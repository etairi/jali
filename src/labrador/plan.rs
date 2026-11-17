//! The prover's choice of a level's parameters: the heuristics of BS23 §5.4, made robust, and
//! the size model of §5.7, with a last witness at the bits of its codes, to choose the base of
//! $`z`$, the splits and when to stop.
//!
//! The prover chooses each level when it starts it, knowing that level's witness: its norm, and
//! its garbage $`g_{ij}=\langle s_i,s_j\rangle`$, which it decomposes exactly. Only the
//! amortized opening $`z`$ and the garbage $`h_{ij}`$ depend on later challenges. Their norms
//! are predicted with a margin, and a level whose next witness exceeds its bound is restarted
//! with the next attempt counter. The bases are odd, so that centred digits are symmetric and
//! the garbage of the next level has no systematic term. The verifier never repeats this
//! choice: it checks the conditions of [`LevelParams::validate`] only.
use super::{
    decompose,
    level::{gaussian_bits, radix_chunk},
    params::{LevelParams, PARAMS_BYTES, Policy, Shape, lambda, projection_max},
    ring::Rq,
    sample::ChallengeSet,
};
use crate::Error;

/// What the prover knows of a level's witness when it chooses the level's parameters.
pub(crate) struct Stats<'a> {
    /// The squared norm of the witness.
    pub(crate) norm_sq: u128,
    /// The garbage $`g_{ij}`$, $`i\le j`$, when known.
    pub(crate) garbage: Option<&'a [Vec<u64>]>,
}

/// A level's parameters before the splits $`\nu,\mu`$, with the predicted bits per
/// coefficient of the next witness in the codes of a last witness (`level::encode_final`).
#[derive(Clone, Copy, Debug)]
struct Fit {
    params: LevelParams,
    bits_z0: f64,
    bits_z1: f64,
    bits_v: f64,
}

/// The bits per digit of the radix code of base $`2h+1`$ (`level::write_radix`): $`k`$ digits
/// in the bits of $`(2h+1)^k-1`$.
fn radix_bits(h: u128) -> f64 {
    if h == 0 {
        return 0.0;
    }
    let radix = 2 * h + 1;
    let k = radix_chunk(radix);
    let power = (0..k).fold(1u128, |p, _| p * radix);
    f64::from(128 - (power - 1).leading_zeros()) / k as f64
}

/// The expected bits per value of the Gaussian code with its best parameter
/// (`level::gaussian_parameter`) for a discrete Gaussian of standard deviation `sigma`.
fn gaussian_code_bits(sigma: f64) -> f64 {
    let sigma = sigma.max(0.1);
    let top = (12.0 * sigma).ceil() as i128;
    let weights: Vec<(i128, f64)> = (-top..=top)
        .map(|x| (x, (-((x * x) as f64) / (2.0 * sigma * sigma)).exp()))
        .collect();
    let total: f64 = weights.iter().map(|(_, w)| w).sum();
    (0..=128 - (top as u128).leading_zeros())
        .map(|p| {
            weights
                .iter()
                .map(|(x, w)| w * gaussian_bits(*x, p) as f64)
                .sum::<f64>()
                / total
        })
        .fold(f64::INFINITY, f64::min)
}

/// The width of the signed code of [`crate::codec::BitWriter::signed`] for magnitudes up to
/// `max`: the least $`w\ge2`$ with $`2^{w-1}-1\ge`$ `max`.
pub(crate) fn signed_width(max: u128) -> u32 {
    (128 - max.leading_zeros() + 1).max(2)
}

/// The nearest odd integer to `x`, at least 3.
fn odd_round(x: f64) -> u64 {
    let v = x.max(3.0).round() as u64;
    if v.is_multiple_of(2) { v + 1 } else { v }
}

/// $`b^t`$, saturating.
fn pow_sat(b: u64, t: usize) -> u128 {
    let mut x = 1u128;
    for _ in 0..t {
        x = x.saturating_mul(u128::from(b));
    }
    x
}

/// The digits that a base near `target` needs for `range` values.
fn digits_for(range: u128, target: u64) -> usize {
    if range <= 1 {
        return 1;
    }
    let t = ((range as f64).ln() / (target as f64).ln()).ceil() as usize;
    t.clamp(1, 64)
}

/// The smallest odd base $`b\ge3`$ with $`b^t\ge`$ `range`.
fn smallest_odd_base(range: u128, t: usize) -> u64 {
    let mut b = odd_round((range as f64).powf(1.0 / t as f64) - 2.0);
    while pow_sat(b, t) < range {
        b += 2;
    }
    while b >= 5 && pow_sat(b - 2, t) >= range {
        b -= 2;
    }
    b
}

/// $`E[X^2]`$ and $`\mathrm{Var}[X^2]`$ for $`X`$ uniform on the integers of $`[-h,h]`$.
fn uniform_moments(h: f64) -> (f64, f64) {
    let e2 = h * (h + 1.0) / 3.0;
    let e4 = h * (h + 1.0) * (3.0 * h * h + 3.0 * h - 1.0) / 15.0;
    (e2, (e4 - e2 * e2).max(0.0))
}

/// The squared norm of the exact decomposition of the garbage, and its largest digit.
fn garbage_digits(ring: &Rq, garbage: &[Vec<u64>], b2: u64, t2: usize) -> (f64, u64) {
    let mut buf = vec![0i64; t2];
    let mut norm = 0u128;
    let mut max = 0u64;
    for poly in garbage {
        for x in poly {
            decompose::digits(ring.center(*x), b2, &mut buf);
            for v in &buf {
                norm += u128::from(v.unsigned_abs()) * u128::from(v.unsigned_abs());
                max = max.max(v.unsigned_abs());
            }
        }
    }
    (norm as f64, max)
}

/// The base of $`z=z_0+bz_1`$ of BS23 §5.4: about $`\sqrt{\sqrt{12}\sigma_z}`$, odd.
fn heuristic_base(set: &ChallengeSet, shape: &Shape, stats: &Stats<'_>, d: usize) -> u64 {
    let nd = (shape.n * d) as f64;
    let tau = set.l2_squared() as f64;
    let gamma_sq = 1.05 * tau * stats.norm_sq as f64 + 4.0 * tau * d as f64;
    odd_round((12f64.sqrt() * (gamma_sq / nd).sqrt()).sqrt())
}

/// The parameters for the base `b` of $`z=z_0+bz_1`$, or the heuristic base of BS23 §5.4.
fn fit(
    ring: &Rq,
    set: &ChallengeSet,
    policy: &Policy,
    shape: &Shape,
    stats: &Stats<'_>,
    last: bool,
    base: Option<u64>,
) -> Option<Fit> {
    let (q, d) = (ring.modulus(), ring.degree());
    let (r, n) = (shape.r, shape.n);
    let nd = (n * d) as f64;
    let pairs = (r * (r + 1) / 2) as f64;
    let tau = set.l2_squared() as f64;
    let w2 = stats.norm_sq as f64;

    // The amortized opening: E||z||^2 = tau ||s||^2 for uniform signs.
    let gamma_sq = 1.05 * tau * w2 + 4.0 * tau * d as f64;
    let sigma_z = (gamma_sq / nd).sqrt();
    let b = base.unwrap_or_else(|| odd_round((12f64.sqrt() * sigma_z).sqrt()));
    let (e0, v0) = uniform_moments(((b - 1) / 2) as f64);
    let z0 = 1.05 * (nd * e0).min(gamma_sq) + 6.0 * (nd * v0).sqrt();
    let z1 = 1.05 * gamma_sq / (b * b) as f64 + 1.0;

    // Inner commitments and h: residues modulo q, t1 centred digits base b1.
    let t1 = digits_for(u128::from(q), b);
    let b1 = smallest_odd_base(u128::from(q), t1);
    let (e1, v1) = uniform_moments(((b1 - 1) / 2) as f64);
    let big = pow_sat(b1, t1 - 1) as f64;
    let htop = ((q as f64 + big - 2.0) / (2.0 * big)).floor().max(0.0);
    let (et, vt) = uniform_moments(htop);
    let per = (t1 - 1) as f64 * e1 + et;
    let per_var = (t1 - 1) as f64 * v1 + vt;
    let residues =
        |count: f64| 1.03 * count * d as f64 * per + 6.0 * (count * d as f64 * per_var).sqrt();

    // The garbage g: decomposed exactly when known, predicted otherwise.
    let (b2, t2, g_norm, g_digit) = match stats.garbage {
        Some(garbage) => {
            let gmax = garbage.iter().map(|p| ring.linf(p)).max().unwrap_or(0);
            let range = 2 * u128::from(gmax) + 1;
            let t2 = digits_for(range, b);
            let b2 = smallest_odd_base(range, t2);
            let (norm, digit) = garbage_digits(ring, garbage, b2, t2);
            (b2, t2, norm, u128::from(digit))
        }
        None => {
            // Coefficients of <s_i, s_j> have standard deviation about sqrt(2 n d) s^2.
            let s2 = w2 / (r as f64 * nd);
            let gmax = (7.0 * (2.0 * nd).sqrt() * s2).ceil().min(q as f64 / 2.0) as u128 + 1;
            let range = 2 * gmax + 1;
            let t2 = digits_for(range, b);
            let b2 = smallest_odd_base(range, t2);
            let (e2, v2) = uniform_moments(((b2 - 1) / 2) as f64);
            let norm = 1.03 * pairs * d as f64 * t2 as f64 * e2
                + 6.0 * (pairs * d as f64 * t2 as f64 * v2).sqrt();
            (b2, t2, norm, u128::from((b2 - 1) / 2))
        }
    };
    let max_v = u128::from((b1 - 1) / 2).max(htop as u128).max(g_digit);
    let bits_z0 = radix_bits(u128::from((b - 1) / 2));
    let bits_z1 = gaussian_code_bits((gamma_sq / ((b * b) as f64 * nd)).sqrt());
    let bits_v = radix_bits(max_v);

    for kappa in 1..=64 {
        let next = z0 + z1 + residues((r * kappa) as f64) + residues(pairs) + g_norm;
        if !next.is_finite() || next >= 1e36 {
            return None;
        }
        let mut params = LevelParams {
            attempt: 0,
            last,
            kappa,
            kappa1: 64,
            kappa2: 64,
            b,
            b1,
            t1,
            b2,
            t2,
            nu: 1,
            mu: 1,
            next_norm_sq: next.ceil() as u128,
        };
        let (inner, outer) = params.msis_bounds(shape, set);
        if !policy.msis_ok(inner, kappa, d, q, true) {
            continue;
        }
        let kappa1 = (1..=64).find(|k| policy.msis_ok(outer, *k, d, q, true))?;
        params.kappa1 = kappa1;
        params.kappa2 = kappa1;
        return Some(Fit {
            params,
            bits_z0,
            bits_z1,
            bits_v,
        });
    }
    None
}

/// The bytes of a level's parameters and messages.
fn level_bytes(ring: &Rq, shape: &Shape, params: &LevelParams) -> f64 {
    let (q, d) = (ring.modulus(), ring.degree());
    let qbits = f64::from(64 - (q - 1).leading_zeros());
    let wp = projection_max(shape.norm_sq).map_or(f64::INFINITY, |m| f64::from(signed_width(m)));
    let polys =
        (params.kappa1 + params.kappa2) as f64 * d as f64 + lambda(q) as f64 * (d - 1) as f64;
    PARAMS_BYTES as f64 + (polys * qbits + 256.0 * wp) / 8.0 + 4.0
}

/// The bytes of the next witness sent in the clear (`level::encode_final`): $`z_0`$ and
/// $`z_1`$ in $`2\nu`$ parts of $`n'`$ polynomials, then the $`m`$ polynomials of $`v`$ in
/// $`\lceil m/n'\rceil`$ parts, each with a header of one byte (and the largest magnitude of a
/// radix part, counted as one more).
fn final_bytes(ring: &Rq, fit: &Fit, nu: usize, m: usize, n_next: usize) -> f64 {
    let d = ring.degree() as f64;
    let mu = m.div_ceil(n_next);
    let z = (nu * n_next) as f64 * d;
    let bits = z * (fit.bits_z0 + fit.bits_z1) + m as f64 * d * fit.bits_v;
    bits / 8.0 + (3 * nu + 2 * mu) as f64 + 1.0
}

/// The splits worth trying for a level: the divisors $`\nu\le256`$ of $`n`$, each with the
/// $`\mu=\lceil m/(n/\nu)\rceil`$ that the layout requires.
fn splits(shape: &Shape, params: &LevelParams) -> Vec<(usize, usize)> {
    let m = params.v_length(shape);
    (1..=shape.n.min(256))
        .filter(|nu| shape.n.is_multiple_of(*nu))
        .map(|nu| (nu, m.div_ceil(shape.n / nu)))
        .collect()
}

/// The cheapest one-more-level continuation from `shape` with predicted statistics, as the
/// last level: its bytes plus the bytes of its witness in the clear.
fn followup(ring: &Rq, set: &ChallengeSet, policy: &Policy, shape: &Shape) -> f64 {
    if super::params::projection_condition(shape.norm_sq, ring.modulus()).is_err() {
        return f64::INFINITY;
    }
    let stats = Stats {
        norm_sq: shape.norm_sq,
        garbage: None,
    };
    let Some(fit) = fit(ring, set, policy, shape, &stats, true, None) else {
        return f64::INFINITY;
    };
    let here = level_bytes(ring, shape, &fit.params);
    splits(shape, &fit.params)
        .into_iter()
        .filter_map(|(nu, mu)| {
            let params = LevelParams {
                nu,
                mu,
                ..fit.params
            };
            let layout = params.layout(shape).ok()?;
            Some(here + final_bytes(ring, &fit, nu, layout.m, layout.n_next))
        })
        .fold(f64::INFINITY, f64::min)
}

/// The parameters of the level with input `shape`, chosen from what the prover knows of its
/// witness (`stats`), minimizing the predicted proof size with a lookahead of one level.
/// Fails with [`Error::Parameter`] when no parameters meet the policy.
pub(crate) fn choose(
    ring: &Rq,
    set: &ChallengeSet,
    policy: &Policy,
    shape: &Shape,
    stats: &Stats<'_>,
) -> Result<LevelParams, Error> {
    choose_with_cost(ring, set, policy, shape, stats).map(|(params, _)| params)
}

/// The odd bases that [`choose`] tries: every odd base from 3 to `top` while there are at most
/// [`EXHAUSTIVE_BASES`] of them (as before, so that such levels choose what they chose), and
/// otherwise [`COARSE_BASES`] spread geometrically over that range together with `heuristic`,
/// which [`choose`] then refines around the best.
fn coarse_bases(top: u64, heuristic: u64) -> Vec<u64> {
    if top.saturating_sub(1) / 2 <= EXHAUSTIVE_BASES {
        return (3..=top).step_by(2).collect();
    }
    let ratio = (top as f64 / 3.0).ln() / (COARSE_BASES - 1) as f64;
    let mut bases: Vec<u64> = (0..COARSE_BASES)
        .map(|k| odd_round((3.0 * (ratio * k as f64).exp()).min(top as f64)))
        .chain([heuristic.clamp(3, top) | 1])
        .filter(|b| (3..=top).contains(b))
        .collect();
    bases.sort_unstable();
    bases.dedup();
    bases
}

/// Odd bases tried exhaustively: up to 160, as for every level of a witness with small digits.
const EXHAUSTIVE_BASES: u64 = 160;
/// Bases of the coarse grid beyond that, and of the refinement around its best.
const COARSE_BASES: usize = 24;
const FINE_BASES: usize = 12;

/// [`choose`] with its predicted proof size from this level on (this level's bytes and the
/// lookahead's). Every candidate base is one the verifier accepts, so the bounded search of
/// [`coarse_bases`] only affects how close to the planner's optimum a large witness gets.
pub(crate) fn choose_with_cost(
    ring: &Rq,
    set: &ChallengeSet,
    policy: &Policy,
    shape: &Shape,
    stats: &Stats<'_>,
) -> Result<(LevelParams, f64), Error> {
    super::params::projection_condition(shape.norm_sq, ring.modulus())?;
    let mut best: Option<(f64, LevelParams)> = None;
    let consider = |last: bool, base: u64, best: &mut Option<(f64, LevelParams)>| {
        let Some(fit) = fit(ring, set, policy, shape, stats, last, Some(base)) else {
            return;
        };
        let here = level_bytes(ring, shape, &fit.params);
        for (nu, mu) in splits(shape, &fit.params) {
            let params = LevelParams {
                nu,
                mu,
                ..fit.params
            };
            let Ok(layout) = params.layout(shape) else {
                continue;
            };
            let rest = if last {
                final_bytes(ring, &fit, nu, layout.m, layout.n_next)
            } else {
                let next = Shape {
                    r: layout.r_next,
                    n: layout.n_next,
                    norm_sq: params.next_norm_sq,
                };
                followup(ring, set, policy, &next)
            };
            let cost = here + rest;
            if best.as_ref().is_none_or(|(c, _)| cost < *c) {
                *best = Some((cost, params));
            }
        }
    };
    // The odd bases from 3 to four times the heuristic one (all of them, or a grid).
    let heuristic = heuristic_base(set, shape, stats, ring.degree());
    let top = 4 * heuristic;
    let bases = coarse_bases(top, heuristic);
    let exhaustive = bases.len() as u64 == top.saturating_sub(1) / 2;
    for (last, base) in [true, false]
        .into_iter()
        .flat_map(|last| bases.iter().map(move |b| (last, *b)))
    {
        consider(last, base, &mut best);
    }
    if !exhaustive {
        // Refine around the best base of the grid, between its neighbours there.
        if let Some((_, p)) = best {
            let at = bases.partition_point(|b| *b < p.b);
            let lo = bases[at.saturating_sub(1)].max(3);
            let hi = bases[(at + 1).min(bases.len() - 1)].max(lo);
            let ratio = (hi as f64 / lo as f64).ln() / (FINE_BASES - 1) as f64;
            let fine: Vec<u64> = (0..FINE_BASES)
                .map(|k| odd_round(lo as f64 * (ratio * k as f64).exp()))
                .filter(|b| (3..=top).contains(b) && !bases.contains(b))
                .collect();
            for last in [true, false] {
                for b in &fine {
                    consider(last, *b, &mut best);
                }
            }
        }
    }
    best.ok_or(Error::Parameter("no LaBRADOR parameters meet the policy"))
        .map(|(c, p)| (p, c))
}

/// The proof size the planner predicts for a statement of `shape`, from its shape and bound
/// alone: the levels [`choose_with_cost`] picks with the bound as the witness's norm and the
/// garbage unknown, each level's bytes ([`level_bytes`]) and, at the last, its prediction of
/// the witness sent in the clear.
pub(crate) fn predict_bytes(
    ring: &Rq,
    set: &ChallengeSet,
    policy: &Policy,
    shape: &Shape,
) -> Result<f64, Error> {
    let mut shape = *shape;
    let mut total = 0.0;
    for _ in 0..super::MAX_LEVELS {
        let stats = Stats {
            norm_sq: shape.norm_sq,
            garbage: None,
        };
        let (params, cost) = choose_with_cost(ring, set, policy, &shape, &stats)?;
        if params.last {
            return Ok(total + cost);
        }
        total += level_bytes(ring, &shape, &params);
        let layout = params.layout(&shape)?;
        shape = Shape {
            r: layout.r_next,
            n: layout.n_next,
            norm_sq: params.next_norm_sq,
        };
    }
    Err(Error::Parameter("too many LaBRADOR levels"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bases_and_digits() {
        assert_eq!(smallest_odd_base(4294967197, 2), 65537);
        assert_eq!(smallest_odd_base(9, 2), 3);
        assert_eq!(smallest_odd_base(10, 2), 5);
        assert_eq!(smallest_odd_base(1, 1), 3);
        for range in [2u128, 3, 100, 1 << 32, 1 << 40] {
            for t in 1..6 {
                let b = smallest_odd_base(range, t);
                assert!(b % 2 == 1 && b >= 3 && pow_sat(b, t) >= range);
                assert!(b == 3 || pow_sat(b - 2, t) < range);
            }
        }
        assert_eq!(digits_for(1, 9), 1);
        assert_eq!(digits_for(1 << 32, 65536), 2);
        let (e, v) = uniform_moments(1.0);
        assert!((e - 2.0 / 3.0).abs() < 1e-12 && (v - (2.0 / 3.0 - 4.0 / 9.0)).abs() < 1e-12);
        assert_eq!(signed_width(0), 2);
        assert_eq!(signed_width(1), 2);
        assert_eq!(signed_width(3), 3);
        assert_eq!(signed_width(4), 4);
    }
}
