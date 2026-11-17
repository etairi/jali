//! The parameters of one LaBRADOR level: their encoding, the conditions that the verifier
//! checks for every level (BS23 Theorem 5.1 and Remark 5.2, adapted to the projection bound
//! $`\sqrt{337}\beta`$ of this implementation), and the prover's choice.
//!
//! **Hardness.** Module-SIS of rank $`\kappa`$ over degree $`d`$ with Euclidean bound $`B<q`$
//! is estimated by the closed form $`\delta=2^{(\log_2B)^2/(4\kappa d\log_2q)}`$ of the
//! methodology of Micciancio and Regev (2009) and Gama and Nguyen (EUROCRYPT 2008), as
//! `docs/parameters.md` describes for the LNP22 sets, and by the smallest BKZ block size
//! $`b\ge50`$ with $`\delta_0(b)\le\delta`$ (Chen's formula). [`Policy`] requires
//! $`\delta<1.0044`$ and a minimum block size: by default 439, about 128 bits under the
//! core-SVP model $`2^{0.292b}`$. These are cost-model heuristics, not proofs.
//!
//! **Conditions per level** (input bound $`\beta`$, next bound $`\beta'`$, slack
//! $`s=\sqrt{337/30}`$ if another level follows and $`s=1`$ if the verifier checks the next
//! witness directly):
//! * $`\sqrt{337/30}\,\beta\le q/125`$ (BS23 Lemma 4.2 for the projection check
//!   $`\|p\|\le\sqrt{337}\beta`$),
//! * M-SIS hard for rank $`\kappa`$ and bound
//!   $`\max(8T(b+1)s\beta',\,2(b+1)s\beta'+4T\sqrt{337/30}\,\beta)`$,
//! * M-SIS hard for ranks $`\kappa_1`$ and $`\kappa_2`$ and bound $`2s\beta'`$.
use super::{ring::Rq, sample::ChallengeSet};
use crate::Error;
use std::f64::consts::{E, PI};

/// The Module-SIS hardness policy that every level must meet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Policy {
    /// The smallest acceptable BKZ block size. At least 346 (the block size of
    /// $`\delta_0\le1.0044`$).
    pub min_block_size: u32,
}

impl Default for Policy {
    /// Block size at least 439: about 128 bits under core-SVP ($`0.292\cdot439\approx128.2`$).
    fn default() -> Self {
        Self {
            min_block_size: 439,
        }
    }
}

/// The ceiling on the M-SIS root Hermite factor, as for the LNP22 sets.
pub(crate) const DELTA_MAX: f64 = 1.0044;
/// The smallest block size the root Hermite factor formula is used for.
const MIN_BLOCK: u32 = 50;
/// The squared slack of the projection check: $`337/30`$.
const SLACK_SQ: f64 = 337.0 / 30.0;

/// The root Hermite factor $`\delta_0(b)`$ of BKZ with block size $`b`$ (Chen's formula).
pub(crate) fn delta_0(b: u32) -> f64 {
    let b = f64::from(b);
    (b / (2.0 * PI * E) * (PI * b).powf(1.0 / b)).powf(1.0 / (2.0 * (b - 1.0)))
}

/// The smallest block size $`b\ge50`$ with $`\delta_0(b)\le\delta`$, `u32::MAX` above $`2^{20}`$.
pub(crate) fn block_size(delta: f64) -> u32 {
    if delta_0(MIN_BLOCK) <= delta {
        return MIN_BLOCK;
    }
    let (mut lo, mut hi) = (MIN_BLOCK, 2 * MIN_BLOCK);
    while delta_0(hi) > delta {
        lo = hi;
        hi *= 2;
        if hi > 1 << 20 {
            return u32::MAX;
        }
    }
    // delta_0(lo) > delta >= delta_0(hi)
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if delta_0(mid) <= delta {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

/// The closed-form M-SIS root Hermite factor for rank `rank`, degree `d`, modulus `q` and
/// Euclidean bound `bound`.
pub(crate) fn msis_delta(bound: f64, rank: usize, d: usize, q: u64) -> f64 {
    let l = bound.max(1.0).log2();
    2f64.powf(l * l / (4.0 * rank as f64 * d as f64 * (q as f64).log2()))
}

impl Policy {
    /// Whether M-SIS of rank `rank` with bound `bound` meets the policy. With `margin` (the
    /// prover's choice) the conditions hold with room to spare, so that a verifier whose
    /// floating-point functions differ in the last place still accepts.
    pub(crate) fn msis_ok(&self, bound: f64, rank: usize, d: usize, q: u64, margin: bool) -> bool {
        if !bound.is_finite() || rank == 0 {
            return false;
        }
        let (bound, delta_max, extra) = if margin {
            (bound * (1.0 + 1e-9), DELTA_MAX * (1.0 - 1e-12), 1)
        } else {
            (bound, DELTA_MAX, 0)
        };
        if bound >= q as f64 {
            return false;
        }
        let delta = msis_delta(bound, rank, d, q);
        delta < delta_max && block_size(delta) >= self.min_block_size.saturating_add(extra)
    }

    /// The policy itself is acceptable: a minimum block size of at least 346.
    pub(crate) fn check(&self) -> Result<(), Error> {
        if self.min_block_size < 346 {
            return Err(Error::Parameter("LaBRADOR hardness policy"));
        }
        Ok(())
    }
}

/// $`\lambda=\lceil128/\lfloor\log_2q\rfloor\rceil`$: the number of aggregations with integer
/// challenges, so that $`q^{-\lambda}\le2^{-128}`$.
pub(crate) fn lambda(q: u64) -> usize {
    128usize.div_ceil((63 - q.leading_zeros()) as usize)
}

/// The input of a level: multiplicity $`r`$, rank $`n`$ and the bound $`\beta^2`$ on
/// $`\sum_i\|s_i\|^2`$.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Shape {
    pub(crate) r: usize,
    pub(crate) n: usize,
    pub(crate) norm_sq: u128,
}

/// The parameters of one level, sent in the proof before the level's messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LevelParams {
    /// The prover's attempt counter, which restarts the level's challenges.
    pub(crate) attempt: u8,
    /// Whether the verifier checks the next witness directly instead of another level.
    pub(crate) last: bool,
    /// Ranks of the inner commitments and of the two outer commitments.
    pub(crate) kappa: usize,
    pub(crate) kappa1: usize,
    pub(crate) kappa2: usize,
    /// The base of $`z=z_0+bz_1`$.
    pub(crate) b: u64,
    /// Base and number of digits of the inner commitments $`t_i`$ and of the garbage $`h_{ij}`$.
    pub(crate) b1: u64,
    pub(crate) t1: usize,
    /// Base and number of digits of the garbage $`g_{ij}`$.
    pub(crate) b2: u64,
    pub(crate) t2: usize,
    /// Parts of $`z_0`$ and $`z_1`$ each, and of $`v=\hat t\|\hat g\|\hat h`$.
    pub(crate) nu: usize,
    pub(crate) mu: usize,
    /// The bound $`\beta'^2`$ on the squared norm of the next witness.
    pub(crate) next_norm_sq: u128,
}

/// The encoded length of [`LevelParams`].
pub(crate) const PARAMS_BYTES: usize = 55;

/// The positions of the next witness, derived from a level's shape and parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Layout {
    /// The level's input multiplicity and rank.
    pub(crate) r: usize,
    pub(crate) n: usize,
    /// $`r(r+1)/2`$: the pairs $`i\le j`$.
    pub(crate) pairs: usize,
    /// The length of $`v`$ in polynomials, and where $`\hat g`$ and $`\hat h`$ start in it.
    pub(crate) m: usize,
    pub(crate) off_g: usize,
    pub(crate) off_h: usize,
    /// The next multiplicity $`2\nu+\mu`$ and rank $`n'`$.
    pub(crate) r_next: usize,
    pub(crate) n_next: usize,
}

/// The index of the pair $`i\le j`$ among $`r(r+1)/2`$ pairs in lexicographic order.
#[inline]
pub(crate) fn pair_index(i: usize, j: usize, r: usize) -> usize {
    debug_assert!(i <= j && j < r);
    // Rows 0 to i-1 hold r, r-1, ..., r-i+1 pairs.
    i * r - i * i.saturating_sub(1) / 2 + (j - i)
}

impl LevelParams {
    pub(crate) fn encode(&self) -> [u8; PARAMS_BYTES] {
        let mut out = [0u8; PARAMS_BYTES];
        out[0] = self.attempt;
        out[1] = u8::from(self.last);
        out[2] = self.kappa as u8;
        out[3] = self.kappa1 as u8;
        out[4] = self.kappa2 as u8;
        out[5] = self.t1 as u8;
        out[6] = self.t2 as u8;
        out[7..15].copy_from_slice(&self.b.to_le_bytes());
        out[15..23].copy_from_slice(&self.b1.to_le_bytes());
        out[23..31].copy_from_slice(&self.b2.to_le_bytes());
        out[31..35].copy_from_slice(&(self.nu as u32).to_le_bytes());
        out[35..39].copy_from_slice(&(self.mu as u32).to_le_bytes());
        out[39..55].copy_from_slice(&self.next_norm_sq.to_le_bytes());
        out
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, Error> {
        let bytes: &[u8; PARAMS_BYTES] = bytes.try_into().map_err(|_| Error::Encoding)?;
        let u64_at = |i: usize| u64::from_le_bytes(bytes[i..i + 8].try_into().expect("8 bytes"));
        let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().expect("4 bytes"));
        if bytes[1] > 1 {
            return Err(Error::Encoding);
        }
        Ok(Self {
            attempt: bytes[0],
            last: bytes[1] == 1,
            kappa: usize::from(bytes[2]),
            kappa1: usize::from(bytes[3]),
            kappa2: usize::from(bytes[4]),
            t1: usize::from(bytes[5]),
            t2: usize::from(bytes[6]),
            b: u64_at(7),
            b1: u64_at(15),
            b2: u64_at(23),
            nu: u32_at(31) as usize,
            mu: u32_at(35) as usize,
            next_norm_sq: u128::from_le_bytes(bytes[39..55].try_into().expect("16 bytes")),
        })
    }

    /// The length $`m`$ of $`v=\hat t\|\hat g\|\hat h`$ in polynomials.
    pub(crate) fn v_length(&self, shape: &Shape) -> usize {
        let pairs = shape.r * (shape.r + 1) / 2;
        shape.r * self.t1 * self.kappa + pairs * (self.t1 + self.t2)
    }

    /// The layout of the next witness: $`\nu`$ divides $`n`$, $`n'=n/\nu`$ and
    /// $`\mu=\lceil m/n'\rceil`$. Fails with [`Error::Parameter`].
    pub(crate) fn layout(&self, shape: &Shape) -> Result<Layout, Error> {
        let bad = Error::Parameter("LaBRADOR level dimensions");
        let (r, n) = (shape.r, shape.n);
        if !(1..=64).contains(&self.kappa)
            || !(1..=64).contains(&self.kappa1)
            || !(1..=64).contains(&self.kappa2)
            || !(1..=64).contains(&self.t1)
            || !(1..=64).contains(&self.t2)
            || [self.b, self.b1, self.b2]
                .iter()
                .any(|b| !(2..=1 << 40).contains(b))
            || r == 0
            || r > super::statement::MAX_MULTIPLICITY
            || n == 0
        {
            return Err(bad);
        }
        let pairs = r * (r + 1) / 2;
        let off_g = r * self.t1 * self.kappa;
        let off_h = off_g + pairs * self.t2;
        let m = off_h + pairs * self.t1;
        // z splits into parts of exactly n / nu polynomials: padding inside a part of z would
        // enter <z, z> unconstrained. v, which the next statement uses only linearly, takes
        // the fewest parts of that length, the last one zero-padded.
        if self.nu == 0 || !n.is_multiple_of(self.nu) {
            return Err(bad);
        }
        let n_next = n / self.nu;
        if self.mu != m.div_ceil(n_next) {
            return Err(bad);
        }
        let r_next = 2 * self.nu + self.mu;
        if r_next > super::statement::MAX_MULTIPLICITY {
            return Err(bad);
        }
        Ok(Layout {
            r,
            n,
            pairs,
            m,
            off_g,
            off_h,
            r_next,
            n_next,
        })
    }

    /// Every condition of the module documentation, and the layout. Fails with
    /// [`Error::Parameter`] naming the condition.
    pub(crate) fn validate(
        &self,
        shape: &Shape,
        ring: &Rq,
        set: &ChallengeSet,
        policy: &Policy,
    ) -> Result<Layout, Error> {
        let layout = self.layout(shape)?;
        if layout
            .r_next
            .checked_mul(layout.n_next * ring.degree())
            .is_none_or(|c| c > super::statement::MAX_WITNESS_COEFFICIENTS)
        {
            return Err(Error::Parameter("LaBRADOR level dimensions"));
        }
        projection_condition(shape.norm_sq, ring.modulus())?;
        if self.next_norm_sq == 0 {
            return Err(Error::Parameter("LaBRADOR next norm bound"));
        }
        if !self.secure(shape, ring, set, policy, false) {
            return Err(Error::Parameter("LaBRADOR Module-SIS hardness"));
        }
        Ok(layout)
    }

    /// The Euclidean bounds of the inner and of the outer M-SIS conditions.
    pub(crate) fn msis_bounds(&self, shape: &Shape, set: &ChallengeSet) -> (f64, f64) {
        let slack_sq = if self.last { 1.0 } else { SLACK_SQ };
        let next = (slack_sq * self.next_norm_sq as f64).sqrt();
        let beta = (shape.norm_sq as f64).sqrt();
        let t = set.op_bound as f64;
        let b1 = self.b as f64 + 1.0;
        let inner = (8.0 * t * b1 * next).max(2.0 * b1 * next + 4.0 * t * SLACK_SQ.sqrt() * beta);
        (inner, 2.0 * next)
    }

    /// The M-SIS conditions.
    pub(crate) fn secure(
        &self,
        shape: &Shape,
        ring: &Rq,
        set: &ChallengeSet,
        policy: &Policy,
        margin: bool,
    ) -> bool {
        let (q, d) = (ring.modulus(), ring.degree());
        let (inner, outer) = self.msis_bounds(shape, set);
        policy.msis_ok(inner, self.kappa, d, q, margin)
            && policy.msis_ok(outer, self.kappa1, d, q, margin)
            && policy.msis_ok(outer, self.kappa2, d, q, margin)
    }
}

/// $`\sqrt{337/30}\,\beta\le q/125`$, in integers: $`337\cdot125^2\beta^2\le30q^2`$.
pub(crate) fn projection_condition(norm_sq: u128, q: u64) -> Result<(), Error> {
    let lhs = norm_sq.checked_mul(337 * 125 * 125);
    // 30 q^2 exceeds u128 for q above 2^61.5, and then every representable lhs passes.
    let rhs = u128::from(q)
        .checked_mul(u128::from(q))
        .and_then(|x| x.checked_mul(30))
        .unwrap_or(u128::MAX);
    match lhs {
        Some(l) if l <= rhs => Ok(()),
        _ => Err(Error::Parameter(
            "LaBRADOR norm bound too large for the modulus",
        )),
    }
}

/// The largest $`|p_j|`$ that a projection passing $`\|p\|^2\le337\beta^2`$ can hold.
pub(crate) fn projection_max(norm_sq: u128) -> Result<u128, Error> {
    Ok(norm_sq.checked_mul(337).ok_or(Error::Overflow)?.isqrt())
}

#[cfg(test)]
mod tests;
