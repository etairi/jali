//! One level of the main protocol of BS23 (Fig. 2 and 3), made non-interactive with the
//! transcript: the prover's messages and the verifier's reduction to the next statement.
//!
//! Order of a level (`docs/labrador.md` gives the encodings): absorb the level's parameters
//! (55 bytes), absorb $`u_1`$, squeeze the 32-byte projection seed, absorb $`p`$, squeeze
//! $`\psi^{(k)},\omega^{(k)}`$ ($`k<\lambda`$, `DecodeUint` each), absorb the non-constant
//! coefficients of $`b''^{(k)}`$, squeeze $`\alpha,\beta`$ (uniform polynomials), absorb
//! $`u_2`$, squeeze the 32-byte challenge seed.
use super::{
    decompose::decompose,
    params::{Layout, LevelParams, PARAMS_BYTES, Policy, Shape, pair_index, projection_max},
    plan::{self, Stats, signed_width},
    recursion::{Next, next_witness},
    ring::{Rq, Sparse},
    sample::{ChallengeSet, Matrix, Projection, purpose},
    statement::{Aggregate, Relation, all_pairs},
};
use crate::{
    Error,
    codec::{BitReader, BitWriter},
    math::{Ring, U256, int},
    transcript::Transcript,
};
use std::sync::Arc;
use zeroize::Zeroizing;

/// The next statement and its witness.
pub(crate) type NextLevel = (Next, Vec<Zeroizing<Vec<u64>>>);

/// The attempts of a level before the prover gives up.
const MAX_ATTEMPTS: u8 = 64;

/// What every level of one proof shares.
pub(crate) struct Ctx {
    pub(crate) ring: Arc<Rq>,
    pub(crate) set: ChallengeSet,
    pub(crate) eta_ring: Arc<Ring>,
    pub(crate) policy: Policy,
    pub(crate) pp_seed: [u8; 32],
    pub(crate) lambda: usize,
}

/// The byte lengths of a level's messages and the width of the projection's entries.
struct Sizes {
    u1: usize,
    p: usize,
    p_width: u32,
    bpp: usize,
    u2: usize,
}

fn qbits(q: u64) -> u32 {
    64 - (q - 1).leading_zeros()
}

fn sizes(ctx: &Ctx, shape: &Shape, params: &LevelParams) -> Result<Sizes, Error> {
    let (q, d) = (ctx.ring.modulus(), ctx.ring.degree());
    let w = qbits(q) as usize;
    let p_width = signed_width(projection_max(shape.norm_sq)?);
    Ok(Sizes {
        u1: (params.kappa1 * d * w).div_ceil(8),
        p: (256 * p_width as usize).div_ceil(8),
        p_width,
        bpp: (ctx.lambda * (d - 1) * w).div_ceil(8),
        u2: (params.kappa2 * d * w).div_ceil(8),
    })
}

/// Residues modulo $`q`$ in $`\lceil\log_2q\rceil`$ bits each, least significant first, zero
/// padded to a byte.
pub(crate) fn encode_residues(q: u64, values: &[u64]) -> Result<Vec<u8>, Error> {
    let w = qbits(q);
    let mut writer = BitWriter::new();
    for x in values {
        writer.uniform(u128::from(*x), u128::from(q))?;
    }
    let pad = (8 - (values.len() * w as usize) % 8) % 8;
    writer.unsigned(0, pad as u32)?;
    writer.into_bytes()
}

/// The inverse of [`encode_residues`] for `count` residues: [`Error::Encoding`] for a value
/// at or above $`q`$, nonzero padding, or a length that is not exact.
pub(crate) fn decode_residues(q: u64, bytes: &[u8], count: usize) -> Result<Vec<u64>, Error> {
    let w = qbits(q) as usize;
    if bytes.len() != (count * w).div_ceil(8) {
        return Err(Error::Encoding);
    }
    let mut reader = BitReader::new(bytes);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(reader.uniform(u128::from(q))? as u64);
    }
    let pad = (8 - (count * w) % 8) % 8;
    if reader.unsigned(pad as u32)? != 0 {
        return Err(Error::Encoding);
    }
    reader.finish_bytes()?;
    Ok(out)
}

/// Signed values in the fixed-width code of [`BitWriter::signed`], zero padded to a byte.
pub(crate) fn encode_signed(values: &[i128], width: u32) -> Result<Vec<u8>, Error> {
    let mut writer = BitWriter::new();
    for x in values {
        writer.signed(*x, width)?;
    }
    let pad = (8 - (values.len() * width as usize) % 8) % 8;
    writer.unsigned(0, pad as u32)?;
    writer.into_bytes()
}

/// The inverse of [`encode_signed`] for `count` values.
pub(crate) fn decode_signed(bytes: &[u8], count: usize, width: u32) -> Result<Vec<i128>, Error> {
    if bytes.len() != (count * width as usize).div_ceil(8) {
        return Err(Error::Encoding);
    }
    let mut reader = BitReader::new(bytes);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(reader.signed(width)?);
    }
    let pad = (8 - (count * width as usize) % 8) % 8;
    if reader.unsigned(pad as u32)? != 0 {
        return Err(Error::Encoding);
    }
    reader.finish_bytes()?;
    Ok(out)
}

/// `count` residues modulo $`q`$ squeezed with `DecodeUint`.
pub(crate) fn squeeze_residues(
    t: &mut Transcript,
    q: u64,
    count: usize,
) -> Result<Vec<u64>, Error> {
    let modulus = U256::from_u64(q);
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let n = (count - out.len()).min(1 << 20);
        for x in t.uints(&modulus, n)? {
            out.push(int::to_u64(&x).ok_or(Error::Overflow)?);
        }
    }
    Ok(out)
}

/// `count` uniform polynomials.
pub(crate) fn squeeze_polys(
    t: &mut Transcript,
    ring: &Rq,
    count: usize,
) -> Result<Vec<Vec<u64>>, Error> {
    let d = ring.degree();
    Ok(squeeze_residues(t, ring.modulus(), count * d)?
        .chunks_exact(d)
        .map(<[u64]>::to_vec)
        .collect())
}

fn squeeze_seed(t: &mut Transcript) -> [u8; 32] {
    let mut seed = [0u8; 32];
    t.squeeze(&mut seed);
    seed
}

/// The next `n` bytes of the proof.
pub(crate) fn take<'a>(input: &mut &'a [u8], n: usize) -> Result<&'a [u8], Error> {
    if input.len() < n {
        return Err(Error::Encoding);
    }
    let (head, rest) = input.split_at(n);
    *input = rest;
    Ok(head)
}

/// $`u=M\cdot v`$ for the public matrix of `purpose` at `level`, `rows` rows, `v` of `cols`
/// polynomials given by its transforms.
fn matvec(
    ctx: &Ctx,
    purpose: u32,
    level: u32,
    rows: usize,
    v_hat: &[u64],
) -> Result<Vec<u64>, Error> {
    let ring = &ctx.ring;
    let (d, k) = (ring.degree(), ring.hat_len());
    let cols = v_hat.len() / k;
    let mut matrix = Matrix::new(ring, &ctx.pp_seed, purpose, level);
    let mut entry = vec![0u64; d];
    let mut entry_hat = vec![0u64; k];
    let mut out = vec![0u64; rows * d];
    for row in 0..rows {
        let mut acc = vec![0u64; k];
        for col in 0..cols {
            matrix.next_poly(&mut entry)?;
            ring.ntt_into(&entry, &mut entry_hat);
            ring.mac(&mut acc, &entry_hat, &v_hat[col * k..(col + 1) * k]);
        }
        ring.intt_into(&mut acc, &mut out[row * d..(row + 1) * d]);
    }
    Ok(out)
}

/// The projection $`p=\sum_i\Pi_is_i`$ of all witness coefficients, in vector order.
fn project(
    ring: &Rq,
    seed: &[u8; 32],
    level: u32,
    witness: &[Zeroizing<Vec<u64>>],
) -> Result<[i128; 256], Error> {
    let mut projection = Projection::new(seed, level);
    let mut p = [0i128; 256];
    let mut column = [0i8; 256];
    for v in witness {
        for x in v.iter() {
            projection.column(&mut column)?;
            let s = i128::from(ring.center(*x));
            if s != 0 {
                for (acc, e) in p.iter_mut().zip(column) {
                    *acc += i128::from(e) * s;
                }
            }
        }
    }
    Ok(p)
}

/// Stream the coefficient polynomials of the aggregated projection constraints (BS23 Fig. 2,
/// $`\varphi''`$): for each witness vector $`i`$ and polynomial $`x`$, call `f(i, x, polys)`
/// with `polys[k]` the polynomial $`x`$ of $`\sigma_{-1}(\sum_j\omega^{(k)}_j\pi^{(j)}_i)`$,
/// $`k<\lambda`$. Nothing of size proportional to the witness is held.
fn for_each_projection_poly(
    ring: &Rq,
    seed: &[u8; 32],
    level: u32,
    omega: &[Vec<u64>],
    shape: &Shape,
    mut f: impl FnMut(usize, usize, &[Vec<u64>]) -> Result<(), Error>,
) -> Result<(), Error> {
    let d = ring.degree();
    let q = ring.modulus() as i64;
    // For each k, each group of 4 rows and each byte, the sum of the 4 weighted entries.
    let tables: Vec<Vec<i64>> = omega
        .iter()
        .map(|w| {
            let mut table = vec![0i64; 64 * 256];
            for group in 0..64 {
                for byte in 0..256usize {
                    let mut s = 0i64;
                    for e in 0..4 {
                        let v = ((byte >> (2 * e)) & 1) as i64 - ((byte >> (2 * e + 1)) & 1) as i64;
                        s += v * w[4 * group + e] as i64;
                    }
                    table[group * 256 + byte] = s;
                }
            }
            table
        })
        .collect();
    let mut projection = Projection::new(seed, level);
    let mut polys = vec![vec![0u64; d]; omega.len()];
    let mut tmp = vec![0u64; d];
    let mut column = [0u8; 64];
    for i in 0..shape.r {
        for x in 0..shape.n {
            for c in 0..d {
                projection.column_bytes(&mut column)?;
                for (poly, table) in polys.iter_mut().zip(&tables) {
                    // 64 terms below 2^50 in magnitude.
                    let s: i64 = column
                        .iter()
                        .enumerate()
                        .map(|(g, b)| table[g * 256 + usize::from(*b)])
                        .sum();
                    poly[c] = s.rem_euclid(q) as u64;
                }
            }
            for poly in &mut polys {
                ring.sigma_minus_one(poly, &mut tmp);
                poly.copy_from_slice(&tmp);
            }
            f(i, x, &polys)?;
        }
    }
    Ok(())
}

/// $`\gamma_l=\sum_k\beta_k\psi^{(k)}_l`$.
fn gammas(ring: &Rq, beta: &[Vec<u64>], psi: &[Vec<u64>], count: usize) -> Vec<Vec<u64>> {
    (0..count)
        .map(|l| {
            let mut g = vec![0u64; ring.degree()];
            for (b, p) in beta.iter().zip(psi) {
                ring.add_scaled_assign(&mut g, b, p[l]);
            }
            g
        })
        .collect()
}

/// Complete an aggregate with the projection constraints and the lifted polynomials:
/// $`\varphi_i\mathrel{+}=\sum_k\beta_k\varphi''^{(k)}_i`$ and
/// $`b\mathrel{+}=\sum_k\beta_kb''^{(k)}`$.
#[allow(clippy::too_many_arguments)]
fn complete(
    ring: &Rq,
    agg: &mut Aggregate,
    seed: &[u8; 32],
    level: u32,
    omega: &[Vec<u64>],
    shape: &Shape,
    beta: &[Vec<u64>],
    bpp: &[Vec<u64>],
) -> Result<(), Error> {
    let (d, k) = (ring.degree(), ring.hat_len());
    let beta_hat: Vec<_> = beta.iter().map(|b| ring.ntt(b)).collect();
    let mut acc = vec![0u64; k];
    let mut poly = vec![0u64; d];
    for_each_projection_poly(ring, seed, level, omega, shape, |i, x, polys| {
        acc.fill(0);
        for (p, b) in polys.iter().zip(&beta_hat) {
            ring.mac(&mut acc, b, &ring.ntt(p));
        }
        ring.intt_into(&mut acc, &mut poly);
        ring.add_assign(&mut agg.phi[i][x * d..(x + 1) * d], &poly);
        Ok(())
    })?;
    for (b, l) in beta.iter().zip(bpp) {
        ring.add_assign(&mut agg.constant, &ring.poly_mul(b, l));
    }
    Ok(())
}

/// $`\varphi_z=\sum_ic_i\varphi_i`$.
fn phi_z(ring: &Rq, phi: &[Vec<u64>], challenges: &[Sparse], n: usize) -> Vec<u64> {
    let d = ring.degree();
    let mut out = vec![0u64; n * d];
    for (c, v) in challenges.iter().zip(phi) {
        for x in 0..n {
            ring.mul_sparse_add(c, &v[x * d..(x + 1) * d], &mut out[x * d..(x + 1) * d]);
        }
    }
    out
}

/// $`F(w)`$ for an aggregate and a witness given by its transforms.
pub(crate) fn evaluate_aggregate(ring: &Rq, agg: &Aggregate, hats: &[Vec<u64>]) -> Vec<u64> {
    let mut value = vec![0u64; ring.degree()];
    for (i, j, a) in &agg.quad {
        ring.add_assign(
            &mut value,
            &ring.poly_mul(a, &ring.inner_hat(&hats[*i], &hats[*j])),
        );
    }
    for (phi, hat) in agg.phi.iter().zip(hats) {
        if phi.iter().any(|x| *x != 0) {
            ring.add_assign(&mut value, &ring.inner_hat(&ring.ntt_many(phi), hat));
        }
    }
    ring.sub_assign(&mut value, &agg.constant);
    value
}

/// Run one level as the prover: choose its parameters, then try attempts until the next
/// witness meets its bound. Appends the level's bytes to `out` and returns the next statement
/// and witness.
pub(crate) fn prove<R: Relation>(
    ctx: &Ctx,
    rel: &R,
    witness: &[Zeroizing<Vec<u64>>],
    level: u32,
    transcript: &mut Transcript,
    out: &mut Vec<u8>,
) -> Result<NextLevel, Error> {
    let ring = &ctx.ring;
    let shape = rel.shape();
    // Everything derived from the witness is wiped on drop.
    let hats: Zeroizing<Vec<Vec<u64>>> =
        Zeroizing::new(witness.iter().map(|v| ring.ntt_many(v)).collect());
    let pairs = Zeroizing::new(all_pairs(ring, &hats));
    let norm_sq: u128 = witness.iter().map(|v| ring.norm_sq(v)).sum();
    if norm_sq > shape.norm_sq {
        return Err(Error::Witness);
    }
    let base = plan::choose(
        ring,
        &ctx.set,
        &ctx.policy,
        &shape,
        &Stats {
            norm_sq,
            garbage: Some(&pairs),
        },
    )?;
    for attempt in 0..MAX_ATTEMPTS {
        let params = LevelParams { attempt, ..base };
        let mut t = transcript.clone();
        let mut bytes = Vec::new();
        if let Some(result) = attempt_level(
            ctx, rel, witness, &hats, &pairs, level, &params, &mut t, &mut bytes,
        )? {
            *transcript = t;
            out.extend_from_slice(&bytes);
            return Ok(result);
        }
    }
    Err(Error::RestartLimit)
}

#[allow(clippy::too_many_arguments)]
fn attempt_level<R: Relation>(
    ctx: &Ctx,
    rel: &R,
    witness: &[Zeroizing<Vec<u64>>],
    hats: &[Vec<u64>],
    pairs: &[Vec<u64>],
    level: u32,
    params: &LevelParams,
    t: &mut Transcript,
    out: &mut Vec<u8>,
) -> Result<Option<NextLevel>, Error> {
    let ring = &ctx.ring;
    let (q, d, k) = (ring.modulus(), ring.degree(), ring.hat_len());
    let shape = rel.shape();
    let (r, n) = (shape.r, shape.n);
    let layout = params.validate(&shape, ring, &ctx.set, &ctx.policy)?;
    let sizes = sizes(ctx, &shape, params)?;
    let encoded = params.encode();
    t.absorb(&encoded);
    out.extend_from_slice(&encoded);

    // Inner commitments t_i = A s_i, all i at once, row by row.
    let kappa = params.kappa;
    let mut t_vecs = Zeroizing::new(vec![vec![0u64; kappa * d]; r]);
    {
        let mut a = Matrix::new(ring, &ctx.pp_seed, purpose::MATRIX_A, level);
        let mut entry = vec![0u64; d];
        let mut entry_hat = vec![0u64; k];
        for row in 0..kappa {
            let mut accs = Zeroizing::new(vec![vec![0u64; k]; r]);
            for x in 0..n {
                a.next_poly(&mut entry)?;
                ring.ntt_into(&entry, &mut entry_hat);
                for (acc, hat) in accs.iter_mut().zip(hats) {
                    ring.mac(acc, &entry_hat, &hat[x * k..(x + 1) * k]);
                }
            }
            for (i, acc) in accs.iter_mut().enumerate() {
                ring.intt_into(acc, &mut t_vecs[i][row * d..(row + 1) * d]);
            }
        }
    }

    // v = t̂ ‖ ĝ ‖ ĥ: the first two parts now.
    let mut v = Zeroizing::new(vec![0u64; layout.m * d]);
    for (i, ti) in t_vecs.iter().enumerate() {
        let digits = Zeroizing::new(decompose(ring, ti, params.b1, params.t1));
        for (digit, part) in digits.iter().enumerate() {
            let y = (i * params.t1 + digit) * kappa;
            v[y * d..(y + kappa) * d].copy_from_slice(part);
        }
    }
    for (pair, g) in pairs.iter().enumerate() {
        let digits = Zeroizing::new(decompose(ring, g, params.b2, params.t2));
        for (digit, part) in digits.iter().enumerate() {
            let y = layout.off_g + pair * params.t2 + digit;
            v[y * d..(y + 1) * d].copy_from_slice(part);
        }
    }
    let u1 = matvec(
        ctx,
        purpose::MATRIX_B,
        level,
        params.kappa1,
        &Zeroizing::new(ring.ntt_many(&v[..layout.off_h * d])),
    )?;
    let bytes = encode_residues(q, &u1)?;
    debug_assert_eq!(bytes.len(), sizes.u1);
    t.absorb(&bytes);
    out.extend_from_slice(&bytes);

    // Projection.
    let jl_seed = squeeze_seed(t);
    let p = project(ring, &jl_seed, level, witness)?;
    let p_norm: u128 = p.iter().map(|x| x.unsigned_abs() * x.unsigned_abs()).sum();
    if p_norm > shape.norm_sq.saturating_mul(337) {
        return Ok(None);
    }
    let bytes = encode_signed(&p, sizes.p_width)?;
    t.absorb(&bytes);
    out.extend_from_slice(&bytes);

    // Aggregation with integer challenges and the lifted polynomials b''.
    let l_count = rel.constant_term_count();
    let mut psi = Vec::with_capacity(ctx.lambda);
    let mut omega = Vec::with_capacity(ctx.lambda);
    for _ in 0..ctx.lambda {
        psi.push(squeeze_residues(t, q, l_count)?);
        omega.push(squeeze_residues(t, q, 256)?);
    }
    let values = Zeroizing::new(rel.constant_term_values(hats, pairs));
    let mut accs = vec![vec![0u64; k]; ctx.lambda];
    for_each_projection_poly(ring, &jl_seed, level, &omega, &shape, |i, x, polys| {
        for (acc, poly) in accs.iter_mut().zip(polys) {
            ring.mac(acc, &ring.ntt(poly), &hats[i][x * k..(x + 1) * k]);
        }
        Ok(())
    })?;
    let mut bpp = Vec::with_capacity(ctx.lambda);
    for (acc, psi_k) in accs.into_iter().zip(&psi) {
        let mut b = ring.intt(acc);
        for (value, weight) in values.iter().zip(psi_k) {
            ring.add_scaled_assign(&mut b, value, *weight);
        }
        bpp.push(b);
    }
    let tails: Vec<u64> = bpp.iter().flat_map(|b| b[1..].iter().copied()).collect();
    let bytes = encode_residues(q, &tails)?;
    debug_assert_eq!(bytes.len(), sizes.bpp);
    t.absorb(&bytes);
    out.extend_from_slice(&bytes);

    // Aggregation with polynomial challenges.
    let alpha = squeeze_polys(t, ring, rel.full_count())?;
    let beta = squeeze_polys(t, ring, ctx.lambda)?;
    let gamma = gammas(ring, &beta, &psi, l_count);
    let mut agg = rel.aggregate(&alpha, &gamma)?;
    complete(ring, &mut agg, &jl_seed, level, &omega, &shape, &beta, &bpp)?;
    if evaluate_aggregate(ring, &agg, hats).iter().any(|x| *x != 0) {
        return Err(Error::Witness);
    }

    // Garbage h and the second outer commitment.
    let phi_hats: Vec<Vec<u64>> = agg.phi.iter().map(|v| ring.ntt_many(v)).collect();
    for i in 0..r {
        for j in i..r {
            let mut h = Zeroizing::new(ring.inner_hat(&phi_hats[i], &hats[j]));
            if i != j {
                ring.add_assign(&mut h, &ring.inner_hat(&phi_hats[j], &hats[i]));
            }
            let pair = pair_index(i, j, r);
            let digits = Zeroizing::new(decompose(ring, &h, params.b1, params.t1));
            for (digit, part) in digits.iter().enumerate() {
                let y = layout.off_h + pair * params.t1 + digit;
                v[y * d..(y + 1) * d].copy_from_slice(part);
            }
        }
    }
    let u2 = matvec(
        ctx,
        purpose::MATRIX_D,
        level,
        params.kappa2,
        &Zeroizing::new(ring.ntt_many(&v[layout.off_h * d..])),
    )?;
    let bytes = encode_residues(q, &u2)?;
    debug_assert_eq!(bytes.len(), sizes.u2);
    t.absorb(&bytes);
    out.extend_from_slice(&bytes);

    // Amortization.
    let c_seed = squeeze_seed(t);
    let challenges = (0..r)
        .map(|i| ctx.set.sample(&c_seed, i as u32, &ctx.eta_ring))
        .collect::<Result<Vec<_>, _>>()?;
    let mut z = Zeroizing::new(vec![0u64; n * d]);
    for (c, s) in challenges.iter().zip(witness) {
        for x in 0..n {
            ring.mul_sparse_add(c, &s[x * d..(x + 1) * d], &mut z[x * d..(x + 1) * d]);
        }
    }
    let next_wit = next_witness(ring, params, &layout, &z, &v);
    let next_norm: u128 = next_wit.iter().map(|w| ring.norm_sq(w)).sum();
    if next_norm > params.next_norm_sq {
        return Ok(None);
    }
    let next = Next {
        ring: ring.clone(),
        pp_seed: ctx.pp_seed,
        level,
        params: *params,
        layout,
        phi_z: phi_z(ring, &agg.phi, &challenges, n),
        challenges,
        u1,
        u2,
        quad: agg.quad,
        constant: agg.constant,
    };
    Ok(Some((next, next_wit)))
}

/// Check one level as the verifier, reading its bytes from `input`: validate its parameters,
/// check the projection, and return the next statement.
pub(crate) fn verify<R: Relation>(
    ctx: &Ctx,
    rel: &R,
    level: u32,
    transcript: &mut Transcript,
    input: &mut &[u8],
) -> Result<Next, Error> {
    let ring = &ctx.ring;
    let (q, d) = (ring.modulus(), ring.degree());
    let shape = rel.shape();
    let encoded = take(input, PARAMS_BYTES)?;
    let params = LevelParams::decode(encoded)?;
    let layout = params
        .validate(&shape, ring, &ctx.set, &ctx.policy)
        .map_err(|_| Error::InvalidProof)?;
    let sizes = sizes(ctx, &shape, &params)?;
    transcript.absorb(encoded);

    let bytes = take(input, sizes.u1)?;
    transcript.absorb(bytes);
    let u1 = decode_residues(q, bytes, params.kappa1 * d)?;

    let jl_seed = squeeze_seed(transcript);
    let bytes = take(input, sizes.p)?;
    transcript.absorb(bytes);
    let p = decode_signed(bytes, 256, sizes.p_width)?;
    let p_norm = p
        .iter()
        .try_fold(0u128, |acc, x| {
            acc.checked_add(x.unsigned_abs() * x.unsigned_abs())
        })
        .ok_or(Error::InvalidProof)?;
    if p_norm > shape.norm_sq.saturating_mul(337) {
        return Err(Error::InvalidProof);
    }

    let l_count = rel.constant_term_count();
    let mut psi = Vec::with_capacity(ctx.lambda);
    let mut omega = Vec::with_capacity(ctx.lambda);
    for _ in 0..ctx.lambda {
        psi.push(squeeze_residues(transcript, q, l_count)?);
        omega.push(squeeze_residues(transcript, q, 256)?);
    }
    let bytes = take(input, sizes.bpp)?;
    transcript.absorb(bytes);
    let tails = decode_residues(q, bytes, ctx.lambda * (d - 1))?;
    // The constant coefficient of b''^(k) is <omega^(k), p> + sum_l psi_l^(k) b'_0^(l).
    let constants = rel.constant_term_constants();
    let bpp: Vec<Vec<u64>> = (0..ctx.lambda)
        .map(|k| {
            let mut ct = p.iter().zip(&omega[k]).fold(0u64, |acc, (x, w)| {
                ring.add(acc, ring.mul(ring.lift_i128(*x), *w))
            });
            for (b0, w) in constants.iter().zip(&psi[k]) {
                ct = ring.add(ct, ring.mul(*b0, *w));
            }
            let mut b = Vec::with_capacity(d);
            b.push(ct);
            b.extend_from_slice(&tails[k * (d - 1)..(k + 1) * (d - 1)]);
            b
        })
        .collect();

    let alpha = squeeze_polys(transcript, ring, rel.full_count())?;
    let beta = squeeze_polys(transcript, ring, ctx.lambda)?;
    let gamma = gammas(ring, &beta, &psi, l_count);
    let mut agg = rel.aggregate(&alpha, &gamma)?;
    complete(ring, &mut agg, &jl_seed, level, &omega, &shape, &beta, &bpp)?;

    let bytes = take(input, sizes.u2)?;
    transcript.absorb(bytes);
    let u2 = decode_residues(q, bytes, params.kappa2 * d)?;

    let c_seed = squeeze_seed(transcript);
    let challenges = (0..shape.r)
        .map(|i| ctx.set.sample(&c_seed, i as u32, &ctx.eta_ring))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Next {
        ring: ring.clone(),
        pp_seed: ctx.pp_seed,
        level,
        params,
        phi_z: phi_z(ring, &agg.phi, &challenges, shape.n),
        layout,
        challenges,
        u1,
        u2,
        quad: agg.quad,
        constant: agg.constant,
    })
}

/// The polynomials of vector `i` of the next witness that carry values: all $`n'`$ of the
/// parts of $`z_0`$ and $`z_1`$, and those of $`v`$ before its zero padding.
fn final_length(layout: &Layout, nu: usize, i: usize) -> usize {
    if i < 2 * nu {
        layout.n_next
    } else {
        let start = (i - 2 * nu) * layout.n_next;
        layout.n_next.min(layout.m - start)
    }
}

/// The final witness: for each vector, one byte holding the width $`w`$ of the signed code,
/// the least for its coefficients, then the coefficients of its polynomials that carry values
/// ([`final_length`]), zero padded to a byte. The padding of $`v`$ is not sent.
pub(crate) fn encode_final(
    ring: &Rq,
    layout: &Layout,
    nu: usize,
    witness: &[Zeroizing<Vec<u64>>],
) -> Result<Vec<u8>, Error> {
    let d = ring.degree();
    let mut out = Vec::new();
    for (i, v) in witness.iter().enumerate() {
        let len = final_length(layout, nu, i) * d;
        debug_assert!(v[len..].iter().all(|x| *x == 0));
        let width = signed_width(u128::from(ring.linf(&v[..len])));
        out.push(width as u8);
        let values: Vec<i128> = v[..len]
            .iter()
            .map(|x| i128::from(ring.center(*x)))
            .collect();
        out.extend_from_slice(&encode_signed(&values, width)?);
    }
    Ok(out)
}

/// The inverse of [`encode_final`]: refuses a width that is not the least, or a coefficient
/// outside $`[-(q-1)/2,(q-1)/2]`$.
pub(crate) fn decode_final(
    ring: &Rq,
    layout: &Layout,
    nu: usize,
    input: &mut &[u8],
) -> Result<Vec<Zeroizing<Vec<u64>>>, Error> {
    let d = ring.degree();
    let half = (ring.modulus() / 2) as u128;
    let max_width = signed_width(half);
    let mut out = Vec::with_capacity(layout.r_next);
    for i in 0..layout.r_next {
        let count = final_length(layout, nu, i) * d;
        let width = u32::from(take(input, 1)?[0]);
        if !(2..=max_width).contains(&width) {
            return Err(Error::Encoding);
        }
        let bytes = take(input, (count * width as usize).div_ceil(8))?;
        let values = decode_signed(bytes, count, width)?;
        let max = values.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0);
        if max > half || signed_width(max) != width {
            return Err(Error::Encoding);
        }
        let mut v: Vec<u64> = values.iter().map(|x| ring.lift_i128(*x)).collect();
        v.resize(layout.n_next * d, 0);
        out.push(Zeroizing::new(v));
    }
    Ok(out)
}
