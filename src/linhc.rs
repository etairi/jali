//! Extractable linear homomorphic commitment (LinHC) of Katsumata (CRYPTO 2021, ePrint
//! 2021/927), the MLWE construction of Sec. 3.4 (Fig. 4 and 5), the building block of
//! straight-line extraction.
//!
//! A message $`e\in R^{m'}`$ over $`R_{q'}=\mathbb Z_{q'}[X]/(X^d+1)`$ is committed as
//! $`t_1=p(As_1+s_2)`$ and $`t_2=p(Bs_1+s_3)+e`$, with uniform ternary $`s_1\in R^n`$ and
//! $`s_2,s_3\in R^{m'}`$, and a mask $`r`$ as $`w_1=p(Ay_1+y_2)`$ and $`w_2=p(By_1+y_3)+r`$,
//! with Gaussian $`y_\ell`$ of width $`\sigma'=1.55\cdot2^{t'}`$. For a challenge $`c`$ the
//! opening $`z'_\ell=cs_\ell+y_\ell`$ passes a rejection test that makes it independent of
//! the $`s_\ell`$, and a verifier recomputes $`w`$ from $`c`$, $`z=ce+r`$ and $`z'`$.
//!
//! Under a trapdoored key $`B=D_1A+D_2`$, with ternary $`D_1`$ and $`D_2`$, every combination
//! $`ct+w`$ that has an opening within the verifier's bounds decrypts:
//! $`(ct_2+w_2)-D_1(ct_1+w_1)=p(D_2z'_1-D_1z'_2+z'_3)+z`$, and [`LinhcParams::check`] makes
//! the right side smaller than $`q'/2`$ in absolute value, so its residue modulo $`p`$ is
//! $`z`$. The uniform key of honest use and the trapdoored key are indistinguishable under
//! MLWE.
//!
//! The constants differ from the paper's in two places, both needed for correctness:
//! - the shift $`cs`$ is bounded with the challenges' operator norm, $`\|cs\|\le\eta\|s\|`$
//!   (`docs/parameters.md`), where the paper's $`T`$ uses $`\|c\|_\infty`$, which bounds
//!   $`cs`$ only for monomial challenges,
//! - each block of the opening is checked against its own bound $`2\dim_\ell\sigma'^2`$ on
//!   its squared norm, where the paper checks every block against $`\sqrt{2nd}\,\phi T`$,
//!   which honest openings of the blocks of dimension $`m'd`$ exceed once $`m'>2n`$.
#[cfg(test)]
use crate::rand::{ShakePrg, uniform_poly};
use crate::{
    Error,
    abdlop::{self, Mask},
    codec::{BitWriter, fields},
    math::{Poly, PolyMat, PolyVec, Ring, U256, int},
    rand::{
        AesPrg, bounded, domain,
        reject::{self, COIN_BYTES, Policy},
    },
    transcript::Transcript,
};
use std::sync::Arc;
use zeroize::Zeroizing;

#[cfg(test)]
mod tests;

/// Domains of the key matrices $`A`$ and $`B`$ under the key seed of the tests' keys ("LHCA",
/// "LHCB").
#[cfg(test)]
const KEY_A: u32 = 0x4c48_4341;
#[cfg(test)]
const KEY_B: u32 = 0x4c48_4342;
/// Domains of the trapdoor matrices $`D_1`$ and $`D_2`$ ("LHD1", "LHD2").
#[cfg(test)]
const TRAPDOOR_D1: u32 = 0x4c48_4431;
#[cfg(test)]
const TRAPDOOR_D2: u32 = 0x4c48_4432;

/// The largest mask exponent: $`2\dim\sigma'^2=961\cdot4^{t'}\dim/200`$ stays far below
/// $`2^{128}`$ for every dimension a key matrix allows.
const MAX_LOG_SIGMA: u32 = 40;

/// Parameters of a LinHC instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LinhcParams {
    /// Ring degree $`d`$, the proof degree, so that the proof's challenges act on the ring.
    pub(crate) degree: usize,
    /// Rank $`n`$ of $`s_1`$: commitments hide under MLWE of dimension $`nd`$.
    pub(crate) rank: usize,
    /// Message length $`m'`$ in polynomials.
    pub(crate) len: usize,
    /// Prime modulus $`q'`$, below $`2^{127}`$.
    pub(crate) modulus: u128,
    /// Odd plaintext modulus $`p<q'`$.
    pub(crate) plaintext: u128,
    /// Exponent $`t'`$ of the mask width $`\sigma'=1.55\cdot2^{t'}`$.
    pub(crate) log_sigma: u32,
    /// Bound $`B`$ on $`\|z\|_\infty`$ for every message the caller's verifier accepts.
    pub(crate) message_bound: u128,
    /// Operator-norm bound $`\eta`$ of the challenges.
    pub(crate) eta: u64,
}

/// The quantities [`LinhcParams::check`] derives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LinhcChecked {
    /// $`T^2=\eta^2(n+2m')d`$, at least $`\|cs\|^2`$ for every challenge and ternary $`s`$.
    pub(crate) shift_bound_squared: u128,
    /// The verifier's bounds $`\lfloor2\dim_\ell\sigma'^2\rfloor`$ on $`\|z'_\ell\|^2`$, for
    /// the dimensions $`nd`$, $`m'd`$ and $`m'd`$.
    pub(crate) block_bounds_squared: [u128; 3],
    /// $`N`$, at least $`\|D_2z'_1-D_1z'_2+z'_3\|_\infty`$ for ternary $`D_1`$ and $`D_2`$
    /// and every opening within the block bounds.
    pub(crate) noise_bound: u128,
    /// $`\lceil M2^{128}\rceil`$ for the rejection test, with $`M`$ at least
    /// $`\exp(\sqrt{384\ln2}/\gamma+1/(2\gamma^2))`$ and $`\gamma=\sigma'/T`$.
    pub(crate) m_scaled: U256,
}

impl LinhcParams {
    /// The ring $`R_{q'}`$.
    pub(crate) fn ring(&self) -> Result<Arc<Ring>, Error> {
        let q = i128::try_from(self.modulus).map_err(|_| Error::Parameter("LinHC modulus"))?;
        Ring::new(q, self.degree)
    }

    /// Check the parameters and derive the bounds.
    ///
    /// Refuses a degree outside the ring's range, empty or oversized shapes, a mask exponent
    /// above 40, a plaintext modulus that is even or at most $`4B`$ (two messages within
    /// $`B`$ then differ by less than $`p/2`$, so each combination decrypts to at most one of
    /// them), a composite modulus or one at most $`p`$, a modulus with
    /// $`2(pN+B)\ge q'`$ (decryption), and a rejection constant of $`2^{16}`$ or more.
    pub(crate) fn check(&self) -> Result<LinhcChecked, Error> {
        if !(64..=1024).contains(&self.degree) || !self.degree.is_power_of_two() {
            return Err(Error::Parameter("LinHC degree"));
        }
        if self.rank == 0 || self.len == 0 {
            return Err(Error::Parameter("LinHC dimensions"));
        }
        abdlop::matrix_size(self.len, self.rank)?;
        if self.log_sigma == 0 || self.log_sigma > MAX_LOG_SIGMA {
            return Err(Error::Parameter("LinHC mask exponent"));
        }
        let p = self.plaintext;
        let b = self.message_bound;
        if p < 3 || p.is_multiple_of(2) || b == 0 || b.checked_mul(4).is_none_or(|x| x >= p) {
            return Err(Error::Parameter("LinHC plaintext modulus"));
        }
        let q = self.modulus;
        if q <= p || q >= 1 << 127 || !is_prime(q) {
            return Err(Error::Parameter("LinHC modulus"));
        }
        let (shift_bound_squared, block_bounds_squared, noise_bound) = self.bounds()?;
        let reach = p
            .checked_mul(noise_bound)
            .and_then(|x| x.checked_add(b))
            .and_then(|x| x.checked_mul(2))
            .ok_or(Error::Overflow)?;
        if reach >= q {
            return Err(Error::Parameter("LinHC modulus below the decryption bound"));
        }
        Ok(LinhcChecked {
            shift_bound_squared,
            block_bounds_squared,
            noise_bound,
            m_scaled: rejection_constant(self.log_sigma, shift_bound_squared)?,
        })
    }

    /// $`T^2`$, the block bounds and $`N`$, which depend on neither modulus.
    fn bounds(&self) -> Result<(u128, [u128; 3], u128), Error> {
        let d = self.degree as u128;
        let (n, m) = (self.rank as u128, self.len as u128);
        let eta = u128::from(self.eta);
        let shift_bound_squared = eta
            .checked_mul(eta)
            .and_then(|x| x.checked_mul(n + 2 * m))
            .and_then(|x| x.checked_mul(d))
            .ok_or(Error::Overflow)?;
        // 2 dim sigma'^2 = 2 dim 961 4^t / 400 = 961 4^t dim / 200, rounded down.
        let variance = 961u128 << (2 * self.log_sigma);
        let dims = [n * d, m * d, m * d];
        let mut block_bounds_squared = [0u128; 3];
        for (bound, dim) in block_bounds_squared.iter_mut().zip(dims) {
            *bound = variance.checked_mul(dim).ok_or(Error::Overflow)? / 200;
        }
        // A coefficient of D_2 z'_1 is the inner product of a ternary vector of length nd with
        // the coefficients of z'_1 (signed by the negacyclic wraparound), so at most
        // sqrt(nd) |z'_1|, and likewise for D_1 z'_2. |z'_3|_inf is at most |z'_3|.
        let beta = block_bounds_squared.map(ceil_sqrt);
        let noise_bound = ceil_sqrt(dims[0])
            .checked_mul(beta[0])
            .and_then(|x| x.checked_add(ceil_sqrt(dims[1]).checked_mul(beta[1])?))
            .and_then(|x| x.checked_add(beta[2]))
            .ok_or(Error::Overflow)?;
        Ok((shift_bound_squared, block_bounds_squared, noise_bound))
    }
}

/// The smallest parameters for a shape: the least $`t'`$ with $`\sigma'\ge\phi T`$,
/// $`p=4B+1`$, and the least prime $`q'`$ above $`2(pN+B)`$.
pub(crate) fn derive(
    degree: usize,
    rank: usize,
    len: usize,
    message_bound: u128,
    eta: u64,
    phi: u32,
) -> Result<LinhcParams, Error> {
    let mut params = LinhcParams {
        degree,
        rank,
        len,
        modulus: 0,
        plaintext: message_bound
            .checked_mul(4)
            .and_then(|x| x.checked_add(1))
            .ok_or(Error::Overflow)?,
        log_sigma: 1,
        message_bound,
        eta,
    };
    let (shift_bound_squared, ..) = params.bounds()?;
    // sigma' >= phi T, that is 961 4^t / 400 >= phi^2 T^2.
    let target = u128::from(phi)
        .checked_mul(u128::from(phi))
        .and_then(|x| x.checked_mul(shift_bound_squared))
        .and_then(|x| x.checked_mul(400))
        .ok_or(Error::Overflow)?;
    while (961u128 << (2 * params.log_sigma)) < target {
        params.log_sigma += 1;
        if params.log_sigma > MAX_LOG_SIGMA {
            return Err(Error::Parameter("LinHC mask exponent"));
        }
    }
    let (_, _, noise_bound) = params.bounds()?;
    let floor = params
        .plaintext
        .checked_mul(noise_bound)
        .and_then(|x| x.checked_add(message_bound))
        .and_then(|x| x.checked_mul(2))
        .and_then(|x| x.checked_add(1))
        .ok_or(Error::Overflow)?;
    params.modulus = next_prime(floor)?;
    params.check()?;
    Ok(params)
}

/// $`\lceil M2^{128}\rceil`$ with $`M`$ an upper bound on
/// $`\exp(\sqrt{384\ln2}/\gamma+1/(2\gamma^2))`$, $`\gamma=\sigma'/T`$: the form of the
/// crate's Rej_1 constant (Lyubashevsky, "Lattice signatures without trapdoors", EUROCRYPT 2012,
/// Lemmas 4.3 and 4.6, as Katsumata's Lemma 2.10 states them), at the statistical
/// error $`2^{-192}`$ instead of Rej_1's $`2^{-129}`$, so that the LinHC test adds a term below
/// every other of the whole-proof bound (`docs/security.md`), for about 5% more attempts at
/// $`\gamma=53`$. Not rounded to an integer: a relative margin of $`2^{-40}`$ covers the
/// rounding of the `f64` operations, so $`M`$ stays an upper bound and the accepted openings'
/// law stays the target's up to that error.
fn rejection_constant(t: u32, shift_bound_squared: u128) -> Result<U256, Error> {
    let sigma = 1.55 * (2.0f64).powi(t as i32);
    let gamma = sigma / (shift_bound_squared as f64).sqrt();
    let x = (384.0 * std::f64::consts::LN_2).sqrt() / gamma + 0.5 / (gamma * gamma);
    let m = x.exp() * (1.0 + (2.0f64).powi(-40));
    if !m.is_finite() || !(1.0..65536.0).contains(&m) {
        return Err(Error::Parameter("LinHC rejection constant"));
    }
    // m = mantissa 2^exponent with exponent in [-52, -37] for m in [1, 2^16), so m 2^128 is
    // the integer mantissa 2^(128 + exponent), exactly.
    let bits = m.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1075;
    let mantissa = (bits & ((1 << 52) - 1)) | (1 << 52);
    Ok(U256::from_u64(mantissa).shl_vartime((128 + exponent) as u32))
}

fn is_prime(q: u128) -> bool {
    match u64::try_from(q) {
        Ok(q) => int::is_prime(q),
        Err(_) => int::is_prime_u256(&U256::from_u128(q)),
    }
}

fn next_prime(from: u128) -> Result<u128, Error> {
    let mut q = from | 1;
    while !is_prime(q) {
        q = q.checked_add(2).ok_or(Error::Overflow)?;
    }
    Ok(q)
}

pub(crate) fn ceil_sqrt(x: u128) -> u128 {
    let mut r = (x as f64).sqrt() as u128;
    while r.checked_mul(r).is_some_and(|s| s < x) {
        r += 1;
    }
    while r > 0 && (r - 1) * (r - 1) >= x {
        r -= 1;
    }
    r
}

/// `p` over `ring`, from its centred coefficients.
pub(crate) fn lift_poly(ring: &Arc<Ring>, p: &Poly) -> Result<Poly, Error> {
    let mut values = p.coefficients_i128()?;
    Poly::new(ring.clone(), std::mem::take(&mut *values))
}

/// A commitment key $`(A,B)`$, both $`m'\times n`$ over $`R_{q'}`$.
#[derive(Clone)]
pub(crate) struct LinhcKey {
    pub(crate) a: PolyMat,
    pub(crate) b: PolyMat,
}

impl LinhcKey {
    /// The key of honest use, squeezed from the transcript: $`A`$ then $`B`$, each entry of
    /// each in row-major order, each coefficient by `DecodeUint` modulo $`q'`$ from
    /// $`N_s+16`$ bytes ([`Transcript::ring_elements`]). The extraction argument models the
    /// sponge as a random oracle with values in the key space and programs it to a key with a
    /// trapdoor.
    pub(crate) fn squeeze(
        ring: &Arc<Ring>,
        params: &LinhcParams,
        transcript: &mut Transcript,
    ) -> Result<Self, Error> {
        let size = abdlop::matrix_size(params.len, params.rank)?;
        let a = PolyMat::new(
            ring.clone(),
            params.len,
            params.rank,
            transcript.ring_elements(ring, size)?,
        )?;
        let b = PolyMat::new(
            ring.clone(),
            params.len,
            params.rank,
            transcript.ring_elements(ring, size)?,
        )?;
        Ok(Self { a, b })
    }
    /// Test only: uniform matrices expanded from `seed` by SHAKE128 ([`shake_matrix`]).
    #[cfg(test)]
    pub(crate) fn expand(
        ring: &Arc<Ring>,
        params: &LinhcParams,
        seed: &[u8; 32],
    ) -> Result<Self, Error> {
        Ok(Self {
            a: shake_matrix(ring, params.len, params.rank, seed, KEY_A)?,
            b: shake_matrix(ring, params.len, params.rank, seed, KEY_B)?,
        })
    }
}

/// Test only: a matrix of uniform entries over `ring`, entry $`i`$ (row-major) read from
/// SHAKE128 of `seed` and the domain $`(\mathrm{dom},i)`$, for the keys of the tests.
#[cfg(test)]
fn shake_matrix(
    ring: &Arc<Ring>,
    rows: usize,
    cols: usize,
    seed: &[u8; 32],
    dom: u32,
) -> Result<PolyMat, Error> {
    let entries = (0..abdlop::matrix_size(rows, cols)?)
        .map(|i| uniform_poly(&mut ShakePrg::new(seed, domain(dom, i as u32)), ring))
        .collect::<Result<Vec<_>, Error>>()?;
    PolyMat::new(ring.clone(), rows, cols, entries)
}

/// The secret of a commitment: ternary $`s_1`$, $`s_2`$ and $`s_3`$, wiped on drop.
pub(crate) struct LinhcSecret {
    s: [PolyVec; 3],
}

/// A commitment $`(t_1,t_2)`$.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LinhcCommitment {
    pub(crate) t1: PolyVec,
    pub(crate) t2: PolyVec,
}

/// Gaussian masks $`y_1`$, $`y_2`$, $`y_3`$ and their commitment $`(w_1,w_2)`$ to a mask
/// message.
pub(crate) struct LinhcMask {
    y: [Mask; 3],
    pub(crate) w1: PolyVec,
    pub(crate) w2: PolyVec,
}

/// An opening $`(z'_1,z'_2,z'_3)`$.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LinhcOpening {
    pub(crate) z: [PolyVec; 3],
}

/// $`(p(Ax_1+x_2),\,p(Bx_1+x_3)+e)`$, the shape shared by commitments, masks and openings.
fn combine(
    key: &LinhcKey,
    params: &LinhcParams,
    x: [&PolyVec; 3],
    message: &PolyVec,
) -> Result<(PolyVec, PolyVec), Error> {
    let p = U256::from_u128(params.plaintext);
    // A x_0 and B x_0 with one transform of x_0.
    let (ax, bx) = key.a.mul_pair(&key.b, x[0])?;
    let u = abdlop::scale_scalar(&abdlop::add(&ax, x[1])?, &p)?;
    let v = abdlop::add(
        &abdlop::scale_scalar(&abdlop::add(&bx, x[2])?, &p)?,
        message,
    )?;
    Ok((u, v))
}

fn check_message(key: &LinhcKey, params: &LinhcParams, message: &PolyVec) -> Result<(), Error> {
    if message.len() != params.len {
        return Err(Error::Dimension);
    }
    if message.ring() != key.a.ring() {
        return Err(Error::RingMismatch);
    }
    Ok(())
}

/// Commit to `message` (over the key's ring) with ternary randomness read from `seed` under
/// the domains $`(\mathrm{dom},i)`$, $`i\ge1`$.
pub(crate) fn commit(
    key: &LinhcKey,
    params: &LinhcParams,
    message: &PolyVec,
    seed: &[u8; 32],
    dom: u32,
) -> Result<(LinhcCommitment, LinhcSecret), Error> {
    check_message(key, params, message)?;
    let ring = key.a.ring().clone();
    let mut index = 0u32;
    let mut ternary = |len: usize| -> Result<PolyVec, Error> {
        let mut entries = Vec::with_capacity(len);
        for _ in 0..len {
            index = index.checked_add(1).ok_or(Error::Overflow)?;
            let mut values = Zeroizing::new(bounded(
                &mut AesPrg::new(seed, domain(dom, index)),
                -1,
                1,
                params.degree,
            )?);
            entries.push(Poly::new(ring.clone(), std::mem::take(&mut *values))?);
        }
        PolyVec::new(ring.clone(), entries)
    };
    let s = [
        ternary(params.rank)?,
        ternary(params.len)?,
        ternary(params.len)?,
    ];
    let (t1, t2) = combine(key, params, [&s[0], &s[1], &s[2]], message)?;
    Ok((LinhcCommitment { t1, t2 }, LinhcSecret { s }))
}

/// Gaussian masks read from `seed` under the domains `dom`, `dom + 1` and `dom + 2`, and
/// their commitment to the mask message `r`.
pub(crate) fn mask(
    key: &LinhcKey,
    params: &LinhcParams,
    r: &PolyVec,
    seed: &[u8; 32],
    dom: u32,
) -> Result<LinhcMask, Error> {
    check_message(key, params, r)?;
    let ring = key.a.ring().clone();
    let doms = [
        dom,
        dom.checked_add(1).ok_or(Error::Overflow)?,
        dom.checked_add(2).ok_or(Error::Overflow)?,
    ];
    let lens = [params.rank, params.len, params.len];
    let mut y = Vec::with_capacity(3);
    for (len, dom) in lens.into_iter().zip(doms) {
        y.push(abdlop::sample_gaussian(
            ring.clone(),
            len,
            params.log_sigma,
            seed,
            dom,
        )?);
    }
    let y: [Mask; 3] = y.try_into().map_err(|_| Error::Dimension)?;
    let (w1, w2) = combine(key, params, [&y[0].values, &y[1].values, &y[2].values], r)?;
    Ok(LinhcMask { y, w1, w2 })
}

/// The opening for challenge `c` (over the key's ring), or `None` when the rejection test,
/// read with the 256-bit `coin`, rejects.
pub(crate) fn open(
    checked: &LinhcChecked,
    c: &Poly,
    secret: &LinhcSecret,
    mask: &LinhcMask,
    coin: &[u8; COIN_BYTES],
) -> Result<Option<LinhcOpening>, Error> {
    let mut shifts = Vec::with_capacity(3);
    let mut z = Vec::with_capacity(3);
    for (s, y) in secret.s.iter().zip(&mask.y) {
        let shift = abdlop::scale(s, c)?;
        z.push(abdlop::add(&shift, &y.values)?);
        shifts.push(shift);
    }
    let mut z_flat = Zeroizing::new(Vec::new());
    let mut v_flat = Zeroizing::new(Vec::new());
    for (z, v) in z.iter().zip(&shifts) {
        z_flat.extend_from_slice(&abdlop::flatten(z)?);
        v_flat.extend_from_slice(&abdlop::flatten(v)?);
    }
    let (dot, norm) = reject::moments(&z_flat, &v_flat)?;
    let accepted = reject::accept(
        Policy::Standard,
        dot,
        norm,
        mask.y[0].variance,
        checked.m_scaled,
        reject::coin(coin),
    )?;
    let z: [PolyVec; 3] = z.try_into().map_err(|_| Error::Dimension)?;
    Ok(accepted.then_some(LinhcOpening { z }))
}

/// The verifier's side: check each block of the opening against its bound, and recompute
/// $`w_1=p(Az'_1+z'_2)-ct_1`$ and $`w_2=p(Bz'_1+z'_3)+z-ct_2`$ for the message `message`
/// $`=z`$. A Fiat–Shamir verifier compares the challenge it derives from these with `c`.
pub(crate) fn recompute(
    key: &LinhcKey,
    params: &LinhcParams,
    checked: &LinhcChecked,
    c: &Poly,
    commitment: &LinhcCommitment,
    message: &PolyVec,
    opening: &LinhcOpening,
) -> Result<(PolyVec, PolyVec), Error> {
    check_message(key, params, message)?;
    let lens = [params.rank, params.len, params.len];
    for ((z, len), bound) in opening.z.iter().zip(lens).zip(checked.block_bounds_squared) {
        if z.len() != len || commitment.t1.len() != params.len || commitment.t2.len() != params.len
        {
            return Err(Error::Dimension);
        }
        if z.ring() != key.a.ring() || c.ring() != key.a.ring() {
            return Err(Error::RingMismatch);
        }
        if z.norm_squared()? > U256::from_u128(bound) {
            return Err(Error::InvalidProof);
        }
    }
    let (u, v) = combine(
        key,
        params,
        [&opening.z[0], &opening.z[1], &opening.z[2]],
        message,
    )?;
    Ok((
        abdlop::sub(&u, &abdlop::scale(&commitment.t1, c)?)?,
        abdlop::sub(&v, &abdlop::scale(&commitment.t2, c)?)?,
    ))
}

/// The mask commitment $`(w_1,w_2)`$ of one attempt, borrowed.
pub(crate) type MaskCommitment<'a> = (&'a PolyVec, &'a PolyVec);

/// What the opening proof's prover needs to run the LinHC beside it: the key, the parameters and
/// the secret of the commitment to $`e=(s_1\|s_{21})`$.
pub(crate) struct ProverExt<'a> {
    pub(crate) key: &'a LinhcKey,
    pub(crate) params: &'a LinhcParams,
    pub(crate) checked: &'a LinhcChecked,
    pub(crate) secret: &'a LinhcSecret,
}

/// What the opening proof's verifier needs: the key, the parameters, and the commitment and
/// opening from the proof.
pub(crate) struct VerifierExt<'a> {
    pub(crate) key: &'a LinhcKey,
    pub(crate) params: &'a LinhcParams,
    pub(crate) checked: &'a LinhcChecked,
    pub(crate) commitment: &'a LinhcCommitment,
    pub(crate) opening: &'a LinhcOpening,
}

/// The LinHC message $`(x\|y)`$ over `ring`, from the centred coefficients of `x` and `y`: the
/// response $`(z_1\|z_{21})`$, its mask $`(y_1\|y_{21})`$ or the witness $`(s_1\|s_{21})`$.
pub(crate) fn message(ring: &Arc<Ring>, x: &PolyVec, y: &PolyVec) -> Result<PolyVec, Error> {
    PolyVec::new(
        ring.clone(),
        x.entries()
            .iter()
            .chain(y.entries())
            .map(|p| lift_poly(ring, p))
            .collect::<Result<_, _>>()?,
    )
}

/// Write a pair of vectors over the LinHC ring, each coefficient modulo $`q'`$ in
/// $`\mathrm{bitlen}(q'-1)`$ bits: the serialization of a commitment $`(t_1,t_2)`$ or a mask
/// commitment $`(w_1,w_2)`$.
pub(crate) fn write_pair(w: &mut BitWriter, u: &PolyVec, v: &PolyVec) -> Result<(), Error> {
    fields::write_below(w, u, &u.ring().modulus())?;
    fields::write_below(w, v, &v.ring().modulus())
}

/// The trapdoor $`(D_1,D_2)`$ of a key with $`B=D_1A+D_2`$.
#[cfg(test)]
pub(crate) struct Trapdoor {
    pub(crate) d1: PolyMat,
    pub(crate) d2: PolyMat,
}

#[cfg(test)]
impl LinhcKey {
    /// A key with a trapdoor: $`A`$ as in [`LinhcKey::expand`], ternary $`D_1`$
    /// ($`m'\times m'`$) and $`D_2`$ ($`m'\times n`$) read from `seed`, and $`B=D_1A+D_2`$.
    pub(crate) fn with_trapdoor(
        ring: &Arc<Ring>,
        params: &LinhcParams,
        seed: &[u8; 32],
    ) -> Result<(Self, Trapdoor), Error> {
        let (m, n) = (params.len, params.rank);
        let a = shake_matrix(ring, m, n, seed, KEY_A)?;
        let ternary = |rows: usize, cols: usize, dom: u32| -> Result<PolyMat, Error> {
            let entries = (0..rows * cols)
                .map(|i| {
                    let values = bounded(
                        &mut AesPrg::new(seed, domain(dom, i as u32)),
                        -1,
                        1,
                        params.degree,
                    )?;
                    Poly::new(ring.clone(), values)
                })
                .collect::<Result<_, _>>()?;
            PolyMat::new(ring.clone(), rows, cols, entries)
        };
        let d1 = ternary(m, m, TRAPDOOR_D1)?;
        let d2 = ternary(m, n, TRAPDOOR_D2)?;
        let mut entries = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = d2.get(i, j)?.clone();
                for k in 0..m {
                    sum = sum.add(&d1.get(i, k)?.mul(a.get(k, j)?)?)?;
                }
                entries.push(sum);
            }
        }
        let b = PolyMat::new(ring.clone(), m, n, entries)?;
        Ok((Self { a, b }, Trapdoor { d1, d2 }))
    }
}

/// Decrypt a combination $`(v_1,v_2)`$: $`v_2-D_1v_1`$ centred modulo $`q'`$, then each
/// coefficient centred modulo $`p`$, one vector of coefficients per message polynomial.
#[cfg(test)]
pub(crate) fn decrypt(
    trapdoor: &Trapdoor,
    plaintext: u128,
    v1: &PolyVec,
    v2: &PolyVec,
) -> Result<Vec<Vec<i128>>, Error> {
    let u = abdlop::sub(v2, &trapdoor.d1.mul(v1)?)?;
    let p = i128::try_from(plaintext).map_err(|_| Error::Overflow)?;
    u.entries()
        .iter()
        .map(|poly| {
            Ok(poly
                .coefficients_i128()?
                .iter()
                .map(|x| {
                    let r = x.rem_euclid(p);
                    if r > p / 2 { r - p } else { r }
                })
                .collect())
        })
        .collect()
}
