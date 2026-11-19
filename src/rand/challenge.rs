//! The challenges of the opening proof: the set of LNP22 §2.7, $`\sigma_{-1}`$-stable
//! polynomials with coefficients in $`[-\omega,\omega]`$ and coefficient $`d/2`$ zero whose
//! operator-norm bound $`\|\sigma_{-1}(c^\kappa)c^\kappa\|_1^{1/(2\kappa)}`$, $`\kappa=32`$,
//! is at most $`\eta`$.
//!
//! **What the bound bounds.** Multiplication by $`c`$ in $`\mathbb Z[X]/(X^d+1)`$ is
//! diagonalised by evaluation at the primitive $`2d`$-th roots of unity $`\zeta`$, with
//! eigenvalues $`c(\zeta)`$. As $`\sigma_{-1}`$ is a ring automorphism,
//! $`\sigma_{-1}(c^\kappa)c^\kappa=(\sigma_{-1}(c)c)^\kappa`$, and for integer coefficients
//! $`(\sigma_{-1}(c)c)(\zeta)=c(\bar\zeta)c(\zeta)=|c(\zeta)|^2`$. With
//! $`|P(\zeta)|\le\|P\|_1`$, every eigenvalue satisfies
//! $`|c(\zeta)|^{2\kappa}\le\|(\sigma_{-1}(c)c)^\kappa\|_1`$.
//!
//! **The exact test** is $`\|(\sigma_{-1}(c)c)^{32}\|_1\le\eta^{64}`$, in integers. The product
//! $`u=\sigma_{-1}(c)c`$ is $`\sigma_{-1}`$-stable for every $`c`$, and so are its powers, so
//! only coefficients $`0`$ to $`d/2-1`$ are computed: $`u_{d/2}=0`$ and $`u_{d-i}=-u_i`$. Five
//! squarings give $`u^{32}`$. For the $`j`$-th product ($`j=1`$ for $`u`$, 2 to 6 for the
//! squarings) every coefficient and every partial sum is at most $`\|a\|_1\|a\|_\infty\le
//! L^{2^j}`$ in absolute value, where $`a`$ is a factor of that product and $`L=\|c\|_1`$
//! (as $`\|xy\|_1\le\|x\|_1\|y\|_1`$). For $`L\le2^{15}`$ that is at most $`2^{30}`$,
//! $`2^{60}`$ and $`2^{120}`$ in `i128`, $`2^{240}`$ in `I256`, $`2^{480}`$ in `I512` and
//! $`2^{960}`$ in `I1024`, and so is the norm; every operation is checked all the same. A
//! challenge has $`L\le(d-1)\omega`$: 504 for $`(d,\omega)=(64,8)`$ and 254 for $`(128,2)`$.
//! No floating point is involved, and the integer types are named by bit width, so the
//! decision is the same on every platform and limb width.
//!
//! **Early acceptance.** As $`\|xy\|_1\le\|x\|_1\|y\|_1`$,
//! $`\|u^{32}\|_1\le\|u^{2^{j-1}}\|_1^{2^{6-j}}`$ for the norm $`N_j=\|u^{2^{j-1}}\|_1`$ after
//! the $`j`$-th product, and $`N_j^{2^{6-j}}\le\eta^{64}`$ if and only if $`N_j\le\eta^{2^j}`$.
//! [`within_eta`] and [`challenge`] therefore accept as soon as some $`N_j\le\eta^{2^j}`$,
//! and otherwise decide by $`N_6=\|u^{32}\|_1`$ itself: the same decision as the exact test,
//! for most challenges after the three products in `i128` (measured on 200,000 uniform draws
//! per degree: 98.9% decided by $`j=3`$ at degree 64 and 98.2% at degree 128).
//!
//! **Early rejection.** The powers $`y=u^{2^{j-1}}`$ are $`\sigma_{-1}`$-stable, so
//! $`y^2=y\,\sigma_{-1}(y)`$, whose constant coefficient is $`\sum_iy_i^2=\|y\|_2^2`$, at
//! least $`\|y\|_1^2/d`$ by Cauchy-Schwarz. Hence $`N_{j+1}\ge N_j^2/d`$, and by induction
//! $`N_6\ge N_j^{2^{6-j}}/d^{2^{6-j}-1}`$. [`within_eta`] and [`challenge`] therefore also reject
//! as soon as $`N_j^{2^{6-j}}>\eta^{64}d^{2^{6-j}-1}`$ for $`j=3`$ to 5, compared exactly: the
//! same decision again. Most draws of LaBRADOR's challenge set are rejected (about five in six),
//! and about 78% of those are decided so before the last product (3,000 draws of its set,
//! computed exactly in Python, 2026-11-19), which is most of the time of the test.
use super::{ByteStream, autostable};
use crate::{
    Error,
    math::{Poly, Ring},
};
use crypto_bigint::{
    CheckedAdd, CheckedSub, Concat, I128, I256, I512, I1024, Int, U1024, U2048, Uint,
};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

/// The draws [`challenge`] makes before it fails. Each draw of the shipped sets is rejected
/// with probability about 1.2% at most, so 64 rejections have probability below $`2^{-400}`$.
pub const MAX_CHALLENGE_DRAWS: u32 = 64;

/// The largest $`\|c\|_1`$ for which the integer types of [`eta_norm_power`] hold every value,
/// $`2^{15}`$ (see the module documentation).
const MAX_L1: u128 = 1 << 15;

/// $`\|(\sigma_{-1}(c)c)^{32}\|_1`$, exactly: the $`2\kappa`$-th power of the operator-norm
/// bound of LNP22 §2.7 for $`\kappa=32`$ (see the module documentation). Fails with
/// [`Error::Parameter`] if $`\|c\|_1>2^{15}`$, beyond the capacity of the integer types, and
/// with [`Error::Overflow`] if a centred coefficient does not fit `i128`.
pub fn eta_norm_power(c: &Poly) -> Result<U1024, Error> {
    stage_norms(c, |_, _| false)
}

/// $`\eta^{2^j}`$ for $`j=0`$ to 6 (at index $`j`$), for $`\eta<2^{16}`$
/// ([`Error::Parameter`] otherwise), so that $`\eta^{64}`$ fits.
fn eta_powers(eta: u64) -> Result<[U1024; 7], Error> {
    if eta >= 1 << 16 {
        return Err(Error::Parameter("challenge eta capacity"));
    }
    let mut powers = [U1024::from_u64(eta); 7];
    for j in 1..7 {
        powers[j] = Option::from(powers[j - 1].checked_square()).ok_or(Error::Overflow)?;
    }
    Ok(powers)
}

/// The thresholds of the exact test for one $`\eta`$ and degree $`d`$: $`\eta^{2^j}`$ for the
/// early acceptance at stage $`j`$ (0 to 6), and $`\eta^{64}d^{2^{6-j}-1}`$ for the early
/// rejection at stages 3 to 5 (module documentation).
struct Bounds {
    accept: [U1024; 7],
    reject: [Option<U2048>; 7],
}

/// The [`Bounds`] of $`(\eta,d)`$, computed once per process for each pair, with the errors of
/// [`eta_powers`].
fn bounds(eta: u64, d: usize) -> Result<Arc<Bounds>, Error> {
    type Memo = Mutex<HashMap<(u64, usize), Arc<Bounds>>>;
    static MEMO: OnceLock<Memo> = OnceLock::new();
    let memo = MEMO.get_or_init(Default::default);
    if let Some(b) = memo.lock().expect("challenge bounds").get(&(eta, d)) {
        return Ok(b.clone());
    }
    let accept = eta_powers(eta)?;
    // eta^64 < 2^1024 and d^31 < 2^320 for the degrees of rings, so no threshold overflows;
    // one that did would only disable its early rejection.
    let top: U2048 = accept[6].resize();
    let degree = U2048::from_u64(d as u64);
    let mut reject = [None; 7];
    for (j, slot) in reject.iter_mut().enumerate().take(6).skip(3) {
        let factor =
            (1..1u32 << (6 - j)).try_fold(U2048::ONE, |f, _| f.checked_mul(&degree).into_option());
        *slot = factor.and_then(|f| top.checked_mul(&f).into_option());
    }
    let b = Arc::new(Bounds { accept, reject });
    memo.lock()
        .expect("challenge bounds")
        .insert((eta, d), b.clone());
    Ok(b)
}

/// Whether $`N^{2^{6-j}}`$ exceeds `threshold`, exactly: by bit lengths where they decide
/// ($`2^{(b-1)e}\le N^e<2^{be}`$ for the bit length $`b`$ of $`N`$ and $`e=2^{6-j}`$), and by the
/// power in 2048 bits otherwise, which then stays below $`2^{be}\le2^{t+e}`$ for the bit length
/// $`t\le1344`$ of the threshold.
fn beyond(norm: &U1024, j: usize, threshold: &U2048) -> bool {
    let e = 1u32 << (6 - j);
    let (b, t) = (norm.bits_vartime(), threshold.bits_vartime());
    if b == 0 {
        return false;
    }
    if (b - 1) * e >= t {
        return true;
    }
    if b * e < t {
        return false;
    }
    let mut x: U2048 = norm.resize();
    for _ in 0..6 - j {
        x = x.wrapping_mul(&x);
    }
    x > *threshold
}

/// The exact test against `b`, accepting early when a stage norm $`N_j`$ is within
/// $`\eta^{2^j}`$ and rejecting early when $`N_j^{2^{6-j}}`$ exceeds $`\eta^{64}d^{2^{6-j}-1}`$
/// (module documentation).
fn decide(c: &Poly, b: &Bounds) -> Result<bool, Error> {
    let mut within = false;
    stage_norms(c, |j, norm| {
        within = *norm <= b.accept[j];
        within || b.reject[j].as_ref().is_some_and(|t| beyond(norm, j, t))
    })?;
    Ok(within)
}

/// Whether $`c`$ meets the operator-norm bound $`\eta`$ of LNP22 §2.7 with $`\kappa=32`$:
/// $`\|(\sigma_{-1}(c)c)^{32}\|_1\le\eta^{64}`$, in exact integer arithmetic (accepting
/// early where a smaller power already shows it, with the same result). The errors are those
/// of [`eta_norm_power`], and [`Error::Parameter`] for $`\eta\ge2^{16}`$.
pub fn within_eta(c: &Poly, eta: u64) -> Result<bool, Error> {
    let bounds = bounds(eta, c.ring().degree())?;
    decide(c, &bounds)
}

/// A challenge of the set: the first [`autostable`] draw from `stream` that is
/// [`within_eta`]. A rejected draw is followed by the next draw from the same stream, which
/// continues where the last read of the rejected one ended, so prover and verifier, reading
/// the same stream, take the same draws. Fails with [`Error::Randomness`] after
/// [`MAX_CHALLENGE_DRAWS`] rejected draws, and with the errors of [`autostable`] and
/// [`within_eta`]. An $`\omega`$ with $`(d-1)\omega>2^{15}`$, for which a draw could exceed
/// the capacity of [`eta_norm_power`], is refused with [`Error::Parameter`] before anything is
/// read, as is an $`\eta\ge2^{16}`$, so that no capacity refusal depends on the stream.
///
/// The time taken depends on the draws, through their number and the stage at which each is
/// decided. For the challenge of the accepted attempt of a proof the draws are public; for a
/// rejected attempt they are a function of the transcript and that attempt's first message
/// $`w_1`$, which no proof reveals. Like the rest of the crate, this runs in variable time
/// (`docs/security.md`).
pub fn challenge(
    stream: &mut impl ByteStream,
    ring: Arc<Ring>,
    omega: i128,
    eta: u64,
) -> Result<Poly, Error> {
    let bounds = bounds(eta, ring.degree())?;
    // ||c||_1 = |c_0| + 2 (|c_1| + ... + |c_{d/2-1}|) <= (d - 1) omega for every draw.
    let largest = (ring.degree() as u128)
        .saturating_sub(1)
        .checked_mul(omega.max(0) as u128);
    if largest.is_none_or(|l1| l1 > MAX_L1) {
        return Err(Error::Parameter("challenge norm capacity"));
    }
    for _ in 0..MAX_CHALLENGE_DRAWS {
        let c = autostable(stream, ring.clone(), omega)?;
        if decide(&c, &bounds)? {
            return Ok(c);
        }
    }
    Err(Error::Randomness)
}

/// The bytes that the proofs squeeze for one challenge and [`challenge_from_bytes`] decodes: 24
/// blocks of the SHAKE128 rate, 4032 bytes. For the two challenge sets of the parameter check,
/// $`(d,\omega)=(64,8)`$ and $`(128,2)`$, they hold no draw within $`\eta`$ with probability at
/// most $`2^{-210.9}`$ and $`2^{-208.0}`$ on uniform bytes if at most a quarter of the draws fail
/// the $`\eta`$ test, and at most $`2^{-582}`$ if a fiftieth do (exact rational computation in
/// Sage over the binomial law of the kept words). About 1% of draws fail it.
pub const CHALLENGE_BYTES: usize = 24 * 168;

/// The challenge decoded from the fixed-length byte string `bytes`, the decoding of the
/// proofs' challenge from [`CHALLENGE_BYTES`] squeezed bytes.
///
/// The bytes are read as $`k`$-bit words, $`k=\mathrm{bitlen}(2\omega)`$, least significant bit
/// first (word $`i`$ is bits $`ik`$ to $`ik+k-1`$, bit $`j`$ being bit $`j\bmod8`$ of byte
/// $`\lfloor j/8\rfloor`$), and the bits after the last whole word are not read. A word $`w`$
/// below $`2\omega+1`$ is kept as the coefficient $`w-\omega`$, the others are skipped. Each
/// $`d/2`$ kept coefficients in order are one draw, the coefficients $`0`$ to $`d/2-1`$ of a
/// $`\sigma_{-1}`$-stable polynomial ([`autostable`]'s form: coefficient $`d/2`$ zero,
/// coefficient $`d-i`$ minus coefficient $`i`$), and the first draw [`within_eta`] is the
/// challenge. If no draw in `bytes` is within $`\eta`$, the result is the constant 1, which is
/// in the challenge set.
///
/// On uniform bytes the result is uniform on the challenge set except for the weight of that
/// fallback, so the decoding's bias is the probability that no draw in `bytes` is within
/// $`\eta`$. The decoding is infallible on every byte string: the errors are the parameter
/// refusals of [`challenge`] ($`\omega<1`$, $`\omega\ge q/2`$, $`(d-1)\omega>2^{15}`$,
/// $`\eta\ge2^{16}`$), which depend on no byte.
pub fn challenge_from_bytes(
    bytes: &[u8],
    ring: Arc<Ring>,
    omega: i128,
    eta: u64,
) -> Result<Poly, Error> {
    let bounds = bounds(eta, ring.degree())?;
    if omega < 1 || crate::math::U256::from_u128(omega as u128) >= ring.half {
        return Err(Error::Parameter("challenge bound"));
    }
    let largest = (ring.degree() as u128)
        .saturating_sub(1)
        .checked_mul(omega as u128);
    if largest.is_none_or(|l1| l1 > MAX_L1) {
        return Err(Error::Parameter("challenge norm capacity"));
    }
    let k = 128 - ((2 * omega) as u128).leading_zeros();
    let half = ring.degree() / 2;
    let mut first = Vec::with_capacity(half);
    for i in 0..bytes.len() * 8 / k as usize {
        let w = super::bits(bytes, i * k as usize, k) as i128;
        if w > 2 * omega {
            continue;
        }
        first.push(w - omega);
        if first.len() == half {
            let c = Poly::new(ring.clone(), stable(&first, 0i128, |x| -x))?;
            if decide(&c, &bounds)? {
                return Ok(c);
            }
            first.clear();
        }
    }
    Ok(Poly::constant(ring, 1))
}

/// Test only: every whole draw of [`challenge_from_bytes`] in `bytes`, in order, before any
/// $`\eta`$ test.
#[cfg(test)]
pub(crate) fn draws_from_bytes(bytes: &[u8], ring: Arc<Ring>, omega: i128) -> Vec<Poly> {
    let k = 128 - ((2 * omega) as u128).leading_zeros();
    let half = ring.degree() / 2;
    let mut draws = Vec::new();
    let mut first = Vec::with_capacity(half);
    for i in 0..bytes.len() * 8 / k as usize {
        let w = super::bits(bytes, i * k as usize, k) as i128;
        if w <= 2 * omega {
            first.push(w - omega);
        }
        if first.len() == half {
            draws.push(Poly::new(ring.clone(), stable(&first, 0i128, |x| -x)).unwrap());
            first.clear();
        }
    }
    draws
}

/// The norms $`N_j=\|u^{2^{j-1}}\|_1`$ of the powers of $`u=\sigma_{-1}(c)c`$ for $`j=1`$ to 6,
/// in order, each passed to `stop` with its $`j`$ once computed. The computation ends after the
/// first $`j`$ for which `stop` returns true, or after $`j=6`$, and returns that $`N_j`$. Fails
/// as [`eta_norm_power`] does.
fn stage_norms(c: &Poly, mut stop: impl FnMut(usize, &U1024) -> bool) -> Result<U1024, Error> {
    let c = c.coefficients_i128()?;
    let d = c.len();
    let l1 = c
        .iter()
        .try_fold(0u128, |sum, x| sum.checked_add(x.unsigned_abs()))
        .ok_or(Error::Overflow)?;
    if l1 > MAX_L1 {
        return Err(Error::Parameter("challenge norm capacity"));
    }
    // sigma_{-1}(c): s_0 = c_0 and s_{d-i} = -c_i.
    let s: Vec<i128> = (0..d)
        .map(|i| if i == 0 { c[0] } else { -c[d - i] })
        .collect();
    // u, u^2 and u^4 (j = 1 to 3), at most 2^30, 2^60 and 2^120.
    let mut u = half_product(
        &s,
        &c,
        0i128,
        |x, y| x.checked_mul(*y),
        |x, y| x.checked_add(*y),
        |x, y| x.checked_sub(*y),
    )?;
    for j in 1..=3 {
        if j > 1 {
            u = half_square(
                &stable(&u, 0, |x| -x),
                0i128,
                |x, y| x.checked_mul(*y),
                |x, y| x.checked_add(*y),
                |x, y| x.checked_sub(*y),
            )?;
        }
        let norm = U1024::from_u128(narrow_norm(&u)?);
        if stop(j, &norm) {
            return Ok(norm);
        }
    }
    // u^8, u^16 and u^32 (j = 4 to 6), at most 2^240, 2^480 and 2^960.
    let u: Vec<I128> = u.iter().map(|x| I128::from_i128(*x)).collect();
    let u: Vec<I256> = square_wide(&u)?;
    let norm = wide_norm(&u)?;
    if stop(4, &norm) {
        return Ok(norm);
    }
    let u: Vec<I512> = square_wide(&u)?;
    let norm = wide_norm(&u)?;
    if stop(5, &norm) {
        return Ok(norm);
    }
    let u: Vec<I1024> = square_wide(&u)?;
    let norm = wide_norm(&u)?;
    stop(6, &norm);
    Ok(norm)
}

/// The norm $`|v_0|+2(|v_1|+\dots+|v_{d/2-1}|)`$ of the $`\sigma_{-1}`$-stable polynomial with
/// first coefficients `half` (coefficient $`d/2`$ zero, coefficient $`d-i`$ minus
/// coefficient $`i`$).
fn narrow_norm(half: &[i128]) -> Result<u128, Error> {
    half.iter()
        .enumerate()
        .try_fold(0u128, |sum, (i, x)| {
            let m = x.unsigned_abs();
            let m = if i == 0 { Some(m) } else { m.checked_mul(2) };
            sum.checked_add(m?)
        })
        .ok_or(Error::Overflow)
}

/// [`narrow_norm`] for wide coefficients, as a `U1024`.
fn wide_norm<const LIMBS: usize>(half: &[Int<LIMBS>]) -> Result<U1024, Error> {
    let mut norm = Uint::<LIMBS>::ZERO;
    for (i, x) in half.iter().enumerate() {
        let m = x.abs();
        let m = if i == 0 {
            m
        } else {
            Option::from(m.checked_add(&m)).ok_or(Error::Overflow)?
        };
        norm = Option::from(norm.checked_add(&m)).ok_or(Error::Overflow)?;
    }
    if LIMBS > U1024::LIMBS {
        return Err(Error::Overflow);
    }
    Ok(norm.resize())
}

/// Coefficients $`0`$ to $`d/2-1`$ of the negacyclic product of `a` and `b` (each of length
/// $`d`$), which the caller knows to be $`\sigma_{-1}`$-stable:
/// $`\sum_{j\le i}a_jb_{i-j}-\sum_{j>i}a_jb_{d+i-j}`$; `None` from an operation is an
/// [`Error::Overflow`].
fn half_product<T, W: Copy>(
    a: &[T],
    b: &[T],
    zero: W,
    mul: impl Fn(&T, &T) -> Option<W>,
    add: impl Fn(&W, &W) -> Option<W>,
    sub: impl Fn(&W, &W) -> Option<W>,
) -> Result<Vec<W>, Error> {
    let d = a.len();
    if b.len() != d || !d.is_multiple_of(2) {
        return Err(Error::Dimension);
    }
    (0..d / 2)
        .map(|i| {
            let mut sum = zero;
            for j in 0..=i {
                sum = add(&sum, &mul(&a[j], &b[i - j])?)?;
            }
            for j in i + 1..d {
                sum = sub(&sum, &mul(&a[j], &b[d + i - j])?)?;
            }
            Some(sum)
        })
        .collect::<Option<Vec<W>>>()
        .ok_or(Error::Overflow)
}

/// Coefficients $`0`$ to $`d/2-1`$ of the negacyclic square of `a` (of length $`d`$), as
/// [`half_product`] computes them for $`b=a`$, with each product $`a_ja_k=a_ka_j`$ taken once:
/// $`2\big(\sum_{j<i-j}a_ja_{i-j}-\sum_{i<j<d+i-j}a_ja_{d+i-j}\big)`$, plus
/// $`a_{i/2}^2-a_{(d+i)/2}^2`$ for even $`i`$. Every partial sum is at most the sum of the
/// absolute values of all $`d`$ products of the coefficient, as in [`half_product`].
fn half_square<T, W: Copy>(
    a: &[T],
    zero: W,
    mul: impl Fn(&T, &T) -> Option<W>,
    add: impl Fn(&W, &W) -> Option<W>,
    sub: impl Fn(&W, &W) -> Option<W>,
) -> Result<Vec<W>, Error> {
    let d = a.len();
    if !d.is_multiple_of(2) {
        return Err(Error::Dimension);
    }
    (0..d / 2)
        .map(|i| {
            let mut sum = zero;
            // j < i - j, and i < j < d + i - j.
            for j in 0..i.div_ceil(2) {
                sum = add(&sum, &mul(&a[j], &a[i - j])?)?;
            }
            for j in i + 1..(d + i).div_ceil(2) {
                sum = sub(&sum, &mul(&a[j], &a[d + i - j])?)?;
            }
            sum = add(&sum, &sum)?;
            if i.is_multiple_of(2) {
                sum = add(&sum, &mul(&a[i / 2], &a[i / 2])?)?;
                sum = sub(&sum, &mul(&a[(d + i) / 2], &a[(d + i) / 2])?)?;
            }
            Some(sum)
        })
        .collect::<Option<Vec<W>>>()
        .ok_or(Error::Overflow)
}

/// The $`d`$ coefficients of a $`\sigma_{-1}`$-stable polynomial from its coefficients $`0`$
/// to $`d/2-1`$: coefficient $`d/2`$ is zero and coefficient $`d-i`$ is minus coefficient $`i`$.
fn stable<T: Copy>(half: &[T], zero: T, neg: impl Fn(&T) -> T) -> Vec<T> {
    let mut out = Vec::with_capacity(2 * half.len());
    out.extend_from_slice(half);
    out.push(zero);
    out.extend(half.iter().skip(1).rev().map(neg));
    out
}

/// The square of a $`\sigma_{-1}`$-stable polynomial given by its first half, into the type of
/// twice the width: the products are exact and the sums checked.
fn square_wide<const LIMBS: usize, const WIDE: usize>(
    half: &[Int<LIMBS>],
) -> Result<Vec<Int<WIDE>>, Error>
where
    Uint<LIMBS>: Concat<LIMBS, Output = Uint<WIDE>>,
{
    half_square(
        &stable(half, Int::ZERO, |x| x.wrapping_neg()),
        Int::ZERO,
        |x, y| Some(x.concatenating_mul(y)),
        |x, y| Option::from(x.checked_add(y)),
        |x, y| Option::from(CheckedSub::checked_sub(x, y)),
    )
}

#[cfg(test)]
mod tests;
