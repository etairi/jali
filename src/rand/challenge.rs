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
use super::{ByteStream, autostable};
use crate::{
    Error,
    math::{Poly, Ring},
};
use crypto_bigint::{CheckedAdd, CheckedSub, Concat, I128, I256, I512, I1024, Int, U1024, Uint};
use std::sync::Arc;

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

/// The exact test against the thresholds of [`eta_powers`], accepting early when a stage norm
/// $`N_j`$ is within $`\eta^{2^j}`$ (module documentation).
fn decide(c: &Poly, powers: &[U1024; 7]) -> Result<bool, Error> {
    let mut within = false;
    stage_norms(c, |j, norm| {
        within = *norm <= powers[j];
        within
    })?;
    Ok(within)
}

/// Whether $`c`$ meets the operator-norm bound $`\eta`$ of LNP22 §2.7 with $`\kappa=32`$:
/// $`\|(\sigma_{-1}(c)c)^{32}\|_1\le\eta^{64}`$, in exact integer arithmetic (accepting
/// early where a smaller power already shows it, with the same result). The errors are those
/// of [`eta_norm_power`], and [`Error::Parameter`] for $`\eta\ge2^{16}`$.
pub fn within_eta(c: &Poly, eta: u64) -> Result<bool, Error> {
    decide(c, &eta_powers(eta)?)
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
    let powers = eta_powers(eta)?;
    // ||c||_1 = |c_0| + 2 (|c_1| + ... + |c_{d/2-1}|) <= (d - 1) omega for every draw.
    let largest = (ring.degree() as u128)
        .saturating_sub(1)
        .checked_mul(omega.max(0) as u128);
    if largest.is_none_or(|l1| l1 > MAX_L1) {
        return Err(Error::Parameter("challenge norm capacity"));
    }
    for _ in 0..MAX_CHALLENGE_DRAWS {
        let c = autostable(stream, ring.clone(), omega)?;
        if decide(&c, &powers)? {
            return Ok(c);
        }
    }
    Err(Error::Randomness)
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
