//! The fixed-width fields of codec C1, from which every message of the proofs is serialized and
//! absorbed (`docs/encodings.md`). Coefficients are written in order, polynomial after
//! polynomial, each least significant bit first, as FIPS 204 §7.1 packs them:
//!
//! * a coefficient modulo $`m`$ as its canonical representative in $`\mathrm{bitlen}(m-1)`$
//!   bits, `SimpleBitPack` of FIPS 204 (commitments, garbage, $`h`$, $`t`$, $`v`$, LinHC
//!   commitments and mask commitments modulo their own modulus, high parts $`w_1`$ modulo
//!   $`m=(q-1)/\gamma`$), read back only if below $`m`$,
//! * a signed coefficient of absolute value at most $`b=2^{w-1}-1`$ as $`b-x`$ in $`w`$ bits,
//!   `BitPack(x, b, b)` of FIPS 204 (Gaussian responses and hints), with the one code above
//!   $`2b`$ refused,
//! * a challenge as its coefficients $`0`$ to $`d/2-1`$, each $`c_i`$ as $`\omega-c_i`$ in
//!   $`\mathrm{bitlen}(2\omega)`$ bits, `BitPack(c, ω, ω)`, with the codes above $`2\omega`$
//!   refused: the other coefficients of a challenge follow from these.
//!
//! With degrees 64 and 128 every field fills whole bytes.
use super::{BitReader, BitWriter};
use crate::{
    Error,
    math::{Poly, PolyVec, Ring, U256, int},
};
use std::sync::Arc;

/// Write the coefficients of `v`, values in $`[0,m)`$, each in $`\mathrm{bitlen}(m-1)`$ bits.
/// With `modulus` the ring's $`q`$ these are the canonical representatives. Fails with
/// [`Error::Encoding`] on a coefficient not below `modulus`.
pub(crate) fn write_below(w: &mut BitWriter, v: &PolyVec, modulus: &U256) -> Result<(), Error> {
    match int::to_u128(modulus) {
        Some(m) => {
            for p in v.entries() {
                for x in p.coefficients() {
                    if x >= modulus {
                        return Err(Error::Encoding);
                    }
                    w.uniform(int::low_u128(x), m)?;
                }
            }
        }
        None => {
            for p in v.entries() {
                for x in p.coefficients() {
                    w.uniform_u256(x, modulus)?;
                }
            }
        }
    }
    Ok(())
}

/// Read `len` polynomials of `ring` written by [`write_below`] with `modulus` at most
/// $`q`$, refusing a value not below `modulus`.
pub(crate) fn read_below(
    r: &mut BitReader<'_>,
    ring: &Arc<Ring>,
    len: usize,
    modulus: &U256,
) -> Result<PolyVec, Error> {
    let narrow = int::to_u128(modulus);
    let mut entries = Vec::with_capacity(len);
    for _ in 0..len {
        let coefficients = (0..ring.degree())
            .map(|_| match narrow {
                Some(m) => Ok(U256::from_u128(r.uniform(m)?)),
                None => r.uniform_u256(modulus),
            })
            .collect::<Result<Vec<_>, Error>>()?;
        entries.push(Poly::from_canonical(ring.clone(), coefficients));
    }
    PolyVec::new(ring.clone(), entries)
}

/// Write the centred coefficients of `v` in the signed code of `width` bits. Fails with
/// [`Error::Encoding`] on a coefficient beyond $`2^{w-1}-1`$ in absolute value.
pub(crate) fn write_signed(w: &mut BitWriter, v: &PolyVec, width: u32) -> Result<(), Error> {
    for p in v.entries() {
        for x in p.coefficients() {
            let x = p.ring().centred_i128(x).ok_or(Error::Encoding)?;
            w.signed(x, width)?;
        }
    }
    Ok(())
}

/// Whether every centred coefficient of `v` fits the signed code of `width` bits: the prover's
/// test before it writes a response or a hint.
pub(crate) fn fits_signed(v: &PolyVec, width: u32) -> bool {
    let b = (1u128 << (width - 1)) - 1;
    v.entries().iter().all(|p| {
        p.coefficients().iter().all(|x| {
            p.ring()
                .centred_i128(x)
                .is_some_and(|x| x.unsigned_abs() <= b)
        })
    })
}

/// Read `len` polynomials of `ring` written by [`write_signed`], refusing values of absolute
/// value $`q/2`$ or more (which no centred coefficient has).
pub(crate) fn read_signed(
    r: &mut BitReader<'_>,
    ring: &Arc<Ring>,
    len: usize,
    width: u32,
) -> Result<PolyVec, Error> {
    let mut entries = Vec::with_capacity(len);
    for _ in 0..len {
        let coefficients = (0..ring.degree())
            .map(|_| {
                let x = r.signed(width)?;
                if U256::from_u128(x.unsigned_abs()) > ring.half {
                    return Err(Error::Encoding);
                }
                Ok(x)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        entries.push(Poly::new(ring.clone(), coefficients)?);
    }
    PolyVec::new(ring.clone(), entries)
}

/// The width of a challenge coefficient's code, $`\mathrm{bitlen}(2\omega)`$.
fn challenge_width(omega: i128) -> u32 {
    128 - ((2 * omega) as u128).leading_zeros()
}

/// Write a challenge as its coefficients $`0`$ to $`d/2-1`$. Fails with [`Error::Encoding`]
/// unless `c` has the challenges' form: $`\sigma_{-1}`$-stable, coefficient $`d/2`$ zero, and
/// coefficients in $`[-\omega,\omega]`$.
pub(crate) fn write_challenge(w: &mut BitWriter, c: &Poly, omega: i128) -> Result<(), Error> {
    let d = c.ring().degree();
    if c.auto() != *c || !c.coefficients()[d / 2].is_zero_vartime() {
        return Err(Error::Encoding);
    }
    let k = challenge_width(omega);
    for x in &c.coefficients()[..d / 2] {
        let x = c.ring().centred_i128(x).ok_or(Error::Encoding)?;
        if x.unsigned_abs() > omega as u128 {
            return Err(Error::Encoding);
        }
        w.unsigned((omega - x) as u128, k)?;
    }
    Ok(())
}

/// Read a challenge written by [`write_challenge`]: the $`\sigma_{-1}`$-stable polynomial of
/// `ring` with the read coefficients $`0`$ to $`d/2-1`$.
pub(crate) fn read_challenge(
    r: &mut BitReader<'_>,
    ring: &Arc<Ring>,
    omega: i128,
) -> Result<Poly, Error> {
    let d = ring.degree();
    let k = challenge_width(omega);
    let mut coefficients = vec![0i128; d];
    for i in 0..d / 2 {
        let u = r.unsigned(k)? as i128;
        if u > 2 * omega {
            return Err(Error::Encoding);
        }
        coefficients[i] = omega - u;
        if i > 0 {
            coefficients[d - i] = u - omega;
        }
    }
    Poly::new(ring.clone(), coefficients)
}
