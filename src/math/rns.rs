use super::{
    Poly, Ring,
    ntt::{add_mod, sub_mod},
    ring::RingCrt,
};
#[cfg(test)]
use super::{
    ntt::mul_mod,
    ring::{Basis, Crt},
};
use crate::{Error, math::int, params::moduli::NADDS};
use crypto_bigint::U256;
#[cfg(test)]
use crypto_bigint::{U64, Uint};
use std::sync::Arc;
use zeroize::Zeroize;

/// Residues per prime, wiped on drop as `Poly` wipes its coefficients: each word once with a
/// volatile write, then any spare capacity. `Vec::zeroize` would wipe every word a second
/// time, bytewise.
#[derive(Clone)]
struct Residues(Vec<Vec<u64>>);
impl Drop for Residues {
    fn drop(&mut self) {
        for values in &mut self.0 {
            values.as_mut_slice().zeroize();
            values.spare_capacity_mut().zeroize();
        }
    }
}

/// NTT representation of a reduced polynomial. Products have a separate type.
///
/// The residues are kept in Montgomery form, times $`2^{64}`$ modulo each prime (see
/// [`NttPlan`](super::ntt::NttPlan)); no method exposes them, and [`PolyNtt::to_poly`] and
/// [`RnsProduct::reduce`] return exactly the coefficients of the exact transforms.
#[derive(Clone)]
pub struct PolyNtt {
    ring: Arc<Ring>,
    residues: Residues,
}
impl core::fmt::Debug for PolyNtt {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PolyNtt").finish_non_exhaustive()
    }
}
impl PolyNtt {
    /// Residues of the centred representatives, so that accumulated products stay below
    /// $`P/2`$ in absolute value.
    pub(crate) fn from_poly(a: &Poly) -> Self {
        let ring = &a.ring;
        let residues = ring
            .plans
            .iter()
            .enumerate()
            .map(|(i, plan)| {
                let mut x = centred_residues(ring, &a.coeffs, i);
                plan.forward_mont(&mut x);
                x
            })
            .collect();
        Self {
            ring: a.ring.clone(),
            residues: Residues(residues),
        }
    }
    /// Recover reduced coefficients using exact integer CRT.
    pub fn to_poly(&self) -> Result<Poly, Error> {
        reconstruct(&self.ring, &self.residues.0)
    }
    /// Pointwise product, ready for at most 128-product accumulation.
    pub fn product(&self, rhs: &Self) -> Result<RnsProduct, Error> {
        if self.ring != rhs.ring {
            return Err(Error::RingMismatch);
        }
        let residues = self
            .residues
            .0
            .iter()
            .zip(&rhs.residues.0)
            .zip(&self.ring.plans)
            .map(|((a, b), plan)| {
                a.iter()
                    .zip(b)
                    .map(|(a, b)| plan.mont_mul(*a, *b))
                    .collect()
            })
            .collect();
        Ok(RnsProduct {
            ring: self.ring.clone(),
            residues: Residues(residues),
            count: 1,
        })
    }
}

/// A sum of at most [`NADDS`] products. The bound is checked in release builds too.
pub struct RnsProduct {
    ring: Arc<Ring>,
    residues: Residues,
    count: usize,
}
impl core::fmt::Debug for RnsProduct {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RnsProduct")
            .field("count", &self.count)
            .finish_non_exhaustive()
    }
}
impl RnsProduct {
    /// The empty sum, of no products.
    pub(crate) fn zero(ring: &Arc<Ring>) -> Self {
        Self {
            ring: ring.clone(),
            residues: Residues(ring.plans.iter().map(|_| vec![0; ring.d]).collect()),
            count: 0,
        }
    }
    /// `self += a * b` without an intermediate product: the value of `add_assign` with
    /// `a.product(b)`, refusing a product past [`NADDS`] without modifying `self`.
    pub(crate) fn mac(&mut self, a: &PolyNtt, b: &PolyNtt) -> Result<(), Error> {
        if self.ring != a.ring || self.ring != b.ring {
            return Err(Error::RingMismatch);
        }
        if self.count >= NADDS {
            return Err(Error::Overflow);
        }
        for (((acc, x), y), plan) in self
            .residues
            .0
            .iter_mut()
            .zip(&a.residues.0)
            .zip(&b.residues.0)
            .zip(&self.ring.plans)
        {
            let p = plan.modulus();
            for ((acc, x), y) in acc.iter_mut().zip(x).zip(y) {
                *acc = add_mod(*acc, plan.mont_mul(*x, *y), p);
            }
        }
        self.count += 1;
        Ok(())
    }
    /// The number of products in the sum.
    pub(crate) fn count(&self) -> usize {
        self.count
    }
    /// Add a product accumulator, rejecting excess capacity without modifying `self`.
    pub fn add_assign(&mut self, rhs: &Self) -> Result<(), Error> {
        if self.ring != rhs.ring {
            return Err(Error::RingMismatch);
        }
        if self.count + rhs.count > NADDS {
            return Err(Error::Overflow);
        }
        for ((a, b), plan) in self
            .residues
            .0
            .iter_mut()
            .zip(&rhs.residues.0)
            .zip(&self.ring.plans)
        {
            let p = plan.modulus();
            for (a, b) in a.iter_mut().zip(b) {
                *a = add_mod(*a, *b, p);
            }
        }
        self.count += rhs.count;
        Ok(())
    }
    /// Reconstruct the exact centred integer product, then reduce modulo $`q`$.
    pub fn reduce(&self) -> Result<Poly, Error> {
        reconstruct(&self.ring, &self.residues.0)
    }
}
/// The residues modulo prime $`i`$ of the centred representatives of canonical coefficients.
fn centred_residues(ring: &Ring, coeffs: &[U256], i: usize) -> Vec<u64> {
    let p = ring.plans[i].modulus();
    let q = ring.q_mod_p[i];
    match ring.small_crt() {
        // A canonical value is below q < 2^63, so its residue is native.
        Some(small) => coeffs
            .iter()
            .map(|x| {
                let v = int::low_u64(x);
                let mut r = v;
                while r >= p {
                    r -= p;
                }
                if v > small.half_q {
                    sub_mod(r, q, p)
                } else {
                    r
                }
            })
            .collect(),
        None => coeffs
            .iter()
            .map(|x| {
                let r = ring.residue(x, i);
                if !ring.is_negative(x) {
                    r
                } else if r >= q {
                    r - q
                } else {
                    r + (p - q)
                }
            })
            .collect(),
    }
}
fn reconstruct(ring: &Arc<Ring>, residues: &[Vec<u64>]) -> Result<Poly, Error> {
    let mut coeffs = Residues(residues.to_vec());
    for (values, plan) in coeffs.0.iter_mut().zip(&ring.plans) {
        plan.inverse_mont(values);
    }
    let out = match &ring.crt {
        RingCrt::Small(small) => (0..ring.d)
            .map(|j| U256::from_u64(small.reconstruct(&coeffs.0, j)))
            .collect(),
        RingCrt::Wide(wide) => (0..ring.d)
            .map(|j| wide.reconstruct(&coeffs.0, j, &ring.nonzero, ring.small.as_ref()))
            .collect(),
    };
    Ok(Poly::from_canonical(ring.clone(), out))
}
/// Mixed-radix CRT of each coefficient in a `C`-limb accumulator, then the canonical
/// representative modulo $`q`$ of the centred integer: the reference reconstruction, kept as
/// the oracle of `crt_tests`.
#[cfg(test)]
fn reconstruct_with<const C: usize>(
    ring: &Ring,
    basis: &Basis<C>,
    coeffs: &[Vec<u64>],
) -> Vec<U256> {
    (0..ring.d)
        .map(|j| {
            let mut x = Uint::<C>::ZERO;
            for (i, plan) in ring.plans.iter().enumerate() {
                let p = plan.modulus();
                let residue = ring.divisors[i].rem(&x);
                let digit = mul_mod((coeffs[i][j] + p - residue) % p, ring.inverses[i], p);
                x = x.wrapping_add(&basis.prefixes[i].wrapping_mul(&U64::from_u64(digit)));
            }
            let negative = x > basis.half;
            let magnitude = if negative {
                basis.product.wrapping_sub(&x)
            } else {
                x
            };
            let r = match &ring.small {
                Some(q) => U256::from_u64(q.rem(&magnitude)),
                None => magnitude.rem_vartime(&ring.nonzero),
            };
            if negative && r != U256::ZERO {
                ring.q.wrapping_sub(&r)
            } else {
                r
            }
        })
        .collect()
}

#[cfg(test)]
mod crt_tests;
