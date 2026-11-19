use super::{
    Poly, Ring,
    ntt::{add_mod, sub_mod},
    poly::Coeffs,
    ring::RingCrt,
};
#[cfg(test)]
use super::{
    ntt::mul_mod,
    ring::{Basis, Crt},
};
use crate::{Error, math::int, params::moduli::NADDS};
#[cfg(test)]
use crypto_bigint::{U64, U256, Uint};
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
                let mut x = centred_residues(ring, a.coeffs(), i);
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
    /// Test only: the empty sum, of no products.
    #[cfg(test)]
    pub(crate) fn zero(ring: &Arc<Ring>) -> Self {
        Self {
            ring: ring.clone(),
            residues: Residues(ring.plans.iter().map(|_| vec![0; ring.d]).collect()),
            count: 0,
        }
    }
    /// Test only, the oracle of [`LazySum`]: `self += a * b` without an intermediate product,
    /// the value of `add_assign` with `a.product(b)`, refusing a product past [`NADDS`] without
    /// modifying `self`.
    #[cfg(test)]
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
    /// Test only: the number of products in the sum.
    #[cfg(test)]
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
/// Products added before a reduction: a sum reduced below $`p<2^{62}`$ plus 15 products of
/// residues below $`p`$ stays below $`p+15(p-1)^2<2^{128}`$.
const LAZY: usize = 15;

/// A sum of at most [`NADDS`] products of transforms, added as 128-bit integers and reduced once
/// per [`LAZY`] products instead of once each.
///
/// Each residue of a transform is the Montgomery form $`\tilde x=xR\bmod p`$, below $`p`$, and
/// [`RnsProduct::mac`] adds $`\mathrm{REDC}(\tilde a\tilde b)=\tilde a\tilde bR^{-1}\bmod p`$
/// for each product. Here the sums $`\sum_k\tilde a_k\tilde b_k`$ are kept exactly, reduced
/// modulo $`p`$ (Barrett) whenever [`LAZY`] products are pending, and $`R^{-1}`$ is applied once
/// at the end (`mont_mul` by 1), so the residues are those of [`RnsProduct::mac`] over the same
/// products: both are $`R^{-1}\sum_k\tilde a_k\tilde b_k\bmod p`$.
pub(crate) struct LazySum {
    ring: Arc<Ring>,
    /// Prime after prime, `d` sums each.
    sums: Vec<u128>,
    pending: usize,
    count: usize,
}
impl LazySum {
    /// The empty sum, of no products.
    pub(crate) fn new(ring: &Arc<Ring>) -> Self {
        Self {
            ring: ring.clone(),
            sums: vec![0; ring.plans.len() * ring.d],
            pending: 0,
            count: 0,
        }
    }
    /// The number of products in the sum.
    pub(crate) fn count(&self) -> usize {
        self.count
    }
    /// `self += a * b`, refusing a product past [`NADDS`] without modifying `self`.
    pub(crate) fn mac(&mut self, a: &PolyNtt, b: &PolyNtt) -> Result<(), Error> {
        if self.ring != a.ring || self.ring != b.ring {
            return Err(Error::RingMismatch);
        }
        if self.count >= NADDS {
            return Err(Error::Overflow);
        }
        if self.pending == LAZY {
            self.reduce_pending();
        }
        for ((sums, x), y) in self
            .sums
            .chunks_exact_mut(self.ring.d)
            .zip(&a.residues.0)
            .zip(&b.residues.0)
        {
            for ((s, x), y) in sums.iter_mut().zip(x).zip(y) {
                *s += u128::from(*x) * u128::from(*y);
            }
        }
        self.pending += 1;
        self.count += 1;
        Ok(())
    }
    fn reduce_pending(&mut self) {
        for (sums, plan) in self
            .sums
            .chunks_exact_mut(self.ring.d)
            .zip(&self.ring.plans)
        {
            for s in sums {
                *s = u128::from(plan.reduce_wide(*s));
            }
        }
        self.pending = 0;
    }
    /// The residues of the sum, in Montgomery form below each prime.
    fn residues(&self) -> Residues {
        Residues(
            self.sums
                .chunks_exact(self.ring.d)
                .zip(&self.ring.plans)
                .map(|(sums, plan)| {
                    sums.iter()
                        .map(|s| plan.mont_mul(plan.reduce_wide(*s), 1))
                        .collect()
                })
                .collect(),
        )
    }
    /// Reconstruct the exact centred integer sum, then reduce modulo $`q`$: what
    /// [`RnsProduct::reduce`] returns for the same products.
    pub(crate) fn reduce(&self) -> Result<Poly, Error> {
        reconstruct_owned(&self.ring, self.residues())
    }
}
impl Drop for LazySum {
    fn drop(&mut self) {
        self.sums.as_mut_slice().zeroize();
    }
}

/// The residues modulo prime $`i`$ of the centred representatives of canonical coefficients.
fn centred_residues(ring: &Ring, coeffs: &Coeffs, i: usize) -> Vec<u64> {
    let p = ring.plans[i].modulus();
    let q = ring.q_mod_p[i];
    // Below 2^128, the sign test and the remainder in words: as q > p from 2^63 on, no value
    // is its own residue there.
    let reducer = ring.reducers[i];
    let narrow = move |v: u128, half: u128| {
        let r = reducer.reduce(v);
        if v > half { sub_mod(r, q, p) } else { r }
    };
    match coeffs {
        // q < 2^63: a canonical value is a word, and its residue is native.
        Coeffs::Word(values) => {
            let half = ring.small_crt().expect("one-word coefficients").half_q;
            values
                .iter()
                .map(|&v| {
                    let mut r = v;
                    while r >= p {
                        r -= p;
                    }
                    if v > half { sub_mod(r, q, p) } else { r }
                })
                .collect()
        }
        Coeffs::Double(values) => {
            let half = ring.narrow.expect("two-word coefficients") >> 1;
            values.iter().map(|&v| narrow(v, half)).collect()
        }
        Coeffs::Wide(values) => match ring.narrow {
            Some(modulus) => values
                .iter()
                .map(|x| narrow(int::low_u128(x), modulus >> 1))
                .collect(),
            None => values
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
        },
    }
}
fn reconstruct(ring: &Arc<Ring>, residues: &[Vec<u64>]) -> Result<Poly, Error> {
    reconstruct_owned(ring, Residues(residues.to_vec()))
}
/// [`reconstruct`] of residues that the caller gives up, transformed back in place.
fn reconstruct_owned(ring: &Arc<Ring>, mut coeffs: Residues) -> Result<Poly, Error> {
    for (values, plan) in coeffs.0.iter_mut().zip(&ring.plans) {
        plan.inverse_mont(values);
    }
    // Each CRT writes the words that the ring's polynomials keep.
    let out = match &ring.crt {
        RingCrt::Small(small) => {
            let mut out = vec![0; ring.d];
            small.reconstruct_into(&coeffs.0, &mut out);
            Coeffs::Word(out)
        }
        RingCrt::Wide(wide) if ring.wide.is_some() => {
            Coeffs::Double(wide.reconstruct_all_u128(&coeffs.0, &ring.nonzero, ring.small.as_ref()))
        }
        RingCrt::Wide(wide) => {
            Coeffs::Wide(wide.reconstruct_all(&coeffs.0, &ring.nonzero, ring.small.as_ref()))
        }
    };
    Ok(Poly::from_coeffs(ring.clone(), out))
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
#[cfg(test)]
mod lazy_tests;
