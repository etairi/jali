use super::{PolyNtt, Ring, int};
use crate::Error;
use crypto_bigint::U256;
use std::sync::Arc;
use zeroize::{Zeroize, Zeroizing};

/// Coefficients stored as canonical representatives in $`[0,q)`$; buffers are wiped on drop.
///
/// Norms and the narrow accessors use the centred representatives in $`(-q/2,q/2]`$.
#[derive(Clone, PartialEq, Eq)]
pub struct Poly {
    pub(crate) ring: Arc<Ring>,
    pub(crate) coeffs: Vec<U256>,
}
impl core::fmt::Debug for Poly {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Poly")
            .field("degree", &self.ring.d)
            .finish_non_exhaustive()
    }
}
impl Drop for Poly {
    fn drop(&mut self) {
        // Every coefficient with one volatile write, then any capacity past the length bytewise.
        // `Vec::zeroize` also wipes the coefficients a second time bytewise, 32 stores per
        // coefficient: a third of the samples in a profile of the possession example.
        self.coeffs.as_mut_slice().zeroize();
        self.coeffs.spare_capacity_mut().zeroize();
    }
}

impl Poly {
    /// Reduce an exact-length coefficient vector into the ring.
    pub fn new(ring: Arc<Ring>, mut coeffs: Vec<i128>) -> Result<Self, Error> {
        if coeffs.len() != ring.d {
            coeffs.zeroize();
            return Err(Error::Dimension);
        }
        let reduced = coeffs.iter().map(|x| ring.reduce_i128(*x)).collect();
        coeffs.zeroize();
        Ok(Self {
            ring,
            coeffs: reduced,
        })
    }
    /// Reduce an exact-length vector of 256-bit values into the ring.
    pub fn from_u256(ring: Arc<Ring>, mut coeffs: Vec<U256>) -> Result<Self, Error> {
        if coeffs.len() != ring.d {
            coeffs.zeroize();
            return Err(Error::Dimension);
        }
        for x in &mut coeffs {
            *x = ring.reduce(x);
        }
        Ok(Self { ring, coeffs })
    }
    /// Values already in $`[0,q)`$, as the arithmetic produces them.
    pub(crate) fn from_canonical(ring: Arc<Ring>, coeffs: Vec<U256>) -> Self {
        debug_assert!(coeffs.len() == ring.d && coeffs.iter().all(|x| x < &ring.q));
        Self { ring, coeffs }
    }
    /// Zero polynomial.
    pub fn zero(ring: Arc<Ring>) -> Self {
        let coeffs = vec![U256::ZERO; ring.d];
        Self { ring, coeffs }
    }
    /// Constant polynomial, reduced modulo the ring modulus.
    pub fn constant(ring: Arc<Ring>, value: i128) -> Self {
        let value = ring.reduce_i128(value);
        Self::constant_u256(ring, &value)
    }
    /// Constant polynomial from a 256-bit value, reduced modulo the ring modulus.
    pub fn constant_u256(ring: Arc<Ring>, value: &U256) -> Self {
        let mut out = Self::zero(ring);
        out.coeffs[0] = out.ring.reduce(value);
        out
    }
    /// Shared ring parameters.
    pub fn ring(&self) -> &Arc<Ring> {
        &self.ring
    }
    /// Canonical coefficients in $`[0,q)`$.
    pub fn coefficients(&self) -> &[U256] {
        &self.coeffs
    }
    /// Centred coefficients in $`(-q/2,q/2]`$ as `i128`, for short vectors. Fails with
    /// [`Error::Overflow`] if one does not fit, which can happen only for $`q\ge2^{128}`$.
    pub fn coefficients_i128(&self) -> Result<Zeroizing<Vec<i128>>, Error> {
        let mut out = Zeroizing::new(Vec::with_capacity(self.ring.d));
        for x in &self.coeffs {
            out.push(self.ring.centred_i128(x).ok_or(Error::Overflow)?);
        }
        Ok(out)
    }
    /// One centred coefficient as `i128`, with the errors of [`Poly::coefficients_i128`] and
    /// [`Error::Index`] for an index outside the degree.
    pub fn coefficient_i128(&self, i: usize) -> Result<i128, Error> {
        self.ring
            .centred_i128(self.coeffs.get(i).ok_or(Error::Index)?)
            .ok_or(Error::Overflow)
    }
    /// Update one coefficient and reduce it.
    pub fn set_coefficient(&mut self, i: usize, value: i128) -> Result<(), Error> {
        *self.coeffs.get_mut(i).ok_or(Error::Index)? = self.ring.reduce_i128(value);
        Ok(())
    }
    pub(crate) fn compatible(&self, rhs: &Self) -> Result<(), Error> {
        if self.ring != rhs.ring {
            Err(Error::RingMismatch)
        } else {
            Ok(())
        }
    }
    fn map(&self, f: impl Fn(&U256) -> U256) -> Self {
        Self::from_canonical(self.ring.clone(), self.coeffs.iter().map(f).collect())
    }
    /// Coefficient-wise addition.
    pub fn add(&self, rhs: &Self) -> Result<Self, Error> {
        self.compatible(rhs)?;
        let ring = &self.ring;
        Ok(Self::from_canonical(
            ring.clone(),
            self.coeffs
                .iter()
                .zip(&rhs.coeffs)
                .map(|(a, b)| ring.add(a, b))
                .collect(),
        ))
    }
    /// Coefficient-wise subtraction.
    pub fn sub(&self, rhs: &Self) -> Result<Self, Error> {
        self.compatible(rhs)?;
        let ring = &self.ring;
        Ok(Self::from_canonical(
            ring.clone(),
            self.coeffs
                .iter()
                .zip(&rhs.coeffs)
                .map(|(a, b)| ring.sub(a, b))
                .collect(),
        ))
    }
    /// `self += rhs`: the value of [`Poly::add`], in place.
    pub(crate) fn add_assign(&mut self, rhs: &Self) -> Result<(), Error> {
        self.compatible(rhs)?;
        let ring = &self.ring;
        for (a, b) in self.coeffs.iter_mut().zip(&rhs.coeffs) {
            *a = ring.add(a, b);
        }
        Ok(())
    }
    /// `self -= rhs`: the value of [`Poly::sub`], in place.
    pub(crate) fn sub_assign(&mut self, rhs: &Self) -> Result<(), Error> {
        self.compatible(rhs)?;
        let ring = &self.ring;
        for (a, b) in self.coeffs.iter_mut().zip(&rhs.coeffs) {
            *a = ring.sub(a, b);
        }
        Ok(())
    }
    /// `self += rhs * scalar` for a canonical scalar: the value of
    /// `self.add(&rhs.scale_u256(scalar)?)`, in place.
    pub(crate) fn add_scaled_assign(&mut self, rhs: &Self, scalar: &U256) -> Result<(), Error> {
        self.compatible(rhs)?;
        let ring = &self.ring;
        debug_assert!(scalar < &ring.q);
        if scalar.is_zero_vartime() {
            return Ok(());
        }
        if scalar.cmp_vartime(&U256::ONE).is_eq() {
            return self.add_assign(rhs);
        }
        if let Some(multiplier) = ring.scalar_multiplier(scalar) {
            for (a, b) in self.coeffs.iter_mut().zip(&rhs.coeffs) {
                *a = ring.add(a, &multiplier.apply(b));
            }
        } else {
            for (a, b) in self.coeffs.iter_mut().zip(&rhs.coeffs) {
                *a = ring.add(a, &ring.mul(b, scalar));
            }
        }
        Ok(())
    }
    /// `self += c X^k rhs` for a canonical scalar $`c`$ and $`0\le k<d`$: the value of
    /// `self.add(&rhs.rotate(k).scale_u256(c)?)`, in place.
    pub(crate) fn add_monomial_product_assign(
        &mut self,
        rhs: &Self,
        k: usize,
        c: &U256,
    ) -> Result<(), Error> {
        self.compatible(rhs)?;
        let ring = &self.ring;
        let d = ring.d;
        debug_assert!(k < d && c < &ring.q);
        if c.is_zero_vartime() {
            return Ok(());
        }
        // X^k x_i is x_i at i + k, negated past X^d = -1.
        let multiplier = ring.scalar_multiplier(c);
        let product = |x: &U256| match &multiplier {
            Some(m) => m.apply(x),
            None => ring.mul(x, c),
        };
        let (low, high) = rhs.coeffs.split_at(d - k);
        for (a, b) in self.coeffs[k..].iter_mut().zip(low) {
            *a = ring.add(a, &product(b));
        }
        for (a, b) in self.coeffs[..k].iter_mut().zip(high) {
            *a = ring.sub(a, &product(b));
        }
        Ok(())
    }
    /// `self *= scalar` for a canonical scalar: the value of [`Poly::scale_u256`], in place.
    pub(crate) fn scale_assign(&mut self, scalar: &U256) {
        let ring = &self.ring;
        debug_assert!(scalar < &ring.q);
        match ring.scalar_multiplier(scalar) {
            Some(m) => self.coeffs.iter_mut().for_each(|a| *a = m.apply(a)),
            None => self
                .coeffs
                .iter_mut()
                .for_each(|a| *a = ring.mul(a, scalar)),
        }
    }
    /// Whether every coefficient but the constant one is zero (in variable time).
    pub(crate) fn is_scalar(&self) -> bool {
        self.coeffs[1..].iter().all(U256::is_zero_vartime)
    }
    /// Additive inverse.
    pub fn neg(&self) -> Self {
        self.map(|x| self.ring.neg(x))
    }
    /// Scalar multiplication modulo $`q`$.
    pub fn scale(&self, scalar: i128) -> Result<Self, Error> {
        self.scale_u256(&self.ring.reduce_i128(scalar))
    }
    /// Scalar multiplication by a 256-bit value, reduced modulo $`q`$.
    pub fn scale_u256(&self, scalar: &U256) -> Result<Self, Error> {
        // Variable-time comparisons, as everywhere in this crate (no constant-time guarantee).
        let scalar = self.ring.reduce(scalar);
        if scalar.is_zero_vartime() {
            return Ok(Self::zero(self.ring.clone()));
        }
        if scalar.cmp_vartime(&U256::ONE).is_eq() {
            return Ok(self.clone());
        }
        if scalar
            .cmp_vartime(&self.ring.q.wrapping_sub(&U256::ONE))
            .is_eq()
        {
            return Ok(self.neg());
        }
        if let Some(multiplier) = self.ring.scalar_multiplier(&scalar) {
            return Ok(self.map(|x| multiplier.apply(x)));
        }
        Ok(self.map(|x| self.ring.mul(x, &scalar)))
    }
    /// Negacyclic multiplication through exact RNS/NTT and integer CRT.
    pub fn mul(&self, rhs: &Self) -> Result<Self, Error> {
        self.compatible(rhs)?;
        // Toolbox forms contain many zero, scalar and monomial coefficients. Preserve exact
        // ring multiplication without paying for transforms in these cases: a monomial
        // cX^k multiplies by rotating and scaling, (cX^k)a = c(X^ka).
        if self.is_scalar() {
            return rhs.scale_u256(&self.coeffs[0]);
        }
        if rhs.is_scalar() {
            return self.scale_u256(&rhs.coeffs[0]);
        }
        if let Some((k, c)) = self.monomial() {
            return rhs.rotate(k as i64).scale_u256(&c);
        }
        if let Some((k, c)) = rhs.monomial() {
            return self.rotate(k as i64).scale_u256(&c);
        }
        self.to_ntt().product(&rhs.to_ntt())?.reduce()
    }
    /// The index and value of the only nonzero coefficient, if exactly one is nonzero.
    pub(crate) fn monomial(&self) -> Option<(usize, U256)> {
        let mut found = None;
        for (i, x) in self.coeffs.iter().enumerate() {
            if !x.is_zero_vartime() {
                if found.is_some() {
                    return None;
                }
                found = Some((i, *x));
            }
        }
        found
    }
    /// Explicit coefficient-to-NTT conversion.
    pub fn to_ntt(&self) -> PolyNtt {
        PolyNtt::from_poly(self)
    }
    /// Automorphism $`\sigma(X)=X^{-1}`$ (LNP22 Lemma 2.4).
    pub fn auto(&self) -> Self {
        let mut out = self.clone();
        for i in 1..self.ring.d {
            out.coeffs[i] = self.ring.neg(&self.coeffs[self.ring.d - i]);
        }
        out
    }
    /// Trace $`(a+\sigma(a))/2`$ (LNP22 §4.4), requiring an odd modulus.
    pub fn trace(&self) -> Result<Self, Error> {
        if !bool::from(self.ring.q.is_odd()) {
            return Err(Error::Parameter("trace requires invertible two"));
        }
        // (q+1)/2 without overflow for odd q.
        let half = self.ring.half.wrapping_add(&U256::ONE);
        self.add(&self.auto()).expect("same ring").scale_u256(&half)
    }
    /// Multiply by $`X^j`$, supporting negative exponents.
    pub fn rotate(&self, j: i64) -> Self {
        let d = self.ring.d;
        let shift = j.rem_euclid(2 * d as i64) as usize;
        let mut out = Self::zero(self.ring.clone());
        for (i, x) in self.coeffs.iter().enumerate() {
            let k = i + shift;
            out.coeffs[k % d] = if (k / d).is_multiple_of(2) {
                *x
            } else {
                self.ring.neg(x)
            };
        }
        out
    }
    /// Squared Euclidean norm of the centred representatives: exact below $`2^{256}`$, and
    /// `U256::MAX` otherwise. A comparison with a bound of at most $`2^{256}-2`$ is therefore
    /// exact; the protocol's bounds are below $`2^{128}`$.
    pub fn norm_squared(&self) -> U256 {
        let mut sum = U256::ZERO;
        for x in &self.coeffs {
            let m = self.ring.magnitude(x);
            let square = match int::to_u128(&m) {
                Some(m) if m >> 64 == 0 => U256::from_u128(m * m),
                Some(_) => m.wrapping_mul(&m),
                None => return U256::MAX,
            };
            match Option::<U256>::from(crypto_bigint::CheckedAdd::checked_add(&sum, &square)) {
                Some(s) => sum = s,
                None => return U256::MAX,
            }
        }
        sum
    }
    /// Maximum absolute value of a centred coefficient.
    pub fn norm_infinity(&self) -> U256 {
        self.coeffs
            .iter()
            .map(|x| self.ring.magnitude(x))
            .max()
            .unwrap_or(U256::ZERO)
    }
    /// Test equality with zero (in variable time).
    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(U256::is_zero_vartime)
    }
}
