use super::{
    PolyNtt, Ring, int,
    mod128::{Mod128, add_mod128, shoup_mul128, sub_mod128},
    ntt::{add_mod, shoup, shoup_mul, sub_mod},
};
use crate::Error;
use crypto_bigint::{CheckedAdd, U256};
use std::sync::{Arc, OnceLock};
use zeroize::{DefaultIsZeroes, Zeroize, Zeroizing};

/// Canonical coefficients in $`[0,q)`$, in the words of the ring's native arithmetic: one word
/// below $`2^{63}`$ (the ring's `word`), two words from $`2^{63}`$ to $`2^{126}`$ (its `wide`
/// context), 256-bit values above. The ring decides the variant, so two polynomials of one ring
/// always have the same one. Wiped on drop.
#[derive(Clone, PartialEq, Eq)]
pub(crate) enum Coeffs {
    Word(Vec<u64>),
    Double(Vec<u128>),
    Wide(Vec<U256>),
}
impl Drop for Coeffs {
    fn drop(&mut self) {
        if !crate::wipe::ENABLED {
            return;
        }
        match self {
            Coeffs::Word(v) => wipe(v),
            Coeffs::Double(v) => wipe(v),
            Coeffs::Wide(v) => wipe_u256(v),
        }
    }
}
/// Each value once with a volatile write, then any spare capacity, as the residues of a
/// transform are wiped. `Vec::zeroize` would wipe every value a second time, bytewise.
fn wipe<T: DefaultIsZeroes>(v: &mut Vec<T>) {
    v.as_mut_slice().zeroize();
    v.spare_capacity_mut().zeroize();
}
/// The limbs of each value, then any spare capacity.
fn wipe_u256(v: &mut Vec<U256>) {
    for x in v.iter_mut() {
        x.as_mut_words().zeroize();
    }
    v.spare_capacity_mut().zeroize();
}
impl Coeffs {
    /// The $`d`$ zero coefficients of `ring`, in its width.
    fn zeros(ring: &Ring) -> Self {
        if ring.word.is_some() {
            Coeffs::Word(vec![0; ring.d])
        } else if ring.wide.is_some() {
            Coeffs::Double(vec![0; ring.d])
        } else {
            Coeffs::Wide(vec![U256::ZERO; ring.d])
        }
    }
    /// Canonical 256-bit values in the width of `ring`. Values that are copied into narrower
    /// words are wiped.
    pub(crate) fn from_u256(ring: &Ring, mut values: Vec<U256>) -> Self {
        if ring.word.is_some() {
            let out = values.iter().map(int::low_u64).collect();
            wipe_u256(&mut values);
            Coeffs::Word(out)
        } else if ring.wide.is_some() {
            let out = values.iter().map(int::low_u128).collect();
            wipe_u256(&mut values);
            Coeffs::Double(out)
        } else {
            Coeffs::Wide(values)
        }
    }
    #[inline]
    fn len(&self) -> usize {
        match self {
            Coeffs::Word(v) => v.len(),
            Coeffs::Double(v) => v.len(),
            Coeffs::Wide(v) => v.len(),
        }
    }
    /// Coefficient `i` as a 256-bit value.
    #[inline]
    pub(crate) fn get(&self, i: usize) -> U256 {
        match self {
            Coeffs::Word(v) => U256::from_u64(v[i]),
            Coeffs::Double(v) => U256::from_u128(v[i]),
            Coeffs::Wide(v) => v[i],
        }
    }
    /// Set coefficient `i` to a canonical value.
    #[inline]
    fn set(&mut self, i: usize, value: &U256) {
        match self {
            Coeffs::Word(v) => v[i] = int::low_u64(value),
            Coeffs::Double(v) => v[i] = int::low_u128(value),
            Coeffs::Wide(v) => v[i] = *value,
        }
    }
    fn to_u256(&self) -> Vec<U256> {
        match self {
            Coeffs::Word(v) => v.iter().map(|x| U256::from_u64(*x)).collect(),
            Coeffs::Double(v) => v.iter().map(|x| U256::from_u128(*x)).collect(),
            Coeffs::Wide(v) => v.clone(),
        }
    }
}

/// The canonical representative of `x` modulo $`q<2^{63}`$.
#[inline]
fn reduce_i128_word(x: i128, q: u64) -> u64 {
    let m = x.unsigned_abs();
    let r = if m < u128::from(q) {
        m as u64
    } else if let Ok(m) = u64::try_from(m) {
        m % q
    } else {
        (m % u128::from(q)) as u64
    };
    if x < 0 && r != 0 { q - r } else { r }
}
/// The canonical representative of `x` modulo $`q<2^{127}`$.
#[inline]
fn reduce_i128_double(x: i128, q: u128) -> u128 {
    let m = x.unsigned_abs();
    let r = if m < q { m } else { m % q };
    if x < 0 && r != 0 { q - r } else { r }
}
/// $`-x\bmod q`$ for a canonical value.
#[inline(always)]
fn neg_word(x: u64, q: u64) -> u64 {
    if x == 0 { 0 } else { q - x }
}
#[inline(always)]
fn neg_double(x: u128, q: u128) -> u128 {
    if x == 0 { 0 } else { q - x }
}
/// The absolute value of the centred representative in $`(-q/2,q/2]`$.
#[inline(always)]
fn magnitude_double(x: u128, q: u128) -> u128 {
    if x > q >> 1 { q - x } else { x }
}
/// The centred representative, which fits `i128` for $`q<2^{127}`$.
#[inline(always)]
fn centred_double(x: u128, q: u128) -> i128 {
    if x > q >> 1 {
        -((q - x) as i128)
    } else {
        x as i128
    }
}
/// $`\sum_i m_i^2`$ for magnitudes $`m_i<2^{128}`$, exact below $`2^{256}`$ and `U256::MAX`
/// otherwise.
fn sum_squares(magnitudes: impl Iterator<Item = u128>) -> U256 {
    let mut sum = U256::ZERO;
    for m in magnitudes {
        let square = if m >> 64 == 0 {
            U256::from_u128(m * m)
        } else {
            let m = U256::from_u128(m);
            m.wrapping_mul(&m)
        };
        match Option::<U256>::from(CheckedAdd::checked_add(&sum, &square)) {
            Some(s) => sum = s,
            None => return U256::MAX,
        }
    }
    sum
}
/// The index of the only value that is not zero: `Some(None)` when all are zero, `None` when
/// two or more are not.
fn sole_nonzero_index<T>(values: &[T], is_zero: impl Fn(&T) -> bool) -> Option<Option<usize>> {
    let mut found = None;
    for (i, x) in values.iter().enumerate() {
        if !is_zero(x) {
            if found.is_some() {
                return None;
            }
            found = Some(i);
        }
    }
    Some(found)
}
/// $`X^j`$ times the coefficients `v` for $`0\le j<2d`$: coefficient $`i`$ moves to
/// $`(i+j)\bmod d`$, negated when $`\lfloor(i+j)/d\rfloor`$ is odd ($`X^d=-1`$).
fn rotated<T: Copy + Default>(v: &[T], shift: usize, neg: impl Fn(T) -> T) -> Vec<T> {
    let d = v.len();
    let mut out = vec![T::default(); d];
    for (i, x) in v.iter().enumerate() {
        let k = i + shift;
        out[k % d] = if (k / d).is_multiple_of(2) {
            *x
        } else {
            neg(*x)
        };
    }
    out
}
/// $`\sigma(X)=X^{-1}`$ on coefficients: $`x_0`$, then $`-x_{d-i}`$ for $`0<i<d`$.
fn automorphism<T: Copy>(v: &[T], neg: impl Fn(T) -> T) -> Vec<T> {
    let d = v.len();
    (0..d)
        .map(|i| if i == 0 { v[0] } else { neg(v[d - i]) })
        .collect()
}

/// Multiplication by one reduced scalar $`w`$ modulo $`q<2^{63}`$ by Shoup products (see
/// `ntt::shoup_mul`): the quotient constant is computed once.
#[derive(Clone, Copy)]
struct ScalarWord {
    w: u64,
    w_shoup: u64,
    q: u64,
}
impl ScalarWord {
    fn new(scalar: &U256, q: u64) -> Self {
        let w = int::low_u64(scalar);
        debug_assert!(w < q && int::to_u64(scalar) == Some(w));
        Self {
            w,
            w_shoup: shoup(w, q),
            q,
        }
    }
    #[inline(always)]
    fn mul(&self, x: u64) -> u64 {
        shoup_mul(x, self.w, self.w_shoup, self.q)
    }
}
/// The same modulo $`2^{63}\le q<2^{126}`$ (see `mod128::shoup_mul128`).
#[derive(Clone, Copy)]
struct ScalarDouble {
    w: u128,
    w_shoup: u128,
    q: u128,
}
impl ScalarDouble {
    fn new(scalar: &U256, m: &Mod128) -> Self {
        let w = int::low_u128(scalar);
        debug_assert!(w < m.modulus() && int::to_u128(scalar) == Some(w));
        Self {
            w,
            w_shoup: m.shoup(w),
            q: m.modulus(),
        }
    }
    #[inline(always)]
    fn mul(&self, x: u128) -> u128 {
        shoup_mul128(x, self.w, self.w_shoup, self.q)
    }
}

/// Coefficients stored as canonical representatives in $`[0,q)`$, in the words of the ring's
/// arithmetic; buffers are wiped on drop.
///
/// Norms and the narrow accessors use the centred representatives in $`(-q/2,q/2]`$.
pub struct Poly {
    pub(crate) ring: Arc<Ring>,
    coeffs: Coeffs,
    /// The 256-bit copy that [`Poly::coefficients`] returns, made on its first call and wiped
    /// when the coefficients change or the polynomial is dropped.
    view: OnceLock<Box<[U256]>>,
}
impl Clone for Poly {
    fn clone(&self) -> Self {
        Self::with(self.ring.clone(), self.coeffs.clone())
    }
}
impl PartialEq for Poly {
    fn eq(&self, rhs: &Self) -> bool {
        self.ring == rhs.ring && self.coeffs == rhs.coeffs
    }
}
impl Eq for Poly {}
impl core::fmt::Debug for Poly {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Poly")
            .field("degree", &self.ring.d)
            .finish_non_exhaustive()
    }
}
impl Zeroize for Poly {
    /// Overwrites every coefficient with zero (one volatile write each, as on drop) and wipes
    /// the copy of [`Poly::coefficients`]: the polynomial becomes the zero polynomial of its
    /// ring.
    fn zeroize(&mut self) {
        if let Some(mut view) = self.view.take() {
            for x in view.iter_mut() {
                x.as_mut_words().zeroize();
            }
        }
        match &mut self.coeffs {
            Coeffs::Word(v) => v.as_mut_slice().zeroize(),
            Coeffs::Double(v) => v.as_mut_slice().zeroize(),
            Coeffs::Wide(v) => {
                for x in v.iter_mut() {
                    x.as_mut_words().zeroize();
                }
            }
        }
    }
}
impl Drop for Poly {
    fn drop(&mut self) {
        // The coefficients wipe themselves (`Coeffs`), the 256-bit copy here.
        if !crate::wipe::ENABLED {
            return;
        }
        if let Some(view) = self.view.get_mut() {
            for x in view.iter_mut() {
                x.as_mut_words().zeroize();
            }
        }
    }
}

impl Poly {
    fn with(ring: Arc<Ring>, coeffs: Coeffs) -> Self {
        debug_assert_eq!(coeffs.len(), ring.d);
        Self {
            ring,
            coeffs,
            view: OnceLock::new(),
        }
    }
    /// Wipe and drop the copy of [`Poly::coefficients`] before the coefficients change.
    fn invalidate(&mut self) {
        if let Some(mut view) = self.view.take() {
            for x in view.iter_mut() {
                crate::wipe::zeroize(x.as_mut_words());
            }
        }
    }
    /// The modulus of a ring with one-word coefficients.
    #[inline]
    fn word_modulus(&self) -> u64 {
        self.ring.word.expect("one-word coefficients")
    }
    /// The context of a ring with two-word coefficients.
    #[inline]
    fn double_modulus(&self) -> u128 {
        self.ring
            .wide
            .as_ref()
            .expect("two-word coefficients")
            .modulus()
    }
    /// Reduce an exact-length coefficient vector into the ring.
    pub fn new(ring: Arc<Ring>, mut coeffs: Vec<i128>) -> Result<Self, Error> {
        if coeffs.len() != ring.d {
            crate::wipe::zeroize(&mut coeffs);
            return Err(Error::Dimension);
        }
        let reduced = if let Some(q) = ring.word {
            Coeffs::Word(coeffs.iter().map(|x| reduce_i128_word(*x, q)).collect())
        } else if let Some(m) = &ring.wide {
            let q = m.modulus();
            Coeffs::Double(coeffs.iter().map(|x| reduce_i128_double(*x, q)).collect())
        } else {
            Coeffs::Wide(coeffs.iter().map(|x| ring.reduce_i128(*x)).collect())
        };
        crate::wipe::zeroize(&mut coeffs);
        Ok(Self::with(ring, reduced))
    }
    /// Reduce an exact-length vector of 256-bit values into the ring.
    pub fn from_u256(ring: Arc<Ring>, mut coeffs: Vec<U256>) -> Result<Self, Error> {
        if coeffs.len() != ring.d {
            crate::wipe::zeroize(&mut coeffs);
            return Err(Error::Dimension);
        }
        for x in &mut coeffs {
            *x = ring.reduce(x);
        }
        let coeffs = Coeffs::from_u256(&ring, coeffs);
        Ok(Self::with(ring, coeffs))
    }
    /// Values already in $`[0,q)`$, as the arithmetic produces them.
    pub(crate) fn from_canonical(ring: Arc<Ring>, coeffs: Vec<U256>) -> Self {
        debug_assert!(coeffs.len() == ring.d && coeffs.iter().all(|x| x < &ring.q));
        let coeffs = Coeffs::from_u256(&ring, coeffs);
        Self::with(ring, coeffs)
    }
    /// Canonical values below $`2^{128}`$, as decoders produce them, in the ring's width.
    pub(crate) fn from_canonical_u128(ring: Arc<Ring>, mut values: Vec<u128>) -> Self {
        debug_assert!(
            values.len() == ring.d && values.iter().all(|x| U256::from_u128(*x) < ring.q)
        );
        let coeffs = if ring.word.is_some() {
            let out = values.iter().map(|x| *x as u64).collect();
            wipe(&mut values);
            Coeffs::Word(out)
        } else if ring.wide.is_some() {
            Coeffs::Double(values)
        } else {
            let out = values.iter().map(|x| U256::from_u128(*x)).collect();
            wipe(&mut values);
            Coeffs::Wide(out)
        };
        Self::with(ring, coeffs)
    }
    /// Canonical coefficients already in the ring's width.
    pub(crate) fn from_coeffs(ring: Arc<Ring>, coeffs: Coeffs) -> Self {
        debug_assert!(match &coeffs {
            Coeffs::Word(v) => ring.word.is_some_and(|q| v.iter().all(|x| *x < q)),
            Coeffs::Double(v) =>
                ring.word.is_none()
                    && ring
                        .wide
                        .is_some_and(|m| v.iter().all(|x| *x < m.modulus())),
            Coeffs::Wide(v) => ring.wide.is_none() && v.iter().all(|x| x < &ring.q),
        });
        Self::with(ring, coeffs)
    }
    /// Zero polynomial.
    pub fn zero(ring: Arc<Ring>) -> Self {
        let coeffs = Coeffs::zeros(&ring);
        Self::with(ring, coeffs)
    }
    /// Constant polynomial, reduced modulo the ring modulus.
    pub fn constant(ring: Arc<Ring>, value: i128) -> Self {
        let value = ring.reduce_i128(value);
        Self::constant_u256(ring, &value)
    }
    /// Constant polynomial from a 256-bit value, reduced modulo the ring modulus.
    pub fn constant_u256(ring: Arc<Ring>, value: &U256) -> Self {
        let mut out = Self::zero(ring);
        let value = out.ring.reduce(value);
        out.coeffs.set(0, &value);
        out
    }
    /// Shared ring parameters.
    #[inline]
    pub fn ring(&self) -> &Arc<Ring> {
        &self.ring
    }
    /// Canonical coefficients in $`[0,q)`$, as 256-bit values. The polynomial keeps its
    /// coefficients in the words of the ring's arithmetic, so the first call makes this copy,
    /// which is kept until the coefficients change.
    pub fn coefficients(&self) -> &[U256] {
        self.view
            .get_or_init(|| self.coeffs.to_u256().into_boxed_slice())
    }
    /// The coefficients in the ring's width.
    #[inline]
    pub(crate) fn coeffs(&self) -> &Coeffs {
        &self.coeffs
    }
    /// Coefficient `i`, canonical, as a 256-bit value.
    #[inline]
    pub(crate) fn coefficient(&self, i: usize) -> U256 {
        self.coeffs.get(i)
    }
    /// The canonical coefficients in order, as 256-bit values, without a copy of them all.
    pub(crate) fn values(&self) -> impl ExactSizeIterator<Item = U256> + '_ {
        (0..self.ring.d).map(|i| self.coeffs.get(i))
    }
    /// The canonical coefficients in order as `u128`, for $`q<2^{128}`$.
    pub(crate) fn values_u128(&self) -> impl ExactSizeIterator<Item = u128> + '_ {
        debug_assert!(self.ring.narrow.is_some());
        (0..self.ring.d).map(|i| match &self.coeffs {
            Coeffs::Word(v) => u128::from(v[i]),
            Coeffs::Double(v) => v[i],
            Coeffs::Wide(v) => int::low_u128(&v[i]),
        })
    }
    /// The centred coefficient `i`, if it fits `i128` (which needs $`q\ge2^{128}`$ not to).
    #[inline]
    pub(crate) fn centred(&self, i: usize) -> Option<i128> {
        match &self.coeffs {
            Coeffs::Word(v) => Some(centred_double(
                u128::from(v[i]),
                u128::from(self.word_modulus()),
            )),
            Coeffs::Double(v) => Some(centred_double(v[i], self.double_modulus())),
            Coeffs::Wide(v) => self.ring.centred_i128(&v[i]),
        }
    }
    /// The magnitude of the centred coefficient `i`.
    #[inline]
    /// The magnitudes of the centred coefficients folded with `f` from `init`, where they
    /// are one or two words (moduli below $`2^{128}`$, magnitudes below $`2^{127}`$), and
    /// `None` for wider coefficients.
    pub(crate) fn fold_magnitudes<A>(&self, init: A, mut f: impl FnMut(A, u128) -> A) -> Option<A> {
        match &self.coeffs {
            Coeffs::Word(v) => {
                let q = u128::from(self.word_modulus());
                Some(
                    v.iter()
                        .fold(init, |a, x| f(a, magnitude_double(u128::from(*x), q))),
                )
            }
            Coeffs::Double(v) => {
                let q = self.double_modulus();
                Some(v.iter().fold(init, |a, x| f(a, magnitude_double(*x, q))))
            }
            Coeffs::Wide(_) => None,
        }
    }
    pub(crate) fn magnitude(&self, i: usize) -> U256 {
        match &self.coeffs {
            Coeffs::Word(v) => U256::from_u128(magnitude_double(
                u128::from(v[i]),
                u128::from(self.word_modulus()),
            )),
            Coeffs::Double(v) => U256::from_u128(magnitude_double(v[i], self.double_modulus())),
            Coeffs::Wide(v) => self.ring.magnitude(&v[i]),
        }
    }
    /// Centred coefficients in $`(-q/2,q/2]`$ as `i128`, for short vectors. Fails with
    /// [`Error::Overflow`] if one does not fit, which can happen only for $`q\ge2^{128}`$.
    pub fn coefficients_i128(&self) -> Result<Zeroizing<Vec<i128>>, Error> {
        let mut out = Zeroizing::new(Vec::with_capacity(self.ring.d));
        for i in 0..self.ring.d {
            out.push(self.centred(i).ok_or(Error::Overflow)?);
        }
        Ok(out)
    }
    /// One centred coefficient as `i128`, with the errors of [`Poly::coefficients_i128`] and
    /// [`Error::Index`] for an index outside the degree.
    pub fn coefficient_i128(&self, i: usize) -> Result<i128, Error> {
        if i >= self.ring.d {
            return Err(Error::Index);
        }
        self.centred(i).ok_or(Error::Overflow)
    }
    /// Update one coefficient and reduce it.
    pub fn set_coefficient(&mut self, i: usize, value: i128) -> Result<(), Error> {
        if i >= self.ring.d {
            return Err(Error::Index);
        }
        self.invalidate();
        let value = self.ring.reduce_i128(value);
        self.coeffs.set(i, &value);
        Ok(())
    }
    #[inline]
    pub(crate) fn compatible(&self, rhs: &Self) -> Result<(), Error> {
        if self.ring != rhs.ring {
            Err(Error::RingMismatch)
        } else {
            Ok(())
        }
    }
    /// Coefficient-wise addition.
    pub fn add(&self, rhs: &Self) -> Result<Self, Error> {
        self.compatible(rhs)?;
        let coeffs = match (&self.coeffs, &rhs.coeffs) {
            (Coeffs::Word(a), Coeffs::Word(b)) => {
                let q = self.word_modulus();
                Coeffs::Word(a.iter().zip(b).map(|(a, b)| add_mod(*a, *b, q)).collect())
            }
            (Coeffs::Double(a), Coeffs::Double(b)) => {
                let q = self.double_modulus();
                Coeffs::Double(
                    a.iter()
                        .zip(b)
                        .map(|(a, b)| add_mod128(*a, *b, q))
                        .collect(),
                )
            }
            (Coeffs::Wide(a), Coeffs::Wide(b)) => {
                Coeffs::Wide(a.iter().zip(b).map(|(a, b)| self.ring.add(a, b)).collect())
            }
            _ => unreachable!("one ring, one width"),
        };
        Ok(Self::with(self.ring.clone(), coeffs))
    }
    /// Coefficient-wise subtraction.
    pub fn sub(&self, rhs: &Self) -> Result<Self, Error> {
        self.compatible(rhs)?;
        let coeffs = match (&self.coeffs, &rhs.coeffs) {
            (Coeffs::Word(a), Coeffs::Word(b)) => {
                let q = self.word_modulus();
                Coeffs::Word(a.iter().zip(b).map(|(a, b)| sub_mod(*a, *b, q)).collect())
            }
            (Coeffs::Double(a), Coeffs::Double(b)) => {
                let q = self.double_modulus();
                Coeffs::Double(
                    a.iter()
                        .zip(b)
                        .map(|(a, b)| sub_mod128(*a, *b, q))
                        .collect(),
                )
            }
            (Coeffs::Wide(a), Coeffs::Wide(b)) => {
                Coeffs::Wide(a.iter().zip(b).map(|(a, b)| self.ring.sub(a, b)).collect())
            }
            _ => unreachable!("one ring, one width"),
        };
        Ok(Self::with(self.ring.clone(), coeffs))
    }
    /// `self += rhs`: the value of [`Poly::add`], in place.
    pub(crate) fn add_assign(&mut self, rhs: &Self) -> Result<(), Error> {
        self.compatible(rhs)?;
        self.invalidate();
        let (word, wide) = (self.ring.word, self.ring.wide);
        match (&mut self.coeffs, &rhs.coeffs) {
            (Coeffs::Word(a), Coeffs::Word(b)) => {
                let q = word.expect("one-word coefficients");
                for (a, b) in a.iter_mut().zip(b) {
                    *a = add_mod(*a, *b, q);
                }
            }
            (Coeffs::Double(a), Coeffs::Double(b)) => {
                let q = wide.expect("two-word coefficients").modulus();
                for (a, b) in a.iter_mut().zip(b) {
                    *a = add_mod128(*a, *b, q);
                }
            }
            (Coeffs::Wide(a), Coeffs::Wide(b)) => {
                for (a, b) in a.iter_mut().zip(b) {
                    *a = self.ring.add(a, b);
                }
            }
            _ => unreachable!("one ring, one width"),
        }
        Ok(())
    }
    /// `self -= rhs`: the value of [`Poly::sub`], in place.
    pub(crate) fn sub_assign(&mut self, rhs: &Self) -> Result<(), Error> {
        self.compatible(rhs)?;
        self.invalidate();
        let (word, wide) = (self.ring.word, self.ring.wide);
        match (&mut self.coeffs, &rhs.coeffs) {
            (Coeffs::Word(a), Coeffs::Word(b)) => {
                let q = word.expect("one-word coefficients");
                for (a, b) in a.iter_mut().zip(b) {
                    *a = sub_mod(*a, *b, q);
                }
            }
            (Coeffs::Double(a), Coeffs::Double(b)) => {
                let q = wide.expect("two-word coefficients").modulus();
                for (a, b) in a.iter_mut().zip(b) {
                    *a = sub_mod128(*a, *b, q);
                }
            }
            (Coeffs::Wide(a), Coeffs::Wide(b)) => {
                for (a, b) in a.iter_mut().zip(b) {
                    *a = self.ring.sub(a, b);
                }
            }
            _ => unreachable!("one ring, one width"),
        }
        Ok(())
    }
    /// `self += rhs * scalar` for a canonical scalar: the value of
    /// `self.add(&rhs.scale_u256(scalar)?)`, in place.
    pub(crate) fn add_scaled_assign(&mut self, rhs: &Self, scalar: &U256) -> Result<(), Error> {
        self.compatible(rhs)?;
        debug_assert!(scalar < &self.ring.q);
        if scalar.is_zero_vartime() {
            return Ok(());
        }
        if scalar.cmp_vartime(&U256::ONE).is_eq() {
            return self.add_assign(rhs);
        }
        self.invalidate();
        let (word, wide) = (self.ring.word, self.ring.wide);
        match (&mut self.coeffs, &rhs.coeffs) {
            (Coeffs::Word(a), Coeffs::Word(b)) => {
                let m = ScalarWord::new(scalar, word.expect("one-word coefficients"));
                for (a, b) in a.iter_mut().zip(b) {
                    *a = add_mod(*a, m.mul(*b), m.q);
                }
            }
            (Coeffs::Double(a), Coeffs::Double(b)) => {
                let m = ScalarDouble::new(scalar, &wide.expect("two-word coefficients"));
                for (a, b) in a.iter_mut().zip(b) {
                    *a = add_mod128(*a, m.mul(*b), m.q);
                }
            }
            (Coeffs::Wide(a), Coeffs::Wide(b)) => {
                let ring = &self.ring;
                for (a, b) in a.iter_mut().zip(b) {
                    *a = ring.add(a, &ring.mul(b, scalar));
                }
            }
            _ => unreachable!("one ring, one width"),
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
        let d = self.ring.d;
        debug_assert!(k < d && c < &self.ring.q);
        if c.is_zero_vartime() {
            return Ok(());
        }
        self.invalidate();
        let (word, wide) = (self.ring.word, self.ring.wide);
        // X^k x_i is x_i at i + k, negated past X^d = -1.
        match (&mut self.coeffs, &rhs.coeffs) {
            (Coeffs::Word(a), Coeffs::Word(b)) => {
                let m = ScalarWord::new(c, word.expect("one-word coefficients"));
                let (low, high) = b.split_at(d - k);
                for (a, b) in a[k..].iter_mut().zip(low) {
                    *a = add_mod(*a, m.mul(*b), m.q);
                }
                for (a, b) in a[..k].iter_mut().zip(high) {
                    *a = sub_mod(*a, m.mul(*b), m.q);
                }
            }
            (Coeffs::Double(a), Coeffs::Double(b)) => {
                let m = ScalarDouble::new(c, &wide.expect("two-word coefficients"));
                let (low, high) = b.split_at(d - k);
                for (a, b) in a[k..].iter_mut().zip(low) {
                    *a = add_mod128(*a, m.mul(*b), m.q);
                }
                for (a, b) in a[..k].iter_mut().zip(high) {
                    *a = sub_mod128(*a, m.mul(*b), m.q);
                }
            }
            (Coeffs::Wide(a), Coeffs::Wide(b)) => {
                let ring = &self.ring;
                let (low, high) = b.split_at(d - k);
                for (a, b) in a[k..].iter_mut().zip(low) {
                    *a = ring.add(a, &ring.mul(b, c));
                }
                for (a, b) in a[..k].iter_mut().zip(high) {
                    *a = ring.sub(a, &ring.mul(b, c));
                }
            }
            _ => unreachable!("one ring, one width"),
        }
        Ok(())
    }
    /// `self *= scalar` for a canonical scalar: the value of [`Poly::scale_u256`], in place.
    pub(crate) fn scale_assign(&mut self, scalar: &U256) {
        debug_assert!(scalar < &self.ring.q);
        self.invalidate();
        let (word, wide) = (self.ring.word, self.ring.wide);
        match &mut self.coeffs {
            Coeffs::Word(a) => {
                let m = ScalarWord::new(scalar, word.expect("one-word coefficients"));
                a.iter_mut().for_each(|a| *a = m.mul(*a));
            }
            Coeffs::Double(a) => {
                let m = ScalarDouble::new(scalar, &wide.expect("two-word coefficients"));
                a.iter_mut().for_each(|a| *a = m.mul(*a));
            }
            Coeffs::Wide(a) => {
                let ring = &self.ring;
                a.iter_mut().for_each(|a| *a = ring.mul(a, scalar));
            }
        }
    }
    /// Whether every coefficient but the constant one is zero (in variable time).
    pub(crate) fn is_scalar(&self) -> bool {
        match &self.coeffs {
            Coeffs::Word(v) => v[1..].iter().all(|x| *x == 0),
            Coeffs::Double(v) => v[1..].iter().all(|x| *x == 0),
            Coeffs::Wide(v) => v[1..].iter().all(U256::is_zero_vartime),
        }
    }
    /// Additive inverse.
    pub fn neg(&self) -> Self {
        let coeffs = match &self.coeffs {
            Coeffs::Word(v) => {
                let q = self.word_modulus();
                Coeffs::Word(v.iter().map(|x| neg_word(*x, q)).collect())
            }
            Coeffs::Double(v) => {
                let q = self.double_modulus();
                Coeffs::Double(v.iter().map(|x| neg_double(*x, q)).collect())
            }
            Coeffs::Wide(v) => Coeffs::Wide(v.iter().map(|x| self.ring.neg(x)).collect()),
        };
        Self::with(self.ring.clone(), coeffs)
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
        let coeffs = match &self.coeffs {
            Coeffs::Word(v) => {
                let m = ScalarWord::new(&scalar, self.word_modulus());
                Coeffs::Word(v.iter().map(|x| m.mul(*x)).collect())
            }
            Coeffs::Double(v) => {
                let m = ScalarDouble::new(&scalar, self.ring.wide.as_ref().expect("two words"));
                Coeffs::Double(v.iter().map(|x| m.mul(*x)).collect())
            }
            Coeffs::Wide(v) => Coeffs::Wide(v.iter().map(|x| self.ring.mul(x, &scalar)).collect()),
        };
        Ok(Self::with(self.ring.clone(), coeffs))
    }
    /// Negacyclic multiplication through exact RNS/NTT and integer CRT.
    pub fn mul(&self, rhs: &Self) -> Result<Self, Error> {
        self.compatible(rhs)?;
        // Toolbox forms contain many zero, scalar and monomial coefficients. Preserve exact
        // ring multiplication without paying for transforms in these cases: a monomial
        // cX^k multiplies by rotating and scaling, (cX^k)a = c(X^ka).
        if self.is_scalar() {
            return rhs.scale_u256(&self.coefficient(0));
        }
        if rhs.is_scalar() {
            return self.scale_u256(&rhs.coefficient(0));
        }
        if let Some((k, c)) = self.monomial() {
            return rhs.rotate(k as i64).scale_u256(&c);
        }
        if let Some((k, c)) = rhs.monomial() {
            return self.rotate(k as i64).scale_u256(&c);
        }
        self.to_ntt().product(&rhs.to_ntt())?.reduce()
    }
    /// The index and value of the only nonzero coefficient: `Some(None)` for the zero
    /// polynomial, `None` when two or more coefficients are nonzero (in variable time).
    pub(crate) fn sole_nonzero(&self) -> Option<Option<(usize, U256)>> {
        let index = match &self.coeffs {
            Coeffs::Word(v) => sole_nonzero_index(v, |x| *x == 0),
            Coeffs::Double(v) => sole_nonzero_index(v, |x| *x == 0),
            Coeffs::Wide(v) => sole_nonzero_index(v, U256::is_zero_vartime),
        }?;
        Some(index.map(|i| (i, self.coefficient(i))))
    }
    /// The index and value of the only nonzero coefficient, if exactly one is nonzero.
    pub(crate) fn monomial(&self) -> Option<(usize, U256)> {
        self.sole_nonzero().flatten()
    }
    /// Explicit coefficient-to-NTT conversion.
    pub fn to_ntt(&self) -> PolyNtt {
        PolyNtt::from_poly(self)
    }
    /// Automorphism $`\sigma(X)=X^{-1}`$ (LNP22 Lemma 2.4).
    pub fn auto(&self) -> Self {
        let coeffs = match &self.coeffs {
            Coeffs::Word(v) => {
                let q = self.word_modulus();
                Coeffs::Word(automorphism(v, |x| neg_word(x, q)))
            }
            Coeffs::Double(v) => {
                let q = self.double_modulus();
                Coeffs::Double(automorphism(v, |x| neg_double(x, q)))
            }
            Coeffs::Wide(v) => Coeffs::Wide(automorphism(v, |x| self.ring.neg(&x))),
        };
        Self::with(self.ring.clone(), coeffs)
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
        let coeffs = match &self.coeffs {
            Coeffs::Word(v) => {
                let q = self.word_modulus();
                Coeffs::Word(rotated(v, shift, |x| neg_word(x, q)))
            }
            Coeffs::Double(v) => {
                let q = self.double_modulus();
                Coeffs::Double(rotated(v, shift, |x| neg_double(x, q)))
            }
            Coeffs::Wide(v) => Coeffs::Wide(rotated(v, shift, |x| self.ring.neg(&x))),
        };
        Self::with(self.ring.clone(), coeffs)
    }
    /// Squared Euclidean norm of the centred representatives: exact below $`2^{256}`$, and
    /// `U256::MAX` otherwise. A comparison with a bound of at most $`2^{256}-2`$ is therefore
    /// exact; the protocol's bounds are below $`2^{128}`$.
    pub fn norm_squared(&self) -> U256 {
        match &self.coeffs {
            Coeffs::Word(v) => {
                let q = u128::from(self.word_modulus());
                sum_squares(v.iter().map(|x| magnitude_double(u128::from(*x), q)))
            }
            Coeffs::Double(v) => {
                let q = self.double_modulus();
                sum_squares(v.iter().map(|x| magnitude_double(*x, q)))
            }
            Coeffs::Wide(v) => {
                let mut sum = U256::ZERO;
                for x in v {
                    let m = self.ring.magnitude(x);
                    let square = match int::to_u128(&m) {
                        Some(m) if m >> 64 == 0 => U256::from_u128(m * m),
                        Some(_) => m.wrapping_mul(&m),
                        None => return U256::MAX,
                    };
                    match Option::<U256>::from(CheckedAdd::checked_add(&sum, &square)) {
                        Some(s) => sum = s,
                        None => return U256::MAX,
                    }
                }
                sum
            }
        }
    }
    /// Maximum absolute value of a centred coefficient.
    pub fn norm_infinity(&self) -> U256 {
        match &self.coeffs {
            Coeffs::Word(v) => {
                let q = u128::from(self.word_modulus());
                let max = v.iter().map(|x| magnitude_double(u128::from(*x), q)).max();
                U256::from_u128(max.unwrap_or(0))
            }
            Coeffs::Double(v) => {
                let q = self.double_modulus();
                U256::from_u128(v.iter().map(|x| magnitude_double(*x, q)).max().unwrap_or(0))
            }
            Coeffs::Wide(v) => v
                .iter()
                .map(|x| self.ring.magnitude(x))
                .max()
                .unwrap_or(U256::ZERO),
        }
    }
    /// Test equality with zero (in variable time).
    pub fn is_zero(&self) -> bool {
        match &self.coeffs {
            Coeffs::Word(v) => v.iter().all(|x| *x == 0),
            Coeffs::Double(v) => v.iter().all(|x| *x == 0),
            Coeffs::Wide(v) => v.iter().all(U256::is_zero_vartime),
        }
    }
}

#[cfg(test)]
mod storage_tests;
