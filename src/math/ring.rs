use super::{
    int::{self, Divisor64},
    ntt::{NttPlan, add_mod, pow_mod, shoup, shoup_mul, sub_mod},
};
use crate::{
    Error,
    params::moduli::{NADDS, NTT_PRIMES},
};
#[cfg(test)]
use crypto_bigint::Uint;
use crypto_bigint::{I256, NonZero, U64, U128, U192, U256, U384, U576};
use std::sync::Arc;

/// Mixed-radix CRT constants in an accumulator of `C` limbs: the reference reconstruction,
/// kept as a test oracle (`rns::reconstruct_with`).
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct Basis<const C: usize> {
    /// Product $`P`$ of the selected primes.
    pub(crate) product: Uint<C>,
    /// $`\lfloor P/2\rfloor`$: larger integers stand for negative values.
    pub(crate) half: Uint<C>,
    /// Products of the primes before each prime.
    pub(crate) prefixes: Vec<Uint<C>>,
}

/// The accumulator width: 256 bits for up to four primes, 384 for up to six, 576 for nine.
#[cfg(test)]
#[derive(Debug)]
pub(crate) enum Crt {
    Four(Basis<{ U256::LIMBS }>),
    Six(Basis<{ U384::LIMBS }>),
    Nine(Basis<{ U576::LIMBS }>),
}

/// Garner's mixed-radix digits and their sign, in native arithmetic.
///
/// With $`M_j=p_0\cdots p_{j-1}`$ and $`P=M_k`$, every $`X\in[0,P)`$ is $`\sum_jv_jM_j`$ with
/// mixed-radix digits $`v_j\in[0,p_j)`$, and $`v_i=(r_i-\sum_{j<i}v_jM_j)M_i^{-1}\bmod p_i`$
/// from its residues $`r_i`$. Digit vectors compare as the integers do, most significant
/// first, which decides whether $`X>\lfloor P/2\rfloor`$, that is whether the centred value
/// $`X-P`$ is meant. Every product is a Shoup product, so no step divides.
#[derive(Debug)]
pub(crate) struct Garner {
    primes: Vec<u64>,
    /// For each prime $`p_i`$: $`M_j\bmod p_i`$ for $`j<i`$, with Shoup constants.
    prefixes: Vec<Vec<(u64, u64)>>,
    /// $`M_i^{-1}\bmod p_i`$, with Shoup constants.
    inverses: Vec<(u64, u64)>,
    /// The mixed-radix digits of $`\lfloor P/2\rfloor`$.
    half: Vec<u64>,
}
impl Garner {
    /// The constants for at most nine distinct primes $`p_i<2^{62}`$, with the products
    /// $`M_j`$ and $`P`$.
    fn new(primes: &[u64]) -> (Self, Vec<U576>, U576) {
        assert!((1..=9).contains(&primes.len()));
        let mut prefixes = vec![U576::ONE];
        for p in &primes[..primes.len() - 1] {
            let last = prefixes[prefixes.len() - 1];
            prefixes.push(last.wrapping_mul(&U64::from_u64(*p)));
        }
        let product =
            prefixes[primes.len() - 1].wrapping_mul(&U64::from_u64(primes[primes.len() - 1]));
        let modulo = |x: &U576, m: u64| Divisor64::new(m).rem(x);
        let half = product.shr_vartime(1);
        let garner = Self {
            primes: primes.to_vec(),
            prefixes: primes
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    prefixes[..i]
                        .iter()
                        .map(|m| (modulo(m, *p), shoup(modulo(m, *p), *p)))
                        .collect()
                })
                .collect(),
            inverses: primes
                .iter()
                .zip(&prefixes)
                .map(|(p, m)| {
                    let inverse = pow_mod(modulo(m, *p), p - 2, *p);
                    (inverse, shoup(inverse, *p))
                })
                .collect(),
            half: primes
                .iter()
                .zip(&prefixes)
                .map(|(p, m)| {
                    let quotient = half.div_rem_vartime(&NonZero::new(*m).expect("nonzero")).0;
                    modulo(&quotient, *p)
                })
                .collect(),
        };
        (garner, prefixes, product)
    }
    /// The digits of the integer $`X\in[0,P)`$ whose residues are `residues[i][j]`, each in
    /// $`[0,p_i)`$, and whether $`X>\lfloor P/2\rfloor`$.
    #[inline(always)]
    fn digits(&self, residues: &[Vec<u64>], j: usize, digits: &mut [u64; 9]) -> bool {
        for (i, p) in self.primes.iter().enumerate() {
            let mut sum = 0;
            for (v, (w, w_shoup)) in digits[..i].iter().zip(&self.prefixes[i]) {
                sum = add_mod(sum, shoup_mul(*v, *w, *w_shoup, *p), *p);
            }
            let (w, w_shoup) = self.inverses[i];
            digits[i] = shoup_mul(sub_mod(residues[i][j], sum, *p), w, w_shoup, *p);
        }
        for (v, h) in digits[..self.primes.len()].iter().zip(&self.half).rev() {
            if v != h {
                return v > h;
            }
        }
        false
    }
}

/// The CRT for moduli $`q<2^{63}`$: [`Garner`]'s digits, then
/// $`X\bmod q=\sum_jv_j(M_j\bmod q)\bmod q`$ by Shoup products modulo $`q`$ (which need
/// $`q<2^{63}`$), less $`P\bmod q`$ for a centred value below zero. The result equals that of
/// the multi-limb reference reconstruction for every input.
#[derive(Debug)]
pub(crate) struct SmallCrt {
    garner: Garner,
    q: u64,
    /// $`\lfloor q/2\rfloor`$: canonical values above it are negative when centred.
    pub(crate) half_q: u64,
    /// $`M_j\bmod q`$, with Shoup constants modulo $`q`$.
    prefixes_q: Vec<(u64, u64)>,
    /// $`P\bmod q`$.
    product_q: u64,
}
impl SmallCrt {
    /// The constants for $`2\le q<2^{63}`$ and at most nine distinct primes $`p_i<2^{62}`$.
    pub(crate) fn new(q: u64, primes: &[u64]) -> Self {
        assert!((2..1 << 63).contains(&q));
        let (garner, prefixes, product) = Garner::new(primes);
        let modulo = |x: &U576| Divisor64::new(q).rem(x);
        Self {
            garner,
            q,
            half_q: q / 2,
            prefixes_q: prefixes
                .iter()
                .map(|m| (modulo(m), shoup(modulo(m), q)))
                .collect(),
            product_q: modulo(&product),
        }
    }
    /// The modulus $`q<2^{63}`$.
    pub(crate) fn modulus(&self) -> u64 {
        self.q
    }
    /// The canonical representative modulo $`q`$ of the centred integer whose residues modulo
    /// the primes are `residues[i][j]`, each in $`[0,p_i)`$.
    #[inline]
    pub(crate) fn reconstruct(&self, residues: &[Vec<u64>], j: usize) -> u64 {
        let mut digits = [0u64; 9];
        let negative = self.garner.digits(residues, j, &mut digits);
        let mut x = 0;
        for (v, (w, w_shoup)) in digits.iter().zip(&self.prefixes_q) {
            x = add_mod(x, shoup_mul(*v, *w, *w_shoup, self.q), self.q);
        }
        if negative {
            sub_mod(x, self.product_q, self.q)
        } else {
            x
        }
    }
}

/// The CRT for moduli $`q\ge2^{63}`$: [`Garner`]'s digits, then
/// $`\sum_jv_j(M_j\bmod q)<k2^{62}q<2^{66}q`$ (at most $`2^{322}`$, six limbs) reduced once
/// modulo $`q`$, less $`P\bmod q`$ for a centred value below zero: what the multi-limb
/// reference reconstruction returns, from a remainder of six limbs instead of up to nine and
/// without a multi-limb remainder per digit.
#[derive(Debug)]
pub(crate) struct WideCrt {
    garner: Garner,
    /// $`M_j\bmod q`$.
    prefixes_q: Vec<U256>,
    /// $`P\bmod q`$.
    product_q: U256,
}
impl WideCrt {
    /// The constants for a modulus $`q`$ and at most nine distinct primes $`p_i<2^{62}`$.
    pub(crate) fn new(q: &NonZero<U256>, primes: &[u64]) -> Self {
        let (garner, prefixes, product) = Garner::new(primes);
        Self {
            garner,
            prefixes_q: prefixes.iter().map(|m| m.rem_vartime(q)).collect(),
            product_q: product.rem_vartime(q),
        }
    }
    /// The canonical representative modulo $`q`$ of the centred integer whose residues modulo
    /// the primes are `residues[i][j]`, each in $`[0,p_i)`$; `small` is $`q`$ as a divisor
    /// when $`q<2^{64}`$.
    #[inline]
    pub(crate) fn reconstruct(
        &self,
        residues: &[Vec<u64>],
        j: usize,
        q: &NonZero<U256>,
        small: Option<&Divisor64>,
    ) -> U256 {
        let mut digits = [0u64; 9];
        let negative = self.garner.digits(residues, j, &mut digits);
        let mut sum = U384::ZERO;
        for (v, m) in digits.iter().zip(&self.prefixes_q) {
            sum = sum.wrapping_add(
                &m.resize::<{ U384::LIMBS }>()
                    .wrapping_mul(&U64::from_u64(*v)),
            );
        }
        let r = match small {
            Some(divisor) => U256::from_u64(divisor.rem(&sum)),
            None => sum.rem_vartime(q),
        };
        if negative {
            r.sub_mod(&self.product_q, q)
        } else {
            r
        }
    }
}

/// The CRT of a ring: native below $`2^{63}`$, six limbs above.
#[derive(Debug)]
pub(crate) enum RingCrt {
    Small(SmallCrt),
    Wide(WideCrt),
}

/// Multiplication by one reduced scalar $`w`$ modulo $`q<2^{63}`$, by Shoup products: the
/// quotient constant $`\lfloor w2^{64}/q\rfloor`$ is computed once (see `ntt::shoup_mul`).
pub(crate) struct ScalarMul {
    w: u64,
    w_shoup: u64,
    q: u64,
}
impl ScalarMul {
    /// $`xw\bmod q`$ for a canonical $`x`$.
    #[inline]
    pub(crate) fn apply(&self, x: &U256) -> U256 {
        U256::from_u64(shoup_mul(int::low_u64(x), self.w, self.w_shoup, self.q))
    }
}

/// Validated ring and exact RNS context, shared by polynomial values.
///
/// Moduli lie in $`[2,2^{256})`$; degrees are powers of two from 64 to 1024. Coefficients are
/// stored as canonical representatives in $`[0,q)`$. Proof-parameter validation imposes the
/// stronger LNP22 restrictions separately.
#[derive(Debug)]
pub struct Ring {
    pub(crate) q: U256,
    pub(crate) nonzero: NonZero<U256>,
    /// $`\lfloor q/2\rfloor`$: canonical values above it are negative when centred.
    pub(crate) half: U256,
    /// The modulus as a divisor when $`q<2^{64}`$.
    pub(crate) small: Option<Divisor64>,
    /// The modulus when $`q<2^{128}`$, for additions in native arithmetic.
    pub(crate) narrow: Option<u128>,
    /// Number of 64-bit words that canonical values can occupy.
    pub(crate) words: usize,
    pub(crate) d: usize,
    pub(crate) plans: Vec<NttPlan>,
    /// The selected primes as divisors.
    pub(crate) divisors: Vec<Divisor64>,
    /// $`q\bmod p_i`$ for each selected prime.
    pub(crate) q_mod_p: Vec<u64>,
    pub(crate) crt: RingCrt,
    /// The multi-limb reference CRT and its digit factors
    /// $`(p_0\cdots p_{i-1})^{-1}\bmod p_i`$, kept as a test oracle.
    #[cfg(test)]
    pub(crate) multi_limb: Crt,
    #[cfg(test)]
    pub(crate) inverses: Vec<u64>,
}
impl PartialEq for Ring {
    fn eq(&self, rhs: &Self) -> bool {
        self.q == rhs.q && self.d == rhs.d
    }
}
impl Eq for Ring {}

#[cfg(test)]
fn basis<const C: usize>(prefixes: &[U576], product: &U576) -> Basis<C> {
    let product = product.resize::<C>();
    Basis {
        product,
        half: product.shr_vartime(1),
        prefixes: prefixes.iter().map(|x| x.resize::<C>()).collect(),
    }
}

impl Ring {
    /// Construct a ring from an `i128` modulus, as [`Ring::with_modulus`].
    pub fn new(q: i128, d: usize) -> Result<Arc<Self>, Error> {
        if q < 2 {
            return Err(Error::Parameter("ring modulus"));
        }
        Self::with_modulus(U256::from_u128(q as u128), d)
    }
    /// Construct a ring, choosing enough RNS primes for 128 accumulated products.
    ///
    /// The primes are taken in the order of `NTT_PRIMES` until their product exceeds
    /// $`(q-1)^2\cdot d\cdot128`$, which nine primes do for every $`q<2^{256}`$.
    pub fn with_modulus(q: U256, d: usize) -> Result<Arc<Self>, Error> {
        if q < U256::from_u8(2) {
            return Err(Error::Parameter("ring modulus"));
        }
        if !(64..=1024).contains(&d) || !d.is_power_of_two() {
            return Err(Error::Parameter("ring degree"));
        }
        // (q-1)^2 d 128 < 2^529 fits 576 bits.
        let qm1 = q.wrapping_sub(&U256::ONE).resize::<{ U576::LIMBS }>();
        let bound = qm1
            .wrapping_mul(&qm1)
            .wrapping_mul(&U576::from_u64((d * NADDS) as u64))
            .wrapping_add(&U576::ONE);
        let mut product = U576::ONE;
        let mut plans = Vec::new();
        let mut divisors = Vec::new();
        let mut q_mod_p = Vec::new();
        #[cfg(test)]
        let (mut prefixes, mut inverses) = (Vec::new(), Vec::new());
        for (p, root) in NTT_PRIMES {
            let divisor = Divisor64::new(p);
            #[cfg(test)]
            {
                prefixes.push(product);
                inverses.push(pow_mod(divisor.rem(&product), p - 2, p));
            }
            plans.push(NttPlan::new(p, d, pow_mod(root, (1024 / d) as u64, p))?);
            q_mod_p.push(divisor.rem(&q));
            divisors.push(divisor);
            product = product.wrapping_mul(&U64::from_u64(p));
            if product > bound {
                break;
            }
        }
        if product <= bound {
            return Err(Error::Overflow);
        }
        let nonzero = NonZero::new(q).expect("modulus at least two");
        let primes: Vec<u64> = plans.iter().map(NttPlan::modulus).collect();
        let crt = match int::to_u64(&q).filter(|q| *q < 1 << 63) {
            Some(q) => RingCrt::Small(SmallCrt::new(q, &primes)),
            None => RingCrt::Wide(WideCrt::new(&nonzero, &primes)),
        };
        Ok(Arc::new(Self {
            q,
            nonzero,
            half: q.shr_vartime(1),
            small: int::to_u64(&q).map(Divisor64::new),
            narrow: int::to_u128(&q),
            words: int::bits(&(q.wrapping_sub(&U256::ONE))).div_ceil(64).max(1) as usize,
            d,
            plans,
            divisors,
            q_mod_p,
            crt,
            #[cfg(test)]
            multi_limb: match prefixes.len() {
                ..=4 => Crt::Four(basis(&prefixes, &product)),
                5..=6 => Crt::Six(basis(&prefixes, &product)),
                _ => Crt::Nine(basis(&prefixes, &product)),
            },
            #[cfg(test)]
            inverses,
        }))
    }
    /// The native CRT, when $`q<2^{63}`$.
    pub(crate) fn small_crt(&self) -> Option<&SmallCrt> {
        match &self.crt {
            RingCrt::Small(small) => Some(small),
            RingCrt::Wide(_) => None,
        }
    }
    /// Coefficient modulus.
    pub fn modulus(&self) -> U256 {
        self.q
    }
    /// Negacyclic degree.
    pub fn degree(&self) -> usize {
        self.d
    }
    /// Number of bits for a canonical coefficient in $`[0,q)`$.
    pub fn coefficient_bits(&self) -> u32 {
        int::bits(&self.q.wrapping_sub(&U256::ONE))
    }
    /// Number of NTT primes selected for exact multiplication.
    pub fn rns_primes(&self) -> usize {
        self.plans.len()
    }

    /// The modulus as `i128`, for tests over moduli below $`2^{127}`$.
    #[cfg(test)]
    pub(crate) fn modulus_i128(&self) -> i128 {
        int::to_u128(&self.q)
            .and_then(|q| i128::try_from(q).ok())
            .expect("test modulus below 2^127")
    }
    /// The canonical representative of an integer.
    pub(crate) fn reduce_i128(&self, x: i128) -> U256 {
        if let Some(q) = &self.small {
            return U256::from_u64(x.rem_euclid(i128::from(q.value())) as u64);
        }
        let magnitude = U256::from_u128(x.unsigned_abs());
        let r = self.reduce(&magnitude);
        if x < 0 { self.neg(&r) } else { r }
    }
    /// $`x\bmod q`$ for any 256-bit value.
    pub(crate) fn reduce(&self, x: &U256) -> U256 {
        if x < &self.q {
            *x
        } else {
            x.rem_vartime(&self.nonzero)
        }
    }
    /// The canonical representative of a signed value.
    pub(crate) fn reduce_i256(&self, x: &I256) -> U256 {
        let (magnitude, negative) = x.abs_sign();
        let r = self.reduce(&magnitude);
        if bool::from(negative) {
            self.neg(&r)
        } else {
            r
        }
    }
    // Canonical inputs. Below 2^128 native arithmetic suffices; the sum can carry out of u128.
    pub(crate) fn add(&self, a: &U256, b: &U256) -> U256 {
        match self.narrow {
            Some(q) => {
                let (s, carry) = int::low_u128(a).overflowing_add(int::low_u128(b));
                U256::from_u128(if carry || s >= q {
                    s.wrapping_sub(q)
                } else {
                    s
                })
            }
            None => a.add_mod(b, &self.nonzero),
        }
    }
    pub(crate) fn sub(&self, a: &U256, b: &U256) -> U256 {
        match self.narrow {
            Some(q) => {
                let (d, borrow) = int::low_u128(a).overflowing_sub(int::low_u128(b));
                U256::from_u128(if borrow { d.wrapping_add(q) } else { d })
            }
            None => a.sub_mod(b, &self.nonzero),
        }
    }
    pub(crate) fn neg(&self, a: &U256) -> U256 {
        match self.narrow {
            Some(q) => {
                let a = int::low_u128(a);
                U256::from_u128(if a == 0 { 0 } else { q - a })
            }
            None => a.neg_mod(&self.nonzero),
        }
    }
    pub(crate) fn mul(&self, a: &U256, b: &U256) -> U256 {
        // Below 2^64 the native remainder is faster than a one-limb reciprocal.
        if let Some(q) = &self.small {
            let product = u128::from(int::low_u64(a)) * u128::from(int::low_u64(b));
            return U256::from_u64((product % u128::from(q.value())) as u64);
        }
        a.mul_mod_vartime(b, &self.nonzero)
    }
    /// A Shoup multiplier for a reduced scalar, when $`q<2^{63}`$.
    pub(crate) fn scalar_multiplier(&self, scalar: &U256) -> Option<ScalarMul> {
        let q = self.small_crt()?.modulus();
        let w = int::low_u64(scalar);
        debug_assert!(w < q && int::to_u64(scalar) == Some(w));
        Some(ScalarMul {
            w,
            w_shoup: shoup(w, q),
            q,
        })
    }
    /// Whether a canonical value is negative when centred in $`(-q/2,q/2]`$.
    pub(crate) fn is_negative(&self, x: &U256) -> bool {
        x > &self.half
    }
    /// Absolute value of the centred representative.
    pub(crate) fn magnitude(&self, x: &U256) -> U256 {
        if self.is_negative(x) {
            self.q.wrapping_sub(x)
        } else {
            *x
        }
    }
    /// The centred representative, if it fits `i128`.
    pub(crate) fn centred_i128(&self, x: &U256) -> Option<i128> {
        let magnitude = int::to_u128(&self.magnitude(x))?;
        if self.is_negative(x) {
            0i128.checked_sub_unsigned(magnitude)
        } else {
            i128::try_from(magnitude).ok()
        }
    }
    /// The centred representative. Every one fits, since $`q/2<2^{255}`$.
    pub(crate) fn centred_i256(&self, x: &U256) -> I256 {
        let magnitude = *self.magnitude(x).as_int();
        if self.is_negative(x) {
            magnitude.wrapping_neg()
        } else {
            magnitude
        }
    }
    /// $`x\bmod p_i`$ for a canonical value, reading only the words that $`q`$ can occupy.
    pub(crate) fn residue(&self, x: &U256, i: usize) -> u64 {
        let p = &self.divisors[i];
        match self.words {
            // q <= p_i: a canonical value is its own residue. The test reads `small`, since
            // q = 2^64 also takes one word.
            1 if self.small.is_some_and(|q| q.value() <= p.value()) => int::low_u64(x),
            1 => p.rem(&x.resize::<{ U64::LIMBS }>()),
            2 => p.rem(&x.resize::<{ U128::LIMBS }>()),
            3 => p.rem(&x.resize::<{ U192::LIMBS }>()),
            _ => p.rem(x),
        }
    }
}
