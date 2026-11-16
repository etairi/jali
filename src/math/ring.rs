use super::{
    int::{self, Divisor64},
    mod128::{Mod128, add_mod128, shoup_mul128, sub_mod128},
    ntt::{NttPlan, add_mod, pow_mod, shoup, shoup_mul, sub_mod},
};
use crate::{
    Error,
    params::moduli::{NADDS, NTT_PRIMES},
};
#[cfg(test)]
use crypto_bigint::Uint;
use crypto_bigint::{I256, NonZero, U64, U192, U256, U384, U576};
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
    /// The same constants in words when $`2^{63}\le q<2^{126}`$.
    native: Option<NativeTail>,
}
/// The sum $`\sum_jv_j(M_j\bmod q)`$ modulo $`2^{63}\le q<2^{126}`$ by Shoup products, one per
/// digit: the canonical value that the six-limb sum reduced once modulo $`q`$ gives.
#[derive(Debug)]
struct NativeTail {
    q: u128,
    /// $`M_j\bmod q`$ with Shoup constants.
    prefixes_q: Vec<(u128, u128)>,
    /// $`P\bmod q`$.
    product_q: u128,
}
impl WideCrt {
    /// The constants for a modulus $`q`$ and at most nine distinct primes $`p_i<2^{62}`$, with
    /// the native tail when `wide` is $`q`$'s context.
    pub(crate) fn new(q: &NonZero<U256>, primes: &[u64], wide: Option<&Mod128>) -> Self {
        let (garner, prefixes, product) = Garner::new(primes);
        let prefixes_q: Vec<U256> = prefixes.iter().map(|m| m.rem_vartime(q)).collect();
        let product_q = product.rem_vartime(q);
        let native = wide.map(|m| NativeTail {
            q: m.modulus(),
            prefixes_q: prefixes_q
                .iter()
                .map(|x| {
                    let v = int::low_u128(x);
                    (v, m.shoup(v))
                })
                .collect(),
            product_q: int::low_u128(&product_q),
        });
        Self {
            garner,
            prefixes_q,
            product_q,
            native,
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
        if let Some(tail) = &self.native {
            let q = tail.q;
            let mut x = 0u128;
            for (v, (m, m_shoup)) in digits.iter().zip(&tail.prefixes_q) {
                x = add_mod128(x, shoup_mul128(u128::from(*v), *m, *m_shoup, q), q);
            }
            return U256::from_u128(if negative {
                sub_mod128(x, tail.product_q, q)
            } else {
                x
            });
        }
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

/// Multiplication by one reduced scalar $`w`$, by Shoup products: the quotient constant is
/// computed once (see `ntt::shoup_mul` and `mod128::shoup_mul128`).
pub(crate) enum ScalarMul {
    /// $`q<2^{63}`$, with $`\lfloor w2^{64}/q\rfloor`$.
    Small { w: u64, w_shoup: u64, q: u64 },
    /// $`2^{63}\le q<2^{126}`$, with $`\lfloor w2^{128}/q\rfloor`$.
    Wide { w: u128, w_shoup: u128, q: u128 },
}
// Each method matches the variant once, outside its loop: a match per value costs about a
// quarter more instructions in a small-modulus scaling (measured 2026-11-16).
impl ScalarMul {
    /// `f(o, xw mod q)` for each pair of `out` and canonical `xs`.
    #[inline(always)]
    pub(crate) fn zip_apply(
        &self,
        out: &mut [U256],
        xs: &[U256],
        mut f: impl FnMut(&mut U256, U256),
    ) {
        match *self {
            ScalarMul::Small { w, w_shoup, q } => {
                for (o, x) in out.iter_mut().zip(xs) {
                    f(o, U256::from_u64(shoup_mul(int::low_u64(x), w, w_shoup, q)));
                }
            }
            ScalarMul::Wide { w, w_shoup, q } => {
                for (o, x) in out.iter_mut().zip(xs) {
                    f(
                        o,
                        U256::from_u128(shoup_mul128(int::low_u128(x), w, w_shoup, q)),
                    );
                }
            }
        }
    }
    /// $`x\gets xw\bmod q`$ for each canonical value.
    #[inline(always)]
    pub(crate) fn apply_in_place(&self, xs: &mut [U256]) {
        match *self {
            ScalarMul::Small { w, w_shoup, q } => {
                for x in xs {
                    *x = U256::from_u64(shoup_mul(int::low_u64(x), w, w_shoup, q));
                }
            }
            ScalarMul::Wide { w, w_shoup, q } => {
                for x in xs {
                    *x = U256::from_u128(shoup_mul128(int::low_u128(x), w, w_shoup, q));
                }
            }
        }
    }
    /// The products $`xw\bmod q`$ of canonical values, in one pass.
    #[inline(always)]
    pub(crate) fn apply_all(&self, xs: &[U256]) -> Vec<U256> {
        match *self {
            ScalarMul::Small { w, w_shoup, q } => xs
                .iter()
                .map(|x| U256::from_u64(shoup_mul(int::low_u64(x), w, w_shoup, q)))
                .collect(),
            ScalarMul::Wide { w, w_shoup, q } => xs
                .iter()
                .map(|x| U256::from_u128(shoup_mul128(int::low_u128(x), w, w_shoup, q)))
                .collect(),
        }
    }
}

/// Remainders of two-word values modulo an NTT prime $`p<2^{62}`$, by two Shoup products:
/// $`x_12^{64}+x_0\equiv x_1(2^{64}\bmod p)+x_0`$, where `shoup_mul` reduces each term from any
/// word.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WordReducer {
    p: u64,
    /// $`2^{64}\bmod p`$ and its Shoup constant.
    c: u64,
    c_shoup: u64,
    /// The Shoup constant $`\lfloor2^{64}/p\rfloor`$ of the multiplier one.
    one_shoup: u64,
}
impl WordReducer {
    fn new(p: u64) -> Self {
        let c = ((1u128 << 64) % u128::from(p)) as u64;
        Self {
            p,
            c,
            c_shoup: shoup(c, p),
            one_shoup: shoup(1, p),
        }
    }
    /// $`x\bmod p`$ for any $`x<2^{128}`$.
    #[inline(always)]
    pub(crate) fn reduce(&self, x: u128) -> u64 {
        let high = shoup_mul((x >> 64) as u64, self.c, self.c_shoup, self.p);
        let low = shoup_mul(x as u64, 1, self.one_shoup, self.p);
        add_mod(high, low, self.p)
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
    /// The modulus when $`q<2^{128}`$, for additions and sign tests in native arithmetic.
    pub(crate) narrow: Option<u128>,
    /// Native products, multipliers and CRT when $`2^{63}\le q<2^{126}`$.
    pub(crate) wide: Option<Mod128>,
    /// Native remainders of two-word values modulo each selected prime.
    pub(crate) reducers: Vec<WordReducer>,
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
        let wide = Mod128::new(&q);
        let crt = match int::to_u64(&q).filter(|q| *q < 1 << 63) {
            Some(q) => RingCrt::Small(SmallCrt::new(q, &primes)),
            None => RingCrt::Wide(WideCrt::new(&nonzero, &primes, wide.as_ref())),
        };
        Ok(Arc::new(Self {
            q,
            nonzero,
            half: q.shr_vartime(1),
            small: int::to_u64(&q).map(Divisor64::new),
            narrow: int::to_u128(&q),
            wide,
            reducers: primes.iter().map(|p| WordReducer::new(*p)).collect(),
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
    #[inline]
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
    #[inline]
    pub(crate) fn reduce(&self, x: &U256) -> U256 {
        if x.cmp_vartime(&self.q).is_lt() {
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
    #[inline]
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
    #[inline]
    pub(crate) fn sub(&self, a: &U256, b: &U256) -> U256 {
        match self.narrow {
            Some(q) => {
                let (d, borrow) = int::low_u128(a).overflowing_sub(int::low_u128(b));
                U256::from_u128(if borrow { d.wrapping_add(q) } else { d })
            }
            None => a.sub_mod(b, &self.nonzero),
        }
    }
    #[inline]
    pub(crate) fn neg(&self, a: &U256) -> U256 {
        match self.narrow {
            Some(q) => {
                let a = int::low_u128(a);
                U256::from_u128(if a == 0 { 0 } else { q - a })
            }
            None => a.neg_mod(&self.nonzero),
        }
    }
    #[inline]
    pub(crate) fn mul(&self, a: &U256, b: &U256) -> U256 {
        // Below 2^64 the native remainder is faster than a one-limb reciprocal.
        if let Some(q) = &self.small {
            let product = u128::from(int::low_u64(a)) * u128::from(int::low_u64(b));
            return U256::from_u64((product % u128::from(q.value())) as u64);
        }
        if let Some(m) = &self.wide {
            return U256::from_u128(m.mul(int::low_u128(a), int::low_u128(b)));
        }
        a.mul_mod_vartime(b, &self.nonzero)
    }
    /// A Shoup multiplier for a reduced scalar, when $`q<2^{126}`$.
    pub(crate) fn scalar_multiplier(&self, scalar: &U256) -> Option<ScalarMul> {
        if let Some(small) = self.small_crt() {
            let q = small.modulus();
            let w = int::low_u64(scalar);
            debug_assert!(w < q && int::to_u64(scalar) == Some(w));
            return Some(ScalarMul::Small {
                w,
                w_shoup: shoup(w, q),
                q,
            });
        }
        let m = self.wide.as_ref()?;
        let w = int::low_u128(scalar);
        debug_assert!(w < m.modulus() && int::to_u128(scalar) == Some(w));
        Some(ScalarMul::Wide {
            w,
            w_shoup: m.shoup(w),
            q: m.modulus(),
        })
    }
    /// Whether a canonical value is negative when centred in $`(-q/2,q/2]`$.
    #[inline]
    pub(crate) fn is_negative(&self, x: &U256) -> bool {
        match self.narrow {
            Some(q) => int::low_u128(x) > q >> 1,
            None => x.cmp_vartime(&self.half).is_gt(),
        }
    }
    /// Absolute value of the centred representative.
    #[inline]
    pub(crate) fn magnitude(&self, x: &U256) -> U256 {
        if let Some(q) = self.narrow {
            let v = int::low_u128(x);
            return if v > q >> 1 {
                U256::from_u128(q - v)
            } else {
                *x
            };
        }
        if self.is_negative(x) {
            self.q.wrapping_sub(x)
        } else {
            *x
        }
    }
    /// The centred representative, if it fits `i128`.
    #[inline]
    pub(crate) fn centred_i128(&self, x: &U256) -> Option<i128> {
        if let Some(q) = self.narrow {
            // Both magnitudes are at most q/2 < 2^127, so both conversions succeed.
            let v = int::low_u128(x);
            return if v > q >> 1 {
                0i128.checked_sub_unsigned(q - v)
            } else {
                i128::try_from(v).ok()
            };
        }
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
    #[inline]
    pub(crate) fn residue(&self, x: &U256, i: usize) -> u64 {
        let p = &self.divisors[i];
        match self.words {
            // q <= p_i: a canonical value is its own residue. The test reads `small`, since
            // q = 2^64 also takes one word.
            1 if self.small.is_some_and(|q| q.value() <= p.value()) => int::low_u64(x),
            1 | 2 => self.reducers[i].reduce(int::low_u128(x)),
            3 => p.rem(&x.resize::<{ U192::LIMBS }>()),
            _ => p.rem(x),
        }
    }
}

#[cfg(test)]
mod wide_tests;
