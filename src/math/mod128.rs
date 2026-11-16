//! Arithmetic modulo $`2^{63}\le q<2^{126}`$ in native `u128` words: Barrett products of two
//! canonical values and Shoup products by a fixed multiplier.
//!
//! Every function returns the canonical representative in $`[0,q)`$, which is unique, so the
//! results are those of `crypto-bigint`'s multi-limb arithmetic that they replace (the oracle
//! tests compare the two). A product by `crypto-bigint` at these moduli costs a 256-bit product
//! and a division with a reciprocal computed again on every call.
use super::int;
use crypto_bigint::{NonZero, U256};

/// The full product $`ab`$ as its low and high words.
#[inline(always)]
pub(crate) fn mul_wide(a: u128, b: u128) -> (u128, u128) {
    let (a0, a1) = (u128::from(a as u64), a >> 64);
    let (b0, b1) = (u128::from(b as u64), b >> 64);
    let (p00, p01, p10, p11) = (a0 * b0, a0 * b1, a1 * b0, a1 * b1);
    // The middle column, at most 3(2^64-1).
    let mid = (p00 >> 64) + u128::from(p01 as u64) + u128::from(p10 as u64);
    let low = u128::from(p00 as u64) | (mid << 64);
    let high = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    (low, high)
}

/// $`xw\bmod q`$ in $`[0,q)`$ for any $`x<2^{128}`$, given $`w<q<2^{127}`$ and
/// `w_shoup` $`=\lfloor w2^{128}/q\rfloor`$.
///
/// As `ntt::shoup_mul` with $`2^{128}`$ for $`2^{64}`$: write $`w'=(w2^{128}-e)/q`$ with
/// $`0\le e<q`$. Then $`k=\lfloor xw'/2^{128}\rfloor`$ satisfies
/// $`0\le xw/q-k<xe/(q2^{128})+1<2`$, so $`xw-kq\in[0,2q)`$. As $`2q<2^{128}`$, the wrapping
/// computation below is exact, and one conditional subtraction reduces it.
#[inline(always)]
pub(crate) fn shoup_mul128(x: u128, w: u128, w_shoup: u128, q: u128) -> u128 {
    let k = mul_wide(x, w_shoup).1;
    let r = x.wrapping_mul(w).wrapping_sub(k.wrapping_mul(q));
    if r >= q { r - q } else { r }
}

/// $`(a+b)\bmod q`$ for $`a,b<q<2^{127}`$.
#[inline(always)]
pub(crate) fn add_mod128(a: u128, b: u128, q: u128) -> u128 {
    let s = a + b;
    if s >= q { s - q } else { s }
}

/// $`(a-b)\bmod q`$ for $`a,b<q`$.
#[inline(always)]
pub(crate) fn sub_mod128(a: u128, b: u128, q: u128) -> u128 {
    if a >= b { a - b } else { a + (q - b) }
}

/// A modulus $`2^{63}\le q<2^{126}`$ with the constants of its Barrett products and of its
/// Shoup multipliers.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Mod128 {
    q: u128,
    /// The bit length $`s`$ of $`q`$: $`2^{s-1}\le q<2^s`$ and $`64\le s\le126`$.
    s: u32,
    /// $`\mu=\lfloor2^{2s}/q\rfloor`$, below $`2^{s+1}`$ since $`q\ge2^{s-1}`$.
    mu: u128,
    /// $`2^{128}\bmod q`$.
    two128: u128,
    /// $`q^{-1}\bmod2^{128}`$ for an odd modulus.
    inverse: Option<u128>,
}
impl Mod128 {
    /// The context for $`2^{63}\le q<2^{126}`$, `None` for every other modulus.
    pub(crate) fn new(q: &U256) -> Option<Self> {
        let value = int::to_u128(q).filter(|v| (1u128 << 63..1u128 << 126).contains(v))?;
        let s = 128 - value.leading_zeros();
        let divisor = NonZero::new(*q).expect("modulus at least 2^63");
        let mu = int::low_u128(&U256::ONE.shl_vartime(2 * s).div_rem_vartime(&divisor).0);
        let two128 = int::low_u128(&U256::ONE.shl_vartime(128).rem_vartime(&divisor));
        // q^-1 mod 2^128 by Newton's iteration x <- x(2 - qx), which doubles the number of
        // correct low bits: x = 1 is correct modulo 2 for odd q, and 2^7 = 128.
        let inverse = (value & 1 == 1).then(|| {
            let mut x = 1u128;
            for _ in 0..7 {
                x = x.wrapping_mul(2u128.wrapping_sub(value.wrapping_mul(x)));
            }
            debug_assert_eq!(value.wrapping_mul(x), 1);
            x
        });
        Some(Self {
            q: value,
            s,
            mu,
            two128,
            inverse,
        })
    }
    /// The modulus.
    pub(crate) fn modulus(&self) -> u128 {
        self.q
    }
    /// $`x\bmod q`$ for $`x=\mathit{high}\cdot2^{128}+\mathit{low}<2^{2s}`$, by Barrett's
    /// reduction.
    ///
    /// With $`q_1=\lfloor x/2^{s-1}\rfloor`$ and $`\hat q=\lfloor q_1\mu/2^{s+1}\rfloor`$:
    /// first $`q_1\mu/2^{s+1}\le(x/2^{s-1})(2^{2s}/q)/2^{s+1}=x/q`$, so
    /// $`\hat q\le\lfloor x/q\rfloor`$. For $`x\ge2^{s-1}`$, the positive lower bounds
    /// $`q_1>x/2^{s-1}-1`$ and $`\mu>2^{2s}/q-1`$ give
    /// $`q_1\mu/2^{s+1}>x/q-x/2^{2s}-2^{s-1}/q>x/q-2`$, as $`x<2^{2s}`$ and $`q\ge2^{s-1}`$, so
    /// $`\hat q\ge\lfloor x/q\rfloor-2`$. Below $`2^{s-1}`$, $`q_1=\hat q=0`$ and $`x<q`$. Hence
    /// $`x-\hat qq\in[0,3q)`$, which fits a word as $`3q<2^{128}`$, and two conditional
    /// subtractions reduce it. The shifts below take $`q_1<2^{s+1}`$ and $`\hat q<2^{s+1}`$
    /// exactly from the two-word values.
    #[inline(always)]
    pub(crate) fn reduce_wide(&self, low: u128, high: u128) -> u128 {
        let s = self.s;
        let q1 = (low >> (s - 1)) | (high << (129 - s));
        let (plow, phigh) = mul_wide(q1, self.mu);
        let qhat = (plow >> (s + 1)) | (phigh << (127 - s));
        let mut r = low.wrapping_sub(qhat.wrapping_mul(self.q));
        if r >= self.q {
            r -= self.q;
        }
        if r >= self.q {
            r -= self.q;
        }
        r
    }
    /// $`ab\bmod q`$ for $`a,b<q`$.
    #[inline(always)]
    pub(crate) fn mul(&self, a: u128, b: u128) -> u128 {
        let (low, high) = mul_wide(a, b);
        self.reduce_wide(low, high)
    }
    /// The Shoup constant $`w'=\lfloor w2^{128}/q\rfloor`$ of a multiplier $`w<q`$.
    ///
    /// For odd $`q`$ without a division: with $`r=w2^{128}\bmod q`$, $`w'q=w2^{128}-r`$, so
    /// $`w'q\equiv-r\pmod{2^{128}}`$ and $`w'\equiv-rq^{-1}`$. As $`w'<2^{128}`$ (from
    /// $`w<q`$), that congruence gives $`w'`$ itself.
    pub(crate) fn shoup(&self, w: u128) -> u128 {
        debug_assert!(w < self.q);
        match self.inverse {
            Some(inverse) => self
                .mul(w, self.two128)
                .wrapping_neg()
                .wrapping_mul(inverse),
            None => {
                let numerator = U256::from_u128(w).shl_vartime(128);
                let divisor = NonZero::new(U256::from_u128(self.q)).expect("nonzero modulus");
                int::low_u128(&numerator.div_rem_vartime(&divisor).0)
            }
        }
    }
}

#[cfg(test)]
mod oracle_tests;
