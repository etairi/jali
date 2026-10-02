//! Dilithium-style compression: LNP22 Fig. 17 and Lemma A.1.
//!
//! Inputs are 256-bit values, reduced modulo $`q`$ first. High parts are nonnegative; low
//! parts and hints are signed, as in the paper.
use crate::{
    Error,
    math::{I256, U256, int},
};
use crypto_bigint::{NonZero, Reciprocal};

/// Validated coefficient compression parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Compression {
    q: NonZero<U256>,
    gamma: NonZero<U256>,
    /// The reciprocal of `gamma` when it fits one limb.
    gamma_limb: Option<Reciprocal>,
    /// $`m=(q-1)/\gamma`$.
    m: U256,
    d_bits: u32,
}

/// The integer with the given magnitude and sign; the magnitude must be below $`2^{255}`$.
fn signed(magnitude: U256, negative: bool) -> I256 {
    let x = *magnitude.as_int();
    if negative { x.wrapping_neg() } else { x }
}

impl Compression {
    /// [`Compression::with_modulus`] for `i128` values; nonpositive values are refused.
    pub fn new(q: i128, gamma: i128, d_bits: u32) -> Result<Self, Error> {
        if q < 3 || gamma < 2 {
            return Err(Error::Parameter("compression"));
        }
        Self::with_modulus(
            U256::from_u128(q as u128),
            U256::from_u128(gamma as u128),
            d_bits,
        )
    }
    /// Require an odd modulus $`q\ge3`$, an even divisor $`\gamma`$ of $`q-1`$, and a rounding
    /// power $`2^D<q`$.
    pub fn with_modulus(q: U256, gamma: U256, d_bits: u32) -> Result<Self, Error> {
        let invalid = Err(Error::Parameter("compression"));
        if q < U256::from_u8(3) || !bool::from(q.is_odd()) {
            return invalid;
        }
        if gamma < U256::from_u8(2) || bool::from(gamma.is_odd()) {
            return invalid;
        }
        let gamma = NonZero::new(gamma).expect("gamma at least two");
        let (m, rest) = q.wrapping_sub(&U256::ONE).div_rem_vartime(&gamma);
        if rest != U256::ZERO || int::power_of_two(d_bits).is_none_or(|p| p >= q) {
            return invalid;
        }
        let gamma_limb = int::to_limb(&gamma)
            .map(|x| Reciprocal::new(NonZero::new(x).expect("gamma at least two")));
        Ok(Self {
            q: NonZero::new(q).expect("q at least three"),
            gamma,
            gamma_limb,
            m,
            d_bits,
        })
    }
    /// The modulus $`q`$.
    pub fn modulus(&self) -> U256 {
        *self.q
    }
    /// $`m=(q-1)/\gamma`$.
    pub fn hint_modulus(&self) -> U256 {
        self.m
    }
    /// Compression divisor.
    pub fn gamma(&self) -> U256 {
        *self.gamma
    }
    /// Number of low bits discarded by Power2Round.
    pub fn d_bits(&self) -> u32 {
        self.d_bits
    }
    fn reduce(&self, r: &U256) -> U256 {
        if r < &*self.q {
            *r
        } else {
            r.rem_vartime(&self.q)
        }
    }
    /// $`(x/\gamma, x\bmod\gamma)`$.
    fn div_rem_gamma(&self, x: &U256) -> (U256, U256) {
        match &self.gamma_limb {
            Some(reciprocal) => {
                let (quotient, rest) = x.div_rem_limb_with_reciprocal(reciprocal);
                (quotient, U256::from_word(rest.0))
            }
            None => x.div_rem_vartime(&self.gamma),
        }
    }
    /// Return `(high, low)` with $`r=2^D high+low \pmod q`$, $`low\in(-2^{D-1},2^{D-1}]`$ and
    /// positive half-ties.
    pub fn power2round(&self, r: &U256) -> (U256, I256) {
        let r = self.reduce(r);
        if self.d_bits == 0 {
            return (r, I256::ZERO);
        }
        let power = U256::ONE.shl_vartime(self.d_bits);
        let low = r.bitand(&power.wrapping_sub(&U256::ONE));
        let high = r.shr_vartime(self.d_bits);
        if low > power.shr_vartime(1) {
            (
                high.wrapping_add(&U256::ONE),
                signed(power.wrapping_sub(&low), true),
            )
        } else {
            (high, signed(low, false))
        }
    }
    /// Decompose with the special wrap at $`q-1`$ from Fig. 17: $`r=\gamma\,high+low`$ with
    /// $`high\in[0,m)`$ and $`low\in(-\gamma/2,\gamma/2]`$, except that $`r-low=q-1`$ gives
    /// $`(0,low-1)`$.
    pub fn decompose(&self, r: &U256) -> (U256, I256) {
        let r = self.reduce(r);
        let (quotient, rest) = self.div_rem_gamma(&r);
        // With low negative, r - low = gamma (quotient + 1) <= q - 1 since gamma divides q - 1.
        let (high, low) = if rest > self.gamma.shr_vartime(1) {
            (
                quotient.wrapping_add(&U256::ONE),
                signed(self.gamma.wrapping_sub(&rest), true),
            )
        } else {
            (quotient, signed(rest, false))
        };
        if high == self.m {
            (U256::ZERO, low.wrapping_sub(&I256::ONE))
        } else {
            (high, low)
        }
    }
    /// Centred difference of high parts, in $`(-m/2,m/2]`$. Inputs are reduced modulo $`q`$.
    pub fn make_hint(&self, z: &U256, r: &U256) -> I256 {
        let r = self.reduce(r);
        let next = self.decompose(&r.add_mod(&self.reduce(z), &self.q)).0;
        let previous = self.decompose(&r).0;
        let difference = if next >= previous {
            next.wrapping_sub(&previous)
        } else {
            next.wrapping_add(&self.m).wrapping_sub(&previous)
        };
        if difference > self.m.shr_vartime(1) {
            signed(self.m.wrapping_sub(&difference), true)
        } else {
            signed(difference, false)
        }
    }
    /// Reconstruct high bits, requiring the canonical hint interval $`(-m/2,m/2]`$.
    pub fn use_hint(&self, hint: &I256, r: &U256) -> Result<U256, Error> {
        let (magnitude, negative) = hint.abs_sign();
        let half = self.m.shr_vartime(1);
        // (-m/2, m/2] holds the integers from floor(m/2) + 1 - m to floor(m/2).
        let fits = if bool::from(negative) {
            magnitude.wrapping_add(&half) < self.m
        } else {
            magnitude <= half
        };
        if !fits {
            return Err(Error::Encoding);
        }
        let high = self.decompose(r).0;
        Ok(if bool::from(negative) {
            if high >= magnitude {
                high.wrapping_sub(&magnitude)
            } else {
                high.wrapping_add(&self.m).wrapping_sub(&magnitude)
            }
        } else {
            let sum = high.wrapping_add(&magnitude);
            if sum >= self.m {
                sum.wrapping_sub(&self.m)
            } else {
                sum
            }
        })
    }
}
