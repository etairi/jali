//! Rejection sampling with exact moments and 192-bit fixed-point exponentials.
//! Arithmetic is variable time. The exponential's absolute error is bounded conservatively
//! by $`2^{-170}`$ on nonnegative rational inputs; probabilities below $`e^{-256}`$ round to
//! zero.
use crate::Error;
use crypto_bigint::{CheckedAdd, I256, I512, NonZero, U256, U512};

/// Rejection policies of LNP22 Fig. 1–2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// Standard rejection, Rej_1.
    Standard,
    /// Rej_2: reject negative inner products before the standard test.
    Rej2,
    /// Bimodal acceptance. The caller MUST randomly flip the displacement first.
    Bimodal,
}

/// A Gaussian variance $`\sigma^2`$ as the exact fraction `numerator / denominator`.
///
/// Rejection sampling reproduces the target Gaussian only if it uses the variance the mask was
/// sampled with. The sampler's variance is fractional, so rounding it to an integer distorts the
/// distribution of accepted responses in a way that depends on the secret.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Variance {
    /// Numerator, nonzero.
    pub numerator: U256,
    /// Denominator, nonzero.
    pub denominator: u64,
}

impl Variance {
    /// The variance $`\sigma^2=961\cdot4^t/400`$ of the width $`\sigma=1.55\cdot2^t`$ that
    /// [`gaussian`](super::gaussian) samples at. It is never an integer. The sampler's fast
    /// Bernoulli decisions (`bernoulli`) are built for exactly this fraction; a test ties them.
    pub fn gaussian(t: u32) -> Result<Self, Error> {
        if t > super::MAX_LOG_SIGMA {
            return Err(Error::Parameter("Gaussian dimensions"));
        }
        Ok(Self {
            numerator: U256::from(961u16).shl_vartime(2 * t),
            denominator: 400,
        })
    }
}

/// Exact dot product and displacement squared norm, with checked accumulation.
pub fn moments(z: &[i128], v: &[i128]) -> Result<(I256, U256), Error> {
    if z.len() != v.len() {
        return Err(Error::Dimension);
    }
    let mut dot = I256::ZERO;
    let mut norm = U256::ZERO;
    for (z, v) in z.iter().zip(v) {
        let product: I256 =
            Option::from(I256::from(*z).checked_mul(&I256::from(*v))).ok_or(Error::Overflow)?;
        dot = Option::from(dot.checked_add(&product)).ok_or(Error::Overflow)?;
        let v = U256::from(v.unsigned_abs());
        norm = Option::from(norm.checked_add(&v.wrapping_mul(&v))).ok_or(Error::Overflow)?;
    }
    Ok((dot, norm))
}

fn div(a: U512, b: U512) -> U512 {
    a.div_rem_vartime(&NonZero::new(b).expect("positive denominator"))
        .0
}

/// Compute $`\lfloor2^{192}e^{-n/d}\rfloor`$ up to at most $`2^{22}`$ rounding units.
/// Inputs are bounded to 256 bits, while all intermediate products use 512 bits.
pub fn exp_negative(numerator: U256, denominator: U256) -> Result<U256, Error> {
    if denominator == U256::ZERO {
        return Err(Error::Parameter("exponential denominator"));
    }
    Ok(exp_impl(numerator.resize(), denominator.resize()).resize())
}

fn exp_impl(n: U512, d: U512) -> U512 {
    let scale = U512::ONE.shl_vartime(192);
    if n >= d.shl_vartime(8) {
        return U512::ZERO;
    }
    let mut x = div(n.shl_vartime(192), d);
    let mut squarings = 0;
    while x > scale.shr_vartime(3) {
        x = x.shr_vartime(1);
        squarings += 1;
    }
    let mut sum = scale;
    let mut term = scale;
    for k in 1..80u64 {
        term = div(term.wrapping_mul(&x).shr_vartime(192), U512::from(k));
        if k % 2 == 1 {
            sum = sum.wrapping_sub(&term);
        } else {
            sum = sum.wrapping_add(&term);
        }
        if term == U512::ZERO {
            break;
        }
    }
    for _ in 0..squarings {
        sum = sum.wrapping_mul(&sum).shr_vartime(192);
    }
    sum
}

/// Decide acceptance from exact moments, the exact variance, $`\mathrm{round}(M2^{128})`$,
/// and a uniform 128-bit integer. Rejects constants outside $`1\le M<2^{16}`$.
///
/// The bimodal comparison is rearranged into nonpositive exponentials, avoiding cosh
/// overflow and preserving the full 128-bit uniform comparison.
pub fn accept(
    policy: Policy,
    dot: I256,
    norm: U256,
    variance: Variance,
    m_scaled: U256,
    u: u128,
) -> Result<bool, Error> {
    if variance.numerator == U256::ZERO
        || variance.denominator == 0
        || m_scaled < U256::ONE.shl_vartime(128)
        || m_scaled >= U256::ONE.shl_vartime(144)
    {
        return Err(Error::Parameter("rejection variance or M"));
    }
    if policy == Policy::Rej2 && bool::from(dot.is_negative()) {
        return Ok(false);
    }
    // With s² = n/d every exponent x/s² equals (x·d)/n, so scale the moments by d and use n as
    // the variance. The scaled moments stay below 2^320 and the exponent numerators below 2^321
    // in absolute value; exp_impl returns zero before shifting any input of at least 2^8 times
    // its denominator.
    let den = U512::from(variance.denominator);
    let norm = norm.resize::<{ U512::LIMBS }>().wrapping_mul(&den);
    let dot = dot.resize::<{ U512::LIMBS }>().wrapping_mul(den.as_int());
    let variance = variance.numerator.resize::<{ U512::LIMBS }>();
    let lhs = m_scaled
        .resize::<{ U512::LIMBS }>()
        .wrapping_mul(&U512::from(u));
    let d = variance.shl_vartime(1);
    let scale = U512::ONE.shl_vartime(192);
    let boundary = U512::ONE.shl_vartime(448);
    if matches!(policy, Policy::Standard | Policy::Rej2) {
        let numerator = norm
            .as_int()
            .wrapping_sub(&dot.wrapping_mul(&I512::from(2i64)));
        let exp = exp_impl(numerator.abs(), d);
        return Ok(if bool::from(numerator.is_negative()) {
            lhs.shl_vartime(192) <= exp.shl_vartime(256)
        } else {
            lhs.wrapping_mul(&exp) <= boundary
        });
    }
    // 2^257 exp(norm/(2s²))/(exp(dot/s²)+exp(-dot/s²)). Factor out exp(|dot|/s²).
    let absdot = dot.abs();
    let numerator = norm.as_int().wrapping_sub(absdot.shl_vartime(1).as_int());
    let exp = exp_impl(numerator.abs(), d);
    let denominator = scale.wrapping_add(&exp_impl(absdot.shl_vartime(1), variance));
    // First round lhs*(1+exp(-2|dot|/s²)) to the scale used by the final comparison.
    let adjusted = lhs.wrapping_mul(&denominator).shr_vartime(192);
    Ok(if bool::from(numerator.is_negative()) {
        adjusted.shl_vartime(192) <= exp.shl_vartime(257)
    } else {
        adjusted.wrapping_mul(&exp) <= boundary.shl_vartime(1)
    })
}
