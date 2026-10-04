//! Rejection sampling with exact moments, 192-bit fixed-point exponentials and 256-bit coins.
//! Arithmetic is variable time. The exponential's absolute error is bounded conservatively
//! by $`2^{-170}`$ on nonnegative rational inputs (`docs/security.md` derives 198 units of
//! $`2^{-192}`$ below 256, and `bernoulli` 12286 below 16); probabilities below $`e^{-256}`$
//! round to zero.
use crate::Error;
use crypto_bigint::{CheckedAdd, I256, I512, NonZero, U256, U512, U1024};

/// Bytes of a rejection coin: a uniform integer in $`[0,2^{256})`$, little-endian ([`coin`]).
///
/// [`accept`] resolves an acceptance probability to $`2^{-256}`$, below the error of its
/// 192-bit exponentials, so the coin adds at most $`2^{-256}`$ to it. A coin of 128 bits would
/// move every acceptance probability by up to $`2^{-128}`$, and an accepted response's law by up
/// to $`2^{-128}`$ divided by the test's acceptance rate.
pub const COIN_BYTES: usize = 32;

/// The coin that `bytes` encode: the integer they give little-endian, uniform in
/// $`[0,2^{256})`$ when the bytes are uniform.
pub fn coin(bytes: &[u8; COIN_BYTES]) -> U256 {
    U256::from_le_slice(bytes)
}

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
/// and a uniform 256-bit coin $`u`$ ([`coin`]). Rejects constants outside $`1\le M<2^{16}`$.
///
/// Let $`p`$ be the policy's acceptance probability, with its exponentials computed by the
/// same 192-bit fixed-point algorithm as [`exp_negative`], on moments scaled to 512 bits. The
/// test accepts exactly when $`u\le2^{256}p`$, comparing exact integer products (below
/// $`2^{785}`$), so with probability
/// $`\min(1,(\lfloor2^{256}p\rfloor+1)/2^{256})`$, within $`2^{-256}`$ of $`\min(1,p)`$: the
/// exponentials' error is the only other approximation.
///
/// The bimodal comparison is rearranged into nonpositive exponentials, avoiding cosh
/// overflow.
pub fn accept(
    policy: Policy,
    dot: I256,
    norm: U256,
    variance: Variance,
    m_scaled: U256,
    u: U256,
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
    // u·M·2^128 < 2^400. The exponentials are at most 2^192 and the bimodal denominator at most
    // 2^193, so no product below reaches 2^785.
    let lhs = m_scaled.resize::<{ U1024::LIMBS }>().wrapping_mul(&u);
    let d = variance.shl_vartime(1);
    let wide = |x: U512| x.resize::<{ U1024::LIMBS }>();
    if matches!(policy, Policy::Standard | Policy::Rej2) {
        // p = exp(numerator/d)/M, with numerator = norm - 2dot.
        let numerator = norm
            .as_int()
            .wrapping_sub(&dot.wrapping_mul(&I512::from(2i64)));
        let exp = wide(exp_impl(numerator.abs(), d));
        return Ok(if bool::from(numerator.is_negative()) {
            // p = (exp/2^192)/M: u/2^256 <= p iff u·M·2^128 <= exp·2^192.
            lhs <= exp.shl_vartime(192)
        } else {
            // p = 2^192/(exp·M): u/2^256 <= p iff u·M·2^128·exp <= 2^576.
            lhs.wrapping_mul(&exp) <= U1024::ONE.shl_vartime(576)
        });
    }
    // p = 2exp(norm/(2s²))/(M(exp(dot/s²)+exp(-dot/s²))). Factor out exp(|dot|/s²):
    // p = 2exp((norm - 2|dot|)/(2s²))/(M(1 + exp(-2|dot|/s²))).
    let absdot = dot.abs();
    let numerator = norm.as_int().wrapping_sub(absdot.shl_vartime(1).as_int());
    let exp = wide(exp_impl(numerator.abs(), d));
    // 2^192(1 + exp(-2|dot|/s²)), at most 2^193.
    let denominator = wide(U512::ONE.shl_vartime(192))
        .wrapping_add(&wide(exp_impl(absdot.shl_vartime(1), variance)));
    let lhs = lhs.wrapping_mul(&denominator);
    Ok(if bool::from(numerator.is_negative()) {
        // p = 2exp/(M·denominator): u/2^256 <= p iff u·M·2^128·denominator <= exp·2^385.
        lhs <= exp.shl_vartime(385)
    } else {
        // p = 2^385/(exp·M·denominator): u/2^256 <= p iff u·M·2^128·denominator·exp <= 2^769.
        lhs.wrapping_mul(&exp) <= U1024::ONE.shl_vartime(769)
    })
}
