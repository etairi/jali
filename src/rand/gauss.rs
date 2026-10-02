use super::{ByteStream, bernoulli, cdf::CDF, reject::Variance};
use crate::Error;
use crypto_bigint::U256;
use zeroize::Zeroizing;

#[cfg(test)]
mod oracle_tests;
#[cfg(test)]
mod table_tests;

#[cfg(test)]
thread_local! {
    /// Test only: candidates of this thread that took the exact test.
    static EXACT_TESTS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

struct Sampler<'a, S> {
    stream: &'a mut S,
    signs: u64,
    remaining: u32,
}
impl<S: ByteStream> Sampler<'_, S> {
    fn word(&mut self) -> Result<u64, Error> {
        let mut b = [0; 8];
        self.stream.fill(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn bit(&mut self) -> Result<i128, Error> {
        if self.remaining == 0 {
            self.signs = self.word()?;
            self.remaining = 64;
        }
        let bit = self.signs & 1;
        self.signs >>= 1;
        self.remaining -= 1;
        Ok(i128::from(bit))
    }
    fn base(&mut self, u: u128, t: u32) -> Result<i128, Error> {
        // The same exact variance that rejection sampling must use for these masks.
        let variance = Variance::gaussian(t)?;
        for _ in 0..4096 {
            let b = self.bit()?;
            // The high 64-bit word first, both words little-endian.
            let value = (u128::from(self.word()?) << 64) | u128::from(self.word()?);
            let k = CDF.iter().take_while(|entry| value < **entry).count() as i128;
            let k = if b == 1 { k + 1 } else { -k };
            let scale = 1i128 << t;
            let a = (k * scale - u as i128).unsigned_abs();
            let v = ((k - b) * scale).unsigned_abs();
            let mut random = [0u8; 32];
            self.stream.fill(&mut random[..24])?;
            // Bytes 16 to 23 are the top word of the 192-bit uniform value. The decision from
            // the cutoff's enclosure equals the exact test's (`bernoulli`).
            let high = u64::from_le_bytes(random[16..24].try_into().expect("eight bytes"));
            let accept = match bernoulli::decide(a, v, t, high) {
                Some(accept) => accept,
                None => {
                    #[cfg(test)]
                    EXACT_TESTS.with(|n| n.set(n.get() + 1));
                    U256::from_le_slice(&random) < exact_cutoff(a, v, variance)?
                }
            };
            if accept {
                return Ok(k);
            }
        }
        Err(Error::Randomness)
    }
}
/// The cutoff $`\lfloor2^{192}e^{-x}\rfloor`$ (up to the error `reject::exp_negative`
/// documents) with $`x=(a^2-v^2)/(2\sigma^2)`$ and $`\sigma^2=n/d`$, computed as
/// $`\exp(-(a^2-v^2)d/(2n))`$.
fn exact_cutoff(a: u128, v: u128, variance: Variance) -> Result<U256, Error> {
    let (a, v) = (U256::from(a), U256::from(v));
    let numerator = a
        .wrapping_mul(&a)
        .wrapping_sub(&v.wrapping_mul(&v))
        .wrapping_mul(&U256::from(variance.denominator));
    let denominator = variance.numerator.shl_vartime(1);
    super::reject::exp_negative(numerator, denominator)
}
impl<S> Drop for Sampler<'_, S> {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.signs.zeroize();
    }
}

/// Sample coefficients at width $`\sigma=1.55\cdot2^t`$ from one continuous stream.
///
/// A coefficient is $`2^tk-u`$ with a uniform offset $`u\in[0,2^t)`$. The candidate $`k`$ comes
/// from a random sign and the 128-bit table of the half-Gaussian of width 1.55 (`cdf`, within
/// statistical distance $`2^{-126.2}`$ of it), and a 192-bit fixed-point Bernoulli test accepts
/// it with the probability that makes $`k`$ follow $`\rho_{1.55}(k-u/2^t)`$. The normalizers
/// $`\rho_{1.55}(\mathbb Z-c)`$ differ from their mean by a relative $`2^{-67.4}`$ at most, so
/// the output probabilities are those of $`D_{\mathbb Z,\sigma}`$ up to a relative error of
/// about $`2^{-67.4}`$ (both figures computed).
///
/// The sign cache belongs to this call; no process-global state or per-coefficient reseeding.
/// Acceptance uses exact rational exponents and integer arithmetic on every platform: nearly
/// every test is decided from the top 64 bits of the uniform value and an integer enclosure of
/// the cutoff, with the same result as the exact comparison (`bernoulli`), and the remaining
/// ones by that comparison. The outputs and the bytes read are those of the exact test.
pub fn gaussian(
    stream: &mut impl ByteStream,
    log_sigma: u32,
    count: usize,
) -> Result<Vec<i128>, Error> {
    if log_sigma > super::MAX_LOG_SIGMA || count > (1 << 24) {
        return Err(Error::Parameter("Gaussian dimensions"));
    }
    let mut bytes = Zeroizing::new(vec![0u8; (count * log_sigma as usize).div_ceil(8)]);
    stream.fill(&mut bytes)?;
    let mut sampler = Sampler {
        stream,
        signs: 0,
        remaining: 0,
    };
    let mut out = Zeroizing::new(Vec::with_capacity(count));
    for i in 0..count {
        let u = super::uniform::bits(&bytes, i * log_sigma as usize, log_sigma);
        let k = sampler.base(u, log_sigma)?;
        out.push(
            k.checked_mul(1i128 << log_sigma)
                .and_then(|x| x.checked_sub(u as i128))
                .ok_or(Error::Overflow)?,
        );
    }
    Ok(std::mem::take(&mut *out))
}
