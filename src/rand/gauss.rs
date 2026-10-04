use super::{
    ByteStream, bernoulli,
    cdf::{HALF_1_55, HALF_3_1},
    reject::Variance,
};
use crate::Error;
use crypto_bigint::U256;
use zeroize::{Zeroize, Zeroizing};

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
    /// The 32 bytes of the current table value.
    value: [u8; 32],
    /// The 24 bytes of the current uniform value of the Bernoulli test; bytes 24 to 31 stay 0.
    uniform: [u8; 32],
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
    /// One sample $`K`$ of a half-Gaussian from its tail table: $`K=\#\{j:V<T_j\}`$ for a
    /// uniform 256-bit $`V`$, 32 bytes little-endian. The entries decrease (a test checks), so
    /// the count stops at the first entry not above $`V`$ and equals the count over all entries
    /// that the reference sampler of the tests takes. The early stop makes the time depend on
    /// $`K`$. For $`t\ge1`$, as in every shipped parameter set, the Bernoulli test's work
    /// depends on $`K`$ too, so counting every entry would not make the time independent of the
    /// sample: the sampler runs in variable time, as `docs/security.md` states.
    fn draw(&mut self, table: &[U256]) -> Result<i128, Error> {
        self.stream.fill(&mut self.value)?;
        let value = U256::from_le_slice(&self.value);
        Ok(table.iter().take_while(|entry| value < **entry).count() as i128)
    }
    /// $`t=0`$: $`k`$ with $`\Pr[k]\propto\rho_{1.55}(k)`$, from a random sign and the
    /// half-Gaussian of width 1.55. A draw of sign 1 and $`K=0`$ is drawn again, so that every
    /// $`k`$ is reached by exactly one pair of sign and $`K`$.
    fn narrow(&mut self) -> Result<i128, Error> {
        for _ in 0..4096 {
            let b = self.bit()?;
            let k = self.draw(&HALF_1_55)?;
            if b == 0 {
                return Ok(-k);
            }
            if k != 0 {
                return Ok(k);
            }
        }
        Err(Error::Randomness)
    }
    /// $`t\ge1`$: $`k`$ with $`\Pr[k]\propto\rho_{3.1}(k-u/2^{t-1})`$, from the bimodal
    /// proposal on the half-Gaussian of width 3.1 and the 192-bit Bernoulli test (`bernoulli`).
    fn base(&mut self, u: u128, t: u32) -> Result<i128, Error> {
        // t = 0 goes to `narrow`; here the scale 2^(t-1) is an integer.
        debug_assert!(t >= 1, "base takes t >= 1");
        // Width 3.1 at scale 2^(t-1) is sigma = 1.55 2^t: the same exact variance that
        // rejection sampling must use for these masks, and the exponent t, not t - 1, for the
        // Bernoulli test.
        let variance = Variance::gaussian(t)?;
        let scale = 1i128 << (t - 1);
        for _ in 0..4096 {
            let b = self.bit()?;
            let k = self.draw(&HALF_3_1)?;
            let k = if b == 1 { k + 1 } else { -k };
            let a = (k * scale - u as i128).unsigned_abs();
            let v = ((k - b) * scale).unsigned_abs();
            self.stream.fill(&mut self.uniform[..24])?;
            // Bytes 16 to 23 are the top word of the 192-bit uniform value. The decision from
            // the cutoff's enclosure equals the exact test's (`bernoulli`).
            let high = u64::from_le_bytes(self.uniform[16..24].try_into().expect("eight bytes"));
            let accept = match bernoulli::decide(a, v, t, high) {
                Some(accept) => accept,
                None => {
                    #[cfg(test)]
                    EXACT_TESTS.with(|n| n.set(n.get() + 1));
                    U256::from_le_slice(&self.uniform) < exact_cutoff(a, v, variance)?
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
        self.signs.zeroize();
        self.value.zeroize();
        self.uniform.zeroize();
    }
}

/// Sample coefficients at width $`\sigma=1.55\cdot2^t`$ from one continuous stream.
///
/// For $`t\ge1`$ a coefficient is $`2^{t-1}k-u`$ with a uniform offset $`u\in[0,2^{t-1})`$.
/// The candidate $`k`$ comes from a random sign and the 256-bit table of the half-Gaussian of
/// width 3.1 (`cdf`), and a 192-bit fixed-point Bernoulli test accepts it with the probability
/// that makes $`k`$ follow $`\rho_{3.1}(k-u/2^{t-1})`$; width 3.1 at scale $`2^{t-1}`$ is
/// $`\sigma`$. For $`t=0`$ a coefficient is $`\pm K`$, from a random sign and the 256-bit
/// table of the half-Gaussian of width 1.55, with no offset and no Bernoulli test. At every
/// $`t`$ the variance is exactly $`961\cdot4^t/400`$.
///
/// Distance from $`D_{\mathbb Z,\sigma}`$, per coefficient (computed; see `docs/security.md`):
/// - the normalizers $`\rho_{3.1}(\mathbb Z-c)`$ differ from their mean by a relative
///   $`2^{-272.6}`$ at most, and not at all for $`t\le1`$, where every offset is 0;
/// - a table sample is within statistical distance $`2^{-252.4}`$ (width 3.1) or
///   $`2^{-253.6}`$ (width 1.55) of its half-Gaussian;
/// - the Bernoulli cutoff is within 12286 of $`2^{192}e^{-x}`$ (`bernoulli`): a statistical
///   distance of $`2^{-178.2}`$ at most per coefficient, but a relative error of the acceptance
///   probabilities, of $`\chi^2`$ divergence $`2^{-356.2}`$ at most.
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
    // Offset bits per coefficient: t - 1, and none at t = 0. At most 99 bits for at most 2^24
    // coefficients, so the bit count stays below 2^31.
    let offset = log_sigma.saturating_sub(1);
    let mut bytes = Zeroizing::new(vec![0u8; (count * offset as usize).div_ceil(8)]);
    stream.fill(&mut bytes)?;
    let mut sampler = Sampler {
        stream,
        signs: 0,
        remaining: 0,
        value: [0; 32],
        uniform: [0; 32],
    };
    let mut out = Zeroizing::new(Vec::with_capacity(count));
    for i in 0..count {
        if log_sigma == 0 {
            out.push(sampler.narrow()?);
            continue;
        }
        let u = super::uniform::bits(&bytes, i * offset as usize, offset);
        let k = sampler.base(u, log_sigma)?;
        out.push(
            k.checked_mul(1i128 << offset)
                .and_then(|x| x.checked_sub(u as i128))
                .ok_or(Error::Overflow)?,
        );
    }
    Ok(std::mem::take(&mut *out))
}
