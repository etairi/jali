use super::ByteStream;
use crate::{
    Error,
    math::{Poly, Ring, U256, int},
};
use std::sync::Arc;
use zeroize::Zeroizing;

#[cfg(test)]
mod oracle_tests;

/// The `count` bits from bit `start` of `buf` on, least significant first, for `count` at
/// most 128: the bytes that hold them (at most 17) are read whole and shifted.
pub(crate) fn bits(buf: &[u8], start: usize, count: u32) -> u128 {
    if count == 0 {
        return 0;
    }
    let shift = (start % 8) as u32;
    let bytes = &buf[start / 8..(start + count as usize).div_ceil(8)];
    let mut value = 0u128;
    for (i, byte) in bytes.iter().take(16).enumerate() {
        value |= u128::from(*byte) << (8 * i);
    }
    value >>= shift;
    if let Some(byte) = bytes.get(16) {
        // Only when count + shift > 128, so shift > 0.
        value |= u128::from(*byte) << (128 - shift);
    }
    if count == 128 {
        value
    } else {
        value & ((1 << count) - 1)
    }
}

/// Uniform values in $`[0,m)`$, including moduli of 60 bits and above.
/// Each rejection pass reads a byte-rounded buffer of $`\lceil\log_2 m\rceil`$-bit words, least
/// significant bit first, for the still-missing elements, and keeps the words below $`m`$.
pub fn uniform(
    stream: &mut impl ByteStream,
    modulus: u128,
    count: usize,
) -> Result<Vec<u128>, Error> {
    if modulus == 0 || count > (1 << 24) {
        return Err(Error::Parameter("uniform sampler dimensions"));
    }
    if modulus == 1 {
        return Ok(vec![0; count]);
    }
    let mbits = 128 - (modulus - 1).leading_zeros();
    let mut out = Zeroizing::new(Vec::with_capacity(count));
    for _ in 0..1024 {
        let missing = count - out.len();
        if missing == 0 {
            return Ok(std::mem::take(&mut *out));
        }
        let mut bytes = Zeroizing::new(vec![0u8; (missing * mbits as usize).div_ceil(8)]);
        stream.fill(&mut bytes)?;
        for i in 0..missing {
            let value = bits(&bytes, i * mbits as usize, mbits);
            if value < modulus {
                out.push(value);
            }
        }
    }
    Err(Error::Randomness)
}

/// Uniform values in $`[0,m)`$ for moduli up to $`2^{256}-1`$, by the rule of [`uniform`]: each
/// pass reads a byte-rounded buffer of $`\lceil\log_2 m\rceil`$-bit words, least significant
/// bit first, for the still-missing elements, and keeps the words below $`m`$. The words are
/// uniform and disjoint, so the kept values are uniform and independent; each word is kept
/// with probability above $`1/2`$. For moduli below $`2^{128}`$ the values and the bytes read
/// equal those of [`uniform`].
pub fn uniform_u256(
    stream: &mut impl ByteStream,
    modulus: &U256,
    count: usize,
) -> Result<Vec<U256>, Error> {
    if let Some(m) = int::to_u128(modulus) {
        let values = Zeroizing::new(uniform(stream, m, count)?);
        return Ok(values.iter().map(|x| U256::from_u128(*x)).collect());
    }
    if count > (1 << 24) {
        return Err(Error::Parameter("uniform sampler dimensions"));
    }
    let mbits = int::bits(&modulus.wrapping_sub(&U256::ONE)) as usize;
    let mut out = Zeroizing::new(Vec::with_capacity(count));
    for _ in 0..1024 {
        let missing = count - out.len();
        if missing == 0 {
            return Ok(std::mem::take(&mut *out));
        }
        let mut bytes = Zeroizing::new(vec![0u8; (missing * mbits).div_ceil(8)]);
        stream.fill(&mut bytes)?;
        for i in 0..missing {
            // mbits >= 128 here: the low 128 bits, then the rest.
            let start = i * mbits;
            let mut word = Zeroizing::new([0u8; 32]);
            word[..16].copy_from_slice(&bits(&bytes, start, 128).to_le_bytes());
            word[16..]
                .copy_from_slice(&bits(&bytes, start + 128, mbits as u32 - 128).to_le_bytes());
            let value = U256::from_le_slice(&word[..]);
            if &value < modulus {
                out.push(value);
            }
        }
    }
    Err(Error::Randomness)
}

/// [`uniform_u256`] modulo the ring modulus.
pub(crate) fn uniform_ring(
    stream: &mut impl ByteStream,
    ring: &Ring,
    count: usize,
) -> Result<Vec<U256>, Error> {
    uniform_u256(stream, &ring.modulus(), count)
}

/// Uniform integer samples from an inclusive interval.
pub fn bounded(
    stream: &mut impl ByteStream,
    low: i128,
    high: i128,
    count: usize,
) -> Result<Vec<i128>, Error> {
    let width = high
        .checked_sub(low)
        .and_then(|x| x.checked_add(1))
        .filter(|x| *x > 0)
        .ok_or(Error::Parameter("uniform interval"))?;
    let values = Zeroizing::new(uniform(stream, width as u128, count)?);
    Ok(values.iter().map(|x| *x as i128 + low).collect())
}

/// Binomial samples, using a contiguous block of positive bits followed by negative bits.
pub fn binomial(stream: &mut impl ByteStream, k: usize, count: usize) -> Result<Vec<i128>, Error> {
    if k == 0 || k > 128 || count > (1 << 24) {
        return Err(Error::Parameter("binomial dimensions"));
    }
    let nbits = k
        .checked_mul(count)
        .and_then(|x| x.checked_mul(2))
        .ok_or(Error::Overflow)?;
    let mut bytes = Zeroizing::new(vec![0u8; nbits.div_ceil(8)]);
    stream.fill(&mut bytes)?;
    Ok((0..count)
        .map(|i| {
            i128::from(bits(&bytes, i * k, k as u32).count_ones())
                - i128::from(bits(&bytes, k * count + i * k, k as u32).count_ones())
        })
        .collect())
}

/// Sample a $`\sigma`$-stable challenge; coefficient $`d/2`$ is zero.
/// Unlike the challenge space of LNP22 §2.7, no challenge is rejected for exceeding the
/// operator-norm bound $`\eta`$.
pub fn autostable(
    stream: &mut impl ByteStream,
    ring: Arc<Ring>,
    omega: i128,
) -> Result<Poly, Error> {
    if omega < 1 || U256::from_u128(omega as u128) >= ring.half {
        return Err(Error::Parameter("challenge bound"));
    }
    let first = Zeroizing::new(bounded(stream, -omega, omega, ring.degree() / 2)?);
    let mut coeffs = vec![0; ring.degree()];
    coeffs[..first.len()].copy_from_slice(&first);
    for i in 1..first.len() {
        coeffs[ring.degree() - i] = -first[i];
    }
    Poly::new(ring, coeffs)
}
