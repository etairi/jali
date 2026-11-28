use super::ByteStream;
use crate::wipe::Secret;
use crate::{
    Error,
    math::{Poly, Ring, U256, int},
};
use std::sync::Arc;

#[cfg(test)]
mod oracle_tests;

/// The `count` bits from bit `start` of `buf` on, least significant first, for `count` at
/// most 128: the bytes that hold them (at most 17) are read whole and shifted. Where the buffer
/// holds sixteen bytes from the first one, they are read as one little-endian word, whose bytes
/// past the field the mask clears.
pub(crate) fn bits(buf: &[u8], start: usize, count: u32) -> u128 {
    if count == 0 {
        return 0;
    }
    let shift = (start % 8) as u32;
    let first = start / 8;
    let bytes = &buf[first..(start + count as usize).div_ceil(8)];
    let mut value = match buf.get(first..first + 16) {
        Some(word) => u128::from_le_bytes(word.try_into().expect("sixteen bytes")),
        None => {
            let mut value = 0u128;
            for (i, byte) in bytes.iter().take(16).enumerate() {
                value |= u128::from(*byte) << (8 * i);
            }
            value
        }
    };
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
    let mut out = Secret::new(Vec::with_capacity(count));
    for _ in 0..1024 {
        let missing = count - out.len();
        if missing == 0 {
            return Ok(std::mem::take(&mut *out));
        }
        let mut bytes = Secret::new(vec![0u8; (missing * mbits as usize).div_ceil(8)]);
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
        let values = Secret::new(uniform(stream, m, count)?);
        return Ok(values.iter().map(|x| U256::from_u128(*x)).collect());
    }
    if count > (1 << 24) {
        return Err(Error::Parameter("uniform sampler dimensions"));
    }
    let mbits = int::bits(&modulus.wrapping_sub(&U256::ONE)) as usize;
    let mut out = Secret::new(Vec::with_capacity(count));
    for _ in 0..1024 {
        let missing = count - out.len();
        if missing == 0 {
            return Ok(std::mem::take(&mut *out));
        }
        let mut bytes = Secret::new(vec![0u8; (missing * mbits).div_ceil(8)]);
        stream.fill(&mut bytes)?;
        for i in 0..missing {
            // mbits >= 128 here: the low 128 bits, then the rest.
            let start = i * mbits;
            let mut word = Secret::new([0u8; 32]);
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

/// A polynomial of [`uniform_u256`] coefficients modulo the ring modulus, the ring's degree of
/// them: below $`2^{128}`$ the values of [`uniform`] from the same bytes, kept in the ring's
/// coefficient width without a 256-bit copy.
pub(crate) fn uniform_poly(stream: &mut impl ByteStream, ring: &Arc<Ring>) -> Result<Poly, Error> {
    match int::to_u128(&ring.modulus()) {
        Some(m) => Ok(Poly::from_canonical_u128(
            ring.clone(),
            uniform(stream, m, ring.degree())?,
        )),
        None => Ok(Poly::from_canonical(
            ring.clone(),
            uniform_u256(stream, &ring.modulus(), ring.degree())?,
        )),
    }
}

/// [`uniform_poly`] for public polynomials, such as the entries of public matrices: the same
/// coefficients from the same bytes, read by [`public_uniform`] without wiping. Moduli from
/// $`2^{128}`$ on take [`uniform_poly`].
pub(crate) fn public_poly(stream: &mut impl ByteStream, ring: &Arc<Ring>) -> Result<Poly, Error> {
    match int::to_u128(&ring.modulus()) {
        Some(m) => Ok(Poly::from_canonical_u128(
            ring.clone(),
            public_uniform(stream, m, ring.degree())?,
        )),
        None => uniform_poly(stream, ring),
    }
}

/// [`uniform`] for public values: each pass reads the byte-rounded buffer that [`uniform`]
/// reads for the missing elements and keeps the same words, without wiping. A field starts at
/// most 7 bits into its first byte, so words of at most 57 bits are read as 64-bit words and
/// words of at most 120 bits as 128-bit words, from a buffer padded with 16 zero bytes that
/// the masks clear. Wider words take [`bits`]. Every word is written to the next free place
/// and kept by advancing past it, without a branch on the comparison: a pass decodes as many
/// words as places are free, so the next free place before a word is at most `count - 1`.
pub(crate) fn public_uniform(
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
    let width = mbits as usize;
    let mut out = vec![0u128; count];
    let mut filled = 0;
    let mut bytes = Vec::new();
    for _ in 0..1024 {
        let missing = count - filled;
        if missing == 0 {
            return Ok(out);
        }
        let len = (missing * width).div_ceil(8);
        bytes.clear();
        bytes.resize(len + 16, 0);
        stream.fill(&mut bytes[..len])?;
        let starts = (0..missing).map(|i| i * width);
        let mut keep = |value: u128| {
            out[filled] = value;
            filled += usize::from(value < modulus);
        };
        if mbits <= 57 {
            let mask = (1u64 << mbits) - 1;
            for start in starts {
                let at = start / 8;
                let word = u64::from_le_bytes(bytes[at..at + 8].try_into().expect("eight bytes"));
                keep(u128::from((word >> (start % 8)) & mask));
            }
        } else if mbits <= 120 {
            let mask = (1u128 << mbits) - 1;
            for start in starts {
                let at = start / 8;
                let word =
                    u128::from_le_bytes(bytes[at..at + 16].try_into().expect("sixteen bytes"));
                keep((word >> (start % 8)) & mask);
            }
        } else {
            for start in starts {
                keep(bits(&bytes[..len], start, mbits));
            }
        }
    }
    Err(Error::Randomness)
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
    let values = Secret::new(uniform(stream, width as u128, count)?);
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
    let mut bytes = Secret::new(vec![0u8; nbits.div_ceil(8)]);
    stream.fill(&mut bytes)?;
    Ok((0..count)
        .map(|i| {
            i128::from(bits(&bytes, i * k, k as u32).count_ones())
                - i128::from(bits(&bytes, k * count + i * k, k as u32).count_ones())
        })
        .collect())
}

/// Sample a $`\sigma`$-stable polynomial with coefficients in $`[-\omega,\omega]`$ and
/// coefficient $`d/2`$ zero, uniformly: one draw, which no operator-norm bound filters. The
/// challenges of the proofs come from [`challenge`](super::challenge), which repeats this
/// draw until it meets the bound $`\eta`$ of LNP22 §2.7.
pub fn autostable(
    stream: &mut impl ByteStream,
    ring: Arc<Ring>,
    omega: i128,
) -> Result<Poly, Error> {
    if omega < 1 || U256::from_u128(omega as u128) >= ring.half {
        return Err(Error::Parameter("challenge bound"));
    }
    let first = Secret::new(bounded(stream, -omega, omega, ring.degree() / 2)?);
    let mut coeffs = vec![0; ring.degree()];
    coeffs[..first.len()].copy_from_slice(&first);
    for i in 1..first.len() {
        coeffs[ring.degree() - i] = -first[i];
    }
    Poly::new(ring, coeffs)
}
