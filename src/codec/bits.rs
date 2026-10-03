use super::segments::{MIN_ZERO_RUN, Segments};
use crate::{Error, math::U256};

#[cfg(test)]
mod writer_tests;

/// The codes that equation encodings are written with. [`BitWriter`] and [`SegmentWriter`]
/// write the same bytes for the same sequence of calls.
pub(crate) trait CodeWriter {
    /// [`BitWriter::unsigned`].
    fn unsigned(&mut self, value: u128, bits: u32) -> Result<(), Error>;
    /// [`BitWriter::uniform`].
    fn uniform(&mut self, value: u128, modulus: u128) -> Result<(), Error>;
    /// [`BitWriter::uniform_u256`].
    fn uniform_u256(&mut self, value: &U256, modulus: &U256) -> Result<(), Error>;
    /// `count` zero bits, as [`BitWriter::zeros`] appends them.
    fn zeros(&mut self, count: u64);
}

/// Append-only LSB-first bit writer.
///
/// Bits collect in a 64-bit word, least significant first, which is appended to the bytes in
/// little-endian order once full; [`BitWriter::finish`] appends the used bytes of the last
/// word. The output is a function of the sequence of written bit fields alone, so it equals
/// that of writing the same fields one bit at a time.
#[derive(Default, Debug)]
pub struct BitWriter {
    bytes: Vec<u8>,
    /// The bits not yet appended, in its low `used` bits; the others are zero.
    word: u64,
    used: u32,
}
impl BitWriter {
    /// Empty writer.
    pub fn new() -> Self {
        Self::default()
    }
    /// Empty writer whose bytes have room for `bytes` bytes.
    fn with_capacity(bytes: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(bytes),
            ..Self::default()
        }
    }
    fn bit(&mut self, value: bool) {
        self.put(u128::from(value), 1);
    }
    /// Append the low `bits` bits of `value`, `bits` at most 128, least significant first.
    #[inline]
    fn put(&mut self, mut value: u128, mut bits: u32) {
        while bits != 0 {
            let take = (64 - self.used).min(bits);
            let field = if take == 64 {
                value as u64
            } else {
                value as u64 & ((1 << take) - 1)
            };
            self.word |= field << self.used;
            self.used += take;
            bits -= take;
            value >>= take;
            if self.used == 64 {
                self.bytes.extend_from_slice(&self.word.to_le_bytes());
                self.word = 0;
                self.used = 0;
            }
        }
    }
    /// Append `count` zero bits: a run of zero codes, such as those of a zero polynomial.
    pub(crate) fn zeros(&mut self, count: u64) {
        let run = self.zero_run(count);
        self.zero_bytes(run);
    }
    /// [`BitWriter::zeros`] up to its whole zero words: fills the current word and appends it
    /// if the run reaches its end, keeps the bits past the last whole word, and returns the
    /// number of zero bytes still to be appended, in whole words, after the bytes so far.
    fn zero_run(&mut self, mut count: u64) -> u64 {
        let free = u64::from(64 - self.used);
        if count < free {
            self.used += count as u32;
            return 0;
        }
        count -= free;
        self.bytes.extend_from_slice(&self.word.to_le_bytes());
        self.word = 0;
        self.used = (count % 64) as u32;
        8 * (count / 64)
    }
    /// Append `count` zero bytes, which [`BitWriter::zero_run`] returned.
    fn zero_bytes(&mut self, count: u64) {
        self.bytes.resize(self.bytes.len() + count as usize, 0);
    }
    /// Write exactly `bits` low bits, rejecting truncation.
    pub fn unsigned(&mut self, value: u128, bits: u32) -> Result<(), Error> {
        if bits > 128 || (bits < 128 && value >> bits != 0) {
            return Err(Error::Encoding);
        }
        self.put(value, bits);
        Ok(())
    }
    /// Encode a coefficient from $`[0,m)`$ using the minimal bit width.
    pub fn uniform(&mut self, value: u128, modulus: u128) -> Result<(), Error> {
        if modulus < 2 || value >= modulus {
            return Err(Error::Encoding);
        }
        self.unsigned(value, 128 - (modulus - 1).leading_zeros())
    }
    /// Write exactly `bits` low bits of a 256-bit value, rejecting truncation.
    pub fn unsigned_u256(&mut self, value: &U256, bits: u32) -> Result<(), Error> {
        if bits > 256 || value.bits_vartime() > bits {
            return Err(Error::Encoding);
        }
        let bytes = value.to_le_bytes();
        let low = u128::from_le_bytes(bytes[..16].try_into().expect("16 bytes"));
        let high = u128::from_le_bytes(bytes[16..32].try_into().expect("16 bytes"));
        self.put(low, bits.min(128));
        self.put(high, bits.saturating_sub(128));
        Ok(())
    }
    /// [`BitWriter::uniform`] for 256-bit moduli: the same code, `bits(m-1)` bits.
    pub fn uniform_u256(&mut self, value: &U256, modulus: &U256) -> Result<(), Error> {
        if modulus < &U256::from_u8(2) || value >= modulus {
            return Err(Error::Encoding);
        }
        self.unsigned_u256(value, modulus.wrapping_sub(&U256::ONE).bits_vartime())
    }
    /// Gaussian code: signed unary quotient and two's-complement remainder.
    /// `max_unary` is an explicit resource limit on the quotient's code length.
    pub fn gaussian(&mut self, z: i128, log_sigma: u32, max_unary: usize) -> Result<(), Error> {
        if log_sigma > crate::rand::MAX_LOG_SIGMA {
            return Err(Error::Parameter("Gaussian exponent"));
        }
        let bits = log_sigma + 1;
        let scale = 1i128 << bits;
        let r = z.rem_euclid(scale);
        let low = if r >= scale / 2 { r - scale } else { r };
        let quotient = z.checked_sub(low).ok_or(Error::Overflow)? / scale;
        let n = if quotient > 0 {
            quotient
                .unsigned_abs()
                .checked_mul(2)
                .and_then(|n| n.checked_sub(1))
        } else {
            quotient.unsigned_abs().checked_mul(2)
        }
        .ok_or(Error::Overflow)?;
        let n = usize::try_from(n).map_err(|_| Error::Encoding)?;
        if n > max_unary {
            return Err(Error::Encoding);
        }
        // n one bits and a zero bit.
        let mut ones = n;
        while ones >= 64 {
            self.put(u128::from(u64::MAX), 64);
            ones -= 64;
        }
        self.put((1 << ones) - 1, ones as u32 + 1);
        self.unsigned(low.rem_euclid(scale) as u128, bits)
    }
    /// Prefix-free Dilithium hint code (LNP22 Table 3).
    pub fn hint(&mut self, value: i128, max_unary: usize) -> Result<(), Error> {
        match value {
            0 => self.put(0b00, 2),
            1 => self.put(0b10, 2),
            -1 => self.put(0b01, 2),
            _ => {
                let n = value
                    .unsigned_abs()
                    .checked_mul(2)
                    .and_then(|n| n.checked_sub(if value > 0 { 4 } else { 3 }))
                    .ok_or(Error::Overflow)?;
                let n = usize::try_from(n).map_err(|_| Error::Encoding)?;
                if n > max_unary {
                    return Err(Error::Encoding);
                }
                // Two one bits, n zero bits and a one bit.
                self.put(0b11, 2);
                self.zeros(n as u64);
                self.bit(true);
            }
        }
        Ok(())
    }
    /// Append the mandatory one bit and zero padding to the byte boundary.
    pub fn finish(mut self) -> Vec<u8> {
        self.bit(true);
        let used = self.used.div_ceil(8) as usize;
        self.bytes
            .extend_from_slice(&self.word.to_le_bytes()[..used]);
        self.bytes
    }
}
impl CodeWriter for BitWriter {
    fn unsigned(&mut self, value: u128, bits: u32) -> Result<(), Error> {
        BitWriter::unsigned(self, value, bits)
    }
    fn uniform(&mut self, value: u128, modulus: u128) -> Result<(), Error> {
        BitWriter::uniform(self, value, modulus)
    }
    fn uniform_u256(&mut self, value: &U256, modulus: &U256) -> Result<(), Error> {
        BitWriter::uniform_u256(self, value, modulus)
    }
    fn zeros(&mut self, count: u64) {
        BitWriter::zeros(self, count)
    }
}

/// Staged bytes at which [`SegmentWriter`] moves them into its segments.
const STAGE: usize = 4096;
/// The most bytes one code adds to the staged bytes: two words for a code of up to 128 bits and
/// four for a 256-bit one; for a zero run that stays staged, one word and fewer than
/// [`MIN_ZERO_RUN`] zero bytes, in whole words; one word for [`BitWriter::finish`]. Fewer than
/// `STAGE` bytes are staged before each code, so the staged bytes never exceed the capacity
/// `STAGE + STAGE_SLACK` they are allocated with, and their buffer is never reallocated.
const STAGE_SLACK: usize = 64;
const _: () = assert!(
    32 <= STAGE_SLACK && MIN_ZERO_RUN.is_multiple_of(8) && MIN_ZERO_RUN as usize <= STAGE_SLACK
);

/// The codes of a [`BitWriter`], written into [`Segments`] as the same bytes.
///
/// An inner [`BitWriter`] packs the codes. Its bytes move into the segments once [`STAGE`] of
/// them are staged, before a zero run that is kept as a count, and at the end, so that its
/// buffer is never reallocated (nor are the chunks of [`Segments`]). Of a zero run
/// ([`CodeWriter::zeros`]), the bits that complete the current word and those past the last
/// whole zero word stay with the codes; the whole zero words are kept as a count if they make
/// at least [`MIN_ZERO_RUN`] bytes, and are appended as bytes otherwise.
#[derive(Debug)]
pub(crate) struct SegmentWriter {
    bits: BitWriter,
    out: Segments,
}
impl SegmentWriter {
    /// Empty writer.
    pub(crate) fn new() -> Self {
        Self {
            bits: BitWriter::with_capacity(STAGE + STAGE_SLACK),
            out: Segments::default(),
        }
    }
    /// Move the staged bytes into the segments if there are at least `threshold` of them.
    fn spill(&mut self, threshold: usize) {
        debug_assert!(self.bits.bytes.capacity() == STAGE + STAGE_SLACK);
        if self.bits.bytes.len() >= threshold {
            self.out.push_literal(&self.bits.bytes);
            self.bits.bytes.clear();
        }
    }
    /// [`BitWriter::finish`], into the segments.
    pub(crate) fn finish(self) -> Segments {
        let mut out = self.out;
        out.push_literal(&self.bits.finish());
        out.shrink_to_fit();
        out
    }
}
impl CodeWriter for SegmentWriter {
    fn unsigned(&mut self, value: u128, bits: u32) -> Result<(), Error> {
        let result = self.bits.unsigned(value, bits);
        self.spill(STAGE);
        result
    }
    fn uniform(&mut self, value: u128, modulus: u128) -> Result<(), Error> {
        let result = self.bits.uniform(value, modulus);
        self.spill(STAGE);
        result
    }
    fn uniform_u256(&mut self, value: &U256, modulus: &U256) -> Result<(), Error> {
        let result = self.bits.uniform_u256(value, modulus);
        self.spill(STAGE);
        result
    }
    fn zeros(&mut self, count: u64) {
        let run = self.bits.zero_run(count);
        if run >= MIN_ZERO_RUN {
            self.spill(0);
            self.out.push_zeros(run);
        } else {
            self.bits.zero_bytes(run);
            self.spill(STAGE);
        }
    }
}

/// A bounded, nonallocating reader for untrusted bit strings.
#[derive(Debug)]
pub struct BitReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}
impl<'a> BitReader<'a> {
    /// Start at bit zero.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn bit(&mut self) -> Result<bool, Error> {
        let byte = self.bytes.get(self.pos / 8).ok_or(Error::Encoding)?;
        let value = byte & (1 << (self.pos % 8)) != 0;
        self.pos += 1;
        Ok(value)
    }
    /// Decode exactly `bits` bits.
    pub fn unsigned(&mut self, bits: u32) -> Result<u128, Error> {
        if bits > 128 {
            return Err(Error::Encoding);
        }
        let mut value = 0;
        for i in 0..bits {
            value |= u128::from(self.bit()?) << i;
        }
        Ok(value)
    }
    /// Read a canonical element of $`[0,m)`$.
    pub fn uniform(&mut self, modulus: u128) -> Result<u128, Error> {
        if modulus < 2 {
            return Err(Error::Parameter("codec modulus"));
        }
        let value = self.unsigned(128 - (modulus - 1).leading_zeros())?;
        if value >= modulus {
            return Err(Error::Encoding);
        }
        Ok(value)
    }
    /// Decode exactly `bits` bits into a 256-bit value.
    pub fn unsigned_u256(&mut self, bits: u32) -> Result<U256, Error> {
        if bits > 256 {
            return Err(Error::Encoding);
        }
        let mut bytes = [0u8; 32];
        for i in 0..bits as usize {
            bytes[i / 8] |= u8::from(self.bit()?) << (i % 8);
        }
        Ok(U256::from_le_slice(&bytes))
    }
    /// [`BitReader::uniform`] for 256-bit moduli.
    pub fn uniform_u256(&mut self, modulus: &U256) -> Result<U256, Error> {
        if modulus < &U256::from_u8(2) {
            return Err(Error::Parameter("codec modulus"));
        }
        let value = self.unsigned_u256(modulus.wrapping_sub(&U256::ONE).bits_vartime())?;
        if &value >= modulus {
            return Err(Error::Encoding);
        }
        Ok(value)
    }
    /// Decode a Gaussian coefficient with bounded unary work and checked integer arithmetic.
    pub fn gaussian(&mut self, log_sigma: u32, max_unary: usize) -> Result<i128, Error> {
        if log_sigma > crate::rand::MAX_LOG_SIGMA {
            return Err(Error::Parameter("Gaussian exponent"));
        }
        let mut n = 0usize;
        while self.bit()? {
            if n == max_unary {
                return Err(Error::Encoding);
            }
            n += 1;
        }
        let high = if n.is_multiple_of(2) {
            -(n as i128 / 2)
        } else {
            n as i128 / 2 + 1
        };
        let bits = log_sigma + 1;
        let scale = 1i128 << bits;
        let r = self.unsigned(bits)? as i128;
        let low = if r >= scale / 2 { r - scale } else { r };
        high.checked_mul(scale)
            .and_then(|x| x.checked_add(low))
            .ok_or(Error::Encoding)
    }
    /// Decode a hint, bounding both the unary code and the output magnitude.
    pub fn hint(&mut self, max_abs: u128, max_unary: usize) -> Result<i128, Error> {
        let a = self.bit()?;
        let b = self.bit()?;
        let value = match (a, b) {
            (false, false) => 0,
            (false, true) => 1,
            (true, false) => -1,
            (true, true) => {
                let mut n = 0usize;
                while !self.bit()? {
                    if n == max_unary {
                        return Err(Error::Encoding);
                    }
                    n += 1;
                }
                if n.is_multiple_of(2) {
                    n as i128 / 2 + 2
                } else {
                    -(n as i128 / 2 + 2)
                }
            }
        };
        if value.unsigned_abs() > max_abs {
            return Err(Error::Encoding);
        }
        Ok(value)
    }
    /// Require exactly the terminal one bit, zero padding, and end of input.
    pub fn finish(mut self) -> Result<(), Error> {
        if !self.bit()? {
            return Err(Error::Encoding);
        }
        while !self.pos.is_multiple_of(8) {
            if self.bit()? {
                return Err(Error::Encoding);
            }
        }
        if self.pos / 8 != self.bytes.len() {
            return Err(Error::Encoding);
        }
        Ok(())
    }
}
