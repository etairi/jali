//! The word writer against a bitwise reference writer, kept below as the oracle: random
//! sequences of every code and of zero runs, at every alignment, give the same bytes and the
//! same errors, and the reader decodes them back. (The word bit reads of the samplers are tested
//! in `rand::uniform`.)
use super::*;

/// The reference writer: one call per bit.
#[derive(Default)]
struct Bitwise {
    bytes: Vec<u8>,
    bit_len: usize,
}
impl Bitwise {
    fn bit(&mut self, value: bool) {
        if self.bit_len.is_multiple_of(8) {
            self.bytes.push(0);
        }
        if value {
            self.bytes[self.bit_len / 8] |= 1 << (self.bit_len % 8);
        }
        self.bit_len += 1;
    }
    fn unsigned(&mut self, value: u128, bits: u32) -> Result<(), Error> {
        if bits > 128 || (bits < 128 && value >> bits != 0) {
            return Err(Error::Encoding);
        }
        for i in 0..bits {
            self.bit((value >> i) & 1 != 0);
        }
        Ok(())
    }
    fn uniform(&mut self, value: u128, modulus: u128) -> Result<(), Error> {
        if modulus < 2 || value >= modulus {
            return Err(Error::Encoding);
        }
        self.unsigned(value, 128 - (modulus - 1).leading_zeros())
    }
    fn unsigned_u256(&mut self, value: &U256, bits: u32) -> Result<(), Error> {
        if bits > 256 || value.bits_vartime() > bits {
            return Err(Error::Encoding);
        }
        for i in 0..bits {
            self.bit(value.bit_vartime(i));
        }
        Ok(())
    }
    fn uniform_u256(&mut self, value: &U256, modulus: &U256) -> Result<(), Error> {
        if modulus < &U256::from_u8(2) || value >= modulus {
            return Err(Error::Encoding);
        }
        self.unsigned_u256(value, modulus.wrapping_sub(&U256::ONE).bits_vartime())
    }
    fn gaussian(&mut self, z: i128, log_sigma: u32, max_unary: usize) -> Result<(), Error> {
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
        for _ in 0..n {
            self.bit(true);
        }
        self.bit(false);
        self.unsigned(low.rem_euclid(scale) as u128, bits)
    }
    fn hint(&mut self, value: i128, max_unary: usize) -> Result<(), Error> {
        match value {
            0 => {
                self.bit(false);
                self.bit(false);
            }
            1 => {
                self.bit(false);
                self.bit(true);
            }
            -1 => {
                self.bit(true);
                self.bit(false);
            }
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
                self.bit(true);
                self.bit(true);
                for _ in 0..n {
                    self.bit(false);
                }
                self.bit(true);
            }
        }
        Ok(())
    }
    fn finish(mut self) -> Vec<u8> {
        self.bit(true);
        self.bytes
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn wide(&mut self) -> u128 {
        u128::from(self.next()) << 64 | u128::from(self.next())
    }
    fn u256(&mut self) -> U256 {
        let words = [self.next(), self.next(), self.next(), self.next()];
        U256::from_le_slice(&words.map(u64::to_le_bytes).concat())
    }
}

/// One random field: its code, a value that fits, one that does not, or a zero run.
#[derive(Clone, Debug)]
enum Field {
    Unsigned(u128, u32),
    Uniform(u128, u128),
    UnsignedWide(U256, u32),
    UniformWide(U256, U256),
    Gaussian(i128, u32, usize),
    Hint(i128, usize),
    Zeros(u64),
}

fn field(rng: &mut Rng) -> Field {
    let bits = |rng: &mut Rng, max: u64| rng.below(max + 1) as u32;
    match rng.below(7) {
        0 => {
            let b = bits(rng, 128);
            let value = rng.wide();
            // Mostly in range; sometimes one bit too wide, which both refuse.
            let value = match rng.below(4) {
                0 => value,
                _ if b == 128 => value,
                _ => value & ((1 << b) - 1),
            };
            Field::Unsigned(value, b)
        }
        1 => {
            let modulus = rng.wide() >> rng.below(127);
            Field::Uniform(
                rng.wide() % modulus.max(1) + u128::from(rng.below(3) == 0),
                modulus,
            )
        }
        2 => {
            let b = bits(rng, 256);
            let value = rng.u256().shr_vartime(rng.below(257) as u32 % 256);
            Field::UnsignedWide(value, b)
        }
        3 => {
            let modulus = rng.u256().shr_vartime(rng.below(255) as u32);
            let value = rng.u256().shr_vartime(rng.below(256) as u32);
            Field::UniformWide(value, modulus)
        }
        4 => {
            // Mostly |z| < 2^(t + 8), so that the unary part fits; sometimes any value, or a
            // width above the cap.
            let t = rng.below(102) as u32;
            let shift = if rng.below(4) == 0 {
                rng.below(127) as u32
            } else {
                118u32.saturating_sub(t).clamp(1, 126)
            };
            let z = (rng.wide() as i128) >> shift;
            Field::Gaussian(z, t, rng.below(600) as usize)
        }
        5 => {
            let value = (rng.next() as i64 >> (54 + rng.below(10))) as i128;
            let value = if rng.below(8) == 0 {
                (rng.next() as i64) as i128
            } else {
                value
            };
            Field::Hint(value, rng.below(300) as usize)
        }
        _ => {
            let max = if rng.below(3) == 0 { 2000 } else { 130 };
            Field::Zeros(rng.below(max))
        }
    }
}

/// Write the field with both writers: equal results, which it returns.
fn write(f: &Field, new: &mut BitWriter, bitwise: &mut Bitwise) -> Result<(), Error> {
    let (a, b) = match f {
        Field::Unsigned(v, b) => (new.unsigned(*v, *b), bitwise.unsigned(*v, *b)),
        Field::Uniform(v, m) => (new.uniform(*v, *m), bitwise.uniform(*v, *m)),
        Field::UnsignedWide(v, b) => (new.unsigned_u256(v, *b), bitwise.unsigned_u256(v, *b)),
        Field::UniformWide(v, m) => (new.uniform_u256(v, m), bitwise.uniform_u256(v, m)),
        Field::Gaussian(z, t, n) => (new.gaussian(*z, *t, *n), bitwise.gaussian(*z, *t, *n)),
        Field::Hint(v, n) => (new.hint(*v, *n), bitwise.hint(*v, *n)),
        Field::Zeros(n) => {
            new.zeros(*n);
            for _ in 0..*n {
                bitwise.bit(false);
            }
            (Ok(()), Ok(()))
        }
    };
    assert_eq!(a, b, "{f:?}");
    a
}

/// Read a field written without error back, as the reader decodes it.
fn read(f: &Field, reader: &mut BitReader) {
    match f {
        Field::Unsigned(v, b) => assert_eq!(reader.unsigned(*b), Ok(*v)),
        Field::Uniform(v, m) => assert_eq!(reader.uniform(*m), Ok(*v)),
        Field::UnsignedWide(v, b) => assert_eq!(reader.unsigned_u256(*b), Ok(*v)),
        Field::UniformWide(v, m) => assert_eq!(reader.uniform_u256(m), Ok(*v)),
        Field::Gaussian(z, t, n) => assert_eq!(reader.gaussian(*t, *n), Ok(*z)),
        Field::Hint(v, n) => {
            assert_eq!(reader.hint(v.unsigned_abs(), *n), Ok(*v))
        }
        Field::Zeros(n) => {
            for _ in 0..*n {
                assert_eq!(reader.unsigned(1), Ok(0));
            }
        }
    }
}

#[test]
fn random_field_sequences_equal_the_bitwise_writer_and_round_trip() {
    let mut rng = Rng(6);
    let mut written_kinds = [0usize; 7];
    for sequence in 0..3000 {
        let mut new = BitWriter::new();
        let mut bitwise = Bitwise::default();
        let mut written = Vec::new();
        for _ in 0..rng.below(24) {
            let f = field(&mut rng);
            if write(&f, &mut new, &mut bitwise).is_ok() {
                written_kinds[kind(&f)] += 1;
                written.push(f);
            }
        }
        let bytes = new.finish();
        assert_eq!(bytes, bitwise.finish(), "sequence {sequence}");
        let mut reader = BitReader::new(&bytes);
        for f in &written {
            read(f, &mut reader);
        }
        assert_eq!(reader.finish(), Ok(()), "sequence {sequence}");
    }
    // Every kind of field was written without error many times.
    assert!(
        written_kinds.iter().all(|n| *n >= 1000),
        "{written_kinds:?}"
    );
}

fn kind(f: &Field) -> usize {
    match f {
        Field::Unsigned(..) => 0,
        Field::Uniform(..) => 1,
        Field::UnsignedWide(..) => 2,
        Field::UniformWide(..) => 3,
        Field::Gaussian(..) => 4,
        Field::Hint(..) => 5,
        Field::Zeros(..) => 6,
    }
}

#[test]
fn zero_runs_equal_zero_bits_at_every_alignment() {
    for offset in 0..=128u32 {
        let ones = if offset == 0 {
            0
        } else {
            u128::MAX >> (128 - offset)
        };
        for run in [0, 1, 2, 7, 8, 63, 64, 65, 127, 128, 129, 191, 4096, 4097] {
            let mut new = BitWriter::new();
            let mut bitwise = Bitwise::default();
            new.unsigned(ones, offset).unwrap();
            bitwise.unsigned(ones, offset).unwrap();
            new.zeros(run);
            for _ in 0..run {
                bitwise.bit(false);
            }
            new.unsigned(0b101, 3).unwrap();
            bitwise.unsigned(0b101, 3).unwrap();
            assert_eq!(new.finish(), bitwise.finish(), "offset {offset}, run {run}");
        }
    }
}
