//! The word-level bit reads against a bitwise reference loop, at every offset and width from
//! 0 to 128, and the samplers built on them against reference samplers on shared streams
//! (values and bytes read).
use super::*;
use crate::rand::{AesPrg, domain};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

/// The reference bit field read: one bit at a time.
fn reference_bits(buf: &[u8], start: usize, count: u32) -> u128 {
    let mut value = 0;
    for j in 0..count as usize {
        value |= u128::from((buf[(start + j) / 8] >> ((start + j) % 8)) & 1) << j;
    }
    value
}

#[test]
fn word_bit_reads_equal_the_bitwise_loop_at_every_offset_and_width() {
    let mut rng = Rng(7);
    let buf: Vec<u8> = (0..64).map(|_| rng.next() as u8).collect();
    for start in 0..=8 * buf.len() - 128 {
        for count in 0..=128 {
            assert_eq!(
                bits(&buf, start, count),
                reference_bits(&buf, start, count),
                "start {start}, count {count}"
            );
        }
    }
    // Fields that end at the last byte, so a read past the needed bytes would fail.
    for count in 0..=128u32 {
        let start = 8 * buf.len() - count as usize;
        assert_eq!(bits(&buf, start, count), reference_bits(&buf, start, count));
    }
}

/// Reference `uniform_u256`, for moduli of 128 bits and more: words assembled bit by bit.
fn reference_uniform_wide(
    stream: &mut impl ByteStream,
    modulus: &U256,
    count: usize,
) -> Result<Vec<U256>, Error> {
    let mbits = int::bits(&modulus.wrapping_sub(&U256::ONE)) as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..1024 {
        let missing = count - out.len();
        if missing == 0 {
            return Ok(out);
        }
        let mut bytes = vec![0u8; (missing * mbits).div_ceil(8)];
        stream.fill(&mut bytes)?;
        for i in 0..missing {
            let mut word = [0u8; 32];
            for j in 0..mbits {
                let index = i * mbits + j;
                word[j / 8] |= ((bytes[index / 8] >> (index % 8)) & 1) << (j % 8);
            }
            let value = U256::from_le_slice(&word[..]);
            if &value < modulus {
                out.push(value);
            }
        }
    }
    Err(Error::Randomness)
}

#[test]
fn wide_uniform_values_equal_the_reference_sampler_on_shared_streams() {
    let two = |k: u32| U256::ONE.shl_vartime(k);
    for (i, modulus) in [
        two(128),
        two(128).wrapping_add(&U256::ONE),
        two(128).wrapping_add(&U256::from_u8(165)),
        two(129).wrapping_sub(&U256::ONE),
        two(200).wrapping_sub(&U256::from_u8(75)),
        two(255).wrapping_sub(&U256::from_u8(19)),
        U256::MAX.wrapping_sub(&U256::from_u16(434)),
        U256::MAX,
    ]
    .iter()
    .enumerate()
    {
        let mut new = AesPrg::new(&[5; 32], domain(9, i as u32));
        let mut reference = AesPrg::new(&[5; 32], domain(9, i as u32));
        assert_eq!(
            uniform_u256(&mut new, modulus, 300).unwrap(),
            reference_uniform_wide(&mut reference, modulus, 300).unwrap(),
            "modulus {modulus}"
        );
        let (mut x, mut y) = ([0u8; 16], [0u8; 16]);
        new.fill(&mut x).unwrap();
        reference.fill(&mut y).unwrap();
        assert_eq!(x, y, "consumption, modulus {modulus}");
    }
}

#[test]
fn public_values_equal_the_uniform_sampler_on_shared_streams() {
    // The public sampler against `uniform` on two copies of a stream: the same values and the
    // same bytes read, at the edges of its 64-bit words (57 and 58 bits), of its 128-bit words
    // (120 and 121 bits), and at moduli with many and few rejections, for counts that leave
    // partial bytes and need several passes.
    let moduli: [u128; 22] = [
        2,
        3,
        5,
        255,
        256,
        257,
        (1 << 31) - 1,
        (1 << 32) - 99,
        (1 << 32) + 15,
        (1 << 56) - 5,
        1 << 57,
        (1 << 57) + 1,
        (1 << 63) - 25,
        (1 << 64) - 59,
        800602353498606932557,
        (1 << 119) - 1,
        1 << 120,
        (1 << 120) + 1,
        (1 << 127) - 1,
        (1 << 127) + 1,
        u128::MAX - 158,
        u128::MAX,
    ];
    for (i, m) in moduli.into_iter().enumerate() {
        for count in [1usize, 7, 64, 300] {
            let mut public = AesPrg::new(&[6; 32], domain(4, i as u32));
            let mut reference = AesPrg::new(&[6; 32], domain(4, i as u32));
            assert_eq!(
                public_uniform(&mut public, m, count).unwrap(),
                uniform(&mut reference, m, count).unwrap(),
                "modulus {m} count {count}"
            );
            let (mut x, mut y) = ([0u8; 16], [0u8; 16]);
            public.fill(&mut x).unwrap();
            reference.fill(&mut y).unwrap();
            assert_eq!(x, y, "consumption, modulus {m} count {count}");
        }
    }
    for q in [(1u128 << 32) - 99, 800602353498606932557] {
        let ring = Ring::new(q as i128, 64).unwrap();
        let mut public = AesPrg::new(&[7; 32], 3);
        let mut reference = AesPrg::new(&[7; 32], 3);
        assert_eq!(
            public_poly(&mut public, &ring).unwrap(),
            uniform_poly(&mut reference, &ring).unwrap()
        );
    }
}
