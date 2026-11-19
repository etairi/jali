//! The transcript against the draft and against independent computations: the tags, the
//! session identifier and the sponge (known answers computed with Python's `hashlib`), the
//! absorption of segments and length-prefixed strings, and the decodings of verifier messages.
use super::*;
use crate::{
    fiat_shamir::decode_uint,
    rand::{challenge::draws_from_bytes, challenge_from_bytes, within_eta},
};

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
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

#[test]
fn the_tags_are_distinct_and_none_is_a_prefix_of_another() {
    for (i, a) in Protocol::ALL.iter().enumerate() {
        assert!(a.tag().is_ascii() && a.tag().starts_with(b"JALI-V02-"));
        assert!(a.tag().ends_with(b"-C1-SHAKE128"));
        for (j, b) in Protocol::ALL.iter().enumerate() {
            if i != j {
                assert!(!b.tag().starts_with(a.tag()), "{a:?} is a prefix of {b:?}");
            }
        }
    }
}

#[test]
fn session_identifier_and_sponge_match_an_independent_computation() {
    // Python: sid = shake_128(b"irtf-cfrg-fiat-shamir/session-id" + bytes(136) + tag).digest(32)
    // with tag = b"JALI-V02-LNP22-TBOX-C1-SHAKE128" + (3).to_bytes(4, "little") + b"ctx", then
    // shake_128(sid + bytes(136) + b"instance").digest(24) for the two squeezes, and
    // shake_128(sid + bytes(136) + b"instance" + b"message").digest(8) after the absorb.
    let mut t = Transcript::new(Protocol::Toolbox, b"ctx").unwrap();
    t.absorb(b"instance");
    assert_eq!(
        hex::encode(t.binding()),
        "eb677f9c3f9cf0571a7c41cfabec100bcdc919faf617de67d7454c770d8930be"
    );
    let mut out = [0u8; 16];
    t.squeeze(&mut out);
    assert_eq!(hex::encode(out), "b6b3acee4f01496b33991bdc9eb6286d");
    let mut next = [0u8; 8];
    t.squeeze(&mut next);
    assert_eq!(hex::encode(next), "14aab9604cdc2880");
    t.absorb(b"message");
    t.squeeze(&mut next);
    assert_eq!(hex::encode(next), "8c1d482c871d4e87");
    // Another context, another protocol, or the context moved into the instance: other states.
    let start = |protocol, context: &[u8], instance: &[u8]| {
        let mut t = Transcript::new(protocol, context).unwrap();
        t.absorb(instance);
        t.peek()
    };
    let base = start(Protocol::Toolbox, b"ctx", b"instance");
    assert_ne!(base, start(Protocol::Toolbox, b"ctx2", b"instance"));
    assert_ne!(
        base,
        start(Protocol::ToolboxExtractable, b"ctx", b"instance")
    );
    assert_ne!(base, start(Protocol::Toolbox, b"", b"ctxinstance"));
}

#[test]
fn absorbing_segments_equals_absorbing_their_bytes() {
    let mut rng = Rng(5);
    for case in 0..300usize {
        let mut segments = Segments::default();
        for _ in 0..rng.below(12) {
            let max = if rng.below(4) == 0 { 80_000 } else { 300 };
            if rng.below(2) == 0 {
                let n = rng.below(max) as usize;
                segments.push_literal(&rng.bytes(n));
            } else {
                segments.push_zeros(rng.below(max));
            }
        }
        segments.shrink_to_fit();
        let bytes = segments.to_vec();
        let mut joined = Transcript::new(Protocol::Quadratic, &[case as u8; 3]).unwrap();
        let mut pieces = joined.clone();
        let mut varlen = joined.clone();
        joined.absorb(&bytes);
        pieces.absorb_segments(&segments);
        assert_eq!(joined.peek(), pieces.peek(), "case {case}");
        // SerializeVarLenString: LE32 of the length, then the bytes.
        varlen.absorb_varlen_segments(&segments).unwrap();
        let mut reference = Transcript::new(Protocol::Quadratic, &[case as u8; 3]).unwrap();
        reference.absorb(&(bytes.len() as u32).to_le_bytes());
        reference.absorb(&bytes);
        assert_eq!(varlen.peek(), reference.peek(), "case {case}");
    }
    let t = Transcript::new(Protocol::Opening, b"").unwrap();
    assert_eq!(t.clone().absorb_len(1 << 32), Err(Error::Dimension));
}

#[test]
fn uints_are_decode_uint_of_consecutive_squeezes() {
    let moduli = [
        U256::from_u64(3329),
        U256::from_u64(2423991946973),
        U256::from_u64(u64::MAX),
        U256::from_u128(1 << 64),
        U256::from_u128((1 << 127) - 1),
        U256::ONE.shl_vartime(255).wrapping_sub(&U256::from_u8(19)),
        U256::MAX,
    ];
    for (i, modulus) in moduli.iter().enumerate() {
        let count = 3000;
        let mut t = Transcript::new(Protocol::Evaluation, &[i as u8]).unwrap();
        let mut reference = t.clone();
        let values = t.uints(modulus, count).unwrap();
        let width = ns(modulus).unwrap() + 16;
        let mut bytes = vec![0u8; width * count];
        reference.squeeze(&mut bytes);
        for (value, chunk) in values.iter().zip(bytes.chunks_exact(width)) {
            assert_eq!(*value, decode_uint(chunk, modulus).unwrap(), "{modulus}");
            assert!(value < modulus);
        }
        assert_eq!(t.peek(), reference.peek());
    }
    // The 64-bit path against the general one, on buffers of every length up to 24 bytes.
    let mut rng = Rng(9);
    for _ in 0..20_000 {
        let m = rng.next().max(2);
        let len = 1 + rng.below(24) as usize;
        let buf = rng.bytes(len);
        assert_eq!(
            U256::from_u64(reduce_le(&buf, m)),
            decode_uint(&buf, &U256::from_u64(m)).unwrap()
        );
    }
    let mut t = Transcript::new(Protocol::Evaluation, b"").unwrap();
    assert_eq!(
        t.uints(&U256::ONE, 1).err(),
        Some(Error::Parameter("decoding modulus"))
    );
}

#[test]
fn ternary_entries_are_the_centred_binomial_decoding_of_fips_203() {
    let mut t = Transcript::new(Protocol::Toolbox, b"ternary").unwrap();
    let mut reference = t.clone();
    let count = 40_001;
    let entries = t.ternary(count).unwrap();
    let mut bytes = vec![0u8; count.div_ceil(4)];
    reference.squeeze(&mut bytes);
    let bit = |j: usize| i8::from(bytes[j / 8] >> (j % 8) & 1 == 1);
    for (i, x) in entries.iter().enumerate() {
        assert_eq!(*x, bit(2 * i) - bit(2 * i + 1), "{i}");
    }
    assert_eq!(t.peek(), reference.peek());
    // P(0) = 1/2 and P(1) = P(-1) = 1/4: counts within 5 standard deviations.
    let n = count as f64;
    for (value, p) in [(-1i8, 0.25), (0, 0.5), (1, 0.25)] {
        let k = entries.iter().filter(|x| **x == value).count() as f64;
        assert!(
            (k - n * p).abs() < 5.0 * (n * p * (1.0 - p)).sqrt(),
            "{value}: {k}"
        );
    }
}

#[test]
fn binding_leaves_the_transcript_unchanged_and_follows_its_state() {
    let mut t = Transcript::new(Protocol::Opening, b"binding").unwrap();
    t.absorb(b"instance");
    let before = t.peek();
    let binding = t.binding();
    assert_eq!(t.peek(), before);
    assert_eq!(t.binding(), binding);
    assert_ne!(binding, before);
    t.absorb(b"message");
    assert_ne!(t.binding(), binding);
}

#[test]
fn the_challenge_is_the_first_draw_within_eta_of_its_squeezed_bytes() {
    let mut rng = Rng(77);
    for (d, omega, eta, words) in [(64usize, 8i128, 140u64, 6451usize), (128, 2, 59, 10752)] {
        let ring = Ring::new(1099511627917, d).unwrap();
        let k = (128 - ((2 * omega) as u128).leading_zeros()) as usize;
        assert_eq!(CHALLENGE_BYTES * 8 / k, words);
        let mut redrawn = 0;
        for case in 0..400 {
            let bytes = rng.bytes(CHALLENGE_BYTES);
            let c = challenge_from_bytes(&bytes, ring.clone(), omega, eta).unwrap();
            let draws = draws_from_bytes(&bytes, ring.clone(), omega);
            let first = draws
                .iter()
                .position(|x| within_eta(x, eta).unwrap())
                .expect("a draw within eta");
            redrawn += usize::from(first > 0);
            assert_eq!(c, draws[first], "{d} {case}");
            assert_eq!(c.auto(), c);
            assert_eq!(c.coefficient_i128(d / 2), Ok(0));
            assert!(c.norm_infinity() <= U256::from_u128(omega as u128));
        }
        // About 1% of first draws exceed eta.
        assert!(redrawn < 20, "{d}: {redrawn} redrawn");
        // The squeeze takes CHALLENGE_BYTES bytes.
        let mut t = Transcript::new(Protocol::Opening, b"challenge").unwrap();
        let mut reference = t.clone();
        let c = t.challenge(ring.clone(), omega, eta).unwrap();
        let mut bytes = [0u8; CHALLENGE_BYTES];
        reference.squeeze(&mut bytes);
        assert_eq!(
            c,
            challenge_from_bytes(&bytes, ring.clone(), omega, eta).unwrap()
        );
        assert_eq!(t.peek(), reference.peek());
        // No draw within eta: the constant challenge 1. With eta = 0 only zero would pass.
        let one = Poly::constant(ring.clone(), 1);
        assert_eq!(
            challenge_from_bytes(&bytes, ring.clone(), omega, 0).unwrap(),
            one
        );
        assert_eq!(
            challenge_from_bytes(&[], ring.clone(), omega, eta).unwrap(),
            one
        );
        assert!(within_eta(&one, eta).unwrap());
        // A refused omega reads nothing and fails alike on any bytes.
        assert_eq!(
            challenge_from_bytes(&bytes, ring.clone(), 0, eta).err(),
            Some(Error::Parameter("challenge bound"))
        );
    }
}

#[test]
fn ring_elements_are_uints_per_coefficient() {
    let ring = Ring::new(3329, 64).unwrap();
    let mut t = Transcript::new(Protocol::ManyQuadratic, b"mu").unwrap();
    let mut reference = t.clone();
    let polys = t.ring_elements(&ring, 5).unwrap();
    let values = reference.uints(&ring.modulus(), 5 * 64).unwrap();
    for (p, chunk) in polys.iter().zip(values.as_chunks::<64>().0) {
        assert_eq!(p.coefficients(), chunk);
    }
    assert_eq!(t.peek(), reference.peek());
}

#[test]
fn barrett_remainders_equal_the_native_ones() {
    let mut rng = Rng(10);
    let mut moduli = vec![
        2u64,
        3,
        u64::MAX,
        u64::MAX - 1,
        1 << 63,
        (1 << 63) + 1,
        1 << 32,
        4294967291,
    ];
    moduli.extend((0..200).map(|_| rng.next().max(2)));
    for m in moduli {
        let b = Barrett64::new(m);
        let m128 = u128::from(m);
        for x in [
            0u128,
            1,
            m128 - 1,
            m128,
            m128 + 1,
            u128::MAX,
            u128::MAX - 1,
            u128::MAX / m128 * m128,
        ] {
            assert_eq!(b.reduce(x), x % m128, "m {m} x {x}");
        }
        for _ in 0..200 {
            let x = (u128::from(rng.next()) << 64) | u128::from(rng.next());
            assert_eq!(b.reduce(x), x % m128, "m {m} x {x}");
        }
    }
}

/// The two-word Horner reduction of `uints` against `DecodeUint`, for moduli from just above
/// $`2^{64}`$ to just below $`2^{126}`$, every chunk length up to 32 bytes, all-ones and random
/// chunks.
#[test]
fn two_word_reductions_equal_decode_uint() {
    let mut state = 0x243f_6a88_85a3_08d3u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let moduli = [
        (1u128 << 64) + 13,
        800602353498606932557,
        (1 << 100) + 277,
        (1 << 125) + 99,
        (1 << 126) - 1,
    ];
    for q in moduli {
        let modulus = U256::from_u128(q);
        let m = Mod128::new(&modulus).expect("a two-word modulus");
        for len in 1..=32usize {
            let mut chunks = vec![vec![0xffu8; len], vec![0u8; len]];
            for _ in 0..50 {
                chunks.push((0..len).map(|_| next() as u8).collect());
            }
            for chunk in chunks {
                assert_eq!(
                    U256::from_u128(reduce_le128(&m, &chunk)),
                    decode_uint(&chunk, &modulus).unwrap(),
                    "q {q} len {len}"
                );
            }
        }
    }
}
