//! Property tests of the decoders: every bit code round-trips and is canonical, the bit reader
//! never panics, every accepted mutation of a valid proof re-encodes to itself and fails
//! verification, and parameter JSON either loads and round-trips or is refused. The generators
//! use fixed seeds and no failure files, so every run checks the same cases; a failure prints
//! its minimal input.
use jali::{
    abdlop::{Abdlop, Commitment},
    codec::{BitReader, BitWriter},
    lnp::{AffineBlock, L2Block, Statement},
    math::{Poly, PolyMat, PolyVec, Ring, U256},
    params::toy_d64,
    statement::Requirements,
    tbox,
};
use proptest::{prelude::*, test_runner::RngSeed};
use std::sync::{Arc, OnceLock};

mod common;
use common::{
    from_u64_words,
    params::{FACTORS_240, even_divisors, fit_wide, power_plus},
};

fn config(cases: u32, seed: u64) -> ProptestConfig {
    ProptestConfig {
        cases,
        failure_persistence: None,
        rng_seed: RngSeed::Fixed(seed),
        ..ProptestConfig::default()
    }
}

#[derive(Clone, Debug)]
enum Op {
    Unsigned(u128, u32),
    Uniform(u128, u128),
    Gaussian(i128, u32),
    Hint(i128),
}
fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        (any::<u128>(), 0u32..=128)
            .prop_map(|(v, b)| Op::Unsigned(if b == 128 { v } else { v & ((1u128 << b) - 1) }, b)),
        (2u128..u128::MAX)
            .prop_flat_map(|m| (0..m, Just(m)))
            .prop_map(|(v, m)| Op::Uniform(v, m)),
        (0u32..30)
            .prop_flat_map(|t| (-(1i128 << (t + 18))..(1i128 << (t + 18)), Just(t)))
            .prop_map(|(z, t)| Op::Gaussian(z, t)),
        (-5000i128..5000).prop_map(Op::Hint),
    ]
}

proptest! {
    #![proptest_config(config(512, 25))]
    /// Encoding any sequence of values and decoding it returns the values and consumes the
    /// terminal bit and padding exactly.
    #[test]
    fn bit_codes_round_trip_and_are_canonical(ops in prop::collection::vec(op(), 0..40)) {
        let mut w = BitWriter::new();
        for o in &ops {
            match o {
                Op::Unsigned(v, b) => w.unsigned(*v, *b).unwrap(),
                Op::Uniform(v, m) => w.uniform(*v, *m).unwrap(),
                Op::Gaussian(z, t) => w.gaussian(*z, *t, 1 << 20).unwrap(),
                Op::Hint(h) => w.hint(*h, 1 << 20).unwrap(),
            }
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for o in &ops {
            match o {
                Op::Unsigned(v, b) => prop_assert_eq!(r.unsigned(*b).unwrap(), *v),
                Op::Uniform(v, m) => prop_assert_eq!(r.uniform(*m).unwrap(), *v),
                Op::Gaussian(z, t) => prop_assert_eq!(r.gaussian(*t, 1 << 20).unwrap(), *z),
                Op::Hint(h) => prop_assert_eq!(r.hint(1 << 20, 1 << 20).unwrap(), *h),
            }
        }
        prop_assert!(r.finish().is_ok());
    }
}

/// A modulus of `bits` bits from random words, at least 2.
fn wide_modulus(words: [u64; 4], bits: u32) -> U256 {
    let x = from_u64_words(words);
    let x = match bits {
        256 => x,
        _ => x.bitand(&U256::ONE.shl_vartime(bits).wrapping_sub(&U256::ONE)),
    };
    x.bitor(&U256::ONE.shl_vartime(bits - 1))
        .max(U256::from_u8(2))
}
/// Uniform codes modulo 256-bit moduli, and fixed-width 256-bit codes, mixed.
fn wide_op() -> impl Strategy<Value = (U256, U256, bool)> {
    (
        any::<[u64; 4]>(),
        any::<[u64; 4]>(),
        1u32..=256,
        any::<bool>(),
    )
        .prop_map(|(m, v, bits, uniform)| {
            let modulus = wide_modulus(m, bits);
            let value = from_u64_words(v).rem_vartime(&modulus.to_nz().unwrap());
            (value, modulus, uniform)
        })
}

proptest! {
    #![proptest_config(config(256, 30))]
    /// Uniform codes modulo any modulus from 2 to 2^256 - 1 and fixed-width codes of up to 256
    /// bits round-trip canonically; a value at or above its modulus is refused by the writer,
    /// and its code, where the width holds it, by the reader.
    #[test]
    fn wide_bit_codes_round_trip_and_are_canonical(ops in prop::collection::vec(wide_op(), 0..24)) {
        let width = |m: &U256| m.wrapping_sub(&U256::ONE).bits_vartime();
        let mut w = BitWriter::new();
        for (value, modulus, uniform) in &ops {
            if *uniform {
                w.uniform_u256(value, modulus).unwrap();
                prop_assert!(w.uniform_u256(modulus, modulus).is_err());
            } else {
                w.unsigned_u256(value, width(modulus)).unwrap();
            }
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for (value, modulus, uniform) in &ops {
            let read = if *uniform {
                r.uniform_u256(modulus).unwrap()
            } else {
                r.unsigned_u256(width(modulus)).unwrap()
            };
            prop_assert_eq!(&read, value);
        }
        prop_assert!(r.finish().is_ok());
        for (_, modulus, _) in &ops {
            if modulus.bits_vartime() == width(modulus) {
                let mut w = BitWriter::new();
                w.unsigned_u256(modulus, width(modulus)).unwrap();
                let bytes = w.finish();
                prop_assert!(BitReader::new(&bytes).uniform_u256(modulus).is_err());
            }
        }
    }
}

proptest! {
    #![proptest_config(config(256, 31))]
    /// The 256-bit readers never panic on arbitrary bytes and widths.
    #[test]
    fn wide_bit_readers_never_panic_on_arbitrary_bytes(
        bytes in prop::collection::vec(any::<u8>(), 0..96),
        reads in prop::collection::vec((any::<[u64; 4]>(), 0u32..300, any::<bool>()), 0..16),
    ) {
        let mut r = BitReader::new(&bytes);
        for (m, bits, uniform) in reads {
            let _ = if uniform {
                r.uniform_u256(&from_u64_words(m)).map(|_| ())
            } else {
                r.unsigned_u256(bits).map(|_| ())
            };
        }
        let _ = r.finish();
    }
}

proptest! {
    #![proptest_config(config(256, 26))]
    /// Reading any bytes with any sequence of codes returns values or errors, never panics.
    #[test]
    fn bit_reader_never_panics_on_arbitrary_bytes(
        bytes in prop::collection::vec(any::<u8>(), 0..64),
        kinds in prop::collection::vec(0u8..4, 0..64),
    ) {
        let mut r = BitReader::new(&bytes);
        for k in kinds {
            let _ = match k {
                0 => r.unsigned(17).map(|_| ()),
                1 => r.uniform(1099511627917).map(|_| ()),
                2 => r.gaussian(12, 64).map(|_| ()),
                _ => r.hint(100, 64).map(|_| ()),
            };
        }
        let _ = r.finish();
    }
}

/// Flip each bit index of `flips` in `bytes`, then drop the last byte (`cut` 1) or append one
/// (`cut` 2).
fn mutate(bytes: &[u8], flips: &[prop::sample::Index], cut: usize) -> Vec<u8> {
    let mut b = bytes.to_vec();
    for f in flips {
        let i = f.index(b.len() * 8);
        b[i / 8] ^= 1 << (i % 8);
    }
    match cut {
        1 => {
            b.pop();
        }
        2 => b.push(0x80),
        _ => {}
    }
    b
}

fn opening_fixture() -> &'static (Abdlop, Commitment, Vec<u8>) {
    static F: OnceLock<(Abdlop, Commitment, Vec<u8>)> = OnceLock::new();
    F.get_or_init(|| {
        let scheme = Abdlop::new([9; 32], toy_d64()).unwrap();
        let ring = scheme.ring().clone();
        let s1 = PolyVec::new(
            ring.clone(),
            vec![Poly::constant(ring.clone(), 1); scheme.bounded_len()],
        )
        .unwrap();
        let m = PolyVec::new(
            ring.clone(),
            vec![Poly::constant(ring, 3); scheme.message_len()],
        )
        .unwrap();
        let (c, o) = scheme.commit_with_seed(s1, m, [5; 32]).unwrap();
        let proof = scheme.prove_with_seed(&c, &o, b"ctx", [8; 32]).unwrap();
        let bytes = scheme.encode_proof(&proof).unwrap();
        (scheme, c, bytes)
    })
}

proptest! {
    #![proptest_config(config(400, 27))]
    /// Every accepted decoding of a mutated opening proof re-encodes to the same bytes (no
    /// malleable encodings), and fails verification unless the bytes are unchanged.
    #[test]
    fn mutated_opening_proofs_are_canonical_or_rejected(
        flips in prop::collection::vec(any::<prop::sample::Index>(), 1..4),
        cut in 0usize..3,
    ) {
        let (scheme, commitment, bytes) = opening_fixture();
        let b = mutate(bytes, &flips, cut);
        if let Ok(p) = scheme.decode_proof(&b) {
            prop_assert_eq!(&scheme.encode_proof(&p).unwrap(), &b);
            if &b != bytes {
                prop_assert!(scheme.verify(commitment, &p, b"ctx").is_err());
            }
        }
    }
}

/// An opening proof at q = 2^240 + 325 with test-only parameters for the toolbox shape.
fn wide_opening_fixture() -> &'static (Abdlop, Commitment, Vec<u8>) {
    static F: OnceLock<(Abdlop, Commitment, Vec<u8>)> = OnceLock::new();
    F.get_or_init(|| {
        let q = power_plus(240, 325);
        let req = Requirements {
            m1: 10,
            l: 2,
            alpha_squared: 640,
            n_bin: 2,
            l2_rows: vec![2, 1],
            l2_bounds_squared: vec![128, 64],
            n_prime: 2,
            linf_bound: 4,
            max_integer_coefficient: U256::ZERO,
            approx_alpha_squared: None,
            lifted_moduli: Vec::new(),
            linf: None,
        };
        let params = fit_wide(
            &req,
            "codec-q240-test-only",
            q,
            &even_divisors(&q, &FACTORS_240),
            64,
        )
        .unwrap();
        let scheme = Abdlop::new([9; 32], params).unwrap();
        let ring = scheme.ring().clone();
        let s1 = PolyVec::new(
            ring.clone(),
            vec![Poly::constant(ring.clone(), 1); scheme.bounded_len()],
        )
        .unwrap();
        let m = PolyVec::new(
            ring.clone(),
            vec![Poly::constant(ring, 3); scheme.message_len()],
        )
        .unwrap();
        let (c, o) = scheme.commit_with_seed(s1, m, [5; 32]).unwrap();
        let proof = scheme.prove_with_seed(&c, &o, b"ctx", [8; 32]).unwrap();
        let bytes = scheme.encode_proof(&proof).unwrap();
        (scheme, c, bytes)
    })
}

proptest! {
    #![proptest_config(config(200, 32))]
    /// The same at a 241-bit modulus, whose hint and response bounds exceed 128 bits.
    #[test]
    fn mutated_wide_opening_proofs_are_canonical_or_rejected(
        flips in prop::collection::vec(any::<prop::sample::Index>(), 1..4),
        cut in 0usize..3,
    ) {
        let (scheme, commitment, bytes) = wide_opening_fixture();
        let b = mutate(bytes, &flips, cut);
        if let Ok(p) = scheme.decode_proof(&b) {
            prop_assert_eq!(&scheme.encode_proof(&p).unwrap(), &b);
            if &b != bytes {
                prop_assert!(scheme.verify(commitment, &p, b"ctx").is_err());
            }
        }
    }
}

fn selector(ring: &Arc<Ring>, columns: usize, indices: &[usize]) -> PolyMat {
    PolyMat::new(
        ring.clone(),
        indices.len(),
        columns,
        indices
            .iter()
            .flat_map(|i| {
                (0..columns).map(move |j| Poly::constant(ring.clone(), i128::from(*i == j)))
            })
            .collect(),
    )
    .unwrap()
}
/// The statement and witness of `tests/toolbox.rs`.
fn toolbox_fixture() -> &'static (Abdlop, Statement, Vec<u8>) {
    static F: OnceLock<(Abdlop, Statement, Vec<u8>)> = OnceLock::new();
    F.get_or_init(|| {
        let scheme = Abdlop::new([1; 32], toy_d64()).unwrap();
        let ring = scheme.ring().clone();
        let select_s = |indices: &[usize]| AffineBlock {
            rows: indices.len(),
            s: Some(selector(&ring, 10, indices)),
            m: None,
            offset: None,
        };
        let statement = Statement {
            quadratic: vec![],
            evaluation: vec![],
            binary: Some(select_s(&[0, 1])),
            l2: vec![
                L2Block {
                    map: select_s(&[2, 3]),
                    bound_squared: 128,
                },
                L2Block {
                    map: select_s(&[4]),
                    bound_squared: 64,
                },
            ],
            arp: Some(AffineBlock {
                rows: 2,
                s: None,
                m: Some(selector(&ring, 2, &[0, 1])),
                offset: None,
            }),
        };
        let s1 = PolyVec::new(
            ring.clone(),
            (0..10)
                .map(|i| {
                    Poly::new(
                        ring.clone(),
                        (0..64).map(|j| ((i + j) % 2) as i128).collect(),
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap();
        let m = PolyVec::new(
            ring.clone(),
            vec![Poly::constant(ring.clone(), 2), Poly::constant(ring, -3)],
        )
        .unwrap();
        let proof = tbox::prove_with_seed(&scheme, &statement, &s1, &m, b"ctx", [2; 32]).unwrap();
        let bytes = jali::codec::proof::encode(&scheme, &proof).unwrap();
        (scheme, statement, bytes)
    })
}

proptest! {
    #![proptest_config(config(32, 28))]
    /// The same for the toolbox codec; each verification takes a fraction of a second.
    #[test]
    fn mutated_toolbox_proofs_are_canonical_or_rejected(
        flips in prop::collection::vec(any::<prop::sample::Index>(), 1..4),
        cut in 0usize..3,
    ) {
        let (scheme, statement, bytes) = toolbox_fixture();
        let b = mutate(bytes, &flips, cut);
        if let Ok(p) = jali::codec::proof::decode(scheme, &b) {
            prop_assert_eq!(&jali::codec::proof::encode(scheme, &p).unwrap(), &b);
            if &b != bytes {
                prop_assert!(tbox::verify(scheme, statement, &p, b"ctx").is_err());
            }
        }
    }
}

#[cfg(feature = "serde")]
mod json {
    use super::*;
    use jali::params::TboxParams;
    use serde_json::{Value, json};

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../examples/fixtures/possession.json")).unwrap()
    }
    /// Replacement values for one field: extremes, wrong types, oversized arrays.
    fn replacement() -> impl Strategy<Value = Value> {
        prop_oneof![
            Just(json!(0)),
            Just(json!(1)),
            Just(json!(-1)),
            Just(json!(u64::MAX)),
            Just(json!(u32::MAX)),
            Just(json!(1.5)),
            Just(json!(1e308)),
            Just(json!("text")),
            Just(json!(null)),
            Just(json!([])),
            Just(json!([0, 0, 0, 0])),
            Just(json!([u64::MAX, 5])),
            Just(Value::Array(vec![json!(1); 1025])),
            Just(Value::Array(vec![json!(13); 70000])),
            Just(json!({"nested": 1})),
            (0u64..200).prop_map(|x| json!(x)),
        ]
    }

    proptest! {
        #![proptest_config(config(512, 29))]
        /// A field replaced, removed, or an unknown field added: `from_json` loads or refuses,
        /// never panics, and what it loads round-trips through JSON.
        #[test]
        fn params_json_never_panics_and_round_trips(
            field in 0usize..20,
            value in replacement(),
            kind in 0u8..3,
        ) {
            let mut v = fixture();
            let object = v.as_object_mut().unwrap();
            let keys: Vec<String> = object.keys().cloned().collect();
            let key = keys[field % keys.len()].clone();
            match kind {
                0 => {
                    object.insert(key, value);
                }
                1 => {
                    object.remove(&key);
                }
                _ => {
                    object.insert(format!("{key}_extra"), value);
                }
            }
            let text = serde_json::to_string(&v).unwrap();
            if let Ok(params) = TboxParams::from_json(&text) {
                let again = serde_json::to_string(&params).unwrap();
                prop_assert_eq!(TboxParams::from_json(&again).unwrap(), params);
            }
        }
    }

    #[test]
    fn the_unmodified_fixture_loads() {
        TboxParams::from_json(&fixture().to_string()).unwrap();
    }
}
