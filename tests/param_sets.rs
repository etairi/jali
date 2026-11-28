//! The parameter sets of the parameter tool that the crate ships (`params::kyber1024_d64`,
//! `kyber1024_d128` and `demo_d64`, from `src/params/sets/`). Each equals its JSON file (the
//! tool's `regenerate --check` checks that the file is the tool's output), passes `check` with
//! the values that the tool's Rust reference data recorded
//! (`tools/params/tests/data/rust_reference.json`), compiles its request's statement in the
//! worst case over public data, while the previous prime of its $`\gamma`$ class fails the
//! lifting check there, and proves and verifies the reference data's seeded random instance,
//! with the recorded proof length. `src/golden_tests.rs` pins their fingerprints.
use jali::{
    Error,
    lin::{self, Block},
    math::{Poly, PolyMat, PolyVec, Ring, U256, int::is_prime},
    params::{
        ExtractionParams, TboxParams, demo_d64, demo_d64_extraction, kyber1024_d64,
        kyber1024_d64_extraction, kyber1024_d128, kyber1024_d128_extraction,
    },
    statement::Norm,
};
use std::{collections::BTreeMap, sync::Arc};

mod common;
use common::{SplitMix64, narrow};

/// One shipped set with what the reference data recorded for it.
struct Set {
    name: &'static str,
    params: fn() -> TboxParams,
    /// The statement modulus and squared bound of the request's `lin-worst-case` statement.
    p: i128,
    bound_squared: u64,
    /// `CheckedParams`: q, lambda, n_ex, l_ext, b_squared, z1 and z3 bounds squared, z4 bound,
    /// exact and approximate alpha squared, estimated proof bytes.
    checked: [u128; 11],
    /// `max_integer_coefficient` and `linf_bound` of the worst-case requirements.
    lifting: (u128, u128),
    /// The previous prime of the set's gamma class.
    previous: u64,
    /// The length of the seeded proof.
    proof_bytes: usize,
}

const SETS: [Set; 3] = [
    Set {
        name: "kyber1024-d64",
        params: kyber1024_d64,
        p: 3329,
        bound_squared: 2950,
        checked: [
            2423991946973,
            4,
            33,
            12,
            36460193441136,
            43586015314575,
            27753065054,
            208037478,
            3014,
            12372557824,
            21004,
        ],
        lifting: (11570096, 3476),
        previous: 2423973072749,
        proof_bytes: 21004,
    },
    Set {
        name: "kyber1024-d128",
        params: kyber1024_d128,
        p: 3329,
        bound_squared: 2950,
        checked: [
            2423989135261,
            4,
            17,
            8,
            112262557837350,
            11226700914360,
            27753065054,
            208037478,
            3078,
            12372557824,
            22552,
        ],
        lifting: (11570096, 3476),
        previous: 2423890569493,
        proof_bytes: 22552,
    },
    Set {
        name: "demo",
        params: demo_d64,
        p: 4294962689,
        bound_squared: 2048,
        checked: [
            1563674726761101613,
            4,
            33,
            12,
            2782693479152,
            43586015314575,
            6938266263,
            104018739,
            2112,
            8594031616,
            24772,
        ],
        lifting: (12441688183056, 2897),
        previous: 1563674726743538501,
        proof_bytes: 24772,
    },
];

/// The statement ring of a set's request: degree 256 modulo `p`.
fn ring(set: &Set) -> Arc<Ring> {
    Ring::new(set.p, 256).unwrap()
}

/// The request's statement, $`Aw+t=0`$ with $`A`$ of size $`4\times8`$ and one block `w`.
fn blocks(set: &Set) -> [Block; 1] {
    [Block {
        name: "w".into(),
        length: 8,
        norm: Norm::L2Squared(set.bound_squared),
    }]
}

/// The worst case over public data: every coefficient of $`A`$ and $`t`$ is
/// $`\lfloor p/2\rfloor`$.
fn worst_case(set: &Set) -> (PolyMat, PolyVec) {
    let ring = ring(set);
    let half = Poly::new(ring.clone(), vec![set.p / 2; 256]).unwrap();
    let a = PolyMat::new(ring.clone(), 4, 8, vec![half.clone(); 32]).unwrap();
    (a, PolyVec::new(ring, vec![half; 4]).unwrap())
}

#[test]
fn each_set_passes_the_check_with_the_recorded_values() {
    for set in &SETS {
        let params = (set.params)();
        let c = params.check().unwrap();
        let got = [
            narrow(&c.q),
            c.lambda as u128,
            c.n_ex as u128,
            c.l_ext as u128,
            c.b_squared,
            c.z1_bound_squared,
            c.z3_bound_squared,
            c.z4_bound,
            c.exact_alpha_squared,
            c.approx_alpha_squared,
            c.estimated_proof_bytes as u128,
        ];
        assert_eq!(got, set.checked, "{}", set.name);
        assert_eq!(c.approx_extraction_bound, 2 * c.z4_bound, "{}", set.name);
        assert_eq!(params.prime_factors, vec![U256::from_u128(set.checked[0])]);
    }
}

#[test]
fn each_set_compiles_its_worst_case_and_the_previous_prime_does_not() {
    for set in &SETS {
        let params = (set.params)();
        let (a, t) = worst_case(set);
        lin::compile(&a, &t, &blocks(set), params.clone()).unwrap();
        // The same statement through the named builder, for its requirements.
        let q = params.check().unwrap().q;
        let mut st = jali::statement::Statement::new(ring(set));
        st.var("w", 8, Norm::L2Squared(set.bound_squared)).unwrap();
        for row in 0..4 {
            let mut eq = st.constant(t.entries()[row].clone()).unwrap();
            for j in 0..8 {
                eq = eq
                    .add(
                        &st.variable("w", j)
                            .unwrap()
                            .scale(a.get(row, j).unwrap())
                            .unwrap(),
                    )
                    .unwrap();
            }
            st.eq_mod_p(eq).unwrap();
        }
        let req = st.requirements(params.degree, q).unwrap();
        assert_eq!(
            (narrow(&req.max_integer_coefficient), req.linf_bound),
            set.lifting,
            "{}",
            set.name
        );
        assert_eq!(
            (req.m1, req.l, req.n_prime, req.alpha_squared),
            (
                params.m1,
                params.l,
                params.n_prime,
                u128::from(params.alpha_squared)
            )
        );
        // The previous prime of the gamma class, found by stepping down the class q = 1 + gamma
        // t (t odd for v2(gamma) = 2, t = 2 mod 4 for v2(gamma) = 1), passes the parameter check
        // but not the lifting check: the tool's modulus is the smallest of its class.
        let gamma = params.gamma;
        let step = gamma * if gamma % 4 == 0 { 2 } else { 4 };
        let q = narrow(&q) as u64;
        let mut previous = q - step;
        while !is_prime(previous) {
            previous -= step;
        }
        assert_eq!(previous, set.previous, "{}", set.name);
        assert_eq!(previous % 8, 5);
        let mut before = params.clone();
        before.prime_factors = vec![U256::from_u64(previous)];
        before.check().unwrap();
        assert_eq!(
            lin::compile(&a, &t, &blocks(set), before).err(),
            Some(Error::Parameter("modulus lifting bound")),
            "{}",
            set.name
        );
    }
}

/// The seeded random instance of `tools/params/rust_reference.py`: SplitMix64 from 2026, a
/// ternary $`w`$, then $`A`$ uniform in $`[-\lfloor p/2\rfloor,p-\lfloor p/2\rfloor)`$, and
/// $`t=-Aw`$.
fn instance(set: &Set) -> (PolyMat, PolyVec, BTreeMap<String, Vec<Poly>>) {
    let ring = ring(set);
    let mut rng = SplitMix64(2026);
    let w: Vec<Poly> = (0..8)
        .map(|_| Poly::new(ring.clone(), rng.ternary(256)).unwrap())
        .collect();
    let half = set.p / 2;
    let entries: Vec<Poly> = (0..32)
        .map(|_| {
            let c = (0..256)
                .map(|_| rng.below(set.p as u64) as i128 - half)
                .collect();
            Poly::new(ring.clone(), c).unwrap()
        })
        .collect();
    let mut t = Vec::new();
    for row in entries.chunks(8) {
        let mut acc = Poly::zero(ring.clone());
        for (aij, wj) in row.iter().zip(&w) {
            acc = acc.add(&aij.mul(wj).unwrap()).unwrap();
        }
        t.push(Poly::zero(ring.clone()).sub(&acc).unwrap());
    }
    let a = PolyMat::new(ring.clone(), 4, 8, entries).unwrap();
    let t = PolyVec::new(ring, t).unwrap();
    (a, t, BTreeMap::from([("w".to_string(), w)]))
}

/// Prove and verify the seeded instance with the seeds and context of the reference data, which
/// recorded the proof's length.
fn proves_the_seeded_instance(set: &Set) {
    const CONTEXT: &[u8] = b"jali tools/params/rust_reference.py";
    let (a, t, witness) = instance(set);
    let compiled = lin::compile(&a, &t, &blocks(set), (set.params)()).unwrap();
    let bytes = compiled
        .prove_bytes_with_seed([7; 32], &witness, CONTEXT, [9; 32])
        .unwrap();
    assert_eq!(bytes.len(), set.proof_bytes, "{}", set.name);
    // The seed by reference gives the same proof.
    let by_ref = compiled
        .prove_bytes_with_seed_ref([7; 32], &witness, CONTEXT, &[9; 32])
        .unwrap();
    assert_eq!(by_ref, bytes, "{}", set.name);
    compiled.verify_bytes([7; 32], &bytes, CONTEXT).unwrap();
    let mut tampered = bytes.clone();
    tampered[bytes.len() / 2] ^= 1;
    assert!(compiled.verify_bytes([7; 32], &tampered, CONTEXT).is_err());
    assert!(
        compiled
            .verify_bytes([7; 32], &bytes, b"another context")
            .is_err()
    );
}

// One test per set, so that the three proofs, about half a minute each, run in parallel.
#[test]
fn kyber1024_d64_proves_the_seeded_instance() {
    proves_the_seeded_instance(&SETS[0]);
}
#[test]
fn kyber1024_d128_proves_the_seeded_instance() {
    proves_the_seeded_instance(&SETS[1]);
}
#[test]
fn demo_d64_proves_the_seeded_instance() {
    proves_the_seeded_instance(&SETS[2]);
}

/// Each set's JSON file loads as its loader function. (`regenerate --check` of the parameter
/// tool checks that these files are the tool's output, byte for byte.)
#[cfg(feature = "serde")]
#[test]
fn each_set_is_its_json_file() {
    let files = [
        include_str!("../src/params/sets/kyber1024-d64.json"),
        include_str!("../src/params/sets/kyber1024-d128.json"),
        include_str!("../src/params/sets/demo.json"),
    ];
    for (set, crate_copy) in SETS.iter().zip(files) {
        assert_eq!(
            TboxParams::from_json(crate_copy).unwrap(),
            (set.params)(),
            "{}",
            set.name
        );
    }
}

/// The extraction sets of the shipped sets, in the order of `SETS`
/// (`tools/params/lnp_params.py extraction --shipped`).
const EXTRACTION: [fn() -> ExtractionParams; 3] = [
    kyber1024_d64_extraction,
    kyber1024_d128_extraction,
    demo_d64_extraction,
];

/// Each extraction set passes its check against its parameter set: the modulus is the one its
/// rank gives there, so the tool's derivation agrees with the crate's.
#[test]
fn each_extraction_set_checks_against_its_set() {
    for (set, extraction) in SETS.iter().zip(EXTRACTION) {
        let params = (set.params)();
        let extraction = extraction();
        extraction.check(&params).unwrap();
        assert_eq!(
            ExtractionParams::modulus_for(&params, extraction.rank).unwrap(),
            extraction.modulus,
            "{}",
            set.name
        );
        // Another rank gives another modulus, which the set does not name.
        let other = ExtractionParams {
            rank: extraction.rank - 1,
            ..extraction.clone()
        };
        assert_eq!(
            other.check(&params),
            Err(Error::Parameter("extraction modulus")),
            "{}",
            set.name
        );
    }
}

/// Each extraction set's JSON file loads as its loader function. (`extraction --shipped
/// --check` of the parameter tool checks that the file is the tool's output, byte for byte.)
#[cfg(feature = "serde")]
#[test]
fn each_extraction_set_is_its_json_file() {
    let files = [
        include_str!("../src/params/sets/kyber1024-d64.extraction.json"),
        include_str!("../src/params/sets/kyber1024-d128.extraction.json"),
        include_str!("../src/params/sets/demo.extraction.json"),
    ];
    for ((set, extraction), file) in SETS.iter().zip(EXTRACTION).zip(files) {
        assert_eq!(
            ExtractionParams::from_json(file, &(set.params)()).unwrap(),
            extraction(),
            "{}",
            set.name
        );
    }
}

/// Prove the seeded instance in the straight-line extractable mode, verify it, and return the
/// proof's length, which the tests pin: about 5.8 times the plain proof's.
fn proves_extractably(set: &Set, extraction: fn() -> ExtractionParams) -> usize {
    const CONTEXT: &[u8] = b"jali extractable mode";
    let (a, t, witness) = instance(set);
    let compiled = lin::compile(&a, &t, &blocks(set), (set.params)()).unwrap();
    let extraction = extraction();
    let bytes = compiled
        .prove_extractable_bytes_with_seed([7; 32], &extraction, &witness, CONTEXT, [9; 32])
        .unwrap();
    compiled
        .verify_extractable_bytes([7; 32], &extraction, &bytes, CONTEXT)
        .unwrap();
    // The seed by reference gives the same proof.
    let by_ref = compiled
        .prove_extractable_bytes_with_seed_ref([7; 32], &extraction, &witness, CONTEXT, &[9; 32])
        .unwrap();
    assert_eq!(by_ref, bytes, "{}", set.name);
    // A flipped bit in the toolbox part and one in the LinHC part.
    for at in [bytes.len() / 4, bytes.len() - 9] {
        let mut tampered = bytes.clone();
        tampered[at] ^= 1;
        assert!(
            compiled
                .verify_extractable_bytes([7; 32], &extraction, &tampered, CONTEXT)
                .is_err()
        );
    }
    bytes.len()
}

#[test]
fn kyber1024_d64_proves_extractably() {
    assert_eq!(proves_extractably(&SETS[0], EXTRACTION[0]), 120_964);
}
#[test]
fn kyber1024_d128_proves_extractably() {
    assert_eq!(proves_extractably(&SETS[1], EXTRACTION[1]), 127_736);
}
#[test]
fn demo_d64_proves_extractably() {
    assert_eq!(proves_extractably(&SETS[2], EXTRACTION[2]), 143_620);
}

/// Wall times of one seeded proof and its verification in both modes, the median of five runs
/// per set (`cargo test --release --test param_sets -- --ignored --nocapture`). The machine's
/// load moves these figures: record it with them.
#[test]
#[ignore = "timing output"]
fn timings_of_both_modes() {
    use std::time::Instant;
    for (set, extraction) in SETS.iter().zip(EXTRACTION) {
        let (a, t, witness) = instance(set);
        let compiled = lin::compile(&a, &t, &blocks(set), (set.params)()).unwrap();
        let extraction = extraction();
        let median = |mut f: Box<dyn FnMut()>| -> f64 {
            let mut times: Vec<f64> = (0..5)
                .map(|_| {
                    let start = Instant::now();
                    f();
                    start.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            times.sort_by(f64::total_cmp);
            times[2]
        };
        let plain = compiled
            .prove_bytes_with_seed([7; 32], &witness, b"timing", [9; 32])
            .unwrap();
        let bytes = compiled
            .prove_extractable_bytes_with_seed([7; 32], &extraction, &witness, b"timing", [9; 32])
            .unwrap();
        let prove_plain = median(Box::new(|| {
            compiled
                .prove_bytes_with_seed([7; 32], &witness, b"timing", [9; 32])
                .unwrap();
        }));
        let verify_plain = median(Box::new(|| {
            compiled.verify_bytes([7; 32], &plain, b"timing").unwrap();
        }));
        let prove_ext = median(Box::new(|| {
            compiled
                .prove_extractable_bytes_with_seed(
                    [7; 32],
                    &extraction,
                    &witness,
                    b"timing",
                    [9; 32],
                )
                .unwrap();
        }));
        let verify_ext = median(Box::new(|| {
            compiled
                .verify_extractable_bytes([7; 32], &extraction, &bytes, b"timing")
                .unwrap();
        }));
        eprintln!(
            "{}: plain {} B, prove {prove_plain:.1} ms, verify {verify_plain:.1} ms | \
             extractable {} B, prove {prove_ext:.1} ms, verify {verify_ext:.1} ms",
            set.name,
            plain.len(),
            bytes.len()
        );
    }
}
