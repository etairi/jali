//! The straight-line extractable mode through the public API: the prover and verifier of
//! `lnp`, the strict wire format, and the extraction set's checks and JSON form.
use jali::{
    Error,
    lnp::{AffineBlock, L2Block, Prover, Statement, Verifier},
    math::{Poly, PolyMat, PolyVec, Ring},
    params::{ExtractionParams, OracleModel, TboxParams, toy_d64},
};
use std::sync::Arc;

mod common;

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

/// The statement of `tests/toolbox.rs`: binary rows 0 and 1, Euclidean blocks on rows 2, 3
/// and on row 4, and a range block on both messages.
fn statement(ring: &Arc<Ring>) -> Statement {
    let select_s = |indices: &[usize]| AffineBlock {
        rows: indices.len(),
        s: Some(selector(ring, 10, indices)),
        m: None,
        offset: None,
    };
    Statement {
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
            m: Some(selector(ring, 2, &[0, 1])),
            offset: None,
        }),
    }
}

fn witness(ring: &Arc<Ring>) -> (PolyVec, PolyVec) {
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
        vec![
            Poly::constant(ring.clone(), 2),
            Poly::constant(ring.clone(), -3),
        ],
    )
    .unwrap();
    (s1, m)
}

/// An extraction set for `params` with `rank`: the modulus that rank gives, and a placeholder
/// estimate for this test shape.
fn extraction(params: &TboxParams, rank: usize) -> ExtractionParams {
    ExtractionParams {
        model: OracleModel::Rom,
        rank,
        modulus: ExtractionParams::modulus_for(params, rank).unwrap(),
        mlwe_delta: 1.004,
        estimator: "test fixture, no estimate".into(),
    }
}

fn parties() -> (Prover, Verifier, ExtractionParams, Arc<TboxParams>) {
    let params = Arc::new(toy_d64());
    let mut prover = Prover::new([1; 32], params.clone()).unwrap();
    let ring = prover.ring().clone();
    prover.set_statement(statement(&ring)).unwrap();
    let (s1, m) = witness(&ring);
    prover.set_witness(&s1, &m).unwrap();
    let mut verifier = Verifier::new([1; 32], params.clone()).unwrap();
    verifier.set_statement(statement(&ring)).unwrap();
    let extraction = extraction(&params, 36);
    (prover, verifier, extraction, params)
}

#[test]
fn extractable_proofs_round_trip_through_the_public_api() {
    let (prover, verifier, extraction, _) = parties();
    let proof = prover
        .prove_extractable_with_seed(&extraction, b"ctx", [3; 32])
        .unwrap();
    verifier
        .verify_extractable(&extraction, &proof, b"ctx")
        .unwrap();
    let bytes = prover
        .prove_extractable_bytes_with_seed(&extraction, b"ctx", [3; 32])
        .unwrap();
    verifier
        .verify_extractable_bytes(&extraction, &bytes, b"ctx")
        .unwrap();
    assert!(
        verifier
            .verify_extractable_bytes(&extraction, &bytes, b"other")
            .is_err()
    );
    // Fresh randomness: another proof, which verifies too.
    let mut rng = common::SeededRng::new(b"extractable");
    let fresh = prover
        .prove_extractable(&extraction, b"ctx", &mut rng)
        .unwrap();
    assert_ne!(fresh, proof);
    verifier
        .verify_extractable(&extraction, &fresh, b"ctx")
        .unwrap();
    // A plain proof is shorter and is not an extractable one, and the other way round.
    let plain = prover.prove_bytes_with_seed(b"ctx", [3; 32]).unwrap();
    assert!(plain.len() < bytes.len());
    assert!(
        verifier
            .verify_extractable_bytes(&extraction, &plain, b"ctx")
            .is_err()
    );
    assert!(verifier.verify_bytes(&bytes, b"ctx").is_err());
}

#[test]
fn the_extractable_decoder_is_strict() {
    let (prover, verifier, ext, params) = parties();
    let bytes = prover
        .prove_extractable_bytes_with_seed(&ext, b"ctx", [4; 32])
        .unwrap();
    let mut longer = bytes.clone();
    longer.push(0);
    for bad in [&bytes[..bytes.len() - 1], &longer[..]] {
        assert!(
            verifier
                .verify_extractable_bytes(&ext, bad, b"ctx")
                .is_err()
        );
    }
    // Another extraction set reads other lengths and another modulus.
    let other = extraction(&params, 20);
    assert!(
        verifier
            .verify_extractable_bytes(&other, &bytes, b"ctx")
            .is_err()
    );
}

#[test]
fn the_extraction_set_is_checked_against_its_parameters() {
    let params = toy_d64();
    let good = extraction(&params, 36);
    good.check(&params).unwrap();
    let refuse = |e: ExtractionParams, why: &'static str| {
        assert_eq!(e.check(&params), Err(Error::Parameter(why)), "{e:?}");
    };
    refuse(
        ExtractionParams {
            modulus: good.modulus + 2,
            ..good.clone()
        },
        "extraction modulus",
    );
    refuse(
        ExtractionParams {
            rank: 0,
            ..good.clone()
        },
        "extraction rank",
    );
    refuse(
        ExtractionParams {
            rank: 10_000,
            ..good.clone()
        },
        "extraction rank",
    );
    refuse(
        ExtractionParams {
            mlwe_delta: 1.0045,
            ..good.clone()
        },
        "extraction MLWE estimate",
    );
    refuse(
        ExtractionParams {
            estimator: String::new(),
            ..good.clone()
        },
        "extraction estimator",
    );
}

#[cfg(feature = "serde")]
#[test]
fn the_extraction_set_reads_strict_json() {
    let params = toy_d64();
    let good = extraction(&params, 36);
    let json = serde_json::to_string(&good).unwrap();
    assert!(json.contains("\"model\":\"rom\""), "{json}");
    assert_eq!(ExtractionParams::from_json(&json, &params).unwrap(), good);
    let unknown = json.replace("\"rank\"", "\"extra\":1,\"rank\"");
    assert_eq!(
        ExtractionParams::from_json(&unknown, &params),
        Err(Error::Encoding)
    );
    let qrom = json.replace("\"rom\"", "\"qrom\"");
    assert_eq!(
        ExtractionParams::from_json(&qrom, &params),
        Err(Error::Encoding)
    );
    let wrong = json.replace(
        &format!("\"modulus\":{}", good.modulus),
        &format!("\"modulus\":{}", good.modulus + 2),
    );
    assert_eq!(
        ExtractionParams::from_json(&wrong, &params),
        Err(Error::Parameter("extraction modulus"))
    );
}
