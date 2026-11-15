//! The straight-line extractable mode (Katsumata's transform): round trips, its separation from
//! the plain mode, tampering, and straight-line extraction of the committed witness from one
//! proof with a trapdoored LinHC key.
use super::*;
use crate::{
    abdlop::{add, part, scale, scale_scalar, sub},
    linhc::{LinhcOpening, decrypt},
    params::{ExtractionParams, OracleModel, toy_d64},
    rand::challenge,
};

/// An extraction set for `toy_d64` with `rank`, the modulus that rank gives, and a placeholder
/// estimate: the toy set is a test shape, its own MLWE estimate is not a security claim either.
fn extraction(rank: usize) -> ExtractionParams {
    ExtractionParams {
        model: OracleModel::Rom,
        rank,
        modulus: ExtractionParams::modulus_for(&toy_d64(), rank).unwrap(),
        mlwe_delta: 1.004,
        estimator: "test fixture, no estimate".into(),
    }
}

fn fixture() -> (Abdlop, Statement, PolyVec, PolyVec) {
    let scheme = Abdlop::new([1; 32], toy_d64()).unwrap();
    let (statement, s1, m) = super::restart_tests::selector_statement(&scheme);
    (scheme, statement, s1, m)
}

#[test]
fn proofs_verify_and_are_separate_from_the_plain_mode() {
    let (scheme, statement, s1, m) = fixture();
    let ext = extraction(36);
    ext.check(&toy_d64()).unwrap();
    let proof =
        prove_extractable_with_seed(&scheme, &ext, &statement, &s1, &m, b"ctx", [7; 32]).unwrap();
    verify_extractable(&scheme, &ext, &statement, &proof, b"ctx").unwrap();
    // Deterministic for a seed, and different for another.
    let again =
        prove_extractable_with_seed(&scheme, &ext, &statement, &s1, &m, b"ctx", [7; 32]).unwrap();
    assert_eq!(again, proof);
    let other =
        prove_extractable_with_seed(&scheme, &ext, &statement, &s1, &m, b"ctx", [8; 32]).unwrap();
    assert_ne!(other, proof);
    // The inner toolbox proof does not verify in the plain mode, and a plain proof is not an
    // extractable one.
    assert_eq!(
        verify(&scheme, &statement, &proof.proof, b"ctx"),
        Err(Error::InvalidProof)
    );
    let plain = prove_with_seed(&scheme, &statement, &s1, &m, b"ctx", [7; 32]).unwrap();
    assert_ne!(plain, proof.proof);
    // The context and the extraction set are bound.
    assert_eq!(
        verify_extractable(&scheme, &ext, &statement, &proof, b"other"),
        Err(Error::InvalidProof)
    );
    let mut renamed = ext.clone();
    renamed.estimator = "another estimate".into();
    assert_eq!(
        verify_extractable(&scheme, &renamed, &statement, &proof, b"ctx"),
        Err(Error::InvalidProof)
    );
    // A set whose modulus does not belong to its rank is refused.
    let mut wrong = ext.clone();
    wrong.rank = 35;
    assert_eq!(
        verify_extractable(&scheme, &wrong, &statement, &proof, b"ctx"),
        Err(Error::Parameter("extraction modulus"))
    );
}

#[test]
fn tampered_proofs_are_rejected() {
    let (scheme, statement, s1, m) = fixture();
    let ext = extraction(36);
    let proof =
        prove_extractable_with_seed(&scheme, &ext, &statement, &s1, &m, b"ctx", [9; 32]).unwrap();
    let bump = |v: &PolyVec, i: usize| -> PolyVec {
        let mut entries = v.entries().to_vec();
        let x = entries[i].coefficient_i128(3).unwrap();
        entries[i].set_coefficient(3, x + 1).unwrap();
        PolyVec::new(v.ring().clone(), entries).unwrap()
    };
    let cases: Vec<(&str, ExtractableProof)> = vec![
        (
            "t1",
            ExtractableProof {
                t1: bump(&proof.t1, 0),
                ..proof.clone()
            },
        ),
        (
            "t2",
            ExtractableProof {
                t2: bump(&proof.t2, 5),
                ..proof.clone()
            },
        ),
        (
            "z1",
            ExtractableProof {
                z1: bump(&proof.z1, 1),
                ..proof.clone()
            },
        ),
        (
            "z2",
            ExtractableProof {
                z2: bump(&proof.z2, 2),
                ..proof.clone()
            },
        ),
        (
            "z3",
            ExtractableProof {
                z3: bump(&proof.z3, 3),
                ..proof.clone()
            },
        ),
        ("opening z1", {
            let mut p = proof.clone();
            let opening = &mut p.proof.evaluation.quadratic.opening;
            opening.z1 = bump(&opening.z1, 0);
            p
        }),
    ];
    for (name, tampered) in cases {
        assert_eq!(
            verify_extractable(&scheme, &ext, &statement, &tampered, b"ctx"),
            Err(Error::InvalidProof),
            "{name}"
        );
    }
    // Wrong shapes are refused before any arithmetic.
    let short = ExtractableProof {
        z1: PolyVec::new(proof.z1.ring().clone(), proof.z1.entries()[1..].to_vec()).unwrap(),
        ..proof.clone()
    };
    assert_eq!(
        verify_extractable(&scheme, &ext, &statement, &short, b"ctx"),
        Err(Error::Dimension)
    );
}

/// $`a^k`$ in the proof ring, by square and multiply.
fn power(a: &Poly, k: &U256) -> Poly {
    let mut out = Poly::constant(a.ring().clone(), 1);
    for i in (0..k.bits()).rev() {
        out = out.mul(&out).unwrap();
        if k.bit(i).into() {
            out = out.mul(a).unwrap();
        }
    }
    out
}

/// $`a^{-1}=a^{q^{d/2}-2}`$: for a prime $`q\equiv5\pmod8`$, $`X^d+1`$ is the product of two
/// irreducible factors of degree $`d/2`$, so the units form a group of exponent $`q^{d/2}-1`$.
/// The exponent's base-$`q`$ digits are $`q-2`$ and then $`d/2-1`$ digits $`q-1`$.
fn invert(a: &Poly) -> Poly {
    let q = a.ring().modulus();
    let half = a.ring().degree() / 2;
    let mut frobenius = a.clone();
    let mut out = power(a, &q.wrapping_sub(&U256::from_u8(2)));
    for _ in 1..half {
        frobenius = power(&frobenius, &q);
        out = out
            .mul(&power(&frobenius, &q.wrapping_sub(&U256::ONE)))
            .unwrap();
    }
    out
}

#[test]
fn ring_inversion_inverts_challenge_differences() {
    let (scheme, ..) = fixture();
    let ring = scheme.ring().clone();
    let draw = |i: u32| {
        challenge(
            &mut AesPrg::new(&[5; 32], domain(1, i)),
            ring.clone(),
            scheme.checked.omega,
            scheme.checked.eta,
        )
        .unwrap()
    };
    let difference = draw(0).sub(&draw(1)).unwrap();
    assert_eq!(
        difference.mul(&invert(&difference)).unwrap(),
        Poly::constant(ring, 1)
    );
}

/// The polynomials of `values` over the proof ring.
fn polys(scheme: &Abdlop, values: &[Vec<i128>]) -> PolyVec {
    PolyVec::new(
        scheme.ring().clone(),
        values
            .iter()
            .map(|v| Poly::new(scheme.ring().clone(), v.clone()).unwrap())
            .collect(),
    )
    .unwrap()
}

#[test]
fn a_trapdoor_extracts_the_committed_witness_from_one_proof() {
    let (scheme, statement, s1, m) = fixture();
    let ext = extraction(8);
    let (params, checked) = ext.linhc(&scheme.parameters, &scheme.checked).unwrap();
    let ring_x = params.ring().unwrap();
    let (key, trapdoor) = LinhcKey::with_trapdoor(&ring_x, &params, &[9; 32]).unwrap();
    let proof = prove_extractable_seeded(
        &scheme,
        &ext,
        &statement,
        &s1,
        &m,
        b"extract",
        &Zeroizing::new([3; 32]),
        Some(&key),
    )
    .unwrap();
    verify_extractable_with_key(&scheme, &ext, &statement, &proof, b"extract", Some(&key)).unwrap();

    // What the extractor sees: one proof. It recomputes the LinHC mask commitment and the
    // opening's first message w1 exactly as the verifier does.
    let opening = &proof.proof.evaluation.quadratic.opening;
    let c = &opening.challenge;
    let commitment = LinhcCommitment {
        t1: proof.t1.clone(),
        t2: proof.t2.clone(),
    };
    let linhc_opening = LinhcOpening {
        z: [proof.z1.clone(), proof.z2.clone(), proof.z3.clone()],
    };
    let (w1x, w2x) = linhc::recompute(
        &key,
        &params,
        &checked,
        &linhc::lift_poly(&ring_x, c).unwrap(),
        &commitment,
        &linhc::message(&ring_x, &opening.z1, &opening.z21).unwrap(),
        &linhc_opening,
    )
    .unwrap();
    let decrypt_at = |challenge: &Poly| -> Vec<Vec<i128>> {
        let cx = linhc::lift_poly(&ring_x, challenge).unwrap();
        let v1 = add(&scale(&commitment.t1, &cx).unwrap(), &w1x).unwrap();
        let v2 = add(&scale(&commitment.t2, &cx).unwrap(), &w2x).unwrap();
        decrypt(&trapdoor, params.plaintext, &v1, &v2).unwrap()
    };
    let n1 = scheme.bounded_len();
    // At the proof's own challenge the decryption is the response itself.
    let own = decrypt_at(c);
    assert_eq!(polys(&scheme, &own[..n1]), opening.z1);
    assert_eq!(polys(&scheme, &own[n1..]), opening.z21);
    let rounded = |challenge: &Poly| -> PolyVec {
        scale(
            &proof.proof.commitment.t_a,
            &challenge.scale_u256(&scheme.rounding_power()).unwrap(),
        )
        .unwrap()
    };
    let r = sub(
        &add(
            &scheme.a1.mul(&opening.z1).unwrap(),
            &scheme.a2.mul(&opening.z21).unwrap(),
        )
        .unwrap(),
        &rounded(c),
    )
    .unwrap();
    let w1 = scheme.use_hints(&opening.hint, &r).unwrap();
    let z22 = sub(&scale_scalar(&w1, &scheme.gamma()).unwrap(), &r).unwrap();

    // Straight-line extraction: fresh challenges, no rewinding. A candidate counts when its
    // decrypted response passes the verifier's predicate for the fixed first message: the
    // norm bounds, with z22 rebuilt from w1.
    let mut found = Vec::new();
    for i in 0..64u32 {
        let candidate = challenge(
            &mut AesPrg::new(&[77; 32], domain(2, i)),
            scheme.ring().clone(),
            scheme.checked.omega,
            scheme.checked.eta,
        )
        .unwrap();
        if candidate == *c {
            continue;
        }
        let z = decrypt_at(&candidate);
        let (zt1, zt21) = (polys(&scheme, &z[..n1]), polys(&scheme, &z[n1..]));
        let rt = sub(
            &add(
                &scheme.a1.mul(&zt1).unwrap(),
                &scheme.a2.mul(&zt21).unwrap(),
            )
            .unwrap(),
            &rounded(&candidate),
        )
        .unwrap();
        let zt22 = sub(&scale_scalar(&w1, &scheme.gamma()).unwrap(), &rt).unwrap();
        if scheme.z1_within_bound(&zt1).unwrap() && scheme.joint_within_bound(&zt21, &zt22).unwrap()
        {
            found.push((candidate, zt1, zt21, zt22));
        }
        if found.len() == 2 {
            break;
        }
    }
    assert_eq!(found.len(), 2, "two further accepting transcripts");

    // Special soundness on each pair (c, z), (c~, z~): s = (c - c~)^-1 (z - z~).
    let mut openings = Vec::new();
    for (candidate, zt1, zt21, zt22) in &found {
        let inverse = invert(&c.sub(candidate).unwrap());
        let s1_bar = scale(&sub(&opening.z1, zt1).unwrap(), &inverse).unwrap();
        let s21_bar = scale(&sub(&opening.z21, zt21).unwrap(), &inverse).unwrap();
        let s22_bar = scale(&sub(&z22, zt22).unwrap(), &inverse).unwrap();
        // An opening of the compressed commitment: A1 s1 + A2 s21 + s22 = 2^D t_A1.
        assert_eq!(
            add(
                &add(
                    &scheme.a1.mul(&s1_bar).unwrap(),
                    &scheme.a2.mul(&s21_bar).unwrap()
                )
                .unwrap(),
                &s22_bar
            )
            .unwrap(),
            scale_scalar(&proof.proof.commitment.t_a, &scheme.rounding_power()).unwrap()
        );
        openings.push((s1_bar, s21_bar));
    }
    // Both pairs give the same opening, which is the committed witness: the original bounded
    // witness first, and the messages t_B - B s21.
    assert_eq!(openings[0], openings[1]);
    let (s1_bar, s21_bar) = &openings[0];
    assert_eq!(&part(s1_bar, 0, s1.len()).unwrap(), &s1);
    let messages = sub(
        &part(&proof.proof.commitment.t_b, 0, m.len()).unwrap(),
        &scheme.b.mul(s21_bar).unwrap(),
    )
    .unwrap();
    assert_eq!(messages, m);
}
