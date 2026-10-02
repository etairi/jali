use jali::{
    Error,
    abdlop::Abdlop,
    math::{Poly, PolyVec, U256},
    params::toy_d64,
    statement::Requirements,
};

mod common;
use common::params::{FACTORS_240, even_divisors, fit_wide, power_plus};

#[test]
fn compressed_opening_proves_verifies_and_rejects_tampering() {
    let scheme = Abdlop::new([9; 32], toy_d64()).unwrap();
    let ring = scheme.ring().clone();
    let s1 = PolyVec::new(
        ring.clone(),
        (0..scheme.bounded_len())
            .map(|i| {
                Poly::new(
                    ring.clone(),
                    (0..64).map(|j| ((i + j) % 3) as i128 - 1).collect(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let m = PolyVec::new(
        ring.clone(),
        vec![
            Poly::constant(ring.clone(), 19),
            Poly::constant(ring.clone(), -7),
        ],
    )
    .unwrap();
    let (commitment, opening) = scheme.commit_with_seed(s1, m, [5; 32]).unwrap();
    let proof = scheme
        .prove_with_seed(&commitment, &opening, b"application-context", [8; 32])
        .unwrap();
    scheme
        .verify(&commitment, &proof, b"application-context")
        .unwrap();
    assert_eq!(
        proof,
        scheme
            .prove_with_seed(&commitment, &opening, b"application-context", [8; 32])
            .unwrap()
    );
    let bytes = scheme.encode_proof(&proof).unwrap();
    assert_eq!(scheme.decode_proof(&bytes).unwrap(), proof);
    assert!(
        scheme
            .verify(&commitment, &proof, b"different context")
            .is_err()
    );
    let other = Abdlop::new([10; 32], toy_d64()).unwrap();
    assert!(
        other
            .verify(&commitment, &proof, b"application-context")
            .is_err()
    );
    let mut changed_params = toy_d64();
    changed_params.estimator.push_str("; changed provenance");
    assert!(
        Abdlop::new([9; 32], changed_params)
            .unwrap()
            .verify(&commitment, &proof, b"application-context")
            .is_err()
    );
    let mut bad = proof.clone();
    bad.challenge
        .set_coefficient(0, bad.challenge.coefficient_i128(0).unwrap() + 1)
        .unwrap();
    assert!(
        scheme
            .verify(&commitment, &bad, b"application-context")
            .is_err()
    );
    let mut bad = proof.clone();
    let mut z = bad.z21.entries().to_vec();
    z[0].set_coefficient(0, 1 << 30).unwrap();
    bad.z21 = PolyVec::new(ring.clone(), z).unwrap();
    assert!(
        scheme
            .verify(&commitment, &bad, b"application-context")
            .is_err()
    );
    let mut bad = proof.clone();
    let mut h = bad.hint.entries().to_vec();
    h[0].set_coefficient(0, 1 << 30).unwrap();
    bad.hint = PolyVec::new(ring.clone(), h).unwrap();
    assert!(
        scheme
            .verify(&commitment, &bad, b"application-context")
            .is_err()
    );
    let mut changed = commitment.clone();
    let mut values = changed.t_b.entries().to_vec();
    values[0] = Poly::constant(ring, 123);
    changed.t_b = PolyVec::new(scheme.ring().clone(), values).unwrap();
    assert!(
        scheme
            .verify(&changed, &proof, b"application-context")
            .is_err()
    );
    assert!(matches!(
        scheme.prove_with_seed(&changed, &opening, b"application-context", [8; 32]),
        Err(Error::Witness)
    ));
    for end in [0, 1, bytes.len() / 2, bytes.len() - 1] {
        assert!(scheme.decode_proof(&bytes[..end]).is_err());
    }
    let mut trailing = bytes;
    trailing.push(0);
    assert!(scheme.decode_proof(&trailing).is_err());
}

#[test]
fn uncompressed_commitment_coefficients_are_canonical_mod_q() {
    let mut params = toy_d64();
    params.d_bits = 0;
    let scheme = Abdlop::new([131; 32], params).unwrap();
    let ring = scheme.ring().clone();
    let (commitment, opening) = scheme
        .commit_with_seed(
            PolyVec::zero(ring.clone(), scheme.bounded_len()),
            PolyVec::zero(ring, scheme.message_len()),
            [132; 32],
        )
        .unwrap();
    assert!(
        commitment
            .t_a
            .entries()
            .iter()
            .any(|p| common::values(p).iter().any(|x| *x < 0))
    );
    let proof = scheme
        .prove_with_seed(&commitment, &opening, b"D=0", [133; 32])
        .unwrap();
    scheme.verify(&commitment, &proof, b"D=0").unwrap();
}

/// A ternary bounded witness and a message, over `toy_d64`.
fn witness(scheme: &Abdlop) -> (PolyVec, PolyVec) {
    let ring = scheme.ring().clone();
    let s1 = PolyVec::new(
        ring.clone(),
        (0..scheme.bounded_len())
            .map(|i| {
                Poly::new(
                    ring.clone(),
                    (0..64).map(|j| ((i * 7 + j * j) % 3) as i128 - 1).collect(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let m = PolyVec::new(
        ring.clone(),
        vec![Poly::constant(ring.clone(), 19), Poly::constant(ring, -7)],
    )
    .unwrap();
    (s1, m)
}
fn difference(a: &PolyVec, b: &PolyVec) -> Vec<Poly> {
    a.entries()
        .iter()
        .zip(b.entries())
        .map(|(a, b)| a.sub(b).unwrap())
        .collect()
}

#[test]
fn two_opening_proofs_under_two_contexts_from_one_seed_never_reveal_s1() {
    // Were the masks a function of the seed alone, two proofs of one commitment under two
    // contexts would share y1, and whenever both accepted at the same attempt,
    // z1 - z1' = (c - c') s1 would give the witness once c - c' is invertible.
    let scheme = Abdlop::new([9; 32], toy_d64()).unwrap();
    let (s1, m) = witness(&scheme);
    let mut leaks = 0;
    for byte in 0..40u8 {
        let seed = [byte; 32];
        let (commitment, opening) = scheme
            .commit_with_seed(s1.clone(), m.clone(), seed)
            .unwrap();
        let a = scheme
            .prove_with_seed(&commitment, &opening, b"context A", seed)
            .unwrap();
        let b = scheme
            .prove_with_seed(&commitment, &opening, b"context B", seed)
            .unwrap();
        scheme.verify(&commitment, &a, b"context A").unwrap();
        scheme.verify(&commitment, &b, b"context B").unwrap();
        assert_ne!(a.challenge, b.challenge);
        let dc = a.challenge.sub(&b.challenge).unwrap();
        let scaled: Vec<_> = s1.entries().iter().map(|s| dc.mul(s).unwrap()).collect();
        if difference(&a.z1, &b.z1) == scaled {
            leaks += 1;
        }
    }
    assert_eq!(leaks, 0, "{leaks} of 40 seeds give z1 - z1' = (c - c') s1");
}

#[test]
fn two_commitments_from_one_seed_share_randomness_only_when_their_values_agree() {
    let scheme = Abdlop::new([9; 32], toy_d64()).unwrap();
    let ring = scheme.ring().clone();
    let (s1, m) = witness(&scheme);
    let other_m = PolyVec::new(
        ring.clone(),
        vec![Poly::constant(ring.clone(), 20), Poly::constant(ring, -7)],
    )
    .unwrap();
    let seed = [21; 32];
    let (c, _) = scheme
        .commit_with_seed(s1.clone(), m.clone(), seed)
        .unwrap();
    let (other, _) = scheme
        .commit_with_seed(s1.clone(), other_m.clone(), seed)
        .unwrap();
    // With one s2 for both, t_B - t_B' = m - m', and equal s1 would give equal t_A.
    assert_ne!(difference(&c.t_b, &other.t_b), difference(&m, &other_m));
    assert_ne!(c.t_a, other.t_a);
    // Identical inputs give the identical commitment: deterministic, and so linkable.
    assert_eq!(scheme.commit_with_seed(s1, m, seed).unwrap().0, c);
}

#[test]
fn opening_proofs_are_deterministic_and_depend_on_the_context() {
    let scheme = Abdlop::new([9; 32], toy_d64()).unwrap();
    let (s1, m) = witness(&scheme);
    let seed = [22; 32];
    let prove = |context: &[u8]| {
        let (commitment, opening) = scheme
            .commit_with_seed(s1.clone(), m.clone(), seed)
            .unwrap();
        let proof = scheme
            .prove_with_seed(&commitment, &opening, context, seed)
            .unwrap();
        (commitment, scheme.encode_proof(&proof).unwrap())
    };
    let (commitment, bytes) = prove(b"context A");
    assert_eq!(prove(b"context A"), (commitment.clone(), bytes.clone()));
    let (same_commitment, other_bytes) = prove(b"context B");
    assert_eq!(same_commitment, commitment);
    assert_ne!(other_bytes, bytes);
    let (_, empty) = prove(b"");
    let proof = scheme.decode_proof(&empty).unwrap();
    scheme.verify(&commitment, &proof, b"").unwrap();
    assert!(scheme.verify(&commitment, &proof, b"context A").is_err());
}

#[test]
fn rng_commitments_and_opening_proofs_are_fresh_and_bound_to_their_context() {
    // `commit` and `prove` draw a fresh seed from the RNG per call, so repeating a call with
    // identical inputs gives an unrelated output.
    let scheme = Abdlop::new([9; 32], toy_d64()).unwrap();
    let (s1, m) = witness(&scheme);
    let mut rng = common::SeededRng::new(b"abdlop");
    let (commitment, opening) = scheme.commit(s1.clone(), m.clone(), &mut rng).unwrap();
    let (again, _) = scheme.commit(s1, m, &mut rng).unwrap();
    assert_ne!(again.t_a, commitment.t_a);
    let prove = |rng: &mut common::SeededRng| {
        scheme
            .prove(&commitment, &opening, b"context A", rng)
            .unwrap()
    };
    let proof = prove(&mut rng);
    scheme.verify(&commitment, &proof, b"context A").unwrap();
    for other in [b"context B".as_slice(), b""] {
        assert!(scheme.verify(&commitment, &proof, other).is_err());
    }
    let second = prove(&mut rng);
    scheme.verify(&commitment, &second, b"context A").unwrap();
    assert_ne!(second.z1, proof.z1);
}

#[test]
fn opening_proofs_at_241_bits_verify_and_reject_saturated_norms() {
    // q = 2^240 + 325 with test-only parameters (synthetic MLWE metadata) for the toolbox shape
    // of `tests/toolbox.rs`. A response coefficient near q/2 has a squared norm above 2^256,
    // which saturates at U256::MAX: the verifier rejects the proof as invalid instead of failing
    // with an arithmetic error, and the encoder refuses it, since no Gaussian code holds it.
    const CTX: &[u8] = b"jali-test/abdlop-wide";
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
        "abdlop-q240-test-only",
        q,
        &even_divisors(&q, &FACTORS_240),
        64,
    )
    .unwrap();
    let scheme = Abdlop::new([11; 32], params).unwrap();
    let ring = scheme.ring().clone();
    assert_eq!(ring.modulus(), q);
    let s1 = PolyVec::new(
        ring.clone(),
        (0..scheme.bounded_len())
            .map(|i| {
                Poly::new(
                    ring.clone(),
                    (0..64).map(|j| ((i + 2 * j) % 3) as i128 - 1).collect(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    // Messages are arbitrary ring elements: one near q/2.
    let m = PolyVec::new(
        ring.clone(),
        vec![
            Poly::constant_u256(ring.clone(), &q.shr_vartime(1)),
            Poly::constant(ring.clone(), -7),
        ],
    )
    .unwrap();
    let (commitment, opening) = scheme.commit_with_seed(s1, m, [12; 32]).unwrap();
    let proof = scheme
        .prove_with_seed(&commitment, &opening, CTX, [13; 32])
        .unwrap();
    scheme.verify(&commitment, &proof, CTX).unwrap();
    assert!(
        scheme
            .verify(&commitment, &proof, b"other context")
            .is_err()
    );
    let bytes = scheme.encode_proof(&proof).unwrap();
    assert_eq!(scheme.decode_proof(&bytes).unwrap(), proof);
    // One coefficient of z1, then of z21, replaced by floor(q/2) or by 2^127, which does not
    // fit the Gaussian code's i128 either.
    let replaced = |v: &PolyVec, value: U256| {
        let mut entries = v.entries().to_vec();
        let mut coefficients = entries[0].coefficients().to_vec();
        coefficients[5] = value;
        entries[0] = Poly::from_u256(ring.clone(), coefficients).unwrap();
        PolyVec::new(ring.clone(), entries).unwrap()
    };
    for value in [q.shr_vartime(1), U256::ONE.shl_vartime(127)] {
        let mut bad = proof.clone();
        bad.z1 = replaced(&proof.z1, value);
        if value == q.shr_vartime(1) {
            assert_eq!(bad.z1.norm_squared(), Ok(U256::MAX));
        }
        assert_eq!(
            scheme.verify(&commitment, &bad, CTX),
            Err(Error::InvalidProof)
        );
        assert_eq!(scheme.encode_proof(&bad).err(), Some(Error::Encoding));
        let mut bad = proof.clone();
        bad.z21 = replaced(&proof.z21, value);
        assert_eq!(
            scheme.verify(&commitment, &bad, CTX),
            Err(Error::InvalidProof)
        );
        assert_eq!(scheme.encode_proof(&bad).err(), Some(Error::Encoding));
    }
}
