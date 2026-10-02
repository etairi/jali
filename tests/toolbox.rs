use jali::{
    Error,
    abdlop::Abdlop,
    lnp::{AffineBlock, L2Block, Prover, Statement, Verifier},
    math::{Poly, PolyMat, PolyVec, Ring, SparsePolyMat, SparsePolyVec, U256},
    params::{TboxParams, toy_d64},
    quad::{self, QuadEq},
    statement::Requirements,
    tbox,
};
use std::sync::Arc;

mod common;
use common::{
    SplitMix64,
    params::{
        FACTORS_128, FACTORS_240, FACTORS_256, even_divisors, fit_upward, fit_wide, power_minus,
        power_plus,
    },
    ring::poly,
};

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
#[test]
fn full_toolbox_and_stateful_api_reject_false_blocks_and_altered_responses() {
    let params = toy_d64();
    let scheme = Abdlop::new([1; 32], params.clone()).unwrap();
    let ring = scheme.ring().clone();
    let statement = statement(&ring);
    let (s1, m) = witness(&ring);
    let context = b"toolbox-test";
    let proof = tbox::prove_with_seed(&scheme, &statement, &s1, &m, context, [2; 32]).unwrap();
    tbox::verify(&scheme, &statement, &proof, context).unwrap();
    let bytes = jali::codec::proof::encode(&scheme, &proof).unwrap();
    assert_eq!(jali::codec::proof::decode(&scheme, &bytes).unwrap(), proof);
    // Gaussian quotients and hints are variable length, so the encoded length varies with the
    // seed; the test checks a range below the set's size estimate of 17,896 bytes.
    assert!(
        bytes.len() <= 17800 && bytes.len() > 16000,
        "proof bytes: {}",
        bytes.len()
    );
    for n in [0, 1, bytes.len() / 2, bytes.len() - 1] {
        assert!(jali::codec::proof::decode(&scheme, &bytes[..n]).is_err());
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(jali::codec::proof::decode(&scheme, &trailing).is_err());
    for coefficient in [0, 17] {
        let mut bad = proof.clone();
        let mut z = bad.z_approx.entries().to_vec();
        let changed = z[0].coefficient_i128(coefficient).unwrap() + 1;
        z[0].set_coefficient(coefficient, changed).unwrap();
        bad.z_approx = PolyVec::new(ring.clone(), z).unwrap();
        assert!(tbox::verify(&scheme, &statement, &bad, context).is_err());
    }
    let mut bad = proof.clone();
    let mut z = bad.z_exact.entries().to_vec();
    z[0].set_coefficient(0, 1 << 30).unwrap();
    bad.z_exact = PolyVec::new(ring.clone(), z).unwrap();
    assert!(tbox::verify(&scheme, &statement, &bad, context).is_err());
    let mut false_statement = statement.clone();
    false_statement.binary.as_mut().unwrap().offset =
        Some(PolyVec::new(ring.clone(), vec![Poly::constant(ring.clone(), 2); 2]).unwrap());
    assert!(tbox::prove_with_seed(&scheme, &false_statement, &s1, &m, context, [2; 32]).is_err());
    assert!(tbox::verify(&scheme, &false_statement, &proof, context).is_err());
    let mut prover = Prover::new([1; 32], Arc::new(params.clone())).unwrap();
    prover.set_statement(statement.clone()).unwrap();
    prover.set_witness(&s1, &m).unwrap();
    assert_eq!(proof, prover.prove_with_seed(context, [2; 32]).unwrap());
    assert_eq!(
        bytes,
        prover.prove_bytes_with_seed(context, [2; 32]).unwrap()
    );
    let mut verifier = Verifier::new([1; 32], Arc::new(params)).unwrap();
    verifier.set_statement(statement).unwrap();
    verifier.verify(&proof, context).unwrap();
    verifier.verify_bytes(&bytes, context).unwrap();
    assert!(verifier.verify(&proof, b"another context").is_err());
    assert!(verifier.verify_bytes(&bytes, b"").is_err());
}

#[test]
fn a_toolbox_proof_verifies_only_under_its_own_context() {
    let scheme = Abdlop::new([1; 32], toy_d64()).unwrap();
    let ring = scheme.ring().clone();
    let statement = statement(&ring);
    let (s1, m) = witness(&ring);
    let prove = |context: &[u8]| {
        let proof = tbox::prove_with_seed(&scheme, &statement, &s1, &m, context, [3; 32]).unwrap();
        (
            proof.clone(),
            jali::codec::proof::encode(&scheme, &proof).unwrap(),
        )
    };
    let (a, a_bytes) = prove(b"context A");
    tbox::verify(&scheme, &statement, &a, b"context A").unwrap();
    // Nor does the empty context, a prefix or an extension of the one used.
    for other in [b"context B".as_slice(), b"", b"context", b"context A\0"] {
        assert!(tbox::verify(&scheme, &statement, &a, other).is_err());
    }
    let (empty, empty_bytes) = prove(b"");
    tbox::verify(&scheme, &statement, &empty, b"").unwrap();
    assert!(tbox::verify(&scheme, &statement, &empty, b"context A").is_err());
    let (_, b_bytes) = prove(b"context B");
    assert_ne!(a_bytes, b_bytes);
    assert_ne!(a_bytes, empty_bytes);
    // Identical inputs, context included, give byte-identical proofs.
    assert_eq!(prove(b"context A").1, a_bytes);
}

#[test]
fn one_seed_gives_unrelated_commitments_across_contexts_statements_and_messages() {
    let scheme = Abdlop::new([1; 32], toy_d64()).unwrap();
    let ring = scheme.ring().clone();
    let statement = statement(&ring);
    let (s1, m) = witness(&ring);
    // The same witness satisfies all three statements: the second swaps the rows of a
    // Euclidean block, the third adds a trivially true quadratic equation.
    let mut other = statement.clone();
    other.l2[0].map.s = Some(selector(&ring, 10, &[3, 2]));
    let mut extra = statement.clone();
    extra.quadratic = vec![QuadEq::zero(ring.clone(), 2 * (s1.len() + m.len())).unwrap()];
    let commit = |statement: &Statement, m: &PolyVec, context: &[u8]| {
        tbox::prove_with_seed(&scheme, statement, &s1, m, context, [4; 32])
            .unwrap()
            .commitment
    };
    let base = commit(&statement, &m, b"context A");
    // One randomness s2 for two proofs of one witness would give one t_A.
    assert_ne!(
        base.t_a,
        commit(&statement, &m, b"context B").t_a,
        "context"
    );
    assert_ne!(base.t_a, commit(&other, &m, b"context A").t_a, "statement");
    assert_ne!(base.t_a, commit(&extra, &m, b"context A").t_a, "equation");
    // For two messages it would give t_B - t_B' = m - m' on the message rows.
    let other_m = PolyVec::new(
        ring.clone(),
        vec![Poly::constant(ring.clone(), 3), Poly::constant(ring, -3)],
    )
    .unwrap();
    let t_b = commit(&statement, &other_m, b"context A").t_b;
    let rows = |a: &PolyVec, b: &PolyVec| -> Vec<Poly> {
        (0..m.len())
            .map(|i| a.entries()[i].sub(&b.entries()[i]).unwrap())
            .collect()
    };
    assert_ne!(rows(&base.t_b, &t_b), rows(&m, &other_m), "message");
}

#[test]
fn rng_toolbox_proofs_are_fresh_and_bound_to_their_context() {
    // `prove` and `prove_bytes` draw a fresh seed from the RNG per call.
    let params = Arc::new(toy_d64());
    let scheme = Abdlop::new([1; 32], (*params).clone()).unwrap();
    let ring = scheme.ring().clone();
    let (s1, m) = witness(&ring);
    let mut prover = Prover::new([1; 32], params.clone()).unwrap();
    prover.set_statement(statement(&ring)).unwrap();
    prover.set_witness(&s1, &m).unwrap();
    let mut verifier = Verifier::new([1; 32], params).unwrap();
    verifier.set_statement(statement(&ring)).unwrap();
    let mut rng = common::SeededRng::new(b"toolbox");
    let proof = prover.prove(b"context A", &mut rng).unwrap();
    verifier.verify(&proof, b"context A").unwrap();
    assert!(verifier.verify(&proof, b"context B").is_err());
    let bytes = prover.prove_bytes(b"context A", &mut rng).unwrap();
    verifier.verify_bytes(&bytes, b"context A").unwrap();
    for other in [b"context B".as_slice(), b""] {
        assert!(verifier.verify_bytes(&bytes, other).is_err());
    }
    let second = jali::codec::proof::decode(&scheme, &bytes).unwrap();
    assert_ne!(second.commitment.t_a, proof.commitment.t_a);
}

/// The shape of `statement` with user equations at proof degree `d`: binary rows 0 and 1,
/// Euclidean blocks on rows 2, 3 (bound 2d) and 4 (bound d), a range block on both messages.
fn user_equation_statement(ring: &Arc<Ring>, s1: &PolyVec, m: &PolyVec) -> Statement {
    let d = ring.degree();
    let dim = 2 * (s1.len() + m.len());
    let select_s = |indices: &[usize]| AffineBlock {
        rows: indices.len(),
        s: Some(selector(ring, 10, indices)),
        m: None,
        offset: None,
    };
    // m_0 m_1 + 6 = 0, on the interleaved indices 2 (m1 + j) of the messages.
    let mut quadratic = QuadEq::zero(ring.clone(), dim).unwrap();
    quadratic.r2 = SparsePolyMat::new(
        ring.clone(),
        dim,
        vec![(20, 22, Poly::constant(ring.clone(), 1))],
    )
    .unwrap();
    quadratic.r0 = Poly::constant(ring.clone(), 6);
    // ct(sigma(a) s_7 + b sigma(s_3)) = value: index 7 = 2 * 3 + 1 is the conjugate of s_3.
    let mut rng = SplitMix64(5 + d as u64);
    let a = poly(ring, rng.uniform(d, 1000));
    let b = poly(ring, rng.uniform(d, 1000));
    let mut evaluation = QuadEq::zero(ring.clone(), dim).unwrap();
    evaluation.r1 =
        SparsePolyVec::new(ring.clone(), dim, vec![(7, b.clone()), (14, a.auto())]).unwrap();
    let value = evaluation
        .evaluate(&quad::interleave(s1, m).unwrap())
        .unwrap()
        .coefficient_i128(0)
        .unwrap();
    let inner: i128 = common::values(&a)
        .iter()
        .zip(common::values(&s1.entries()[7]))
        .map(|(x, y)| x * y)
        .sum();
    let conjugate = b
        .mul(&s1.entries()[3].auto())
        .unwrap()
        .coefficient_i128(0)
        .unwrap();
    assert_eq!(value, inner + conjugate);
    // Non-constant coefficients of r0 are unconstrained.
    let mut r0 = vec![0; d];
    r0[0] = -value;
    r0[9] = 12345;
    evaluation.r0 = Poly::new(ring.clone(), r0).unwrap();
    Statement {
        quadratic: vec![quadratic],
        evaluation: vec![evaluation],
        binary: Some(select_s(&[0, 1])),
        l2: vec![
            L2Block {
                map: select_s(&[2, 3]),
                bound_squared: 2 * d as u64,
            },
            L2Block {
                map: select_s(&[4]),
                bound_squared: d as u64,
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
fn user_equation_witness(ring: &Arc<Ring>) -> (PolyVec, PolyVec) {
    let mut rng = SplitMix64(15);
    let s1 = PolyVec::new(
        ring.clone(),
        (0..10)
            .map(|_| poly(ring, rng.binary(ring.degree())))
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
fn prove_and_break_user_equations(params: TboxParams) {
    const CTX: &[u8] = b"jali-test/toolbox-user-equations";
    let scheme = Abdlop::new([1; 32], params).unwrap();
    let ring = scheme.ring().clone();
    let (s1, m) = user_equation_witness(&ring);
    let statement = user_equation_statement(&ring, &s1, &m);
    let prove =
        |st: &Statement, seed: u8| tbox::prove_with_seed(&scheme, st, &s1, &m, CTX, [seed; 32]);
    let verify = |st: &Statement, proof: &tbox::Proof| tbox::verify(&scheme, st, proof, CTX);
    let proof = prove(&statement, 2).unwrap();
    verify(&statement, &proof).unwrap();
    assert!(tbox::verify(&scheme, &statement, &proof, b"other").is_err());
    let bytes = jali::codec::proof::encode(&scheme, &proof).unwrap();
    assert_eq!(jali::codec::proof::decode(&scheme, &bytes).unwrap(), proof);
    // A false evaluation equation: its constant coefficient off by one.
    let mut false_eval = statement.clone();
    false_eval.evaluation[0].r0 = false_eval.evaluation[0]
        .r0
        .add(&Poly::constant(ring.clone(), 1))
        .unwrap();
    assert_eq!(prove(&false_eval, 2).err(), Some(Error::Witness));
    assert!(verify(&false_eval, &proof).is_err());
    // Another non-constant coefficient keeps the equation true but changes the statement.
    let mut other = statement.clone();
    other.evaluation[0].r0.set_coefficient(9, 1).unwrap();
    assert!(verify(&other, &proof).is_err());
    let proof = prove(&other, 3).unwrap();
    verify(&other, &proof).unwrap();
    // A false quadratic equation.
    let mut false_quadratic = statement.clone();
    false_quadratic.quadratic[0].r0 = Poly::constant(ring.clone(), 7);
    assert_eq!(prove(&false_quadratic, 2).err(), Some(Error::Witness));
    assert!(verify(&false_quadratic, &proof).is_err());
}

#[test]
fn toolbox_with_user_quadratic_and_evaluation_equations_at_degree_64() {
    prove_and_break_user_equations(toy_d64());
}

#[test]
fn toolbox_with_user_quadratic_and_evaluation_equations_at_degree_128() {
    let req = Requirements {
        m1: 10,
        l: 2,
        alpha_squared: 1280,
        n_bin: 2,
        l2_rows: vec![2, 1],
        l2_bounds_squared: vec![256, 128],
        n_prime: 2,
        linf_bound: 4,
        max_integer_coefficient: U256::ZERO,
        approx_alpha_squared: None,
        lifted_moduli: Vec::new(),
        linf: None,
    };
    let (_, params) = fit_upward(
        &req,
        "toolbox-user-equations-128-test-only",
        128,
        1099511627917,
    );
    prove_and_break_user_equations(params);
}

#[test]
fn stateful_setters_build_the_same_statement_and_refuse_bad_blocks() {
    const CTX: &[u8] = b"jali-test/stateful-setters";
    let params = Arc::new(toy_d64());
    let scheme = Abdlop::new([1; 32], (*params).clone()).unwrap();
    let ring = scheme.ring().clone();
    let (s1, m) = user_equation_witness(&ring);
    let statement = user_equation_statement(&ring, &s1, &m);
    let mut whole = Prover::new([1; 32], params.clone()).unwrap();
    whole.set_statement(statement.clone()).unwrap();
    whole.set_witness(&s1, &m).unwrap();
    let mut parts = Prover::new([1; 32], params.clone()).unwrap();
    assert_eq!(
        parts.prove_with_seed(CTX, [2; 32]).err(),
        Some(Error::Witness)
    );
    parts.set_quadratic_eqs(&statement.quadratic).unwrap();
    parts.set_eval_eqs(&statement.evaluation).unwrap();
    parts.set_l2_blocks(&statement.l2).unwrap();
    parts.set_binary_block(statement.binary.clone()).unwrap();
    parts.set_arp_block(statement.arp.clone()).unwrap();
    parts.set_witness(&s1, &m).unwrap();
    let bytes = parts.prove_bytes_with_seed(CTX, [2; 32]).unwrap();
    assert_eq!(bytes, whole.prove_bytes_with_seed(CTX, [2; 32]).unwrap());
    let mut verifier = Verifier::new([1; 32], params.clone()).unwrap();
    verifier.set_quadratic_eqs(&statement.quadratic).unwrap();
    verifier.set_eval_eqs(&statement.evaluation).unwrap();
    verifier.set_l2_blocks(&statement.l2).unwrap();
    verifier.set_binary_block(statement.binary.clone()).unwrap();
    verifier.set_arp_block(statement.arp.clone()).unwrap();
    verifier.verify_bytes(&bytes, CTX).unwrap();
    assert!(verifier.verify_bytes(&bytes, b"other").is_err());
    // Refusals: wrong dimensions and rings, a missing or extra block, another bound.
    let other_ring = Ring::new(13, 64).unwrap();
    let wrong_space = QuadEq::zero(ring.clone(), 2).unwrap();
    assert_eq!(
        parts
            .set_quadratic_eqs(std::slice::from_ref(&wrong_space))
            .err(),
        Some(Error::Dimension)
    );
    assert_eq!(
        parts.set_eval_eqs(&[wrong_space]).err(),
        Some(Error::Dimension)
    );
    let foreign = QuadEq::zero(other_ring.clone(), 24).unwrap();
    assert_eq!(
        verifier.set_eval_eqs(&[foreign]).err(),
        Some(Error::RingMismatch)
    );
    assert_eq!(
        parts.set_l2_blocks(&statement.l2[..1]).err(),
        Some(Error::Dimension)
    );
    let mut swapped = statement.l2.clone();
    swapped.swap(0, 1);
    assert_eq!(parts.set_l2_blocks(&swapped).err(), Some(Error::Dimension));
    let mut rebound = statement.l2.clone();
    rebound[1].bound_squared += 1;
    assert_eq!(
        parts.set_l2_blocks(&rebound).err(),
        Some(Error::Parameter("exact block bound"))
    );
    assert_eq!(parts.set_binary_block(None).err(), Some(Error::Dimension));
    let mut short = statement.binary.clone().unwrap();
    short.rows = 1;
    assert_eq!(
        verifier.set_binary_block(Some(short)).err(),
        Some(Error::Dimension)
    );
    assert_eq!(parts.set_arp_block(None).err(), Some(Error::Dimension));
    let mut foreign = statement.arp.clone().unwrap();
    foreign.m = Some(selector(&other_ring, 2, &[0, 1]));
    assert_eq!(
        verifier.set_arp_block(Some(foreign)).err(),
        Some(Error::RingMismatch)
    );
    assert_eq!(
        parts
            .set_witness(&s1, &PolyVec::zero(ring.clone(), 3))
            .err(),
        Some(Error::Dimension)
    );
    assert_eq!(
        parts
            .set_witness(&s1, &PolyVec::zero(other_ring.clone(), 2))
            .err(),
        Some(Error::RingMismatch)
    );
    // A set without an approximate range block refuses one.
    let mut no_arp = toy_d64();
    no_arp.n_prime = 0;
    no_arp.linf_bound = 0;
    no_arp.mlwe_rank += 256 / 64;
    let mut prover = Prover::new([1; 32], Arc::new(no_arp)).unwrap();
    assert_eq!(
        prover.set_arp_block(statement.arp.clone()).err(),
        Some(Error::Dimension)
    );
    prover.set_arp_block(None).unwrap();
    // The refused calls left the statement as it was.
    assert_eq!(bytes, parts.prove_bytes_with_seed(CTX, [2; 32]).unwrap());
}

/// The shape of the user-equation statement at degree 64.
fn user_equation_requirements() -> Requirements {
    Requirements {
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
    }
}

/// The user-equation statement at degree 64 with test-only parameters (synthetic MLWE metadata)
/// fitted at a prime `q` above $`2^{64}`$: the proof verifies only under its context and public
/// seed, and byte flips and tampered responses and commitments are refused. Returns the set.
fn toolbox_proof_at_wide_modulus(
    q: U256,
    factors: &[(&str, u32)],
    id: &str,
    primes: usize,
) -> TboxParams {
    const CTX: &[u8] = b"jali-test/toolbox-wide";
    let divisors = even_divisors(&q, factors);
    let params = fit_wide(&user_equation_requirements(), id, q, &divisors, 64).unwrap();
    let scheme = Abdlop::new([1; 32], params.clone()).unwrap();
    assert_eq!(scheme.ring().modulus(), q);
    assert_eq!(scheme.ring().rns_primes(), primes, "{id}");
    prove_and_break_user_equations(params.clone());
    let ring = scheme.ring().clone();
    let (s1, m) = user_equation_witness(&ring);
    let statement = user_equation_statement(&ring, &s1, &m);
    let proof = tbox::prove_with_seed(&scheme, &statement, &s1, &m, CTX, [2; 32]).unwrap();
    tbox::verify(&scheme, &statement, &proof, CTX).unwrap();
    assert!(tbox::verify(&scheme, &statement, &proof, b"wrong context").is_err());
    let other = Abdlop::new([9; 32], params.clone()).unwrap();
    assert!(tbox::verify(&other, &statement, &proof, CTX).is_err());
    let bytes = jali::codec::proof::encode(&scheme, &proof).unwrap();
    assert_eq!(jali::codec::proof::decode(&scheme, &bytes).unwrap(), proof);
    // Every tenth byte flipped in turn: the decoder or the verifier refuses each.
    for i in (0..bytes.len()).step_by(bytes.len() / 10) {
        let mut bad = bytes.clone();
        bad[i] ^= 0x10;
        if let Ok(p) = jali::codec::proof::decode(&scheme, &bad) {
            assert!(
                tbox::verify(&scheme, &statement, &p, CTX).is_err(),
                "byte {i}"
            );
        }
    }
    // A response coefficient moved by one, and a commitment coefficient moved by q - 1.
    let mut bad = proof.clone();
    let mut z = bad.z_approx.entries().to_vec();
    let changed = z[0].coefficient_i128(3).unwrap() + 1;
    z[0].set_coefficient(3, changed).unwrap();
    bad.z_approx = PolyVec::new(ring.clone(), z).unwrap();
    assert!(tbox::verify(&scheme, &statement, &bad, CTX).is_err());
    let mut bad = proof.clone();
    let mut t = bad.commitment.t_b.entries().to_vec();
    t[0] = t[0].sub(&Poly::constant(ring.clone(), 1)).unwrap();
    bad.commitment.t_b = PolyVec::new(ring.clone(), t).unwrap();
    assert!(tbox::verify(&scheme, &statement, &bad, CTX).is_err());
    // The estimate of the proof length stays an upper bound at these widths.
    let estimate = params.check().unwrap().estimated_proof_bytes;
    assert!(bytes.len() <= estimate, "{} > {estimate}", bytes.len());
    params
}

#[test]
fn toolbox_proofs_at_moduli_of_241_and_129_bits() {
    // q = 2^240 + 325 and q = 2^128 + 165, both 5 mod 8 and proven prime in Sage. Moduli above
    // 2^64 are accepted after the Baillie-PSW test.
    let q240 = power_plus(240, 325);
    toolbox_proof_at_wide_modulus(q240, &FACTORS_240, "toolbox-q240-test-only", 8);
    let q128 = power_plus(128, 165);
    toolbox_proof_at_wide_modulus(q128, &FACTORS_128, "toolbox-q128-test-only", 5);
}

#[test]
fn a_toolbox_proof_at_the_largest_prime_modulus_5_mod_8() {
    // q = 2^256 - 435, the largest prime 5 mod 8 below 2^256 (Sage), with all nine RNS primes.
    // The high part of q - 1 = 2^256 - 436 rounds up to 2^(256 - D) once 2^(D - 1) > 436, so
    // the compressed commitment fits only for D <= 9, and the fit lowers D to fit it.
    let q = power_minus(256, 435);
    let params = toolbox_proof_at_wide_modulus(q, &FACTORS_256, "toolbox-q256-test-only", 9);
    assert!(params.d_bits <= 9, "D = {}", params.d_bits);
    let mut wider = params.clone();
    wider.d_bits = 10;
    assert_eq!(
        wider.check().err(),
        Some(Error::Parameter("compressed commitment bit width"))
    );
}

/// The user-equation statement at $`q=2^{240}+325`$ with the approximate range Gaussian at
/// width $`1.55\cdot2^t`$: the proof verifies, its response is as wide as the Gaussian, and
/// a response coefficient moved by one or past `z4_bound` is refused.
fn toolbox_proof_with_range_exponent(t: u32) {
    const CTX: &[u8] = b"jali-test/toolbox-range-width";
    let q = power_plus(240, 325);
    let divisors = even_divisors(&q, &FACTORS_240);
    let id = format!("toolbox-sigma4-{t}-test-only");
    let mut params = fit_wide(&user_equation_requirements(), &id, q, &divisors, 64).unwrap();
    params.log_sigma[3] = t;
    let checked = params.check().unwrap();
    let scheme = Abdlop::new([1; 32], params).unwrap();
    let ring = scheme.ring().clone();
    let (s1, m) = user_equation_witness(&ring);
    let statement = user_equation_statement(&ring, &s1, &m);
    let proof = tbox::prove_with_seed(&scheme, &statement, &s1, &m, CTX, [2; 32]).unwrap();
    tbox::verify(&scheme, &statement, &proof, CTX).unwrap();
    assert!(tbox::verify(&scheme, &statement, &proof, b"wrong context").is_err());
    let largest = proof
        .z_approx
        .entries()
        .iter()
        .map(|p| p.norm_infinity())
        .max()
        .unwrap();
    assert!(largest > U256::ONE.shl_vartime(t - 1), "t = {t}");
    assert!(largest <= U256::from_u128(checked.z4_bound), "t = {t}");
    let bytes = jali::codec::proof::encode(&scheme, &proof).unwrap();
    assert_eq!(jali::codec::proof::decode(&scheme, &bytes).unwrap(), proof);
    assert!(bytes.len() <= checked.estimated_proof_bytes, "t = {t}");
    let bound = i128::try_from(checked.z4_bound).unwrap();
    for value in [
        proof.z_approx.entries()[0].coefficient_i128(5).unwrap() + 1,
        bound + 1,
    ] {
        let mut bad = proof.clone();
        let mut z = bad.z_approx.entries().to_vec();
        z[0].set_coefficient(5, value).unwrap();
        bad.z_approx = PolyVec::new(ring.clone(), z).unwrap();
        assert!(
            tbox::verify(&scheme, &statement, &bad, CTX).is_err(),
            "t = {t}"
        );
    }
}

#[test]
fn toolbox_proofs_with_range_exponents_41_and_68() {
    // Two range exponents between 40 and the cap of 100.
    for t in [41, 68] {
        toolbox_proof_with_range_exponent(t);
    }
}

#[test]
fn a_toolbox_proof_at_the_gaussian_exponent_cap() {
    toolbox_proof_with_range_exponent(100);
}
