//! End-to-end tests: proofs of random statements with known witnesses verify, and changed
//! proofs, statements, seeds or contexts do not.
use super::*;
use crate::labrador::test_rng::TestRng;
use std::sync::Arc;

const Q: u64 = 4294967197;

/// A random statement with `full` full constraints and `ct` constant-term constraints, each
/// with one quadratic term and two linear terms, built around a random witness with
/// coefficients in $`[-bound,bound]`$, and the witness.
pub(crate) fn instance(
    rng: &mut TestRng,
    r: usize,
    n: usize,
    bound: u64,
    full: usize,
    ct: usize,
) -> (Statement, Witness) {
    instance_mod(rng, Q, r, n, bound, full, ct)
}

/// [`instance`] over another modulus.
pub(crate) fn instance_mod(
    rng: &mut TestRng,
    q: u64,
    r: usize,
    n: usize,
    bound: u64,
    full: usize,
    ct: usize,
) -> (Statement, Witness) {
    let ring = Rq::new(q, 64).unwrap();
    let d = 64;
    let vectors: Vec<Vec<i64>> = (0..r)
        .map(|_| (0..n * d).map(|_| rng.small(bound)).collect())
        .collect();
    let witness = Witness::new(&ring, vectors);
    let norm: u128 = witness.vectors.iter().map(|v| ring.norm_sq(v)).sum();
    let mut statement = Statement::new(ring.clone(), r, n, norm + 17).unwrap();
    let random_poly = |rng: &mut TestRng| (0..d).map(|_| rng.below(q)).collect::<Vec<u64>>();
    for index in 0..full + ct {
        let (i, j) = (rng.below(r as u64) as usize, rng.below(r as u64) as usize);
        let a = random_poly(rng);
        let (l1, o1) = (rng.below(n as u64) as usize + 1, 0);
        let o2 = rng.below(n as u64) as usize;
        let l2 = n - o2;
        let phi1: Vec<u64> = (0..l1 * d).map(|_| rng.below(q)).collect();
        let phi2: Vec<u64> = (0..l2 * d).map(|_| rng.below(q)).collect();
        let (k1, k2) = (rng.below(r as u64) as usize, rng.below(r as u64) as usize);
        // The value of the terms on the witness.
        let mut value = ring.poly_mul(&a, &inner(&ring, &witness.vectors[i], &witness.vectors[j]));
        ring.add_assign(
            &mut value,
            &inner(&ring, &phi1, &witness.vectors[k1][o1 * d..(o1 + l1) * d]),
        );
        ring.add_assign(
            &mut value,
            &inner(&ring, &phi2, &witness.vectors[k2][o2 * d..]),
        );
        let c = Constraint::new(value)
            .quadratic(i, j, a)
            .linear(k1, o1, phi1)
            .linear(k2, o2, phi2);
        if index < full {
            statement.push(c).unwrap();
        } else {
            // Only the constant coefficient must vanish: change the others.
            let mut c = c;
            for x in &mut c.constant_mut()[1..] {
                *x = rng.below(q);
            }
            statement.push_constant_term(c).unwrap();
        }
    }
    statement.check(&witness).unwrap();
    (statement, witness)
}

/// $`\sum_xa_xb_x`$ over two polynomial vectors.
fn inner(ring: &Arc<Rq>, a: &[u64], b: &[u64]) -> Vec<u64> {
    ring.inner_hat(&ring.ntt_many(a), &ring.ntt_many(b))
}

fn roundtrip(statement: &Statement, witness: &Witness) -> Proof {
    let policy = Policy::default();
    let proof = prove(statement, witness, &[9u8; 32], b"ctx", &policy).unwrap();
    verify(statement, &proof, &[9u8; 32], b"ctx", &policy).unwrap();
    proof
}

#[test]
fn small_statements_prove_and_verify() {
    let mut rng = TestRng::new(10);
    for (r, n, bound, full, ct) in [
        (1, 1, 1, 0, 0),
        (2, 3, 1, 1, 1),
        (3, 8, 4, 2, 3),
        (1, 16, 100, 3, 0),
    ] {
        let (statement, witness) = instance(&mut rng, r, n, bound, full, ct);
        let proof = roundtrip(&statement, &witness);
        assert!(!proof.is_empty());
    }
}

#[test]
fn a_larger_statement_takes_several_levels() {
    let mut rng = TestRng::new(11);
    let (statement, witness) = instance(&mut rng, 4, 512, 1, 3, 3);
    let proof = roundtrip(&statement, &witness);
    let witness_bytes = 4 * 512 * 64 * 32 / 8;
    eprintln!(
        "proof {} bytes for a witness of {} bytes",
        proof.len(),
        witness_bytes
    );
    assert!(proof.len() < witness_bytes / 4);
}

#[test]
fn changed_proofs_statements_seeds_and_contexts_are_rejected() {
    let mut rng = TestRng::new(12);
    let (statement, witness) = instance(&mut rng, 2, 32, 2, 2, 2);
    let policy = Policy::default();
    let proof = prove(&statement, &witness, &[9u8; 32], b"ctx", &policy).unwrap();
    let bytes = proof.as_bytes().to_vec();
    // Every 97th byte, flipped.
    for i in (0..bytes.len()).step_by(97) {
        let mut changed = bytes.clone();
        changed[i] ^= 0x10;
        let changed = Proof::from_bytes(changed).unwrap();
        assert!(
            verify(&statement, &changed, &[9u8; 32], b"ctx", &policy).is_err(),
            "byte {i}"
        );
    }
    // Truncated and extended.
    let short = Proof::from_bytes(bytes[..bytes.len() - 1].to_vec()).unwrap();
    assert!(verify(&statement, &short, &[9u8; 32], b"ctx", &policy).is_err());
    let mut long = bytes.clone();
    long.push(0);
    let long = Proof::from_bytes(long).unwrap();
    assert!(verify(&statement, &long, &[9u8; 32], b"ctx", &policy).is_err());
    // Another seed or context.
    assert!(verify(&statement, &proof, &[8u8; 32], b"ctx", &policy).is_err());
    assert!(verify(&statement, &proof, &[9u8; 32], b"ctX", &policy).is_err());
    // Another statement: one constant changed.
    let mut other = statement.clone();
    other.full_mut()[0].constant_mut()[5] ^= 1;
    assert!(verify(&other, &proof, &[9u8; 32], b"ctx", &policy).is_err());
    let mut other = statement.clone();
    other.constant_term_mut()[1].constant_mut()[0] ^= 1;
    assert!(verify(&other, &proof, &[9u8; 32], b"ctx", &policy).is_err());
}

#[test]
fn the_prover_refuses_a_witness_that_does_not_satisfy_the_statement() {
    let mut rng = TestRng::new(13);
    let (statement, witness) = instance(&mut rng, 2, 4, 3, 1, 1);
    let policy = Policy::default();
    let mut bad = witness.clone();
    bad.vectors[1][7] = (bad.vectors[1][7] + 1) % Q;
    assert_eq!(
        prove(&statement, &bad, &[0u8; 32], b"", &policy).err(),
        Some(Error::Witness)
    );
    // Over the norm bound.
    let ring = statement.ring().clone();
    let mut tight = Statement::new(ring, 2, 4, 1).unwrap();
    tight.push(Constraint::new(vec![0; 64])).unwrap();
    assert_eq!(
        prove(&tight, &witness, &[0u8; 32], b"", &policy).err(),
        Some(Error::Witness)
    );
}

#[test]
fn proofs_are_deterministic() {
    let mut rng = TestRng::new(14);
    let (statement, witness) = instance(&mut rng, 2, 8, 1, 1, 1);
    let policy = Policy::default();
    let a = prove(&statement, &witness, &[1u8; 32], b"x", &policy).unwrap();
    let b = prove(&statement, &witness, &[1u8; 32], b"x", &policy).unwrap();
    assert_eq!(a, b);
}

#[test]
fn other_moduli_ranks_and_shapes() {
    let mut rng = TestRng::new(15);
    // 2^36 - 243 and 2^38 - 107, a prime rank, many vectors, a 20-bit modulus.
    for (q, r, n, bound, full, ct) in [
        (68719476493u64, 3, 16, 50, 2, 1),
        (274877906837, 2, 64, 1000, 1, 1),
        (Q, 2, 97, 1, 1, 1),
        (Q, 16, 4, 1, 2, 2),
        (1048573, 2, 4, 1, 1, 0),
    ] {
        let (statement, witness) = instance_mod(&mut rng, q, r, n, bound, full, ct);
        roundtrip(&statement, &witness);
    }
}

#[test]
fn statements_without_constraints_or_with_a_zero_witness() {
    let ring = Rq::new(Q, 64).unwrap();
    // Only the norm bound.
    let mut rng = TestRng::new(16);
    let vectors: Vec<Vec<i64>> = (0..2)
        .map(|_| (0..8 * 64).map(|_| rng.small(1)).collect())
        .collect();
    let witness = Witness::new(&ring, vectors);
    let norm: u128 = witness.vectors.iter().map(|v| ring.norm_sq(v)).sum();
    let statement = Statement::new(ring.clone(), 2, 8, norm).unwrap();
    roundtrip(&statement, &witness);
    // The zero witness of a homogeneous constraint.
    let zero = Witness::new(&ring, vec![vec![0; 4 * 64]; 3]);
    let mut statement = Statement::new(ring.clone(), 3, 4, 0).unwrap();
    statement
        .push(Constraint::new(vec![0; 64]).linear(1, 0, vec![1; 4 * 64]))
        .unwrap();
    roundtrip(&statement, &zero);
}

#[test]
fn the_norm_bound_is_checked_against_the_modulus() {
    // 337 * 125^2 * beta^2 <= 30 q^2 fails for a bound close to q.
    let ring = Rq::new(1048573, 64).unwrap();
    let statement = Statement::new(ring.clone(), 1, 1, 1 << 40).unwrap();
    let witness = Witness::new(&ring, vec![vec![0; 64]]);
    assert!(matches!(
        prove(&statement, &witness, &[0u8; 32], b"", &Policy::default()),
        Err(Error::Parameter(_))
    ));
}

/// Every byte of one proof, flipped in one bit, is refused. About a minute in release mode.
#[test]
#[ignore = "slow: verifies thousands of changed proofs"]
fn every_changed_byte_is_rejected() {
    let mut rng = TestRng::new(17);
    let (statement, witness) = instance(&mut rng, 2, 4, 1, 1, 1);
    let policy = Policy::default();
    let proof = prove(&statement, &witness, &[3u8; 32], b"", &policy).unwrap();
    let bytes = proof.as_bytes().to_vec();
    for i in 0..bytes.len() {
        for bit in [0, 3, 7] {
            let mut changed = bytes.clone();
            changed[i] ^= 1 << bit;
            let changed = Proof::from_bytes(changed).unwrap();
            assert!(
                verify(&statement, &changed, &[3u8; 32], b"", &policy).is_err(),
                "byte {i} bit {bit}"
            );
        }
    }
}
