//! No stream is read twice, whichever sampler, index or width reads it. Every stream is keyed
//! by a key derived from the caller's seed and the call's inputs, so the log records
//! `(key, domain)` pairs. Each test watches its own seed, so tests running in parallel do not
//! see each other's streams.
use super::*;
use crate::lnp::{AffineBlock, L2Block, Statement};
use crate::math::{PolyMat, Ring};
use crate::rand::stream_log::{Stream, opened_with};
use std::sync::Arc;

/// No `(key, domain)` pair is opened twice, and every stream is keyed by an output of
/// `derive_key`: neither by the seed itself nor by a seed drawn from a stream, such as the
/// toolbox's evaluation-proof seed.
fn assert_no_stream_opened_twice(what: &str, seed: [u8; 32], streams: &[Stream]) {
    assert!(!streams.is_empty(), "{what}: nothing recorded");
    let name = |s: &Stream| {
        format!(
            "stream ({:#x}, {}) of key {}",
            s.domain >> 32,
            s.domain & 0xffff_ffff,
            hex::encode(&s.key[..4])
        )
    };
    let mut seen = std::collections::BTreeSet::new();
    for s in streams {
        assert_ne!(s.key, seed, "{what}: {} keyed by the seed itself", name(s));
        assert!(
            s.derived,
            "{what}: {} keyed by a seed, not a derived key",
            name(s)
        );
        assert!(
            seen.insert((s.key, s.domain)),
            "{what}: {} opened twice",
            name(s)
        );
    }
}

/// The high domain words recorded, whichever key they were read under.
fn words(streams: &[Stream]) -> std::collections::BTreeSet<u32> {
    streams.iter().map(|s| (s.domain >> 32) as u32).collect()
}

/// The first `rows` polynomials of `a - b`.
fn difference(a: &PolyVec, b: &PolyVec, rows: usize) -> Vec<Poly> {
    (0..rows)
        .map(|i| a.entries()[i].sub(&b.entries()[i]).unwrap())
        .collect()
}

fn small_opening(scheme: &Abdlop) -> (PolyVec, PolyVec) {
    let ring = scheme.ring().clone();
    let s1 = PolyVec::new(
        ring.clone(),
        vec![Poly::constant(ring.clone(), 1); scheme.bounded_len()],
    )
    .unwrap();
    let m = PolyVec::new(
        ring.clone(),
        vec![Poly::constant(ring.clone(), 3); scheme.message_len()],
    )
    .unwrap();
    (s1, m)
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

/// A toolbox statement with binary, Euclidean and approximate range blocks, and a witness.
fn toolbox_instance(scheme: &Abdlop) -> (Statement, PolyVec, PolyVec) {
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
        vec![
            Poly::constant(ring.clone(), 2),
            Poly::constant(ring.clone(), -3),
        ],
    )
    .unwrap();
    (statement, s1, m)
}

/// The witness of `toolbox_instance` with its last, unconstrained polynomial changed.
fn other_witness(s1: &PolyVec) -> PolyVec {
    let ring = s1.ring().clone();
    let mut entries = s1.entries().to_vec();
    entries[9] = Poly::new(ring.clone(), (0..64).map(|j| (j % 2) as i128).collect()).unwrap();
    assert_ne!(entries[9], s1.entries()[9]);
    PolyVec::new(ring, entries).unwrap()
}

#[test]
fn a_commitment_and_its_opening_proof_from_one_seed_read_each_stream_once() {
    let scheme = Abdlop::new([91; 32], crate::params::toy_d64()).unwrap();
    let (s1, m) = small_opening(&scheme);
    let seed = [201; 32];
    let streams = opened_with(seed, || {
        let (commitment, opening) = scheme.commit_with_seed(s1, m, seed).unwrap();
        scheme
            .prove_with_seed(&commitment, &opening, b"", seed)
            .unwrap();
    });
    assert_no_stream_opened_twice("opening proof", seed, &streams);
}

#[test]
fn a_commitment_and_its_evaluation_proof_from_one_seed_read_each_stream_once() {
    // The evaluation proof runs the quadratic proofs on the same seed, so this covers them too.
    let scheme = Abdlop::new([91; 32], crate::params::toy_d64()).unwrap();
    let (s1, m) = small_opening(&scheme);
    let seed = [202; 32];
    let streams = opened_with(seed, || {
        let (commitment, opening) = scheme.commit_with_seed(s1, m, seed).unwrap();
        crate::quad_eval::prove_with_seed(&scheme, &commitment, &opening, &[], &[], b"", seed)
            .unwrap();
    });
    assert_no_stream_opened_twice("evaluation proof", seed, &streams);
}

#[test]
fn a_toolbox_proof_reads_each_stream_of_its_seed_once() {
    let scheme = Abdlop::new([1; 32], crate::params::toy_d64()).unwrap();
    let (statement, s1, m) = toolbox_instance(&scheme);
    let seed = [203; 32];
    let streams = opened_with(seed, || {
        prove_with_seed(&scheme, &statement, &s1, &m, b"", seed).unwrap();
    });
    assert_no_stream_opened_twice("toolbox", seed, &streams);
    // The log follows the evaluation-proof seed that the toolbox draws, down to the masks.
    for word in [
        secret::TOOLBOX_COMMITMENT,
        secret::TOOLBOX,
        secret::RANGE_MASKS,
        secret::GARBAGE,
        secret::OPENING_COINS,
        secret::OPENING_MASKS,
    ] {
        assert!(
            words(&streams).contains(&word),
            "word {word:#x} not recorded"
        );
    }
}

#[test]
fn a_toolbox_proof_shares_no_stream_with_a_commitment_from_the_same_seed() {
    // The toolbox commits internally. Were its commitment randomness that of a standalone
    // commitment from the same seed, t_B - t_B' would equal m - m'.
    let scheme = Abdlop::new([1; 32], crate::params::toy_d64()).unwrap();
    let (statement, s1, m) = toolbox_instance(&scheme);
    let (other_s1, other_m) = small_opening(&scheme);
    let seed = [204; 32];
    let streams = opened_with(seed, || {
        scheme.commit_with_seed(other_s1, other_m, seed).unwrap();
        prove_with_seed(&scheme, &statement, &s1, &m, b"", seed).unwrap();
    });
    assert_no_stream_opened_twice("standalone commitment and toolbox", seed, &streams);
}

#[test]
fn two_toolbox_proofs_under_two_contexts_from_one_seed_share_no_stream() {
    let scheme = Abdlop::new([1; 32], crate::params::toy_d64()).unwrap();
    let (statement, s1, m) = toolbox_instance(&scheme);
    let seed = [205; 32];
    let mut proofs = Vec::new();
    let streams = opened_with(seed, || {
        for context in [b"context A".as_slice(), b"context B"] {
            proofs.push(prove_with_seed(&scheme, &statement, &s1, &m, context, seed).unwrap());
        }
    });
    assert_no_stream_opened_twice("one statement, two contexts", seed, &streams);
    // One bounded witness: equal t_A would mean equal randomness s2. Comparing whole
    // commitments would not show it, because the range masks inside t_B differ anyway.
    assert_ne!(proofs[0].commitment.t_a, proofs[1].commitment.t_a);
    // Identical inputs give the identical proof: a reused seed reveals the equality pattern.
    assert_eq!(
        prove_with_seed(&scheme, &statement, &s1, &m, b"context A", seed).unwrap(),
        proofs[0]
    );
}

#[test]
fn toolbox_proofs_of_three_statements_with_one_witness_from_one_seed_share_no_stream() {
    // One witness satisfies all three statements: the second swaps the rows of the first
    // Euclidean block, the third adds a trivially true equation. A key without the statement
    // would commit to the witness with one randomness under all three, linking the proofs.
    let scheme = Abdlop::new([1; 32], crate::params::toy_d64()).unwrap();
    let (statement, s1, m) = toolbox_instance(&scheme);
    let mut swapped = statement.clone();
    swapped.l2[0].map.s = Some(selector(scheme.ring(), 10, &[3, 2]));
    let mut extended = statement.clone();
    let p = &scheme.parameters;
    extended.quadratic = vec![QuadEq::zero(scheme.ring().clone(), 2 * (p.m1 + p.l)).unwrap()];
    let statements = [statement, swapped, extended];
    let seed = [208; 32];
    let mut proofs = Vec::new();
    let streams = opened_with(seed, || {
        for statement in &statements {
            proofs.push(prove_with_seed(&scheme, statement, &s1, &m, b"", seed).unwrap());
        }
    });
    assert_no_stream_opened_twice("one witness, three statements", seed, &streams);
    for (i, proof) in proofs.iter().enumerate() {
        let next = (i + 1) % proofs.len();
        verify(&scheme, &statements[i], proof, b"").unwrap();
        assert!(verify(&scheme, &statements[next], proof, b"").is_err());
        assert_ne!(proof.commitment.t_a, proofs[next].commitment.t_a);
    }
}

#[test]
fn toolbox_proofs_and_commitments_under_two_public_seeds_from_one_seed_share_no_stream() {
    let schemes =
        [[1; 32], [2; 32]].map(|pp_seed| Abdlop::new(pp_seed, crate::params::toy_d64()).unwrap());
    let (statement, s1, m) = toolbox_instance(&schemes[0]);
    let seed = [209; 32];
    let streams = opened_with(seed, || {
        for scheme in &schemes {
            prove_with_seed(scheme, &statement, &s1, &m, b"", seed).unwrap();
        }
    });
    assert_no_stream_opened_twice("toolbox under two public seeds", seed, &streams);
    let (s1, m) = small_opening(&schemes[0]);
    let seed = [210; 32];
    let streams = opened_with(seed, || {
        for scheme in &schemes {
            scheme
                .commit_with_seed(s1.clone(), m.clone(), seed)
                .unwrap();
        }
    });
    assert_no_stream_opened_twice("commitments under two public seeds", seed, &streams);
}

#[test]
fn two_toolbox_proofs_of_two_messages_from_one_seed_share_no_stream() {
    // The bounded witness is the same. One randomness s2 for both would give equal t_A and
    // t_B - t_B' = m - m' on the message rows.
    let scheme = Abdlop::new([1; 32], crate::params::toy_d64()).unwrap();
    let (statement, s1, m) = toolbox_instance(&scheme);
    let ring = scheme.ring().clone();
    let other_m = PolyVec::new(
        ring.clone(),
        vec![Poly::constant(ring.clone(), 1), Poly::constant(ring, -3)],
    )
    .unwrap();
    let seed = [211; 32];
    let mut proofs = Vec::new();
    let streams = opened_with(seed, || {
        for m in [&m, &other_m] {
            proofs.push(prove_with_seed(&scheme, &statement, &s1, m, b"", seed).unwrap());
        }
    });
    assert_no_stream_opened_twice("one statement, two messages", seed, &streams);
    let [a, b] = [&proofs[0].commitment, &proofs[1].commitment];
    assert_ne!(a.t_a, b.t_a);
    assert_ne!(
        difference(&a.t_b, &b.t_b, m.len()),
        difference(&m, &other_m, m.len())
    );
}

#[test]
fn every_layer_on_one_commitment_under_two_contexts_from_one_seed_shares_no_stream() {
    // The opening, quadratic, many-quadratic and evaluation proofs of one commitment, each
    // under two contexts, all from the seed of the commitment.
    let scheme = Abdlop::new([91; 32], crate::params::toy_d64()).unwrap();
    let (s1, m) = small_opening(&scheme);
    let zero = QuadEq::zero(
        scheme.ring().clone(),
        2 * (scheme.bounded_len() + scheme.message_len()),
    )
    .unwrap();
    let eqs = [zero.clone()];
    let seed = [212; 32];
    let streams = opened_with(seed, || {
        let (c, o) = scheme.commit_with_seed(s1, m, seed).unwrap();
        for (context, other) in [(b"X".as_slice(), b"Y".as_slice()), (b"Y", b"X")] {
            let p = scheme.prove_with_seed(&c, &o, context, seed).unwrap();
            scheme.verify(&c, &p, context).unwrap();
            assert!(scheme.verify(&c, &p, other).is_err());
            let q = quad::prove_with_seed(&scheme, &c, &o, &zero, context, seed).unwrap();
            quad::verify(&scheme, &c, &zero, &q, context).unwrap();
            assert!(quad::verify(&scheme, &c, &zero, &q, other).is_err());
            let q = quad_many::prove_with_seed(&scheme, &c, &o, &eqs, context, seed).unwrap();
            quad_many::verify(&scheme, &c, &eqs, &q, context).unwrap();
            assert!(quad_many::verify(&scheme, &c, &eqs, &q, other).is_err());
            let e = quad_eval::prove_with_seed(&scheme, &c, &o, &[], &[], context, seed).unwrap();
            quad_eval::verify(&scheme, &c, &[], &[], &e, context).unwrap();
            assert!(quad_eval::verify(&scheme, &c, &[], &[], &e, other).is_err());
        }
    });
    assert_no_stream_opened_twice("every layer, two contexts", seed, &streams);
}

#[test]
fn two_toolbox_proofs_of_two_witnesses_from_one_seed_share_no_stream() {
    let scheme = Abdlop::new([1; 32], crate::params::toy_d64()).unwrap();
    let (statement, s1, m) = toolbox_instance(&scheme);
    let other = other_witness(&s1);
    let seed = [206; 32];
    let streams = opened_with(seed, || {
        for s1 in [&s1, &other] {
            prove_with_seed(&scheme, &statement, s1, &m, b"", seed).unwrap();
        }
    });
    assert_no_stream_opened_twice("one statement, two witnesses", seed, &streams);
}

#[test]
fn two_opening_proofs_under_two_contexts_and_two_commitments_from_one_seed_share_no_stream() {
    let scheme = Abdlop::new([91; 32], crate::params::toy_d64()).unwrap();
    let (s1, m) = small_opening(&scheme);
    let seed = [207; 32];
    let streams = opened_with(seed, || {
        let (commitment, opening) = scheme
            .commit_with_seed(s1.clone(), m.clone(), seed)
            .unwrap();
        for context in [b"context A".as_slice(), b"context B"] {
            scheme
                .prove_with_seed(&commitment, &opening, context, seed)
                .unwrap();
        }
        let other_m = PolyVec::new(
            m.ring().clone(),
            vec![Poly::constant(m.ring().clone(), 4); m.len()],
        )
        .unwrap();
        scheme.commit_with_seed(s1, other_m, seed).unwrap();
    });
    assert_no_stream_opened_twice("two contexts, two commitments", seed, &streams);
}
