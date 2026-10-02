//! Each check of the opening-proof verifier, at its boundary where it has one. Structural
//! rejections must come before the challenge is recomputed: the closure passed to
//! `verify_core` panics if it is reached.
use super::*;

fn fixture() -> (Abdlop, Commitment, OpeningProof) {
    let scheme = Abdlop::new([91; 32], crate::params::toy_d64()).unwrap();
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
    let (commitment, opening) = scheme.commit_with_seed(s1, m, [92; 32]).unwrap();
    let proof = scheme
        .prove_with_seed(&commitment, &opening, b"ctx", [93; 32])
        .unwrap();
    scheme.verify(&commitment, &proof, b"ctx").unwrap();
    (scheme, commitment, proof)
}
fn unreachable(_: &PolyVec) -> Result<Poly, Error> {
    panic!("challenge recomputed: a structural check did not fire")
}
/// Four integers whose squares sum to `n` (Lagrange), found greedily with backtracking.
fn four_squares(n: u128) -> [i128; 4] {
    let root = |x: u128| {
        let mut r = (x as f64).sqrt() as u128;
        while r * r > x {
            r -= 1;
        }
        while (r + 1) * (r + 1) <= x {
            r += 1;
        }
        r
    };
    for a in (0..=root(n)).rev() {
        let n1 = n - a * a;
        for b in (0..=root(n1).min(a)).rev() {
            let n2 = n1 - b * b;
            for c in (0..=root(n2).min(b)).rev() {
                let n3 = n2 - c * c;
                let d = root(n3);
                if d * d == n3 {
                    return [a as i128, b as i128, c as i128, d as i128];
                }
            }
        }
    }
    unreachable!("every natural number is a sum of four squares")
}
/// A vector of `len` polynomials whose squared norm is exactly `n`.
pub(crate) fn with_norm(ring: &Arc<Ring>, len: usize, n: u128) -> PolyVec {
    let mut entries = vec![Poly::zero(ring.clone()); len];
    for (i, v) in four_squares(n).into_iter().enumerate() {
        entries[i % len].set_coefficient(i / len, v).unwrap();
    }
    let out = PolyVec::new(ring.clone(), entries).unwrap();
    assert_eq!(out.norm_squared().unwrap(), U256::from(n));
    out
}

#[test]
fn challenges_outside_the_set_are_rejected_before_the_challenge_is_recomputed() {
    let (scheme, commitment, proof) = fixture();
    let omega = scheme.checked.omega;
    let d = scheme.ring().degree();
    assert!(scheme.challenge_in_set(&proof.challenge));
    // Not sigma-stable.
    let mut p = proof.clone();
    let x = p.challenge.coefficient_i128(1).unwrap();
    p.challenge
        .set_coefficient(1, if x == 0 { 1 } else { 0 })
        .unwrap();
    assert_ne!(p.challenge.auto(), p.challenge);
    assert_eq!(
        scheme.verify_core(&commitment, &p, unreachable),
        Err(Error::InvalidProof)
    );
    // Coefficient bound: omega is in the set, omega + 1 is not; both are sigma-stable.
    let mut c = Poly::zero(scheme.ring().clone());
    c.set_coefficient(0, omega).unwrap();
    assert!(scheme.challenge_in_set(&c));
    c.set_coefficient(0, -omega - 1).unwrap();
    assert_eq!(c.auto(), c);
    assert!(!scheme.challenge_in_set(&c));
    let mut p = proof.clone();
    p.challenge = c;
    assert_eq!(
        scheme.verify_core(&commitment, &p, unreachable),
        Err(Error::InvalidProof)
    );
    // A nonzero coefficient at d/2. Since sigma(X^(d/2)) = -X^(d/2) and q is odd, such a
    // challenge is never sigma-stable, so this check cannot fire alone: it is implied by the
    // sigma-stability check.
    let mut p = proof.clone();
    p.challenge.set_coefficient(d / 2, 1).unwrap();
    assert_ne!(p.challenge.auto(), p.challenge);
    assert_eq!(
        scheme.verify_core(&commitment, &p, unreachable),
        Err(Error::InvalidProof)
    );
    // A challenge from another ring.
    let mut p = proof.clone();
    p.challenge = Poly::zero(Ring::new(13, d).unwrap());
    assert_eq!(
        scheme.verify_core(&commitment, &p, unreachable),
        Err(Error::InvalidProof)
    );
}

#[test]
fn response_shapes_are_checked_first() {
    let (scheme, commitment, proof) = fixture();
    let ring = scheme.ring().clone();
    let mut p = proof.clone();
    p.z1 = PolyVec::zero(ring.clone(), scheme.bounded_len() - 1);
    assert_eq!(
        scheme.verify_core(&commitment, &p, unreachable),
        Err(Error::Dimension)
    );
    let mut p = proof.clone();
    p.z21 = PolyVec::zero(ring.clone(), scheme.a2.cols() + 1);
    assert_eq!(
        scheme.verify_core(&commitment, &p, unreachable),
        Err(Error::Dimension)
    );
    let mut p = proof;
    p.hint = PolyVec::zero(Ring::new(13, 64).unwrap(), scheme.parameters.n_msis);
    assert_eq!(
        scheme.verify_core(&commitment, &p, unreachable),
        Err(Error::RingMismatch)
    );
}

#[test]
fn z1_bound_is_exact() {
    let (scheme, commitment, proof) = fixture();
    let ring = scheme.ring().clone();
    let bound = scheme.checked.z1_bound_squared;
    assert!(scheme.z1_within_bound(&proof.z1).unwrap());
    assert!(
        scheme
            .z1_within_bound(&with_norm(&ring, scheme.bounded_len(), bound))
            .unwrap()
    );
    let long = with_norm(&ring, scheme.bounded_len(), bound + 1);
    assert!(!scheme.z1_within_bound(&long).unwrap());
    let mut p = proof;
    p.z1 = long;
    assert_eq!(
        scheme.verify_core(&commitment, &p, unreachable),
        Err(Error::InvalidProof)
    );
}

#[test]
fn joint_bound_is_exact() {
    let (scheme, _, proof) = fixture();
    let ring = scheme.ring().clone();
    let bound = scheme.checked.b_squared;
    let zero = PolyVec::zero(ring.clone(), scheme.parameters.n_msis);
    let cols = scheme.a2.cols();
    assert!(
        scheme
            .joint_within_bound(&with_norm(&ring, cols, bound), &zero)
            .unwrap()
    );
    assert!(
        !scheme
            .joint_within_bound(&with_norm(&ring, cols, bound + 1), &zero)
            .unwrap()
    );
    // Both parts count: half of the bound in each is accepted, one more anywhere is not.
    let half = with_norm(&ring, cols, bound / 2);
    let rest = with_norm(&ring, scheme.parameters.n_msis, bound - bound / 2);
    assert!(scheme.joint_within_bound(&half, &rest).unwrap());
    let over = with_norm(&ring, scheme.parameters.n_msis, bound - bound / 2 + 1);
    assert!(!scheme.joint_within_bound(&half, &over).unwrap());
    assert!(scheme.joint_within_bound(&proof.z21, &zero).unwrap());
}

#[test]
fn t_a_above_the_compressed_range_is_rejected() {
    let (scheme, commitment, proof) = fixture();
    let high_max = crate::math::int::low_u128(&scheme.max_high()) as i128;
    let with_t_a = |value: i128| {
        let mut c = commitment.clone();
        let mut t = c.t_a.entries().to_vec();
        t[0].set_coefficient(0, value).unwrap();
        c.t_a = PolyVec::new(scheme.ring.clone(), t).unwrap();
        c
    };
    scheme.prefix(&with_t_a(high_max), b"ctx").unwrap();
    assert_eq!(
        scheme.prefix(&with_t_a(high_max + 1), b"ctx").err(),
        Some(Error::InvalidProof)
    );
    assert_eq!(
        scheme.verify(&with_t_a(high_max + 1), &proof, b"ctx"),
        Err(Error::InvalidProof)
    );
}
