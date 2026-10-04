//! The rejection loop of the opening proof: the attempt limit, a restart after a rejected
//! first attempt, and the restart on the joint norm. Also the sampler's refusal of a width that
//! reaches q/2, and the refusals of `commit_with_randomness`.
use super::*;
use crate::abdlop::verifier_check_tests::with_norm;

#[test]
fn restarts_are_taken_and_the_attempt_limit_is_enforced() {
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
    let prefix = scheme.prefix(&commitment, b"restarts").unwrap();
    let prove = |seed: u8, attempts: u32| {
        scheme
            .prove_core_with_attempts(
                &commitment,
                &opening,
                Caller::Opening,
                &prefix.digest(),
                &Zeroizing::new([seed; 32]),
                attempts,
                |_, _, w1| Ok((scheme.challenge(&prefix, w1)?, ())),
            )
            .map(|(proof, ())| proof)
    };
    assert_eq!(prove(0, 0).err(), Some(Error::RestartLimit));
    // A seed whose first attempt is rejected and whose second is accepted (about 1 in 32).
    let seed = (0..=255)
        .find(|seed| prove(*seed, 1).err() == Some(Error::RestartLimit) && prove(*seed, 2).is_ok())
        .expect("a seed with exactly one restart among 256");
    let proof = prove(seed, 2).unwrap();
    scheme.verify(&commitment, &proof, b"restarts").unwrap();
    assert_eq!(prove(seed, secret::MAX_ATTEMPTS).unwrap(), proof);
    // The limit never exceeds MAX_ATTEMPTS, which bounds the mask domains.
    assert_eq!(prove(seed, u32::MAX).unwrap(), proof);
}

#[test]
fn a_mask_width_reaching_q_over_2_is_a_parameter_error() {
    let key = Zeroizing::new([5; 32]);
    let ring = Ring::new(13, 64).unwrap();
    assert_eq!(
        sample_gaussian(ring.clone(), 2, 10, &key, 0).err(),
        Some(Error::Parameter("Gaussian mask above q/2"))
    );
    // Width 1.55: every one of 64 samples stays within 6 = (13 - 1) / 2 for this key.
    let mask = sample_gaussian(ring, 1, 0, &key, 0).unwrap();
    assert_eq!(mask.values.len(), 1);
}

#[test]
fn the_joint_norm_check_restarts_the_prover() {
    let scheme = Abdlop::new([91; 32], crate::params::toy_d64()).unwrap();
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
    let (commitment, opening) = scheme.commit_with_seed(s1, m, [92; 32]).unwrap();
    let prefix = scheme.prefix(&commitment, b"joint-norm").unwrap();
    let prove = |opening: &Opening, attempts: u32| {
        scheme
            .prove_core_with_attempts(
                &commitment,
                opening,
                Caller::Opening,
                &prefix.digest(),
                &Zeroizing::new([94; 32]),
                attempts,
                |_, _, w1| Ok((scheme.challenge(&prefix, w1)?, ())),
            )
            .map(|(proof, ())| proof)
    };
    // The attempt at which the honest opening is accepted.
    let k = (1..=64)
        .find(|k| prove(&opening, *k).is_ok())
        .expect("an accepted attempt among the first 64");
    // The low part of t_A enters only z22 = y22 + c s22 - c low - w0: not the key, the masks,
    // the challenge, the rejection coins or z1. Shifted by K > B in every constant coefficient,
    // it moves each row of z22 by K c. The attempts before k fail as before, and attempt k now
    // fails on the joint norm alone.
    let shift = (scheme.checked.b_squared as f64).sqrt() as i128 + 1;
    let mut far = opening.clone();
    far.low = PolyVec::new(
        ring.clone(),
        opening
            .low
            .entries()
            .iter()
            .map(|p| p.add(&Poly::constant(ring.clone(), shift)).unwrap())
            .collect(),
    )
    .unwrap();
    assert_eq!(prove(&far, k).err(), Some(Error::RestartLimit));
}

#[test]
fn commit_with_randomness_refuses_a_long_s1_and_non_ternary_s2() {
    let scheme = Abdlop::new([95; 32], crate::params::toy_d64()).unwrap();
    let ring = scheme.ring().clone();
    let p = &scheme.parameters;
    // The bound covers the witness and its slack rows, one binary polynomial per block.
    let bound = u128::from(p.alpha_squared) + (p.l2_rows.len() * p.degree) as u128;
    let m = PolyVec::zero(ring.clone(), scheme.message_len());
    let s2 = |value: i128| {
        let mut entries = vec![Poly::constant(ring.clone(), -1); p.m2];
        entries[p.m2 - 1]
            .set_coefficient(p.degree - 1, value)
            .unwrap();
        PolyVec::new(ring.clone(), entries).unwrap()
    };
    let commit = |s1: PolyVec, s2: PolyVec| scheme.commit_with_randomness(s1, m.clone(), s2);
    let len = scheme.bounded_len();
    commit(with_norm(&ring, len, bound), s2(1)).unwrap();
    assert_eq!(
        commit(with_norm(&ring, len, bound + 1), s2(1)).err(),
        Some(Error::Witness)
    );
    for value in [2, -2] {
        assert_eq!(
            commit(with_norm(&ring, len, 1), s2(value)).err(),
            Some(Error::Witness),
            "{value}"
        );
    }
}
