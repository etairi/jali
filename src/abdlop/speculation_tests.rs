//! Speculative attempts of the opening proof (`parallel`): at every width the same accepted
//! attempt, proof, side data and attempt limit as one attempt at a time. The widths run on the
//! current pool's threads with the `parallel` feature, and one after the other without it.
use super::*;

type Fixture = (Abdlop, Commitment, Opening, Transcript);

/// An opening of the restart tests' shape and its transcript prefix.
fn fixture() -> Fixture {
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
    let prefix = scheme.prefix(&commitment, b"speculation").unwrap();
    (scheme, commitment, opening, prefix)
}

/// Prover seeds of the fixture and the attempt each accepts, found one attempt at a time (the
/// least seed of 0 to 255 for each attempt, with `prove` at width 1): 0, 1, 2, 3, 4, 7 and 8
/// are K - 1 and K for K in {1, 2, 3, 4, 8}, and 9, 15, 16 and 17 lie in and at the ends of
/// the second batch of 8.
const SEEDS: [(u8, u32); 11] = [
    (28, 0),
    (8, 1),
    (14, 2),
    (48, 3),
    (4, 4),
    (19, 7),
    (42, 8),
    (16, 9),
    (11, 15),
    (52, 16),
    (21, 17),
];
const WIDTHS: [u32; 6] = [1, 2, 3, 4, 8, 16];

/// The accepted attempt and proof, with the mask $`y_1`$ that the challenge function saw in
/// that attempt as side data.
fn prove(
    (scheme, commitment, opening, prefix): &Fixture,
    seed: u8,
    max_attempts: u32,
    width: u32,
) -> Result<(u32, OpeningProof, PolyVec), Error> {
    scheme.prove_core_speculative(
        commitment,
        opening,
        Caller::Opening,
        &prefix.digest(),
        &Zeroizing::new([seed; 32]),
        max_attempts,
        width,
        |y1, _, w1| Ok((scheme.challenge(prefix, w1)?, y1.clone())),
    )
}

#[test]
fn every_width_accepts_the_attempt_of_one_at_a_time_with_the_same_proof() {
    let f = fixture();
    let (scheme, commitment, _, _) = &f;
    for (seed, index) in SEEDS {
        let (accepted, proof, y1) = prove(&f, seed, secret::MAX_ATTEMPTS, 1).unwrap();
        assert_eq!(accepted, index, "seed {seed}");
        scheme.verify(commitment, &proof, b"speculation").unwrap();
        for width in WIDTHS {
            let (accepted, other, other_y1) = prove(&f, seed, secret::MAX_ATTEMPTS, width).unwrap();
            assert_eq!(
                (accepted, &other, &other_y1),
                (index, &proof, &y1),
                "{seed} {width}"
            );
        }
    }
}

#[test]
fn the_side_data_is_that_of_the_accepted_attempt() {
    let f = fixture();
    let (scheme, _, opening, prefix) = &f;
    for (seed, index) in SEEDS {
        let key = Abdlop::opening_key(opening, Caller::Opening, &prefix.digest(), &[seed; 32]);
        let mask = sample_gaussian(
            scheme.ring.clone(),
            scheme.bounded_len(),
            scheme.parameters.log_sigma[0],
            &key,
            secret::OPENING_MASKS + 2 * index,
        )
        .unwrap()
        .values;
        for width in WIDTHS {
            let (_, proof, y1) = prove(&f, seed, secret::MAX_ATTEMPTS, width).unwrap();
            // The mask of the accepted attempt, and z1 = y1 + c s1 of the same attempt.
            assert_eq!(y1, mask, "{seed} {width}");
            assert_eq!(
                add(&y1, &scale(&opening.s1, &proof.challenge).unwrap()).unwrap(),
                proof.z1
            );
        }
    }
}

#[test]
fn the_attempt_limit_and_its_error_are_unchanged_at_every_width() {
    let f = fixture();
    let (scheme, commitment, opening, prefix) = &f;
    for (seed, index) in SEEDS {
        let expected = prove(&f, seed, secret::MAX_ATTEMPTS, 1).unwrap();
        let key = Abdlop::opening_key(opening, Caller::Opening, &prefix.digest(), &[seed; 32]);
        let masks: Vec<PolyVec> = (0..index)
            .map(|a| {
                sample_gaussian(
                    scheme.ring.clone(),
                    scheme.bounded_len(),
                    scheme.parameters.log_sigma[0],
                    &key,
                    secret::OPENING_MASKS + 2 * a,
                )
                .unwrap()
                .values
            })
            .collect();
        for width in WIDTHS {
            // Acceptance at the last attempt the limit allows; one attempt fewer exhausts it,
            // after every attempt below the limit and none at or beyond it.
            assert_eq!(prove(&f, seed, index + 1, width).unwrap(), expected);
            let seen = std::sync::Mutex::new(Vec::new());
            let exhausted = scheme.prove_core_speculative(
                commitment,
                opening,
                Caller::Opening,
                &prefix.digest(),
                &Zeroizing::new([seed; 32]),
                index,
                width,
                |y1, _, w1| {
                    seen.lock().unwrap().push(y1.clone());
                    Ok((scheme.challenge(prefix, w1)?, ()))
                },
            );
            assert_eq!(exhausted.err(), Some(Error::RestartLimit), "{seed} {width}");
            let seen = seen.into_inner().unwrap();
            assert_eq!(seen.len(), index as usize, "{seed} {width}");
            assert!(seen.iter().all(|y1| masks.contains(y1)), "{seed} {width}");
            assert_eq!(prove(&f, seed, 0, width).err(), Some(Error::RestartLimit));
        }
    }
    // Limits above MAX_ATTEMPTS are MAX_ATTEMPTS, at every width.
    let expected = prove(&f, 12, secret::MAX_ATTEMPTS, 1).unwrap();
    for width in WIDTHS {
        assert_eq!(prove(&f, 12, u32::MAX, width).unwrap(), expected);
    }
}

#[test]
fn an_attempt_that_fails_ends_the_loop_where_one_at_a_time_would() {
    let f = fixture();
    let (scheme, commitment, opening, prefix) = &f;
    // Seed 19 accepts at attempt 7. The challenge function fails at attempt e, recognised by
    // its mask: the proof fails if e <= 7 and is unchanged if e > 7, in the accepted attempt's
    // batch or a later one.
    let (seed, index) = (19u8, 7u32);
    let key = Abdlop::opening_key(opening, Caller::Opening, &prefix.digest(), &[seed; 32]);
    let mask = |a: u32| {
        sample_gaussian(
            scheme.ring.clone(),
            scheme.bounded_len(),
            scheme.parameters.log_sigma[0],
            &key,
            secret::OPENING_MASKS + 2 * a,
        )
        .unwrap()
        .values
    };
    let expected = prove(&f, seed, secret::MAX_ATTEMPTS, 1).unwrap();
    assert_eq!(expected.0, index);
    for e in [0, 3, 6, 7, 8, 9, 15, 16] {
        let failing = mask(e);
        for width in WIDTHS {
            let result = scheme.prove_core_speculative(
                commitment,
                opening,
                Caller::Opening,
                &prefix.digest(),
                &Zeroizing::new([seed; 32]),
                secret::MAX_ATTEMPTS,
                width,
                |y1, _, w1| {
                    if *y1 == failing {
                        return Err(Error::Encoding);
                    }
                    Ok((scheme.challenge(prefix, w1)?, y1.clone()))
                },
            );
            if e <= index {
                assert_eq!(result.err(), Some(Error::Encoding), "{e} {width}");
            } else {
                assert_eq!(result.unwrap(), expected, "{e} {width}");
            }
        }
    }
}
