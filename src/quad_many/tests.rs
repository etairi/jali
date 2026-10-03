//! `absorb_equations`, and `encode_equations` with `absorb_encoded`, against the transcript of
//! the encodings before segments: the count, then each equation's bytes under `equation`. The
//! same states for families of every length around the encoding chunk and the parallel
//! threshold, the empty family included; for a family with equations that fail to encode, the
//! same error as the first failure in order, after the same absorptions.
use super::*;
use crate::{
    math::{Ring, SparsePolyVec},
    quad::encoding_tests::{Rng, random_equation, reference_to_bytes},
};

/// `absorb_equations` as it was before segments.
fn reference_absorb(
    transcript: &mut Transcript,
    label: &[u8],
    equations: &[QuadEq],
) -> Result<(), Error> {
    transcript.absorb(label, &(equations.len() as u64).to_le_bytes());
    for equation in equations {
        transcript.absorb(b"equation", &reference_to_bytes(equation)?);
    }
    Ok(())
}

fn start() -> Transcript {
    Transcript::new(b"quad-many-tests", b"parameters", &[1; 32], b"statement")
}

#[test]
fn absorbed_families_equal_the_transcript_of_their_bytes() {
    let mut rng = Rng(19);
    let ring = Ring::new(1099511627917, 64).unwrap();
    let equations: Vec<QuadEq> = (0..70)
        .map(|_| {
            let (keys, zero_percent) = (1 + rng.below(30), rng.below(101));
            random_equation(&mut rng, &ring, 40, keys, zero_percent)
        })
        .collect();
    for count in [0, 1, 7, 8, 9, 31, 32, 33, 64, 65, 70] {
        let family = &equations[..count];
        let (mut reference, mut absorbed, mut encoded) = (start(), start(), start());
        reference_absorb(&mut reference, b"family", family).unwrap();
        absorb_equations(&mut absorbed, b"family", family).unwrap();
        absorb_encoded(&mut encoded, b"family", &encode_equations(family).unwrap());
        assert_eq!(absorbed, reference, "{count} equations");
        assert_eq!(encoded, reference, "{count} equations");
        assert_ne!(reference, start());
    }
}

#[test]
fn a_family_stops_at_its_first_equation_that_fails_to_encode() {
    let ring = Ring::new(1099511627917, 64).unwrap();
    let mut rng = Rng(23);
    let mut wrong_dimension = random_equation(&mut rng, &ring, 40, 5, 50);
    wrong_dimension.r1 = SparsePolyVec::new(ring.clone(), 42, vec![]).unwrap();
    let mut wrong_ring = random_equation(&mut rng, &ring, 40, 5, 50);
    wrong_ring.r0 = Poly::zero(Ring::new(13, 64).unwrap());
    // Failures at (first, second), within and across chunks of 32 equations.
    for (first, second) in [(0, 5), (3, 40), (30, 31), (31, 32), (32, 33), (40, 44)] {
        for (a, b, error) in [
            (&wrong_ring, &wrong_dimension, Error::RingMismatch),
            (&wrong_dimension, &wrong_ring, Error::Dimension),
        ] {
            let mut family: Vec<QuadEq> = (0..45)
                .map(|_| random_equation(&mut rng, &ring, 40, 5, 50))
                .collect();
            family[first] = a.clone();
            family[second] = b.clone();
            let (mut reference, mut absorbed) = (start(), start());
            assert_eq!(
                reference_absorb(&mut reference, b"family", &family),
                Err(error.clone())
            );
            assert_eq!(
                absorb_equations(&mut absorbed, b"family", &family),
                Err(error.clone())
            );
            // The count and the equations before the failure, and nothing more.
            assert_eq!(absorbed, reference, "failure at {first}");
            assert_eq!(encode_equations(&family).err(), Some(error));
        }
    }
}
