//! `absorb_equations` against the absorption of the encodings before segments: the count as
//! `LE32`, then each equation's bytes as `SerializeVarLenString`. The same states for families of
//! every length around the encoding chunk and the parallel threshold, the empty family
//! included; for a family with equations that fail to encode, the same error as the first
//! failure in order, after the same absorptions.
use super::*;
use crate::{
    math::{Poly, Ring, SparsePolyVec},
    quad::encoding_tests::{Rng, random_equation, reference_to_bytes},
};

/// `absorb_equations` from the bytes of each equation, without segments.
fn reference_absorb(transcript: &mut Transcript, equations: &[QuadEq]) -> Result<(), Error> {
    transcript.absorb(&(equations.len() as u32).to_le_bytes());
    for equation in equations {
        let bytes = reference_to_bytes(equation)?;
        transcript.absorb(&(bytes.len() as u32).to_le_bytes());
        transcript.absorb(&bytes);
    }
    Ok(())
}

fn start() -> Transcript {
    let mut transcript = Transcript::new(Protocol::ManyQuadratic, b"quad-many-tests").unwrap();
    transcript.absorb(b"instance");
    transcript
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
        let (mut reference, mut absorbed) = (start(), start());
        reference_absorb(&mut reference, family).unwrap();
        absorb_equations(&mut absorbed, family).unwrap();
        assert_eq!(absorbed.peek(), reference.peek(), "{count} equations");
        assert_ne!(reference.peek(), start().peek());
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
                reference_absorb(&mut reference, &family),
                Err(error.clone())
            );
            assert_eq!(absorb_equations(&mut absorbed, &family), Err(error));
            // The count and the equations before the failure, and nothing more.
            assert_eq!(absorbed.peek(), reference.peek(), "failure at {first}");
        }
    }
}
