//! The evaluation verifier's checks of its garbage rows and responses.
use super::*;
use crate::math::{Poly, Ring};

#[test]
fn response_counts_rings_and_reserved_coefficients_are_checked() {
    let scheme = Abdlop::new([1; 32], crate::params::toy_d64()).unwrap();
    let ring = scheme.ring().clone();
    let s1 = PolyVec::new(
        ring.clone(),
        vec![Poly::constant(ring.clone(), 1); scheme.bounded_len()],
    )
    .unwrap();
    let m = PolyVec::new(
        ring.clone(),
        vec![Poly::constant(ring.clone(), 7); scheme.message_len()],
    )
    .unwrap();
    let (commitment, opening) = scheme.commit_with_seed(s1, m, [2; 32]).unwrap();
    let proof = prove_with_seed(&scheme, &commitment, &opening, &[], &[], b"ctx", [3; 32]).unwrap();
    verify(&scheme, &commitment, &[], &[], &proof, b"ctx").unwrap();
    check_responses(&scheme, &proof).unwrap();
    let count = scheme.checked.lambda / 2;
    let d = ring.degree();
    let replace_h = |f: &dyn Fn(&mut Vec<Poly>)| {
        let mut p = proof.clone();
        let mut h = p.h.entries().to_vec();
        f(&mut h);
        p.h = PolyVec::new(ring.clone(), h).unwrap();
        p
    };
    assert_eq!(
        check_responses(&scheme, &replace_h(&|h| h.truncate(count - 1))),
        Err(Error::Dimension)
    );
    let mut p = proof.clone();
    p.garbage_commitments = PolyVec::zero(ring.clone(), count + 1);
    assert_eq!(check_responses(&scheme, &p), Err(Error::Dimension));
    let mut p = proof.clone();
    p.h = PolyVec::zero(Ring::new(13, d).unwrap(), count);
    assert_eq!(check_responses(&scheme, &p), Err(Error::RingMismatch));
    for reserved in [0, d / 2] {
        let p = replace_h(&|h| h[count - 1].set_coefficient(reserved, 1).unwrap());
        assert_eq!(check_responses(&scheme, &p), Err(Error::InvalidProof));
        assert_eq!(
            verify(&scheme, &commitment, &[], &[], &p, b"ctx"),
            Err(Error::InvalidProof)
        );
    }
    // Other coefficients pass these checks; the transcript then rejects the change.
    let p = replace_h(&|h| {
        let x = h[0].coefficient_i128(1).unwrap();
        h[0].set_coefficient(1, x + 1).unwrap()
    });
    check_responses(&scheme, &p).unwrap();
    assert!(verify(&scheme, &commitment, &[], &[], &p, b"ctx").is_err());
}
