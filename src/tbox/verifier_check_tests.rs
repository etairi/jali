//! The toolbox verifier's checks of the range responses, at their boundaries. They run before
//! any transcript is built, so a proof whose other parts are zero reaches them.
use super::*;
use crate::{
    abdlop::{OpeningProof, verifier_check_tests::with_norm},
    math::Ring,
    quad::QuadProof,
};

/// A proof of the right shape with zero commitment, evaluation proof and range responses.
fn zero_proof(scheme: &Abdlop) -> Proof {
    let ring = scheme.ring().clone();
    let p = &scheme.parameters;
    let (e, d) = (
        if scheme.checked.n_ex > 0 {
            256 / p.degree
        } else {
            0
        },
        if p.n_prime > 0 { 256 / p.degree } else { 0 },
    );
    let g = scheme.checked.lambda / 2;
    Proof {
        commitment: Commitment {
            t_a: PolyVec::zero(ring.clone(), p.n_msis),
            t_b: PolyVec::zero(ring.clone(), p.l + e + d + 1),
        },
        z_exact: PolyVec::zero(ring.clone(), e),
        z_approx: PolyVec::zero(ring.clone(), d),
        evaluation: EvalProof {
            garbage_commitments: PolyVec::zero(ring.clone(), g),
            h: PolyVec::zero(ring.clone(), g),
            quadratic: QuadProof {
                t: Poly::zero(ring.clone()),
                opening: OpeningProof {
                    challenge: Poly::zero(ring.clone()),
                    z1: PolyVec::zero(ring.clone(), scheme.bounded_len()),
                    z21: PolyVec::zero(ring.clone(), p.m2 - p.n_msis),
                    hint: PolyVec::zero(ring, p.n_msis),
                },
            },
        },
    }
}

#[test]
fn range_response_bounds_are_exact() {
    let scheme = Abdlop::new([1; 32], crate::params::toy_d64()).unwrap();
    let ring = scheme.ring().clone();
    let statement = Statement {
        binary: Some(crate::lnp::AffineBlock {
            rows: 2,
            s: None,
            m: None,
            offset: None,
        }),
        l2: [(2, 128), (1, 64)]
            .map(|(rows, bound_squared)| crate::lnp::L2Block {
                map: crate::lnp::AffineBlock {
                    rows,
                    s: None,
                    m: None,
                    offset: None,
                },
                bound_squared,
            })
            .to_vec(),
        arp: Some(crate::lnp::AffineBlock {
            rows: 2,
            s: None,
            m: None,
            offset: None,
        }),
        ..Statement::default()
    };
    let base = zero_proof(&scheme);
    assert_eq!(response_shape_and_bounds(&scheme, &base), Ok((4, 4)));
    let z3 = scheme.checked.z3_bound_squared;
    let mut proof = base.clone();
    proof.z_exact = with_norm(&ring, 4, z3);
    response_shape_and_bounds(&scheme, &proof).unwrap();
    proof.z_exact = with_norm(&ring, 4, z3 + 1);
    assert_eq!(
        response_shape_and_bounds(&scheme, &proof),
        Err(Error::InvalidProof)
    );
    assert_eq!(
        verify(&scheme, &statement, &proof, b""),
        Err(Error::InvalidProof)
    );
    let z4 = scheme.checked.z4_bound as i128;
    for (value, ok) in [(z4, true), (-z4, true), (z4 + 1, false), (-z4 - 1, false)] {
        let mut proof = base.clone();
        let mut entries = proof.z_approx.entries().to_vec();
        entries[3].set_coefficient(63, value).unwrap();
        proof.z_approx = PolyVec::new(ring.clone(), entries).unwrap();
        if ok {
            response_shape_and_bounds(&scheme, &proof).unwrap();
        } else {
            assert_eq!(
                response_shape_and_bounds(&scheme, &proof),
                Err(Error::InvalidProof)
            );
            assert_eq!(
                verify(&scheme, &statement, &proof, b""),
                Err(Error::InvalidProof)
            );
        }
    }
    // Shapes: one slot too few or too many, and another ring.
    let mut proof = base.clone();
    proof.z_exact = PolyVec::zero(ring.clone(), 3);
    assert_eq!(
        response_shape_and_bounds(&scheme, &proof),
        Err(Error::Dimension)
    );
    let mut proof = base.clone();
    proof.z_approx = PolyVec::zero(ring.clone(), 5);
    assert_eq!(
        response_shape_and_bounds(&scheme, &proof),
        Err(Error::Dimension)
    );
    let mut proof = base.clone();
    proof.z_approx = PolyVec::zero(Ring::new(13, 64).unwrap(), 4);
    assert_eq!(
        response_shape_and_bounds(&scheme, &proof),
        Err(Error::RingMismatch)
    );
    // Without range blocks, both responses must be empty.
    let mut p = crate::params::toy_d64();
    p.n_bin = 0;
    p.l2_rows.clear();
    p.l2_bounds_squared.clear();
    p.n_prime = 0;
    p.linf_bound = 0;
    p.mlwe_rank += 8;
    let bare = Abdlop::new([1; 32], p).unwrap();
    let proof = zero_proof(&bare);
    assert_eq!(response_shape_and_bounds(&bare, &proof), Ok((0, 0)));
    let mut long = proof.clone();
    long.z_exact = PolyVec::zero(bare.ring().clone(), 1);
    assert_eq!(
        response_shape_and_bounds(&bare, &long),
        Err(Error::Dimension)
    );
}
