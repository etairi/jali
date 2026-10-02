use jali::{
    Error,
    abdlop::Abdlop,
    lnp::{AffineBlock, L2Block, Statement},
    math::{Poly, PolyMat, PolyVec},
    params::toy_d64,
    statement::Requirements,
    tbox,
};

mod common;
use common::{SplitMix64, params::fit_upward, ring::poly};

#[test]
fn each_block_can_be_present_or_absent_at_both_proof_degrees() {
    for degree in [64, 128] {
        for flags in 0u8..8 {
            let mut p = toy_d64();
            p.degree = degree;
            if degree == 64 && flags == 0 {
                p.d_bits = 0;
            }
            p.n_msis = 32;
            p.m2 = 96;
            p.log_sigma[1] = 13;
            p.n_bin = if flags & 1 != 0 { 2 } else { 0 };
            if flags & 2 == 0 {
                p.l2_rows.clear();
                p.l2_bounds_squared.clear();
            }
            if flags & 4 == 0 {
                p.n_prime = 0;
                p.linf_bound = 0;
            }
            // Size sigma4 of these test-only variants from the Euclidean bound sqrt(n'd)*4, as
            // tools/params/lnp_params.py does, so the range rejection constant stays small.
            p.log_sigma[3] = if degree == 64 { 11 } else { 12 };
            let has_exact = p.n_bin > 0 || !p.l2_rows.is_empty();
            let lext = usize::from(has_exact) * 256 / degree
                + usize::from(p.n_prime > 0) * 256 / degree
                + 4;
            p.mlwe_rank = p.m2 - p.n_msis - p.l - lext;
            p.id = format!("test-only-degree-{degree}-blocks-{flags}");
            p.estimator =
                "test-only dimensions with an increased MLWE rank; not an estimate".into();
            let scheme = Abdlop::new([flags; 32], p.clone()).unwrap();
            let ring = scheme.ring().clone();
            let block = |rows| AffineBlock {
                rows,
                s: None,
                m: None,
                offset: None,
            };
            let statement = Statement {
                quadratic: vec![],
                evaluation: vec![],
                binary: if p.n_bin > 0 {
                    Some(block(p.n_bin))
                } else {
                    None
                },
                l2: p
                    .l2_rows
                    .iter()
                    .zip(&p.l2_bounds_squared)
                    .map(|(rows, bound)| L2Block {
                        map: block(*rows),
                        bound_squared: *bound,
                    })
                    .collect(),
                arp: if p.n_prime > 0 {
                    Some(block(p.n_prime))
                } else {
                    None
                },
            };
            let proof = tbox::prove_with_seed(
                &scheme,
                &statement,
                &PolyVec::zero(ring.clone(), p.m1),
                &PolyVec::zero(ring, p.l),
                b"block-combinations",
                [flags + 16; 32],
            )
            .unwrap();
            tbox::verify(&scheme, &statement, &proof, b"block-combinations").unwrap();
            let bytes = jali::codec::proof::encode(&scheme, &proof).unwrap();
            assert_eq!(jali::codec::proof::decode(&scheme, &bytes).unwrap(), proof);
        }
    }
}

/// One block combination at proof degree 128 (bit 0 binary, bit 1 Euclidean, bit 2 range),
/// with dense uniform maps on both witness parts, offsets that make each block evaluate to a
/// valid target, and a random ternary witness and uniform messages. Parameters are fitted to
/// the shape. Each present block, pushed out of its range by one offset coefficient, makes the
/// prover refuse. The verifier rejects the honest proof for the changed range block, and with
/// `every_rejection` for every changed block and under another context; each verification of
/// these dense forms takes seconds.
fn dense_case(flags: u8, every_rejection: bool) {
    const CTX: &[u8] = b"jali-test/dense-blocks";
    let d = 128;
    let req = Requirements {
        m1: 10,
        l: 2,
        alpha_squared: 1280,
        n_bin: if flags & 1 != 0 { 2 } else { 0 },
        l2_rows: if flags & 2 != 0 { vec![2, 1] } else { vec![] },
        l2_bounds_squared: if flags & 2 != 0 {
            vec![128, 64]
        } else {
            vec![]
        },
        n_prime: if flags & 4 != 0 { 2 } else { 0 },
        linf_bound: if flags & 4 != 0 { 4 } else { 0 },
        max_integer_coefficient: jali::math::U256::ZERO,
        approx_alpha_squared: None,
        lifted_moduli: Vec::new(),
        linf: None,
    };
    let id = format!("dense-blocks-{flags}-test-only");
    let (q, params) = fit_upward(&req, &id, d, 1099511627917);
    let q = q as i128;
    let scheme = Abdlop::new([61; 32], params).unwrap();
    let ring = scheme.ring().clone();
    let mut rng = SplitMix64(66 + u64::from(flags));
    let s1 = PolyVec::new(
        ring.clone(),
        (0..10).map(|_| poly(&ring, rng.ternary(d))).collect(),
    )
    .unwrap();
    let m = PolyVec::new(
        ring.clone(),
        (0..2).map(|_| poly(&ring, rng.uniform(d, q))).collect(),
    )
    .unwrap();
    let mut dense = |rows: usize, target: Vec<Vec<i128>>| {
        let mut uniform = |count: usize| -> Vec<Poly> {
            (0..count).map(|_| poly(&ring, rng.uniform(d, q))).collect()
        };
        let es = PolyMat::new(ring.clone(), rows, 10, uniform(rows * 10)).unwrap();
        let em = PolyMat::new(ring.clone(), rows, 2, uniform(rows * 2)).unwrap();
        let (image_s, image_m) = (es.mul(&s1).unwrap(), em.mul(&m).unwrap());
        let offset = (0..rows)
            .map(|i| {
                poly(&ring, target[i].clone())
                    .sub(&image_s.entries()[i])
                    .unwrap()
                    .sub(&image_m.entries()[i])
                    .unwrap()
            })
            .collect();
        AffineBlock {
            rows,
            s: Some(es),
            m: Some(em),
            offset: Some(PolyVec::new(ring.clone(), offset).unwrap()),
        }
    };
    let mut targets = SplitMix64(8 + u64::from(flags));
    let binary_target: Vec<Vec<i128>> = (0..2).map(|_| targets.binary(d)).collect();
    let mut sparse = |weight: usize| {
        let mut c = vec![0i128; d];
        for _ in 0..weight {
            c[targets.below(d as u64) as usize] = targets.below(3) as i128 - 1;
        }
        c
    };
    let l2_targets = [vec![sparse(60), sparse(60)], vec![sparse(60)]];
    let range_target: Vec<Vec<i128>> = (0..2)
        .map(|i| (0..d).map(|j| ((i + j) % 9) as i128 - 4).collect())
        .collect();
    let statement = Statement {
        quadratic: vec![],
        evaluation: vec![],
        binary: (flags & 1 != 0).then(|| dense(2, binary_target)),
        l2: if flags & 2 != 0 {
            let [a, b] = l2_targets;
            vec![
                L2Block {
                    map: dense(2, a),
                    bound_squared: 128,
                },
                L2Block {
                    map: dense(1, b),
                    bound_squared: 64,
                },
            ]
        } else {
            vec![]
        },
        arp: (flags & 4 != 0).then(|| dense(2, range_target)),
    };
    let proof = tbox::prove_with_seed(&scheme, &statement, &s1, &m, CTX, [62; 32]).unwrap();
    tbox::verify(&scheme, &statement, &proof, CTX).unwrap();
    if every_rejection {
        assert!(tbox::verify(&scheme, &statement, &proof, b"other").is_err());
    }
    let bytes = jali::codec::proof::encode(&scheme, &proof).unwrap();
    assert_eq!(jali::codec::proof::decode(&scheme, &bytes).unwrap(), proof);
    // One offset coefficient out of range in each present block: binary 2, range 5 or more,
    // Euclidean norm above its bound.
    for block in 0..4 {
        let mut bad = statement.clone();
        let (target, shift) = match block {
            0 => (bad.binary.as_mut(), 2),
            1 => (bad.l2.get_mut(0).map(|b| &mut b.map), 100),
            2 => (bad.l2.get_mut(1).map(|b| &mut b.map), 100),
            _ => (bad.arp.as_mut(), 9),
        };
        let Some(target) = target else {
            continue;
        };
        let offset = target.offset.as_mut().unwrap();
        let mut entries = offset.entries().to_vec();
        let c = entries[0].coefficient_i128(5).unwrap();
        entries[0].set_coefficient(5, c + shift).unwrap();
        *offset = PolyVec::new(ring.clone(), entries).unwrap();
        assert_eq!(
            tbox::prove_with_seed(&scheme, &bad, &s1, &m, CTX, [62; 32]).err(),
            Some(Error::Witness),
            "flags {flags}, block {block}"
        );
        if every_rejection || block == 3 {
            assert!(
                tbox::verify(&scheme, &bad, &proof, CTX).is_err(),
                "flags {flags}, block {block}"
            );
        }
    }
}

#[test]
fn dense_maps_offsets_and_random_witnesses_with_every_block() {
    dense_case(7, false);
}

#[test]
#[ignore = "dense maps for all eight block combinations at degree 128: minutes"]
fn dense_maps_offsets_and_random_witnesses_for_every_combination() {
    for flags in 0..8 {
        dense_case(flags, true);
    }
}
