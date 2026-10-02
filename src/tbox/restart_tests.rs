//! The range round of the toolbox prover: the attempt limit, a restart after a rejected first
//! attempt, and the restarts of `range_response` on a coefficient above q/2 and on the bounds
//! `z3_bound_squared` and `z4_bound`. Also the witness check's norm budget.
use super::*;
use crate::{
    abdlop::verifier_check_tests::with_norm, lnp::AffineBlock, math::PolyMat,
    rand::reject::Variance,
};

fn selector(
    ring: &std::sync::Arc<crate::math::Ring>,
    columns: usize,
    indices: &[usize],
) -> PolyMat {
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

/// The statement and witness of `tests/toolbox.rs`: binary rows 0 and 1, Euclidean blocks on
/// rows 2, 3 and on row 4, and a range block on both messages.
pub(super) fn selector_statement(scheme: &Abdlop) -> (Statement, PolyVec, PolyVec) {
    let ring = scheme.ring().clone();
    let select_s = |indices: &[usize]| AffineBlock {
        rows: indices.len(),
        s: Some(selector(&ring, 10, indices)),
        m: None,
        offset: None,
    };
    let statement = Statement {
        binary: Some(select_s(&[0, 1])),
        l2: vec![
            crate::lnp::L2Block {
                map: select_s(&[2, 3]),
                bound_squared: 128,
            },
            crate::lnp::L2Block {
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
        ..Statement::default()
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
        vec![Poly::constant(ring.clone(), 2), Poly::constant(ring, -3)],
    )
    .unwrap();
    (statement, s1, m)
}

#[test]
fn range_round_restarts_and_its_attempt_limit() {
    let scheme = Abdlop::new([1; 32], crate::params::toy_d64()).unwrap();
    let (statement, s1, m) = selector_statement(&scheme);
    let prove = |seed: u8, attempts: u32| {
        prove_with_attempts(
            &scheme,
            &statement,
            &s1,
            &m,
            b"restarts",
            &Zeroizing::new([seed; 32]),
            attempts,
        )
    };
    assert_eq!(prove(0, 0).err(), Some(Error::RestartLimit));
    // A seed whose first range attempt is rejected and whose second is accepted.
    let seed = (0..=255)
        .find(|seed| prove(*seed, 1).err() == Some(Error::RestartLimit) && prove(*seed, 2).is_ok())
        .expect("a seed with exactly one range restart among 256");
    let proof = prove(seed, 2).unwrap();
    verify(&scheme, &statement, &proof, b"restarts").unwrap();
    assert_eq!(prove(seed, secret::MAX_ATTEMPTS).unwrap(), proof);
    assert_eq!(prove(seed, u32::MAX).unwrap(), proof);
}

#[test]
fn range_responses_restart_above_q_over_2_and_above_their_bounds() {
    let scheme = Abdlop::new([1; 32], crate::params::toy_d64()).unwrap();
    let ring = scheme.ring().clone();
    let slots = 256 / ring.degree();
    let seed = [9; 32];
    let coins = |domain: u64| AesPrg::new(&[7; 32], domain);
    // The response to the projections of `values` under `mask`, and the next 16 bytes of the
    // coin stream after it.
    let respond = |exact: bool, values: &[i128], mask: PolyVec, domain: u64| {
        let t = scheme.parameters.log_sigma[if exact { 2 } else { 3 }];
        let mask = abdlop::Mask {
            values: mask,
            variance: Variance::gaussian(t).unwrap(),
        };
        let mut random = coins(domain);
        let z = range_response(&scheme, &seed, exact, values, &mask, 1, &mut random).unwrap();
        let mut next = [0u8; 16];
        random.fill(&mut next).unwrap();
        (z, next)
    };
    // Zero values project to zero, so the response is the mask itself, and rejection sampling
    // accepts with probability 1/M whatever the mask: the coin stream alone decides.
    let zeros = [0i128; 2];
    let zero = PolyVec::zero(ring.clone(), slots);
    let accepting = |exact: bool| {
        (0..64)
            .find(|domain| respond(exact, &zeros, zero.clone(), *domain).0.is_some())
            .expect("an accepting coin among 64")
    };
    // Exact block: squared norm z3_bound_squared is kept, one more restarts.
    let z3 = scheme.checked.z3_bound_squared;
    let domain = accepting(true);
    let at = with_norm(&ring, slots, z3);
    assert_eq!(respond(true, &zeros, at.clone(), domain).0, Some(at));
    let above = with_norm(&ring, slots, z3 + 1);
    assert_eq!(respond(true, &zeros, above, domain).0, None);
    // Approximate block: a coefficient of absolute value z4_bound is kept, one more restarts.
    let z4 = scheme.checked.z4_bound as i128;
    let domain = accepting(false);
    let one = |value: i128| {
        let mut entries = zero.entries().to_vec();
        entries[slots - 1].set_coefficient(5, value).unwrap();
        PolyVec::new(ring.clone(), entries).unwrap()
    };
    for (value, kept) in [(z4, true), (-z4, true), (z4 + 1, false), (-z4 - 1, false)] {
        let z = respond(false, &zeros, one(value), domain).0;
        assert_eq!(z.is_some(), kept, "{value}");
    }
    // A mask coefficient is at most q/2, so only a long projection exceeds it. Values (h, 1),
    // with h = (q - 1)/2, project to at most h + 1 in absolute value, and the prover restarts
    // before it reads a coin; values (h, 0) project to at most h, and it reads one before the
    // z4 bound refuses.
    let half = ring.modulus_i128() / 2;
    let mut fresh = [0u8; 16];
    coins(domain).fill(&mut fresh).unwrap();
    for (second, reads_a_coin) in [(1, false), (0, true)] {
        let values = [half, second];
        let most = projections(&seed, false, &values)
            .unwrap()
            .iter()
            .map(|v| v.unsigned_abs())
            .max();
        assert_eq!(most, Some((half + second) as u128));
        let (z, next) = respond(false, &values, zero.clone(), domain);
        assert_eq!(z, None);
        assert_eq!(next != fresh, reads_a_coin, "{second}");
    }
}

#[test]
fn the_witness_check_refuses_a_witness_above_the_norm_budget() {
    let scheme = Abdlop::new([1; 32], crate::params::toy_d64()).unwrap();
    let (statement, s1, m) = selector_statement(&scheme);
    // Rows 5 to 9 are in no block, so only the norm budget alpha_squared constrains them.
    let budget = u128::from(scheme.parameters.alpha_squared);
    let with_budget = |n: u128| {
        let mut entries = s1.entries().to_vec();
        let used: u128 = s1.entries()[..5]
            .iter()
            .flat_map(|p| p.coefficients_i128().unwrap().to_vec())
            .map(|x| x.unsigned_abs().pow(2))
            .sum();
        let rest = with_norm(scheme.ring(), 5, n - used);
        entries[5..].clone_from_slice(rest.entries());
        let out = PolyVec::new(scheme.ring().clone(), entries).unwrap();
        assert_eq!(out.norm_squared().unwrap(), U256::from(n));
        out
    };
    statement
        .check_witness(&scheme, &with_budget(budget), &m)
        .unwrap();
    assert_eq!(
        statement
            .check_witness(&scheme, &with_budget(budget + 1), &m)
            .err(),
        Some(Error::Witness)
    );
}
