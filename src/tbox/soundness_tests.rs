use super::*;
use crate::lnp::{AffineBlock, L2Block};

#[test]
fn borrowing_slack_between_exact_blocks_does_not_cancel_independent_equations() {
    let scheme = Abdlop::new([101; 32], crate::params::toy_d64()).unwrap();
    let ring = scheme.ring().clone();
    let zero_map = |rows| AffineBlock {
        rows,
        s: None,
        m: None,
        offset: None,
    };
    let statement = Statement {
        quadratic: vec![],
        evaluation: vec![],
        binary: Some(zero_map(2)),
        arp: Some(zero_map(2)),
        l2: vec![
            L2Block {
                map: AffineBlock {
                    offset: Some(
                        PolyVec::new(
                            ring.clone(),
                            vec![Poly::constant(ring.clone(), 12), Poly::zero(ring.clone())],
                        )
                        .unwrap(),
                    ),
                    ..zero_map(2)
                },
                bound_squared: 128,
            },
            L2Block {
                map: zero_map(1),
                bound_squared: 64,
            },
        ],
    };
    let extended = scheme.extend_messages(9).unwrap();
    let forms = forms(&scheme, &extended, &statement).unwrap();
    let mut bounded = vec![Poly::zero(ring.clone()); scheme.bounded_len()];
    // First block: 12² - 128 = 16; second: 0 + 48 - 64 = -16. Both slack
    // polynomials are binary, so a single shared weight would hide the violation.
    bounded[scheme.parameters.m1 + 1]
        .set_coefficient(4, 1)
        .unwrap();
    bounded[scheme.parameters.m1 + 1]
        .set_coefficient(5, 1)
        .unwrap();
    let mut messages = vec![Poly::zero(ring.clone()); extended.message_len()];
    *messages.last_mut().unwrap() = Poly::constant(ring.clone(), 1)
        .sub(&Poly::constant(ring.clone(), 1).rotate(32))
        .unwrap();
    let witness = quad::interleave(
        &PolyVec::new(ring.clone(), bounded).unwrap(),
        &PolyVec::new(ring.clone(), messages).unwrap(),
    )
    .unwrap();
    let values: Vec<_> = forms
        .evals
        .iter()
        .map(|eq| eq.evaluate(&witness).unwrap().coefficient_i128(0).unwrap())
        .collect();
    assert!(values[..values.len() - 2].iter().all(|x| *x == 0));
    assert_eq!(&values[values.len() - 2..], &[16, -16]);
    assert_eq!(values.iter().sum::<i128>(), 0);
    for row in 0..scheme.checked.lambda {
        let weights = crate::rand::uniform(
            &mut AesPrg::new(&[102; 32], domain(0x47414d4d, row as u32)),
            ring.modulus_i128() as u128,
            values.len(),
        )
        .unwrap();
        let folded = values
            .iter()
            .zip(weights)
            .map(|(v, w)| v * w as i128)
            .sum::<i128>()
            .rem_euclid(ring.modulus_i128());
        assert_ne!(folded, 0);
    }
    assert_eq!(
        statement.check_witness(
            &scheme,
            &PolyVec::zero(ring.clone(), scheme.parameters.m1),
            &PolyVec::zero(ring, scheme.message_len())
        ),
        Err(Error::Witness)
    );
}

#[test]
fn approximate_range_rejection_is_exact_for_a_worst_case_witness() {
    // Bimodal rejection reproduces the Gaussian only if M exp(-|v|^2/(2 sigma^2)) >= 1 for the
    // projected displacement v = R e. Every coefficient of the approximate-range vector may sit
    // at linf_bound, so |e|^2 can reach n'd linf^2; sizing M from linf^2 alone is too small.
    // At the narrow width 1.55 * 2^8, sized from linf_bound alone, the check takes M = 9.
    let mut params = crate::params::toy_d64();
    params.log_sigma[3] = 8;
    let scheme = Abdlop::new([103; 32], params).unwrap();
    let p = &scheme.parameters;
    let values = vec![p.linf_bound as i128; p.n_prime * p.degree];
    let variance = (1.55 * 2f64.powi(p.log_sigma[3] as i32)).powi(2);
    let m = range_rejection_m(&scheme, false).unwrap() as f64;
    let linf_only = crate::params::range_rejection_constant(
        p.log_sigma[3],
        u128::from(p.linf_bound) * u128::from(p.linf_bound),
    )
    .unwrap() as f64;
    let mut worst = 0f64;
    for seed in 0..8u8 {
        let v = projections(&[seed; 32], false, &values).unwrap();
        let norm: f64 = v.iter().map(|x| (*x as f64).powi(2)).sum();
        let needed = (norm / (2.0 * variance)).exp();
        assert!(m >= needed, "M = {m} below exp(|v|^2/2s^2) = {needed}");
        worst = worst.max(needed);
    }
    assert!(
        linf_only < worst,
        "the infinity-norm sizing {linf_only} was sufficient"
    );
}

#[test]
fn approximate_range_rejection_is_exact_at_every_accepted_width() {
    // The parameter tool's per-slot rule may narrow sigma_4 down to the guard, the smallest
    // width whose M for n'd linf^2 is 2; the check accepts narrower widths with a larger M.
    // The prover's M always comes from n'd linf^2, which its witness check enforces, so for a
    // worst-case witness (every range coefficient at linf_bound) M >= exp(|v|^2 / (2 s^2)) at
    // every accepted width, the guard among them.
    let base = crate::params::toy_d64();
    let alpha = crate::params::range_rejection_constant;
    let squared = u128::from(base.linf_bound).pow(2) * (base.n_prime * base.degree) as u128;
    let guard = (0..=20).find(|t| alpha(*t, squared) == Ok(2)).unwrap();
    let values = vec![base.linf_bound as i128; base.n_prime * base.degree];
    let mut accepted = Vec::new();
    for t in 0..=guard + 2 {
        let mut p = base.clone();
        p.log_sigma[3] = t;
        let Ok(scheme) = Abdlop::new([105; 32], p) else {
            continue;
        };
        let m = range_rejection_m(&scheme, false).unwrap() as f64;
        assert_eq!(m, alpha(t, squared).unwrap() as f64);
        let variance = (1.55 * 2f64.powi(t as i32)).powi(2);
        for seed in 0..8u8 {
            let v = projections(&[seed; 32], false, &values).unwrap();
            let norm: f64 = v.iter().map(|x| (*x as f64).powi(2)).sum();
            let needed = (norm / (2.0 * variance)).exp();
            assert!(m >= needed, "t = {t}: M = {m} below {needed}");
        }
        accepted.push(t);
    }
    // Computed for toy_d64 (n'd linf^2 = 2048): the guard is 9 (M = 2), and the check
    // accepts 7 and 8 below it, with M = 6419 and 9; toy_d64's own width is 11.
    assert_eq!(guard, 9);
    assert_eq!(accepted, [7, 8, 9, 10, 11]);
    assert_eq!([7, 8].map(|t| alpha(t, squared)), [Ok(6419), Ok(9)]);
}

#[test]
fn a_two_prime_square_root_of_one_passes_the_sign_equations() {
    // Why range blocks need a prime modulus: with q = q1 q2, the constant u = (1 mod q1,
    // -1 mod q2) satisfies both sign equations and all integrality equations of the toolbox.
    let mut p = crate::params::toy_d64();
    p.prime_factors = vec![U256::from_u64(1048589), U256::from_u64(4194389)];
    p.degree = 128;
    p.m2 = 45;
    p.n_msis = 7;
    p.log_sigma = [15, 11, 11, 8];
    p.gamma = 123280;
    p.d_bits = 9;
    p.n_bin = 0;
    p.l2_rows.clear();
    p.l2_bounds_squared.clear();
    p.n_prime = 0;
    p.linf_bound = 0;
    p.mlwe_rank = 30;
    let scheme = Abdlop::new([104; 32], p).unwrap();
    let ring = scheme.ring().clone();
    let q = ring.modulus_i128();
    let u = (1..4194389i128)
        .map(|k| 1 + k * 1048589)
        .find(|x| (x + 1) % 4194389 == 0)
        .unwrap();
    assert!(u != 1 && u != q - 1);
    let extended = scheme.extend_messages(1).unwrap();
    let forms = forms(&scheme, &extended, &Statement::default()).unwrap();
    let mut messages = vec![Poly::zero(ring.clone()); extended.message_len()];
    // Packed signs b_e - X^(d/2) b_d with b_e = b_d = u.
    *messages.last_mut().unwrap() = Poly::constant(ring.clone(), u)
        .sub(&Poly::constant(ring.clone(), u).rotate((ring.degree() / 2) as i64))
        .unwrap();
    let witness = quad::interleave(
        &PolyVec::zero(ring.clone(), scheme.bounded_len()),
        &PolyVec::new(ring, messages).unwrap(),
    )
    .unwrap();
    assert_eq!(forms.eqs.len(), 2);
    for eq in &forms.eqs {
        assert!(eq.evaluate(&witness).unwrap().is_zero());
    }
    for eq in &forms.evals {
        assert_eq!(eq.evaluate(&witness).unwrap().coefficient_i128(0), Ok(0));
    }
}
