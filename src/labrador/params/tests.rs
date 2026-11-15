//! The hardness estimate, the encoding of the parameters and the limits of a level.
use super::*;

#[test]
fn block_sizes_invert_the_root_hermite_factor() {
    // docs/parameters.md: block size 346 is the smallest with delta_0 <= 1.0044.
    assert_eq!(block_size(1.0044), 346);
    for b in [50u32, 100, 346, 439, 600] {
        assert_eq!(block_size(delta_0(b)), b);
        assert!(delta_0(b + 1) < delta_0(b));
    }
    assert_eq!(block_size(1.5), 50);
    assert_eq!(block_size(1.0), u32::MAX);
}

#[test]
fn the_policy_tightens_with_the_bound_and_loosens_with_the_rank() {
    let policy = Policy::default();
    let q = 4294967197;
    let rank = (1..=64)
        .find(|k| policy.msis_ok(1e6, *k, 64, q, false))
        .unwrap();
    assert!(policy.msis_ok(1e6, rank + 1, 64, q, false));
    assert!(!policy.msis_ok(1e6, rank - 1, 64, q, false));
    assert!(!policy.msis_ok(1e7, rank - 1, 64, q, false));
    assert!(!policy.msis_ok(q as f64, 64, 64, q, false));
    assert!(policy.check().is_ok());
    assert!(
        Policy {
            min_block_size: 345
        }
        .check()
        .is_err()
    );
}

#[test]
fn lambda_makes_the_integer_aggregation_negligible() {
    for q in [65557u64, 4294967197, 68719476493, 281474976710597] {
        let l = lambda(q) as f64;
        assert!(l * (q as f64).log2() >= 128.0);
        assert!((l - 1.0) * (q as f64).log2() < 128.0 + 64.0);
    }
    assert_eq!(lambda(4294967197), 5);
}

fn sample_params() -> LevelParams {
    LevelParams {
        attempt: 3,
        last: true,
        kappa: 9,
        kappa1: 6,
        kappa2: 7,
        b: 311,
        b1: 1625,
        t1: 3,
        b2: 401,
        t2: 2,
        nu: 5,
        mu: 11,
        next_norm_sq: 123456789012345678901234567,
    }
}

#[test]
fn parameters_round_trip_and_refuse_a_bad_flag() {
    let p = sample_params();
    let bytes = p.encode();
    assert_eq!(LevelParams::decode(&bytes).unwrap(), p);
    let mut bad = bytes;
    bad[1] = 2;
    assert_eq!(LevelParams::decode(&bad), Err(Error::Encoding));
    assert_eq!(LevelParams::decode(&bytes[..54]), Err(Error::Encoding));
}

#[test]
fn layouts_follow_the_paper_without_padding_inside_z() {
    let p = sample_params();
    let shape = Shape {
        r: 4,
        n: 100,
        norm_sq: 1000,
    };
    let (m, n_next): (usize, usize) = (4 * 3 * 9 + 10 * (2 + 3), 20);
    let p = LevelParams {
        mu: m.div_ceil(n_next),
        ..p
    };
    let l = p.layout(&shape).unwrap();
    assert_eq!(l.pairs, 10);
    assert_eq!(l.off_g, 4 * 3 * 9);
    assert_eq!(l.off_h, l.off_g + 10 * 2);
    assert_eq!(l.m, m);
    assert_eq!(l.n_next, n_next);
    assert_eq!(l.r_next, 2 * 5 + p.mu);
    // nu must divide n, and mu is the least that holds v.
    for bad in [
        LevelParams { nu: 3, ..p },
        LevelParams { mu: p.mu + 1, ..p },
        LevelParams { mu: p.mu - 1, ..p },
        LevelParams { kappa: 0, ..p },
        LevelParams { t1: 0, ..p },
        LevelParams { b2: 1, ..p },
        LevelParams { nu: 0, ..p },
    ] {
        assert!(bad.layout(&shape).is_err(), "{bad:?}");
    }
}

#[test]
fn pair_indices_enumerate_the_upper_triangle() {
    for r in 1..9 {
        let mut next = 0;
        for i in 0..r {
            for j in i..r {
                assert_eq!(pair_index(i, j, r), next);
                next += 1;
            }
        }
        assert_eq!(next, r * (r + 1) / 2);
    }
}

#[test]
fn the_projection_condition_is_exact() {
    let q = 4294967197u64;
    // 337 * 125^2 * beta^2 <= 30 q^2.
    let max = 30 * u128::from(q) * u128::from(q) / (337 * 125 * 125);
    assert!(projection_condition(max, q).is_ok());
    assert!(projection_condition(max + 1, q).is_err());
    assert!(projection_condition(u128::MAX, q).is_err());
    assert_eq!(projection_max(100).unwrap(), 183);
}
