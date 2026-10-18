//! Two-sided exact Euclidean bounds ([`Norm::L2SquaredRange`]): $`\mathrm{min}\le\|s\|^2\le
//! \mathrm{max}`$. The upper bound is the exact-norm proof of `L2Squared(max)`; for the lower bound
//! the compiler commits one binary proof-ring polynomial $`y`$ in the Ajtai part and proves
//! $`\mathrm{ct}(\sum_c\sigma(s_c)s_c)-\sum_bc_by_b-\mathrm{min}=0`$ at the proof modulus. The
//! tests check the requirements (one more Ajtai row and binary row, and $`d`$ more of
//! `alpha_squared`, than `L2Squared(max)`, with the same exact-norm rows and bound, wherever the
//! block is placed), the declaration checks (a lower bound of 0 or above the upper one refused),
//! the witness map ($`\|s\|^2`$ at `min` and at `max` admitted, one below and one above refused,
//! $`s=0`$ refused for `min = 1`, and the bits it writes), proofs at both proof degrees from both
//! parts of the commitment, which verify and report `Extraction::L2SquaredRange`, that the
//! parameters of one bound do not compile the other, and that a witness below `min`, which the
//! plain bound proves, has no binary $`y`$ that satisfies the compiled equations, so that the
//! prover refuses it with the witness map bypassed. Parameters come from the test port of the
//! parameter tool (`common::params`); each test fixes its seeds.
use jali::{
    Error,
    abdlop::Abdlop,
    math::{Poly, PolyVec, Ring, U256},
    params::TboxParams,
    statement::{Compiled, Extraction, Norm, Placement, Statement},
    tbox,
};
use std::collections::BTreeMap;

mod common;
use common::{
    SplitMix64,
    params::{fit_upward, lifting_threshold, range_width},
    ring::{integer_value, poly},
};

/// The prime of `toy_d64`, where every search for a proof modulus starts.
const Q: u64 = 1099511627917;
/// The statement modulus.
const P: i128 = 1313;
const CTX: &[u8] = b"jali-test/l2-range";
const PP: [u8; 32] = [7; 32];
const MIN: u64 = 5;
const MAX: u64 = 64;
const RANGE: Norm = Norm::L2SquaredRange { min: MIN, max: MAX };

type Witness = BTreeMap<String, Vec<Poly>>;

/// The integer witness, in declaration order: $`a_0..a_4`$ with 16 ternary coefficients each,
/// $`s_0,s_1`$ of degree 128 with $`\|s\|^2=`$ `norm` (one coefficient $`\pm2`$ when `norm` is at
/// least 4, the rest $`\pm1`$, at distinct positions), and a binary $`x`$.
fn integers(norm: u64, seed: u64) -> Vec<Vec<i128>> {
    let mut rng = SplitMix64(seed);
    let mut w = Vec::new();
    for _ in 0..5 {
        let mut c = vec![0i128; 128];
        for _ in 0..16 {
            c[rng.below(128) as usize] = rng.below(3) as i128 - 1;
        }
        w.push(c);
    }
    let mut s = vec![0i128; 256];
    let mut left = norm as i128;
    let mut free: Vec<usize> = (0..256).collect();
    while left > 0 {
        let i = free.swap_remove(rng.below(free.len() as u64) as usize);
        let v = if left >= 4 { 2 } else { 1 };
        s[i] = if rng.below(2) == 0 { v } else { -v };
        left -= v * v;
    }
    w.push(s[..128].to_vec());
    w.push(s[128..].to_vec());
    w.push(rng.binary(128));
    w
}

/// The named witness of `build` (`with_x`) or of `alone`.
fn named(w: &[Vec<i128>], with_x: bool) -> Witness {
    let ring = Ring::new(P, 128).unwrap();
    let polys = |range: std::ops::Range<usize>| {
        range.map(|i| poly(&ring, w[i].clone())).collect::<Vec<_>>()
    };
    let mut out = BTreeMap::from([
        ("a".to_string(), polys(0..5)),
        ("s".to_string(), polys(5..7)),
    ]);
    if with_x {
        out.insert("x".to_string(), polys(7..8));
    }
    out
}

/// Over $`\mathbb Z_{1313}[X]/(X^{128}+1)`$: `a` (five polynomials, $`\|a\|^2\le640`$, in the
/// Ajtai part), `s` (two, under `norm`, placed as `placement` says) and a binary `x`, with the
/// constraint $`g\,s_0+h\,s_1+i\,x+j\,a_0+t=0`$ modulo 1313 for sparse public $`g,h,i,j`$ and the
/// $`t`$ that makes `w` satisfy it.
fn build(norm: Norm, placement: Placement, w: &[Vec<i128>]) -> (Statement, Witness) {
    let ring = Ring::new(P, 128).unwrap();
    let mut st = Statement::new(ring.clone());
    st.var("a", 5, Norm::L2Squared(640)).unwrap();
    st.var_placed("s", 2, norm, placement).unwrap();
    st.var("x", 1, Norm::Binary).unwrap();
    let mut rng = SplitMix64(0x12a4_6e01);
    let mut sparse = || {
        let mut c = vec![0i128; 128];
        for _ in 0..4 {
            c[rng.below(128) as usize] = rng.below(13) as i128 - 6;
        }
        poly(&ring, c)
    };
    let mut form = st.variable("a", 0).unwrap().scale(&sparse()).unwrap();
    for (name, i) in [("s", 0), ("s", 1), ("x", 0)] {
        let term = st.variable(name, i).unwrap().scale(&sparse()).unwrap();
        form = form.add(&term).unwrap();
    }
    let value = poly(&ring, integer_value(&form, w));
    st.eq_mod_p(form.add(&st.constant(value.neg()).unwrap()).unwrap())
        .unwrap();
    (st, named(w, true))
}

/// `a` and `s` alone, both in the Ajtai part, without a constraint: the norms are all that a
/// witness must meet.
fn alone(norm: Norm) -> Statement {
    let mut st = Statement::new(Ring::new(P, 128).unwrap());
    st.var("a", 5, Norm::L2Squared(640)).unwrap();
    st.var("s", 2, norm).unwrap();
    st
}

/// `st` compiled at proof degree `degree` with parameters that the test port of the parameter
/// tool fits to it, at the first prime above the lifting threshold of the statement modulus.
fn compiled(st: &Statement, degree: usize) -> (Compiled, TboxParams) {
    let req = st.requirements(degree, U256::from_u64(Q)).unwrap();
    let threshold = lifting_threshold(&req, P as u128, range_width(&req, degree));
    let (_, params) = fit_upward(
        &req,
        "jali-test/l2-range",
        degree,
        (threshold as u64 + 1).max(Q),
    );
    (st.compile(params.clone()).unwrap(), params)
}

/// The weights $`c_b=\lfloor W/2^b\rfloor-\lfloor W/2^{b+1}\rfloor`$ of $`W`$, from their
/// definition.
fn weights(w: u64) -> Vec<i128> {
    (0..u64::BITS - w.leading_zeros())
        .map(|b| ((w >> b) - (w >> (b + 1))) as i128)
        .collect()
}

#[test]
fn the_weights_of_the_lower_bound_reach_exactly_the_slack_interval() {
    for w in 0..=1u64 << 10 {
        let c = weights(w);
        assert_eq!(c.iter().sum::<i128>(), w as i128);
        let mut reached = vec![false; w as usize + 1];
        for subset in 0u32..1 << c.len() {
            let sum: i128 = (0..c.len())
                .filter(|b| subset >> b & 1 == 1)
                .map(|b| c[b])
                .sum();
            reached[sum as usize] = true;
        }
        assert!(reached.iter().all(|r| *r), "W = {w}");
    }
}

#[test]
fn requirements_add_one_binary_row_to_the_plain_bound() {
    let w = integers(MIN, 1);
    for placement in [Placement::Ajtai, Placement::Bdlop] {
        let (range, _) = build(RANGE, placement, &w);
        let (plain, _) = build(Norm::L2Squared(MAX), placement, &w);
        for degree in [64, 128] {
            let r = range.requirements(degree, U256::from_u64(Q)).unwrap();
            let p = plain.requirements(degree, U256::from_u64(Q)).unwrap();
            assert_eq!(r.m1, p.m1 + 1, "{placement:?}, degree {degree}");
            assert_eq!(r.n_bin, p.n_bin + 1);
            assert_eq!(r.alpha_squared, p.alpha_squared + degree as u128);
            assert_eq!(
                (r.l, r.l2_rows.clone(), r.l2_bounds_squared.clone()),
                (p.l, p.l2_rows.clone(), p.l2_bounds_squared.clone())
            );
            assert_eq!(r.l2_bounds_squared, vec![640, MAX]);
            assert_eq!(
                (r.n_prime, r.linf_bound, r.max_integer_coefficient),
                (p.n_prime, p.linf_bound, p.max_integer_coefficient)
            );
            let blocks = range.blocks(degree).unwrap();
            assert_eq!(blocks[1].norm, RANGE);
            assert_eq!(blocks[1].rows, plain.blocks(degree).unwrap()[1].rows);
        }
    }
}

#[test]
fn declarations_refuse_a_lower_bound_of_zero_or_above_the_upper_one() {
    let mut st = Statement::new(Ring::new(P, 128).unwrap());
    for (min, max) in [(0, 5), (6, 5), (1, 0)] {
        assert_eq!(
            st.var("s", 1, Norm::L2SquaredRange { min, max }),
            Err(Error::Parameter("variable declaration")),
            "min {min}, max {max}"
        );
    }
    st.var("s", 1, Norm::L2SquaredRange { min: 5, max: 5 })
        .unwrap();
}

#[test]
fn the_witness_map_admits_the_range_and_writes_the_lower_bound_bits() {
    for degree in [64, 128] {
        // the constant of the constraint depends on the witness: one statement per witness
        for norm in [MIN, 17, MAX] {
            let (st, witness) = build(RANGE, Placement::Ajtai, &integers(norm, 3 + norm));
            let (c, params) = compiled(&st, degree);
            assert_eq!(params.l2_bounds_squared, vec![640, MAX]);
            let (s1, _) = c.map_witness(&witness).unwrap();
            // the lower-bound polynomial is the last of the Ajtai part
            let y = s1.entries().last().unwrap().coefficients_i128().unwrap();
            let w = weights(MAX - MIN);
            assert!(y.iter().all(|b| *b == 0 || *b == 1));
            assert!(y[w.len()..].iter().all(|b| *b == 0));
            let sum: i128 = w.iter().zip(y.iter()).map(|(c, b)| c * b).sum();
            assert_eq!(sum, (norm - MIN) as i128, "norm {norm}, degree {degree}");
        }
        // one below min and one above max are refused
        for norm in [MIN - 1, MAX + 1] {
            let (st, witness) = build(RANGE, Placement::Ajtai, &integers(norm, 40 + norm));
            let (c, _) = compiled(&st, degree);
            assert_eq!(c.map_witness(&witness), Err(Error::Witness), "norm {norm}");
        }
    }
    // s = 0 is refused for min = 1
    let one = Norm::L2SquaredRange { min: 1, max: MAX };
    let (st, witness) = build(one, Placement::Ajtai, &integers(0, 4));
    let (c, _) = compiled(&st, 64);
    assert_eq!(c.map_witness(&witness), Err(Error::Witness));
}

#[test]
fn range_blocks_prove_and_verify_at_both_degrees_from_both_parts() {
    for placement in [Placement::Ajtai, Placement::Bdlop] {
        for degree in [64, 128] {
            for norm in [MIN, MAX] {
                let (st, witness) = build(RANGE, placement, &integers(norm, 50 + norm));
                let (c, _) = compiled(&st, degree);
                let proof = c.prove_with_seed(PP, &witness, CTX, [9; 32]).unwrap();
                c.verify(PP, &proof, CTX).unwrap();
                assert!(c.verify(PP, &proof, b"another context").is_err());
                assert_eq!(
                    c.extraction_bound("s"),
                    Ok(Extraction::L2SquaredRange { min: MIN, max: MAX })
                );
            }
        }
    }
}

#[test]
fn the_parameters_of_one_bound_do_not_compile_the_other() {
    let w = integers(MIN, 5);
    let (range, _) = build(RANGE, Placement::Ajtai, &w);
    let (plain, _) = build(Norm::L2Squared(MAX), Placement::Ajtai, &w);
    let (_, range_params) = compiled(&range, 64);
    let (_, plain_params) = compiled(&plain, 64);
    assert!(range.compile(plain_params).is_err());
    assert!(plain.compile(range_params).is_err());
}

#[test]
fn a_witness_below_the_lower_bound_has_no_binary_lower_bound_polynomial() {
    let honest = integers(MIN, 6);
    let mut zero = honest.clone();
    zero[5].fill(0);
    zero[6].fill(0);
    // s = 0 meets the plain bound, which proves it
    let (plain, _) = compiled(&alone(Norm::L2Squared(MAX)), 64);
    let proof = plain
        .prove_with_seed(PP, &named(&zero, false), CTX, [3; 32])
        .unwrap();
    plain.verify(PP, &proof, CTX).unwrap();
    // Under the range, the mapped witness of an honest s of norm MIN with s replaced by 0: the
    // lower-bound equation reads -sum_b c_b y_b - MIN, in [-MAX, -MIN], for every binary y.
    // The Ajtai part holds a (10 rows at degree 64), s (4) and y.
    let (range, params) = compiled(&alone(RANGE), 64);
    let (s1, m) = range.map_witness(&named(&honest, false)).unwrap();
    assert_eq!(s1.len(), 15);
    tbox::prove_with_seed(
        &Abdlop::new(PP, params.clone()).unwrap(),
        range.statement(),
        &s1,
        &m,
        CTX,
        [5; 32],
    )
    .unwrap();
    let scheme = Abdlop::new(PP, params).unwrap();
    let ring = s1.ring().clone();
    let bits = weights(MAX - MIN).len();
    let mut rng = SplitMix64(7);
    let mut candidates = vec![vec![0i128; 64], {
        let mut all = vec![0i128; 64];
        all[..bits].fill(1);
        all
    }];
    for _ in 0..6 {
        let mut y = rng.binary(64);
        y[bits..].fill(0);
        candidates.push(y);
    }
    for y in candidates {
        let mut entries = s1.entries().to_vec();
        for row in &mut entries[10..14] {
            *row = Poly::zero(ring.clone());
        }
        entries[14] = Poly::new(ring.clone(), y).unwrap();
        let forged = PolyVec::new(ring.clone(), entries).unwrap();
        assert_eq!(
            tbox::prove_with_seed(&scheme, range.statement(), &forged, &m, CTX, [5; 32]),
            Err(Error::Witness)
        );
    }
}

#[cfg(feature = "serde")]
#[test]
fn the_range_norm_has_a_json_form() {
    let json = serde_json::to_value(RANGE).unwrap();
    assert_eq!(
        json,
        serde_json::json!({"l2_squared_range": {"min": MIN, "max": MAX}})
    );
    assert_eq!(serde_json::from_value::<Norm>(json).unwrap(), RANGE);
    let blocks = build(RANGE, Placement::Bdlop, &integers(MIN, 8))
        .0
        .blocks(64)
        .unwrap();
    assert_eq!(
        serde_json::to_value(&blocks[1]).unwrap(),
        serde_json::json!({
            "name": "s",
            "rows": 4,
            "norm": {"l2_squared_range": {"min": MIN, "max": MAX}},
            "placement": "bdlop"
        })
    );
}
