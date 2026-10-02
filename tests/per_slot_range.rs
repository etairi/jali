//! Per-slot sizing of the approximate range proof: `Requirements::approx_alpha_squared`
//! bounds $`\|e\|^2`$ of every honest range vector by
//! $`\sum_js_j\lceil F_j/p_j\rceil^2+\sum_v\mathrm{count}_v\,d'\beta_v^2`$ ($`s_j=d'`$ for an
//! `eq_mod_p`, 1 for a constant-coefficient clause) and is present exactly when that is below
//! $`n'd\beta_\infty^2`$. The tests recompute it from the integer bounds of `common::ring`, check
//! its serde forms, and fit parameters with the port's per-slot rule (`fit_per_slot`), whose
//! $`\sigma_4`$ is never wider than the default rule's and keeps the prover's rejection constant
//! at 2; the sets compile, prove and verify. Parameters come from the test port of the
//! parameter tool (`common::params`).
use jali::{
    math::{Poly, Ring, U256},
    params::{TboxParams, range_rejection_constant},
    quad::QuadEq,
    statement::{Norm, Requirements, Statement},
};
use std::collections::BTreeMap;

mod common;
use common::{
    SplitMix64,
    params::{
        extraction_bound, fit, fit_per_slot, fit_upward, lifting_threshold, next_prime_5_mod_8,
        per_slot_rule, per_slot_width, range_width,
    },
    ring::{VarBound, f_bound, f_bound_linf, integer_value, poly},
};

const Q: u64 = 1099511627917;
/// Both 12 and 13 divide the statement modulus, so the range condition does not apply.
const P: i128 = 156;
const CTX: &[u8] = b"jali-test/per-slot-range";

type Witness = BTreeMap<String, Vec<Poly>>;

/// One lifted constraint of a test statement: its form and whether it is a clause.
struct Lifted {
    form: QuadEq,
    clause: bool,
}

/// Over $`\mathbb Z_{156}[X]/(X^{128}+1)`$: `s` (four polynomials, $`\|s\|^2\le64`$), a binary
/// `x` and, with `beta`, `y` bounded by it in $`\ell_\infty`$, with a witness drawn from `seed`.
/// `build` adds the constraints, each made true on the witness.
struct Built {
    st: Statement,
    lifted: Vec<Lifted>,
    witness: Witness,
    bounds: Vec<VarBound>,
}
fn build(seed: u64, beta: Option<u64>, kinds: &[(&str, i128)]) -> Built {
    let ring = Ring::new(P, 128).unwrap();
    let mut rng = SplitMix64(seed);
    let mut st = Statement::new(ring.clone());
    st.var("s", 4, Norm::L2Squared(64)).unwrap();
    st.var("x", 1, Norm::Binary).unwrap();
    let mut s = vec![vec![0i128; 128]; 4];
    for _ in 0..64 {
        let (i, j) = (rng.below(4) as usize, rng.below(128) as usize);
        s[i][j] = rng.below(2) as i128 * 2 - 1;
    }
    let x = rng.binary(128);
    let mut w: Vec<Vec<i128>> = s.iter().cloned().chain([x.clone()]).collect();
    let mut bounds = vec![VarBound::Squared(64); 4];
    bounds.push(VarBound::Squared(128));
    let mut witness = Witness::from([
        (
            "s".into(),
            s.iter().map(|c| poly(&ring, c.clone())).collect(),
        ),
        ("x".into(), vec![poly(&ring, x)]),
    ]);
    if let Some(beta) = beta {
        st.var("y", 1, Norm::Linf(beta)).unwrap();
        let b = beta as i128;
        let y: Vec<i128> = (0..128)
            .map(|_| rng.below(2 * beta + 1) as i128 - b)
            .collect();
        witness.insert("y".into(), vec![poly(&ring, y.clone())]);
        w.push(y);
        bounds.push(VarBound::Linf(u128::from(beta)));
    }
    let mut lifted = Vec::new();
    for (kind, modulus) in kinds {
        let r = Ring::new(*modulus, 128).unwrap();
        let mut uniform = || poly(&r, rng.uniform(128, *modulus));
        let v = |name: &str, i: usize| st.variable_in(&r, name, i).unwrap();
        let form = match *kind {
            // ct(sigma(a) s_0 + b x_0 + c), or with y in place of s_0.
            "clause" => v("s", 0).scale(&uniform().auto()).unwrap(),
            "clause-y" => v("y", 0).scale(&uniform().auto()).unwrap(),
            "quadratic" => v("s", 1)
                .product_affine(&v("x", 0))
                .unwrap()
                .scale(&uniform())
                .unwrap()
                .add(&v("s", 2).scale(&uniform()).unwrap())
                .unwrap(),
            "linear" => v("s", 3).scale(&uniform()).unwrap(),
            "linear-small" => v("s", 3).scale(&Poly::constant(r.clone(), 1)).unwrap(),
            other => panic!("unknown kind {other}"),
        };
        let form = form.add(&v("x", 0).scale(&uniform()).unwrap()).unwrap();
        // Minus its value on the witness, reduced modulo the constraint's modulus.
        let value = poly(&r, integer_value(&form, &w));
        let form = form.add(&st.constant_in(value.neg()).unwrap()).unwrap();
        lifted.push(Lifted {
            form,
            clause: kind.starts_with("clause"),
        });
    }
    for l in &lifted {
        if l.clause {
            st.const_coeff_zero(l.form.clone()).unwrap();
        } else {
            st.eq_mod_p(l.form.clone()).unwrap();
        }
    }
    Built {
        st,
        lifted,
        witness,
        bounds,
    }
}

/// The per-slot sum recomputed from the integer bounds: $`d'=128`$ slots of
/// $`\lceil F_j/p_j\rceil`$ for an `eq_mod_p`, one for a clause, and $`128\beta^2`$ for `y`.
fn oracle(b: &Built, beta: Option<u64>) -> u128 {
    let mut sum = 0u128;
    for l in &b.lifted {
        let p = common::modulus(l.form.r0.ring()) as u128;
        let f = if b.bounds.len() > 5 {
            f_bound_linf(&l.form, &b.bounds)
        } else {
            let squared: Vec<u128> = b
                .bounds
                .iter()
                .map(|x| match x {
                    VarBound::Squared(v) => *v,
                    VarBound::Linf(_) | VarBound::LinfIn(..) => unreachable!(),
                })
                .collect();
            f_bound(&l.form, &squared)
        };
        let quotient = f.div_ceil(p);
        sum += if l.clause { 1 } else { 128 } * quotient * quotient;
    }
    sum + beta.map_or(0, |beta| 128 * u128::from(beta * beta))
}

/// Every carry and quotient of an honest witness is at most its per-row bound: the squared
/// norm of the honest range vector, from the witness map, is at most the per-slot sum.
fn honest_norm(b: &Built, params: &TboxParams) -> u128 {
    let compiled = b.st.compile(params.clone()).unwrap();
    let (s1, m) = compiled.map_witness(&b.witness).unwrap();
    let e = compiled
        .statement()
        .arp
        .as_ref()
        .unwrap()
        .evaluate(&s1, &m)
        .unwrap();
    e.entries()
        .iter()
        .flat_map(common::values)
        .map(|x| (x * x) as u128)
        .sum()
}

fn fitted(req: &Requirements, degree: usize, id: &str) -> TboxParams {
    let threshold = lifting_threshold(req, P as u128, range_width(req, degree));
    fit_upward(req, id, degree, (threshold as u64 + 1).max(Q)).1
}

#[test]
fn the_bound_is_the_per_slot_sum_and_present_only_below_the_default() {
    // (kinds, y's bound, whether the per-slot sum is below n' d linf_bound^2)
    type Case<'a> = (&'a [(&'a str, i128)], Option<u64>, bool);
    let cases: [Case; 6] = [
        // Clauses, a quadratic with committed carries and a linear form with quotients.
        (
            &[
                ("clause", 12),
                ("quadratic", 13),
                ("clause", 13),
                ("linear", 12),
            ],
            None,
            true,
        ),
        // One linear form: its k rows are dense at one bound, so the sums are equal.
        (&[("linear", 12)], None, false),
        // Two linear forms with different bounds.
        (&[("linear", 12), ("linear-small", 13)], None, true),
        // A clause on y and y's own rows.
        (&[("clause-y", 13)], Some(2), true),
        // y alone: its rows are dense at beta.
        (&[], Some(3), false),
        // One clause: one slot of the packed polynomial's d.
        (&[("clause", 13)], None, true),
    ];
    for (i, (kinds, beta, below)) in cases.into_iter().enumerate() {
        let b = build(0x2567_2400 + i as u64, beta, kinds);
        let sum = oracle(&b, beta);
        for degree in [64, 128] {
            let req = b.st.requirements(degree, U256::from_u64(Q)).unwrap();
            let default = req.linf_bound * req.linf_bound * (req.n_prime * degree) as u128;
            assert!(sum <= default, "case {i}");
            assert_eq!(sum < default, below, "case {i}, degree {degree}");
            assert_eq!(
                req.approx_alpha_squared,
                below.then_some(sum),
                "case {i}, degree {degree}"
            );
            if degree == 64 {
                let params = fitted(&req, degree, "per-slot-sum-test-only");
                let norm = honest_norm(&b, &params);
                assert!(norm <= sum && norm > 0, "case {i}: {norm} > {sum}");
            }
        }
    }
    // Nothing lifted and no l_inf variable: no range block, no bound.
    let b = build(0x2567_2410, None, &[]);
    let req = b.st.requirements(64, U256::from_u64(Q)).unwrap();
    assert_eq!((req.n_prime, req.approx_alpha_squared), (0, None));
}

#[test]
fn the_per_slot_width_keeps_the_rejection_constant_at_2() {
    // A quadratic, two clauses and a linear form, and one clause alone, at both degrees: the
    // per-slot width is at most the default and has M4 = 2, and one step narrower either the
    // per-slot width is larger or M4 exceeds 2. Each set compiles, and its honest range vector
    // is within the per-slot bound; proofs at degree 64.
    let kinds: [&[(&str, i128)]; 2] = [
        &[
            ("clause", 12),
            ("quadratic", 13),
            ("clause", 13),
            ("linear", 12),
        ],
        &[("clause", 13)],
    ];
    let mut narrower = 0;
    for (i, kinds) in kinds.into_iter().enumerate() {
        let b = build(0x2567_2420 + i as u64, None, kinds);
        for degree in [64, 128] {
            let req = b.st.requirements(degree, U256::from_u64(Q)).unwrap();
            let alpha = req.approx_alpha_squared.unwrap();
            let default = req.linf_bound * req.linf_bound * (req.n_prime * degree) as u128;
            let (t, slot, guard) = per_slot_rule(&req, degree, alpha);
            assert_eq!(per_slot_width(&req, degree), t);
            let t_default = range_width(&req, degree);
            assert!(t <= t_default, "case {i}, degree {degree}");
            assert_eq!(range_rejection_constant(t, default), Ok(2));
            assert_eq!(range_rejection_constant(t_default, default), Ok(2));
            assert!(slot > t - 1 || range_rejection_constant(t - 1, default) != Ok(2));
            assert!(range_rejection_constant(guard - 1, default) != Ok(2));
            narrower += usize::from(t < t_default);
            let threshold = lifting_threshold(&req, P as u128, t);
            let mut q = next_prime_5_mod_8((threshold as u64 + 1).max(Q));
            let params = loop {
                if let Ok(p) = fit_per_slot(&req, "per-slot-width-test-only", vec![q], degree, 26) {
                    break p;
                }
                q = next_prime_5_mod_8(q + 1);
            };
            assert_eq!(params.log_sigma[3], t);
            // The same set but the width: the per-slot rule only lowers log_sigma[3].
            let mut default_set =
                fit(&req, "per-slot-width-test-only", vec![q], degree, 26).unwrap();
            assert_eq!(default_set.log_sigma[3], t_default);
            default_set.log_sigma[3] = t;
            assert_eq!(default_set, params);
            assert!(honest_norm(&b, &params) <= alpha);
            if degree == 64 {
                let compiled = b.st.compile(params.clone()).unwrap();
                let bytes = compiled
                    .prove_bytes_with_seed([5; 32], &b.witness, CTX, [6; 32])
                    .unwrap();
                compiled.verify_bytes([5; 32], &bytes, CTX).unwrap();
                assert!(compiled.verify_bytes([5; 32], &bytes, b"other").is_err());
            }
            // A narrower width, a smaller extraction bound.
            let e = params.check().unwrap().approx_extraction_bound;
            assert_eq!(e, extraction_bound(t));
            assert!(t == t_default || e < extraction_bound(t_default));
        }
    }
    // Computed for these statements: the rule narrows sigma_4 in some of them.
    assert!(narrower > 0);
}

/// The serde forms of `approx_alpha_squared`: after `max_integer_coefficient` and before
/// `lifted_moduli`, left out of JSON when absent, a decimal string from 2^64 on, and always
/// written by postcard.
#[cfg(feature = "serde")]
mod serde_forms {
    use super::*;
    use serde_json::{Value, json};

    fn requirements() -> Requirements {
        let b = build(0x2567_2430, None, &[("clause", 12), ("linear", 13)]);
        b.st.requirements(64, U256::from_u64(Q)).unwrap()
    }

    #[test]
    fn json_writes_the_bound_before_lifted_moduli_and_leaves_absent_out() {
        let req = requirements();
        let alpha = req.approx_alpha_squared.unwrap();
        let text = serde_json::to_string(&req).unwrap();
        let (a, b, c) = (
            text.find("\"max_integer_coefficient\"").unwrap(),
            text.find(&format!("\"approx_alpha_squared\":{alpha},"))
                .unwrap(),
            text.find("\"lifted_moduli\"").unwrap(),
        );
        assert!(a < b && b < c);
        assert!(text.ends_with("}]}"));
        assert_eq!(serde_json::from_str::<Requirements>(&text).unwrap(), req);
        // Absent: the key is left out; null reads as absent.
        let mut none = req.clone();
        none.approx_alpha_squared = None;
        let value = serde_json::to_value(&none).unwrap();
        assert!(value.get("approx_alpha_squared").is_none());
        assert_eq!(
            serde_json::from_value::<Requirements>(value.clone()).unwrap(),
            none
        );
        let mut null = value;
        null["approx_alpha_squared"] = Value::Null;
        assert_eq!(serde_json::from_value::<Requirements>(null).unwrap(), none);
        // The number rule: a decimal string from 2^64 on, and either spelling reads.
        let mut wide = req.clone();
        wide.approx_alpha_squared = Some(u128::MAX);
        let value = serde_json::to_value(&wide).unwrap();
        assert_eq!(value["approx_alpha_squared"], json!(u128::MAX.to_string()));
        assert_eq!(serde_json::from_value::<Requirements>(value).unwrap(), wide);
        let mut small = serde_json::to_value(&req).unwrap();
        small["approx_alpha_squared"] = json!(alpha.to_string());
        assert_eq!(serde_json::from_value::<Requirements>(small).unwrap(), req);
        // Refused: a negative number, a float, a number of 2^64 or more, a leading zero.
        for bad in [
            json!(-1),
            json!(1.5),
            json!(u64::MAX as f64 * 2.0),
            json!("01"),
        ] {
            let mut v = serde_json::to_value(&req).unwrap();
            v["approx_alpha_squared"] = bad;
            assert!(serde_json::from_value::<Requirements>(v).is_err());
        }
    }

    #[test]
    fn postcard_always_writes_the_bound() {
        let req = requirements();
        for approx in [None, Some(5), Some(u128::MAX)] {
            let mut r = req.clone();
            r.approx_alpha_squared = approx;
            let bytes = postcard::to_allocvec(&r).unwrap();
            assert_eq!(postcard::from_bytes::<Requirements>(&bytes).unwrap(), r);
            // After max_integer_coefficient's 32 bytes: the option's tag and its varint.
            let tail = postcard::to_allocvec(&(&r.lifted_moduli, &r.linf)).unwrap();
            let field = postcard::to_allocvec(&approx).unwrap();
            let at = bytes.len() - tail.len() - field.len();
            assert_eq!(bytes[at..at + field.len()], field[..]);
            assert_eq!(
                bytes[at - 32..at],
                r.max_integer_coefficient.to_le_bytes()[..]
            );
            assert_eq!(field[0], u8::from(approx.is_some()));
            // One byte for None, then the varint: 5 in one byte, 2^128 - 1 in nineteen.
            let varint = match approx {
                None => 0,
                Some(5) => 1,
                Some(_) => 19,
            };
            assert_eq!(field.len(), 1 + varint);
            assert!(postcard::from_bytes::<Requirements>(&bytes[..bytes.len() - 1]).is_err());
        }
    }
}
