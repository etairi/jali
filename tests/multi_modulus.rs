//! Statements whose constraints have their own source moduli, built with
//! `Statement::{variable_in, constant_in}`. A prime and a composite modulus on shared variables
//! prove and verify, with carries and implicit quotients equal to independently computed
//! quotients; a witness true modulo one modulus but not the other is refused; the lifting check
//! is per constraint and exact at its boundary; the quotient bound is checked per constraint
//! kind under its own modulus, and quotients at the bound prove; a constraint at the proof
//! modulus stays native among lifted ones; forms of another degree and moduli above the proof
//! modulus are refused; witness values are the centred representatives of the statement
//! modulus; the range condition refuses statements with no witness in the statement ring,
//! exactly at its boundaries; and the requirements list the lifted moduli, in JSON only when
//! needed. Parameters come from the test port of the parameter tool (`common::params`); each
//! test fixes its seeds.
use jali::{
    Error,
    math::{Poly, Ring, U256, iso},
    params::{TboxParams, toy_d64},
    quad::QuadEq,
    statement::{LiftedModulus, Norm, Requirements, Statement},
};
use std::{collections::BTreeMap, sync::Arc};

mod common;
use common::{
    SplitMix64, narrow,
    params::{fit_upward, lifting_threshold, range_width},
    ring::{ceil_sqrt, f_bound, integer_value, poly},
};

/// The prime of `toy_d64`, where every search for a proof modulus starts.
const Q: u64 = 1099511627917;

/// The refusal of a statement that violates the range condition of the witness relation.
const OUT_OF_RANGE: Option<Error> =
    Some(Error::Parameter("variable range above statement modulus"));

type Witness = BTreeMap<String, Vec<Poly>>;

fn entry(modulus: u128, f: u128) -> LiftedModulus {
    LiftedModulus {
        modulus: U256::from_u128(modulus),
        max_integer_coefficient: U256::from_u128(f),
    }
}

/// The exact quotients of an integer value by `p`, after checking that `p` divides it.
fn quotients(f: &[i128], p: i128) -> Vec<i128> {
    assert!(f.iter().all(|x| x % p == 0), "not a multiple of {p}");
    f.iter().map(|x| x / p).collect()
}

/// Parameters for lifted requirements: the first prime that clears the compiler's lifting
/// threshold, recomputed independently, and fits.
fn lifted_params(req: &Requirements, statement_modulus: u128, id: &str) -> TboxParams {
    let threshold = lifting_threshold(req, statement_modulus, range_width(req, 64));
    fit_upward(req, id, 64, (threshold as u64 + 1).max(Q)).1
}

fn uniform(rng: &mut SplitMix64, ring: &Arc<Ring>) -> Poly {
    poly(ring, rng.uniform(ring.degree(), common::modulus(ring)))
}

/// Over `ring`, the quadratic $`a\,s_0x_0+b\,s_1+c`$ with uniform $`a,b`$ and $`c`$ such that
/// the integer witness $`(s,x)`$ satisfies it.
fn quadratic(
    st: &Statement,
    ring: &Arc<Ring>,
    rng: &mut SplitMix64,
    s: &[Vec<i128>],
    x: &[i128],
) -> QuadEq {
    let (a, b) = (uniform(rng, ring), uniform(rng, ring));
    let c = a
        .mul(
            &poly(ring, s[0].clone())
                .mul(&poly(ring, x.to_vec()))
                .unwrap(),
        )
        .unwrap()
        .add(&b.mul(&poly(ring, s[1].clone())).unwrap())
        .unwrap()
        .neg();
    st.variable_in(ring, "s", 0)
        .unwrap()
        .product_affine(&st.variable_in(ring, "x", 0).unwrap())
        .unwrap()
        .scale(&a)
        .unwrap()
        .add(&st.variable_in(ring, "s", 1).unwrap().scale(&b).unwrap())
        .unwrap()
        .add(&st.constant_in(c).unwrap())
        .unwrap()
}

/// Over `ring`, the constant-coefficient clause $`\mathrm{ct}(a\,s_2+b\,x_0+c)`$ with uniform
/// $`a,b`$ and a uniform $`c`$ whose constant coefficient makes the witness satisfy it.
fn clause(
    st: &Statement,
    ring: &Arc<Ring>,
    rng: &mut SplitMix64,
    s: &[Vec<i128>],
    x: &[i128],
) -> QuadEq {
    let (a, b) = (uniform(rng, ring), uniform(rng, ring));
    let v = a
        .mul(&poly(ring, s[2].clone()))
        .unwrap()
        .add(&b.mul(&poly(ring, x.to_vec())).unwrap())
        .unwrap()
        .coefficient_i128(0)
        .unwrap();
    let mut c = uniform(rng, ring);
    c.set_coefficient(0, -v).unwrap();
    st.variable_in(ring, "s", 2)
        .unwrap()
        .scale(&a)
        .unwrap()
        .add(&st.variable_in(ring, "x", 0).unwrap().scale(&b).unwrap())
        .unwrap()
        .add(&st.constant_in(c).unwrap())
        .unwrap()
}

/// Over `ring`, the linear $`a\,s_0+b\,s_1+c`$, with $`c`$ as in `quadratic`.
fn linear(st: &Statement, ring: &Arc<Ring>, rng: &mut SplitMix64, s: &[Vec<i128>]) -> QuadEq {
    let (a, b) = (uniform(rng, ring), uniform(rng, ring));
    let c = a
        .mul(&poly(ring, s[0].clone()))
        .unwrap()
        .add(&b.mul(&poly(ring, s[1].clone())).unwrap())
        .unwrap()
        .neg();
    st.variable_in(ring, "s", 0)
        .unwrap()
        .scale(&a)
        .unwrap()
        .add(&st.variable_in(ring, "s", 1).unwrap().scale(&b).unwrap())
        .unwrap()
        .add(&st.constant_in(c).unwrap())
        .unwrap()
}

/// The variables `s` (four polynomials, $`\|s\|^2\le512`$) and `x` (one binary polynomial)
/// over `ring`, and a witness drawn from `rng`, ternary for `s`: the statement, the named
/// witness and the integer witness in declaration order.
fn variables(ring: &Arc<Ring>, rng: &mut SplitMix64) -> (Statement, Witness, Vec<Vec<i128>>) {
    let mut st = Statement::new(ring.clone());
    st.var("s", 4, Norm::L2Squared(512)).unwrap();
    st.var("x", 1, Norm::Binary).unwrap();
    let s: Vec<Vec<i128>> = (0..4).map(|_| rng.ternary(128)).collect();
    let x = rng.binary(128);
    let witness = Witness::from([
        (
            "s".into(),
            s.iter().map(|c| poly(ring, c.clone())).collect(),
        ),
        ("x".into(), vec![poly(ring, x.clone())]),
    ]);
    let w = s.iter().cloned().chain([x]).collect();
    (st, witness, w)
}

/// Over the statement ring `ring` of degree 128: modulo 13 the quadratic of `quadratic`
/// (committed carries), and modulo 12 the clause of `clause` (one committed carry) and the
/// linear form of `linear` (implicit quotients). $`s_0,s_1,x_0`$ appear under both moduli.
/// The tests take $`\mathbb Z_{156}`$: both moduli divide 156, so the range condition does
/// not apply (and the values of `s`, at most $`\lfloor\sqrt{512}\rfloor=22`$ in absolute
/// value, would meet it). Over $`\mathbb Z_{13}`$ they would not.
fn two_moduli(ring: &Arc<Ring>, seed: u64) -> (Statement, Vec<QuadEq>, Witness, Vec<Vec<i128>>) {
    let r13 = Ring::new(13, 128).unwrap();
    let r12 = Ring::new(12, 128).unwrap();
    let mut rng = SplitMix64(seed);
    let (mut st, witness, w) = variables(ring, &mut rng);
    let (s, x) = (&w[..4], &w[4]);
    let forms = vec![
        quadratic(&st, &r13, &mut rng, s, x),
        clause(&st, &r12, &mut rng, s, x),
        linear(&st, &r12, &mut rng, s),
    ];
    st.eq_mod_p(forms[0].clone()).unwrap();
    st.const_coeff_zero(forms[1].clone()).unwrap();
    st.eq_mod_p(forms[2].clone()).unwrap();
    (st, forms, witness, w)
}

/// The squared bounds $`\beta_i`$ of `variables`: 512 for each `s_i`, $`1\cdot128`$ for `x`.
const BETA: [u128; 5] = [512, 512, 512, 512, 128];

/// $`\mathbb Z_{156}[X]/(X^{128}+1)`$, the statement ring of `two_moduli` in the tests.
fn r156() -> Arc<Ring> {
    Ring::new(156, 128).unwrap()
}

#[test]
fn a_prime_and_a_composite_modulus_on_shared_variables_prove_and_verify() {
    const CTX: &[u8] = b"jali-test/multi-modulus";
    let (st, forms, witness, w) = two_moduli(&r156(), 0x2567_1312);
    // Over Z_13, s_0 and s_1 under modulus 12 could take values outside [-6, 6].
    let (over_13, _, _, _) = two_moduli(&Ring::new(13, 128).unwrap(), 0x2567_1312);
    assert_eq!(
        over_13.requirements(64, U256::from_u64(Q)).err(),
        OUT_OF_RANGE
    );
    let req = st.requirements(64, U256::from_u64(Q)).unwrap();
    // Two carries for the quadratic, one for the clause; range rows for those and for the two
    // implicit quotients of the linear form.
    assert_eq!((req.m1, req.l, req.n_bin, req.n_prime), (10, 3, 2, 5));
    let f: Vec<u128> = forms.iter().map(|form| f_bound(form, &BETA)).collect();
    assert_eq!(
        req.lifted_moduli,
        vec![entry(12, f[1].max(f[2])), entry(13, f[0])]
    );
    assert_eq!(
        narrow(&req.max_integer_coefficient),
        *f.iter().max().unwrap()
    );
    let v = [f[0].div_ceil(13), f[1].div_ceil(12), f[2].div_ceil(12)];
    assert_eq!(req.linf_bound, *v.iter().max().unwrap());
    let params = lifted_params(&req, 156, "multi-modulus-test-only");
    let q = narrow(&params.prime_factors[0]) as i128;
    let compiled = st.compile(params.clone()).unwrap();
    let statement = compiled.statement();
    assert_eq!(
        (statement.quadratic.len(), statement.evaluation.len()),
        (2, 1)
    );
    let (s1, m) = compiled.map_witness(&witness).unwrap();
    // The quotients of the integer values by each constraint's own modulus, computed here.
    let proof_ring = Ring::new(q, 64).unwrap();
    let lifted = Ring::new(q, 128).unwrap();
    let split = |c: Vec<i128>| iso::split(&poly(&lifted, c), proof_ring.clone()).unwrap();
    let c1 = quotients(&integer_value(&forms[0], &w), 13);
    let c2 = quotients(&integer_value(&forms[1], &w)[..1], 12)[0];
    let c3 = quotients(&integer_value(&forms[2], &w), 12);
    for c in [&c1, &c3] {
        assert!(c.iter().any(|x| *x > 0) && c.iter().any(|x| *x < 0));
    }
    assert_ne!(c2, 0);
    let carries = [split(c1), vec![Poly::constant(proof_ring.clone(), c2)]].concat();
    // The messages are the committed carries; the range rows are those carries and then the
    // implicit quotients.
    assert_eq!(m.entries(), carries.as_slice());
    let arp = statement.arp.as_ref().unwrap().evaluate(&s1, &m).unwrap();
    assert_eq!(arp.entries(), [carries, split(c3)].concat().as_slice());
    let proof = compiled
        .prove_with_seed([1; 32], &witness, CTX, [2; 32])
        .unwrap();
    compiled.verify([1; 32], &proof, CTX).unwrap();
    assert!(compiled.verify([1; 32], &proof, b"other").is_err());
    let bytes = compiled
        .prove_bytes_with_seed([1; 32], &witness, CTX, [3; 32])
        .unwrap();
    compiled.verify_bytes([1; 32], &bytes, CTX).unwrap();
    // The same statement with the linear form's constant changed by one modulo 12: the
    // witness map refuses the witness, and the verifier the proof.
    let (mut other, _, _) = variables(&r156(), &mut SplitMix64(0));
    let r12 = Ring::new(12, 128).unwrap();
    let one = other.constant_in(Poly::constant(r12, 1)).unwrap();
    other.eq_mod_p(forms[0].clone()).unwrap();
    other.const_coeff_zero(forms[1].clone()).unwrap();
    other.eq_mod_p(forms[2].add(&one).unwrap()).unwrap();
    let other = other.compile(params).unwrap();
    assert_eq!(other.map_witness(&witness).err(), Some(Error::Witness));
    assert!(other.verify([1; 32], &proof, CTX).is_err());
}

/// Over a statement ring modulo 1009, which holds every value used below: `y` (five
/// polynomials, $`\|y\|^2\le1024`$) and, modulo 13 and then modulo 12, the clause
/// $`\mathrm{ct}(y_0)=v`$, the linear $`y_1=t`$ and the quadratic $`y_2y_3=u`$, with the right
/// sides those of `honest` reduced modulo that modulus. Returns the statement and its forms.
fn one_of_each(honest: &[Vec<i128>]) -> (Statement, Vec<QuadEq>) {
    let mut st = Statement::new(Ring::new(1009, 128).unwrap());
    st.var("y", 5, Norm::L2Squared(1024)).unwrap();
    let mut forms = Vec::new();
    for p in [13, 12] {
        let ring = Ring::new(p, 128).unwrap();
        let y = |i: usize| st.variable_in(&ring, "y", i).unwrap();
        let minus = |c: Poly| st.constant_in(c.neg()).unwrap();
        let product = poly(&ring, honest[2].clone())
            .mul(&poly(&ring, honest[3].clone()))
            .unwrap();
        forms.extend([
            y(0).add(&minus(Poly::constant(ring.clone(), honest[0][0])))
                .unwrap(),
            y(1).add(&minus(poly(&ring, honest[1].clone()))).unwrap(),
            y(2).product_affine(&y(3))
                .unwrap()
                .add(&minus(product))
                .unwrap(),
        ]);
    }
    for (i, form) in forms.iter().enumerate() {
        if i.is_multiple_of(3) {
            st.const_coeff_zero(form.clone()).unwrap();
        } else {
            st.eq_mod_p(form.clone()).unwrap();
        }
    }
    (st, forms)
}

#[test]
fn a_witness_true_modulo_one_modulus_only_is_refused() {
    let mut rng = SplitMix64(0x2567_1213);
    let honest: Vec<Vec<i128>> = (0..5).map(|_| rng.ternary(128)).collect();
    assert!(honest[3].iter().any(|x| *x != 0));
    let (st, forms) = one_of_each(&honest);
    let req = st.requirements(64, U256::from_u64(Q)).unwrap();
    // Two quadratics with two carries each and the two clauses' carries in one packed
    // polynomial; range rows for those and for the two linear forms' implicit quotients.
    assert_eq!((req.m1, req.l, req.n_prime), (10, 5, 9));
    let compiled = st
        .compile(lifted_params(
            &req,
            1009,
            "multi-modulus-refusals-test-only",
        ))
        .unwrap();
    let source = Ring::new(1009, 128).unwrap();
    let named = |y: &[Vec<i128>]| {
        Witness::from([(
            "y".into(),
            y.iter().map(|c| poly(&source, c.clone())).collect(),
        )])
    };
    compiled.map_witness(&named(&honest)).unwrap();
    // One coefficient moved by 13 or 12: each constraint modulo that number still holds, and
    // the one of the same kind modulo the other number fails. Forms 0, 1, 2 are modulo 13 and
    // 3, 4, 5 modulo 12, in the order clause, linear, quadratic.
    let holds = |form: usize, y: &[Vec<i128>], p: i128| {
        let f = integer_value(&forms[form], y);
        if form.is_multiple_of(3) {
            f[0] % p == 0
        } else {
            f.iter().all(|x| x % p == 0)
        }
    };
    for kind in 0..3 {
        for (delta, kept, broken) in [(13, 13, 12), (12, 12, 13)] {
            let mut bad = honest.clone();
            let (polynomial, coefficient) = [(0, 0), (1, 7), (2, 5)][kind];
            bad[polynomial][coefficient] += delta;
            let index = |p: i128| kind + if p == 13 { 0 } else { 3 };
            assert!(holds(index(kept), &bad, kept) && !holds(index(broken), &bad, broken));
            assert_eq!(
                compiled.map_witness(&named(&bad)).err(),
                Some(Error::Witness),
                "kind {kind}, broken modulo {broken}"
            );
        }
    }
}

/// The possession fixture's parameters (`tests/statement.rs`): $`q=1099511627917`$, blocks of
/// 16 and 4 bounded polynomials with squared bounds 128 and 64, two carries with range rows.
fn possession_params() -> TboxParams {
    let mut p = toy_d64();
    p.id = "quadratic-possession-test-only".into();
    p.m1 = 20;
    p.m2 = 55;
    p.n_msis = 15;
    p.l = 2;
    p.alpha_squared = 192;
    p.n_bin = 0;
    p.l2_rows = vec![16, 4];
    p.n_prime = 2;
    p.linf_bound = 13;
    p.log_sigma = [14, 12, 10, 13];
    p.d_bits = 6;
    p
}

/// `possession_params` with one packed carry polynomial and its range row, for up to 64 lifted
/// clauses (the MLWE rank one larger for the same $`m_2`$): the shape of `two_clauses`.
fn packed_possession_params() -> TboxParams {
    let mut p = possession_params();
    p.id = "packed-possession-test-only".into();
    p.l = 1;
    p.n_prime = 1;
    p.mlwe_rank += 1;
    p
}

/// Over a statement ring modulo 1009, `s` (8, 128) and `x` (2, 64), the shape of
/// `packed_possession_params`, with $`\mathrm{ct}(a\,s_0)=0`$ modulo `p_a`, where the
/// coefficients 0, 1 and 2 of $`a`$ equal `a` and the others are zero, and
/// $`\mathrm{ct}(s_1+1)=0`$ modulo `p_b`: their carries are coefficients 0 and 1 of one packed
/// polynomial.
fn two_clauses(p_a: U256, a: i128, p_b: U256) -> Statement {
    let mut st = Statement::new(Ring::new(1009, 128).unwrap());
    st.var("s", 8, Norm::L2Squared(128)).unwrap();
    st.var("x", 2, Norm::L2Squared(64)).unwrap();
    let ring_a = Ring::with_modulus(p_a, 128).unwrap();
    let mut coefficients = vec![0; 128];
    coefficients[..3].fill(a);
    let first = st
        .variable_in(&ring_a, "s", 0)
        .unwrap()
        .scale(&poly(&ring_a, coefficients))
        .unwrap();
    let ring_b = Ring::with_modulus(p_b, 128).unwrap();
    let second = st
        .variable_in(&ring_b, "s", 1)
        .unwrap()
        .add(&st.constant_in(Poly::constant(ring_b, 1)).unwrap())
        .unwrap();
    st.const_coeff_zero(first).unwrap();
    st.const_coeff_zero(second).unwrap();
    st
}

#[test]
fn the_lifting_check_is_exact_for_each_constraint() {
    // Soundness needs q > F_j + p_j * approx_extraction_bound for every lifted constraint; the
    // compiler checks q > 2 (F_j + p_j * linf_bound * psi). Clause a has a large F and a
    // modulus 2^20, clause b a small F: the largest modulus admitted for b is set by b's own
    // F, which a bound shared by both constraints would lower. Both carries share one packed
    // polynomial, and each clause keeps its own check.
    let params = packed_possession_params();
    let checked = params.check().unwrap();
    let q = narrow(&checked.q);
    let extraction = checked.approx_extraction_bound;
    let psi =
        (28.0 * 1.55 * 2f64.powi(params.log_sigma[3] as i32) / params.linf_bound as f64).ceil();
    let slack = u128::from(params.linf_bound) * psi as u128;
    // The largest p with 2 (f + p slack) < q.
    let largest = |f: u128| (q - 1 - 2 * f) / (2 * slack);
    let a = (1i128 << 19) - 1;
    let (f_a, f_b) = (ceil_sqrt(3 * (a * a) as u128 * 128), 1 + ceil_sqrt(128));
    let compile = |p_a: u128, p_b: u128| {
        two_clauses(U256::from_u128(p_a), a, U256::from_u128(p_b)).compile(params.clone())
    };
    let refused = Some(Error::Parameter("modulus lifting bound"));
    let p_a = 1 << 20;
    let p_b = largest(f_b);
    let req = two_clauses(U256::from_u128(p_a), a, U256::from_u128(p_b))
        .requirements(64, checked.q)
        .unwrap();
    assert_eq!(req.lifted_moduli, vec![entry(p_a, f_a), entry(p_b, f_b)]);
    // Each quotient bound is over the constraint's own modulus, not the statement's 1009.
    assert_eq!(req.linf_bound, f_a.div_ceil(p_a));
    assert!(req.linf_bound <= u128::from(params.linf_bound));
    compile(p_a, p_b).unwrap();
    assert_eq!(compile(p_a, p_b + 1).err(), refused);
    assert!(largest(f_a) < p_b, "a shared bound would refuse {p_b}");
    // With b fixed, the largest modulus admitted for a is set by a's F.
    let p_a = largest(f_a);
    compile(p_a, 13).unwrap();
    assert_eq!(compile(p_a + 1, 13).err(), refused);
    // At both boundaries the lifted relation cannot wrap around.
    assert!(f_b + p_b * extraction < q);
    assert!(f_a + p_a * extraction < q);
}

/// Over the ring modulo `ring`'s modulus: `s`, `x` as in `variables` and an unbounded `u`,
/// with the native quadratic $`a\,s_0x_0-u_0=0`$ and, lifted, the quadratic of `quadratic`
/// modulo 13 and the clause of `clause` modulo 12.
fn native_and_lifted(
    ring: &Arc<Ring>,
    native: i128,
) -> (Statement, Vec<QuadEq>, Witness, Vec<Vec<i128>>) {
    let mut rng = SplitMix64(0x2567_0071);
    let mut st = Statement::new(ring.clone());
    st.var("s", 4, Norm::L2Squared(512)).unwrap();
    st.var("x", 1, Norm::Binary).unwrap();
    st.var("u", 1, Norm::Unbounded).unwrap();
    let s: Vec<Vec<i128>> = (0..4).map(|_| rng.ternary(128)).collect();
    let x = rng.binary(128);
    let a = uniform(&mut rng, ring);
    let u = a
        .mul(
            &poly(ring, s[0].clone())
                .mul(&poly(ring, x.clone()))
                .unwrap(),
        )
        .unwrap()
        .add(&Poly::constant(ring.clone(), native))
        .unwrap();
    let first = st
        .variable("s", 0)
        .unwrap()
        .product_affine(&st.variable("x", 0).unwrap())
        .unwrap()
        .scale(&a)
        .unwrap()
        .add(
            &st.variable("u", 0)
                .unwrap()
                .scale(&Poly::constant(ring.clone(), -1))
                .unwrap(),
        )
        .unwrap();
    let r13 = Ring::new(13, 128).unwrap();
    let r12 = Ring::new(12, 128).unwrap();
    let forms = vec![
        first,
        quadratic(&st, &r13, &mut rng, &s, &x),
        clause(&st, &r12, &mut rng, &s, &x),
    ];
    st.eq_mod_p(forms[0].clone()).unwrap();
    st.eq_mod_p(forms[1].clone()).unwrap();
    st.const_coeff_zero(forms[2].clone()).unwrap();
    let witness = Witness::from([
        (
            "s".into(),
            s.iter().map(|c| poly(ring, c.clone())).collect(),
        ),
        ("x".into(), vec![poly(ring, x.clone())]),
        ("u".into(), vec![u]),
    ]);
    let w = s.into_iter().chain([x]).collect();
    (st, forms, witness, w)
}

#[test]
fn a_constraint_at_the_proof_modulus_stays_native_among_lifted_ones() {
    const CTX: &[u8] = b"jali-test/multi-modulus-native";
    // The shape does not depend on the proof modulus as long as the native constraint has it:
    // probe it at another prime, fit, and rebuild at the prime chosen.
    let probe = Ring::new(1152921504606847009, 128).unwrap();
    let req = native_and_lifted(&probe, 0)
        .0
        .requirements(64, probe.modulus())
        .unwrap();
    // Messages: u (two components), two carries modulo 13, one modulo 12; no range row and no
    // bound for the native constraint.
    assert_eq!((req.m1, req.l, req.n_prime), (10, 5, 3));
    let threshold = lifting_threshold(&req, 0, range_width(&req, 64));
    let (q, params) = fit_upward(
        &req,
        "multi-modulus-native-test-only",
        64,
        (threshold as u64 + 1).max(Q),
    );
    let ring = Ring::new(q as i128, 128).unwrap();
    let (st, forms, witness, w) = native_and_lifted(&ring, 0);
    assert_eq!(st.requirements(64, ring.modulus()).unwrap(), req);
    // At any other proof modulus the native constraint is lifted, and its unbounded u refused.
    assert_eq!(
        st.requirements(64, ring.modulus().wrapping_add(&U256::from_u8(2)))
            .err(),
        Some(Error::Parameter("unbounded variable in lifted relation"))
    );
    let compiled = st.compile(params.clone()).unwrap();
    let statement = compiled.statement();
    assert_eq!(
        (statement.quadratic.len(), statement.evaluation.len()),
        (4, 1)
    );
    assert_eq!(statement.arp.as_ref().unwrap().rows, 3);
    let (s1, m) = compiled.map_witness(&witness).unwrap();
    let proof_ring = Ring::new(q as i128, 64).unwrap();
    let u = iso::split(&witness["u"][0], proof_ring.clone()).unwrap();
    let c1 = quotients(&integer_value(&forms[1], &w), 13);
    let c2 = quotients(&integer_value(&forms[2], &w)[..1], 12)[0];
    let carries = [
        iso::split(&poly(&ring, c1), proof_ring.clone()).unwrap(),
        vec![Poly::constant(proof_ring.clone(), c2)],
    ]
    .concat();
    assert_eq!(m.entries(), [u, carries.clone()].concat().as_slice());
    let arp = statement.arp.as_ref().unwrap().evaluate(&s1, &m).unwrap();
    assert_eq!(arp.entries(), carries.as_slice());
    let proof = compiled
        .prove_with_seed([5; 32], &witness, CTX, [6; 32])
        .unwrap();
    compiled.verify([5; 32], &proof, CTX).unwrap();
    assert!(compiled.verify([5; 32], &proof, b"other").is_err());
    // The native constraint with u_0 off by one is false modulo q only.
    let (_, _, off, _) = native_and_lifted(&ring, 1);
    assert_eq!(compiled.map_witness(&off).err(), Some(Error::Witness));
}

#[test]
fn forms_of_another_degree_and_moduli_above_the_proof_modulus_are_refused() {
    let r13 = Ring::new(13, 128).unwrap();
    let mut st = Statement::new(r13.clone());
    st.var("s", 8, Norm::L2Squared(128)).unwrap();
    st.var("x", 2, Norm::L2Squared(64)).unwrap();
    // Another degree, whatever the modulus.
    for ring in [Ring::new(12, 64).unwrap(), Ring::new(13, 256).unwrap()] {
        let mismatch = Some(Error::RingMismatch);
        assert_eq!(st.variable_in(&ring, "s", 0).err(), mismatch);
        assert_eq!(
            st.constant_in(Poly::constant(ring.clone(), 1)).err(),
            mismatch
        );
        let form = QuadEq::zero(ring, 10).unwrap();
        assert_eq!(st.clone().eq_mod_p(form.clone()).err(), mismatch);
        assert_eq!(st.clone().const_coeff_zero(form).err(), mismatch);
    }
    // The same degree with another modulus is a constraint of its own; `constant` still takes
    // the statement ring only.
    let r12 = Ring::new(12, 128).unwrap();
    let form = st.variable_in(&r12, "s", 0).unwrap();
    st.clone().eq_mod_p(form).unwrap();
    assert_eq!(
        st.constant(Poly::constant(r12, 1)).err(),
        Some(Error::RingMismatch)
    );
    // A modulus below 2 has no ring, so no form.
    for m in [1, 0, -12] {
        assert_eq!(
            Ring::new(m, 128).err(),
            Some(Error::Parameter("ring modulus"))
        );
    }
    for m in [U256::ZERO, U256::ONE] {
        assert_eq!(
            Ring::with_modulus(m, 128).err(),
            Some(Error::Parameter("ring modulus"))
        );
    }
    // A modulus above q fails the lifting check, and a multiple of q has no inverse.
    let params = packed_possession_params();
    let q = params.check().unwrap().q;
    let above = [
        q.wrapping_add(&U256::ONE),
        q.wrapping_add(&U256::from_u8(2)),
        common::params::power_minus(200, 75),
        common::params::power_minus(256, 435),
    ];
    for p_b in above {
        assert_eq!(
            two_clauses(U256::from_u8(13), 1, p_b)
                .compile(params.clone())
                .err(),
            Some(Error::Parameter("modulus lifting bound"))
        );
    }
    for multiple in [2u8, 3] {
        assert_eq!(
            two_clauses(
                U256::from_u8(13),
                1,
                q.wrapping_mul(&U256::from_u8(multiple))
            )
            .compile(params.clone())
            .err(),
            Some(Error::Parameter("noninvertible constraint modulus"))
        );
    }
}

#[test]
fn witness_values_are_the_centred_representatives_of_the_statement_modulus() {
    // Over Z_17, where ||x||^2 <= 64 keeps every value in [-8, 8]: x_0 = 7 stands for 7, so
    // ct(x_0) = 7 holds modulo 12 and ct(x_0) = 6 does not; x_0 = 16 stands for -1, so
    // ct(x_0) = 11 holds and ct(x_0) = 4, true of the canonical 16, does not. The carry is the
    // quotient of the integer value by 12, with the constant -v centred modulo 12 as its ring
    // stores it. Over Z_13, where 7 would stand for -6 while a proof accepts the integer 7
    // (see the next test), the statement is refused.
    let r12 = Ring::new(12, 128).unwrap();
    let statement = |source: &Arc<Ring>, v: i128| {
        let mut st = Statement::new(source.clone());
        st.var("x", 5, Norm::L2Squared(64)).unwrap();
        let form = st
            .variable_in(&r12, "x", 0)
            .unwrap()
            .add(&st.constant_in(Poly::constant(r12.clone(), -v)).unwrap())
            .unwrap();
        st.const_coeff_zero(form).unwrap();
        st
    };
    let r17 = Ring::new(17, 128).unwrap();
    // v = 6 has the largest F of the values used (8 + 6), so its parameters serve them all.
    let req = statement(&r17, 6)
        .requirements(64, U256::from_u64(Q))
        .unwrap();
    let params = lifted_params(&req, 17, "multi-modulus-values-test-only");
    for (stored, value, accepted, refused) in [(7, 7, 7, 6), (16, -1, 11, 4)] {
        let mut x = vec![Poly::zero(r17.clone()); 5];
        x[0] = Poly::constant(r17.clone(), stored);
        assert_eq!(x[0].coefficient_i128(0), Ok(value));
        // The refused value is the one the canonical representative satisfies, where it
        // differs from the centred one.
        assert_eq!((stored - refused) % 12 == 0, stored != value);
        let witness = Witness::from([("x".into(), x)]);
        let (_, m) = statement(&r17, accepted)
            .compile(params.clone())
            .unwrap()
            .map_witness(&witness)
            .unwrap();
        let f = value + common::ring::centred(-accepted, 12);
        assert_eq!(f % 12, 0);
        assert_eq!(m.entries()[0].coefficient_i128(0), Ok(f / 12));
        assert_eq!(
            statement(&r17, refused)
                .compile(params.clone())
                .unwrap()
                .map_witness(&witness)
                .err(),
            Some(Error::Witness)
        );
    }
    let r13 = Ring::new(13, 128).unwrap();
    for v in [6, 7] {
        let st = statement(&r13, v);
        assert_eq!(st.requirements(64, U256::from_u64(Q)).err(), OUT_OF_RANGE);
        assert_eq!(st.compile(params.clone()).err(), OUT_OF_RANGE);
    }
}

/// Over $`\mathbb Z_p`$, `x` (five polynomials, $`\|x\|^2\le64`$) and the clause
/// $`\mathrm{ct}(x_0)=7`$ modulo each of `moduli`.
fn sevens(p: i128, moduli: &[i128]) -> Statement {
    let mut st = Statement::new(Ring::new(p, 128).unwrap());
    st.var("x", 5, Norm::L2Squared(64)).unwrap();
    for m in moduli {
        let ring = Ring::new(*m, 128).unwrap();
        let form = st
            .variable_in(&ring, "x", 0)
            .unwrap()
            .add(&st.constant_in(Poly::constant(ring, -7)).unwrap())
            .unwrap();
        st.const_coeff_zero(form).unwrap();
    }
    st
}

/// Over `ring`, `x` as in `sevens` and an unbounded `u`, with $`u_0=100`$ over
/// $`\mathbb Z_q`$, which is native when `q` is the proof modulus.
fn hundred(ring: Arc<Ring>, q: i128) -> Statement {
    let mut st = Statement::new(ring);
    st.var("x", 5, Norm::L2Squared(64)).unwrap();
    st.var("u", 1, Norm::Unbounded).unwrap();
    let rq = Ring::new(q, 128).unwrap();
    let form = st
        .variable_in(&rq, "u", 0)
        .unwrap()
        .add(&st.constant_in(Poly::constant(rq, -100)).unwrap())
        .unwrap();
    st.eq_mod_p(form).unwrap();
    st
}

/// The named witness of `sevens` and `hundred` over `ring`: $`x_0=7`$ and, with `u`,
/// $`u_0=100`$.
fn seven_and_hundred(ring: &Arc<Ring>, u: bool) -> Witness {
    let mut x = vec![Poly::zero(ring.clone()); 5];
    x[0] = Poly::constant(ring.clone(), 7);
    let mut witness = Witness::from([("x".into(), x)]);
    if u {
        witness.insert("u".into(), vec![Poly::constant(ring.clone(), 100)]);
    }
    witness
}

/// The first prime from `Q` on that fits the shape of a statement whose only constraint is
/// native, and the statement built at it: the shape does not depend on that prime.
fn native_at(build: impl Fn(i128) -> Statement, id: &str) -> (Statement, TboxParams) {
    let probe = 1152921504606847009;
    let req = build(probe)
        .requirements(64, U256::from_u128(probe as u128))
        .unwrap();
    assert_eq!(req.n_prime, 0);
    let (q, params) = fit_upward(&req, id, 64, Q);
    let st = build(q as i128);
    assert_eq!(st.requirements(64, U256::from_u64(q)).unwrap(), req);
    (st, params)
}

#[test]
fn statements_without_a_witness_in_the_statement_ring_are_refused() {
    // Three statements over Z_13 with the proof modulus Q and no witness: the clauses
    // ct(x_0) = 7 modulo 12 and modulo 13, the clause ct(x_0) = 7 over Z_Q (native), and
    // u_0 = 100 over Z_Q (native). No value in [-6, 6] satisfies them, but the integers 7
    // (49 <= 64) and 100 do, and without the range condition proofs from those would verify:
    // a proof binds integers within the norms (|x_0| <= 8, any u) and each constraint modulo
    // its own modulus, while Z_13 holds only [-6, 6].
    let q = Q as i128;
    for v in -6..=6i128 {
        assert!((v - 7) % 12 != 0 || (v - 7) % 13 != 0);
        assert!((v - 7) % q != 0 && (v - 100) % q != 0);
    }
    let r13 = Ring::new(13, 128).unwrap();
    for st in [sevens(13, &[12, 13]), sevens(13, &[q]), hundred(r13, q)] {
        assert_eq!(st.requirements(64, U256::from_u64(Q)).err(), OUT_OF_RANGE);
        assert_eq!(st.compile(toy_d64()).err(), OUT_OF_RANGE);
    }
    // Over statement moduli that hold the values, the same constraints prove and verify with
    // the same integers: Z_17 holds [-8, 8], and Z_(2^61-1) every value modulo the proof
    // modulus.
    const CTX: &[u8] = b"jali-test/multi-modulus-ranges";
    let r17 = Ring::new(17, 128).unwrap();
    let lifted = sevens(17, &[12, 13]);
    let req = lifted.requirements(64, U256::from_u64(Q)).unwrap();
    let (bounded, bounded_params) = native_at(|q| sevens(17, &[q]), "ranges-bounded-test-only");
    let wide = Ring::new((1 << 61) - 1, 128).unwrap();
    let (unbounded, unbounded_params) =
        native_at(|q| hundred(wide.clone(), q), "ranges-unbounded-test-only");
    for (st, params, witness) in [
        (
            lifted,
            lifted_params(&req, 17, "ranges-lifted-test-only"),
            seven_and_hundred(&r17, false),
        ),
        (bounded, bounded_params, seven_and_hundred(&r17, false)),
        (unbounded, unbounded_params, seven_and_hundred(&wide, true)),
    ] {
        let compiled = st.compile(params).unwrap();
        let proof = compiled
            .prove_with_seed([9; 32], &witness, CTX, [10; 32])
            .unwrap();
        compiled.verify([9; 32], &proof, CTX).unwrap();
    }
}

/// `requirements` at `Q` for a statement over $`\mathbb Z_p`$ with `x` (five polynomials,
/// $`\|x\|^2\le`$ `bound`) and a binary `y`, declared in that order or, with `y_first`, the
/// other way round, and the clause `form` builds over the ring modulo `modulus`.
fn clause_over(
    p: i128,
    bound: u64,
    y_first: bool,
    modulus: i128,
    form: impl Fn(&Statement, &Arc<Ring>) -> QuadEq,
) -> Result<Requirements, Error> {
    let mut st = Statement::new(Ring::new(p, 128).unwrap());
    if y_first {
        st.var("y", 1, Norm::Binary).unwrap();
    }
    st.var("x", 5, Norm::L2Squared(bound)).unwrap();
    if !y_first {
        st.var("y", 1, Norm::Binary).unwrap();
    }
    let ring = Ring::new(modulus, 128).unwrap();
    let form = form(&st, &ring);
    st.const_coeff_zero(form).unwrap();
    st.requirements(64, U256::from_u64(Q))
}

#[test]
fn the_range_condition_is_exact_at_its_boundaries() {
    let x0 = |st: &Statement, r: &Arc<Ring>| st.variable_in(r, "x", 0).unwrap();
    let y0 = |st: &Statement, r: &Arc<Ring>| st.variable_in(r, "y", 0).unwrap();
    let scaled_x0 = |c: i128| {
        move |st: &Statement, r: &Arc<Ring>| {
            st.variable_in(r, "x", 0)
                .unwrap()
                .scale(&Poly::constant(r.clone(), c))
                .unwrap()
                .add(&st.variable_in(r, "y", 0).unwrap())
                .unwrap()
        }
    };
    let product = |a: &'static str, b: &'static str| {
        move |st: &Statement, r: &Arc<Ring>| {
            st.variable_in(r, a, 0)
                .unwrap()
                .product_affine(&st.variable_in(r, b, 0).unwrap())
                .unwrap()
        }
    };
    let ok = |r: Result<Requirements, Error>| r.map(|_| ()).err();
    // A bound B keeps values within floor(sqrt(B)); the statement ring holds them if twice
    // that is below p. Not when a modulus divides the other way round (156 does not divide 12).
    for (p, bound, modulus, admitted) in [
        (13, 64, 12, false),
        (16, 64, 12, false),
        (17, 64, 12, true),
        (17, 80, 12, true),
        (17, 81, 12, false),
        (19, 81, 12, true),
        (12, 64, 156, false),
        (155, 6084, 12, false),
        (157, 6084, 12, true),
    ] {
        let expected = if admitted { None } else { OUT_OF_RANGE };
        assert_eq!(
            ok(clause_over(p, bound, false, modulus, x0)),
            expected,
            "p {p}"
        );
    }
    // A constraint whose modulus divides p needs no range: 156 = 12 * 13, 26 = 2 * 13 and
    // 24 = 2 * 12.
    for (p, bound, modulus) in [
        (156, 6084, 12),
        (156, 6084, 13),
        (26, 400, 13),
        (24, 400, 12),
    ] {
        assert_eq!(ok(clause_over(p, bound, false, modulus, x0)), None);
    }
    assert_eq!(ok(clause_over(26, 400, false, 12, x0)), OUT_OF_RANGE);
    // Binary variables fit every statement ring, also Z_2; the tag clauses of
    // `tests/tag_preimage.rs` use only those, while its long s appears only modulo the
    // statement modulus.
    assert_eq!(ok(clause_over(13, 2048, false, 12, y0)), None);
    assert_eq!(ok(clause_over(2, 64, false, 3, y0)), None);
    assert_eq!(ok(clause_over(2, 64, false, 3, x0)), OUT_OF_RANGE);
    // A variable counts only with a nonzero coefficient: 12 x_0 vanishes modulo 12.
    assert_eq!(ok(clause_over(13, 64, false, 12, scaled_x0(12))), None);
    assert_eq!(
        ok(clause_over(13, 64, false, 12, scaled_x0(11))),
        OUT_OF_RANGE
    );
    // Quadratic terms count, at either index of the pair.
    for y_first in [false, true] {
        let refusal = ok(clause_over(13, 64, y_first, 12, product("x", "y")));
        assert_eq!(refusal, OUT_OF_RANGE, "y first: {y_first}");
    }
    assert_eq!(ok(clause_over(13, 64, false, 12, product("y", "y"))), None);
    // An unbounded u in a native constraint at Q takes any value modulo Q: the statement
    // modulus must be Q itself (Q divides it) or at least Q.
    let q = Q as i128;
    for (p, admitted) in [
        (13, false),
        (q - 2, false),
        (q, true),
        (q + 2, true),
        ((1 << 61) - 1, true),
    ] {
        let st = hundred(Ring::new(p, 128).unwrap(), q);
        let expected = if admitted { None } else { OUT_OF_RANGE };
        assert_eq!(
            ok(st.requirements(64, U256::from_u64(Q))),
            expected,
            "p {p}"
        );
    }
}

/// Over $`\mathbb Z_{1009}`$, `x` (five polynomials, $`\|x\|^2\le64`$) and `z` (one,
/// $`\|z\|^2\le9`$), one constraint whose bound is $`F=4p_j`$: the clause
/// $`\mathrm{ct}(5x_0+4\epsilon)`$ or the full-ring linear $`5x_0+4\epsilon`$ modulo
/// $`p_j=11`$, or the full-ring quadratic $`\epsilon(3z_0^2+1)`$ modulo $`p_j=7`$, with
/// $`\epsilon=`$ `sign`; and the witness $`x_0=8\epsilon`$, $`z_0=3`$, on which the
/// constraint's value is $`\epsilon F`$, so that its carries or quotients reach
/// $`4\epsilon`$.
fn at_the_bound(kind: usize, sign: i128) -> (Statement, Witness, i128) {
    let source = Ring::new(1009, 128).unwrap();
    let mut st = Statement::new(source.clone());
    st.var("x", 5, Norm::L2Squared(64)).unwrap();
    st.var("z", 1, Norm::L2Squared(9)).unwrap();
    let modulus = if kind == 2 { 7 } else { 11 };
    let ring = Ring::new(modulus, 128).unwrap();
    let c = |v: i128| Poly::constant(ring.clone(), v);
    let form = if kind == 2 {
        let z = st.variable_in(&ring, "z", 0).unwrap();
        let square = z.product_affine(&z).unwrap().scale(&c(3 * sign)).unwrap();
        square.add(&st.constant_in(c(sign)).unwrap())
    } else {
        let x = st.variable_in(&ring, "x", 0).unwrap().scale(&c(5)).unwrap();
        x.add(&st.constant_in(c(4 * sign)).unwrap())
    };
    if kind == 0 {
        st.const_coeff_zero(form.unwrap()).unwrap();
    } else {
        st.eq_mod_p(form.unwrap()).unwrap();
    }
    let mut x = vec![Poly::zero(source.clone()); 5];
    x[0] = Poly::constant(source.clone(), 8 * sign);
    let witness = Witness::from([
        ("x".into(), x),
        ("z".into(), vec![Poly::constant(source, 3)]),
    ]);
    (st, witness, modulus)
}

#[test]
fn quotients_at_their_bound_prove_and_a_smaller_bound_is_refused() {
    // The compiler checks ceil(F_j / p_j) against linf_bound for each lifted constraint, with
    // its own modulus p_j, not the statement's 1009 (which would give a bound of 1): for a
    // clause (one committed carry), a linear constraint (implicit quotients) and a quadratic
    // one (committed carries). Values of +-F give carries of +-4, which prove and verify at
    // linf_bound 4; parameters with linf_bound 3 are refused.
    const CTX: &[u8] = b"jali-test/multi-modulus-quotient-bound";
    for kind in 0..3 {
        let (st, _, modulus) = at_the_bound(kind, 1);
        let req = st.requirements(64, U256::from_u64(Q)).unwrap();
        assert_eq!(narrow(&req.max_integer_coefficient), 4 * modulus as u128);
        assert_eq!(req.linf_bound, 4, "kind {kind}");
        let params = lifted_params(&req, 1009, "quotient-bound-test-only");
        for sign in [1, -1] {
            let (st, witness, _) = at_the_bound(kind, sign);
            assert_eq!(st.requirements(64, U256::from_u64(Q)).unwrap(), req);
            let compiled = st.compile(params.clone()).unwrap();
            let (s1, m) = compiled.map_witness(&witness).unwrap();
            let rows = compiled
                .statement()
                .arp
                .as_ref()
                .unwrap()
                .evaluate(&s1, &m)
                .unwrap();
            // The first row is coefficient 0 of the first carry or quotient; all others vanish.
            assert_eq!(rows.entries()[0].coefficient_i128(0), Ok(4 * sign));
            let rest = rows.entries()[0].coefficients()[1..]
                .iter()
                .all(|c| *c == U256::ZERO);
            assert!(
                rest && rows.entries()[1..].iter().all(Poly::is_zero),
                "kind {kind}"
            );
            let proof = compiled
                .prove_with_seed([11; 32], &witness, CTX, [12; 32])
                .unwrap();
            compiled.verify([11; 32], &proof, CTX).unwrap();
        }
        let mut tight = req.clone();
        tight.linf_bound -= 1;
        let q = narrow(&params.prime_factors[0]) as u64;
        let (_, tight) = fit_upward(&tight, "quotient-bound-tight-test-only", 64, q);
        assert_eq!(
            st.compile(tight).err(),
            Some(Error::Parameter("carry or quotient bound")),
            "kind {kind}"
        );
    }
}

/// The serde forms of `Requirements::lifted_moduli`: JSON leaves an empty list out, so that the
/// text of a statement with one modulus does not change, and writes the entries by the number
/// rule; postcard, which reads fields by position, always writes it.
#[cfg(feature = "serde")]
mod serde_forms {
    use super::*;
    use serde_json::json;

    const KEYS: [&str; 9] = [
        "m1",
        "l",
        "alpha_squared",
        "n_bin",
        "l2_rows",
        "l2_bounds_squared",
        "n_prime",
        "linf_bound",
        "max_integer_coefficient",
    ];

    #[test]
    fn json_lists_lifted_moduli_only_when_a_constraint_has_its_own() {
        // One modulus: the nine fields as before, and text that re-serializes byte for byte.
        let mut single = Statement::new(Ring::new(13, 128).unwrap());
        single.var("s", 4, Norm::L2Squared(512)).unwrap();
        let form = single.variable("s", 0).unwrap();
        single.eq_mod_p(form).unwrap();
        let req = single.requirements(64, U256::from_u64(Q)).unwrap();
        assert!(req.lifted_moduli.is_empty());
        let value = serde_json::to_value(&req).unwrap();
        let keys: Vec<&str> = value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        let mut expected = KEYS.to_vec();
        expected.sort_unstable();
        assert_eq!(keys, expected);
        let text = concat!(
            r#"{"m1":8,"l":0,"alpha_squared":512,"n_bin":0,"l2_rows":[8],"#,
            r#""l2_bounds_squared":[512],"n_prime":2,"linf_bound":2,"#,
            r#""max_integer_coefficient":"340282366920938463463374607431768211456"}"#
        );
        let read: Requirements = serde_json::from_str(text).unwrap();
        assert!(read.lifted_moduli.is_empty());
        assert_eq!(serde_json::to_string(&read).unwrap(), text);
        // Two moduli: the entries, ascending, after the other fields.
        let (st, forms, _, _) = two_moduli(&r156(), 0x2567_1312);
        let req = st.requirements(64, U256::from_u64(Q)).unwrap();
        let f: Vec<u128> = forms.iter().map(|form| f_bound(form, &BETA)).collect();
        let value = serde_json::to_value(&req).unwrap();
        assert_eq!(
            value["lifted_moduli"],
            json!([
                {"modulus": 12, "max_integer_coefficient": f[1].max(f[2]) as u64},
                {"modulus": 13, "max_integer_coefficient": f[0] as u64}
            ])
        );
        let text = serde_json::to_string(&req).unwrap();
        assert!(text.ends_with(r#"}]}"#) && text.contains(r#","lifted_moduli":[{"#));
        assert_eq!(serde_json::from_str::<Requirements>(&text).unwrap(), req);
        // Entries of 2^64 and more are decimal strings.
        let mut wide = req.clone();
        wide.lifted_moduli = vec![
            entry(3, u128::from(u64::MAX)),
            LiftedModulus {
                modulus: common::params::power_minus(256, 435),
                max_integer_coefficient: U256::from_u128(1 << 64),
            },
        ];
        let value = serde_json::to_value(&wide).unwrap();
        assert_eq!(
            value["lifted_moduli"],
            json!([
                {"modulus": 3, "max_integer_coefficient": u64::MAX},
                {
                    "modulus": concat!(
                        "11579208923731619542357098500868790785326998466564",
                        "0564039457584007913129639501"
                    ),
                    "max_integer_coefficient": "18446744073709551616"
                }
            ])
        );
        assert_eq!(serde_json::from_value::<Requirements>(value).unwrap(), wide);
    }

    #[test]
    fn postcard_always_writes_lifted_moduli() {
        let (st, _, _, _) = two_moduli(&r156(), 0x2567_1312);
        let req = st.requirements(64, U256::from_u64(Q)).unwrap();
        let bytes = postcard::to_allocvec(&req).unwrap();
        assert_eq!(postcard::from_bytes::<Requirements>(&bytes).unwrap(), req);
        // The count, then each modulus and bound as 32 little-endian bytes, then the absent
        // `linf`, one byte (tests/linf.rs round-trips a present one).
        assert_eq!(bytes[bytes.len() - 1], 0);
        let tail = &bytes[bytes.len() - 130..bytes.len() - 1];
        assert_eq!(tail[0], 2);
        for (i, m) in req.lifted_moduli.iter().enumerate() {
            assert_eq!(tail[1 + 64 * i..33 + 64 * i], m.modulus.to_le_bytes()[..]);
            assert_eq!(
                tail[33 + 64 * i..65 + 64 * i],
                m.max_integer_coefficient.to_le_bytes()[..]
            );
        }
        let mut single = req.clone();
        single.lifted_moduli.clear();
        let bytes = postcard::to_allocvec(&single).unwrap();
        assert_eq!(bytes[bytes.len() - 2..], [0, 0]);
        assert_eq!(
            postcard::from_bytes::<Requirements>(&bytes).unwrap(),
            single
        );
        assert!(postcard::from_bytes::<Requirements>(&bytes[..bytes.len() - 1]).is_err());
    }
}
