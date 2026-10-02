//! Carry packing: the carries of all lifted `const_coeff_zero` clauses share committed
//! polynomials, $`d`$ to a polynomial of the proof ring $`\mathbb Z_q[X]/(X^d+1)`$. Clause $`i`$
//! holds coefficient $`t=i\bmod d`$ of packed polynomial $`\lfloor i/d\rfloor`$ and reads it
//! through $`\mathrm{ct}(X^{-t}C)=C_t`$. The tests check the shapes for
//! $`m\in\{1,2,d,d+1,2d+1\}`$ clauses at both proof degrees, each compiled equation's read
//! coefficient against the negacyclic sign rule, the read identity itself, the carries and range
//! rows of the witness map, clauses with two moduli in one packed polynomial, which prove and
//! verify, the per-clause lifting boundary, and where the packed polynomials sit among other
//! carries. Parameters come from the test port of the parameter tool (`common::params`); the
//! soundness of a false clause against a prover that skips its witness check is in
//! `src/tbox/lifting_soundness_tests.rs`.
use jali::{
    Error,
    math::{Poly, Ring, U256, iso},
    params::TboxParams,
    quad::QuadEq,
    statement::{Norm, Requirements, Statement},
};
use std::{collections::BTreeMap, sync::Arc};

mod common;
use common::{
    SplitMix64, narrow,
    params::{fit_upward, lifting_threshold, range_width},
    ring::{centred, integer_value, negacyclic, poly},
};

/// The prime of `toy_d64`, where every search for a proof modulus starts.
const Q: u64 = 1099511627917;
/// The statement modulus: both clause moduli, 12 and 13, divide it, so the range condition
/// does not apply.
const P: i128 = 156;
const CTX: &[u8] = b"jali-test/packed-carries";

type Witness = BTreeMap<String, Vec<Poly>>;

/// The first prime above the compiler's lifting threshold that the port fits.
fn fitted(req: &Requirements, degree: usize, id: &str) -> TboxParams {
    let threshold = lifting_threshold(req, P as u128, range_width(req, degree));
    fit_upward(req, id, degree, (threshold as u64 + 1).max(Q)).1
}

/// Over $`\mathbb Z_{156}[X]/(X^{128}+1)`$: `s` (four polynomials, $`\|s\|^2\le64`$), a binary
/// `x`, and `m` clauses $`\mathrm{ct}(\sigma(a_i)s_{i\bmod4}+b_ix_0+c_i)\equiv0`$ modulo 13 for
/// even $`i`$ and 12 for odd $`i`$, with uniform $`a_i,b_i`$ and $`c_i`$ chosen so that the
/// witness satisfies each. Returns the statement, its clause forms, the named witness and the
/// integer witness in declaration order.
struct Clauses {
    st: Statement,
    forms: Vec<QuadEq>,
    witness: Witness,
    w: Vec<Vec<i128>>,
}
fn clauses(m: usize, seed: u64) -> Clauses {
    let ring = Ring::new(P, 128).unwrap();
    let mut rng = SplitMix64(seed);
    let mut st = Statement::new(ring.clone());
    st.var("s", 4, Norm::L2Squared(64)).unwrap();
    st.var("x", 1, Norm::Binary).unwrap();
    // At most 64 entries of +-1.
    let mut s = vec![vec![0i128; 128]; 4];
    for _ in 0..64 {
        let (i, j) = (rng.below(4) as usize, rng.below(128) as usize);
        s[i][j] = rng.below(2) as i128 * 2 - 1;
    }
    let x = rng.binary(128);
    let w: Vec<Vec<i128>> = s.iter().cloned().chain([x.clone()]).collect();
    let mut forms = Vec::new();
    for i in 0..m {
        let modulus = if i % 2 == 0 { 13 } else { 12 };
        let r = Ring::new(modulus, 128).unwrap();
        let a = poly(&r, rng.uniform(128, modulus));
        let b = poly(&r, rng.uniform(128, modulus));
        let form = st
            .variable_in(&r, "s", i % 4)
            .unwrap()
            .scale(&a.auto())
            .unwrap()
            .add(&st.variable_in(&r, "x", 0).unwrap().scale(&b).unwrap())
            .unwrap();
        // The constant coefficient's value modulo the clause's modulus, negated; noise
        // elsewhere, which the clause does not read.
        let value = integer_value(&form, &w)[0];
        let mut c = poly(&r, rng.uniform(128, modulus));
        c.set_coefficient(0, -value).unwrap();
        forms.push(form.add(&st.constant_in(c).unwrap()).unwrap());
    }
    for form in &forms {
        st.const_coeff_zero(form.clone()).unwrap();
    }
    let witness = Witness::from([
        (
            "s".into(),
            s.iter().map(|c| poly(&ring, c.clone())).collect(),
        ),
        ("x".into(), vec![poly(&ring, x)]),
    ]);
    Clauses {
        st,
        forms,
        witness,
        w,
    }
}

/// The exact integer carries of the clauses: constant coefficient over the clause modulus.
fn carries(c: &Clauses) -> Vec<i128> {
    c.forms
        .iter()
        .map(|form| {
            let p = common::modulus(form.r0.ring());
            let f = integer_value(form, &c.w)[0];
            assert_eq!(f % p, 0);
            f / p
        })
        .collect()
}

/// $`-p\,X^{-t}`$ in the proof ring from the sign rule, independently of `Poly::rotate`:
/// $`X^{-t}=-X^{d-t}`$ for $`0<t<d`$.
fn read_coefficient(ring: &Arc<Ring>, p: i128, t: usize) -> Poly {
    let mut c = vec![0i128; ring.degree()];
    if t == 0 {
        c[0] = -p;
    } else {
        c[ring.degree() - t] = p;
    }
    poly(ring, c)
}

#[test]
fn packed_rows_follow_the_clause_count_at_both_degrees() {
    for degree in [64, 128] {
        for m in [1, 2, degree, degree + 1, 2 * degree + 1] {
            let c = clauses(m, 0x2567_2100 + m as u64);
            let rows = m.div_ceil(degree);
            let req = c.st.requirements(degree, U256::from_u64(Q)).unwrap();
            let k = 128 / degree;
            assert_eq!(
                (req.m1, req.l, req.n_prime),
                (5 * k, rows, rows),
                "degree {degree}, {m} clauses"
            );
            let params = fitted(&req, degree, "packed-shape-test-only");
            let q = narrow(&params.prime_factors[0]) as i128;
            let compiled = c.st.compile(params.clone()).unwrap();
            let statement = compiled.statement();
            assert_eq!(statement.evaluation.len(), m);
            assert!(statement.quadratic.is_empty());
            assert_eq!(statement.arp.as_ref().unwrap().rows, rows);
            // Clause i reads packed polynomial i/d, message i/d after the Ajtai part, through
            // -p_i X^-(i mod d), and no other message.
            let proof_ring = Ring::new(q, degree).unwrap();
            let first = 2 * req.m1;
            for (i, equation) in statement.evaluation.iter().enumerate() {
                let p = if i % 2 == 0 { 13 } else { 12 };
                let messages: Vec<(u16, &Poly)> = equation
                    .r1
                    .entries()
                    .filter(|(j, _)| usize::from(*j) >= first)
                    .collect();
                assert_eq!(messages.len(), 1, "clause {i}");
                assert_eq!(usize::from(messages[0].0), first + 2 * (i / degree));
                assert_eq!(
                    *messages[0].1,
                    read_coefficient(&proof_ring, p, i % degree),
                    "clause {i}"
                );
            }
            // Carry i at coefficient i mod d of packed polynomial i/d, the other coefficients
            // 0; the range rows are the packed polynomials themselves.
            let (s1, packed) = compiled.map_witness(&c.witness).unwrap();
            let mut expected = carries(&c);
            assert!(expected.iter().any(|x| *x != 0));
            expected.resize(rows * degree, 0);
            let expected: Vec<Poly> = expected
                .chunks(degree)
                .map(|chunk| poly(&proof_ring, chunk.to_vec()))
                .collect();
            assert_eq!(packed.entries(), expected.as_slice());
            let arp = statement
                .arp
                .as_ref()
                .unwrap()
                .evaluate(&s1, &packed)
                .unwrap();
            assert_eq!(arp.entries(), expected.as_slice());
        }
    }
}

#[test]
fn one_clause_compiles_to_one_unrotated_carry_polynomial() {
    // One lifted clause: its carry polynomial holds the carry at coefficient 0, the equation
    // reads it through -p (no rotation), and its range row is that polynomial.
    for degree in [64, 128] {
        let c = clauses(1, 0x2567_2101);
        let req = c.st.requirements(degree, U256::from_u64(Q)).unwrap();
        assert_eq!((req.l, req.n_prime), (1, 1));
        let params = fitted(&req, degree, "packed-one-test-only");
        let q = narrow(&params.prime_factors[0]) as i128;
        let compiled = c.st.compile(params).unwrap();
        let equation = &compiled.statement().evaluation[0];
        let ring = Ring::new(q, degree).unwrap();
        let carry = equation.r1.entries().last().unwrap();
        assert_eq!(usize::from(carry.0), 2 * req.m1);
        assert_eq!(*carry.1, Poly::constant(ring.clone(), -13));
        let (_, m) = compiled.map_witness(&c.witness).unwrap();
        assert_eq!(m.entries(), [Poly::constant(ring, carries(&c)[0])]);
    }
}

#[test]
fn the_constant_coefficient_of_x_to_the_minus_t_times_c_is_coefficient_t() {
    // ct(X^-t C) = C_t for every t < d in Z_q[X]/(X^d + 1), with the library's product and with
    // an independent negacyclic schoolbook product on centred values.
    let mut rng = SplitMix64(0x2567_2102);
    for degree in [64, 128] {
        let ring = Ring::new(Q as i128, degree).unwrap();
        let c = poly(&ring, rng.uniform(degree, Q as i128));
        let values = common::values(&c);
        for t in 0..degree {
            let read = Poly::constant(ring.clone(), 1).rotate(-(t as i64));
            assert_eq!(
                read.mul(&c).unwrap().coefficient_i128(0),
                Ok(values[t]),
                "t = {t}"
            );
            let mut monomial = vec![0i128; degree];
            if t == 0 {
                monomial[0] = 1;
            } else {
                monomial[degree - t] = -1;
            }
            assert_eq!(common::values(&read), monomial);
            let product = negacyclic(&monomial, &values);
            assert_eq!(centred(product[0], Q as i128), values[t]);
        }
    }
}

#[test]
fn clauses_with_two_moduli_share_packed_polynomials_and_prove() {
    // 65 clauses at degree 64: two packed polynomials, the first with carries modulo 13 and 12
    // in alternate slots, the second with the last clause's carry at coefficient 0.
    let c = clauses(65, 0x2567_2103);
    let req = c.st.requirements(64, U256::from_u64(Q)).unwrap();
    assert_eq!((req.l, req.n_prime), (2, 2));
    assert_eq!(
        req.lifted_moduli
            .iter()
            .map(|m| narrow(&m.modulus))
            .collect::<Vec<_>>(),
        vec![12, 13]
    );
    let params = fitted(&req, 64, "packed-two-moduli-test-only");
    let compiled = c.st.compile(params.clone()).unwrap();
    let values = carries(&c);
    let (_, m) = compiled.map_witness(&c.witness).unwrap();
    assert_eq!(common::values(&m.entries()[0]), values[..64].to_vec());
    assert_eq!(common::values(&m.entries()[1])[0], values[64]);
    let bytes = compiled
        .prove_bytes_with_seed([1; 32], &c.witness, CTX, [2; 32])
        .unwrap();
    compiled.verify_bytes([1; 32], &bytes, CTX).unwrap();
    assert!(compiled.verify_bytes([1; 32], &bytes, b"other").is_err());
    // Clause 63, modulo 12 at the last slot of the first polynomial, with its constant one
    // larger: the witness map refuses the witness, and the verifier the proof.
    let mut st = Statement::new(Ring::new(P, 128).unwrap());
    st.var("s", 4, Norm::L2Squared(64)).unwrap();
    st.var("x", 1, Norm::Binary).unwrap();
    for (i, form) in c.forms.iter().enumerate() {
        let form = if i == 63 {
            let one = Poly::constant(form.r0.ring().clone(), 1);
            form.add(&st.constant_in(one).unwrap()).unwrap()
        } else {
            form.clone()
        };
        st.const_coeff_zero(form).unwrap();
    }
    let other = st.compile(params).unwrap();
    assert_eq!(other.map_witness(&c.witness).err(), Some(Error::Witness));
    assert!(other.verify_bytes([1; 32], &bytes, CTX).is_err());
    // The lifting check stays per clause; `the_lifting_check_is_exact_for_each_constraint` in
    // tests/multi_modulus.rs checks it at its exact boundary for two clauses in one packed
    // polynomial. Here the threshold is far below the modulus that the other checks need.
}

#[test]
fn packed_polynomials_sit_where_the_first_clause_is() {
    // A clause modulo 12, a quadratic modulo 13 and a clause modulo 13, at degree 64 (k = 2):
    // the packed polynomial comes first among the messages, in the first clause's place, then
    // the quadratic's two carry components; the range rows follow the same order.
    let ring = Ring::new(P, 128).unwrap();
    let mut rng = SplitMix64(0x2567_2104);
    let c = clauses(2, 0x2567_2105);
    let mut st = Statement::new(ring.clone());
    st.var("s", 4, Norm::L2Squared(64)).unwrap();
    st.var("x", 1, Norm::Binary).unwrap();
    let r13 = Ring::new(13, 128).unwrap();
    let a = poly(&r13, rng.uniform(128, 13));
    let product = st
        .variable_in(&r13, "s", 0)
        .unwrap()
        .product_affine(&st.variable_in(&r13, "x", 0).unwrap())
        .unwrap()
        .scale(&a)
        .unwrap();
    // Minus its value modulo 13, so that the witness satisfies it.
    let value = poly(&r13, integer_value(&product, &c.w));
    let quadratic = product.add(&st.constant_in(value.neg()).unwrap()).unwrap();
    // Clause 1 of `clauses` is modulo 12 and clause 0 modulo 13.
    st.const_coeff_zero(c.forms[1].clone()).unwrap();
    st.eq_mod_p(quadratic.clone()).unwrap();
    st.const_coeff_zero(c.forms[0].clone()).unwrap();
    let req = st.requirements(64, U256::from_u64(Q)).unwrap();
    assert_eq!((req.l, req.n_prime), (3, 3));
    let params = fitted(&req, 64, "packed-layout-test-only");
    let q = narrow(&params.prime_factors[0]) as i128;
    let compiled = st.compile(params).unwrap();
    let (s1, m) = compiled.map_witness(&c.witness).unwrap();
    let values = carries(&c);
    let proof_ring = Ring::new(q, 64).unwrap();
    let mut first = vec![values[1], values[0]];
    first.resize(64, 0);
    let f = integer_value(&quadratic, &c.w);
    assert!(f.iter().all(|x| x % 13 == 0));
    let lifted = Ring::new(q, 128).unwrap();
    let components = iso::split(
        &poly(&lifted, f.iter().map(|x| x / 13).collect()),
        proof_ring.clone(),
    )
    .unwrap();
    let expected: Vec<Poly> = std::iter::once(poly(&proof_ring, first))
        .chain(components)
        .collect();
    assert_eq!(m.entries(), expected.as_slice());
    let arp = compiled
        .statement()
        .arp
        .as_ref()
        .unwrap()
        .evaluate(&s1, &m)
        .unwrap();
    assert_eq!(arp.entries(), expected.as_slice());
    let proof = compiled
        .prove_with_seed([3; 32], &c.witness, CTX, [4; 32])
        .unwrap();
    compiled.verify([3; 32], &proof, CTX).unwrap();
}
