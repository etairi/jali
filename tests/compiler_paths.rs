//! Statement-compiler paths that the other tests leave out: `lin::compile` end to end,
//! `const_coeff_zero` at the proof modulus and lifted, `Norm::Binary` and `Norm::Unbounded`,
//! the refusals of the statement builder, the compiler and the witness map, and the size
//! estimate at a small $`\gamma`$. Parameters come from the test port of the parameter tool
//! (`common::params`); each test fixes its seeds.
use jali::{
    Error, lin,
    math::{Poly, PolyMat, PolyVec, Ring, U256, iso},
    params::TboxParams,
    quad::QuadEq,
    statement::{self, Extraction, Norm, Requirements, Statement},
};
use std::{collections::BTreeMap, sync::Arc};

mod common;
use common::{
    SplitMix64, narrow,
    params::{fit, fit_upward, lifting_threshold, range_width},
    ring::{centred, negacyclic, norm2, poly},
    values as centred_values,
};

/// The prime of `toy_d64`, where every search for a proof modulus starts.
const Q: u64 = 1099511627917;

type Witness = BTreeMap<String, Vec<Poly>>;

/// Parameters for a statement lifted from `statement_modulus`: the first prime that clears the
/// compiler's lifting threshold (recomputed independently) and fits.
fn lifted_params(req: &Requirements, statement_modulus: i128, id: &str) -> TboxParams {
    let threshold = lifting_threshold(req, statement_modulus as u128, range_width(req, 64));
    fit_upward(req, id, 64, (threshold as u64 + 1).max(Q)).1
}

/// The statement `lin::compile` builds for $`Aw+t=0`$, so that its requirements can be read.
fn linear_statement(a: &PolyMat, t: &PolyVec, blocks: &[lin::Block]) -> Statement {
    let mut st = Statement::new(a.ring().clone());
    for block in blocks {
        st.var(&block.name, block.length, block.norm).unwrap();
    }
    for row in 0..a.rows() {
        let mut e = st.constant(t.entries()[row].clone()).unwrap();
        let mut col = 0;
        for block in blocks {
            for j in 0..block.length {
                let term = st.variable(&block.name, j).unwrap();
                e = e
                    .add(&term.scale(a.get(row, col).unwrap()).unwrap())
                    .unwrap();
                col += 1;
            }
        }
        st.eq_mod_p(e).unwrap();
    }
    st
}

#[test]
fn lin_compile_proves_and_verifies_with_implicit_quotients() {
    const CTX: &[u8] = b"jali-test/lin-compile";
    let ring = Ring::new(13, 128).unwrap();
    let mut rng = SplitMix64(1);
    let s: Vec<Poly> = (0..4).map(|_| poly(&ring, rng.ternary(128))).collect();
    let x = vec![poly(&ring, rng.binary(128))];
    assert!(norm2(&s) <= 512);
    let w: Vec<Poly> = s.iter().chain(&x).cloned().collect();
    let a = PolyMat::new(
        ring.clone(),
        2,
        5,
        (0..10).map(|_| poly(&ring, rng.uniform(128, 13))).collect(),
    )
    .unwrap();
    let aw = a
        .mul(&PolyVec::new(ring.clone(), w.clone()).unwrap())
        .unwrap();
    let t = PolyVec::new(ring.clone(), aw.entries().iter().map(Poly::neg).collect()).unwrap();
    let blocks = [
        lin::Block {
            name: "s".into(),
            length: 4,
            norm: Norm::L2Squared(512),
        },
        lin::Block {
            name: "x".into(),
            length: 1,
            norm: Norm::Binary,
        },
    ];
    let req = linear_statement(&a, &t, &blocks)
        .requirements(64, U256::from_u64(Q))
        .unwrap();
    // Two rows of degree 128: four implicit quotients and no committed carry.
    assert_eq!((req.m1, req.l, req.n_bin, req.n_prime), (10, 0, 2, 4));
    let params = lifted_params(&req, 13, "lin-compile-test-only");
    assert_eq!(u128::from(params.linf_bound), req.linf_bound);
    let compiled = lin::compile(&a, &t, &blocks, params.clone()).unwrap();
    assert_eq!(
        compiled.statement(),
        linear_statement(&a, &t, &blocks)
            .compile(params.clone())
            .unwrap()
            .statement()
    );
    assert!(compiled.statement().quadratic.is_empty());
    assert!(compiled.statement().evaluation.is_empty());
    assert!(compiled.statement().binary.is_some());
    let witness = Witness::from([("s".into(), s.clone()), ("x".into(), x.clone())]);
    let (s1, m) = compiled.map_witness(&witness).unwrap();
    assert!(
        m.is_empty(),
        "linear rows lift implicitly, without committed carries"
    );
    // The ARP rows are the integer quotients (Aw + t) / 13, computed here independently.
    let quotients = compiled
        .statement()
        .arp
        .as_ref()
        .unwrap()
        .evaluate(&s1, &m)
        .unwrap();
    let proof_ring = Ring::new(narrow(&params.prime_factors[0]) as i128, 64).unwrap();
    let integer_ring = Ring::new(narrow(&params.prime_factors[0]) as i128, 128).unwrap();
    let mut expected = Vec::new();
    for row in 0..2 {
        let mut f = centred_values(&t.entries()[row]);
        for (col, w) in w.iter().enumerate() {
            let product = negacyclic(
                &centred_values(a.get(row, col).unwrap()),
                &centred_values(w),
            );
            f.iter_mut().zip(product).for_each(|(f, p)| *f += p);
        }
        assert!(f.iter().all(|x| x % 13 == 0));
        let quotient = poly(&integer_ring, f.iter().map(|x| x / 13).collect());
        expected.extend(iso::split(&quotient, proof_ring.clone()).unwrap());
    }
    assert_eq!(quotients.entries(), expected.as_slice());
    let bytes = compiled
        .prove_bytes_with_seed([7; 32], &witness, CTX, [8; 32])
        .unwrap();
    compiled.verify_bytes([7; 32], &bytes, CTX).unwrap();
    assert!(compiled.verify_bytes([6; 32], &bytes, CTX).is_err());
    assert!(compiled.verify_bytes([7; 32], &bytes, b"other").is_err());
    // One changed witness coefficient breaks the relation.
    let mut bad = witness.clone();
    let c = bad["s"][0].coefficient_i128(3).unwrap();
    bad.get_mut("s").unwrap()[0]
        .set_coefficient(3, if c == 1 { 0 } else { c + 1 })
        .unwrap();
    assert_eq!(compiled.map_witness(&bad).err(), Some(Error::Witness));
    // A false statement: t + X^5 in row 1.
    let mut false_t = t.entries().to_vec();
    false_t[1] = false_t[1]
        .add(&Poly::constant(ring.clone(), 1).rotate(5))
        .unwrap();
    let false_t = PolyVec::new(ring.clone(), false_t).unwrap();
    let other = lin::compile(&a, &false_t, &blocks, params.clone()).unwrap();
    assert!(other.verify_bytes([7; 32], &bytes, CTX).is_err());
    assert_eq!(other.map_witness(&witness).err(), Some(Error::Witness));
    // The quotient bound is exact: one less is refused.
    let mut tight = req.clone();
    tight.linf_bound -= 1;
    let tight = fit(
        &tight,
        "lin-compile-tight-test-only",
        params
            .prime_factors
            .iter()
            .map(|p| narrow(p) as u64)
            .collect(),
        64,
        params.mlwe_rank,
    )
    .unwrap();
    assert_eq!(
        lin::compile(&a, &t, &blocks, tight).err(),
        Some(Error::Parameter("carry or quotient bound"))
    );
}

/// Six clauses $`\mathrm{ct}(\sigma(a_j)s_{j\bmod 5})=v_j`$, the inner products
/// $`\langle a_j,s_{j\bmod5}\rangle=v_j`$, with a constant polynomial $`-v_j+X^3\cdot`$`extra`.
fn inner_products(ring: &Arc<Ring>, a: &[Poly], values: &[i128], extra: i128) -> Statement {
    inner_products_at(ring, a, values, 3, extra)
}
/// As `inner_products`, with the constant polynomial $`-v_j+X^i\cdot`$`extra`.
fn inner_products_at(
    ring: &Arc<Ring>,
    a: &[Poly],
    values: &[i128],
    i: usize,
    extra: i128,
) -> Statement {
    let mut st = Statement::new(ring.clone());
    st.var("s", 5, Norm::L2Squared(1024)).unwrap();
    for (j, (a, v)) in a.iter().zip(values).enumerate() {
        let mut constant = Poly::constant(ring.clone(), -v);
        constant.set_coefficient(i, extra).unwrap();
        let e = st
            .variable("s", j % 5)
            .unwrap()
            .scale(&a.auto())
            .unwrap()
            .add(&st.constant(constant).unwrap())
            .unwrap();
        st.const_coeff_zero(e).unwrap();
    }
    st
}
/// A ternary witness `s` of five degree-128 polynomials and six public `a_j` with small
/// coefficients, over a ring of modulus `modulus`.
fn inner_product_instance(modulus: i128) -> (Arc<Ring>, Vec<Poly>, Vec<Poly>) {
    let ring = Ring::new(modulus, 128).unwrap();
    let mut rng = SplitMix64(2);
    let s: Vec<Poly> = (0..5).map(|_| poly(&ring, rng.ternary(128))).collect();
    assert!(norm2(&s) <= 1024);
    let a: Vec<Poly> = (0..6)
        .map(|_| poly(&ring, rng.uniform(128, 13).iter().map(|x| x - 6).collect()))
        .collect();
    (ring, s, a)
}
fn inner(a: &Poly, b: &Poly) -> i128 {
    centred_values(a)
        .iter()
        .zip(centred_values(b))
        .map(|(x, y)| x * y)
        .sum()
}

#[test]
fn const_coeff_zero_at_the_proof_modulus_proves_inner_products() {
    const CTX: &[u8] = b"jali-test/const-coeff-native";
    // The shape does not depend on the modulus when nothing is lifted.
    let (probe, _, a) = inner_product_instance(13);
    let req = inner_products(&probe, &a, &[0; 6], 0)
        .requirements(64, U256::from_u8(13))
        .unwrap();
    assert_eq!((req.m1, req.l, req.n_prime, req.linf_bound), (10, 0, 0, 0));
    let (q, params) = fit_upward(&req, "const-coeff-native-test-only", 64, Q);
    let (ring, s, a) = inner_product_instance(q as i128);
    let values: Vec<i128> = (0..6).map(|j| inner(&a[j], &s[j % 5])).collect();
    for (j, a) in a.iter().enumerate() {
        assert_eq!(
            a.auto().mul(&s[j % 5]).unwrap().coefficient_i128(0),
            Ok(values[j])
        );
    }
    let compiled = inner_products(&ring, &a, &values, 0)
        .compile(params.clone())
        .unwrap();
    let st = compiled.statement();
    assert_eq!(st.evaluation.len(), 6);
    assert!(st.quadratic.is_empty() && st.arp.is_none() && st.binary.is_none());
    let witness = Witness::from([("s".into(), s.clone())]);
    let (_, m) = compiled.map_witness(&witness).unwrap();
    assert!(m.is_empty(), "no carries at the proof modulus");
    let proof = compiled
        .prove_with_seed([1; 32], &witness, CTX, [2; 32])
        .unwrap();
    compiled.verify([1; 32], &proof, CTX).unwrap();
    assert!(compiled.verify([1; 32], &proof, b"other").is_err());
    // A changed coefficient of s_1 at a position where a_1 is nonzero changes clause 1.
    let i = (0..128)
        .find(|i| a[1].coefficient_i128(*i) != Ok(0))
        .unwrap();
    let mut bad = witness.clone();
    let c = bad["s"][1].coefficient_i128(i).unwrap();
    bad.get_mut("s").unwrap()[1]
        .set_coefficient(i, if c == 1 { 0 } else { c + 1 })
        .unwrap();
    assert!(norm2(&bad["s"]) <= 1024);
    assert_eq!(compiled.map_witness(&bad).err(), Some(Error::Witness));
    // Only the constant coefficient is constrained, so a term X^i in every constant polynomial
    // leaves the relation true. Lowering puts X^i into component i mod 2, and only component 0
    // of a constant-coefficient clause is compiled: X^3 gives the same compiled statement, under
    // which the proof verifies, and X^2 another one, under which it does not.
    let odd = inner_products(&ring, &a, &values, 5)
        .compile(params.clone())
        .unwrap();
    odd.map_witness(&witness).unwrap();
    assert_eq!(odd.statement(), compiled.statement());
    odd.verify([1; 32], &proof, CTX).unwrap();
    let even = inner_products_at(&ring, &a, &values, 2, 5)
        .compile(params.clone())
        .unwrap();
    even.map_witness(&witness).unwrap();
    assert_ne!(even.statement(), compiled.statement());
    assert!(even.verify([1; 32], &proof, CTX).is_err());
    let mut wrong = values.clone();
    wrong[5] += 1;
    let wrong = inner_products(&ring, &a, &wrong, 0)
        .compile(params)
        .unwrap();
    assert_eq!(wrong.map_witness(&witness).err(), Some(Error::Witness));
    assert!(wrong.verify([1; 32], &proof, CTX).is_err());
}

#[test]
fn const_coeff_zero_lifted_packs_the_carries_of_all_clauses() {
    const CTX: &[u8] = b"jali-test/const-coeff-lifted";
    let (ring, s, a) = inner_product_instance(13);
    let values: Vec<i128> = (0..6)
        .map(|j| inner(&a[j], &s[j % 5]).rem_euclid(13))
        .collect();
    let st = inner_products(&ring, &a, &values, 0);
    let req = st.requirements(64, U256::from_u64(Q)).unwrap();
    // The six carries share one packed polynomial with one range row.
    assert_eq!((req.m1, req.l, req.n_prime), (10, 1, 1));
    let params = lifted_params(&req, 13, "const-coeff-lifted-test-only");
    let compiled = st.compile(params.clone()).unwrap();
    assert_eq!(compiled.statement().evaluation.len(), 6);
    assert!(compiled.statement().arp.is_some());
    let witness = Witness::from([("s".into(), s.clone())]);
    let (s1, m) = compiled.map_witness(&witness).unwrap();
    // Carry j is the exact integer quotient (<a_j, s> - v_j) / 13, with the constant -v_j
    // centred modulo 13 as the statement ring stores it, at coefficient j of the packed
    // polynomial; the other coefficients are 0.
    let carries: Vec<i128> = (0..6)
        .map(|j| {
            let f = inner(&a[j], &s[j % 5]) + centred(-values[j], 13);
            assert_eq!(f % 13, 0);
            f / 13
        })
        .collect();
    assert!(carries.iter().any(|c| *c < 0) && carries.iter().any(|c| *c > 0));
    let proof_ring = Ring::new(narrow(&params.prime_factors[0]) as i128, 64).unwrap();
    let mut packed = carries.clone();
    packed.resize(64, 0);
    let expected = vec![poly(&proof_ring, packed)];
    assert_eq!(m.entries(), expected.as_slice());
    // The range row of the carries is the packed polynomial itself.
    let arp = compiled
        .statement()
        .arp
        .as_ref()
        .unwrap()
        .evaluate(&s1, &m)
        .unwrap();
    assert_eq!(arp.entries(), expected.as_slice());
    let proof = compiled
        .prove_with_seed([1; 32], &witness, CTX, [2; 32])
        .unwrap();
    compiled.verify([1; 32], &proof, CTX).unwrap();
    assert!(compiled.verify([1; 32], &proof, b"other").is_err());
    let i = (0..128)
        .find(|i| a[1].coefficient_i128(*i) != Ok(0))
        .unwrap();
    let mut bad = witness.clone();
    let c = bad["s"][1].coefficient_i128(i).unwrap();
    bad.get_mut("s").unwrap()[1]
        .set_coefficient(i, if c == 1 { 0 } else { c + 1 })
        .unwrap();
    assert_eq!(compiled.map_witness(&bad).err(), Some(Error::Witness));
    let mut wrong = values.clone();
    wrong[5] = (wrong[5] + 1) % 13;
    let wrong = inner_products(&ring, &a, &wrong, 0)
        .compile(params)
        .unwrap();
    assert_eq!(wrong.map_witness(&witness).err(), Some(Error::Witness));
    assert!(wrong.verify([1; 32], &proof, CTX).is_err());
}

/// `b_0 b_1 + X^3 s_0 - u = 0` with a binary block `b`, an unbounded `u` and an L2 block `s`.
fn binary_unbounded(ring: &Arc<Ring>) -> Statement {
    let mut st = Statement::new(ring.clone());
    st.var("b", 3, Norm::Binary).unwrap();
    st.var("u", 1, Norm::Unbounded).unwrap();
    st.var("s", 3, Norm::L2Squared(1024)).unwrap();
    let x3 = Poly::constant(ring.clone(), 1).rotate(3);
    let e = st
        .variable("b", 0)
        .unwrap()
        .product_affine(&st.variable("b", 1).unwrap())
        .unwrap()
        .add(&st.variable("s", 0).unwrap().scale(&x3).unwrap())
        .unwrap()
        .add(
            &st.variable("u", 0)
                .unwrap()
                .scale(&Poly::constant(ring.clone(), -1))
                .unwrap(),
        )
        .unwrap();
    st.eq_mod_p(e).unwrap();
    st
}

#[test]
fn binary_and_unbounded_blocks_at_the_proof_modulus() {
    const CTX: &[u8] = b"jali-test/binary-unbounded";
    let req = binary_unbounded(&Ring::new(13, 128).unwrap())
        .requirements(64, U256::from_u8(13))
        .unwrap();
    assert_eq!((req.m1, req.l, req.n_bin, req.n_prime), (12, 2, 6, 0));
    assert_eq!(req.l2_rows, vec![6]);
    let (q, params) = fit_upward(&req, "binary-unbounded-test-only", 64, Q);
    let ring = Ring::new(q as i128, 128).unwrap();
    let compiled = binary_unbounded(&ring).compile(params).unwrap();
    assert!(compiled.statement().binary.is_some());
    // What a proof guarantees for each block: the declared bounds, and nothing for u.
    for (name, bound) in [
        ("b", Ok(Extraction::Binary)),
        ("u", Ok(Extraction::Unbounded)),
        ("s", Ok(Extraction::L2Squared(1024))),
        ("v", Err(Error::Index)),
    ] {
        assert_eq!(compiled.extraction_bound(name), bound, "{name}");
    }
    let mut rng = SplitMix64(3);
    let b: Vec<Poly> = (0..3).map(|_| poly(&ring, rng.binary(128))).collect();
    let s: Vec<Poly> = (0..3).map(|_| poly(&ring, rng.ternary(128))).collect();
    let u = b[0].mul(&b[1]).unwrap().add(&s[0].rotate(3)).unwrap();
    let witness = Witness::from([
        ("b".into(), b.clone()),
        ("u".into(), vec![u.clone()]),
        ("s".into(), s.clone()),
    ]);
    let (s1, m) = compiled.map_witness(&witness).unwrap();
    // The unbounded block becomes the BDLOP messages; the bounded blocks the Ajtai part.
    let proof_ring = Ring::new(q as i128, 64).unwrap();
    assert_eq!(s1.len(), 12);
    assert_eq!(m.entries(), iso::split(&u, proof_ring).unwrap().as_slice());
    let proof = compiled
        .prove_with_seed([4; 32], &witness, CTX, [5; 32])
        .unwrap();
    compiled.verify([4; 32], &proof, CTX).unwrap();
    assert!(compiled.verify([4; 32], &proof, b"other").is_err());
    // A coefficient 2 in b_2, which no relation uses, is refused by the binary check.
    let mut bad = witness.clone();
    bad.get_mut("b").unwrap()[2].set_coefficient(0, 2).unwrap();
    assert_eq!(compiled.map_witness(&bad).err(), Some(Error::Witness));
    // A flipped bit of b_0 keeps b binary but breaks the relation.
    let mut bad = witness.clone();
    let bit = bad["b"][0].coefficient_i128(7).unwrap();
    bad.get_mut("b").unwrap()[0]
        .set_coefficient(7, 1 - bit)
        .unwrap();
    assert!(!b[1].is_zero());
    assert_eq!(compiled.map_witness(&bad).err(), Some(Error::Witness));
}

/// The possession relation `(s_0 - 1)(5 - x_0) = 1` modulo 13, with `x` binary.
fn binary_possession(ring: &Arc<Ring>) -> Statement {
    let mut st = Statement::new(ring.clone());
    st.var("s", 8, Norm::L2Squared(128)).unwrap();
    st.var("x", 2, Norm::Binary).unwrap();
    let minus = |e: QuadEq| e.scale(&Poly::constant(ring.clone(), -1)).unwrap();
    let one = st.constant(Poly::constant(ring.clone(), 1)).unwrap();
    let five = st.constant(Poly::constant(ring.clone(), 5)).unwrap();
    let left = st
        .variable("s", 0)
        .unwrap()
        .add(&minus(one.clone()))
        .unwrap();
    let right = five.add(&minus(st.variable("x", 0).unwrap())).unwrap();
    let e = left
        .product_affine(&right)
        .unwrap()
        .add(&minus(one))
        .unwrap();
    st.eq_mod_p(e).unwrap();
    st
}

#[test]
fn binary_block_lifted_with_committed_carries() {
    const CTX: &[u8] = b"jali-test/binary-lifted";
    let ring = Ring::new(13, 128).unwrap();
    let st = binary_possession(&ring);
    let req = st.requirements(64, U256::from_u64(Q)).unwrap();
    assert_eq!((req.n_bin, req.n_prime, req.l), (4, 2, 2));
    let compiled = st
        .compile(lifted_params(&req, 13, "binary-lifted-test-only"))
        .unwrap();
    // x_0 = 1 and s_0 = -2 (11 modulo 13): (-2 - 1)(5 - 1) - 1 = -13, so the carry is -1.
    let mut s = vec![Poly::zero(ring.clone()); 8];
    s[0] = Poly::constant(ring.clone(), -2);
    let mut x = vec![Poly::zero(ring.clone()); 2];
    x[0] = Poly::constant(ring.clone(), 1);
    let witness = Witness::from([("s".into(), s), ("x".into(), x)]);
    let (_, m) = compiled.map_witness(&witness).unwrap();
    assert_eq!(m.entries()[0].coefficient_i128(0), Ok(-1));
    assert!(
        centred_values(&m.entries()[0])[1..].iter().all(|c| *c == 0) && m.entries()[1].is_zero()
    );
    let proof = compiled
        .prove_with_seed([9; 32], &witness, CTX, [10; 32])
        .unwrap();
    compiled.verify([9; 32], &proof, CTX).unwrap();
    assert!(compiled.verify([9; 32], &proof, b"other").is_err());
    // A bit set in x_1, outside the relation, keeps the witness valid; one in x_0 does not.
    let mut outside = witness.clone();
    outside.get_mut("x").unwrap()[1]
        .set_coefficient(5, 1)
        .unwrap();
    compiled.map_witness(&outside).unwrap();
    let mut inside = outside;
    inside.get_mut("x").unwrap()[0]
        .set_coefficient(1, 1)
        .unwrap();
    assert_eq!(compiled.map_witness(&inside).err(), Some(Error::Witness));
}

#[test]
fn unbounded_block_in_a_lifted_relation_is_refused() {
    let ring = Ring::new(13, 128).unwrap();
    let mut st = Statement::new(ring.clone());
    st.var("s", 10, Norm::L2Squared(64)).unwrap();
    st.var("u", 1, Norm::Unbounded).unwrap();
    let e = st
        .variable("s", 0)
        .unwrap()
        .add(&st.variable("u", 0).unwrap())
        .unwrap();
    st.eq_mod_p(e).unwrap();
    let refusal = Some(Error::Parameter("unbounded variable in lifted relation"));
    assert_eq!(st.requirements(64, U256::from_u64(Q)).err(), refusal);
    // Any checked set of another modulus reaches the same refusal inside `compile`.
    assert_eq!(st.compile(jali::params::toy_d64()).err(), refusal);
    assert!(st.requirements(64, U256::from_u8(13)).is_ok());
}

/// A statement over `ring` with `u` unbounded and `x` of norm `norm`, declared in the given
/// order, and the constraint `x_0 u_0 = 0` over the statement ring (a product, so lifting
/// commits a carry). With `norm` unbounded, `x` is a second unbounded block.
fn unbounded_product(ring: &Arc<Ring>, norm: Norm, u_first: bool) -> Statement {
    let mut st = Statement::new(ring.clone());
    let declare = |st: &mut Statement, name: &str| match name {
        "u" => st.var("u", 1, Norm::Unbounded).unwrap(),
        _ => st.var("x", 2, norm).unwrap(),
    };
    for name in if u_first { ["u", "x"] } else { ["x", "u"] } {
        declare(&mut st, name);
    }
    let form = st
        .variable("x", 0)
        .unwrap()
        .product_affine(&st.variable("u", 0).unwrap())
        .unwrap();
    st.eq_mod_p(form).unwrap();
    st
}

#[test]
fn an_unbounded_block_in_a_lifted_product_is_refused() {
    let refusal = Some(Error::Parameter("unbounded variable in lifted relation"));
    let ring = Ring::new(13, 128).unwrap();
    // The final Rust verifier's case: s u with s exact-norm, over Z_13, lifted to Q. Only the
    // quadratic arm of `form_bound` refuses it: the linear coefficients are explicit zeros.
    let mut st = Statement::new(ring.clone());
    st.var("s", 1, Norm::L2Squared(64)).unwrap();
    st.var("u", 1, Norm::Unbounded).unwrap();
    let form = st
        .variable("s", 0)
        .unwrap()
        .product_affine(&st.variable("u", 0).unwrap())
        .unwrap();
    assert!(form.r1.entries().all(|(_, p)| p.is_zero()));
    st.eq_mod_p(form).unwrap();
    assert_eq!(st.requirements(64, U256::from_u64(Q)).err(), refusal);
    assert_eq!(st.compile(jali::params::toy_d64()).err(), refusal);
    // Every norm of the other factor, in both orders of the pair, and u times u; at the
    // statement modulus the same products compile natively.
    for norm in [
        Norm::L2Squared(64),
        Norm::Binary,
        Norm::Linf(1),
        Norm::Unbounded,
    ] {
        for u_first in [false, true] {
            let st = unbounded_product(&ring, norm, u_first);
            assert_eq!(
                st.requirements(64, U256::from_u64(Q)).err(),
                refusal,
                "{norm:?}, u first: {u_first}"
            );
            assert!(st.requirements(64, U256::from_u8(13)).is_ok());
        }
    }
    // A false statement that the refusal keeps out (the verifier forged a proof of it with the
    // refusal switched off). Over Z_4 with ||s||^2 <= 4: A, s_0 = 2 modulo 4, forces s_0 = 2,
    // so ct(s_0 u) = 2 u_0 is even, and B, ct(s_0 u) = 1 modulo 2, has no witness. Both moduli
    // divide 4, so the range condition does not read B.
    let (ring4, ring2) = (Ring::new(4, 128).unwrap(), Ring::new(2, 128).unwrap());
    let mut st = Statement::new(ring4.clone());
    st.var("s", 5, Norm::L2Squared(4)).unwrap();
    st.var("u", 1, Norm::Unbounded).unwrap();
    let a = st
        .variable("s", 0)
        .unwrap()
        .add(&st.constant(Poly::constant(ring4, -2)).unwrap())
        .unwrap();
    st.eq_mod_p(a).unwrap();
    let b = st
        .variable_in(&ring2, "s", 0)
        .unwrap()
        .product_affine(&st.variable_in(&ring2, "u", 0).unwrap())
        .unwrap()
        .add(&st.constant_in(Poly::constant(ring2, -1)).unwrap())
        .unwrap();
    st.const_coeff_zero(b).unwrap();
    assert_eq!(st.requirements(64, U256::from_u64(Q)).err(), refusal);
    assert_eq!(st.compile(jali::params::toy_d64()).err(), refusal);
}

#[test]
fn explicit_zero_coefficients_are_not_terms() {
    // `s_0 + x_0 u_0 - x_0 u_0` and `s_0 + y_0 y_0 - y_0 y_0` keep an explicit zero quadratic
    // coefficient. Lifted, each is the linear constraint s_0 = 0: neither u nor y is read, so
    // there is no refusal, no carry and no lifting entry for y.
    let ring = Ring::new(13, 128).unwrap();
    // s_0 plus the form `extra` builds, over Z_13.
    let with = |extra: &dyn Fn(&Statement) -> QuadEq| {
        let mut st = Statement::new(ring.clone());
        st.var("s", 10, Norm::L2Squared(64)).unwrap();
        st.var("x", 1, Norm::L2Squared(64)).unwrap();
        st.var("y", 1, Norm::Linf(2)).unwrap();
        st.var("u", 1, Norm::Unbounded).unwrap();
        let form = st.variable("s", 0).unwrap().add(&extra(&st)).unwrap();
        st.eq_mod_p(form).unwrap();
        st
    };
    let cancelled = |st: &Statement, a: &str, b: &str| {
        let product = st
            .variable(a, 0)
            .unwrap()
            .product_affine(&st.variable(b, 0).unwrap())
            .unwrap();
        let minus = product.scale(&Poly::constant(ring.clone(), -1)).unwrap();
        let sum = product.add(&minus).unwrap();
        assert_eq!(sum.r2.entries().count(), 1);
        assert!(sum.r2.entries().all(|(_, p)| p.is_zero()));
        sum
    };
    let linear = with(&|st| st.constant(Poly::zero(ring.clone())).unwrap());
    let want = linear.requirements(64, U256::from_u64(Q)).unwrap();
    // u's two proof-ring rows and no carry; y is declared but read by no lifted constraint.
    assert_eq!((want.l, want.n_prime), (2, 2 + 2));
    assert_eq!(want.linf.as_ref().unwrap().lifting, vec![]);
    for (a, b) in [("x", "u"), ("y", "y")] {
        let st = with(&|st| cancelled(st, a, b));
        assert_eq!(
            st.requirements(64, U256::from_u64(Q)).unwrap(),
            want,
            "{a} {b}"
        );
    }
}

/// The possession relation of `tests/statement.rs` over Z_13[X]/(X^128+1): `s` (8, L2 128)
/// and `x` (2, L2 64), `(s_0 - 1)(5 - x_0) = 1`, with parameters regenerated by the port.
fn l2_possession() -> (Arc<Ring>, Statement, statement::Compiled) {
    let ring = Ring::new(13, 128).unwrap();
    let mut st = Statement::new(ring.clone());
    st.var("s", 8, Norm::L2Squared(128)).unwrap();
    st.var("x", 2, Norm::L2Squared(64)).unwrap();
    let minus = |e: QuadEq| e.scale(&Poly::constant(ring.clone(), -1)).unwrap();
    let one = st.constant(Poly::constant(ring.clone(), 1)).unwrap();
    let five = st.constant(Poly::constant(ring.clone(), 5)).unwrap();
    let left = st
        .variable("s", 0)
        .unwrap()
        .add(&minus(one.clone()))
        .unwrap();
    let right = five.add(&minus(st.variable("x", 0).unwrap())).unwrap();
    let e = left
        .product_affine(&right)
        .unwrap()
        .add(&minus(one))
        .unwrap();
    st.eq_mod_p(e).unwrap();
    let req = st.requirements(64, U256::from_u64(Q)).unwrap();
    let params = fit(&req, "possession-test-only", vec![Q], 64, 26).unwrap();
    let compiled = st.compile(params).unwrap();
    (ring, st, compiled)
}

#[test]
fn the_size_estimate_covers_the_hints_at_a_small_gamma() {
    // A hint is about round(z_22 / gamma), so at gamma = 676, an even divisor of
    // Q - 1 = 2^2 * 3 * 13^2 * 10867 * 49891, and sigma_2 / gamma = 9.4 it takes about 14 bits,
    // not the 2.25 of LNP22 Section 6.1; D = 1 keeps 2^(D-1) omega d = 512 below gamma.
    const CTX: &[u8] = b"jali-test/small-gamma";
    let (ring, st, compiled) = l2_possession();
    let mut params = compiled.parameters().clone();
    (params.gamma, params.d_bits) = (676, 1);
    let checked = params.check().unwrap();
    let compiled = st.compile(params.clone()).unwrap();
    let mut s = vec![Poly::zero(ring.clone()); 8];
    s[0] = Poly::constant(ring.clone(), 5);
    let mut x = vec![Poly::zero(ring.clone()); 2];
    x[0] = Poly::constant(ring.clone(), -5);
    let witness = Witness::from([("s".into(), s), ("x".into(), x)]);
    let bytes = compiled
        .prove_bytes_with_seed([23; 32], &witness, CTX, [24; 32])
        .unwrap();
    compiled.verify_bytes([23; 32], &bytes, CTX).unwrap();
    // The estimate covers the proof; with 2.25 bits per hint coefficient it would not.
    let s = 1.55 * 2f64.powi(params.log_sigma[1] as i32) / params.gamma as f64;
    let hints = (params.n_msis * params.degree) as f64;
    let flat = checked.estimated_proof_bytes - ((1.6 * s + 0.7 - 2.25) * hints / 8.0) as usize;
    assert!(
        flat < bytes.len() && bytes.len() <= checked.estimated_proof_bytes,
        "{flat} < {} <= {}",
        bytes.len(),
        checked.estimated_proof_bytes
    );
}

#[test]
fn compiler_refuses_malformed_statements_and_witnesses() {
    let declaration = Some(Error::Parameter("variable declaration"));
    let target = Some(Error::Parameter("target ring"));
    let (ring, st, compiled) = l2_possession();
    // Proof rings: degree 32 or 256, a degree that does not divide the source's, modulus 1.
    for (degree, modulus) in [(32, Q), (256, Q), (64, 1)] {
        assert_eq!(
            st.requirements(degree, U256::from_u64(modulus)).err(),
            target
        );
    }
    let small = Statement::new(Ring::new(13, 64).unwrap());
    assert_eq!(small.requirements(128, U256::from_u64(Q)).err(), target);
    // `lower` refuses a target degree that does not divide the source degree.
    let form = QuadEq::zero(Ring::new(13, 64).unwrap(), 1).unwrap();
    assert_eq!(
        statement::lower(&form, Ring::new(13, 128).unwrap()).err(),
        Some(Error::Dimension)
    );
    // Variable declarations.
    let mut fresh = Statement::new(ring.clone());
    assert_eq!(fresh.var("", 1, Norm::Binary).err(), declaration);
    assert_eq!(
        fresh.var("n".repeat(257), 1, Norm::Binary).err(),
        declaration
    );
    fresh.var("n".repeat(256), 1, Norm::Binary).unwrap();
    assert_eq!(fresh.var("zero", 0, Norm::Binary).err(), declaration);
    assert_eq!(
        fresh.var("n".repeat(256), 1, Norm::Unbounded).err(),
        declaration
    );
    assert_eq!(
        fresh.var("large", 32768, Norm::Unbounded).err(),
        Some(Error::Dimension)
    );
    fresh.var("fits", 32767, Norm::Unbounded).unwrap();
    let mut late = st.clone();
    assert_eq!(late.var("late", 1, Norm::Binary).err(), declaration);
    // Expressions.
    assert_eq!(st.variable("y", 0).err(), Some(Error::Index));
    assert_eq!(st.variable("x", 2).err(), Some(Error::Index));
    st.variable("x", 1).unwrap();
    let other_ring = Poly::constant(Ring::new(17, 128).unwrap(), 1);
    assert_eq!(st.constant(other_ring).err(), Some(Error::RingMismatch));
    let wrong_space = QuadEq::zero(ring.clone(), 3).unwrap();
    assert_eq!(late.eq_mod_p(wrong_space).err(), Some(Error::Dimension));
    // A product of two wide variables with coefficients 2^97 over a 99-bit ring: the bound
    // 64 * 2^97 * (2^64 - 1), about 2^167, exceeds 128 bits and fits. Over a 251-bit ring
    // with coefficients 2^249 it is about 2^319 and does not.
    let wide_product = |ring: &Arc<Ring>, coefficient: &Poly| {
        let mut wide = Statement::new(ring.clone());
        wide.var("w", 2, Norm::L2Squared(u64::MAX)).unwrap();
        let e = wide
            .variable("w", 0)
            .unwrap()
            .product_affine(&wide.variable("w", 1).unwrap())
            .unwrap()
            .scale(coefficient)
            .unwrap();
        wide.eq_mod_p(e).unwrap();
        wide.requirements(64, U256::from_u64(Q))
    };
    let ring99 = Ring::new((1i128 << 99) - 1, 64).unwrap();
    let req = wide_product(
        &ring99,
        &Poly::new(ring99.clone(), vec![1i128 << 97; 64]).unwrap(),
    )
    .unwrap();
    assert_eq!(
        req.max_integer_coefficient,
        U256::from_u64(u64::MAX).shl_vartime(103)
    );
    let ring251 =
        Ring::with_modulus(U256::ONE.shl_vartime(250).wrapping_add(&U256::ONE), 64).unwrap();
    let big = Poly::constant_u256(ring251.clone(), &U256::ONE.shl_vartime(249));
    let dense = (1..64).fold(big.clone(), |acc, i| acc.add(&big.rotate(i)).unwrap());
    assert_eq!(wide_product(&ring251, &dense).err(), Some(Error::Overflow));
    // Parameters that do not match the statement.
    let params = compiled.parameters().clone();
    let mut p = params.clone();
    p.alpha_squared -= 1;
    assert_eq!(
        st.compile(p).err(),
        Some(Error::Parameter("bounded witness norm budget"))
    );
    let dimensions = Some(Error::Parameter("compiled witness dimensions"));
    let mut p = params.clone();
    p.m1 -= 1;
    p.check().unwrap();
    assert_eq!(st.compile(p).err(), dimensions);
    for delta in [1i64, -1] {
        let mut p = params.clone();
        p.l = (p.l as i64 + delta) as usize;
        p.mlwe_rank = (p.mlwe_rank as i64 - delta) as usize;
        p.check().unwrap();
        assert_eq!(st.compile(p).err(), dimensions);
    }
    // A statement modulus that the proof modulus divides has no inverse.
    let q = narrow(&params.prime_factors[0]) as i128;
    let multiple = Ring::new(2 * q, 128).unwrap();
    let mut lifted = Statement::new(multiple.clone());
    lifted.var("s", 8, Norm::L2Squared(128)).unwrap();
    lifted.var("x", 2, Norm::L2Squared(64)).unwrap();
    let e = lifted.variable("s", 0).unwrap();
    lifted.eq_mod_p(e).unwrap();
    let req = lifted.requirements(64, U256::from_u128(q as u128)).unwrap();
    let mut p = params.clone();
    p.l = req.l;
    p.mlwe_rank += params.l - req.l;
    assert_eq!(
        lifted.compile(p).err(),
        Some(Error::Parameter("noninvertible statement modulus"))
    );
    // Witness blocks: missing, extra, renamed, wrong length, wrong ring.
    let mut s = vec![Poly::zero(ring.clone()); 8];
    s[0] = Poly::constant(ring.clone(), 5);
    let mut x = vec![Poly::zero(ring.clone()); 2];
    x[0] = Poly::constant(ring.clone(), -5);
    let witness = Witness::from([("s".into(), s.clone()), ("x".into(), x.clone())]);
    compiled.map_witness(&witness).unwrap();
    let mut w = witness.clone();
    w.remove("x");
    assert_eq!(compiled.map_witness(&w).err(), Some(Error::Dimension));
    let mut w = witness.clone();
    w.insert("y".into(), x.clone());
    assert_eq!(compiled.map_witness(&w).err(), Some(Error::Dimension));
    let mut w = witness.clone();
    let renamed = w.remove("x").unwrap();
    w.insert("z".into(), renamed);
    assert_eq!(compiled.map_witness(&w).err(), Some(Error::Witness));
    let mut w = witness.clone();
    w.get_mut("x").unwrap().pop();
    assert_eq!(compiled.map_witness(&w).err(), Some(Error::Dimension));
    let mut w = witness.clone();
    w.get_mut("x").unwrap()[1] = Poly::zero(Ring::new(17, 128).unwrap());
    assert_eq!(compiled.map_witness(&w).err(), Some(Error::Dimension));
    // The L2 bound is exact: ||x||^2 = 64 is accepted and 65 refused. x_1 is outside the
    // relation, and 25 + 36 + 1 + 1 + 1 = 64.
    let mut w = witness.clone();
    let mut x1 = Poly::zero(ring.clone());
    for (i, v) in [6, 1, -1, 1].into_iter().enumerate() {
        x1.set_coefficient(i, v).unwrap();
    }
    w.get_mut("x").unwrap()[1] = x1.clone();
    assert_eq!(norm2(&w["x"]), 64);
    compiled.map_witness(&w).unwrap();
    x1.set_coefficient(9, 1).unwrap();
    w.get_mut("x").unwrap()[1] = x1;
    assert_eq!(norm2(&w["x"]), 65);
    assert_eq!(compiled.map_witness(&w).err(), Some(Error::Witness));
}
