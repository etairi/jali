//! Why the compiler's lifting is sound, on the possession statement modulo 13 of
//! `tests/statement.rs`. A relation false modulo p = 13 is made true modulo the proof modulus q
//! by the carry $`f\cdot p^{-1}\bmod q`$, and by no other: every such carry is far above the
//! bound that extraction guarantees ($`2\cdot`$`z4_bound`), so an extracted short carry
//! gives the relation over the integers and hence modulo p. The honest prover refuses the long
//! carry, and a prover that skips its witness check cannot produce an accepted proof. The same
//! holds for lifted `const_coeff_zero` clauses, whose carries share packed polynomials with a
//! range row each, whichever slot or packed polynomial the false clause reads, and for clauses
//! with moduli of their own: a witness true modulo one needs a long carry for the other. Short
//! carries in the wrong slots fail the clauses' equations.
use super::*;
use crate::{
    math::{Ring, iso},
    params::TboxParams,
    statement::{Norm, Statement as Source},
};
use std::{collections::BTreeMap, sync::Arc};

/// The possession set: two carry polynomials, each with a range row.
fn params() -> TboxParams {
    let mut p = crate::params::toy_d64();
    p.id = "quadratic-possession-test-only".into();
    p.m1 = 20;
    p.m2 = 55;
    p.n_msis = 15;
    p.alpha_squared = 192;
    p.n_bin = 0;
    p.l2_rows = vec![16, 4];
    p.n_prime = 2;
    p.linf_bound = 13;
    p.log_sigma = [14, 12, 10, 13];
    p.d_bits = 6;
    p
}

/// `params` for up to 64 clauses: one packed carry polynomial with one range row, and the MLWE
/// rank one larger for the same $`m_2`$.
fn packed_params() -> TboxParams {
    let mut p = params();
    p.id = "packed-possession-test-only".into();
    p.l = 1;
    p.n_prime = 1;
    p.mlwe_rank += 1;
    p
}

/// $`p^{-1}`$ modulo $`q`$ for a small $`p`$.
fn inverse(p: i128, q: i128) -> i128 {
    (1..p)
        .map(|k| (1 + k * q) / p)
        .find(|x| (p * x) % q == 1)
        .unwrap()
}

/// A polynomial of the proof ring with the given values at the first coefficients.
fn slots(ring: &Arc<Ring>, values: &[i128]) -> Poly {
    let mut c = vec![0; ring.degree()];
    c[..values.len()].copy_from_slice(values);
    Poly::new(ring.clone(), c).unwrap()
}

/// The proof-ring Ajtai part of $`(s,x)`$ with $`s_0`$ and $`x_0`$ the given constants: each
/// statement polynomial split into its two components.
fn bounded(ring: &Arc<Ring>, s0: i128, x0: i128) -> PolyVec {
    let lifted = Ring::new(ring.modulus_i128(), 128).unwrap();
    let mut s = vec![Poly::zero(lifted.clone()); 10];
    s[0] = Poly::constant(lifted.clone(), s0);
    s[8] = Poly::constant(lifted, x0);
    PolyVec::new(
        ring.clone(),
        s.iter()
            .flat_map(|p| iso::split(p, ring.clone()).unwrap())
            .collect(),
    )
    .unwrap()
}

/// Every compiled equation holds on the proof-ring witness, only the range bound fails, and a
/// prover that skips the witness check gets no accepted proof.
fn only_the_range_fails(
    scheme: &Abdlop,
    statement: &Statement,
    s1: &PolyVec,
    m: &PolyVec,
    seed: u8,
) {
    let witness = crate::quad::interleave(s1, m).unwrap();
    for equation in &statement.quadratic {
        assert!(equation.evaluate(&witness).unwrap().is_zero());
    }
    for equation in &statement.evaluation {
        assert_eq!(
            equation.evaluate(&witness).unwrap().coefficient_i128(0),
            Ok(0)
        );
    }
    assert_eq!(
        statement.check_witness(scheme, s1, m).err(),
        Some(Error::Witness)
    );
    assert_eq!(
        prove_with_seed(scheme, statement, s1, m, b"lifting", [44; 32]).err(),
        Some(Error::Witness)
    );
    // Eight range attempts fail to hide a projection of the long carry (their responses
    // exceed q/2 or z4_bound, or rejection sampling refuses them); a proof, if one came out,
    // would have to pass the unmodified verifier.
    let seed = Zeroizing::new([seed; 32]);
    match prove_rounds(scheme, statement, s1, m, b"lifting", &seed, 8) {
        Err(e) => assert_eq!(e, Error::RestartLimit),
        Ok(proof) => assert!(verify(scheme, statement, &proof, b"lifting").is_err()),
    }
}

#[test]
fn a_relation_false_modulo_p_is_not_provable_modulo_q() {
    let source = Ring::new(13, 128).unwrap();
    let mut st = Source::new(source.clone());
    st.var("s", 8, Norm::L2Squared(128)).unwrap();
    st.var("x", 2, Norm::L2Squared(64)).unwrap();
    let minus = |e: QuadEq| e.scale(&Poly::constant(source.clone(), -1)).unwrap();
    let one = st.constant(Poly::constant(source.clone(), 1)).unwrap();
    let five = st.constant(Poly::constant(source.clone(), 5)).unwrap();
    let left = st
        .variable("s", 0)
        .unwrap()
        .add(&minus(one.clone()))
        .unwrap();
    let right = five.add(&minus(st.variable("x", 0).unwrap())).unwrap();
    st.eq_mod_p(
        left.product_affine(&right)
            .unwrap()
            .add(&minus(one))
            .unwrap(),
    )
    .unwrap();
    let compiled = st.compile(params()).unwrap();
    let scheme = Abdlop::new([33; 32], params()).unwrap();
    let ring = scheme.ring().clone();
    let q = ring.modulus_i128();
    // s_0 = 5, x_0 = -4: f = (5 - 1)(5 + 4) - 1 = 35, which 13 does not divide.
    let f = 35i128;
    let inverse = (1..13)
        .map(|k| (1 + k * q) / 13)
        .find(|x| (13 * x) % q == 1)
        .unwrap();
    let carry = crate::math::int::center(f * inverse % q, q).unwrap();
    assert_eq!((f - 13 * carry).rem_euclid(q), 0);
    assert!(carry.unsigned_abs() > 1 << 30);
    assert!(carry.unsigned_abs() > scheme.checked.approx_extraction_bound);
    // The witness in the proof ring: bounded variables in declaration order, each split into
    // its two degree-64 components, then the carry's components as messages.
    let lifted = Ring::new(q, 128).unwrap();
    let mut s = vec![Poly::zero(lifted.clone()); 10];
    s[0] = Poly::constant(lifted.clone(), 5);
    s[8] = Poly::constant(lifted.clone(), -4);
    let s1 = PolyVec::new(
        ring.clone(),
        s.iter()
            .flat_map(|p| iso::split(p, ring.clone()).unwrap())
            .collect(),
    )
    .unwrap();
    let m = PolyVec::new(
        ring.clone(),
        iso::split(&Poly::constant(lifted, carry), ring.clone()).unwrap(),
    )
    .unwrap();
    // Every lowered equation holds modulo q; only the range bound on the carry fails.
    let statement = compiled.statement();
    let witness = crate::quad::interleave(&s1, &m).unwrap();
    assert!(!statement.quadratic.is_empty());
    for equation in &statement.quadratic {
        assert!(equation.evaluate(&witness).unwrap().is_zero());
    }
    let arp = statement.arp.as_ref().unwrap().evaluate(&s1, &m).unwrap();
    assert_eq!(arp.entries()[0].coefficient_i128(0), Ok(carry));
    assert_eq!(
        statement.check_witness(&scheme, &s1, &m).err(),
        Some(Error::Witness)
    );
    assert_eq!(
        prove_with_seed(&scheme, statement, &s1, &m, b"lifting", [44; 32]).err(),
        Some(Error::Witness)
    );
    // Without the witness check, eight range attempts fail to hide a projection of the carry
    // (their responses exceed q/2 or z4_bound, or rejection sampling refuses them); a proof,
    // if one came out, would have to pass the unmodified verifier.
    let seed = Zeroizing::new([44; 32]);
    match prove_rounds(&scheme, statement, &s1, &m, b"lifting", &seed, 8) {
        Err(e) => assert_eq!(e, Error::RestartLimit),
        Ok(proof) => assert!(verify(&scheme, statement, &proof, b"lifting").is_err()),
    }
}

#[test]
fn a_constant_coefficient_clause_false_modulo_p_is_not_provable_modulo_q() {
    // ct(s_0 - 6) = 0 and ct(x_0 + 4) = 0 modulo 13, in either order: their carries are the
    // first two coefficients of one packed polynomial with one range row, so `packed_params`
    // fits. s_0 = 5 makes the clause on s false, in slot 0 or in slot 1.
    let source = Ring::new(13, 128).unwrap();
    for s_first in [true, false] {
        let mut st = Source::new(source.clone());
        st.var("s", 8, Norm::L2Squared(128)).unwrap();
        st.var("x", 2, Norm::L2Squared(64)).unwrap();
        let six = st.constant(Poly::constant(source.clone(), -6)).unwrap();
        let on_s = st.variable("s", 0).unwrap().add(&six).unwrap();
        let four = st.constant(Poly::constant(source.clone(), 4)).unwrap();
        let on_x = st.variable("x", 0).unwrap().add(&four).unwrap();
        let order = if s_first { [on_s, on_x] } else { [on_x, on_s] };
        for clause in order {
            st.const_coeff_zero(clause).unwrap();
        }
        let compiled = st.compile(packed_params()).unwrap();
        let scheme = Abdlop::new([33; 32], packed_params()).unwrap();
        let ring = scheme.ring().clone();
        let q = ring.modulus_i128();
        let carry = crate::math::int::center((-inverse(13, q)).rem_euclid(q), q).unwrap();
        assert_eq!((-1 - 13 * carry).rem_euclid(q), 0);
        assert!(carry.unsigned_abs() > scheme.checked.approx_extraction_bound);
        let s1 = bounded(&ring, 5, -4);
        // The long carry in the slot of the clause on s; the true clause's carry is 0.
        let slot = usize::from(!s_first);
        let mut values = [0; 2];
        values[slot] = carry;
        let m = PolyVec::new(ring.clone(), vec![slots(&ring, &values)]).unwrap();
        let statement = compiled.statement();
        assert!(statement.quadratic.is_empty());
        assert_eq!(statement.evaluation.len(), 2);
        // The one range row is the packed polynomial itself.
        let arp = statement.arp.as_ref().unwrap().evaluate(&s1, &m).unwrap();
        assert_eq!(arp.entries(), m.entries());
        only_the_range_fails(&scheme, statement, &s1, &m, 45);
    }
}

#[test]
fn a_clause_false_modulo_one_of_two_moduli_is_not_provable_modulo_q() {
    // ct(s_0 - 5) = 0 modulo 13 and modulo 12 on the same s_0: one packed carry polynomial,
    // slot 0 for the clause modulo 13 and slot 1 for the one modulo 12, so `packed_params`
    // fits. The statement ring, modulo 1009, holds the values used: s_0 = -8 satisfies the
    // clause modulo 13 only, and s_0 = -7 the one modulo 12.
    let source = Ring::new(1009, 128).unwrap();
    let mut st = Source::new(source.clone());
    st.var("s", 8, Norm::L2Squared(128)).unwrap();
    st.var("x", 2, Norm::L2Squared(64)).unwrap();
    for p in [13, 12] {
        let ring = Ring::new(p, 128).unwrap();
        let clause = st
            .variable_in(&ring, "s", 0)
            .unwrap()
            .add(&st.constant_in(Poly::constant(ring, -5)).unwrap())
            .unwrap();
        st.const_coeff_zero(clause).unwrap();
    }
    let compiled = st.compile(packed_params()).unwrap();
    let scheme = Abdlop::new([33; 32], packed_params()).unwrap();
    let ring = scheme.ring().clone();
    let q = ring.modulus_i128();
    let statement = compiled.statement();
    assert_eq!(statement.evaluation.len(), 2);
    for (value, false_modulus) in [(-8i128, 12i128), (-7, 13)] {
        let mut s = vec![Poly::zero(source.clone()); 8];
        s[0] = Poly::constant(source.clone(), value);
        let x = vec![Poly::zero(source.clone()); 2];
        let witness = BTreeMap::from([("s".to_string(), s), ("x".to_string(), x)]);
        assert_eq!(compiled.map_witness(&witness).err(), Some(Error::Witness));
        // The true clause's integer quotient, and for the false one the carry f * p^-1 mod q,
        // the only carry that makes its equation hold modulo q, far above the extraction bound.
        let f = value - 5;
        let carries = [13, 12].map(|p| {
            if p == false_modulus {
                crate::math::int::center(f * inverse(p, q), q).unwrap()
            } else {
                assert_eq!(f % p, 0);
                f / p
            }
        });
        let long = carries[usize::from(false_modulus == 12)];
        assert_eq!((f - false_modulus * long).rem_euclid(q), 0);
        assert!(long.unsigned_abs() > scheme.checked.approx_extraction_bound);
        let s1 = bounded(&ring, value, 0);
        let m = PolyVec::new(ring.clone(), vec![slots(&ring, &carries)]).unwrap();
        let arp = statement.arp.as_ref().unwrap().evaluate(&s1, &m).unwrap();
        assert_eq!(arp.entries(), m.entries());
        only_the_range_fails(&scheme, statement, &s1, &m, 46);
    }
}

#[test]
fn a_false_clause_in_the_second_packed_polynomial_is_not_provable_modulo_q() {
    // 65 clauses modulo 13, clause j reading coefficient j/8 of s_(j mod 8) through
    // ct(X^-(j/8) s_(j mod 8)): at degree 64 their carries fill two packed polynomials, the
    // shape of `params`. With s = 0 every clause holds but the last, ct(X^-8 s_0 + 1), which
    // sits in slot 0 of the second packed polynomial.
    let source = Ring::new(13, 128).unwrap();
    let mut st = Source::new(source.clone());
    st.var("s", 8, Norm::L2Squared(128)).unwrap();
    st.var("x", 2, Norm::L2Squared(64)).unwrap();
    for j in 0..65i64 {
        let read = Poly::constant(source.clone(), 1).rotate(-(j / 8));
        let mut clause = st
            .variable("s", (j % 8) as usize)
            .unwrap()
            .scale(&read)
            .unwrap();
        if j == 64 {
            let one = st.constant(Poly::constant(source.clone(), 1)).unwrap();
            clause = clause.add(&one).unwrap();
        }
        st.const_coeff_zero(clause).unwrap();
    }
    let compiled = st.compile(params()).unwrap();
    let scheme = Abdlop::new([33; 32], params()).unwrap();
    let ring = scheme.ring().clone();
    let q = ring.modulus_i128();
    let statement = compiled.statement();
    assert_eq!(statement.evaluation.len(), 65);
    assert_eq!(statement.arp.as_ref().unwrap().rows, 2);
    // The honest witness with the last clause's constant 0 would map; this one does not.
    let zero = |n| vec![Poly::zero(source.clone()); n];
    let witness = BTreeMap::from([("s".to_string(), zero(8)), ("x".to_string(), zero(2))]);
    assert_eq!(compiled.map_witness(&witness).err(), Some(Error::Witness));
    // f = 1: the long carry 13^-1 mod q at coefficient 0 of the second packed polynomial.
    let carry = crate::math::int::center(inverse(13, q), q).unwrap();
    assert!(carry.unsigned_abs() > scheme.checked.approx_extraction_bound);
    let s1 = bounded(&ring, 0, 0);
    let m = PolyVec::new(
        ring.clone(),
        vec![Poly::zero(ring.clone()), slots(&ring, &[carry])],
    )
    .unwrap();
    let arp = statement.arp.as_ref().unwrap().evaluate(&s1, &m).unwrap();
    assert_eq!(arp.entries(), m.entries());
    only_the_range_fails(&scheme, statement, &s1, &m, 47);
}

#[test]
fn short_carries_in_the_wrong_slots_fail_the_clauses() {
    // ct(s_0 + 2) = 0 and ct(x_0 - 5) = 0 modulo 13 on s_0 = 11 and x_0 = -8, over a statement
    // ring modulo 1009, which holds these values: integer values 13 and -13, carries 1 and -1 in
    // slots 0 and 1. Swapped, both stay within the range bound, but each clause reads the
    // other's carry: the witness check refuses, and a prover that skips it is refused by the
    // evaluation proof's own check of the equations.
    let source = Ring::new(1009, 128).unwrap();
    let r13 = Ring::new(13, 128).unwrap();
    let mut st = Source::new(source.clone());
    st.var("s", 8, Norm::L2Squared(128)).unwrap();
    st.var("x", 2, Norm::L2Squared(64)).unwrap();
    for (name, v) in [("s", 2), ("x", -5)] {
        let c = st.constant_in(Poly::constant(r13.clone(), v)).unwrap();
        let clause = st.variable_in(&r13, name, 0).unwrap().add(&c).unwrap();
        st.const_coeff_zero(clause).unwrap();
    }
    let compiled = st.compile(packed_params()).unwrap();
    let scheme = Abdlop::new([33; 32], packed_params()).unwrap();
    let ring = scheme.ring().clone();
    let named = |s0: i128, x0: i128| {
        let mut s = vec![Poly::zero(source.clone()); 8];
        s[0] = Poly::constant(source.clone(), s0);
        let mut x = vec![Poly::zero(source.clone()); 2];
        x[0] = Poly::constant(source.clone(), x0);
        BTreeMap::from([("s".to_string(), s), ("x".to_string(), x)])
    };
    let (s1, m) = compiled.map_witness(&named(11, -8)).unwrap();
    assert_eq!(m.entries(), [slots(&ring, &[1, -1])]);
    let statement = compiled.statement();
    statement.check_witness(&scheme, &s1, &m).unwrap();
    let swapped = PolyVec::new(ring.clone(), vec![slots(&ring, &[-1, 1])]).unwrap();
    assert_eq!(
        statement.check_witness(&scheme, &s1, &swapped).err(),
        Some(Error::Witness)
    );
    let seed = Zeroizing::new([48; 32]);
    assert_eq!(
        prove_rounds(&scheme, statement, &s1, &swapped, b"lifting", &seed, 64).err(),
        Some(Error::Witness)
    );
    // The honest carries prove.
    let proof = prove_with_seed(&scheme, statement, &s1, &m, b"lifting", [49; 32]).unwrap();
    verify(&scheme, statement, &proof, b"lifting").unwrap();
}
