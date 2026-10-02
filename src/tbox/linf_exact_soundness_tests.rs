//! What a verified proof says about a variable bounded exactly ([`Norm::LinfExact`]) and about a
//! subring variable ([`Source::var_subring`]), with a prover that skips its witness check
//! (`prove_rounds`). A `LinfExact` coefficient is $`\sum_bc_bx_b-\beta`$ over bits in the binary
//! block, whose weights reach exactly $`[0,2\beta]`$: a clause that needs $`\beta+1`$ has no
//! binary encoding, only a non-binary one, which the binary proof refuses, or, lifted, a carry
//! far above the extraction bound, which the range proof refuses (compare
//! `linf_soundness_tests`, where $`\beta+1`$ proves for [`Norm::Linf`]). So does a square, whose
//! substituted form expands the products of the bits. A flipped bit breaks the substituted
//! equation. A clause or a full equation that reads a coefficient outside a subring compiles to
//! an equation without a term for it, natively and lifted, and the witness map refuses a nonzero
//! coefficient outside the subring in every element of a block. Parameters: `toy_d64`
//! with the dimensions that `requirements` exports, checked.
use super::*;
use crate::{
    math::Ring,
    params::TboxParams,
    statement::{Extraction, Norm, Placement, Requirements, Statement as Source},
};
use std::sync::Arc;

/// `toy_d64` for the shape of `req` at proof degree 64: its dimensions and bounds, the
/// possession widths, and the smallest MSIS rank from 15 that the check accepts, with the MLWE
/// rank 26 of `toy_d64`.
fn params(req: &Requirements, id: &str) -> TboxParams {
    let mut p = crate::params::toy_d64();
    p.id = id.into();
    p.m1 = req.m1;
    p.l = req.l;
    p.alpha_squared = u64::try_from(req.alpha_squared).unwrap();
    p.n_bin = req.n_bin;
    p.l2_rows = req.l2_rows.clone();
    p.l2_bounds_squared = req.l2_bounds_squared.clone();
    p.n_prime = req.n_prime;
    p.linf_bound = u64::try_from(req.linf_bound).unwrap();
    p.log_sigma = [14, 12, 10, 13];
    p.d_bits = 6;
    // m2 = rank + n + l + l_ext, l_ext = 4 exact and 4 range slots when present, the sign,
    // lambda/2 = 2 garbage polynomials and one more.
    let n_ex = p.n_bin + p.l2_rows.iter().sum::<usize>() + p.l2_rows.len();
    let l_ext = 4 * usize::from(n_ex > 0) + 4 * usize::from(p.n_prime > 0) + 4;
    for n in 15..40 {
        p.n_msis = n;
        p.m2 = 26 + n + p.l + l_ext;
        if p.check().is_ok() {
            return p;
        }
    }
    panic!("no MSIS rank fits {id}: {:?}", p.check());
}

/// The possession modulus of `toy_d64`, a native modulus below. Read from every limb: with
/// 32-bit limbs the first holds only its low 32 bits.
fn q() -> i128 {
    let q = &crate::params::toy_d64().prime_factors[0];
    i128::from(crate::math::int::to_u64(q).expect("a 41-bit modulus"))
}

/// $`p^{-1}\bmod q`$ for $`1<p<q`$, $`q`$ prime: the $`k<p`$ with $`p\mid1+kq`$.
fn inverse_mod(p: i128, q: i128) -> i128 {
    (1..p)
        .map(|k| (1 + k * q) / p)
        .find(|x| (p * x) % q == 1)
        .unwrap()
}

/// Every compiled equation holds on the proof-ring witness, a prover that skips the witness
/// check gets no accepted proof (the evaluation proof's own check of the equations, which
/// include the binary one, refuses it, eight range attempts fail, or the unmodified verifier
/// rejects what comes out), and the witness check and the honest prover refuse it.
fn not_provable(scheme: &Abdlop, statement: &Statement, s1: &PolyVec, m: &PolyVec, seed: u8) {
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
    let key = Zeroizing::new([seed; 32]);
    match prove_rounds(scheme, statement, s1, m, b"linf-exact", &key, 8) {
        Err(e) => assert!(matches!(e, Error::Witness | Error::RestartLimit), "{e:?}"),
        Ok(proof) => assert!(verify(scheme, statement, &proof, b"linf-exact").is_err()),
    }
    assert_eq!(
        statement.check_witness(scheme, s1, m).err(),
        Some(Error::Witness)
    );
    assert_eq!(
        prove_with_seed(scheme, statement, s1, m, b"linf-exact", [seed; 32]).err(),
        Some(Error::Witness)
    );
}

/// Over `source` (degree 128): `s` (eight polynomials, $`\|s\|^2\le128`$) and `u`
/// ($`|u_i|\le\beta`$ exactly), and the clause $`\mathrm{ct}(u)=c`$ over `clause_ring`.
fn clause_on_u(source: &Arc<Ring>, beta: u64, clause_ring: &Arc<Ring>, c: i128) -> Source {
    let mut st = Source::new(source.clone());
    st.var("s", 8, Norm::L2Squared(128)).unwrap();
    st.var("u", 1, Norm::LinfExact(beta)).unwrap();
    let clause = st
        .variable_in(clause_ring, "u", 0)
        .unwrap()
        .add(
            &st.constant_in(Poly::constant(clause_ring.clone(), -c))
                .unwrap(),
        )
        .unwrap();
    st.const_coeff_zero(clause).unwrap();
    st
}

/// The Ajtai part for `clause_on_u` at proof degree 64: `s` zero (16 polynomials), then for each
/// of u's two components its bits, set to the weights' greedy bits of $`0+\beta`$ at every
/// position but position 0 of component 0, which gets `first`.
fn bits_witness(ring: &Arc<Ring>, weights: &[i128], first: &[i128]) -> PolyVec {
    let n = weights.len();
    let mut polys = vec![Poly::zero(ring.clone()); 16 + 2 * n];
    let beta: i128 = weights.iter().sum::<i128>() / 2;
    // The greedy bits of beta, from the first weight on.
    let mut rest = beta;
    for (b, w) in weights.iter().enumerate() {
        if rest >= *w {
            rest -= w;
            for component in 0..2 {
                polys[16 + component * n + b] = Poly::new(ring.clone(), vec![1; 64]).unwrap();
            }
        }
    }
    for (b, bit) in first.iter().enumerate() {
        polys[16 + b].set_coefficient(0, *bit).unwrap();
    }
    PolyVec::new(ring.clone(), polys).unwrap()
}

#[test]
fn a_value_beyond_beta_has_no_binary_encoding_and_is_not_provable() {
    // Native over Z_q: ct(u) = 6 = beta + 1 for beta = 5, whose weights (5, 3, 1, 1) reach at
    // most 10 = 2 beta, while 6 needs x = u + beta = 11.
    let q = q();
    let rq = Ring::new(q, 128).unwrap();
    let st = clause_on_u(&rq, 5, &rq, 6);
    let req = st.requirements(64, U256::from_u128(q as u128)).unwrap();
    // s: 16 rows; u: two components of four bits, in the binary block; nothing lifted.
    assert_eq!(
        (req.m1, req.l, req.n_bin, req.n_prime, req.alpha_squared),
        (24, 0, 8, 0, 128 + 128 * 4)
    );
    let params = params(&req, "linf-exact-native-test-only");
    let compiled = st.compile(params.clone()).unwrap();
    assert_eq!(compiled.extraction_bound("u"), Ok(Extraction::LinfExact(5)));
    let scheme = Abdlop::new([71; 32], params).unwrap();
    let ring = scheme.ring().clone();
    let statement = compiled.statement();
    assert_eq!(statement.evaluation.len(), 1);
    // The binary block selects the eight bit polynomials.
    let binary = statement.binary.as_ref().unwrap();
    let probe = PolyVec::new(
        ring.clone(),
        (0..24)
            .map(|i| Poly::constant(ring.clone(), i as i128))
            .collect(),
    )
    .unwrap();
    let selected = binary
        .evaluate(&probe, &PolyVec::zero(ring.clone(), 0))
        .unwrap();
    assert_eq!(selected.entries(), &probe.entries()[16..]);
    let messages = PolyVec::zero(ring.clone(), 0);
    let weights = [5, 3, 1, 1];
    // No binary bits at position 0 satisfy the clause: the substituted equation reads
    // sum c_b x_b - 5 - 6 there.
    let at = |bits: &[i128]| {
        let s1 = bits_witness(&ring, &weights, bits);
        statement.evaluation[0]
            .evaluate(&crate::quad::interleave(&s1, &messages).unwrap())
            .unwrap()
            .coefficient_i128(0)
            .unwrap()
    };
    let patterns: Vec<Vec<i128>> = (0..16i128)
        .map(|p| (0..4).map(|b| (p >> b) & 1).collect())
        .collect();
    // Whatever the compiled weights: no binary bits satisfy the clause.
    for bits in &patterns {
        assert_ne!(at(bits), 0, "the binary bits {bits:?} encode 6");
    }
    for bits in &patterns {
        let x: i128 = bits.iter().zip(weights).map(|(b, w)| b * w).sum();
        assert_eq!(at(bits), x - 11, "bits {bits:?}");
    }
    // A non-binary "bit": 2 * 5 + 1 = 11. Only the binary check fails.
    let s1 = bits_witness(&ring, &weights, &[2, 0, 0, 1]);
    not_provable(&scheme, statement, &s1, &messages, 72);
}

#[test]
fn a_flipped_bit_breaks_the_substituted_equation() {
    // ct(u) = 5 = beta: x = 10, bits (1, 1, 1, 1) at position 0. The honest witness proves;
    // with one bit flipped the equation reads 5 - 10 at position 0 and the evaluation proof's
    // own check of the equations refuses the witness.
    let q = q();
    let rq = Ring::new(q, 128).unwrap();
    let st = clause_on_u(&rq, 5, &rq, 5);
    let req = st.requirements(64, U256::from_u128(q as u128)).unwrap();
    let params = params(&req, "linf-exact-native-test-only");
    let compiled = st.compile(params.clone()).unwrap();
    let scheme = Abdlop::new([73; 32], params).unwrap();
    let ring = scheme.ring().clone();
    let mut u = Poly::zero(rq.clone());
    u.set_coefficient(0, 5).unwrap();
    let named = std::collections::BTreeMap::from([
        ("s".to_string(), vec![Poly::zero(rq.clone()); 8]),
        ("u".to_string(), vec![u]),
    ]);
    let (s1, m) = compiled.map_witness(&named).unwrap();
    assert_eq!(s1, bits_witness(&ring, &[5, 3, 1, 1], &[1, 1, 1, 1]));
    let statement = compiled.statement();
    let proof = prove_with_seed(&scheme, statement, &s1, &m, b"linf-exact", [74; 32]).unwrap();
    verify(&scheme, statement, &proof, b"linf-exact").unwrap();
    let flipped = bits_witness(&ring, &[5, 3, 1, 1], &[0, 1, 1, 1]);
    assert_eq!(
        statement.check_witness(&scheme, &flipped, &m).err(),
        Some(Error::Witness)
    );
    let key = Zeroizing::new([75; 32]);
    assert_eq!(
        prove_rounds(&scheme, statement, &flipped, &m, b"linf-exact", &key, 8).err(),
        Some(Error::Witness)
    );
}

#[test]
fn a_lifted_clause_beyond_beta_needs_a_non_binary_bit_or_a_long_carry() {
    // Over Z_13: ct(u) = 2 = beta + 1 for beta = 1 (weights (1, 1)), lifted with a packed
    // carry. With bits (2, 1) the carry is 0 and only the binary check fails; with binary bits
    // u_0 is -1, 0 or 1, and the carry that makes the equation hold modulo q is
    // (u_0 - 2) 13^-1 mod q, far above the extraction bound, so only the range fails.
    let r13 = Ring::new(13, 128).unwrap();
    let st = clause_on_u(&r13, 1, &r13, 2);
    let req = st.requirements(64, U256::from_u128(q() as u128)).unwrap();
    assert_eq!(
        (req.m1, req.l, req.n_bin, req.n_prime, req.linf_bound),
        (20, 1, 4, 1, 1)
    );
    let params = params(&req, "linf-exact-lifted-test-only");
    let compiled = st.compile(params.clone()).unwrap();
    let scheme = Abdlop::new([76; 32], params).unwrap();
    let ring = scheme.ring().clone();
    let q = ring.modulus_i128();
    let statement = compiled.statement();
    let zero = PolyVec::new(ring.clone(), vec![Poly::zero(ring.clone())]).unwrap();
    let s1 = bits_witness(&ring, &[1, 1], &[2, 1]);
    not_provable(&scheme, statement, &s1, &zero, 77);
    let inverse = (1..13)
        .map(|k| (1 + k * q) / 13)
        .find(|x| (13 * x) % q == 1)
        .unwrap();
    for (bits, u0) in [([0, 0], -1i128), ([1, 0], 0), ([1, 1], 1)] {
        let carry = crate::math::int::center((u0 - 2) * inverse % q, q).unwrap();
        assert!(carry.unsigned_abs() > scheme.checked.approx_extraction_bound);
        let mut c = Poly::zero(ring.clone());
        c.set_coefficient(0, carry).unwrap();
        let m = PolyVec::new(ring.clone(), vec![c]).unwrap();
        let s1 = bits_witness(&ring, &[1, 1], &bits);
        // The range row is the packed carry itself.
        let arp = statement.arp.as_ref().unwrap().evaluate(&s1, &m).unwrap();
        assert_eq!(arp.entries(), m.entries());
        not_provable(&scheme, statement, &s1, &m, 78);
    }
}

#[test]
fn a_square_beyond_beta_needs_a_non_binary_bit_or_a_long_carry() {
    // Over Z_12289, lifted with a packed carry: ct(u u) = 16 for beta = 3 (weights (3, 2, 1)).
    // With u_0 at coefficient 0 and 0 elsewhere, ct(u u) = u_0^2, so the clause needs |u_0| = 4.
    // The bits (1, 1, 2) give u_0 = 4 with carry 0, and only the binary check fails; binary
    // bits give |u_0| <= 3, whose carry (u_0^2 - 16) 12289^-1 mod q is far above the extraction
    // bound, and only the range fails. The substituted square must expand the products of the
    // bits exactly, cross terms twice, for these equations to hold.
    let p = 12289;
    let source = Ring::new(p, 128).unwrap();
    let mut st = Source::new(source.clone());
    st.var("s", 8, Norm::L2Squared(128)).unwrap();
    st.var("u", 1, Norm::LinfExact(3)).unwrap();
    let u = st.variable("u", 0).unwrap();
    let clause = u
        .product_affine(&u)
        .unwrap()
        .add(&st.constant(Poly::constant(source.clone(), -16)).unwrap())
        .unwrap();
    st.const_coeff_zero(clause).unwrap();
    let req = st.requirements(64, U256::from_u128(q() as u128)).unwrap();
    // s: 16 rows; u: two components of three bits; one packed carry.
    assert_eq!((req.m1, req.l, req.n_bin, req.n_prime), (22, 1, 6, 1));
    let params = params(&req, "linf-exact-square-test-only");
    let compiled = st.compile(params.clone()).unwrap();
    for u0 in [4, -4] {
        let mut u = Poly::zero(source.clone());
        u.set_coefficient(0, u0).unwrap();
        let named = std::collections::BTreeMap::from([
            ("s".to_string(), vec![Poly::zero(source.clone()); 8]),
            ("u".to_string(), vec![u]),
        ]);
        assert_eq!(compiled.map_witness(&named).err(), Some(Error::Witness));
    }
    let scheme = Abdlop::new([81; 32], params).unwrap();
    let ring = scheme.ring().clone();
    let q = ring.modulus_i128();
    let statement = compiled.statement();
    let weights = [3, 2, 1];
    let zero = PolyVec::new(ring.clone(), vec![Poly::zero(ring.clone())]).unwrap();
    not_provable(
        &scheme,
        statement,
        &bits_witness(&ring, &weights, &[1, 1, 2]),
        &zero,
        82,
    );
    let inverse = inverse_mod(p, q);
    for pattern in 0..8 {
        let bits: Vec<i128> = (0..3).map(|b| (pattern >> b) & 1).collect();
        let u0 = bits.iter().zip(weights).map(|(b, w)| b * w).sum::<i128>() - 3;
        let carry = crate::math::int::center((u0 * u0 - 16) * inverse % q, q).unwrap();
        assert!(
            carry.unsigned_abs() > scheme.checked.approx_extraction_bound,
            "u_0 {u0}"
        );
        // u_0 = 3, the largest binary value, with its carry.
        if pattern == 7 {
            let mut c = Poly::zero(ring.clone());
            c.set_coefficient(0, carry).unwrap();
            let m = PolyVec::new(ring.clone(), vec![c]).unwrap();
            not_provable(
                &scheme,
                statement,
                &bits_witness(&ring, &weights, &bits),
                &m,
                83,
            );
        }
    }
}

#[test]
fn a_clause_on_a_coefficient_outside_the_subring_compiles_without_a_term_for_it() {
    // Over Z_q, degree 128: v in the subring of degree 64 (K = 2), whose coefficient 1 is 0 in
    // every witness, and the clause ct(X^-1 v) = 1, which reads it: no witness exists. At proof
    // degree 64, v commits only component 0; coefficient 1 lies in component 1, which is not
    // committed, so the compiled equation reads the constant -1 and nothing else.
    let q = q();
    let rq = Ring::new(q, 128).unwrap();
    let mut st = Source::new(rq.clone());
    st.var("s", 8, Norm::L2Squared(128)).unwrap();
    st.var_subring("v", 1, 64, Norm::L2Squared(64), Placement::Ajtai)
        .unwrap();
    let read = Poly::constant(rq.clone(), 1).rotate(-1);
    let clause = st
        .variable("v", 0)
        .unwrap()
        .scale(&read)
        .unwrap()
        .add(&st.constant(Poly::constant(rq.clone(), -1)).unwrap())
        .unwrap();
    st.const_coeff_zero(clause).unwrap();
    let req = st.requirements(64, U256::from_u128(q as u128)).unwrap();
    // s: 16 rows, v: one; two exact-norm blocks.
    assert_eq!((req.m1, req.l2_rows.clone()), (17, vec![16, 1]));
    let params = params(&req, "subring-native-test-only");
    let compiled = st.compile(params.clone()).unwrap();
    let statement = compiled.statement();
    let equation = &statement.evaluation[0];
    assert!(equation.r2.entries().all(|(_, p)| p.is_zero()));
    assert!(equation.r1.entries().all(|(_, p)| p.is_zero()));
    assert_eq!(equation.r0.coefficient_i128(0), Ok(-1));
    // v with coefficient 1 set is refused by the witness map; a prover that skips the check,
    // with v's committed component 1 at every coefficient, is refused by the evaluation
    // proof's check of the equation.
    let mut v = Poly::zero(rq.clone());
    v.set_coefficient(1, 1).unwrap();
    let named = std::collections::BTreeMap::from([
        ("s".to_string(), vec![Poly::zero(rq.clone()); 8]),
        ("v".to_string(), vec![v]),
    ]);
    assert_eq!(compiled.map_witness(&named).err(), Some(Error::Witness));
    let scheme = Abdlop::new([79; 32], params).unwrap();
    let ring = scheme.ring().clone();
    let mut polys = vec![Poly::zero(ring.clone()); 17];
    polys[16] = Poly::new(ring.clone(), vec![1; 64]).unwrap();
    let s1 = PolyVec::new(ring.clone(), polys).unwrap();
    let m = PolyVec::zero(ring.clone(), 0);
    assert_eq!(
        statement.check_witness(&scheme, &s1, &m).err(),
        Some(Error::Witness)
    );
    let key = Zeroizing::new([80; 32]);
    assert_eq!(
        prove_rounds(&scheme, statement, &s1, &m, b"subring", &key, 8).err(),
        Some(Error::Witness)
    );
}

/// Over `source` (degree 128): `s` (eight polynomials, $`\|s\|^2\le128`$) and `v` (one element
/// of the subring of degree 64, $`K=2`$, $`\|v\|^2\le64`$), the constraint built from
/// $`X^{-1}v-1`$ by `constrain`, and `v` = $`X`$, which satisfies it but lies outside the
/// subring.
fn on_x_inverse_v(
    source: &Arc<Ring>,
    constrain: fn(&mut Source, crate::quad::QuadEq),
) -> (Source, std::collections::BTreeMap<String, Vec<Poly>>) {
    let mut st = Source::new(source.clone());
    st.var("s", 8, Norm::L2Squared(128)).unwrap();
    st.var_subring("v", 1, 64, Norm::L2Squared(64), Placement::Ajtai)
        .unwrap();
    let form = st
        .variable("v", 0)
        .unwrap()
        .scale(&Poly::constant(source.clone(), 1).rotate(-1))
        .unwrap()
        .add(&st.constant(Poly::constant(source.clone(), -1)).unwrap())
        .unwrap();
    constrain(&mut st, form);
    let mut v = Poly::zero(source.clone());
    v.set_coefficient(1, 1).unwrap();
    let named = std::collections::BTreeMap::from([
        ("s".to_string(), vec![Poly::zero(source.clone()); 8]),
        ("v".to_string(), vec![v]),
    ]);
    (st, named)
}

#[test]
fn a_full_equation_on_a_component_outside_the_subring_has_no_term_for_it() {
    // X^-1 v - 1 = 0, whose only solution v = X lies outside the subring, over Z_q and over
    // Z_13. At proof degree 64, component 0 of X^-1 v reads v's component 1, which is not
    // committed: natively the compiled component-0 equation is the constant -1, and lifted
    // modulo 13 its implicit quotient's range row is the constant -13^-1 mod q, far above the
    // extraction bound, whatever the committed component of v holds.
    let q = q();
    for (p, lifted) in [(q, false), (13, true)] {
        let source = Ring::new(p, 128).unwrap();
        let (st, named) = on_x_inverse_v(&source, |st, form| {
            st.eq_mod_p(form).unwrap();
        });
        let req = st.requirements(64, U256::from_u128(q as u128)).unwrap();
        // v commits one component; lifted, both components have implicit quotients.
        assert_eq!(
            (req.m1, req.l, req.n_prime),
            (17, 0, 2 * usize::from(lifted))
        );
        let params = params(&req, "subring-equation-test-only");
        let compiled = st.compile(params.clone()).unwrap();
        assert_eq!(compiled.map_witness(&named).err(), Some(Error::Witness));
        let scheme = Abdlop::new([84; 32], params).unwrap();
        let ring = scheme.ring().clone();
        let statement = compiled.statement();
        let m = PolyVec::zero(ring.clone(), 0);
        // v's committed component, within the bound 64 (`prove_rounds` needs the norms).
        let mut sparse = vec![0; 64];
        (sparse[0], sparse[5]) = (7, -3);
        for (case, component) in [vec![0; 64], vec![1; 64], vec![-1; 64], sparse]
            .into_iter()
            .enumerate()
        {
            let mut polys = vec![Poly::zero(ring.clone()); 17];
            polys[16] = Poly::new(ring.clone(), component).unwrap();
            let s1 = PolyVec::new(ring.clone(), polys).unwrap();
            if lifted {
                assert!(statement.quadratic.is_empty());
                let row = crate::math::int::center(-inverse_mod(13, q), q).unwrap();
                assert!(row.unsigned_abs() > scheme.checked.approx_extraction_bound);
                let arp = statement.arp.as_ref().unwrap().evaluate(&s1, &m).unwrap();
                assert_eq!(
                    arp.entries()[0],
                    Poly::constant(ring.clone(), row),
                    "case {case}"
                );
                if case == 0 {
                    not_provable(&scheme, statement, &s1, &m, 85);
                }
            } else {
                let equation = &statement.quadratic[0];
                assert!(equation.r1.entries().all(|(_, p)| p.is_zero()));
                assert!(equation.r2.entries().all(|(_, p)| p.is_zero()));
                let witness = crate::quad::interleave(&s1, &m).unwrap();
                assert_eq!(
                    equation.evaluate(&witness).unwrap(),
                    Poly::constant(ring.clone(), -1)
                );
                assert_eq!(
                    statement.check_witness(&scheme, &s1, &m).err(),
                    Some(Error::Witness)
                );
                let key = Zeroizing::new([86; 32]);
                assert_eq!(
                    prove_rounds(&scheme, statement, &s1, &m, b"subring", &key, 8).err(),
                    Some(Error::Witness),
                    "case {case}"
                );
            }
        }
    }
}

#[test]
fn a_lifted_clause_on_a_coefficient_outside_the_subring_needs_a_long_carry() {
    // ct(X^-1 v) = 1 over Z_13, lifted with a packed carry, reads coefficient 1 of v, which is
    // 0 in every witness. The compiled equation's only variable is the carry, which must be
    // -13^-1 mod q, far above the extraction bound, whatever the committed component of v.
    let q = q();
    let source = Ring::new(13, 128).unwrap();
    let (st, named) = on_x_inverse_v(&source, |st, form| {
        st.const_coeff_zero(form).unwrap();
    });
    let req = st.requirements(64, U256::from_u128(q as u128)).unwrap();
    assert_eq!((req.m1, req.l, req.n_prime), (17, 1, 1));
    let params = params(&req, "subring-clause-test-only");
    let compiled = st.compile(params.clone()).unwrap();
    assert_eq!(compiled.map_witness(&named).err(), Some(Error::Witness));
    let scheme = Abdlop::new([87; 32], params).unwrap();
    let ring = scheme.ring().clone();
    let statement = compiled.statement();
    // The packed carry is message 0, witness polynomial 17, interleaved index 34.
    let equation = &statement.evaluation[0];
    assert!(equation.r2.entries().all(|(_, p)| p.is_zero()));
    let terms: Vec<u16> = equation
        .r1
        .entries()
        .filter(|(_, p)| !p.is_zero())
        .map(|(i, _)| i)
        .collect();
    assert_eq!(terms, [34]);
    let carry = crate::math::int::center(-inverse_mod(13, q), q).unwrap();
    assert!(carry.unsigned_abs() > scheme.checked.approx_extraction_bound);
    let mut c = Poly::zero(ring.clone());
    c.set_coefficient(0, carry).unwrap();
    let m = PolyVec::new(ring.clone(), vec![c]).unwrap();
    for fill in [0, 1, 5, q - 1] {
        let mut polys = vec![Poly::zero(ring.clone()); 17];
        polys[16] = Poly::new(ring.clone(), vec![fill; 64]).unwrap();
        let s1 = PolyVec::new(ring.clone(), polys).unwrap();
        let witness = crate::quad::interleave(&s1, &m).unwrap();
        assert_eq!(
            equation.evaluate(&witness).unwrap().coefficient_i128(0),
            Ok(0),
            "fill {fill}"
        );
    }
    not_provable(&scheme, statement, &PolyVec::zero(ring.clone(), 17), &m, 88);
}

#[test]
fn junk_outside_the_subring_is_refused_in_every_element() {
    // v: two elements of the subring of degree 64 (K = 2) that no constraint reads, so that
    // only the witness map's zero check can refuse a nonzero coefficient 3 in either element.
    let q = q();
    let rq = Ring::new(q, 128).unwrap();
    let mut st = Source::new(rq.clone());
    st.var("s", 8, Norm::L2Squared(128)).unwrap();
    st.var_subring("v", 2, 64, Norm::L2Squared(64), Placement::Ajtai)
        .unwrap();
    let clause = st.variable("s", 0).unwrap();
    st.const_coeff_zero(clause).unwrap();
    let req = st.requirements(64, U256::from_u128(q as u128)).unwrap();
    let compiled = st.compile(params(&req, "subring-junk-test-only")).unwrap();
    let zero = |n| vec![Poly::zero(rq.clone()); n];
    let named = |v: Vec<Poly>| {
        std::collections::BTreeMap::from([("s".to_string(), zero(8)), ("v".to_string(), v)])
    };
    compiled.map_witness(&named(zero(2))).unwrap();
    for element in [0, 1] {
        let mut v = zero(2);
        v[element].set_coefficient(3, 1).unwrap();
        assert_eq!(
            compiled.map_witness(&named(v)).err(),
            Some(Error::Witness),
            "element {element}"
        );
    }
}
