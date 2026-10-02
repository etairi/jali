use jali::{
    Error,
    abdlop::Abdlop,
    math::{Poly, PolyVec, Ring, SparsePolyMat, SparsePolyVec, U256},
    params::{TboxParams, toy_d64},
    quad::{self, QuadEq},
    quad_eval, quad_many,
};

#[test]
fn quadratic_many_and_evaluation_proofs_bind_the_entire_statement() {
    let scheme = Abdlop::new([1; 32], toy_d64()).unwrap();
    let ring = scheme.ring().clone();
    let s1 = PolyVec::new(
        ring.clone(),
        (0..scheme.bounded_len())
            .map(|i| Poly::constant(ring.clone(), (i % 3) as i128))
            .collect(),
    )
    .unwrap();
    let m = PolyVec::new(
        ring.clone(),
        vec![
            Poly::constant(ring.clone(), 7),
            Poly::constant(ring.clone(), 11),
        ],
    )
    .unwrap();
    let witness = quad::interleave(&s1, &m).unwrap();
    let dim = witness.len();
    let one = Poly::constant(ring.clone(), 1);
    let mut equation = QuadEq {
        r2: SparsePolyMat::new(
            ring.clone(),
            dim,
            vec![
                (0, 2, one.clone()),
                (2, 3, one.rotate(1)),
                (0, 25, one.neg()),
            ],
        )
        .unwrap(),
        r1: SparsePolyVec::new(ring.clone(), dim, vec![(0, one.clone()), (24, one.clone())])
            .unwrap(),
        r0: Poly::zero(ring.clone()),
    };
    equation.r0 = equation.evaluate(&witness).unwrap().neg();
    assert_eq!(
        equation.trace().unwrap().evaluate(&witness).unwrap(),
        equation.evaluate(&witness).unwrap().trace().unwrap()
    );
    let (commitment, opening) = scheme.commit_with_seed(s1, m, [2; 32]).unwrap();
    let proof =
        quad::prove_with_seed(&scheme, &commitment, &opening, &equation, b"quad", [3; 32]).unwrap();
    quad::verify(&scheme, &commitment, &equation, &proof, b"quad").unwrap();
    let mut bad = proof.clone();
    bad.t = bad.t.add(&one).unwrap();
    assert!(quad::verify(&scheme, &commitment, &equation, &bad, b"quad").is_err());
    let mut false_eq = equation.clone();
    false_eq.r0 = false_eq.r0.add(&one).unwrap();
    assert!(quad::verify(&scheme, &commitment, &false_eq, &proof, b"quad").is_err());
    assert!(matches!(
        quad::prove_with_seed(&scheme, &commitment, &opening, &false_eq, b"quad", [3; 32]),
        Err(Error::Witness)
    ));
    let equations = vec![equation.clone(), equation.scale(&one.rotate(3)).unwrap()];
    let proof =
        quad_many::prove_with_seed(&scheme, &commitment, &opening, &equations, b"many", [4; 32])
            .unwrap();
    quad_many::verify(&scheme, &commitment, &equations, &proof, b"many").unwrap();
    let mut eval = equation.clone();
    eval.r0 = eval.r0.add(&one.rotate(1)).unwrap();
    assert!(!eval.evaluate(&witness).unwrap().is_zero());
    let proof = quad_eval::prove_with_seed(
        &scheme,
        &commitment,
        &opening,
        &equations,
        &[eval.clone()],
        b"eval",
        [5; 32],
    )
    .unwrap();
    quad_eval::verify(
        &scheme,
        &commitment,
        &equations,
        &[eval.clone()],
        &proof,
        b"eval",
    )
    .unwrap();
    for coefficient in [0, 32] {
        let mut bad = proof.clone();
        let mut h = bad.h.entries().to_vec();
        h[0].set_coefficient(coefficient, 1).unwrap();
        bad.h = PolyVec::new(ring.clone(), h).unwrap();
        assert!(
            quad_eval::verify(
                &scheme,
                &commitment,
                &equations,
                &[eval.clone()],
                &bad,
                b"eval"
            )
            .is_err()
        );
    }
    let mut bad = proof.clone();
    let mut h = bad.h.entries().to_vec();
    h[0] = h[0].add(&one.rotate(3)).unwrap();
    bad.h = PolyVec::new(ring, h).unwrap();
    assert!(quad_eval::verify(&scheme, &commitment, &equations, &[eval], &bad, b"eval").is_err());
}

#[test]
fn evaluation_proofs_under_two_contexts_from_one_seed_commit_to_different_garbage() {
    // The two proofs share the opening, so one garbage vector would give equal garbage rows.
    let scheme = Abdlop::new([1; 32], toy_d64()).unwrap();
    let ring = scheme.ring().clone();
    let s1 = PolyVec::new(
        ring.clone(),
        (0..scheme.bounded_len())
            .map(|i| Poly::constant(ring.clone(), (i % 3) as i128))
            .collect(),
    )
    .unwrap();
    let m = PolyVec::new(
        ring.clone(),
        vec![Poly::constant(ring.clone(), 7), Poly::constant(ring, 11)],
    )
    .unwrap();
    let seed = [5; 32];
    let (commitment, opening) = scheme.commit_with_seed(s1, m, seed).unwrap();
    let [a, b] = [b"context A".as_slice(), b"context B"].map(|context| {
        quad_eval::prove_with_seed(&scheme, &commitment, &opening, &[], &[], context, seed).unwrap()
    });
    assert_ne!(a.garbage_commitments, b.garbage_commitments);
}

/// Algebraic coverage only, with synthetic MLWE metadata; not a shipped parameter set. A
/// two-prime modulus admits no binary, exact-norm or range block, so none is declared.
fn two_prime_params() -> TboxParams {
    let mut p = toy_d64();
    p.id = "two-prime-test-only".into();
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
    // m2 - n_msis - l - l_ext with l_ext = 1 + lambda/2 + 1 = 6 once no range slot is reserved.
    p.mlwe_rank = 30;
    p.estimator = "synthetic test metadata; no hardness estimate".into();
    p
}

#[test]
fn two_prime_moduli_are_refused_for_binary_norm_and_range_blocks() {
    let p = two_prime_params();
    let q: i128 = 1048589 * 4194389;
    // The CRT value (1 mod q1, -1 mod q2) squares to one without being a sign, which is why the
    // range proofs' sign check needs a prime modulus.
    let u = (1..4194389i128)
        .map(|k| 1 + k * 1048589)
        .find(|x| (x + 1) % 4194389 == 0)
        .unwrap();
    let ring = Ring::new(q, 128).unwrap();
    let b = Poly::constant(ring.clone(), u);
    assert_eq!(b.mul(&b).unwrap(), Poly::constant(ring, 1));
    assert!(u != 1 && u != q - 1);
    assert!(p.check().is_ok());
    for block in 0..3 {
        let mut p = p.clone();
        match block {
            0 => p.n_bin = 2,
            1 => {
                p.l2_rows = vec![1];
                p.l2_bounds_squared = vec![64];
            }
            _ => {
                p.n_prime = 2;
                p.linf_bound = 4;
            }
        }
        assert_eq!(
            p.check().err(),
            Some(Error::Parameter(
                "binary, exact-norm and range blocks require a prime modulus"
            )),
            "block {block}"
        );
    }
}

#[test]
fn sixteen_relations_at_degree_128_over_a_two_prime_modulus() {
    let scheme = Abdlop::new([121; 32], two_prime_params()).unwrap();
    let ring = scheme.ring().clone();
    let s = PolyVec::new(
        ring.clone(),
        vec![Poly::constant(ring.clone(), 1).rotate(7); scheme.bounded_len()],
    )
    .unwrap();
    let m = PolyVec::zero(ring.clone(), scheme.message_len());
    let witness = quad::interleave(&s, &m).unwrap();
    let dim = witness.len();
    let mut equations = Vec::new();
    let mut evaluations = Vec::new();
    for i in 0..16 {
        let one = Poly::constant(ring.clone(), 1).rotate(i);
        let mut eq = QuadEq::zero(ring.clone(), dim).unwrap();
        eq.r2 = SparsePolyMat::new(
            ring.clone(),
            dim,
            vec![
                (i as u16, i as u16, one.clone()),
                (i as u16, i as u16 + 1, one.auto()),
            ],
        )
        .unwrap();
        eq.r0 = eq.evaluate(&witness).unwrap().neg();
        equations.push(eq.clone());
        eq.r0 = eq
            .r0
            .add(&Poly::constant(ring.clone(), 1).rotate(i + 1))
            .unwrap();
        evaluations.push(eq);
    }
    let (commitment, opening) = scheme.commit_with_seed(s, m, [122; 32]).unwrap();
    let proof = quad_eval::prove_with_seed(
        &scheme,
        &commitment,
        &opening,
        &equations,
        &evaluations,
        b"two-prime-test",
        [123; 32],
    )
    .unwrap();
    quad_eval::verify(
        &scheme,
        &commitment,
        &equations,
        &evaluations,
        &proof,
        b"two-prime-test",
    )
    .unwrap();
    evaluations[15].r0 = evaluations[15].r0.add(&Poly::constant(ring, 1)).unwrap();
    assert!(
        quad_eval::verify(
            &scheme,
            &commitment,
            &equations,
            &evaluations,
            &proof,
            b"two-prime-test"
        )
        .is_err()
    );
}

#[test]
fn algebra_and_provers_refuse_what_they_cannot_express_or_prove() {
    let scheme = Abdlop::new([1; 32], toy_d64()).unwrap();
    let ring = scheme.ring().clone();
    let s1 = PolyVec::new(
        ring.clone(),
        (0..scheme.bounded_len())
            .map(|i| Poly::constant(ring.clone(), (i % 3) as i128))
            .collect(),
    )
    .unwrap();
    let m = PolyVec::new(
        ring.clone(),
        vec![
            Poly::constant(ring.clone(), 7),
            Poly::constant(ring.clone(), 11),
        ],
    )
    .unwrap();
    let witness = quad::interleave(&s1, &m).unwrap();
    let dim = witness.len();
    let one = Poly::constant(ring.clone(), 1);
    let mut linear = QuadEq::zero(ring.clone(), dim).unwrap();
    linear.r1 = SparsePolyVec::new(ring.clone(), dim, vec![(2, one.clone())]).unwrap();
    let quadratic = linear.product_affine(&linear).unwrap();
    // A product with a quadratic factor would have degree three or four.
    for (a, b) in [(&quadratic, &linear), (&linear, &quadratic)] {
        assert_eq!(
            a.product_affine(b).err(),
            Some(Error::Parameter("expression degree exceeds two"))
        );
    }
    // The trace halves, which needs 2 to be invertible modulo q.
    let even = Ring::new(1 << 20, 64).unwrap();
    assert_eq!(
        QuadEq::zero(even, 2).unwrap().trace().err(),
        Some(Error::Parameter("trace requires invertible two"))
    );
    // One false equation among true ones: every prover refuses the witness.
    let mut true_eq = quadratic.clone();
    true_eq.r0 = true_eq.evaluate(&witness).unwrap().neg();
    let mut false_eq = true_eq.clone();
    false_eq.r0 = false_eq.r0.add(&one).unwrap();
    let (commitment, opening) = scheme
        .commit_with_seed(s1.clone(), m.clone(), [2; 32])
        .unwrap();
    assert_eq!(
        quad_many::prove_with_seed(
            &scheme,
            &commitment,
            &opening,
            &[true_eq.clone(), false_eq.clone()],
            b"refusals",
            [3; 32]
        )
        .err(),
        Some(Error::Witness)
    );
    assert_eq!(
        quad_eval::prove_with_seed(
            &scheme,
            &commitment,
            &opening,
            &[true_eq.clone(), false_eq],
            &[],
            b"refusals",
            [3; 32]
        )
        .err(),
        Some(Error::Witness)
    );
    let mut false_eval = true_eq.clone();
    false_eval.r0 = false_eval.r0.add(&one).unwrap();
    assert_eq!(
        quad_eval::prove_with_seed(
            &scheme,
            &commitment,
            &opening,
            &[true_eq.clone()],
            &[true_eq.clone(), false_eval],
            b"refusals",
            [3; 32]
        )
        .err(),
        Some(Error::Witness)
    );
    // An evaluation equation whose constant coefficient vanishes is accepted even though the
    // whole polynomial does not.
    let mut eval = true_eq.clone();
    eval.r0 = eval.r0.add(&one.rotate(5)).unwrap();
    quad_eval::prove_with_seed(
        &scheme,
        &commitment,
        &opening,
        &[],
        &[eval],
        b"refusals",
        [3; 32],
    )
    .unwrap();
    // An opening of another commitment is refused, by every layer.
    let (other, _) = scheme.commit_with_seed(s1, m, [4; 32]).unwrap();
    assert_ne!(other, commitment);
    assert_eq!(
        quad::prove_with_seed(&scheme, &other, &opening, &true_eq, b"refusals", [3; 32]).err(),
        Some(Error::Witness)
    );
    assert_eq!(
        quad_eval::prove_with_seed(&scheme, &other, &opening, &[], &[], b"refusals", [3; 32]).err(),
        Some(Error::Witness)
    );
}
