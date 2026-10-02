//! Deliberately weakened transcript experiments.
use super::*;
use crate::params::toy_d64;

fn weights(scheme: &Abdlop, seed: &[u8; 32], count: usize) -> Vec<Poly> {
    (0..count)
        .map(|i| {
            Poly::from_u256(
                scheme.ring().clone(),
                uniform_ring(
                    &mut AesPrg::new(seed, domain(0x4d550000, i as u32 + 1)),
                    scheme.ring(),
                    scheme.ring().degree(),
                )
                .unwrap(),
            )
            .unwrap()
        })
        .collect()
}

// Solve sum_i mu_i*h_i = target with h_i[0] = h_i[d/2] = 0. This is the
// adversary's post-challenge choice, using Gaussian elimination over the prime field.
fn adaptive_h(mu: &[Poly], target: &Poly) -> PolyVec {
    let ring = target.ring();
    let d = ring.degree();
    let q = ring.modulus_i128();
    let free: Vec<_> = (0..mu.len())
        .flat_map(|i| (1..d).filter(move |j| *j != d / 2).map(move |j| (i, j)))
        .collect();
    let columns: Vec<_> = free.iter().map(|(i, j)| mu[*i].rotate(*j as i64)).collect();
    let mut matrix: Vec<Vec<i128>> = (0..d)
        .map(|row| {
            columns
                .iter()
                .map(|p| crate::math::int::low_u128(&p.coefficients()[row]) as i128)
                .chain(std::iter::once(
                    crate::math::int::low_u128(&target.coefficients()[row]) as i128,
                ))
                .collect()
        })
        .collect();
    let power = |mut x: i128, mut e: i128| {
        let mut out = 1;
        while e > 0 {
            if e & 1 != 0 {
                out = out * x % q;
            }
            x = x * x % q;
            e >>= 1;
        }
        out
    };
    let mut pivots = Vec::new();
    for col in 0..free.len() {
        let row = pivots.len();
        let Some(pivot) = (row..d).find(|i| matrix[*i][col] != 0) else {
            continue;
        };
        matrix.swap(row, pivot);
        let inverse = power(matrix[row][col], q - 2);
        for x in &mut matrix[row][col..] {
            *x = *x * inverse % q;
        }
        let pivot_values = matrix[row][col..].to_vec();
        for (i, values) in matrix.iter_mut().enumerate() {
            if i == row {
                continue;
            }
            let factor = values[col];
            for (x, pivot) in values[col..].iter_mut().zip(&pivot_values) {
                *x = (*x - factor * pivot).rem_euclid(q);
            }
        }
        pivots.push(col);
        if pivots.len() == d {
            break;
        }
    }
    assert_eq!(
        pivots.len(),
        d,
        "chosen test weights must have full row rank"
    );
    let mut h = vec![Poly::zero(ring.clone()); mu.len()];
    for (row, col) in pivots.into_iter().enumerate() {
        let (i, j) = free[col];
        h[i].set_coefficient(j, matrix[row][free.len()]).unwrap();
    }
    PolyVec::new(ring.clone(), h).unwrap()
}

#[test]
fn adaptive_h_after_mu_can_forge_a_weakened_round_but_fails_default_verification() {
    let scheme = Abdlop::new([70; 32], toy_d64()).unwrap();
    let ring = scheme.ring().clone();
    let (commitment, opening) = scheme
        .commit_with_seed(
            PolyVec::zero(ring.clone(), scheme.bounded_len()),
            PolyVec::zero(ring.clone(), scheme.message_len()),
            [71; 32],
        )
        .unwrap();
    let dim = 2 * (scheme.bounded_len() + scheme.message_len());
    let mut false_eq = QuadEq::zero(ring.clone(), dim).unwrap();
    false_eq.r0 = Poly::constant(ring.clone(), 1);
    let eqs = vec![false_eq]; // An impossible public statement: 1 = 0.
    let extended = scheme.extend_messages(scheme.checked.lambda / 2).unwrap();
    let (full_commitment, full_opening) = extended
        .commit_with_randomness(
            opening.s1.clone(),
            PolyVec::zero(ring.clone(), extended.message_len()),
            opening.s2.clone(),
        )
        .unwrap();
    let mut prefix =
        setup_prefix(&extended, &full_commitment, &eqs, &[], b"adversarial-test").unwrap();
    let gamma = combined(&extended, &prefix, &[], dim).unwrap();
    let old_seed = prefix.challenge_seed(b"mu");
    let mu = weights(&extended, &old_seed, 1 + extended.checked.lambda / 2);
    let h = adaptive_h(&mu[1..], &mu[0]);
    assert!(
        h.entries()
            .iter()
            .all(|h| h.coefficient_i128(0) == Ok(0)
                && h.coefficient_i128(ring.degree() / 2) == Ok(0))
    );
    let equations = final_equations(&extended, &eqs, gamma.clone(), &h, dim).unwrap();
    let mut folded = QuadEq::zero(
        ring.clone(),
        2 * (extended.bounded_len() + extended.message_len()),
    )
    .unwrap();
    for (eq, mu) in equations.iter().zip(&mu) {
        folded = folded.add(&eq.scale(mu).unwrap()).unwrap();
    }
    assert!(
        folded
            .evaluate(&quad::interleave(&full_opening.s1, &full_opening.m).unwrap())
            .unwrap()
            .is_zero()
    );
    let quadratic = quad::prove_with_seed(
        &extended,
        &full_commitment,
        &full_opening,
        &folded,
        &prefix.digest(),
        [72; 32],
    )
    .unwrap();
    quad::verify(
        &extended,
        &full_commitment,
        &folded,
        &quadratic,
        &prefix.digest(),
    )
    .unwrap();
    bind_h(&mut prefix, &h).unwrap();
    assert_ne!(old_seed, prefix.challenge_seed(b"mu"));
    let proof = EvalProof {
        garbage_commitments: abdlop::part(
            &full_commitment.t_b,
            scheme.message_len(),
            extended.message_len(),
        )
        .unwrap(),
        h,
        quadratic,
    };
    assert!(verify(&scheme, &commitment, &eqs, &[], &proof, b"adversarial-test").is_err());
}

#[test]
fn garbage_chosen_after_gamma_absorbs_a_false_row_but_fails_default_verification() {
    let scheme = Abdlop::new([80; 32], toy_d64()).unwrap();
    let ring = scheme.ring().clone();
    let (commitment, opening) = scheme
        .commit_with_seed(
            PolyVec::zero(ring.clone(), scheme.bounded_len()),
            PolyVec::zero(ring.clone(), scheme.message_len()),
            [81; 32],
        )
        .unwrap();
    let witness = quad::interleave(&opening.s1, &opening.m).unwrap();
    let dim = witness.len();
    let mut false_row = QuadEq::zero(ring.clone(), dim).unwrap();
    false_row.r0 = Poly::constant(ring.clone(), 1);
    let evals = [false_row];
    // Deliberately sample Gamma before committing to g. The reserved coefficients of
    // g can now cancel an arbitrary false evaluation row, while transmitted h is zero.
    let mut prefix = setup_prefix(&scheme, &commitment, &[], &evals, b"early-Gamma-test").unwrap();
    let gamma = combined(&scheme, &prefix, &evals, dim).unwrap();
    let g: Vec<_> = (0..scheme.checked.lambda / 2)
        .map(|i| {
            gamma[2 * i]
                .evaluate(&witness)
                .unwrap()
                .add(
                    &gamma[2 * i + 1]
                        .evaluate(&witness)
                        .unwrap()
                        .rotate((ring.degree() / 2) as i64),
                )
                .unwrap()
                .neg()
        })
        .collect();
    assert!(g.iter().any(|g| g.coefficient_i128(0) != Ok(0)));
    let extended = scheme.extend_messages(g.len()).unwrap();
    let mut messages = opening.m.entries().to_vec();
    messages.extend(g);
    let (full_commitment, full_opening) = extended
        .commit_with_randomness(
            opening.s1.clone(),
            PolyVec::new(ring.clone(), messages).unwrap(),
            opening.s2.clone(),
        )
        .unwrap();
    let h = PolyVec::zero(ring, scheme.checked.lambda / 2);
    let equations = final_equations(&extended, &[], gamma.clone(), &h, dim).unwrap();
    bind_h(&mut prefix, &h).unwrap();
    let quadratic = quad_many::prove_with_seed(
        &extended,
        &full_commitment,
        &full_opening,
        &equations,
        &prefix.digest(),
        [82; 32],
    )
    .unwrap();
    quad_many::verify(
        &extended,
        &full_commitment,
        &equations,
        &quadratic,
        &prefix.digest(),
    )
    .unwrap();
    let proof = EvalProof {
        garbage_commitments: abdlop::part(
            &full_commitment.t_b,
            scheme.message_len(),
            extended.message_len(),
        )
        .unwrap(),
        h,
        quadratic,
    };
    assert!(
        verify(
            &scheme,
            &commitment,
            &[],
            &evals,
            &proof,
            b"early-Gamma-test"
        )
        .is_err()
    );
}
