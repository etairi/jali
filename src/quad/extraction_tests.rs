//! Special soundness of the quadratic proof (LNP22 Fig. 6), with injected challenges. With
//! masks $`y`$ and the witness $`s`$, the verifier's reconstruction of the prover's $`v`$ is
//! $`v+c^2E-c\,\delta`$, where $`E`$ is the equation's value at $`s`$ and $`\delta`$ the change a
//! prover makes to $`t`$. For a false equation ($`E\ne0`$) a prover that guesses $`c^*`$ and
//! sends $`t+c^*E`$ passes at $`c^*`$ and fails at every other $`c`$ with $`c(c-c^*)E\ne0`$;
//! for a true one, two accepting challenges of one first message give the witness back.
use super::*;

/// The equation of `tests/quadratic.rs`, whose value at the committed witness is `offset`.
fn fixture(offset: i128) -> (Abdlop, Commitment, Opening, QuadEq) {
    let scheme = Abdlop::new([1; 32], crate::params::toy_d64()).unwrap();
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
    let witness = interleave(&s1, &m).unwrap();
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
        r1: SparsePolyVec::new(ring.clone(), dim, vec![(0, one.clone()), (24, one)]).unwrap(),
        r0: Poly::zero(ring.clone()),
    };
    equation.r0 = equation
        .evaluate(&witness)
        .unwrap()
        .neg()
        .add(&Poly::constant(ring.clone(), offset))
        .unwrap();
    assert_eq!(
        equation.evaluate(&witness).unwrap(),
        Poly::constant(ring, offset)
    );
    let (commitment, opening) = scheme.commit_with_seed(s1, m, [2; 32]).unwrap();
    (scheme, commitment, opening, equation)
}

/// The first message `(t, v, w1)` of the accepted attempt.
type First = (Poly, Poly, PolyVec);

/// Run the prover (without its witness check) with challenge `c` injected.
fn fork(
    scheme: &Abdlop,
    commitment: &Commitment,
    opening: &Opening,
    equation: &QuadEq,
    seed: u8,
    c: &Poly,
) -> (QuadProof, First) {
    let binding = prefix(scheme, commitment, equation, b"forks")
        .unwrap()
        .digest();
    prove_core(
        scheme,
        commitment,
        opening,
        equation,
        &binding,
        &Zeroizing::new([seed; 32]),
        |t, v, w1| Ok((c.clone(), (t.clone(), v.clone(), w1.clone()))),
    )
    .unwrap()
}

/// The interactive verifier: it receives `first`, sends `c` and checks the response.
fn accepts(
    scheme: &Abdlop,
    commitment: &Commitment,
    equation: &QuadEq,
    proof: &QuadProof,
    first: &First,
    c: &Poly,
) -> bool {
    verify_core(scheme, commitment, equation, proof, |v, w1| {
        if proof.t == first.0 && *v == first.1 && *w1 == first.2 {
            Ok(c.clone())
        } else {
            Err(Error::InvalidProof)
        }
    })
    .is_ok()
}

/// Two forks at challenges 1 and 2 that accepted at the same attempt, so share the masks.
fn same_attempt_forks(
    scheme: &Abdlop,
    commitment: &Commitment,
    opening: &Opening,
    equation: &QuadEq,
) -> [(Poly, QuadProof, First); 2] {
    let ring = scheme.ring().clone();
    let [c1, c2] = [1, 2].map(|c| Poly::constant(ring.clone(), c));
    for seed in 0..64 {
        let (p1, f1) = fork(scheme, commitment, opening, equation, seed, &c1);
        let (p2, f2) = fork(scheme, commitment, opening, equation, seed, &c2);
        if f1.2 == f2.2 {
            return [(c1, p1, f1), (c2, p2, f2)];
        }
    }
    panic!("no seed among 64 accepts both challenges at the same attempt");
}

#[test]
fn two_accepting_challenges_of_a_true_equation_give_the_witness_back() {
    let (scheme, commitment, opening, equation) = fixture(0);
    let [(c1, p1, f1), (c2, p2, f2)] =
        same_attempt_forks(&scheme, &commitment, &opening, &equation);
    assert!(accepts(&scheme, &commitment, &equation, &p1, &f1, &c1));
    assert!(accepts(&scheme, &commitment, &equation, &p2, &f2, &c2));
    // c2 - c1 = 1: z(c2) - z(c1) = s.
    let s1 = abdlop::sub(&p2.opening.z1, &p1.opening.z1).unwrap();
    let s21 = abdlop::sub(&p2.opening.z21, &p1.opening.z21).unwrap();
    let m = abdlop::sub(&commitment.t_b, &scheme.b.mul(&s21).unwrap()).unwrap();
    assert_eq!(s1, opening.s1);
    assert_eq!(m, opening.m);
    assert!(
        equation
            .evaluate(&interleave(&s1, &m).unwrap())
            .unwrap()
            .is_zero()
    );
}

#[test]
fn a_false_equation_passes_only_at_the_challenge_a_cheating_prover_guessed() {
    let (scheme, commitment, opening, equation) = fixture(5);
    let ring = scheme.ring().clone();
    let e = Poly::constant(ring, 5);
    let forks = same_attempt_forks(&scheme, &commitment, &opening, &equation);
    // The honest algorithm on a false witness fails at both challenges.
    for (c, proof, first) in &forks {
        assert!(!accepts(&scheme, &commitment, &equation, proof, first, c));
    }
    // One first message for both forks: t and v do not depend on the challenge.
    let [(_, _, f1), (_, _, f2)] = &forks;
    assert_eq!((&f1.0, &f1.1), (&f2.0, &f2.1));
    for (guess, _, _) in &forks {
        let t = f1.0.add(&guess.mul(&e).unwrap()).unwrap();
        for (c, proof, first) in &forks {
            let mut cheat = proof.clone();
            cheat.t = t.clone();
            let first = (t.clone(), first.1.clone(), first.2.clone());
            assert_eq!(
                accepts(&scheme, &commitment, &equation, &cheat, &first, c),
                c == guess
            );
        }
    }
}
