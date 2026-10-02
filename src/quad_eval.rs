//! Vanishing constant-coefficient proofs, LNP22 Fig. 8 and §4.4.
//! Both garbage coefficients are zero; independent Gamma rows follow the garbage
//! commitments, and h is absorbed before ring-valued folding challenges are derived.
use crate::{
    Error,
    abdlop::{self, Abdlop, Commitment, Opening},
    codec::BitWriter,
    math::{Poly, PolyVec, SparsePolyVec},
    quad::{self, QuadEq, QuadProof},
    quad_many,
    rand::{AesPrg, derive_key, domain, secret, take_seed, uniform_ring},
    transcript::Transcript,
};
use zeroize::Zeroizing;

#[cfg(test)]
mod soundness_tests;
#[cfg(test)]
mod verifier_check_tests;

/// Evaluation proof with lambda/2 garbage commitments and responses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalProof {
    /// Additional BDLOP commitment rows for the garbage polynomials.
    pub garbage_commitments: PolyVec,
    /// Responses with zero coefficients at 0 and d/2.
    pub h: PolyVec,
    /// Final quadratic proof after folding.
    pub quadratic: QuadProof,
}

fn setup_prefix(
    scheme: &Abdlop,
    commitment: &Commitment,
    eqs: &[QuadEq],
    evals: &[QuadEq],
    context: &[u8],
) -> Result<Transcript, Error> {
    let mut prefix = scheme.prefix(commitment, context)?;
    quad_many::absorb_equations(&mut prefix, b"quadratic-inputs", eqs)?;
    quad_many::absorb_equations(&mut prefix, b"evaluation-inputs", evals)?;
    Ok(prefix)
}
fn combined(
    scheme: &Abdlop,
    prefix: &Transcript,
    evals: &[QuadEq],
    input_dim: usize,
) -> Result<Vec<QuadEq>, Error> {
    let ring = scheme.ring();
    let seed = prefix.challenge_seed(b"Gamma");
    // Each row reads its own stream: the rows are computed in parallel (`parallel`) and kept
    // in order.
    crate::par::try_map(scheme.checked.lambda, |row| {
        let weights = uniform_ring(
            &mut AesPrg::new(&seed, domain(0x47414d4d, row as u32)),
            ring,
            evals.len(),
        )?;
        // In place: the keys and values of `equation.add(&eq.scale(&constant(weight)))`.
        let mut equation = QuadEq::zero(ring.clone(), input_dim)?;
        for (eq, weight) in evals.iter().zip(weights) {
            eq.check(ring, input_dim)?;
            equation.add_scaled_scalar_assign(eq, &weight)?;
        }
        equation.into_trace()
    })
}
fn final_equations(
    scheme: &Abdlop,
    eqs: &[QuadEq],
    combined: Vec<QuadEq>,
    h: &PolyVec,
    input_dim: usize,
) -> Result<Vec<QuadEq>, Error> {
    let dim = 2 * (scheme.bounded_len() + scheme.message_len());
    let ring = scheme.ring().clone();
    let mut out = eqs
        .iter()
        .map(|eq| {
            eq.check(&ring, input_dim)?;
            eq.resized(dim)
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let rotation = Poly::constant(ring.clone(), 1).rotate((ring.degree() / 2) as i64);
    let mut combined = combined.into_iter();
    for i in 0..scheme.checked.lambda / 2 {
        // In place: combined[2i] + combined[2i+1] X^{d/2}, resized, plus the garbage term.
        let (Some(mut equation), Some(odd)) = (combined.next(), combined.next()) else {
            return Err(Error::Dimension);
        };
        equation.add_scaled_assign(&odd, &rotation)?;
        equation.resize(dim)?;
        let mut garbage = QuadEq::zero(ring.clone(), dim)?;
        garbage.r1 = SparsePolyVec::new(
            ring.clone(),
            dim,
            vec![(
                u16::try_from(input_dim + 2 * i).map_err(|_| Error::Dimension)?,
                Poly::constant(ring.clone(), 1),
            )],
        )?;
        garbage.r0 = h.entries()[i].neg();
        equation.add_assign(&garbage)?;
        out.push(equation);
    }
    Ok(out)
}
fn bind_h(prefix: &mut Transcript, h: &PolyVec) -> Result<(), Error> {
    let mut w = BitWriter::new();
    abdlop::encode_vec(&mut w, h, &h.ring().modulus())?;
    prefix.absorb(b"h-before-mu", &w.finish());
    Ok(())
}

/// Prove full ring equations and vanishing constant coefficients simultaneously.
///
/// The seed must be secret and uniformly random, and it may be reused, also for the commitment
/// being proven. The garbage polynomials are read under a key derived from the seed, the
/// transcript before they are committed (parameters, public seed, context, commitment and both
/// equation lists) and the opening, and the masks under a key bound to the later transcript:
/// identical calls return identical proofs, and calls that differ in any input read
/// independent randomness (assuming SHAKE128 and AES-256 behave as pseudorandom functions).
///
/// With a reused seed, zero-knowledge holds only relative to the equality pattern of the
/// inputs: identical inputs give identical proofs. Uses that need multi-theorem
/// zero-knowledge, and settings where faults can be injected (a fault that changes a challenge
/// but not the key can reveal the witness), need a fresh seed per call, drawn from a
/// `rand_core::CryptoRng`.
pub fn prove_with_seed(
    scheme: &Abdlop,
    commitment: &Commitment,
    opening: &Opening,
    eqs: &[QuadEq],
    evals: &[QuadEq],
    context: &[u8],
    mut seed: [u8; 32],
) -> Result<EvalProof, Error> {
    prove_seeded(
        scheme,
        commitment,
        opening,
        eqs,
        evals,
        context,
        &take_seed(&mut seed),
    )
}
pub(crate) fn prove_seeded(
    scheme: &Abdlop,
    commitment: &Commitment,
    opening: &Opening,
    eqs: &[QuadEq],
    evals: &[QuadEq],
    context: &[u8],
    seed: &Zeroizing<[u8; 32]>,
) -> Result<EvalProof, Error> {
    let witness = quad::interleave(&opening.s1, &opening.m)?;
    // Evaluated in parallel (`parallel`); the first failure in order is returned.
    crate::par::try_map(evals.len(), |i| {
        match evals[i].evaluate(&witness)?.coefficients()[0].is_zero_vartime() {
            true => Ok(()),
            false => Err(Error::Witness),
        }
    })?;
    let key = derive_key(
        b"quad-eval/garbage",
        seed,
        &[
            &setup_prefix(scheme, commitment, eqs, evals, context)?.digest(),
            &abdlop::secret_bytes(&opening.s1),
            &abdlop::secret_bytes(&opening.m),
            &abdlop::secret_bytes(&opening.s2),
        ],
    );
    let ring = scheme.ring();
    let mut garbage = Vec::new();
    for i in 0..scheme.checked.lambda / 2 {
        let values = uniform_ring(
            &mut AesPrg::new(&key, domain(secret::GARBAGE, i as u32)),
            ring,
            ring.degree(),
        )?;
        let mut g = Poly::from_canonical(ring.clone(), values);
        g.set_coefficient(0, 0)?;
        g.set_coefficient(ring.degree() / 2, 0)?;
        garbage.push(g);
    }
    let extended = scheme.extend_messages(garbage.len())?;
    let mut messages = opening.m.entries().to_vec();
    messages.extend(garbage.iter().cloned());
    let (full_commitment, full_opening) = extended.commit_with_randomness(
        opening.s1.clone(),
        PolyVec::new(ring.clone(), messages)?,
        opening.s2.clone(),
    )?;
    if full_commitment.t_a != commitment.t_a
        || full_commitment.t_b.entries()[..scheme.message_len()] != *commitment.t_b.entries()
    {
        return Err(Error::Witness);
    }
    let mut prefix = setup_prefix(&extended, &full_commitment, eqs, evals, context)?;
    let combined = combined(&extended, &prefix, evals, witness.len())?;
    // The rows' values, in parallel (`parallel`) and in order.
    let values = crate::par::try_map(combined.len(), |k| combined[k].evaluate(&witness))?;
    let h = PolyVec::new(
        ring.clone(),
        garbage
            .iter()
            .enumerate()
            .map(|(i, g)| {
                g.add(&values[2 * i])?
                    .add(&values[2 * i + 1].rotate((ring.degree() / 2) as i64))
            })
            .collect::<Result<_, _>>()?,
    )?;
    let equations = final_equations(&extended, eqs, combined, &h, witness.len())?;
    bind_h(&mut prefix, &h)?;
    let quadratic = quad_many::prove_seeded(
        &extended,
        &full_commitment,
        &full_opening,
        &equations,
        &prefix.digest(),
        seed,
    )?;
    Ok(EvalProof {
        garbage_commitments: abdlop::part(
            &full_commitment.t_b,
            scheme.message_len(),
            extended.message_len(),
        )?,
        h,
        quadratic,
    })
}

/// The verifier's checks of the garbage rows and responses: lambda/2 of each, in the proof
/// ring, and every response zero at coefficients 0 and d/2.
fn check_responses(scheme: &Abdlop, proof: &EvalProof) -> Result<(), Error> {
    let count = scheme.checked.lambda / 2;
    let ring = scheme.ring();
    if proof.h.len() != count || proof.garbage_commitments.len() != count {
        return Err(Error::Dimension);
    }
    if proof.h.ring() != ring || proof.garbage_commitments.ring() != ring {
        return Err(Error::RingMismatch);
    }
    if proof.h.entries().iter().any(|h| {
        !h.coefficients()[0].is_zero_vartime()
            || !h.coefficients()[ring.degree() / 2].is_zero_vartime()
    }) {
        return Err(Error::InvalidProof);
    }
    Ok(())
}

/// Verify both reserved coefficients, then reconstruct Gamma, equations and Fiat–Shamir order.
pub fn verify(
    scheme: &Abdlop,
    commitment: &Commitment,
    eqs: &[QuadEq],
    evals: &[QuadEq],
    proof: &EvalProof,
    context: &[u8],
) -> Result<(), Error> {
    check_responses(scheme, proof)?;
    let count = scheme.checked.lambda / 2;
    let ring = scheme.ring();
    let extended = scheme.extend_messages(count)?;
    if commitment.t_b.len() != scheme.message_len() {
        return Err(Error::Dimension);
    }
    let mut tb = commitment.t_b.entries().to_vec();
    tb.extend(proof.garbage_commitments.entries().iter().cloned());
    let full_commitment = Commitment {
        t_a: commitment.t_a.clone(),
        t_b: PolyVec::new(ring.clone(), tb)?,
    };
    let dim = 2 * (scheme.bounded_len() + scheme.message_len());
    let mut prefix = setup_prefix(&extended, &full_commitment, eqs, evals, context)?;
    let combined = combined(&extended, &prefix, evals, dim)?;
    let equations = final_equations(&extended, eqs, combined, &proof.h, dim)?;
    bind_h(&mut prefix, &proof.h)?;
    quad_many::verify(
        &extended,
        &full_commitment,
        &equations,
        &proof.quadratic,
        &prefix.digest(),
    )
}
