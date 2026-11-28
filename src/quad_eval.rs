//! Vanishing constant-coefficient proofs, LNP22 Fig. 8 and §4.4.
//!
//! Both garbage coefficients are zero. On its own the proof runs the transcript
//! `JALI-V02-LNP22-QUAD-EVAL-C1-SHAKE128`, whose instance is the parameters, the public seed,
//! the commitment and both equation lists. Inside the toolbox it continues the toolbox's
//! transcript after the range responses. Either way the garbage commitments are its first
//! prover message, $`\Gamma`$ its first verifier message, $`h`$ its second prover message,
//! and the many-equation proof of the final equations follows (`docs/encodings.md`).
use crate::wipe::Secret;
use crate::{
    Error,
    abdlop::{self, Abdlop, Commitment, Opening},
    codec::{BitWriter, fields},
    linhc,
    math::{Poly, PolyVec, Ring, SparsePolyVec, U256},
    quad::{self, QuadEq, QuadProof},
    quad_many,
    rand::{AesPrg, derive_key, domain, secret, take_seed, uniform_poly},
    transcript::{Protocol, Transcript},
};
use std::{borrow::Cow, sync::Arc};

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

/// The transcript of the proof on its own, after its instance: the scheme, the commitment's
/// serialization, and both equation lists (`quad_many::absorb_equations`).
fn transcript(
    scheme: &Abdlop,
    commitment: &Commitment,
    eqs: &[QuadEq],
    evals: &[QuadEq],
    context: &[u8],
) -> Result<Transcript, Error> {
    let mut transcript = Transcript::new(Protocol::Evaluation, context)?;
    scheme.absorb_scheme(&mut transcript)?;
    transcript.absorb(&scheme.commitment_bytes(commitment)?);
    quad_many::absorb_equations(&mut transcript, eqs)?;
    quad_many::absorb_equations(&mut transcript, evals)?;
    Ok(transcript)
}
/// The serialization of a vector modulo $`q`$: the garbage commitments, or $`h`$.
fn vector_bytes(v: &PolyVec) -> Result<Vec<u8>, Error> {
    let mut w = BitWriter::new();
    fields::write_below(&mut w, v, &v.ring().modulus())?;
    w.into_bytes()
}
/// Squeeze $`\Gamma`$, $`\lambda`$ rows of one scalar modulo $`q`$ per evaluation equation,
/// row-major, each by `DecodeUint`.
fn squeeze_gamma(
    scheme: &Abdlop,
    transcript: &mut Transcript,
    evals: usize,
) -> Result<Vec<U256>, Error> {
    let count = scheme
        .checked
        .lambda
        .checked_mul(evals)
        .ok_or(Error::Dimension)?;
    transcript.uints(&scheme.ring().modulus(), count)
}
/// The evaluation equations of a proof, in order: a list, or the toolbox's projection rows,
/// which it adds to the combination already combined (`tbox::ProjectionRows`).
pub(crate) trait EvalRows {
    /// The number of equations.
    fn count(&self) -> usize;
    /// Add the equations, in order, to `combination`. The prover passes its witness, and the
    /// equations are checked against it before they are added.
    fn add_to(
        &self,
        combination: &mut Combination<'_>,
        witness: Option<&PolyVec>,
    ) -> Result<(), Error>;
}
impl EvalRows for [QuadEq] {
    fn count(&self) -> usize {
        self.len()
    }
    fn add_to(
        &self,
        combination: &mut Combination<'_>,
        witness: Option<&PolyVec>,
    ) -> Result<(), Error> {
        if let Some(witness) = witness {
            check_constants(self, witness, &mut quad::PairProducts::empty())?;
        }
        combination.add(self)
    }
}
/// The traces of the $`\lambda`$ combinations $`\sum_j\Gamma_{rj}f_j`$ of the evaluation
/// equations $`f_j`$, for the rows $`r`$ of `gamma`, added up as the equations arrive: each
/// row takes the additions of one pass over the whole list, in its order, so the rows do not
/// depend on the chunks.
pub(crate) struct Combination<'a> {
    ring: &'a Arc<Ring>,
    gamma: &'a [U256],
    count: usize,
    added: usize,
    input_dim: usize,
    rows: Vec<QuadEq>,
}
impl<'a> Combination<'a> {
    pub(crate) fn new(
        scheme: &'a Abdlop,
        gamma: &'a [U256],
        count: usize,
        input_dim: usize,
    ) -> Result<Self, Error> {
        if gamma.len() != scheme.checked.lambda * count {
            return Err(Error::Dimension);
        }
        let ring = scheme.ring();
        let rows = (0..scheme.checked.lambda)
            .map(|_| QuadEq::zero(ring.clone(), input_dim))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            ring,
            gamma,
            count,
            added: 0,
            input_dim,
            rows,
        })
    }
    /// Each row's weights of the next `count` equations.
    pub(crate) fn next_weights(&self, count: usize) -> Result<Vec<&'a [U256]>, Error> {
        let (start, end) = (
            self.added,
            self.added.checked_add(count).ok_or(Error::Dimension)?,
        );
        if end > self.count {
            return Err(Error::Dimension);
        }
        let (gamma, total) = (self.gamma, self.count);
        Ok((0..self.rows.len())
            .map(|row| &gamma[row * total + start..row * total + end])
            .collect())
    }
    /// Add to each row `rows[row]`, which the caller formed as the combination of the next
    /// `count` equations with that row's weights ([`Combination::next_weights`]).
    pub(crate) fn add_combined(&mut self, rows: Vec<QuadEq>, count: usize) -> Result<(), Error> {
        let end = self.added.checked_add(count).ok_or(Error::Dimension)?;
        if rows.len() != self.rows.len() || end > self.count {
            return Err(Error::Dimension);
        }
        for eq in &rows {
            eq.check(self.ring, self.input_dim)?;
        }
        // The rows are computed in parallel (`parallel`).
        let pairs: Vec<(&mut QuadEq, QuadEq)> = self.rows.iter_mut().zip(rows).collect();
        crate::par::try_for_each_min(pairs, 2, |(row, eq)| row.add_assign(&eq))?;
        self.added = end;
        Ok(())
    }
    /// Add the next equations.
    pub(crate) fn add(&mut self, chunk: &[QuadEq]) -> Result<(), Error> {
        for eq in chunk {
            eq.check(self.ring, self.input_dim)?;
        }
        let (start, end) = (self.added, self.added + chunk.len());
        if end > self.count {
            return Err(Error::Dimension);
        }
        let (gamma, count) = (self.gamma, self.count);
        // The rows are computed in parallel (`parallel`).
        let rows: Vec<(usize, &mut QuadEq)> = self.rows.iter_mut().enumerate().collect();
        crate::par::try_for_each_min(rows, 2, |(row, equation)| {
            let weights = &gamma[row * count + start..row * count + end];
            // In place: the keys and values of `equation.add(&eq.scale(&constant(weight)))`.
            for (eq, weight) in chunk.iter().zip(weights) {
                equation.add_scaled_scalar_assign(eq, weight)?;
            }
            Ok(())
        })?;
        self.added = end;
        Ok(())
    }
    /// The traces of the rows, once every equation was added.
    pub(crate) fn finish(mut self) -> Result<Vec<QuadEq>, Error> {
        if self.added != self.count {
            return Err(Error::Dimension);
        }
        let ring = self.ring;
        let rows: Vec<&mut QuadEq> = self.rows.iter_mut().collect();
        // In parallel (`parallel`).
        crate::par::try_for_each_min(rows, 2, |row| {
            let equation = std::mem::replace(row, QuadEq::zero(ring.clone(), 0)?);
            *row = equation.into_trace()?;
            Ok(())
        })?;
        Ok(self.rows)
    }
}
/// [`Combination`] of a list.
#[cfg(test)]
fn combined(
    scheme: &Abdlop,
    gamma: &[U256],
    evals: &[QuadEq],
    input_dim: usize,
) -> Result<Vec<QuadEq>, Error> {
    let mut combination = Combination::new(scheme, gamma, evals.len(), input_dim)?;
    combination.add(evals)?;
    combination.finish()
}
/// The honest prover's check that `witness` satisfies each of `eqs` at the constant
/// coefficient, with the witness products of `pairs`, extended by those of `eqs`.
pub(crate) fn check_constants(
    eqs: &[QuadEq],
    witness: &PolyVec,
    pairs: &mut quad::PairProducts,
) -> Result<(), Error> {
    let covered = pairs.add_for(eqs, witness)?;
    let pairs = &*pairs;
    // Evaluated in parallel (`parallel`), and the first failure in order is returned.
    crate::par::try_map(eqs.len(), |i| {
        let constant = match covered {
            true => eqs[i].evaluate_constant_with(witness, pairs)?,
            false => eqs[i].evaluate_constant(witness)?,
        };
        match constant.is_zero_vartime() {
            true => Ok(()),
            false => Err(Error::Witness),
        }
    })?;
    Ok(())
}
fn final_equations(
    scheme: &Abdlop,
    eqs: Cow<'_, [QuadEq]>,
    combined: Vec<QuadEq>,
    h: &PolyVec,
    input_dim: usize,
) -> Result<Vec<QuadEq>, Error> {
    let dim = 2 * (scheme.bounded_len() + scheme.message_len());
    let ring = scheme.ring().clone();
    // Equations the caller hands over are resized in place, with the keys and values that
    // `QuadEq::resized` copies.
    let mut out = match eqs {
        Cow::Owned(eqs) => eqs
            .into_iter()
            .map(|mut eq| {
                eq.check(&ring, input_dim)?;
                eq.resize(dim)?;
                Ok(eq)
            })
            .collect::<Result<Vec<_>, Error>>()?,
        Cow::Borrowed(eqs) => eqs
            .iter()
            .map(|eq| {
                eq.check(&ring, input_dim)?;
                eq.resized(dim)
            })
            .collect::<Result<Vec<_>, Error>>()?,
    };
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
    seed: &Secret<[u8; 32]>,
) -> Result<EvalProof, Error> {
    let transcript = transcript(scheme, commitment, eqs, evals, context)?;
    Ok(prove_in(
        scheme,
        commitment,
        opening,
        Cow::Borrowed(eqs),
        evals,
        transcript,
        seed,
        None,
    )?
    .0)
}
/// The proof in `transcript`, positioned before the garbage commitments, with the extractable
/// commitment beside the final opening when `ext` is given, returning the accepted attempt's
/// LinHC opening too. The garbage polynomials' key binds the transcript through
/// `Transcript::binding`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_in<E: EvalRows + ?Sized>(
    scheme: &Abdlop,
    commitment: &Commitment,
    opening: &Opening,
    eqs: Cow<'_, [QuadEq]>,
    evals: &E,
    mut transcript: Transcript,
    seed: &Secret<[u8; 32]>,
    ext: Option<&linhc::ProverExt<'_>>,
) -> Result<(EvalProof, Option<linhc::LinhcOpening>), Error> {
    let witness = quad::interleave(&opening.s1, &opening.m)?;
    // "/v2" since the duplex-sponge transcript of version 0.2, which the binding comes from.
    let key = derive_key(
        b"quad-eval/garbage/v2",
        seed,
        &[
            &transcript.binding(),
            &abdlop::secret_bytes(&opening.s1),
            &abdlop::secret_bytes(&opening.m),
            &abdlop::secret_bytes(&opening.s2),
        ],
    );
    let ring = scheme.ring();
    let mut garbage = Vec::new();
    for i in 0..scheme.checked.lambda / 2 {
        let mut g = uniform_poly(
            &mut AesPrg::new(&key, domain(secret::GARBAGE, i as u32)),
            ring,
        )?;
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
    let garbage_commitments = abdlop::part(
        &full_commitment.t_b,
        scheme.message_len(),
        extended.message_len(),
    )?;
    transcript.absorb(&vector_bytes(&garbage_commitments)?);
    let gamma = squeeze_gamma(&extended, &mut transcript, evals.count())?;
    // The evaluation equations are checked against the witness as they are added to the
    // combination. Nothing is returned before every check passed.
    let mut combination = Combination::new(&extended, &gamma, evals.count(), witness.len())?;
    evals.add_to(&mut combination, Some(&witness))?;
    let combined = combination.finish()?;
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
    transcript.absorb(&vector_bytes(&h)?);
    let (quadratic, linhc_opening) = quad_many::prove_in(
        &extended,
        &full_commitment,
        &full_opening,
        &equations,
        transcript,
        seed,
        ext,
    )?;
    Ok((
        EvalProof {
            garbage_commitments,
            h,
            quadratic,
        },
        linhc_opening,
    ))
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
        !h.coefficient(0).is_zero_vartime() || !h.coefficient(ring.degree() / 2).is_zero_vartime()
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
    let transcript = transcript(scheme, commitment, eqs, evals, context)?;
    verify_in(
        scheme,
        commitment,
        Cow::Borrowed(eqs),
        evals,
        proof,
        transcript,
        None,
    )
}
/// [`verify`] in `transcript`, as [`prove_in`] proves, with the extractable commitment's checks
/// when `ext` is given.
pub(crate) fn verify_in<E: EvalRows + ?Sized>(
    scheme: &Abdlop,
    commitment: &Commitment,
    eqs: Cow<'_, [QuadEq]>,
    evals: &E,
    proof: &EvalProof,
    mut transcript: Transcript,
    ext: Option<&linhc::VerifierExt<'_>>,
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
    transcript.absorb(&vector_bytes(&proof.garbage_commitments)?);
    let gamma = squeeze_gamma(&extended, &mut transcript, evals.count())?;
    let mut combination = Combination::new(&extended, &gamma, evals.count(), dim)?;
    evals.add_to(&mut combination, None)?;
    let combined = combination.finish()?;
    let equations = final_equations(&extended, eqs, combined, &proof.h, dim)?;
    transcript.absorb(&vector_bytes(&proof.h)?);
    quad_many::verify_in(
        &extended,
        &full_commitment,
        &equations,
        &proof.quadratic,
        transcript,
        ext,
    )
}
