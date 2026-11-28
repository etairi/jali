//! Independent ring-valued Schwartz–Zippel folding, LNP22 Fig. 7.
//!
//! On its own the proof runs the transcript `JALI-V02-LNP22-QUAD-MANY-C1-SHAKE128`, whose
//! instance is the parameters, the public seed, the commitment and the equations. Inside the
//! evaluation proof it continues that proof's transcript after $`h`$. Either way its verifier
//! message is the folding challenge $`\mu`$, one ring element per equation, and the quadratic
//! proof of the folded equation follows in the same transcript (`docs/encodings.md`).
use crate::wipe::Secret;
use crate::{
    Error,
    abdlop::{Abdlop, Commitment, Opening},
    linhc,
    quad::{self, QuadEq, QuadProof},
    rand::take_seed,
    transcript::{Protocol, Transcript},
};

#[cfg(test)]
mod tests;

/// Equations encoded at once before they are absorbed: in parallel with the `parallel`
/// feature (from 8 equations on), with the encodings of one chunk held at a time.
const ENCODING_CHUNK: usize = 32;
/// Absorb a list of equations as part of an instance: their number as `LE32`, then each one's
/// encoding ([`QuadEq::to_bytes`], held as segments of at most 32 KiB) as
/// `SerializeVarLenString`.
pub(crate) fn absorb_equations(
    transcript: &mut Transcript,
    equations: &[QuadEq],
) -> Result<(), Error> {
    transcript.absorb_len(equations.len() as u64)?;
    // Absorbed in order, up to the first equation that fails to encode.
    for chunk in equations.chunks(ENCODING_CHUNK) {
        for encoding in crate::par::map_min(chunk.len(), 8, |i| chunk[i].to_segments()) {
            transcript.absorb_varlen_segments(&encoding?)?;
        }
    }
    Ok(())
}
/// Squeeze the folding challenge $`\mu`$, one ring element per equation with each coefficient
/// by `DecodeUint` modulo $`q`$, and return $`\sum_i\mu_if_i`$ for the equations $`f_i`$.
fn fold(
    scheme: &Abdlop,
    equations: &[QuadEq],
    transcript: &mut Transcript,
) -> Result<QuadEq, Error> {
    let dim = quad::dimension(scheme);
    for equation in equations {
        equation.check(scheme.ring(), dim)?;
    }
    let weights = transcript.ring_elements(scheme.ring(), equations.len())?;
    // The keys and values of `add_scaled_assign` of each equation in turn on a zero equation.
    QuadEq::combination(scheme.ring(), dim, equations, &weights)
}
/// The transcript of the proof on its own, after its instance: the scheme, the commitment's
/// serialization and the equations ([`absorb_equations`]).
fn transcript(
    scheme: &Abdlop,
    commitment: &Commitment,
    equations: &[QuadEq],
    context: &[u8],
) -> Result<Transcript, Error> {
    let mut transcript = Transcript::new(Protocol::ManyQuadratic, context)?;
    scheme.absorb_scheme(&mut transcript)?;
    transcript.absorb(&scheme.commitment_bytes(commitment)?);
    absorb_equations(&mut transcript, equations)?;
    Ok(transcript)
}
/// Prove all equations after checking the original witness against every equation.
///
/// The seed must be secret and uniformly random, and it may be reused, also for the commitment
/// being proven. The masks are read under a key derived from the seed, the transcript before
/// the first message (parameters, public seed, context, commitment and equations) and the
/// opening: identical calls return identical proofs, and calls that differ in any input read
/// independent masks (assuming SHAKE128 and AES-256 behave as pseudorandom functions).
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
    equations: &[QuadEq],
    context: &[u8],
    mut seed: [u8; 32],
) -> Result<QuadProof, Error> {
    prove_seeded(
        scheme,
        commitment,
        opening,
        equations,
        context,
        &take_seed(&mut seed),
    )
}
pub(crate) fn prove_seeded(
    scheme: &Abdlop,
    commitment: &Commitment,
    opening: &Opening,
    equations: &[QuadEq],
    context: &[u8],
    seed: &Secret<[u8; 32]>,
) -> Result<QuadProof, Error> {
    let transcript = transcript(scheme, commitment, equations, context)?;
    Ok(prove_in(
        scheme, commitment, opening, equations, transcript, seed, None,
    )?
    .0)
}
/// The proof in `transcript`, positioned before the folding challenge, with the extractable
/// commitment beside the final opening when `ext` is given, returning the accepted attempt's
/// LinHC opening too.
pub(crate) fn prove_in(
    scheme: &Abdlop,
    commitment: &Commitment,
    opening: &Opening,
    equations: &[QuadEq],
    mut transcript: Transcript,
    seed: &Secret<[u8; 32]>,
    ext: Option<&linhc::ProverExt<'_>>,
) -> Result<(QuadProof, Option<linhc::LinhcOpening>), Error> {
    let witness = quad::interleave(&opening.s1, &opening.m)?;
    // Evaluated in parallel (`parallel`); the first failure in order is returned.
    crate::par::try_map(equations.len(), |i| {
        match equations[i].evaluate(&witness)?.is_zero() {
            true => Ok(()),
            false => Err(Error::Witness),
        }
    })?;
    let equation = fold(scheme, equations, &mut transcript)?;
    quad::prove_in(
        scheme,
        commitment,
        opening,
        &equation,
        &transcript,
        seed,
        ext,
    )
}
/// Verify the folded proof, deriving fresh coefficients from the full public statement.
pub fn verify(
    scheme: &Abdlop,
    commitment: &Commitment,
    equations: &[QuadEq],
    proof: &QuadProof,
    context: &[u8],
) -> Result<(), Error> {
    let transcript = transcript(scheme, commitment, equations, context)?;
    verify_in(scheme, commitment, equations, proof, transcript, None)
}
/// [`verify`] in `transcript`, as [`prove_in`] proves, with the extractable commitment's checks
/// when `ext` is given.
pub(crate) fn verify_in(
    scheme: &Abdlop,
    commitment: &Commitment,
    equations: &[QuadEq],
    proof: &QuadProof,
    mut transcript: Transcript,
    ext: Option<&linhc::VerifierExt<'_>>,
) -> Result<(), Error> {
    let equation = fold(scheme, equations, &mut transcript)?;
    quad::verify_in(scheme, commitment, &equation, proof, &transcript, ext)
}
