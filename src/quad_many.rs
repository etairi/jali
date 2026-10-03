//! Independent ring-valued Schwartz–Zippel folding, LNP22 Fig. 7.
use crate::{
    Error,
    abdlop::{Abdlop, Commitment, Opening},
    codec::Segments,
    math::Poly,
    quad::{self, QuadEq, QuadProof},
    rand::{AesPrg, domain, take_seed, uniform_ring},
    transcript::Transcript,
};
use zeroize::Zeroizing;

#[cfg(test)]
mod tests;

/// Equations encoded at once before they are absorbed: in parallel with the `parallel`
/// feature (from 8 equations on), with the encodings of one chunk held at a time.
const ENCODING_CHUNK: usize = 32;
/// Absorb the number of `equations`, then each one's encoding ([`QuadEq::to_bytes`], held as
/// segments of at most 32 KiB) under the label `equation`.
pub(crate) fn absorb_equations(
    transcript: &mut Transcript,
    label: &[u8],
    equations: &[QuadEq],
) -> Result<(), Error> {
    transcript.absorb(label, &(equations.len() as u64).to_le_bytes());
    // Absorbed in order, up to the first equation that fails to encode.
    for chunk in equations.chunks(ENCODING_CHUNK) {
        for encoding in crate::par::map_min(chunk.len(), 8, |i| chunk[i].to_segments()) {
            transcript.absorb_segments(b"equation", &encoding?);
        }
    }
    Ok(())
}
/// The encodings of `equations`, for absorbing one list into several transcripts; computed in
/// parallel with the `parallel` feature. Each is held as segments, in chunks of at most 32 KiB
/// in which long runs of zero bytes, such as the codes of zero polynomials, are counts.
pub(crate) fn encode_equations(equations: &[QuadEq]) -> Result<Vec<Segments>, Error> {
    crate::par::try_map_min(equations.len(), 8, |i| equations[i].to_segments())
}
/// [`absorb_equations`] from the encodings of [`encode_equations`]: the same absorptions.
pub(crate) fn absorb_encoded(transcript: &mut Transcript, label: &[u8], encoded: &[Segments]) {
    transcript.absorb(label, &(encoded.len() as u64).to_le_bytes());
    for encoding in encoded {
        transcript.absorb_segments(b"equation", encoding);
    }
}
fn fold(
    scheme: &Abdlop,
    commitment: &Commitment,
    equations: &[QuadEq],
    context: &[u8],
) -> Result<(QuadEq, [u8; 32]), Error> {
    let dim = 2 * (scheme.bounded_len() + scheme.message_len());
    let mut prefix = scheme.prefix(commitment, context)?;
    absorb_equations(&mut prefix, b"many-quadratics", equations)?;
    let seed = prefix.challenge_seed(b"mu");
    let mut folded = QuadEq::zero(scheme.ring().clone(), dim)?;
    for (i, equation) in equations.iter().enumerate() {
        equation.check(scheme.ring(), dim)?;
        let index = u32::try_from(i + 1).map_err(|_| Error::Dimension)?;
        let values = uniform_ring(
            &mut AesPrg::new(&seed, domain(0x4d550000, index)),
            scheme.ring(),
            scheme.ring().degree(),
        )?;
        let weight = Poly::from_canonical(scheme.ring().clone(), values);
        // In place: the keys and values of `folded.add(&equation.scale(&weight)?)`.
        folded.add_scaled_assign(equation, &weight)?;
    }
    Ok((folded, prefix.digest()))
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
    seed: &Zeroizing<[u8; 32]>,
) -> Result<QuadProof, Error> {
    let witness = quad::interleave(&opening.s1, &opening.m)?;
    // Evaluated in parallel (`parallel`); the first failure in order is returned.
    crate::par::try_map(equations.len(), |i| {
        match equations[i].evaluate(&witness)?.is_zero() {
            true => Ok(()),
            false => Err(Error::Witness),
        }
    })?;
    let (equation, binding) = fold(scheme, commitment, equations, context)?;
    quad::prove_seeded(scheme, commitment, opening, &equation, &binding, seed)
}
/// Verify the folded proof, deriving fresh coefficients from the full public statement.
pub fn verify(
    scheme: &Abdlop,
    commitment: &Commitment,
    equations: &[QuadEq],
    proof: &QuadProof,
    context: &[u8],
) -> Result<(), Error> {
    let (equation, binding) = fold(scheme, commitment, equations, context)?;
    quad::verify(scheme, commitment, &equation, proof, &binding)
}
