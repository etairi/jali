//! The toolbox proof's NARG string: the serializations of its prover messages in protocol
//! order, with the final round in the compact flavor (the challenge in place of the final
//! message's recomputed part). Every field has a fixed width that the validated parameters
//! determine (`docs/encodings.md`):
//!
//! 1. the commitment, $`t_{A,1}`$ in $`\mathrm{bitlen}(q-1)-D`$ bits per coefficient and
//!    $`t_B`$ modulo $`q`$ ($`\ell+e+a+1`$ polynomials, for the $`e`$ exact and $`a`$
//!    approximate slots), and in the extractable mode the LinHC commitment $`t'_1,t'_2`$ modulo
//!    $`q'`$,
//! 2. the range responses $`z^{(e)}`$ and $`z^{(d)}`$ in their response widths, then the
//!    garbage commitments modulo $`q`$,
//! 3. $`h`$ modulo $`q`$,
//! 4. $`t`$ modulo $`q`$, the challenge, $`z_1`$ and $`z_{21}`$ in their response widths and
//!    the hints in the hint width, and in the extractable mode the LinHC opening
//!    $`z'_1,z'_2,z'_3`$ in the width $`t'+4`$.
//!
//! Parts 1 to 3 and $`t`$ are the bytes the transcript absorbs for those messages.
use super::{BitReader, BitWriter, fields};
use crate::{
    Error,
    abdlop::Abdlop,
    math::{Poly, PolyVec},
    params::{ExtractionParams, response_width},
    quad::QuadProof,
    quad_eval::EvalProof,
    tbox::{ExtractableProof, Proof},
};

/// The exact and approximate range slots and the garbage rows of `scheme`.
fn dimensions(scheme: &Abdlop) -> (usize, usize, usize) {
    (
        if scheme.checked.n_ex > 0 {
            256 / scheme.ring().degree()
        } else {
            0
        },
        if scheme.parameters.n_prime > 0 {
            256 / scheme.ring().degree()
        } else {
            0
        },
        scheme.checked.lambda / 2,
    )
}
fn shape(scheme: &Abdlop, v: &PolyVec, len: usize) -> Result<(), Error> {
    if v.len() != len {
        return Err(Error::Dimension);
    }
    if v.ring() != scheme.ring() {
        return Err(Error::RingMismatch);
    }
    Ok(())
}

/// Encode a toolbox proof as its NARG string (module documentation).
pub fn encode(scheme: &Abdlop, proof: &Proof) -> Result<Vec<u8>, Error> {
    let mut writer = BitWriter::new();
    encode_into(&mut writer, scheme, proof, None)?;
    writer.into_bytes()
}

/// Encode a proof of the straight-line extractable mode as its NARG string: the toolbox proof's
/// fields as in [`encode`], with the LinHC commitment $`t'_1,t'_2`$ (modulo $`q'`$) at the end of
/// the first message and the LinHC opening $`z'_1,z'_2,z'_3`$ (in the width $`t'+4`$) at the
/// end, every length from the parameters and the extraction set.
pub fn encode_extractable(
    scheme: &Abdlop,
    extraction: &ExtractionParams,
    proof: &ExtractableProof,
) -> Result<Vec<u8>, Error> {
    let (params, _) = extraction.linhc(&scheme.parameters, &scheme.checked)?;
    let ring = params.ring()?;
    for (v, len) in [
        (&proof.t1, params.len),
        (&proof.t2, params.len),
        (&proof.z1, params.rank),
        (&proof.z2, params.len),
        (&proof.z3, params.len),
    ] {
        if v.len() != len {
            return Err(Error::Dimension);
        }
        if v.ring() != &ring {
            return Err(Error::RingMismatch);
        }
    }
    let mut writer = BitWriter::new();
    encode_into(
        &mut writer,
        scheme,
        &proof.proof,
        Some((&proof.t1, &proof.t2)),
    )?;
    let width = response_width(params.log_sigma);
    for z in [&proof.z1, &proof.z2, &proof.z3] {
        fields::write_signed(&mut writer, z, width)?;
    }
    writer.into_bytes()
}

/// The fields of [`encode`], written to `writer`, with the LinHC commitment after the toolbox's
/// commitment when given.
fn encode_into(
    writer: &mut BitWriter,
    scheme: &Abdlop,
    proof: &Proof,
    linhc_commitment: Option<(&PolyVec, &PolyVec)>,
) -> Result<(), Error> {
    let (e, d, g) = dimensions(scheme);
    let ring = scheme.ring();
    let opening = &proof.evaluation.quadratic.opening;
    shape(
        scheme,
        &proof.commitment.t_b,
        scheme.message_len() + e + d + 1,
    )?;
    shape(scheme, &proof.commitment.t_a, scheme.parameters.n_msis)?;
    shape(scheme, &proof.evaluation.garbage_commitments, g)?;
    shape(scheme, &proof.evaluation.h, g)?;
    shape(scheme, &proof.z_exact, e)?;
    shape(scheme, &proof.z_approx, d)?;
    if proof.evaluation.quadratic.t.ring() != ring {
        return Err(Error::RingMismatch);
    }
    scheme.write_commitment(writer, &proof.commitment)?;
    if let Some((t1, t2)) = linhc_commitment {
        crate::linhc::write_pair(writer, t1, t2)?;
    }
    fields::write_signed(
        writer,
        &proof.z_exact,
        response_width(scheme.parameters.log_sigma[2]),
    )?;
    fields::write_signed(
        writer,
        &proof.z_approx,
        response_width(scheme.parameters.log_sigma[3]),
    )?;
    fields::write_below(writer, &proof.evaluation.garbage_commitments, &ring.q)?;
    fields::write_below(writer, &proof.evaluation.h, &ring.q)?;
    fields::write_below(
        writer,
        &PolyVec::new(ring.clone(), vec![proof.evaluation.quadratic.t.clone()])?,
        &ring.q,
    )?;
    scheme.write_opening(writer, opening)
}

/// Strict decoder with dimensions taken solely from validated parameters: every field in its
/// range, and no trailing bytes. Inputs above 16 MiB are refused before any parsing.
pub fn decode(scheme: &Abdlop, bytes: &[u8]) -> Result<Proof, Error> {
    let mut reader = super::proof_reader(bytes)?;
    let (proof, _) = decode_from(&mut reader, scheme, None)?;
    reader.finish_bytes()?;
    Ok(proof)
}

/// Strictly decode a proof of the straight-line extractable mode ([`encode_extractable`]):
/// every length, the modulus $`q'`$ and the width $`t'+4`$ come from the parameters and the
/// extraction set, the LinHC commitment's coefficients are below $`q'`$, and no bytes may
/// follow.
pub fn decode_extractable(
    scheme: &Abdlop,
    extraction: &ExtractionParams,
    bytes: &[u8],
) -> Result<ExtractableProof, Error> {
    let (params, _) = extraction.linhc(&scheme.parameters, &scheme.checked)?;
    let ring = params.ring()?;
    let mut reader = super::proof_reader(bytes)?;
    let (proof, linhc_commitment) = decode_from(&mut reader, scheme, Some((&ring, params.len)))?;
    let (t1, t2) = linhc_commitment.ok_or(Error::Encoding)?;
    let width = response_width(params.log_sigma);
    let z1 = fields::read_signed(&mut reader, &ring, params.rank, width)?;
    let z2 = fields::read_signed(&mut reader, &ring, params.len, width)?;
    let z3 = fields::read_signed(&mut reader, &ring, params.len, width)?;
    reader.finish_bytes()?;
    Ok(ExtractableProof {
        proof,
        t1,
        t2,
        z1,
        z2,
        z3,
    })
}

/// The fields of [`decode`] from `reader`, which the caller finishes, with the LinHC
/// commitment over `linhc` (its ring and length) after the toolbox's commitment when given.
#[allow(clippy::type_complexity)]
fn decode_from(
    reader: &mut BitReader<'_>,
    scheme: &Abdlop,
    linhc: Option<(&std::sync::Arc<crate::math::Ring>, usize)>,
) -> Result<(Proof, Option<(PolyVec, PolyVec)>), Error> {
    let (e, d, g) = dimensions(scheme);
    let ring = scheme.ring();
    let commitment = scheme.read_commitment(reader, scheme.message_len() + e + d + 1)?;
    let linhc_commitment = match linhc {
        Some((linhc_ring, len)) => {
            let q = linhc_ring.modulus();
            Some((
                fields::read_below(reader, linhc_ring, len, &q)?,
                fields::read_below(reader, linhc_ring, len, &q)?,
            ))
        }
        None => None,
    };
    let z_exact = fields::read_signed(
        reader,
        ring,
        e,
        response_width(scheme.parameters.log_sigma[2]),
    )?;
    let z_approx = fields::read_signed(
        reader,
        ring,
        d,
        response_width(scheme.parameters.log_sigma[3]),
    )?;
    let garbage_commitments = fields::read_below(reader, ring, g, &ring.q)?;
    let h = fields::read_below(reader, ring, g, &ring.q)?;
    let t: Poly = fields::read_below(reader, ring, 1, &ring.q)?.entries()[0].clone();
    let opening = scheme.read_opening(reader)?;
    Ok((
        Proof {
            commitment,
            z_exact,
            z_approx,
            evaluation: EvalProof {
                garbage_commitments,
                h,
                quadratic: QuadProof { t, opening },
            },
        },
        linhc_commitment,
    ))
}
