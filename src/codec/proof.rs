//! Fixed-shape toolbox proof encoding: validated parameters determine every field and length.
use super::{BitReader, BitWriter};
use crate::{
    Error,
    abdlop::{self, Abdlop, Commitment, OpeningProof},
    math::{Poly, PolyVec, Ring, U256},
    params::ExtractionParams,
    quad::QuadProof,
    quad_eval::EvalProof,
    tbox::{ExtractableProof, Proof},
};
use std::sync::Arc;

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
/// The width of a compressed high part: $`\mathrm{bits}(q-1)-D`$.
fn high_bits(scheme: &Abdlop) -> u32 {
    scheme.ring().coefficient_bits() - scheme.parameters.d_bits
}

/// Encode `tB, h, tA1, c, hint, z1, z21, z3, z4, end` without wire-controlled lengths.
pub fn encode(scheme: &Abdlop, proof: &Proof) -> Result<Vec<u8>, Error> {
    let mut writer = BitWriter::new();
    encode_into(&mut writer, scheme, proof)?;
    Ok(writer.finish())
}

/// Encode a proof of the straight-line extractable mode: the toolbox proof's fields as in
/// [`encode`], then the LinHC commitment $`t_1,t_2`$ in the uniform code modulo $`q'`$ and the
/// LinHC opening $`z'_1,z'_2,z'_3`$ in the Gaussian code of width $`1.55\cdot2^{t'}`$, with
/// every length from the parameters and the extraction set.
pub fn encode_extractable(
    scheme: &Abdlop,
    extraction: &ExtractionParams,
    proof: &ExtractableProof,
) -> Result<Vec<u8>, Error> {
    let (params, _) = extraction.linhc(&scheme.parameters, &scheme.checked)?;
    let ring = params.ring()?;
    let mut writer = BitWriter::new();
    encode_into(&mut writer, scheme, &proof.proof)?;
    for (v, len) in [(&proof.t1, params.len), (&proof.t2, params.len)] {
        shape_in(&ring, v, len)?;
        abdlop::encode_vec(&mut writer, v, &ring.modulus())?;
    }
    for (v, len) in [
        (&proof.z1, params.rank),
        (&proof.z2, params.len),
        (&proof.z3, params.len),
    ] {
        shape_in(&ring, v, len)?;
        abdlop::write_gaussians(&mut writer, v, params.log_sigma)?;
    }
    Ok(writer.finish())
}

fn shape_in(ring: &Arc<Ring>, v: &PolyVec, len: usize) -> Result<(), Error> {
    if v.len() != len {
        return Err(Error::Dimension);
    }
    if v.ring() != ring {
        return Err(Error::RingMismatch);
    }
    Ok(())
}

/// The fields of [`encode`], written to `writer`.
fn encode_into(writer: &mut BitWriter, scheme: &Abdlop, proof: &Proof) -> Result<(), Error> {
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
    shape(scheme, &opening.z1, scheme.bounded_len())?;
    shape(
        scheme,
        &opening.z21,
        scheme.parameters.m2 - scheme.parameters.n_msis,
    )?;
    shape(scheme, &opening.hint, scheme.parameters.n_msis)?;
    shape(scheme, &proof.z_exact, e)?;
    shape(scheme, &proof.z_approx, d)?;
    if proof.evaluation.quadratic.t.ring() != ring || opening.challenge.ring() != ring {
        return Err(Error::RingMismatch);
    }
    let full = PolyVec::new(
        ring.clone(),
        proof
            .commitment
            .t_b
            .entries()
            .iter()
            .chain(proof.evaluation.garbage_commitments.entries())
            .chain(std::iter::once(&proof.evaluation.quadratic.t))
            .chain(proof.evaluation.h.entries())
            .cloned()
            .collect(),
    )?;
    abdlop::encode_vec(writer, &full, &ring.modulus())?;
    // A high part below 2^k written in k bits is the uniform code modulo 2^k.
    let bits = high_bits(scheme);
    let high_max = scheme.max_high();
    for p in proof.commitment.t_a.entries() {
        for x in p.coefficients() {
            if *x > high_max {
                return Err(Error::Encoding);
            }
            writer.unsigned_u256(x, bits)?;
        }
    }
    scheme.write_challenge(writer, &opening.challenge)?;
    scheme.write_hints(writer, &opening.hint)?;
    for (v, t) in [
        (&opening.z1, scheme.parameters.log_sigma[0]),
        (&opening.z21, scheme.parameters.log_sigma[1]),
        (&proof.z_exact, scheme.parameters.log_sigma[2]),
        (&proof.z_approx, scheme.parameters.log_sigma[3]),
    ] {
        abdlop::write_gaussians(writer, v, t)?;
    }
    Ok(())
}

/// Read `len` polynomials; `coeff` returns canonical values in $`[0,q)`$.
fn vector(
    reader: &mut BitReader<'_>,
    scheme: &Abdlop,
    len: usize,
    coeff: impl FnMut(&mut BitReader<'_>) -> Result<U256, Error>,
) -> Result<PolyVec, Error> {
    vector_in(reader, scheme.ring(), len, coeff)
}
/// [`vector`] over `ring`.
fn vector_in(
    reader: &mut BitReader<'_>,
    ring: &Arc<Ring>,
    len: usize,
    mut coeff: impl FnMut(&mut BitReader<'_>) -> Result<U256, Error>,
) -> Result<PolyVec, Error> {
    let mut entries = Vec::with_capacity(len);
    for _ in 0..len {
        entries.push(Poly::from_canonical(
            ring.clone(),
            (0..ring.degree())
                .map(|_| coeff(reader))
                .collect::<Result<_, _>>()?,
        ));
    }
    PolyVec::new(ring.clone(), entries)
}

/// Strict decoder with dimensions taken solely from validated parameters.
/// Maximum input length is 16 MiB; all unary codes and coefficient ranges are bounded.
pub fn decode(scheme: &Abdlop, bytes: &[u8]) -> Result<Proof, Error> {
    let mut reader = super::proof_reader(bytes)?;
    let proof = decode_from(&mut reader, scheme)?;
    reader.finish()?;
    Ok(proof)
}

/// Strictly decode a proof of the straight-line extractable mode ([`encode_extractable`]):
/// every length, the modulus $`q'`$ and the width $`t'`$ come from the parameters and the
/// extraction set, the uniform codes are below $`q'`$, the Gaussian codes are bounded and below
/// $`q'/2`$, and no bits may follow.
pub fn decode_extractable(
    scheme: &Abdlop,
    extraction: &ExtractionParams,
    bytes: &[u8],
) -> Result<ExtractableProof, Error> {
    let (params, _) = extraction.linhc(&scheme.parameters, &scheme.checked)?;
    let ring = params.ring()?;
    let q = ring.modulus();
    let half = q.shr_vartime(1);
    let mut reader = super::proof_reader(bytes)?;
    let proof = decode_from(&mut reader, scheme)?;
    let t1 = vector_in(&mut reader, &ring, params.len, |r| r.uniform_u256(&q))?;
    let t2 = vector_in(&mut reader, &ring, params.len, |r| r.uniform_u256(&q))?;
    let mut openings = Vec::with_capacity(3);
    for len in [params.rank, params.len, params.len] {
        openings.push(vector_in(&mut reader, &ring, len, |r| {
            let x = r.gaussian(params.log_sigma, 1 << 20)?;
            if U256::from_u128(x.unsigned_abs()) > half {
                return Err(Error::Encoding);
            }
            Ok(ring.reduce_i128(x))
        })?);
    }
    reader.finish()?;
    let [z1, z2, z3]: [PolyVec; 3] = openings.try_into().map_err(|_| Error::Encoding)?;
    Ok(ExtractableProof {
        proof,
        t1,
        t2,
        z1,
        z2,
        z3,
    })
}

/// The fields of [`decode`] from `reader`, which the caller finishes.
fn decode_from(reader: &mut BitReader<'_>, scheme: &Abdlop) -> Result<Proof, Error> {
    let (e, d, g) = dimensions(scheme);
    let ring = scheme.ring();
    let q = ring.modulus();
    let tb = vector(reader, scheme, scheme.message_len() + e + d + 1, |r| {
        r.uniform_u256(&q)
    })?;
    let garbage = vector(reader, scheme, g, |r| r.uniform_u256(&q))?;
    let t = vector(reader, scheme, 1, |r| r.uniform_u256(&q))?.entries()[0].clone();
    let h = vector(reader, scheme, g, |r| r.uniform_u256(&q))?;
    let bits = high_bits(scheme);
    let high_max = scheme.max_high();
    let ta = vector(reader, scheme, scheme.parameters.n_msis, |r| {
        let x = r.unsigned_u256(bits)?;
        if x > high_max {
            return Err(Error::Encoding);
        }
        Ok(x)
    })?;
    let c = vector(reader, scheme, 1, |r| {
        Ok(ring.reduce_i128(
            r.uniform((2 * scheme.checked.omega + 1) as u128)? as i128 - scheme.checked.omega,
        ))
    })?
    .entries()[0]
        .clone();
    let hint = vector(reader, scheme, scheme.parameters.n_msis, |r| {
        Ok(ring.reduce_i128(scheme.read_hint(r)?))
    })?;
    let mut responses = Vec::new();
    for (len, t) in [
        (scheme.bounded_len(), scheme.parameters.log_sigma[0]),
        (
            scheme.parameters.m2 - scheme.parameters.n_msis,
            scheme.parameters.log_sigma[1],
        ),
        (e, scheme.parameters.log_sigma[2]),
        (d, scheme.parameters.log_sigma[3]),
    ] {
        responses.push(vector(reader, scheme, len, |r| {
            let x = r.gaussian(t, 1 << 20)?;
            if U256::from_u128(x.unsigned_abs()) > ring.half {
                return Err(Error::Encoding);
            }
            Ok(ring.reduce_i128(x))
        })?);
    }
    let mut responses = responses.into_iter();
    Ok(Proof {
        commitment: Commitment { t_a: ta, t_b: tb },
        evaluation: EvalProof {
            garbage_commitments: garbage,
            h,
            quadratic: QuadProof {
                t,
                opening: OpeningProof {
                    challenge: c,
                    z1: responses.next().expect("four responses"),
                    z21: responses.next().expect("four responses"),
                    hint,
                },
            },
        },
        z_exact: responses.next().expect("four responses"),
        z_approx: responses.next().expect("four responses"),
    })
}
