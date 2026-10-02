//! Fixed-shape toolbox proof encoding: validated parameters determine every field and length.
use super::{BitReader, BitWriter};
use crate::{
    Error,
    abdlop::{self, Abdlop, Commitment, OpeningProof},
    math::{Poly, PolyVec, U256},
    quad::QuadProof,
    quad_eval::EvalProof,
    tbox::Proof,
};

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
    let mut writer = BitWriter::new();
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
    abdlop::encode_vec(&mut writer, &full, &ring.modulus())?;
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
    scheme.write_challenge(&mut writer, &opening.challenge)?;
    scheme.write_hints(&mut writer, &opening.hint)?;
    for (v, t) in [
        (&opening.z1, scheme.parameters.log_sigma[0]),
        (&opening.z21, scheme.parameters.log_sigma[1]),
        (&proof.z_exact, scheme.parameters.log_sigma[2]),
        (&proof.z_approx, scheme.parameters.log_sigma[3]),
    ] {
        abdlop::write_gaussians(&mut writer, v, t)?;
    }
    Ok(writer.finish())
}

/// Read `len` polynomials; `coeff` returns canonical values in $`[0,q)`$.
fn vector(
    reader: &mut BitReader<'_>,
    scheme: &Abdlop,
    len: usize,
    mut coeff: impl FnMut(&mut BitReader<'_>) -> Result<U256, Error>,
) -> Result<PolyVec, Error> {
    let ring = scheme.ring();
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
    let (e, d, g) = dimensions(scheme);
    let ring = scheme.ring();
    let q = ring.modulus();
    let tb = vector(&mut reader, scheme, scheme.message_len() + e + d + 1, |r| {
        r.uniform_u256(&q)
    })?;
    let garbage = vector(&mut reader, scheme, g, |r| r.uniform_u256(&q))?;
    let t = vector(&mut reader, scheme, 1, |r| r.uniform_u256(&q))?.entries()[0].clone();
    let h = vector(&mut reader, scheme, g, |r| r.uniform_u256(&q))?;
    let bits = high_bits(scheme);
    let high_max = scheme.max_high();
    let ta = vector(&mut reader, scheme, scheme.parameters.n_msis, |r| {
        let x = r.unsigned_u256(bits)?;
        if x > high_max {
            return Err(Error::Encoding);
        }
        Ok(x)
    })?;
    let c = vector(&mut reader, scheme, 1, |r| {
        Ok(ring.reduce_i128(
            r.uniform((2 * scheme.checked.omega + 1) as u128)? as i128 - scheme.checked.omega,
        ))
    })?
    .entries()[0]
        .clone();
    let hint = vector(&mut reader, scheme, scheme.parameters.n_msis, |r| {
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
        responses.push(vector(&mut reader, scheme, len, |r| {
            let x = r.gaussian(t, 1 << 20)?;
            if U256::from_u128(x.unsigned_abs()) > ring.half {
                return Err(Error::Encoding);
            }
            Ok(ring.reduce_i128(x))
        })?);
    }
    reader.finish()?;
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
