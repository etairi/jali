//! LaBRADOR (Beullens and Seiler, ePrint 2022/1341, CRYPTO 2023): succinct proofs of knowledge
//! of short solutions to dot-product constraints over $`\mathcal R_q`$, by recursion of an
//! amortized base protocol.
//!
//! **Not zero knowledge.** The proof reveals information about the witness: the paper
//! disregards zero knowledge (BS23 §1), and so does this module. Use it where the witness may
//! leak, such as aggregating signatures, or behind a zero-knowledge layer.
//!
//! A [`Statement`] of the principal relation of BS23 §5.1 holds $`r`$ witness vectors of $`n`$
//! polynomials over a [`Rq`], a bound $`\beta^2`$ on the sum of their squared norms, and
//! dot-product constraints that must vanish in $`\mathcal R_q`$ ([`Statement::push`]) or in
//! their constant coefficient ([`Statement::push_constant_term`]). [`prove`] runs levels of the
//! main protocol of BS23 Fig. 2, each reducing the witness to a shorter witness of a new
//! statement (§5.3), and sends the last witness in the clear. [`verify`] checks each level's
//! parameters and projection, follows the reduction, and checks the last witness.
//! `docs/labrador.md` describes the protocol as implemented, its parameters, its encodings and
//! where it departs from the paper.
//!
//! The extractor only guarantees a witness of norm at most $`\sqrt{337/30}\,\beta`$ for the
//! input statement (the slack of the projection check), and the hardness of Module-SIS rests
//! on the cost-model estimate of [`Policy`]. Like the rest of the crate, the code runs in
//! variable time.
mod decompose;
mod level;
mod params;
mod plan;
mod recursion;
mod ring;
mod sample;
mod statement;
#[cfg(test)]
pub(crate) mod test_rng;
#[cfg(test)]
mod tests;

pub use params::Policy;
pub use ring::Rq;
pub use statement::{Constraint, MAX_WITNESS_COEFFICIENTS, Statement, Witness};

use crate::{
    Error,
    codec::MAX_PROOF_BYTES,
    math::Ring,
    transcript::{Protocol, Transcript},
};
use level::Ctx;
use sample::ChallengeSet;
use statement::Relation;

/// The most levels a proof may have.
pub const MAX_LEVELS: u32 = 32;

/// A LaBRADOR proof: the encoding of `docs/labrador.md`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    bytes: Vec<u8>,
}

impl Proof {
    /// The encoding.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// A proof from its encoding, which [`verify`] parses. Refuses encodings above 16 MiB with
    /// [`Error::Encoding`].
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, Error> {
        if bytes.len() > MAX_PROOF_BYTES {
            return Err(Error::Encoding);
        }
        Ok(Self { bytes })
    }

    /// The length of the encoding in bytes.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the encoding is empty (never, for a proof that [`prove`] made).
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

fn context(
    statement: &Statement,
    pp_seed: &[u8; 32],
    app_context: &[u8],
    policy: &Policy,
) -> Result<(Ctx, Transcript), Error> {
    policy.check()?;
    let ring = statement.ring().clone();
    let (q, d) = (ring.modulus(), ring.degree());
    let set = ChallengeSet::for_degree(d)?;
    let ctx = Ctx {
        eta_ring: Ring::new(i128::from(q), d)?,
        set,
        policy: *policy,
        pp_seed: *pp_seed,
        lambda: params::lambda(q),
        ring,
    };
    let mut transcript = Transcript::new(Protocol::Labrador, app_context)?;
    transcript.absorb(pp_seed);
    transcript.absorb(&policy.min_block_size.to_le_bytes());
    statement.absorb(&mut transcript)?;
    Ok((ctx, transcript))
}

/// Prove that `witness` satisfies `statement`, with the public matrices expanded from
/// `pp_seed`, under the application context `context` and the hardness `policy`.
///
/// Fails with [`Error::Witness`] if the witness does not satisfy the statement, with
/// [`Error::Parameter`] if the ring has no challenge set (only degree 64 has one), if the norm
/// bound is too large for the modulus ($`337\cdot125^2\beta^2\le30q^2`$ is required) or if no
/// parameters meet the policy, and with [`Error::RestartLimit`] in the unlikely case that a
/// level exceeds its norm bounds in every attempt.
pub fn prove(
    statement: &Statement,
    witness: &Witness,
    pp_seed: &[u8; 32],
    context_bytes: &[u8],
    policy: &Policy,
) -> Result<Proof, Error> {
    statement.check(witness)?;
    let (ctx, mut transcript) = context(statement, pp_seed, context_bytes, policy)?;
    let mut out = Vec::new();
    let (mut next, mut wit) = level::prove(
        &ctx,
        statement,
        &witness.vectors,
        0,
        &mut transcript,
        &mut out,
    )?;
    let mut level = 1;
    while !next.params.last {
        if level >= MAX_LEVELS {
            return Err(Error::Parameter("too many LaBRADOR levels"));
        }
        let (n2, w2) = level::prove(&ctx, &next, &wit, level, &mut transcript, &mut out)?;
        next = n2;
        wit = w2;
        level += 1;
    }
    let bytes = level::encode_final(&ctx.ring, &next.layout, next.params.nu, &wit)?;
    out.extend_from_slice(&bytes);
    Proof::from_bytes(out)
}

/// Verify `proof` for `statement` with the same public seed, context and policy as the
/// prover. Fails with [`Error::InvalidProof`] or [`Error::Encoding`].
pub fn verify(
    statement: &Statement,
    proof: &Proof,
    pp_seed: &[u8; 32],
    context_bytes: &[u8],
    policy: &Policy,
) -> Result<(), Error> {
    let (ctx, mut transcript) = context(statement, pp_seed, context_bytes, policy)?;
    let mut input = proof.as_bytes();
    let mut next = level::verify(&ctx, statement, 0, &mut transcript, &mut input)?;
    let mut level = 1;
    while !next.params.last {
        if level >= MAX_LEVELS {
            return Err(Error::InvalidProof);
        }
        next = level::verify(&ctx, &next, level, &mut transcript, &mut input)?;
        level += 1;
    }
    // The last witness, in the clear: its norm, then all equations of the last statement at
    // once, aggregated with challenges squeezed after the witness is absorbed.
    let before = input;
    let wit = level::decode_final(&ctx.ring, &next.layout, next.params.nu, &mut input)?;
    if !input.is_empty() {
        return Err(Error::Encoding);
    }
    transcript.absorb(&before[..before.len() - input.len()]);
    let norm: u128 = wit.iter().map(|w| ctx.ring.norm_sq(w)).sum();
    if norm > next.params.next_norm_sq {
        return Err(Error::InvalidProof);
    }
    let alpha = level::squeeze_polys(&mut transcript, &ctx.ring, next.full_count())?;
    let agg = next.aggregate(&alpha, &[])?;
    let hats: Vec<Vec<u64>> = wit.iter().map(|w| ctx.ring.ntt_many(w)).collect();
    if level::evaluate_aggregate(&ctx.ring, &agg, &hats)
        .iter()
        .any(|x| *x != 0)
    {
        return Err(Error::InvalidProof);
    }
    Ok(())
}
