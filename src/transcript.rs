//! The Fiat–Shamir transcript of every proof: the duplex sponge of
//! draft-irtf-cfrg-fiat-shamir-04 over SHAKE128 ([`crate::fiat_shamir`]), one per proof.
//!
//! A proof starts the sponge with the session identifier `DeriveSessionID(tag)` (§5.1), where
//! the tag is the protocol's ASCII identifier ([`Protocol::tag`]) followed by
//! $`\mathrm{LE32}(|ctx|)\,\|\,ctx`$ for the caller's application context. It then absorbs the
//! instance once (§5.2), then each prover message as its serialization and squeezes each
//! verifier message, in protocol order (`docs/encodings.md` lists the order of every proof).
//! Nothing else enters the sponge: data that the instance and the messages determine, such as
//! folded or projected equations, is not absorbed.
//!
//! Verifier messages are squeezed with a fixed length each and decoded by infallible maps
//! (§3.2): integers modulo $`M`$ by `DecodeUint` from $`N_s+16`$ bytes ([`Transcript::uints`]),
//! entries of the projection matrices by the centred binomial decoding of FIPS 203 with
//! $`\eta=1`$ from 2 bits ([`Transcript::ternary_packed`]), and the challenge polynomial by
//! [`crate::rand::challenge_from_bytes`] from [`CHALLENGE_BYTES`] bytes.
use crate::{
    Error,
    codec::Segments,
    fiat_shamir::{DuplexSponge, decode_uint, derive_session_id, ns},
    math::{Poly, Ring, U256, int},
};
use std::sync::Arc;

#[cfg(test)]
pub(crate) mod record;
#[cfg(test)]
mod tests;

/// The bytes squeezed for one challenge polynomial ([`crate::rand::CHALLENGE_BYTES`]), 24 blocks
/// of the rate of SHAKE128.
pub(crate) const CHALLENGE_BYTES: usize = crate::rand::CHALLENGE_BYTES;
const _: () = assert!(CHALLENGE_BYTES == 24 * crate::fiat_shamir::RATE);

/// The arguments whose transcripts this crate runs, each with its own tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Protocol {
    /// The toolbox proof (`tbox::prove_with_seed`).
    Toolbox,
    /// The toolbox proof in the straight-line extractable mode
    /// (`tbox::prove_extractable_with_seed`).
    ToolboxExtractable,
    /// The opening proof on its own (`Abdlop::prove_with_seed`).
    Opening,
    /// One quadratic equation on its own (`quad::prove_with_seed`).
    Quadratic,
    /// Several quadratic equations on their own (`quad_many::prove_with_seed`).
    ManyQuadratic,
    /// Quadratic equations and vanishing constant coefficients on their own
    /// (`quad_eval::prove_with_seed`).
    Evaluation,
    /// A LaBRADOR proof (`labrador::prove`).
    Labrador,
}

impl Protocol {
    /// Every protocol, for the tests of the tags.
    #[cfg(test)]
    pub(crate) const ALL: [Self; 7] = [
        Self::Toolbox,
        Self::ToolboxExtractable,
        Self::Opening,
        Self::Quadratic,
        Self::ManyQuadratic,
        Self::Evaluation,
        Self::Labrador,
    ];

    /// The ASCII part of the tag: the crate, the transcript version `V02`, the argument, the
    /// codec version `C1` (the serializations and decodings of `docs/encodings.md`) and the
    /// hash suite. No tag is a prefix of another, so the tag and the context that follows it
    /// are recovered from their concatenation.
    pub(crate) const fn tag(self) -> &'static [u8] {
        match self {
            Self::Toolbox => b"JALI-V02-LNP22-TBOX-C1-SHAKE128",
            Self::ToolboxExtractable => b"JALI-V02-LNP22-TBOX-SL-C1-SHAKE128",
            Self::Opening => b"JALI-V02-ABDLOP-OPENING-C1-SHAKE128",
            Self::Quadratic => b"JALI-V02-LNP22-QUAD-C1-SHAKE128",
            Self::ManyQuadratic => b"JALI-V02-LNP22-QUAD-MANY-C1-SHAKE128",
            Self::Evaluation => b"JALI-V02-LNP22-QUAD-EVAL-C1-SHAKE128",
            Self::Labrador => b"JALI-V02-LABRADOR-C1-SHAKE128",
        }
    }
}

/// The label of the 32 bytes that bind a prover's keys to the transcript
/// ([`Transcript::binding`]).
const BINDING_LABEL: &[u8] = b"JALI-V02-PROVER-KEY-BINDING";

/// The duplex sponge of one proof. Cloning it forks the transcript, as the prover does for each
/// attempt of a rejection-sampling loop.
#[derive(Clone)]
pub(crate) struct Transcript {
    sponge: DuplexSponge,
}

impl core::fmt::Debug for Transcript {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Transcript").finish_non_exhaustive()
    }
}

impl Transcript {
    /// `Init(DeriveSessionID(tag))` with the tag `protocol.tag() ‖ LE32(|context|) ‖ context`.
    /// Fails with [`Error::Parameter`] for a context of $`2^{32}`$ bytes or more.
    pub(crate) fn new(protocol: Protocol, context: &[u8]) -> Result<Self, Error> {
        let len = u32::try_from(context.len()).map_err(|_| Error::Parameter("context length"))?;
        let mut tag = Vec::with_capacity(protocol.tag().len() + 4 + context.len());
        tag.extend_from_slice(protocol.tag());
        tag.extend_from_slice(&len.to_le_bytes());
        tag.extend_from_slice(context);
        let out = Self {
            sponge: DuplexSponge::new(&derive_session_id(&tag)),
        };
        #[cfg(test)]
        record::start(protocol, context);
        Ok(out)
    }

    /// Absorb bytes: part of the instance or of a prover message.
    pub(crate) fn absorb(&mut self, bytes: &[u8]) {
        #[cfg(test)]
        record::absorb(bytes);
        self.sponge.absorb(bytes);
    }

    /// Absorb the bytes held by `segments`, piece by piece: the sponge state of absorbing their
    /// joined bytes.
    pub(crate) fn absorb_segments(&mut self, segments: &Segments) {
        segments.for_each_piece(|piece| self.absorb(piece));
    }

    /// Absorb `SerializeVarLenString(bytes)`: $`\mathrm{LE32}(|bytes|)\,\|\,bytes`$. Fails
    /// with [`Error::Dimension`] for $`2^{32}`$ bytes or more.
    pub(crate) fn absorb_varlen(&mut self, bytes: &[u8]) -> Result<(), Error> {
        self.absorb_len(bytes.len() as u64)?;
        self.absorb(bytes);
        Ok(())
    }

    /// [`Transcript::absorb_varlen`] of the bytes held by `segments`.
    pub(crate) fn absorb_varlen_segments(&mut self, segments: &Segments) -> Result<(), Error> {
        self.absorb_len(segments.len())?;
        self.absorb_segments(segments);
        Ok(())
    }

    /// Absorb a length or a count as `LE32`. Fails with [`Error::Dimension`] from $`2^{32}`$ on.
    pub(crate) fn absorb_len(&mut self, len: u64) -> Result<(), Error> {
        let len = u32::try_from(len).map_err(|_| Error::Dimension)?;
        self.absorb(&len.to_le_bytes());
        Ok(())
    }

    /// `Squeeze(len(out))` into `out`.
    pub(crate) fn squeeze(&mut self, out: &mut [u8]) {
        #[cfg(test)]
        record::squeeze(out.len());
        self.sponge.squeeze(out);
    }

    /// `count` integers modulo `modulus` by `DecodeUint`, each from $`N_s+16`$ squeezed bytes,
    /// in order. Fails with [`Error::Parameter`] for a modulus below 2 or above $`2^{256}-1`$,
    /// and with [`Error::Dimension`] for a count above $`2^{24}`$.
    pub(crate) fn uints(&mut self, modulus: &U256, count: usize) -> Result<Vec<U256>, Error> {
        let width = ns(modulus).ok_or(Error::Parameter("decoding modulus"))? + 16;
        if count > 1 << 24 {
            return Err(Error::Dimension);
        }
        let narrow = int::to_u64(modulus).map(Barrett64::new);
        let mut out = Vec::with_capacity(count);
        // Squeezed in pieces of at most 64 KiB: consecutive squeezes continue one stream, so the
        // bytes are those of one squeeze of the whole length.
        let per_piece = (65536 / width).max(1);
        let mut buf = vec![0u8; per_piece * width];
        while out.len() < count {
            let n = per_piece.min(count - out.len());
            let piece = &mut buf[..n * width];
            self.squeeze(piece);
            for chunk in piece.chunks_exact(width) {
                out.push(match narrow {
                    Some(m) => U256::from_u64(m.reduce_le(chunk)),
                    None => {
                        decode_uint(chunk, modulus).ok_or(Error::Parameter("decoding modulus"))?
                    }
                });
            }
        }
        Ok(out)
    }

    /// `count` polynomials of `ring`, each coefficient by `DecodeUint` modulo $`q`$
    /// ([`Transcript::uints`]), in order.
    pub(crate) fn ring_elements(
        &mut self,
        ring: &Arc<Ring>,
        count: usize,
    ) -> Result<Vec<Poly>, Error> {
        let d = ring.degree();
        let values = self.uints(
            &ring.modulus(),
            count.checked_mul(d).ok_or(Error::Dimension)?,
        )?;
        Ok(values
            .chunks_exact(d)
            .map(|c| Poly::from_canonical(ring.clone(), c.to_vec()))
            .collect())
    }

    /// The $`\lceil count/4\rceil`$ squeezed bytes that hold `count` entries of
    /// $`\{-1,0,1\}`$, 2 bits each ([`ternary_entry`] decodes entry $`i`$). Fails with
    /// [`Error::Dimension`] above $`2^{32}`$ entries.
    pub(crate) fn ternary_packed(&mut self, count: usize) -> Result<Vec<u8>, Error> {
        if count > 1 << 32 {
            return Err(Error::Dimension);
        }
        let mut bytes = vec![0u8; count.div_ceil(4)];
        self.squeeze(&mut bytes);
        Ok(bytes)
    }

    /// `count` entries of $`\{-1,0,1\}`$ from $`\lceil count/4\rceil`$ squeezed bytes
    /// ([`Transcript::ternary_packed`], [`ternary_entry`]).
    #[cfg(test)]
    pub(crate) fn ternary(&mut self, count: usize) -> Result<Vec<i8>, Error> {
        let bytes = self.ternary_packed(count)?;
        Ok((0..count).map(|i| ternary_entry(&bytes, i)).collect())
    }

    /// The challenge polynomial of the set with $`\omega`$ and $`\eta`$ over `ring`, from
    /// [`CHALLENGE_BYTES`] squeezed bytes ([`crate::rand::challenge_from_bytes`]).
    pub(crate) fn challenge(
        &mut self,
        ring: Arc<Ring>,
        omega: i128,
        eta: u64,
    ) -> Result<Poly, Error> {
        let mut bytes = [0u8; CHALLENGE_BYTES];
        self.squeeze(&mut bytes);
        crate::rand::challenge_from_bytes(&bytes, ring, omega, eta)
    }

    /// Test only: the next 32 bytes a squeeze would return, from a copy, which tells transcript
    /// states apart (also by the position of their output streams).
    #[cfg(test)]
    pub(crate) fn peek(&self) -> [u8; 32] {
        self.sponge.clone().squeeze_array()
    }

    /// 32 bytes that bind a prover's keys to the transcript so far: squeezed from a copy of the
    /// sponge after absorbing the label `JALI-V02-PROVER-KEY-BINDING`, so the transcript itself
    /// is unchanged. They are a function of public data, which the verifier could compute too.
    pub(crate) fn binding(&self) -> [u8; 32] {
        let mut sponge = self.sponge.clone();
        sponge.absorb(BINDING_LABEL);
        sponge.squeeze_array()
    }
}

/// Entry $`i`$ of packed ternary bytes: $`b_{2i}-b_{2i+1}`$ for the bits $`b`$ of the bytes,
/// least significant first, the centred binomial decoding `SamplePolyCBD` of FIPS 203 with
/// $`\eta=1`$ ($`P(0)=1/2`$, $`P(\pm1)=1/4`$ on uniform bytes).
pub(crate) fn ternary_entry(bytes: &[u8], i: usize) -> i8 {
    let byte = bytes[i / 4] >> (2 * (i % 4));
    (byte & 1) as i8 - ((byte >> 1) & 1) as i8
}

/// $`\mathrm{LE2IP}(buf)\bmod m`$ for a nonzero 64-bit $`m`$, by Horner's rule on the 64-bit
/// words of `buf` (bytes $`8j`$ to $`8j+7`$ for word $`j`$, the last one possibly shorter) from
/// the most significant one: the `DecodeUint` value for moduli below $`2^{64}`$.
#[cfg(test)]
pub(crate) fn reduce_le(buf: &[u8], m: u64) -> u64 {
    Barrett64::new(m).reduce_le(buf)
}

/// Remainders modulo a 64-bit $`m\ge2`$ of 128-bit values, by Barrett's reduction with
/// $`\mu=\lfloor(2^{128}-1)/m\rfloor`$.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Barrett64 {
    m: u128,
    mu: u128,
}
impl Barrett64 {
    pub(crate) fn new(m: u64) -> Self {
        assert!(m >= 2, "Barrett64 needs a modulus of at least 2");
        let m = u128::from(m);
        Self {
            m,
            mu: u128::MAX / m,
        }
    }
    /// $`x\bmod m`$. As $`\mu\le2^{128}/m`$, the estimate $`\hat q=\lfloor x\mu/2^{128}\rfloor`$
    /// is at most $`\lfloor x/m\rfloor`$. From $`\mu>2^{128}/m-2`$ and $`x<2^{128}`$ it is
    /// above $`x/m-3`$, so $`x-\hat qm<4m`$ and at most three subtractions follow.
    #[inline]
    pub(crate) fn reduce(&self, x: u128) -> u128 {
        let q = crate::math::mod128::mul_wide(x, self.mu).1;
        let mut r = x - q * self.m;
        while r >= self.m {
            r -= self.m;
        }
        r
    }
    /// $`\mathrm{LE2IP}(buf)\bmod m`$ by Horner's rule on the 64-bit words from the most
    /// significant one ([`reduce_le`]).
    pub(crate) fn reduce_le(&self, buf: &[u8]) -> u64 {
        let mut r = 0u128;
        for word in buf.chunks(8).rev() {
            let mut le = [0u8; 8];
            le[..word.len()].copy_from_slice(word);
            r = self.reduce((r << 64) | u128::from(u64::from_le_bytes(le)));
        }
        r as u64
    }
}
