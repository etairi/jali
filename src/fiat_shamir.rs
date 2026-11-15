//! The duplex sponge of draft-irtf-cfrg-fiat-shamir-04 over SHAKE128, its session identifiers
//! and the codecs of its Appendix A that the transcript uses.
//!
//! * [`DuplexSponge`] is the XOF duplex sponge of §4.2 with the suite SHAKE128 of §9.1 (rate
//!   $`R = 168`$): `Init(session_id)` absorbs `session_id ‖ zeros(R - 32)`, `Absorb` appends
//!   with no separator (an empty absorb is a no-op), and consecutive squeezes continue one output
//!   stream, which a non-empty absorb restarts over everything absorbed so far.
//! * [`derive_session_id`] is `DeriveSessionID` of §5.1.
//! * [`decode_uint`] is `DecodeUint` of Appendix A.1.2 and [`ns`] its $`N_s`$. The
//!   serializations of Appendix A.2 and A.3 for integers and byte strings
//!   (`serialize_uint`, `deserialize_uint`, `serialize_varlen`, `deserialize_varlen`) are
//!   compiled for the tests, which run them against the draft's vectors. The transcript writes
//!   `SerializeVarLenString` itself (`Transcript::absorb_varlen`).
//!
//! The unit tests run the draft's SHAKE128 vectors (Appendix C), and the draft's sumcheck
//! example (Appendix B) against its NARG vectors.
use crypto_bigint::{NonZero, U256, U512};
use shake::{ExtendableOutput, Shake128, Shake128Reader, Update, XofReader};

#[cfg(test)]
mod tests;

/// The rate $`R`$ of SHAKE128 in bytes.
pub(crate) const RATE: usize = 168;
/// The 32-byte domain separator of `DeriveSessionID` (§5.1).
const SESSION_ID_DOMAIN: &[u8; 32] = b"irtf-cfrg-fiat-shamir/session-id";

/// The XOF duplex sponge of §4.2 over SHAKE128.
///
/// The state is the incremental SHAKE128 context of everything absorbed, and the reader of its
/// output stream since the last non-empty absorb, created at the first squeeze after it.
#[derive(Clone)]
pub(crate) struct DuplexSponge {
    absorbed: Shake128,
    reader: Option<Shake128Reader>,
}

impl DuplexSponge {
    /// `Init(session_id)`.
    pub(crate) fn new(session_id: &[u8; 32]) -> Self {
        let mut absorbed = Shake128::default();
        absorbed.update(session_id);
        absorbed.update(&[0u8; RATE - 32]);
        Self {
            absorbed,
            reader: None,
        }
    }

    /// `Absorb(x)`. Absorbing the empty string leaves the state unchanged.
    pub(crate) fn absorb(&mut self, x: &[u8]) {
        if !x.is_empty() {
            self.absorbed.update(x);
            self.reader = None;
        }
    }

    /// `Squeeze(len(out))` into `out`.
    pub(crate) fn squeeze(&mut self, out: &mut [u8]) {
        let absorbed = &self.absorbed;
        self.reader
            .get_or_insert_with(|| absorbed.clone().finalize_xof())
            .read(out);
    }

    /// `Squeeze(N)` as an array.
    pub(crate) fn squeeze_array<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        self.squeeze(&mut out);
        out
    }
}

/// `DeriveSessionID(tag)`: the 32 bytes squeezed after absorbing `tag` into the sponge started
/// with the domain separator `irtf-cfrg-fiat-shamir/session-id`.
pub(crate) fn derive_session_id(tag: &[u8]) -> [u8; 32] {
    let mut sponge = DuplexSponge::new(SESSION_ID_DOMAIN);
    sponge.absorb(tag);
    sponge.squeeze_array()
}

/// $`N_s`$: the least number of bytes with $`256^{N_s} \ge M`$. `None` for $`M < 2`$.
pub(crate) fn ns(modulus: &U256) -> Option<usize> {
    if *modulus < U256::from_u8(2) {
        return None;
    }
    // 256^k >= M  <=>  8k >= bits(M - 1)
    Some((modulus.wrapping_sub(&U256::ONE).bits_vartime() as usize).div_ceil(8))
}

/// `DecodeUint(buf, M)`: `LE2IP(buf) mod M`, for a `buf` of at most 64 bytes (the caller
/// squeezes $`N_s + 16`$ of them, at most 48 for a modulus below $`2^{256}`$). `None` for a
/// longer `buf` or $`M < 2`$.
pub(crate) fn decode_uint(buf: &[u8], modulus: &U256) -> Option<U256> {
    if buf.len() > 64 || *modulus < U256::from_u8(2) {
        return None;
    }
    let mut le = [0u8; 64];
    le[..buf.len()].copy_from_slice(buf);
    let x = U512::from_le_slice(&le);
    let m = NonZero::new(modulus.resize::<{ U512::LIMBS }>()).into_option()?;
    Some(x.rem_vartime(&m).resize::<{ U256::LIMBS }>())
}

/// `SerializeUint(x, M)`: the $`N_s`$-byte little-endian representative of `x` in $`[0, M)`$.
/// `None` if `x ≥ M` or $`M < 2`$.
#[cfg(test)]
pub(crate) fn serialize_uint(x: &U256, modulus: &U256) -> Option<Vec<u8>> {
    let n = ns(modulus)?;
    if x >= modulus {
        return None;
    }
    Some(x.to_le_bytes()[..n].to_vec())
}

/// `DeserializeUint(input, M)`: the integer of the next $`N_s`$ bytes, read little-endian, and
/// the unread remainder. `None` on a short input or a value `≥ M`.
#[cfg(test)]
pub(crate) fn deserialize_uint<'a>(input: &'a [u8], modulus: &U256) -> Option<(U256, &'a [u8])> {
    let n = ns(modulus)?;
    if input.len() < n {
        return None;
    }
    let mut le = [0u8; 32];
    le[..n].copy_from_slice(&input[..n]);
    let x = U256::from_le_slice(&le);
    if &x >= modulus {
        return None;
    }
    Some((x, &input[n..]))
}

/// `SerializeVarLenString(s)`: `LE(len(s), 4) ‖ s`. `None` for a string of `2^32` bytes or more.
#[cfg(test)]
pub(crate) fn serialize_varlen(s: &[u8]) -> Option<Vec<u8>> {
    let len = u32::try_from(s.len()).ok()?;
    let mut out = Vec::with_capacity(4 + s.len());
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(s);
    Some(out)
}

/// `DeserializeVarLenString(input)`: the string after a 4-byte little-endian length, and the
/// unread remainder. `None` if the input is shorter than the length it states. The length is
/// compared with the remaining input, so no addition can overflow.
#[cfg(test)]
pub(crate) fn deserialize_varlen(input: &[u8]) -> Option<(&[u8], &[u8])> {
    let (head, rest) = input.split_first_chunk::<4>()?;
    let n = usize::try_from(u32::from_le_bytes(*head)).ok()?;
    if rest.len() < n {
        return None;
    }
    Some(rest.split_at(n))
}
