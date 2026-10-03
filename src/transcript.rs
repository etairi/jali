//! Length-framed, domain-separated SHAKE128 chaining for the default transcript.
//! The protocol layer must absorb messages before deriving their dependent challenges.
use crate::codec::Segments;
use shake::{ExtendableOutput, Shake128, Update, XofReader};

#[cfg(test)]
mod tests;

/// A 32-byte chaining state. Does not expose secret prover randomness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transcript {
    state: [u8; 32],
}

/// SHAKE128 after absorbing `parts` in order.
fn absorbed(parts: &[&[u8]]) -> Shake128 {
    let mut h = Shake128::default();
    for part in parts {
        h.update(part);
    }
    h
}
fn squeeze(h: Shake128) -> [u8; 32] {
    let mut out = [0u8; 32];
    h.finalize_xof().read(&mut out);
    out
}
fn hash(parts: &[&[u8]]) -> [u8; 32] {
    squeeze(absorbed(parts))
}
impl Transcript {
    /// Bind a protocol version, the full parameter encoding, public seed, and statement.
    pub fn new(protocol: &[u8], params: &[u8], pp_seed: &[u8; 32], statement: &[u8]) -> Self {
        let mut out = Self {
            state: hash(&[b"LNP22-RUST-v1"]),
        };
        out.absorb(b"protocol", protocol);
        out.absorb(b"parameters", params);
        out.absorb(b"pp-seed", pp_seed);
        out.absorb(b"statement", statement);
        out
    }
    /// Hash a message with lengths of both its label and payload.
    pub fn absorb(&mut self, label: &[u8], payload: &[u8]) {
        self.state = hash(&[
            &self.state,
            &(label.len() as u64).to_le_bytes(),
            label,
            &(payload.len() as u64).to_le_bytes(),
            payload,
        ]);
    }
    /// [`Transcript::absorb`] of a payload held as [`Segments`]: the same state as absorbing
    /// its bytes, which SHAKE128 takes piece by piece instead of joined.
    pub(crate) fn absorb_segments(&mut self, label: &[u8], payload: &Segments) {
        let mut h = absorbed(&[
            &self.state,
            &(label.len() as u64).to_le_bytes(),
            label,
            &payload.len().to_le_bytes(),
        ]);
        payload.for_each_piece(|piece| h.update(piece));
        self.state = squeeze(h);
    }
    /// Derive a fresh challenge seed with an explicit domain; leaves the prefix unchanged.
    pub fn challenge_seed(&self, label: &[u8]) -> [u8; 32] {
        hash(&[
            &self.state,
            b"challenge",
            &(label.len() as u64).to_le_bytes(),
            label,
        ])
    }
    /// Current digest, for reproducibility and transcript test vectors.
    pub fn digest(&self) -> [u8; 32] {
        self.state
    }
}
