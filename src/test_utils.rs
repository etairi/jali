//! Deterministic byte-stream capture and replay for independent sampler tests.
//! These helpers expose the random tape and belong only in tests.
use crate::{Error, rand::ByteStream};
use zeroize::Zeroizing;

/// A finite known-answer tape. Exhaustion fails without consuming any bytes.
pub struct ReplayStream<'a> {
    remaining: &'a [u8],
}
impl<'a> ReplayStream<'a> {
    /// Read a public test vector from its first byte.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { remaining: bytes }
    }
    /// Number of bytes not yet consumed.
    pub fn remaining(&self) -> usize {
        self.remaining.len()
    }
}
impl ByteStream for ReplayStream<'_> {
    fn fill(&mut self, output: &mut [u8]) -> Result<(), Error> {
        let (head, tail) = self
            .remaining
            .split_at_checked(output.len())
            .ok_or(Error::Randomness)?;
        output.copy_from_slice(head);
        self.remaining = tail;
        Ok(())
    }
}

/// A sampler stream recording its exact successful byte consumption.
pub struct RecordingStream<S> {
    inner: S,
    tape: Zeroizing<Vec<u8>>,
}
impl<S: ByteStream> RecordingStream<S> {
    /// Wrap a stream; the captured bytes must never be treated as secret afterwards.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            tape: Zeroizing::new(Vec::new()),
        }
    }
    /// Captured random bytes, for an independent known-answer oracle.
    pub fn tape(&self) -> &[u8] {
        &self.tape
    }
}
impl<S: ByteStream> ByteStream for RecordingStream<S> {
    fn fill(&mut self, output: &mut [u8]) -> Result<(), Error> {
        self.inner.fill(output)?;
        self.tape.extend_from_slice(output);
        Ok(())
    }
}
