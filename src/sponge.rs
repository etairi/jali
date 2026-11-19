//! SHAKE128 (FIPS 202) over the `keccak` crate's Keccak-f[1600] permutation (with the SHA3
//! instructions on aarch64 where present), in the crate so that it can choose the permutation
//! per processor. The output is SHAKE128's on every target, which the tests check against the
//! `shake` crate.
//!
//! Bytes go into and out of the lanes of the state directly, whole lanes where the position
//! allows and the bytes of one lane at the ends, so that a context and a reader hold only the
//! state, which their drops wipe.
use zeroize::Zeroize;

/// The rate of SHAKE128 in bytes, 21 lanes of the 25.
const RATE: usize = 168;

/// Keccak-f[1600] on `state`, lane $`i`$ the little-endian word $`x+5y=i`$ of FIPS 202.
#[inline]
fn permute(keccak: &keccak::Keccak, state: &mut [u64; 25]) {
    keccak.with_f1600(|f| f(state));
}

/// XOR `data` into the state from byte `pos` on, within the rate: the bytes up to a lane
/// boundary, whole lanes, then the bytes of the last lane.
#[inline]
fn xor_at(state: &mut [u64; 25], pos: usize, data: &[u8]) {
    let (mut lane, off) = (pos / 8, pos % 8);
    let mut data = data;
    if off != 0 {
        let n = (8 - off).min(data.len());
        let mut word = [0u8; 8];
        word[off..off + n].copy_from_slice(&data[..n]);
        state[lane] ^= u64::from_le_bytes(word);
        data = &data[n..];
        lane += 1;
    }
    let (words, tail) = data.as_chunks::<8>();
    for (s, w) in state[lane..lane + words.len()].iter_mut().zip(words) {
        *s ^= u64::from_le_bytes(*w);
    }
    if !tail.is_empty() {
        let mut word = [0u8; 8];
        word[..tail.len()].copy_from_slice(tail);
        state[lane + words.len()] ^= u64::from_le_bytes(word);
    }
}

/// The bytes of the state from byte `pos` on into `out`, within the rate, in the pieces of
/// [`xor_at`].
#[inline]
fn copy_at(state: &[u64; 25], pos: usize, out: &mut [u8]) {
    let (mut lane, off) = (pos / 8, pos % 8);
    let mut out = out;
    if off != 0 {
        let n = (8 - off).min(out.len());
        let (head, rest) = out.split_at_mut(n);
        head.copy_from_slice(&state[lane].to_le_bytes()[off..off + n]);
        out = rest;
        lane += 1;
    }
    let (words, tail) = out.as_chunks_mut::<8>();
    let whole = words.len();
    for (w, s) in words.iter_mut().zip(&state[lane..lane + whole]) {
        *w = s.to_le_bytes();
    }
    if !tail.is_empty() {
        let n = tail.len();
        tail.copy_from_slice(&state[lane + whole].to_le_bytes()[..n]);
    }
}

/// An incremental SHAKE128 context: [`Shake128::update`] absorbs, and
/// [`Shake128::finalize_xof`] pads and returns the reader of the output stream.
#[derive(Clone)]
pub(crate) struct Shake128 {
    state: [u64; 25],
    /// Bytes of the current block absorbed, below [`RATE`].
    pos: usize,
    keccak: keccak::Keccak,
}

impl Default for Shake128 {
    fn default() -> Self {
        Self {
            state: [0; 25],
            pos: 0,
            keccak: keccak::Keccak::new(),
        }
    }
}

impl Shake128 {
    /// Absorb `data`.
    pub(crate) fn update(&mut self, mut data: &[u8]) {
        while !data.is_empty() {
            let take = (RATE - self.pos).min(data.len());
            let (part, rest) = data.split_at(take);
            xor_at(&mut self.state, self.pos, part);
            self.pos += take;
            data = rest;
            if self.pos == RATE {
                permute(&self.keccak, &mut self.state);
                self.pos = 0;
            }
        }
    }

    /// Pad with SHAKE's domain bits (`1111`, then `10*1`) and return the output stream.
    pub(crate) fn finalize_xof(mut self) -> Shake128Reader {
        xor_at(&mut self.state, self.pos, &[0x1f]);
        xor_at(&mut self.state, RATE - 1, &[0x80]);
        permute(&self.keccak, &mut self.state);
        Shake128Reader {
            state: self.state,
            pos: 0,
            keccak: self.keccak,
        }
    }
}

impl Drop for Shake128 {
    fn drop(&mut self) {
        self.state.zeroize();
    }
}

/// The output stream of a finalized [`Shake128`].
#[derive(Clone)]
pub(crate) struct Shake128Reader {
    state: [u64; 25],
    /// Bytes of the current block read, at most [`RATE`].
    pos: usize,
    keccak: keccak::Keccak,
}

impl Shake128Reader {
    /// Fill `out` with the next bytes of the stream.
    pub(crate) fn read(&mut self, mut out: &mut [u8]) {
        while !out.is_empty() {
            if self.pos == RATE {
                permute(&self.keccak, &mut self.state);
                self.pos = 0;
            }
            let take = (RATE - self.pos).min(out.len());
            let (part, rest) = out.split_at_mut(take);
            copy_at(&self.state, self.pos, part);
            self.pos += take;
            out = rest;
        }
    }
}

impl Drop for Shake128Reader {
    fn drop(&mut self) {
        self.state.zeroize();
    }
}

#[cfg(test)]
mod tests;
