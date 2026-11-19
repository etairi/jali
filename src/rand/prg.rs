use crate::Error;
use crate::sponge::{Shake128, Shake128Reader};
use aes::Aes256;
use ctr::cipher::{KeyIvInit, StreamCipher, StreamCipherSeek};
use zeroize::Zeroize;

/// A deterministic byte stream with explicit failure on exhaustion.
pub trait ByteStream {
    /// Fill the entire destination with fresh bytes.
    fn fill(&mut self, output: &mut [u8]) -> Result<(), Error>;
}

/// $`(domain_{32}\ll32)\mathbin{|}index_{32}`$, independent of host byte order.
pub fn domain(domain: u32, index: u32) -> u64 {
    (u64::from(domain) << 32) | u64::from(index)
}

/// High words of the streams drawn from a secret key. Each call derives its keys from the
/// caller's seed and its own inputs (`rand::derive_key`), and each purpose reads its own words
/// under its key, so no two purposes share a keystream even if two keys were ever equal. Masks
/// of attempt `a` use the words `base + 2a` and `base + 2a + 1`.
pub(crate) mod secret {
    /// Attempts of a prover's rejection loop, which bound the mask words below.
    pub(crate) const MAX_ATTEMPTS: u32 = 4096;
    /// Commitment randomness $`s_2`$ of `Abdlop::commit_with_seed`.
    pub(crate) const COMMITMENT: u32 = 0;
    /// Commitment randomness $`s_2`$ of the toolbox's own commitment ("TCOM").
    pub(crate) const TOOLBOX_COMMITMENT: u32 = 0x5443_4f4d;
    /// Rejection coins of the opening proof.
    pub(crate) const OPENING_COINS: u32 = 3;
    /// Masks $`y_1`$ and $`y_2`$ of the opening proof.
    pub(crate) const OPENING_MASKS: u32 = 0x2000_0000;
    /// Masks $`y^{(e)}`$ and $`y^{(d)}`$ of the range proofs.
    pub(crate) const RANGE_MASKS: u32 = 0x1000_0000;
    /// Signs, rejection coins and the evaluation-proof seed of the toolbox ("TBOX").
    pub(crate) const TOOLBOX: u32 = 0x5442_4f58;
    /// Garbage polynomials of the evaluation proof ("GARB").
    pub(crate) const GARBAGE: u32 = 0x4741_5242;
    /// Randomness $`s_1,s_2,s_3`$ of the extractable mode's LinHC commitment, under the
    /// extractable toolbox key ("LCOM").
    pub(crate) const LINHC_COMMITMENT: u32 = 0x4c43_4f4d;
    /// LinHC masks of opening attempt `a`, words `base + 3a` to `base + 3a + 2`, under the
    /// extractable opening key.
    pub(crate) const LINHC_MASKS: u32 = 0x3000_0000;
    /// LinHC rejection coin of opening attempt `a`, the first 32 bytes of the stream
    /// `(LINHC_COINS, a)` under the extractable opening key.
    pub(crate) const LINHC_COINS: u32 = 5;
}

/// AES-256-CTR keystream under the seed, with IV `LE64(domain) || 0^8` and a big-endian
/// 128-bit counter.
///
/// Reads shorter than 256 bytes are served from a buffer of the next keystream bytes,
/// so that one call to the cipher covers many blocks. The bytes of the reads are those of the
/// keystream in order, whatever their sizes. The buffer is wiped on drop.
pub struct AesPrg {
    cipher: ctr::Ctr128BE<Aes256>,
    /// The next keystream bytes from `pos` on, which `fill` returns before any other.
    buffer: [u8; AES_BUFFER],
    pos: usize,
}
/// The keystream bytes that [`AesPrg`] computes ahead for short reads.
const AES_BUFFER: usize = 256;
impl Drop for AesPrg {
    fn drop(&mut self) {
        self.buffer.zeroize();
    }
}
impl core::fmt::Debug for AesPrg {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AesPrg").finish_non_exhaustive()
    }
}
impl AesPrg {
    /// Initialize from a seed and 64-bit domain.
    pub fn new(seed: &[u8; 32], domain: u64) -> Self {
        #[cfg(test)]
        stream_log::record(seed, domain);
        let mut iv = [0u8; 16];
        iv[..8].copy_from_slice(&domain.to_le_bytes());
        Self {
            cipher: ctr::Ctr128BE::<Aes256>::new(seed.into(), (&iv).into()),
            buffer: [0; AES_BUFFER],
            pos: AES_BUFFER,
        }
    }
    /// The stream of [`AesPrg::new`] from keystream byte `offset` on: what that stream returns
    /// after reads of `offset` bytes in total, whatever their sizes.
    pub(crate) fn at(seed: &[u8; 32], domain: u64, offset: u64) -> Result<Self, Error> {
        let mut prg = Self::new(seed, domain);
        prg.cipher.try_seek(offset).map_err(|_| Error::Randomness)?;
        Ok(prg)
    }
}
impl ByteStream for AesPrg {
    fn fill(&mut self, output: &mut [u8]) -> Result<(), Error> {
        // The buffered bytes come first. Once they are used up, the cipher stands right after
        // them, so a long read takes the keystream directly and a short one refills the buffer.
        let buffered = (AES_BUFFER - self.pos).min(output.len());
        output[..buffered].copy_from_slice(&self.buffer[self.pos..self.pos + buffered]);
        self.pos += buffered;
        let rest = &mut output[buffered..];
        if rest.is_empty() {
            return Ok(());
        }
        if rest.len() >= AES_BUFFER {
            rest.fill(0);
            return self
                .cipher
                .try_apply_keystream(rest)
                .map_err(|_| Error::Randomness);
        }
        self.buffer.fill(0);
        self.cipher
            .try_apply_keystream(&mut self.buffer)
            .map_err(|_| Error::Randomness)?;
        rest.copy_from_slice(&self.buffer[..rest.len()]);
        self.pos = rest.len();
        Ok(())
    }
}

/// A stream read in batches of `N` bytes for its short reads: the bytes of `inner` in order,
/// whatever the read sizes. A read of `N` bytes or more takes the inner stream directly. The
/// batch read ahead leaves `inner` past the bytes returned, so the wrapper is for streams that
/// are dropped with it, such as a mask's AES stream (on x86 `aes` encrypts 64 blocks at a time
/// with VAES-512, 1 KiB, while `AesPrg` refills 256 bytes for short reads). The bytes read
/// ahead are wiped on drop.
#[cfg(any(target_arch = "x86", target_arch = "x86_64", test))]
pub(crate) struct ReadAhead<S, const N: usize> {
    inner: S,
    bytes: [u8; N],
    pos: usize,
    len: usize,
    used: usize,
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64", test))]
impl<S: ByteStream, const N: usize> ReadAhead<S, N> {
    pub(crate) fn new(inner: S) -> Self {
        Self {
            inner,
            bytes: [0; N],
            pos: 0,
            len: 0,
            used: 0,
        }
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64", test))]
impl<S: ByteStream, const N: usize> ByteStream for ReadAhead<S, N> {
    fn fill(&mut self, output: &mut [u8]) -> Result<(), Error> {
        let buffered = (self.len - self.pos).min(output.len());
        output[..buffered].copy_from_slice(&self.bytes[self.pos..self.pos + buffered]);
        self.pos += buffered;
        let rest = &mut output[buffered..];
        if rest.is_empty() {
            return Ok(());
        }
        if rest.len() >= N {
            return self.inner.fill(rest);
        }
        self.inner.fill(&mut self.bytes)?;
        rest.copy_from_slice(&self.bytes[..rest.len()]);
        (self.pos, self.len, self.used) = (rest.len(), N, N);
        Ok(())
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64", test))]
impl<S, const N: usize> Drop for ReadAhead<S, N> {
    fn drop(&mut self) {
        self.bytes[..self.used].zeroize();
    }
}

/// Alternative SHAKE128 stream over `seed || LE64(domain)`.
pub struct ShakePrg(Shake128Reader);
impl core::fmt::Debug for ShakePrg {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ShakePrg").finish_non_exhaustive()
    }
}
impl ShakePrg {
    /// Initialize from a seed and 64-bit domain.
    pub fn new(seed: &[u8; 32], domain: u64) -> Self {
        #[cfg(test)]
        stream_log::record(seed, domain);
        let mut h = Shake128::default();
        h.update(seed);
        h.update(&domain.to_le_bytes());
        Self(h.finalize_xof())
    }
}
impl ByteStream for ShakePrg {
    fn fill(&mut self, output: &mut [u8]) -> Result<(), Error> {
        self.0.read(output);
        Ok(())
    }
}

/// Test only: the streams opened with a watched seed, or with a key or seed that descends from
/// one, across all threads.
#[cfg(test)]
pub(crate) mod stream_log {
    use std::sync::{Mutex, MutexGuard};

    /// One stream opened with a watched key.
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct Stream {
        pub(crate) key: [u8; 32],
        pub(crate) domain: u64,
        /// Whether `key` is an output of `derive_key`, rather than a seed.
        pub(crate) derived: bool,
    }

    /// A watched seed, what descends from it, and the streams opened with any of those.
    struct Watch {
        /// The seed first, then derived keys and child seeds.
        keys: Vec<[u8; 32]>,
        /// The outputs of `derive_key` among `keys`.
        derived: Vec<[u8; 32]>,
        streams: Vec<Stream>,
    }
    static WATCHED: Mutex<Vec<Watch>> = Mutex::new(Vec::new());

    fn watched() -> MutexGuard<'static, Vec<Watch>> {
        WATCHED.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn record(key: &[u8; 32], domain: u64) {
        for watch in watched().iter_mut().filter(|w| w.keys.contains(key)) {
            let derived = watch.derived.contains(key);
            watch.streams.push(Stream {
                key: *key,
                domain,
                derived,
            });
        }
    }

    fn watch_descendant(parent: &[u8; 32], child: &[u8; 32], derived: bool) {
        for watch in watched().iter_mut().filter(|w| w.keys.contains(parent)) {
            if !watch.keys.contains(child) {
                watch.keys.push(*child);
            }
            if derived && !watch.derived.contains(child) {
                watch.derived.push(*child);
            }
        }
    }

    /// Watch `key` wherever `seed` is watched: `derive_key` returned it for `seed`.
    pub(crate) fn derived(seed: &[u8; 32], key: &[u8; 32]) {
        watch_descendant(seed, key, true);
    }

    /// Watch `child` wherever `parent` is watched: a seed read from one of its streams.
    pub(crate) fn child(parent: &[u8; 32], child: &[u8; 32]) {
        watch_descendant(parent, child, false);
    }

    /// Run `f` and return every stream opened meanwhile with `seed` or with a key or seed that
    /// descends from it. Each caller must watch a seed that no concurrently running test uses.
    pub(crate) fn opened_with(seed: [u8; 32], f: impl FnOnce()) -> Vec<Stream> {
        watched().push(Watch {
            keys: vec![seed],
            derived: Vec::new(),
            streams: Vec::new(),
        });
        f();
        let mut watched = watched();
        let i = watched.iter().position(|w| w.keys[0] == seed).unwrap();
        watched.remove(i).streams
    }
}

#[cfg(test)]
mod tests {
    use super::ByteStream;
    /// The keystream through reads of every size around the buffer's, against one long read and
    /// against the cipher itself: the same bytes in the same order.
    #[test]
    fn readers_at_an_offset_continue_the_stream() {
        use super::AesPrg;
        let seed = [11u8; 32];
        let mut all = vec![0u8; 200_000];
        AesPrg::new(&seed, 7).fill(&mut all).unwrap();
        for offset in [
            0usize,
            1,
            15,
            16,
            17,
            255,
            256,
            4096,
            65536,
            3 * 65536 + 64,
            150_001,
        ] {
            let mut prg = AesPrg::at(&seed, 7, offset as u64).unwrap();
            let mut got = vec![0u8; all.len() - offset];
            // Short and long reads, as the buffered reader serves them.
            let (head, tail) = got.split_at_mut(7.min(all.len() - offset));
            prg.fill(head).unwrap();
            prg.fill(tail).unwrap();
            assert_eq!(&got[..], &all[offset..], "offset {offset}");
        }
    }
    #[test]
    fn buffered_reads_return_the_keystream() {
        use ctr::cipher::{KeyIvInit, StreamCipher};
        for (seed, domain) in [([1u8; 32], 0u64), ([2u8; 32], 7), ([3u8; 32], u64::MAX)] {
            let total = 5000;
            let mut whole = vec![0u8; total];
            super::AesPrg::new(&seed, domain).fill(&mut whole).unwrap();
            let mut iv = [0u8; 16];
            iv[..8].copy_from_slice(&domain.to_le_bytes());
            let mut raw = vec![0u8; total];
            ctr::Ctr128BE::<aes::Aes256>::new((&seed).into(), (&iv).into())
                .apply_keystream(&mut raw);
            assert_eq!(whole, raw);
            for sizes in [
                vec![1usize, 7, 8, 24, 32],
                vec![255, 1, 256, 257, 3],
                vec![16, 240, 1, 255, 512, 2],
            ] {
                let mut prg = super::AesPrg::new(&seed, domain);
                let mut got = Vec::with_capacity(total);
                let mut k = 0;
                while got.len() < total {
                    let n = sizes[k % sizes.len()].min(total - got.len());
                    let mut buf = vec![0u8; n];
                    prg.fill(&mut buf).unwrap();
                    got.extend_from_slice(&buf);
                    k += 1;
                }
                assert_eq!(got, raw, "{sizes:?}");
            }
        }
    }
    #[test]
    fn read_ahead_returns_the_inner_stream() {
        // Reads of every size around the batch, against one long read of the same stream.
        let (seed, domain, total) = ([5u8; 32], 9u64, 7000);
        let mut whole = vec![0u8; total];
        super::AesPrg::new(&seed, domain).fill(&mut whole).unwrap();
        for sizes in [
            vec![1usize, 7, 8, 24, 32],
            vec![1023, 1, 1024, 1025, 600, 5],
            vec![255, 256, 257, 3, 2048],
        ] {
            let mut ahead = super::ReadAhead::<_, 1024>::new(super::AesPrg::new(&seed, domain));
            let mut got = Vec::with_capacity(total);
            let mut k = 0;
            while got.len() < total {
                let n = sizes[k % sizes.len()].min(total - got.len());
                let mut buf = vec![0u8; n];
                ahead.fill(&mut buf).unwrap();
                got.extend_from_slice(&buf);
                k += 1;
            }
            assert_eq!(got, whole, "{sizes:?}");
        }
    }
    use super::secret::*;

    #[test]
    fn secret_seed_purposes_own_disjoint_domain_words() {
        // Word ranges [start, end): masks take two words per attempt.
        let ranges = [
            (COMMITMENT, COMMITMENT + 1),
            (TOOLBOX_COMMITMENT, TOOLBOX_COMMITMENT + 1),
            (OPENING_COINS, OPENING_COINS + 1),
            (OPENING_MASKS, OPENING_MASKS + 2 * MAX_ATTEMPTS),
            (RANGE_MASKS, RANGE_MASKS + 2 * MAX_ATTEMPTS),
            (TOOLBOX, TOOLBOX + 1),
            (GARBAGE, GARBAGE + 1),
        ];
        for (i, a) in ranges.iter().enumerate() {
            for b in &ranges[i + 1..] {
                assert!(a.1 <= b.0 || b.1 <= a.0, "{a:?} overlaps {b:?}");
            }
        }
    }
}
