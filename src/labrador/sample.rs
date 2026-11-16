//! The public randomness of LaBRADOR: the commitment matrices, expanded from the public seed,
//! and the Johnson–Lindenstrauss matrices and challenge polynomials, expanded from seeds that
//! the transcript squeezes. Every stream is AES-256-CTR ([`AesPrg`]) under its own domain:
//! the high 32 bits name the purpose, the low 32 bits the level (or the challenge index).
use super::ring::{Rq, Sparse};
use crate::{
    Error,
    math::{Poly, Ring},
    rand::{AesPrg, ByteStream, domain, within_eta},
};
use std::sync::Arc;

/// The purposes of the streams.
pub(crate) mod purpose {
    /// The inner commitment matrix $`A`$ of a level.
    pub(crate) const MATRIX_A: u32 = 0x4c42_4141;
    /// The first outer commitment matrix $`(B\mid C)`$ of a level.
    pub(crate) const MATRIX_B: u32 = 0x4c42_4242;
    /// The second outer commitment matrix $`D`$ of a level.
    pub(crate) const MATRIX_D: u32 = 0x4c42_4444;
    /// The projection matrices of a level.
    pub(crate) const PROJECTION: u32 = 0x4c42_4a4c;
    /// The amortization challenges of a level, one stream per challenge.
    pub(crate) const CHALLENGE: u32 = 0x4c42_4348;
}

/// A buffered byte stream over AES-256-CTR.
struct Bytes {
    prg: AesPrg,
    buf: Vec<u8>,
    pos: usize,
}

impl Bytes {
    fn new(seed: &[u8; 32], domain: u64, buffer: usize) -> Self {
        Self {
            prg: AesPrg::new(seed, domain),
            buf: vec![0; buffer],
            pos: buffer,
        }
    }

    /// The reader of `new(seed, domain, buffer)` after `blocks` whole buffers, for streams whose
    /// reads fill each buffer exactly: the keystream from byte `blocks * buffer` on.
    fn at(seed: &[u8; 32], domain: u64, buffer: usize, blocks: u64) -> Result<Self, Error> {
        let offset = blocks.checked_mul(buffer as u64).ok_or(Error::Randomness)?;
        Ok(Self {
            prg: AesPrg::at(seed, domain, offset)?,
            buf: vec![0; buffer],
            pos: buffer,
        })
    }

    /// The next `n` bytes, `n` at most the buffer length. A read that does not fit in the rest
    /// of the buffer refills it and skips that rest, in the same way for every reader.
    fn take(&mut self, n: usize) -> Result<&[u8], Error> {
        if self.pos + n > self.buf.len() {
            self.prg.fill(&mut self.buf)?;
            self.pos = 0;
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
}

/// Uniform residues modulo $`q`$: little-endian words of $`\lceil\log_2q\rceil`$ bits, read
/// from whole bytes, the words at or above $`q`$ skipped.
pub(crate) struct Uniform {
    bytes: Bytes,
    q: u64,
    width: usize,
    mask: u64,
}

impl Uniform {
    pub(crate) fn new(seed: &[u8; 32], purpose: u32, index: u32, q: u64) -> Self {
        let bits = 64 - (q - 1).leading_zeros();
        let width = bits.div_ceil(8) as usize;
        Self {
            bytes: Bytes::new(seed, domain(purpose, index), width * 4096),
            q,
            width,
            mask: if bits == 64 {
                u64::MAX
            } else {
                (1 << bits) - 1
            },
        }
    }

    /// The next residue.
    pub(crate) fn next(&mut self) -> Result<u64, Error> {
        loop {
            let mut le = [0u8; 8];
            le[..self.width].copy_from_slice(self.bytes.take(self.width)?);
            let v = u64::from_le_bytes(le) & self.mask;
            if v < self.q {
                return Ok(v);
            }
        }
    }

    /// Fill `out` with the next residues.
    pub(crate) fn fill(&mut self, out: &mut [u64]) -> Result<(), Error> {
        for x in out {
            *x = self.next()?;
        }
        Ok(())
    }
}

/// The columns of the projection matrices: for each witness coefficient in order, the 256
/// entries of its column, 2 bits each with the centred binomial decoding of FIPS 203 for
/// $`\eta=1`$ (the entry is $`b_0-b_1`$, so $`P(0)=1/2`$ and $`P(\pm1)=1/4`$), from 64 bytes.
/// The bytes of one buffer of the projection stream.
const PROJECTION_BUFFER: usize = 64 * 1024;
/// The columns of one buffer of the projection stream, a block of [`Projection::at`].
pub(crate) const PROJECTION_BLOCK: usize = PROJECTION_BUFFER / 64;

pub(crate) struct Projection {
    bytes: Bytes,
}

impl Projection {
    /// Test only: the whole stream from its first column, the reference of [`Projection::at`].
    #[cfg(test)]
    pub(crate) fn new(seed: &[u8; 32], level: u32) -> Self {
        Self {
            bytes: Bytes::new(seed, domain(purpose::PROJECTION, level), PROJECTION_BUFFER),
        }
    }

    /// The stream of [`Projection::new`] from column `block` [`PROJECTION_BLOCK`] on: each
    /// column takes 64 bytes and the buffer holds a whole number of them, so column $`c`$ is at
    /// keystream byte $`64c`$ and the block starts on a buffer of its own.
    pub(crate) fn at(seed: &[u8; 32], level: u32, block: usize) -> Result<Self, Error> {
        let domain = domain(purpose::PROJECTION, level);
        Ok(Self {
            bytes: Bytes::at(seed, domain, PROJECTION_BUFFER, block as u64)?,
        })
    }

    /// The next column as its 64 bytes: entry $`4g+e`$ is in bits $`2e`$ and $`2e+1`$ of
    /// byte $`g`$.
    pub(crate) fn column_bytes(&mut self, out: &mut [u8; 64]) -> Result<(), Error> {
        out.copy_from_slice(self.bytes.take(64)?);
        Ok(())
    }

    /// The next column: 256 entries in $`\{-1,0,1\}`$.
    pub(crate) fn column(&mut self, out: &mut [i8; 256]) -> Result<(), Error> {
        let bytes = self.bytes.take(64)?;
        for (j, entry) in out.iter_mut().enumerate() {
            let byte = bytes[j / 4] >> (2 * (j % 4));
            *entry = (byte & 1) as i8 - ((byte >> 1) & 1) as i8;
        }
        Ok(())
    }
}

/// A challenge set of BS23 §2: polynomials with `ones` coefficients $`\pm1`$, `twos`
/// coefficients $`\pm2`$ and zeros elsewhere, whose operator norm is at most `op_bound`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ChallengeSet {
    pub(crate) d: usize,
    pub(crate) ones: usize,
    pub(crate) twos: usize,
    /// $`T`$: every challenge has operator norm at most $`T`$.
    pub(crate) op_bound: u64,
}

/// The draws a challenge may take before the sampler gives up.
pub(crate) const MAX_CHALLENGE_DRAWS: u32 = 4096;

impl ChallengeSet {
    /// The set for degree `d`. Degree 64 is the set of BS23 §2: 31 coefficients $`\pm1`$ and
    /// 10 coefficients $`\pm2`$ ($`\|c\|_2^2=71`$, about $`2^{128.09}`$ elements before the
    /// operator-norm filter), with operator norm at most 15.
    pub(crate) fn for_degree(d: usize) -> Result<Self, Error> {
        match d {
            64 => Ok(Self {
                d,
                ones: 31,
                twos: 10,
                op_bound: 15,
            }),
            _ => Err(Error::Parameter(
                "no LaBRADOR challenge set for this degree",
            )),
        }
    }

    /// $`\|c\|_2^2`$ of every challenge (BS23 §5.4 calls it $`\tau`$).
    pub(crate) fn l2_squared(&self) -> u64 {
        (self.ones + 4 * self.twos) as u64
    }

    /// The challenge of index `index` from the stream of `seed`: the first draw whose operator
    /// norm passes the exact integer test of [`within_eta`] for $`\eta=T`$, which accepts
    /// only polynomials with $`\|c\|_{op}\le T`$ (its module documentation gives the
    /// argument). A draw places the $`\pm2`$ and then the $`\pm1`$ coefficients at the first
    /// positions of a Fisher–Yates shuffle, and takes the signs from the next bytes. Fails with
    /// [`Error::Randomness`] after [`MAX_CHALLENGE_DRAWS`] rejected draws. `eta_ring` is any
    /// ring of degree $`d`$: the test reads only the centred coefficients.
    pub(crate) fn sample(
        &self,
        seed: &[u8; 32],
        index: u32,
        eta_ring: &Arc<Ring>,
    ) -> Result<Sparse, Error> {
        self.sample_counting(seed, index, eta_ring).map(|(c, _)| c)
    }

    /// [`ChallengeSet::sample`] and the number of draws it took.
    pub(crate) fn sample_counting(
        &self,
        seed: &[u8; 32],
        index: u32,
        eta_ring: &Arc<Ring>,
    ) -> Result<(Sparse, u32), Error> {
        let d = self.d;
        let weight = self.ones + self.twos;
        let mut bytes = Bytes::new(seed, domain(purpose::CHALLENGE, index), 1024);
        for draw in 1..=MAX_CHALLENGE_DRAWS {
            let mut positions: Vec<usize> = (0..d).collect();
            for i in 0..weight {
                let range = d - i;
                let limit = 256 - 256 % range;
                let j = loop {
                    let b = usize::from(bytes.take(1)?[0]);
                    if b < limit {
                        break i + b % range;
                    }
                };
                positions.swap(i, j);
            }
            let signs = bytes.take(weight.div_ceil(8))?;
            let mut terms: Vec<(usize, i64)> = (0..weight)
                .map(|idx| {
                    let magnitude = if idx < self.twos { 2 } else { 1 };
                    let negative = (signs[idx / 8] >> (idx % 8)) & 1 == 1;
                    (
                        positions[idx],
                        if negative { -magnitude } else { magnitude },
                    )
                })
                .collect();
            terms.sort_unstable();
            let mut coeffs = vec![0i128; d];
            for &(pos, value) in &terms {
                coeffs[pos] = i128::from(value);
            }
            let poly = Poly::new(eta_ring.clone(), coeffs)?;
            if within_eta(&poly, self.op_bound)? {
                return Ok((Sparse { terms }, draw));
            }
        }
        Err(Error::Randomness)
    }
}

/// The row-major entries of a public matrix of `rows` rows and `cols` polynomials each, as a
/// stream of polynomials.
pub(crate) struct Matrix {
    uniform: Uniform,
    d: usize,
}

impl Matrix {
    pub(crate) fn new(ring: &Rq, seed: &[u8; 32], purpose: u32, level: u32) -> Self {
        Self {
            uniform: Uniform::new(seed, purpose, level, ring.modulus()),
            d: ring.degree(),
        }
    }

    /// The next entry.
    pub(crate) fn next_poly(&mut self, out: &mut [u64]) -> Result<(), Error> {
        debug_assert_eq!(out.len(), self.d);
        self.uniform.fill(out)
    }
}

#[cfg(test)]
mod tests;
