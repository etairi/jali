//! One level of the main protocol of BS23 (Fig. 2 and 3), made non-interactive with the
//! transcript: the prover's messages and the verifier's reduction to the next statement.
//!
//! Order of a level (`docs/labrador.md` gives the encodings): absorb the level's parameters
//! (55 bytes), absorb $`u_1`$, squeeze the 32-byte projection seed, absorb $`p`$, squeeze
//! $`\psi^{(k)},\omega^{(k)}`$ ($`k<\lambda`$, `DecodeUint` each), absorb the non-constant
//! coefficients of $`b''^{(k)}`$, squeeze $`\alpha,\beta`$ (uniform polynomials), absorb
//! $`u_2`$, squeeze the 32-byte challenge seed.
use super::{
    decompose::decompose,
    params::{Layout, LevelParams, PARAMS_BYTES, Policy, Shape, pair_index, projection_max},
    plan::{self, Stats, signed_width},
    recursion::{Next, next_witness},
    ring::{Rq, Sparse},
    sample::{ChallengeSet, Matrix, PROJECTION_BLOCK, Projection, purpose},
    statement::{Aggregate, Derived, Relation},
};
use crate::{
    Error,
    codec::{BitReader, BitWriter},
    math::{
        Ring, U256, int,
        ternary::{DECODE, PATTERN},
    },
    transcript::Transcript,
};
use std::sync::Arc;
use zeroize::Zeroizing;

/// The next statement and its witness.
pub(crate) type NextLevel = (Next, Vec<Zeroizing<Vec<u64>>>);

/// The attempts of a level before the prover gives up.
const MAX_ATTEMPTS: u8 = 64;

/// What every level of one proof shares.
pub(crate) struct Ctx {
    pub(crate) ring: Arc<Rq>,
    pub(crate) set: ChallengeSet,
    pub(crate) eta_ring: Arc<Ring>,
    pub(crate) policy: Policy,
    pub(crate) pp_seed: [u8; 32],
    pub(crate) lambda: usize,
}

/// The byte lengths of a level's messages and the width of the projection's entries.
struct Sizes {
    u1: usize,
    p: usize,
    p_width: u32,
    bpp: usize,
    u2: usize,
}

fn qbits(q: u64) -> u32 {
    64 - (q - 1).leading_zeros()
}

fn sizes(ctx: &Ctx, shape: &Shape, params: &LevelParams) -> Result<Sizes, Error> {
    let (q, d) = (ctx.ring.modulus(), ctx.ring.degree());
    let w = qbits(q) as usize;
    let p_width = signed_width(projection_max(shape.norm_sq)?);
    Ok(Sizes {
        u1: (params.kappa1 * d * w).div_ceil(8),
        p: (256 * p_width as usize).div_ceil(8),
        p_width,
        bpp: (ctx.lambda * (d - 1) * w).div_ceil(8),
        u2: (params.kappa2 * d * w).div_ceil(8),
    })
}

/// Residues modulo $`q`$ in $`\lceil\log_2q\rceil`$ bits each, least significant first, zero
/// padded to a byte.
pub(crate) fn encode_residues(q: u64, values: &[u64]) -> Result<Vec<u8>, Error> {
    let w = qbits(q);
    let mut writer = BitWriter::new();
    for x in values {
        writer.uniform(u128::from(*x), u128::from(q))?;
    }
    let pad = (8 - (values.len() * w as usize) % 8) % 8;
    writer.unsigned(0, pad as u32)?;
    writer.into_bytes()
}

/// The inverse of [`encode_residues`] for `count` residues: [`Error::Encoding`] for a value
/// at or above $`q`$, nonzero padding, or a length that is not exact.
pub(crate) fn decode_residues(q: u64, bytes: &[u8], count: usize) -> Result<Vec<u64>, Error> {
    let w = qbits(q) as usize;
    if bytes.len() != (count * w).div_ceil(8) {
        return Err(Error::Encoding);
    }
    let mut reader = BitReader::new(bytes);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(reader.uniform(u128::from(q))? as u64);
    }
    let pad = (8 - (count * w) % 8) % 8;
    if reader.unsigned(pad as u32)? != 0 {
        return Err(Error::Encoding);
    }
    reader.finish_bytes()?;
    Ok(out)
}

/// Signed values in the fixed-width code of [`BitWriter::signed`], zero padded to a byte.
pub(crate) fn encode_signed(values: &[i128], width: u32) -> Result<Vec<u8>, Error> {
    let mut writer = BitWriter::new();
    for x in values {
        writer.signed(*x, width)?;
    }
    let pad = (8 - (values.len() * width as usize) % 8) % 8;
    writer.unsigned(0, pad as u32)?;
    writer.into_bytes()
}

/// The inverse of [`encode_signed`] for `count` values.
pub(crate) fn decode_signed(bytes: &[u8], count: usize, width: u32) -> Result<Vec<i128>, Error> {
    if bytes.len() != (count * width as usize).div_ceil(8) {
        return Err(Error::Encoding);
    }
    let mut reader = BitReader::new(bytes);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(reader.signed(width)?);
    }
    let pad = (8 - (count * width as usize) % 8) % 8;
    if reader.unsigned(pad as u32)? != 0 {
        return Err(Error::Encoding);
    }
    reader.finish_bytes()?;
    Ok(out)
}

/// `count` residues modulo $`q`$ squeezed with `DecodeUint`.
pub(crate) fn squeeze_residues(
    t: &mut Transcript,
    q: u64,
    count: usize,
) -> Result<Vec<u64>, Error> {
    let modulus = U256::from_u64(q);
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let n = (count - out.len()).min(1 << 20);
        for x in t.uints(&modulus, n)? {
            out.push(int::to_u64(&x).ok_or(Error::Overflow)?);
        }
    }
    Ok(out)
}

/// `count` uniform polynomials.
pub(crate) fn squeeze_polys(
    t: &mut Transcript,
    ring: &Rq,
    count: usize,
) -> Result<Vec<Vec<u64>>, Error> {
    let d = ring.degree();
    Ok(squeeze_residues(t, ring.modulus(), count * d)?
        .chunks_exact(d)
        .map(<[u64]>::to_vec)
        .collect())
}

fn squeeze_seed(t: &mut Transcript) -> [u8; 32] {
    let mut seed = [0u8; 32];
    t.squeeze(&mut seed);
    seed
}

/// The next `n` bytes of the proof.
pub(crate) fn take<'a>(input: &mut &'a [u8], n: usize) -> Result<&'a [u8], Error> {
    if input.len() < n {
        return Err(Error::Encoding);
    }
    let (head, rest) = input.split_at(n);
    *input = rest;
    Ok(head)
}

/// $`u=M\cdot v`$ for the public matrix of `purpose` at `level`, `rows` rows, `v` of `cols`
/// polynomials given by its transforms.
fn matvec(
    ctx: &Ctx,
    purpose: u32,
    level: u32,
    rows: usize,
    v_hat: &[u64],
) -> Result<Vec<u64>, Error> {
    let ring = &ctx.ring;
    let (d, k) = (ring.degree(), ring.hat_len());
    let cols = v_hat.len() / k;
    let mut matrix = Matrix::new(ring, &ctx.pp_seed, purpose, level);
    let mut entry = vec![0u64; d];
    let mut entry_hat = vec![0u64; k];
    let mut out = vec![0u64; rows * d];
    for row in 0..rows {
        let mut acc = vec![0u64; k];
        for col in 0..cols {
            matrix.next_poly(&mut entry)?;
            ring.ntt_into(&entry, &mut entry_hat);
            ring.mac(&mut acc, &entry_hat, &v_hat[col * k..(col + 1) * k]);
        }
        ring.intt_into(&mut acc, &mut out[row * d..(row + 1) * d]);
    }
    Ok(out)
}

/// The projection $`p=\sum_i\Pi_is_i`$ of all witness coefficients, in vector order.
fn project(
    ring: &Rq,
    seed: &[u8; 32],
    level: u32,
    witness: &[Zeroizing<Vec<u64>>],
) -> Result<[i128; 256], Error> {
    // With every centred coefficient at most M in magnitude, a lane of i32 takes
    // floor((2^31 - 1)/M) steps of at most M before it could overflow, so the lanes are added
    // into the i128 sums that often. The same integers as summing in i128 throughout.
    let max = witness
        .iter()
        .flat_map(|v| v.iter())
        .map(|x| ring.center(*x).unsigned_abs())
        .max()
        .unwrap_or(0);
    let flush = (max <= i32::MAX as u64).then(|| (i32::MAX as u64 / max.max(1)) as usize);
    // Coefficients of at most 127 in magnitude (ternary witnesses, small digits) go through
    // 8-bit lanes, added into the sums every floor(127/M) steps.
    let byte_flush = (max <= i8::MAX as u64).then(|| (i8::MAX as u64 / max.max(1)) as usize);
    let total: usize = witness.iter().map(|v| v.len()).sum();
    let per_block = BLOCK_BUFFERS * PROJECTION_BLOCK;
    let blocks = total.div_ceil(per_block);
    // Block by block, each on its own reader, and the exact integer sums of the blocks.
    let mut p = [0i128; 256];
    for wave in (0..blocks).step_by(WAVE) {
        let parts = Zeroizing::new(crate::par::try_map(WAVE.min(blocks - wave), |w| {
            let block = wave + w;
            let range = block_range(block, per_block, total);
            let mut projection = Projection::at(seed, level, block * BLOCK_BUFFERS)?;
            let coefficients = coefficients(witness, range);
            match (byte_flush, flush) {
                (Some(flush), _) => project_bytes(ring, &mut projection, coefficients, flush),
                (None, Some(flush)) => project_lanes(ring, &mut projection, coefficients, flush),
                (None, None) => project_block_wide(ring, &mut projection, coefficients),
            }
        })?);
        for part in parts.iter() {
            for (t, x) in p.iter_mut().zip(part) {
                *t += *x;
            }
        }
    }
    Ok(p)
}

/// Blocks of the projection stream processed at once, in parallel with `parallel`: also the
/// most results held at a time.
const WAVE: usize = 64;

/// The coefficients `range` of the witness vectors put end to end, in order.
fn coefficients(
    witness: &[Zeroizing<Vec<u64>>],
    range: core::ops::Range<usize>,
) -> impl Iterator<Item = &u64> {
    let mut start = 0;
    witness.iter().flat_map(move |v| {
        let (s, e) = (start, start + v.len());
        start = e;
        v[range.start.clamp(s, e) - s..range.end.clamp(s, e) - s].iter()
    })
}

/// The projection of `coefficients` with the columns of `projection`, for coefficients of at
/// most 127 in magnitude: in 8-bit lanes added into 32-bit lanes every `flush` steps. Entry
/// $`e`$ of row $`4g+e`$ is bit $`2e`$ less bit $`2e+1`$ of byte $`g`$ ([`DECODE`]), so its
/// product with $`s`$ is $`s`$ masked by the first bit less $`s`$ masked by the second: no
/// multiplication, and the lanes of one entry position side by side, so that the loops run on
/// vectors. The same integers as [`project_lanes`].
fn project_bytes<'a>(
    ring: &Rq,
    projection: &mut Projection,
    mut coefficients: impl Iterator<Item = &'a u64>,
    flush: usize,
) -> Result<[i128; 256], Error> {
    // Lane g of `lanes[e]` and of `wide[e]` is row 4g + e. The byte lanes are added into the
    // 32-bit lanes after each batch, at most 127 in magnitude each time, and those into p
    // every `BATCHES` batches and at the end: the same sums as adding into p each time.
    const BATCHES: usize = (i32::MAX / 127) as usize;
    fn add_into(p: &mut [i128; 256], wide: &mut [[i32; 64]; 4]) {
        for (e, lane) in wide.iter_mut().enumerate() {
            for (g, a) in lane.iter_mut().enumerate() {
                p[4 * g + e] += i128::from(*a);
                *a = 0;
            }
        }
    }
    let mut p = [0i128; 256];
    let mut lanes = Zeroizing::new([[0i8; 64]; 4]);
    let mut wide = Zeroizing::new([[0i32; 64]; 4]);
    // Up to `flush` columns of nonzero coefficients at a time, then their sums into the 32-bit
    // lanes.
    let mut columns = vec![[0u8; 64]; flush];
    let mut values = Zeroizing::new(vec![0i8; flush]);
    let mut batches = 0;
    loop {
        let (mut count, mut more) = (0, true);
        while count < flush {
            let Some(x) = coefficients.next() else {
                more = false;
                break;
            };
            // Every coefficient takes its column from the stream, also a zero one, whose
            // place the next coefficient's column then takes: no branch on the value.
            projection.column_bytes(&mut columns[count])?;
            let s = ring.center(*x) as i8;
            values[count] = s;
            count += usize::from(s != 0);
        }
        add_columns(&mut lanes, &columns[..count], &values[..count]);
        for (w, lane) in wide.iter_mut().zip(lanes.iter_mut()) {
            for (t, a) in w.iter_mut().zip(lane.iter_mut()) {
                *t += i32::from(*a);
                *a = 0;
            }
        }
        batches += 1;
        if batches == BATCHES || !more {
            add_into(&mut p, &mut wide);
            batches = 0;
        }
        if !more {
            return Ok(p);
        }
    }
}

/// The columns `columns` with their coefficients `values` added into the 8-bit lanes of
/// [`project_bytes`]: the AVX-512 byte kernel where it runs, loops on vectors of bytes
/// otherwise. Every partial sum of a call stays within ±127 (at most `flush` columns).
fn add_columns(lanes: &mut [[i8; 64]; 4], columns: &[[u8; 64]], values: &[i8]) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if crate::simd::project_bytes(lanes, columns, values) {
        return;
    }
    for (column, s) in columns.iter().zip(values) {
        for (e, lane) in lanes.iter_mut().enumerate() {
            for (a, b) in lane.iter_mut().zip(column) {
                let b = b >> (2 * e);
                let plus = (b & 1).wrapping_neg() as i8;
                let minus = ((b >> 1) & 1).wrapping_neg() as i8;
                *a += (*s & plus) - (*s & minus);
            }
        }
    }
}

/// The projection of `coefficients` with the columns of `projection`, in i32 lanes added into
/// the i128 sums every `flush` steps.
fn project_lanes<'a>(
    ring: &Rq,
    projection: &mut Projection,
    coefficients: impl Iterator<Item = &'a u64>,
    flush: usize,
) -> Result<[i128; 256], Error> {
    let mut p = [0i128; 256];
    let mut lanes = Zeroizing::new([0i32; 256]);
    let mut steps = 0;
    let mut column = [0u8; 64];
    for x in coefficients {
        // Every coefficient takes its column from the stream, also a zero one.
        projection.column_bytes(&mut column)?;
        let s = ring.center(*x) as i32;
        if s == 0 {
            continue;
        }
        for (acc, b) in lanes.as_chunks_mut::<4>().0.iter_mut().zip(column) {
            for (a, e) in acc.iter_mut().zip(DECODE[usize::from(b)]) {
                *a += i32::from(e) * s;
            }
        }
        steps += 1;
        if steps == flush {
            for (t, a) in p.iter_mut().zip(lanes.iter_mut()) {
                *t += i128::from(*a);
                *a = 0;
            }
            steps = 0;
        }
    }
    for (t, a) in p.iter_mut().zip(lanes.iter()) {
        *t += i128::from(*a);
    }
    Ok(p)
}

/// [`project_lanes`] in i128 throughout, for witnesses with coefficients beyond $`2^{31}`$.
fn project_block_wide<'a>(
    ring: &Rq,
    projection: &mut Projection,
    coefficients: impl Iterator<Item = &'a u64>,
) -> Result<[i128; 256], Error> {
    let mut p = [0i128; 256];
    let mut column = [0i8; 256];
    for x in coefficients {
        projection.column(&mut column)?;
        let s = i128::from(ring.center(*x));
        if s != 0 {
            for (acc, e) in p.iter_mut().zip(column) {
                *acc += i128::from(e) * s;
            }
        }
    }
    Ok(p)
}

/// Test only: the projection in i128 throughout from one reader of the whole stream, the
/// reference of [`project`].
#[cfg(test)]
fn project_wide(
    ring: &Rq,
    seed: &[u8; 32],
    level: u32,
    witness: &[Zeroizing<Vec<u64>>],
) -> Result<[i128; 256], Error> {
    let total = witness.iter().map(|v| v.len()).sum();
    let mut projection = Projection::new(seed, level);
    project_block_wide(ring, &mut projection, coefficients(witness, 0..total))
}

/// The most memory the prover spends on keeping the polynomials of [`projection_blocks`]
/// from the lifted polynomials $`b''`$ to [`complete`], instead of streaming them again.
const KEEP_BUDGET: usize = 64 << 20;

#[cfg(test)]
thread_local! {
    /// Test only: keep no polynomials, so that [`complete`] streams them a second time.
    pub(crate) static STREAM_TWICE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The polynomials of [`projection_blocks`] for every `(i, x)`, in its order, kept by the
/// prover for [`complete`]: in words of 32 bits when $`q<2^{32}`$, of 64 otherwise. They are
/// public (the projection and $`\omega`$ are), and [`complete`] reads exactly the values a
/// second stream would give.
enum Kept {
    Narrow(Vec<u32>),
    Wide(Vec<u64>),
}
impl Kept {
    /// Room for `lambda` polynomials per `(i, x)` of `shape`, if that fits [`KEEP_BUDGET`].
    fn new(ring: &Rq, lambda: usize, shape: &Shape) -> Option<Self> {
        #[cfg(test)]
        if STREAM_TWICE.with(|c| c.get()) {
            return None;
        }
        let count = lambda
            .checked_mul(shape.r)?
            .checked_mul(shape.n)?
            .checked_mul(ring.degree())?;
        let narrow = ring.modulus() <= u64::from(u32::MAX);
        if count.checked_mul(if narrow { 4 } else { 8 })? > KEEP_BUDGET {
            return None;
        }
        Some(if narrow {
            Kept::Narrow(vec![0; count])
        } else {
            Kept::Wide(vec![0; count])
        })
    }
    /// The storage in slots of `len` values, one per block of [`projection_blocks`], so that
    /// each block writes its own polynomials and no block's are held for another.
    fn slots(&mut self, len: usize) -> KeptSlots<'_> {
        match self {
            Kept::Narrow(kept) => {
                KeptSlots::Narrow(kept.chunks_mut(len).map(std::sync::Mutex::new).collect())
            }
            Kept::Wide(kept) => {
                KeptSlots::Wide(kept.chunks_mut(len).map(std::sync::Mutex::new).collect())
            }
        }
    }
    /// The polynomials of index `index`, side by side, into `polys`.
    fn read(&self, index: usize, polys: &mut [u64]) {
        let (at, len) = (index * polys.len(), polys.len());
        match self {
            Kept::Narrow(kept) => {
                for (out, c) in polys.iter_mut().zip(&kept[at..at + len]) {
                    *out = u64::from(*c);
                }
            }
            Kept::Wide(kept) => polys.copy_from_slice(&kept[at..at + len]),
        }
    }
}

/// The slots of [`Kept::slots`], each locked by the task of its block alone.
enum KeptSlots<'a> {
    Narrow(Vec<std::sync::Mutex<&'a mut [u32]>>),
    Wide(Vec<std::sync::Mutex<&'a mut [u64]>>),
}
impl KeptSlots<'_> {
    /// Store the polynomials of block `block`, each index's side by side.
    fn write(&self, block: usize, polys: &[u64]) {
        match self {
            KeptSlots::Narrow(slots) => {
                let mut slot = slots[block].lock().unwrap_or_else(|e| e.into_inner());
                for (out, c) in slot.iter_mut().zip(polys) {
                    *out = *c as u32;
                }
            }
            KeptSlots::Wide(slots) => {
                let mut slot = slots[block].lock().unwrap_or_else(|e| e.into_inner());
                slot.copy_from_slice(polys);
            }
        }
    }
}

/// Buffers of the projection stream per block of [`projection_blocks`] and [`project`]: work
/// enough per reader that setting one up costs little.
const BLOCK_BUFFERS: usize = 4;

/// The polynomials of a block of [`projection_blocks`]: for each polynomial index from `first`
/// on, its $`\lambda`$ polynomials side by side.
struct BlockPolys {
    first: usize,
    stride: usize,
    data: Vec<u64>,
}
impl BlockPolys {
    fn len(&self) -> usize {
        self.data.len() / self.stride
    }
    /// Each polynomial index of the block with its polynomials.
    fn iter(&self) -> impl Iterator<Item = (usize, &[u64])> {
        let first = self.first;
        self.data
            .chunks_exact(self.stride)
            .enumerate()
            .map(move |(j, polys)| (first + j, polys))
    }
}

/// Indices `per_block` at a time, the last block possibly short: the indices of block `block`.
fn block_range(block: usize, per_block: usize, total: usize) -> core::ops::Range<usize> {
    block * per_block..((block + 1) * per_block).min(total)
}

/// The coefficient polynomials of the aggregated projection constraints (BS23 Fig. 2,
/// $`\varphi''`$): for each witness vector $`i`$ and polynomial $`x`$, `polys[k]` the
/// polynomial $`x`$ of $`\sigma_{-1}(\sum_j\omega^{(k)}_j\pi^{(j)}_i)`$, $`k<\lambda`$, computed
/// one block of the stream at a time, each on its own reader ([`Projection::at`]). `work` maps
/// the `(i, x, polys)` of a block, in order, to a result and `absorb` takes the results in block
/// order. The blocks of a wave run in parallel with `parallel`, and at most a wave of results is
/// held, nothing of size proportional to the witness.
fn projection_blocks<T: Send>(
    ring: &Rq,
    seed: &[u8; 32],
    level: u32,
    omega: &[Vec<u64>],
    shape: &Shape,
    work: impl Fn(BlockPolys) -> Result<T, Error> + Sync,
    mut absorb: impl FnMut(T) -> Result<(), Error>,
) -> Result<(), Error> {
    let table = projection_table(ring, omega);
    let per_block = BLOCK_BUFFERS * PROJECTION_BLOCK / ring.degree();
    let total = shape.r * shape.n;
    let blocks = total.div_ceil(per_block);
    for wave in (0..blocks).step_by(WAVE) {
        let results = crate::par::try_map(WAVE.min(blocks - wave), |w| {
            let block = wave + w;
            let mut projection = Projection::at(seed, level, block * BLOCK_BUFFERS)?;
            let indices = block_range(block, per_block, total);
            work(block_polys(ring, &table, &mut projection, indices)?)
        })?;
        for result in results {
            absorb(result)?;
        }
    }
    Ok(())
}

/// For each group of 4 rows, the sums of the 4 weighted entries for every k, side by side, so
/// that one lookup reads the λ sums of a byte of a column. Each sum is stored as its centred
/// residue modulo $`q`$ (at most $`q/2`$ in magnitude, while the sum of 4 canonical residues
/// reaches $`4q`$, beyond `i64` for $`q>2^{61}`$). The polynomials of [`block_polys`] only take
/// residues of sums of entries, so they do not change.
///
/// The table holds one sum per sign pattern ([`PATTERN`], 81 per group), in 32 bits below
/// $`2^{32}`$ (about 100 KB at λ = 5, within the 128 KB first-level data cache of an Apple M4
/// core). With the `simd` feature on x86-64, where the AVX-512 kernels run, a kernel reads the
/// table of patterns below $`2^{32}`$ padded to eight sums per pattern, from $`2^{32}`$ to
/// $`2^{58}`$ the table holds one sum per byte instead, the layout that another kernel reads,
/// and from $`2^{58}`$ a kernel reads it padded to four sums per pattern for λ at most 4.
struct ProjectionTable {
    lambda: usize,
    entries: TableEntries,
}

enum TableEntries {
    /// One sum per pattern, below $`2^{32}`$ in magnitude.
    Narrow(Vec<i32>),
    /// One sum per pattern.
    Wide(Vec<i64>),
    /// One sum per byte, for the AVX-512 kernel ($`2^{32}\le q<2^{58}`$).
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    Bytes(Vec<i64>),
    /// The sums of [`TableEntries::Narrow`] padded to eight per pattern (the first λ used, the
    /// others zero) from the index `start`, past which the table starts on 32 bytes so that no
    /// entry crosses a cache line, for the AVX-512VL kernel.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    NarrowPadded { table: Vec<i32>, start: usize },
    /// The sums of [`TableEntries::Wide`] from $`2^{58}`$ padded to four per pattern for λ at
    /// most 4, from `start` as above, for the wide AVX-512VL kernel.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    WidePadded { table: Vec<i64>, start: usize },
}

fn projection_table(ring: &Rq, omega: &[Vec<u64>]) -> ProjectionTable {
    let lambda = omega.len();
    debug_assert!(lambda <= MAX_LAMBDA);
    // The centred residue of the sum of the weighted entries of a group with signs `signs`:
    // entries in {-1, 0, 1}, so modular additions and subtractions of the canonical `w`.
    let sum = |k: usize, group: usize, signs: &[i8; 4]| -> i64 {
        let mut s = 0u64;
        for (e, v) in signs.iter().enumerate() {
            match v {
                1 => s = ring.add(s, omega[k][4 * group + e]),
                -1 => s = ring.sub(s, omega[k][4 * group + e]),
                _ => {}
            }
        }
        ring.center(s)
    };
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if (1 << 32..1 << 58).contains(&ring.modulus()) && crate::simd::avx512() {
        let mut table = vec![0i64; 64 * 256 * lambda];
        for group in 0..64 {
            for (byte, signs) in DECODE.iter().enumerate() {
                for k in 0..lambda {
                    table[(group * 256 + byte) * lambda + k] = sum(k, group, signs);
                }
            }
        }
        return ProjectionTable {
            lambda,
            entries: TableEntries::Bytes(table),
        };
    }
    // Every pattern has bytes, any of which stands for it.
    let mut patterns = [[0i8; 4]; 81];
    for (byte, signs) in DECODE.iter().enumerate() {
        patterns[usize::from(PATTERN[byte])] = *signs;
    }
    // The padded layouts where their AVX-512 kernels run: eight i32 per pattern below 2^32,
    // four i64 from 2^58 for λ at most 4, from the first index past which the table starts on
    // 32 bytes.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        let q = ring.modulus();
        let aligned = |address: usize, size: usize| (32 - address % 32) % 32 / size;
        if crate::simd::avx512vl() && q < 1 << 32 {
            let mut table = vec![0i32; 64 * 81 * 8 + 8];
            let start = aligned(table.as_ptr() as usize, 4);
            for group in 0..64 {
                for (pattern, signs) in patterns.iter().enumerate() {
                    for k in 0..lambda {
                        table[start + (group * 81 + pattern) * 8 + k] =
                            i32::try_from(sum(k, group, signs)).expect("a centred residue");
                    }
                }
            }
            return ProjectionTable {
                lambda,
                entries: TableEntries::NarrowPadded { table, start },
            };
        }
        if crate::simd::avx512vl() && q >= 1 << 58 && lambda <= 4 {
            let mut table = vec![0i64; 64 * 81 * 4 + 4];
            let start = aligned(table.as_ptr() as usize, 8);
            for group in 0..64 {
                for (pattern, signs) in patterns.iter().enumerate() {
                    for k in 0..lambda {
                        table[start + (group * 81 + pattern) * 4 + k] = sum(k, group, signs);
                    }
                }
            }
            return ProjectionTable {
                lambda,
                entries: TableEntries::WidePadded { table, start },
            };
        }
    }
    let mut table = vec![0i64; 64 * 81 * lambda];
    for group in 0..64 {
        for (pattern, signs) in patterns.iter().enumerate() {
            for k in 0..lambda {
                table[(group * 81 + pattern) * lambda + k] = sum(k, group, signs);
            }
        }
    }
    let entries = if ring.modulus() < 1 << 32 {
        TableEntries::Narrow(
            table
                .iter()
                .map(|x| i32::try_from(*x).expect("a centred residue below 2^31"))
                .collect(),
        )
    } else {
        TableEntries::Wide(table)
    };
    ProjectionTable { lambda, entries }
}

/// The largest λ: $`\lceil128/\lfloor\log_2q\rfloor\rceil`$ for the moduli above $`2^{16}`$
/// (`ring::MIN_MODULUS`), the length of the sums of [`ProjectionTable::sums`].
const MAX_LAMBDA: usize = 8;

/// Columns whose sums [`block_polys`] takes side by side, as many as keep their sums in
/// registers: four with the 31 general registers of aarch64, two elsewhere.
#[cfg(target_arch = "aarch64")]
const COLUMNS: usize = 4;
#[cfg(not(target_arch = "aarch64"))]
const COLUMNS: usize = 2;

/// The polynomials of [`projection_blocks`] for the polynomial indices `indices` $`=in+x`$,
/// whose columns `projection` reads in order.
fn block_polys(
    ring: &Rq,
    table: &ProjectionTable,
    projection: &mut Projection,
    indices: core::ops::Range<usize>,
) -> Result<BlockPolys, Error> {
    let (d, lambda) = (ring.degree(), table.lambda);
    let stride = lambda * d;
    let mut data = vec![0u64; indices.len() * stride];
    let mut tmp = vec![0u64; d];
    for polys in data.chunks_exact_mut(stride) {
        // Eight columns at a time in AVX-512 registers where they run (the table of patterns
        // below 2^32, the byte table up to 2^58, with the narrow sums of `ProjectionTable::sums`
        // and its reduction), then `COLUMNS` at a time, then one by one. Whether a kernel runs is
        // known before a column is read, and a kernel that declines leaves its columns to
        // `ProjectionTable::sums`, so every column is summed once in stream order.
        #[allow(unused_mut)]
        let mut first = 0;
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            let kernel = match &table.entries {
                TableEntries::Narrow(_) => crate::simd::avx512vl(),
                TableEntries::Bytes(_)
                | TableEntries::NarrowPadded { .. }
                | TableEntries::WidePadded { .. } => true,
                TableEntries::Wide(_) => false,
            };
            let mut columns = [[0u8; 64]; 8];
            let mut raw = [[0i64; 8]; 8];
            // The narrow kernels run below 2^58, where 32q does not overflow, the wide one from
            // there, whose sums take i128 (`ProjectionTable::sums`).
            let narrow = ring.modulus() < 1 << 58;
            let offset = if kernel && narrow {
                32 * ring.modulus()
            } else {
                0
            };
            while kernel && first + 8 <= d {
                for c in columns.iter_mut() {
                    projection.column_bytes(c)?;
                }
                let done = match &table.entries {
                    TableEntries::NarrowPadded { table: t, start } => {
                        crate::simd::padded_sums8(&t[*start..], &PATTERN, &columns, &mut raw)
                    }
                    TableEntries::Narrow(t) => {
                        crate::simd::pattern_sums8(t, lambda, &PATTERN, &columns, &mut raw)
                    }
                    TableEntries::Bytes(t) => {
                        crate::simd::table_sums8(t, lambda, &columns, &mut raw)
                    }
                    TableEntries::Wide(_) => false,
                    TableEntries::WidePadded { table: t, start } => {
                        let (mut low, mut high) = ([[0u64; 4]; 8], [[0i64; 4]; 8]);
                        let done = crate::simd::wide_sums8(
                            &t[*start..],
                            &PATTERN,
                            &columns,
                            &mut low,
                            &mut high,
                        );
                        if done {
                            // The canonical residues of hi 2^32 + lo, in lanes where they run.
                            let mut reduced = [[0u64; 4]; 8];
                            let q = ring.modulus();
                            let shift_shoup = ((1u128 << 96) / u128::from(q)) as u64;
                            if !crate::simd::reduce_wide8(&low, &high, q, shift_shoup, &mut reduced)
                            {
                                let offset = 32 * i128::from(q);
                                for ((o, l), h) in reduced.iter_mut().zip(&low).zip(&high) {
                                    for ((x, l), h) in o.iter_mut().zip(l).zip(h) {
                                        let a = (i128::from(*h) << 32) + i128::from(*l);
                                        *x = ring.reduce_wide((a + offset) as u128);
                                    }
                                }
                            }
                            for (j, r) in reduced.iter().enumerate() {
                                for (k, s) in r[..lambda].iter().enumerate() {
                                    polys[k * d + first + j] = *s;
                                }
                            }
                            first += 8;
                            continue;
                        }
                        false
                    }
                };
                if done {
                    // The lanes past λ hold zero sums, whose residues are not written.
                    let mut reduced = [[0u64; 8]; 8];
                    let q = ring.modulus();
                    if !crate::simd::reduce_sums8(&raw, offset, q, ring.one_shoup(), &mut reduced) {
                        for (o, r) in reduced.iter_mut().zip(&raw) {
                            for (x, s) in o.iter_mut().zip(r) {
                                *x = ring.reduce_word((*s as u64).wrapping_add(offset));
                            }
                        }
                    }
                    for (j, r) in reduced.iter().enumerate() {
                        for (k, s) in r[..lambda].iter().enumerate() {
                            polys[k * d + first + j] = *s;
                        }
                    }
                } else {
                    let mut sums = [[0u64; MAX_LAMBDA]; 8];
                    table.sums(ring, &columns, &mut sums);
                    for (j, s) in sums.iter().enumerate() {
                        for (k, s) in s[..lambda].iter().enumerate() {
                            polys[k * d + first + j] = *s;
                        }
                    }
                }
                first += 8;
            }
        }
        let mut columns = [[0u8; 64]; COLUMNS];
        let mut sums = [[0u64; 8]; COLUMNS];
        while first + COLUMNS <= d {
            for c in columns.iter_mut() {
                projection.column_bytes(c)?;
            }
            table.sums(ring, &columns, &mut sums);
            for (j, s) in sums.iter().enumerate() {
                for (k, s) in s[..lambda].iter().enumerate() {
                    polys[k * d + first + j] = *s;
                }
            }
            first += COLUMNS;
        }
        let (mut column, mut sum) = ([[0u8; 64]; 1], [[0u64; 8]; 1]);
        for c in first..d {
            projection.column_bytes(&mut column[0])?;
            table.sums(ring, &column, &mut sum);
            for (k, s) in sum[0][..lambda].iter().enumerate() {
                polys[k * d + c] = *s;
            }
        }
        for poly in polys.chunks_exact_mut(d) {
            ring.sigma_minus_one(poly, &mut tmp);
            poly.copy_from_slice(&tmp);
        }
    }
    Ok(BlockPolys {
        first: indices.start,
        stride,
        data,
    })
}

impl ProjectionTable {
    /// `sums[c][k]` $`=\sum_g`$ (the sum of group $`g`$ at byte `columns[c][g]`, entry $`k`$)
    /// $`\bmod q`$ for $`k<\lambda`$, canonical, for each column of `columns`.
    ///
    /// The 64 entries of a sum are centred residues, at most $`(q-1)/2`$ in magnitude, so the
    /// sum lies within $`\pm32(q-1)`$ and adding $`32q`$ gives a positive value below $`64q`$:
    /// one word while $`q<2^{58}`$, where the sums are taken in `i64`, two from there, in
    /// `i128`. One Shoup reduction of that value replaces the division. Specialized for each λ
    /// of the supported moduli (3 to 8), so that the sums of all the columns stay in
    /// registers, with a loop per lane for any other λ and for the byte and padded tables.
    fn sums<const B: usize>(
        &self,
        ring: &Rq,
        columns: &[[u8; 64]; B],
        sums: &mut [[u64; MAX_LAMBDA]; B],
    ) {
        let narrow = ring.modulus() < 1 << 58;
        match (&self.entries, self.lambda) {
            (TableEntries::Narrow(t), 3) => narrow_sums::<i32, 3, B>(ring, t, columns, sums),
            (TableEntries::Narrow(t), 4) => narrow_sums::<i32, 4, B>(ring, t, columns, sums),
            (TableEntries::Narrow(t), 5) => narrow_sums::<i32, 5, B>(ring, t, columns, sums),
            (TableEntries::Narrow(t), 6) => narrow_sums::<i32, 6, B>(ring, t, columns, sums),
            (TableEntries::Narrow(t), 7) => narrow_sums::<i32, 7, B>(ring, t, columns, sums),
            (TableEntries::Narrow(t), 8) => narrow_sums::<i32, 8, B>(ring, t, columns, sums),
            (TableEntries::Wide(t), 3) if narrow => {
                narrow_sums::<i64, 3, B>(ring, t, columns, sums)
            }
            (TableEntries::Wide(t), 4) if narrow => {
                narrow_sums::<i64, 4, B>(ring, t, columns, sums)
            }
            (TableEntries::Wide(t), 5) if narrow => {
                narrow_sums::<i64, 5, B>(ring, t, columns, sums)
            }
            (TableEntries::Wide(t), 6) if narrow => {
                narrow_sums::<i64, 6, B>(ring, t, columns, sums)
            }
            (TableEntries::Wide(t), 7) if narrow => {
                narrow_sums::<i64, 7, B>(ring, t, columns, sums)
            }
            (TableEntries::Wide(t), 8) if narrow => {
                narrow_sums::<i64, 8, B>(ring, t, columns, sums)
            }
            (TableEntries::Wide(t), 3) => wide_sums::<3, B>(ring, t, columns, sums),
            (TableEntries::Wide(t), 4) => wide_sums::<4, B>(ring, t, columns, sums),
            (TableEntries::Wide(t), 5) => wide_sums::<5, B>(ring, t, columns, sums),
            (TableEntries::Wide(t), 6) => wide_sums::<6, B>(ring, t, columns, sums),
            (TableEntries::Wide(t), 7) => wide_sums::<7, B>(ring, t, columns, sums),
            (TableEntries::Wide(t), 8) => wide_sums::<8, B>(ring, t, columns, sums),
            _ => {
                let offset = 32 * i128::from(ring.modulus());
                for (s, column) in sums.iter_mut().zip(columns) {
                    for (k, s) in s[..self.lambda].iter_mut().enumerate() {
                        let a: i128 = column
                            .iter()
                            .enumerate()
                            .map(|(g, b)| i128::from(self.entry(g, *b, k)))
                            .sum();
                        *s = ring.reduce_wide((a + offset) as u128);
                    }
                }
            }
        }
    }

    /// The sum of group `g` at byte `byte`, entry `k`.
    fn entry(&self, g: usize, byte: u8, k: usize) -> i64 {
        let pattern = || (g * 81 + usize::from(PATTERN[usize::from(byte)])) * self.lambda + k;
        match &self.entries {
            TableEntries::Narrow(t) => i64::from(t[pattern()]),
            TableEntries::Wide(t) => t[pattern()],
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            TableEntries::Bytes(t) => t[(g * 256 + usize::from(byte)) * self.lambda + k],
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            TableEntries::NarrowPadded { table, start } => {
                let at = (g * 81 + usize::from(PATTERN[usize::from(byte)])) * 8 + k;
                i64::from(table[start + at])
            }
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            TableEntries::WidePadded { table, start } => {
                table[start + (g * 81 + usize::from(PATTERN[usize::from(byte)])) * 4 + k]
            }
        }
    }
}

/// [`ProjectionTable::sums`] for λ = `L` over pattern sums of type `T`, in `i64`
/// ($`q<2^{58}`$).
fn narrow_sums<T: Copy + Into<i64>, const L: usize, const B: usize>(
    ring: &Rq,
    table: &[T],
    columns: &[[u8; 64]; B],
    sums: &mut [[u64; 8]; B],
) {
    let table = table.as_chunks::<L>().0;
    let mut acc = [[0i64; L]; B];
    for g in 0..64 {
        let group = &table[g * 81..(g + 1) * 81];
        for (a, column) in acc.iter_mut().zip(columns) {
            let entry = &group[usize::from(PATTERN[usize::from(column[g])])];
            for (a, t) in a.iter_mut().zip(entry) {
                *a += (*t).into();
            }
        }
    }
    // The sum plus 32q is below 64q < 2^64, so the wrapping addition is exact.
    let offset = 32 * ring.modulus();
    for (s, a) in sums.iter_mut().zip(acc) {
        for (s, a) in s.iter_mut().zip(a) {
            *s = ring.reduce_word((a as u64).wrapping_add(offset));
        }
    }
}

/// [`ProjectionTable::sums`] for λ = `L`, in `i128` ($`q\ge2^{58}`$).
fn wide_sums<const L: usize, const B: usize>(
    ring: &Rq,
    table: &[i64],
    columns: &[[u8; 64]; B],
    sums: &mut [[u64; 8]; B],
) {
    // Each entry t is hi 2^32 + lo for its low 32 bits lo (unsigned) and hi = t >> 32, below
    // 2^31 in magnitude: the 64 entries of a column sum to less than 2^38 in lo and to less
    // than 2^37 in magnitude in hi, so both sums take 64-bit lanes, and they are joined once.
    let table = table.as_chunks::<L>().0;
    let mut low = [[0u64; L]; B];
    let mut high = [[0i64; L]; B];
    for g in 0..64 {
        let group = &table[g * 81..(g + 1) * 81];
        for ((lo, hi), column) in low.iter_mut().zip(high.iter_mut()).zip(columns) {
            let entry = &group[usize::from(PATTERN[usize::from(column[g])])];
            for ((l, h), t) in lo.iter_mut().zip(hi.iter_mut()).zip(entry) {
                *l += u64::from(*t as u32);
                *h += *t >> 32;
            }
        }
    }
    let offset = 32 * i128::from(ring.modulus());
    for ((s, lo), hi) in sums.iter_mut().zip(low).zip(high) {
        for ((s, l), h) in s.iter_mut().zip(lo).zip(hi) {
            let a = (i128::from(h) << 32) + i128::from(l);
            *s = ring.reduce_wide((a + offset) as u128);
        }
    }
}

/// $`\gamma_l=\sum_k\beta_k\psi^{(k)}_l`$.
fn gammas(ring: &Rq, beta: &[Vec<u64>], psi: &[Vec<u64>], count: usize) -> Vec<Vec<u64>> {
    (0..count)
        .map(|l| {
            let mut g = vec![0u64; ring.degree()];
            for (b, p) in beta.iter().zip(psi) {
                ring.add_scaled_assign(&mut g, b, p[l]);
            }
            g
        })
        .collect()
}

/// The projection of a level as [`complete`] needs it: its seed and $`\omega`$, and the
/// polynomials of [`projection_blocks`] if the prover kept them.
struct Projected<'a> {
    seed: &'a [u8; 32],
    level: u32,
    omega: &'a [Vec<u64>],
    kept: Option<&'a Kept>,
}

/// Complete an aggregate with the projection constraints and the lifted polynomials:
/// $`\varphi_i\mathrel{+}=\sum_k\beta_k\varphi''^{(k)}_i`$ and
/// $`b\mathrel{+}=\sum_k\beta_kb''^{(k)}`$. The polynomials $`\varphi''^{(k)}_i`$ are the kept
/// ones if there are, the streamed ones otherwise: the same values.
fn complete(
    ring: &Rq,
    agg: &mut Aggregate,
    projection: &Projected<'_>,
    shape: &Shape,
    beta: &[Vec<u64>],
    bpp: &[Vec<u64>],
) -> Result<(), Error> {
    let (d, k) = (ring.degree(), ring.hat_len());
    let beta_hat: Vec<_> = beta.iter().map(|b| ring.ntt(b)).collect();
    let n = shape.n;
    // The polynomial of sum_k beta_k phi''^(k) at one index, from its polynomials side by side.
    let term = |polys: &[u64], acc: &mut [u64], hat: &mut [u64], out: &mut [u64]| {
        acc.fill(0);
        for (p, b) in polys.chunks_exact(d).zip(&beta_hat) {
            ring.ntt_into(p, hat);
            ring.mac(acc, b, hat);
        }
        ring.intt_into(acc, out);
    };
    // The terms of a block, from its first index on.
    let mut add = |(first, terms): (usize, Vec<u64>)| -> Result<(), Error> {
        for (j, poly) in terms.chunks_exact(d).enumerate() {
            let (i, x) = ((first + j) / n, (first + j) % n);
            ring.add_assign(&mut agg.phi[i][x * d..(x + 1) * d], poly);
        }
        Ok(())
    };
    match projection.kept {
        Some(kept) => {
            // The blocks of the stream's order, from the kept polynomials.
            let per_block = BLOCK_BUFFERS * PROJECTION_BLOCK / d;
            let total = shape.r * n;
            let blocks = total.div_ceil(per_block);
            let stride = projection.omega.len() * d;
            for wave in (0..blocks).step_by(WAVE) {
                let results = crate::par::try_map(WAVE.min(blocks - wave), |w| {
                    let indices = block_range(wave + w, per_block, total);
                    let (mut acc, mut hat) = (vec![0u64; k], vec![0u64; k]);
                    let mut polys = vec![0u64; stride];
                    let mut terms = vec![0u64; indices.len() * d];
                    for (index, out) in indices.clone().zip(terms.chunks_exact_mut(d)) {
                        kept.read(index, &mut polys);
                        term(&polys, &mut acc, &mut hat, out);
                    }
                    Ok((indices.start, terms))
                })?;
                for result in results {
                    add(result)?;
                }
            }
        }
        None => projection_blocks(
            ring,
            projection.seed,
            projection.level,
            projection.omega,
            shape,
            |block| {
                let (mut acc, mut hat) = (vec![0u64; k], vec![0u64; k]);
                let mut terms = vec![0u64; block.len() * d];
                for ((_, polys), out) in block.iter().zip(terms.chunks_exact_mut(d)) {
                    term(polys, &mut acc, &mut hat, out);
                }
                Ok((block.first, terms))
            },
            &mut add,
        )?,
    }
    for (b, l) in beta.iter().zip(bpp) {
        ring.add_assign(&mut agg.constant, &ring.poly_mul(b, l));
    }
    Ok(())
}

/// The transforms of the challenges, for [`combine_hats`] and [`combine`].
fn challenge_hats(ring: &Rq, challenges: &[Sparse]) -> Vec<Vec<u64>> {
    challenges
        .iter()
        .map(|c| ring.ntt(&c.dense(ring)))
        .collect()
}

/// $`\sum_ic_iv_i`$ for the challenges whose transforms are `c_hats` and $`r`$ vectors of `n`
/// polynomials whose transforms are `v_hats`: the products accumulated in the transform domain
/// and one inverse transform per polynomial. Each coefficient of the sum, from centred
/// operands, is an integer below $`r\|c\|_1(q-1)/2`$ in magnitude, which the CRT reconstructs
/// exactly (`Rq::intt_into`), so the residues are those of the sparse products
/// ([`Rq::mul_sparse_add`]). The sums are wiped on drop, as $`z`$ is secret.
fn combine_hats(
    ring: &Rq,
    c_hats: &[Vec<u64>],
    v_hats: &[impl AsRef<[u64]>],
    n: usize,
) -> Zeroizing<Vec<u64>> {
    let (d, k) = (ring.degree(), ring.hat_len());
    let mut out = Zeroizing::new(vec![0u64; n * d]);
    let mut acc = Zeroizing::new(vec![0u64; k]);
    for (x, poly) in out.chunks_exact_mut(d).enumerate() {
        acc.fill(0);
        for (c_hat, v) in c_hats.iter().zip(v_hats) {
            ring.mac(&mut acc, c_hat, &v.as_ref()[x * k..(x + 1) * k]);
        }
        ring.intt_into(&mut acc, poly);
    }
    out
}

/// [`combine_hats`] of vectors given by their coefficients, each polynomial transformed where
/// it is used: $`\varphi_z=\sum_ic_i\varphi_i`$ for the verifier, which keeps no transforms.
fn combine(ring: &Rq, c_hats: &[Vec<u64>], v: &[Vec<u64>], n: usize) -> Vec<u64> {
    let (d, k) = (ring.degree(), ring.hat_len());
    let mut out = vec![0u64; n * d];
    let (mut acc, mut hat) = (vec![0u64; k], vec![0u64; k]);
    for (x, poly) in out.chunks_exact_mut(d).enumerate() {
        acc.fill(0);
        for (c_hat, v) in c_hats.iter().zip(v) {
            ring.ntt_into(&v[x * d..(x + 1) * d], &mut hat);
            ring.mac(&mut acc, c_hat, &hat);
        }
        ring.intt_into(&mut acc, poly);
    }
    out
}

/// $`F(w)`$ for an aggregate and a witness given by its transforms.
pub(crate) fn evaluate_aggregate(ring: &Rq, agg: &Aggregate, hats: &[Vec<u64>]) -> Vec<u64> {
    let phi_hats: Vec<Option<Vec<u64>>> = agg
        .phi
        .iter()
        .map(|phi| phi.iter().any(|x| *x != 0).then(|| ring.ntt_many(phi)))
        .collect();
    let phi_hats: Vec<Option<&[u64]>> = phi_hats.iter().map(|h| h.as_deref()).collect();
    evaluate_with(ring, agg, &phi_hats, hats)
}

/// [`evaluate_aggregate`] with the transforms of the nonzero vectors $`\varphi_i`$ given, as
/// the prover has them for its garbage terms.
fn evaluate_with(
    ring: &Rq,
    agg: &Aggregate,
    phi_hats: &[Option<&[u64]>],
    hats: &[Vec<u64>],
) -> Vec<u64> {
    let mut value = vec![0u64; ring.degree()];
    for (i, j, a) in &agg.quad {
        ring.add_assign(
            &mut value,
            &ring.poly_mul(a, &ring.inner_hat(&hats[*i], &hats[*j])),
        );
    }
    for (phi_hat, hat) in phi_hats.iter().zip(hats) {
        if let Some(phi_hat) = phi_hat {
            ring.add_assign(&mut value, &ring.inner_hat(phi_hat, hat));
        }
    }
    ring.sub_assign(&mut value, &agg.constant);
    value
}

/// Run one level as the prover: choose its parameters, then try attempts until the next
/// witness meets its bound. Appends the level's bytes to `out` and returns the next statement
/// and witness. `derived` is what the statement's check already derived from `witness`, if it
/// ran (the first level), and is computed here otherwise.
pub(crate) fn prove<R: Relation>(
    ctx: &Ctx,
    rel: &R,
    witness: &[Zeroizing<Vec<u64>>],
    level: u32,
    transcript: &mut Transcript,
    out: &mut Vec<u8>,
    derived: Option<Derived>,
) -> Result<NextLevel, Error> {
    let ring = &ctx.ring;
    let shape = rel.shape();
    // Everything derived from the witness is wiped on drop.
    let Derived {
        hats,
        pairs,
        norm_sq,
    } = derived.unwrap_or_else(|| Derived::of(ring, witness));
    if norm_sq > shape.norm_sq {
        return Err(Error::Witness);
    }
    let base = plan::choose(
        ring,
        &ctx.set,
        &ctx.policy,
        &shape,
        &Stats {
            norm_sq,
            garbage: Some(&pairs),
            memo: plan::GarbageMemo::default(),
        },
    )?;
    for attempt in 0..MAX_ATTEMPTS {
        let params = LevelParams { attempt, ..base };
        let mut t = transcript.clone();
        let mut bytes = Vec::new();
        if let Some(result) = attempt_level(
            ctx, rel, witness, &hats, &pairs, level, &params, &mut t, &mut bytes,
        )? {
            *transcript = t;
            out.extend_from_slice(&bytes);
            return Ok(result);
        }
    }
    Err(Error::RestartLimit)
}

#[allow(clippy::too_many_arguments)]
fn attempt_level<R: Relation>(
    ctx: &Ctx,
    rel: &R,
    witness: &[Zeroizing<Vec<u64>>],
    hats: &[Vec<u64>],
    pairs: &[Vec<u64>],
    level: u32,
    params: &LevelParams,
    t: &mut Transcript,
    out: &mut Vec<u8>,
) -> Result<Option<NextLevel>, Error> {
    let ring = &ctx.ring;
    let (q, d, k) = (ring.modulus(), ring.degree(), ring.hat_len());
    let shape = rel.shape();
    let (r, n) = (shape.r, shape.n);
    let layout = params.validate(&shape, ring, &ctx.set, &ctx.policy)?;
    let sizes = sizes(ctx, &shape, params)?;
    let encoded = params.encode();
    t.absorb(&encoded);
    out.extend_from_slice(&encoded);

    // Inner commitments t_i = A s_i, all i at once, row by row.
    let kappa = params.kappa;
    let mut t_vecs = Zeroizing::new(vec![vec![0u64; kappa * d]; r]);
    {
        let mut a = Matrix::new(ring, &ctx.pp_seed, purpose::MATRIX_A, level);
        let mut entry = vec![0u64; d];
        let mut entry_hat = vec![0u64; k];
        for row in 0..kappa {
            let mut accs = Zeroizing::new(vec![vec![0u64; k]; r]);
            for x in 0..n {
                a.next_poly(&mut entry)?;
                ring.ntt_into(&entry, &mut entry_hat);
                for (acc, hat) in accs.iter_mut().zip(hats) {
                    ring.mac(acc, &entry_hat, &hat[x * k..(x + 1) * k]);
                }
            }
            for (i, acc) in accs.iter_mut().enumerate() {
                ring.intt_into(acc, &mut t_vecs[i][row * d..(row + 1) * d]);
            }
        }
    }

    // v = t̂ ‖ ĝ ‖ ĥ: the first two parts now.
    let mut v = Zeroizing::new(vec![0u64; layout.m * d]);
    for (i, ti) in t_vecs.iter().enumerate() {
        let digits = Zeroizing::new(decompose(ring, ti, params.b1, params.t1));
        for (digit, part) in digits.iter().enumerate() {
            let y = (i * params.t1 + digit) * kappa;
            v[y * d..(y + kappa) * d].copy_from_slice(part);
        }
    }
    for (pair, g) in pairs.iter().enumerate() {
        let digits = Zeroizing::new(decompose(ring, g, params.b2, params.t2));
        for (digit, part) in digits.iter().enumerate() {
            let y = layout.off_g + pair * params.t2 + digit;
            v[y * d..(y + 1) * d].copy_from_slice(part);
        }
    }
    let u1 = matvec(
        ctx,
        purpose::MATRIX_B,
        level,
        params.kappa1,
        &Zeroizing::new(ring.ntt_many(&v[..layout.off_h * d])),
    )?;
    let bytes = encode_residues(q, &u1)?;
    debug_assert_eq!(bytes.len(), sizes.u1);
    t.absorb(&bytes);
    out.extend_from_slice(&bytes);

    // Projection.
    let jl_seed = squeeze_seed(t);
    let p = project(ring, &jl_seed, level, witness)?;
    let p_norm: u128 = p.iter().map(|x| x.unsigned_abs() * x.unsigned_abs()).sum();
    if p_norm > shape.norm_sq.saturating_mul(337) {
        return Ok(None);
    }
    let bytes = encode_signed(&p, sizes.p_width)?;
    t.absorb(&bytes);
    out.extend_from_slice(&bytes);

    // Aggregation with integer challenges and the lifted polynomials b''.
    let l_count = rel.constant_term_count();
    let mut psi = Vec::with_capacity(ctx.lambda);
    let mut omega = Vec::with_capacity(ctx.lambda);
    for _ in 0..ctx.lambda {
        psi.push(squeeze_residues(t, q, l_count)?);
        omega.push(squeeze_residues(t, q, 256)?);
    }
    let values = Zeroizing::new(rel.constant_term_values(hats, pairs));
    let lambda = ctx.lambda;
    let mut accs = vec![vec![0u64; k]; lambda];
    let mut kept = Kept::new(ring, lambda, &shape);
    let n = shape.n;
    let per_block = BLOCK_BUFFERS * PROJECTION_BLOCK / d;
    let slots = kept.as_mut().map(|kept| kept.slots(per_block * lambda * d));
    // Each block's sums, added in block order: the residues of one sum in any order.
    projection_blocks(
        ring,
        &jl_seed,
        level,
        &omega,
        &shape,
        |block| {
            let mut part = vec![0u64; lambda * k];
            let mut hat = vec![0u64; k];
            for (index, polys) in block.iter() {
                let (i, x) = (index / n, index % n);
                for (acc, poly) in part.chunks_exact_mut(k).zip(polys.chunks_exact(d)) {
                    ring.ntt_into(poly, &mut hat);
                    ring.mac(acc, &hat, &hats[i][x * k..(x + 1) * k]);
                }
            }
            if let Some(slots) = &slots {
                slots.write(block.first / per_block, &block.data);
            }
            Ok(part)
        },
        |part| {
            for (acc, p) in accs.iter_mut().zip(part.chunks_exact(k)) {
                ring.hat_add_assign(acc, p);
            }
            Ok(())
        },
    )?;
    drop(slots);
    let mut bpp = Vec::with_capacity(ctx.lambda);
    for (acc, psi_k) in accs.into_iter().zip(&psi) {
        let mut b = ring.intt(acc);
        for (value, weight) in values.iter().zip(psi_k) {
            ring.add_scaled_assign(&mut b, value, *weight);
        }
        bpp.push(b);
    }
    let tails: Vec<u64> = bpp.iter().flat_map(|b| b[1..].iter().copied()).collect();
    let bytes = encode_residues(q, &tails)?;
    debug_assert_eq!(bytes.len(), sizes.bpp);
    t.absorb(&bytes);
    out.extend_from_slice(&bytes);

    // Aggregation with polynomial challenges.
    let alpha = squeeze_polys(t, ring, rel.full_count())?;
    let beta = squeeze_polys(t, ring, ctx.lambda)?;
    let gamma = gammas(ring, &beta, &psi, l_count);
    let mut agg = rel.aggregate(&alpha, &gamma)?;
    let projection = Projected {
        seed: &jl_seed,
        level,
        omega: &omega,
        kept: kept.as_ref(),
    };
    complete(ring, &mut agg, &projection, &shape, &beta, &bpp)?;
    drop(kept);
    // The transforms of phi, for the check and the garbage terms both.
    let phi_hats: Vec<Vec<u64>> = agg.phi.iter().map(|v| ring.ntt_many(v)).collect();
    let nonzero: Vec<Option<&[u64]>> = agg
        .phi
        .iter()
        .zip(&phi_hats)
        .map(|(phi, hat)| phi.iter().any(|x| *x != 0).then_some(&hat[..]))
        .collect();
    if evaluate_with(ring, &agg, &nonzero, hats)
        .iter()
        .any(|x| *x != 0)
    {
        return Err(Error::Witness);
    }

    // Garbage h and the second outer commitment.
    for i in 0..r {
        for j in i..r {
            let mut h = Zeroizing::new(ring.inner_hat(&phi_hats[i], &hats[j]));
            if i != j {
                ring.add_assign(&mut h, &ring.inner_hat(&phi_hats[j], &hats[i]));
            }
            let pair = pair_index(i, j, r);
            let digits = Zeroizing::new(decompose(ring, &h, params.b1, params.t1));
            for (digit, part) in digits.iter().enumerate() {
                let y = layout.off_h + pair * params.t1 + digit;
                v[y * d..(y + 1) * d].copy_from_slice(part);
            }
        }
    }
    let u2 = matvec(
        ctx,
        purpose::MATRIX_D,
        level,
        params.kappa2,
        &Zeroizing::new(ring.ntt_many(&v[layout.off_h * d..])),
    )?;
    let bytes = encode_residues(q, &u2)?;
    debug_assert_eq!(bytes.len(), sizes.u2);
    t.absorb(&bytes);
    out.extend_from_slice(&bytes);

    // Amortization.
    let c_seed = squeeze_seed(t);
    let challenges = (0..r)
        .map(|i| ctx.set.sample(&c_seed, i as u32, &ctx.eta_ring))
        .collect::<Result<Vec<_>, _>>()?;
    // z = sum_i c_i s_i from the transforms of the witness (`combine_hats`).
    let c_hats = challenge_hats(ring, &challenges);
    let z = combine_hats(ring, &c_hats, hats, n);
    let next_wit = next_witness(ring, params, &layout, &z, &v);
    let next_norm: u128 = next_wit
        .iter()
        .map(|w| ring.norm_sq(w))
        .fold(0, u128::saturating_add);
    if next_norm > params.next_norm_sq {
        return Ok(None);
    }
    let next = Next {
        ring: ring.clone(),
        pp_seed: ctx.pp_seed,
        level,
        params: *params,
        layout,
        phi_z: combine_hats(ring, &c_hats, &phi_hats, n).to_vec(),
        challenges,
        u1,
        u2,
        quad: agg.quad,
        constant: agg.constant,
    };
    Ok(Some((next, next_wit)))
}

/// Check one level as the verifier, reading its bytes from `input`: validate its parameters,
/// check the projection, and return the next statement.
pub(crate) fn verify<R: Relation>(
    ctx: &Ctx,
    rel: &R,
    level: u32,
    transcript: &mut Transcript,
    input: &mut &[u8],
) -> Result<Next, Error> {
    let ring = &ctx.ring;
    let (q, d) = (ring.modulus(), ring.degree());
    let shape = rel.shape();
    let encoded = take(input, PARAMS_BYTES)?;
    let params = LevelParams::decode(encoded)?;
    let layout = params
        .validate(&shape, ring, &ctx.set, &ctx.policy)
        .map_err(|_| Error::InvalidProof)?;
    let sizes = sizes(ctx, &shape, &params)?;
    transcript.absorb(encoded);

    let bytes = take(input, sizes.u1)?;
    transcript.absorb(bytes);
    let u1 = decode_residues(q, bytes, params.kappa1 * d)?;

    let jl_seed = squeeze_seed(transcript);
    let bytes = take(input, sizes.p)?;
    transcript.absorb(bytes);
    let p = decode_signed(bytes, 256, sizes.p_width)?;
    let p_norm = p
        .iter()
        .try_fold(0u128, |acc, x| {
            acc.checked_add(x.unsigned_abs() * x.unsigned_abs())
        })
        .ok_or(Error::InvalidProof)?;
    if p_norm > shape.norm_sq.saturating_mul(337) {
        return Err(Error::InvalidProof);
    }

    let l_count = rel.constant_term_count();
    let mut psi = Vec::with_capacity(ctx.lambda);
    let mut omega = Vec::with_capacity(ctx.lambda);
    for _ in 0..ctx.lambda {
        psi.push(squeeze_residues(transcript, q, l_count)?);
        omega.push(squeeze_residues(transcript, q, 256)?);
    }
    let bytes = take(input, sizes.bpp)?;
    transcript.absorb(bytes);
    let tails = decode_residues(q, bytes, ctx.lambda * (d - 1))?;
    // The constant coefficient of b''^(k) is <omega^(k), p> + sum_l psi_l^(k) b'_0^(l).
    let constants = rel.constant_term_constants();
    let bpp: Vec<Vec<u64>> = (0..ctx.lambda)
        .map(|k| {
            let mut ct = p.iter().zip(&omega[k]).fold(0u64, |acc, (x, w)| {
                ring.add(acc, ring.mul(ring.lift_i128(*x), *w))
            });
            for (b0, w) in constants.iter().zip(&psi[k]) {
                ct = ring.add(ct, ring.mul(*b0, *w));
            }
            let mut b = Vec::with_capacity(d);
            b.push(ct);
            b.extend_from_slice(&tails[k * (d - 1)..(k + 1) * (d - 1)]);
            b
        })
        .collect();

    let alpha = squeeze_polys(transcript, ring, rel.full_count())?;
    let beta = squeeze_polys(transcript, ring, ctx.lambda)?;
    let gamma = gammas(ring, &beta, &psi, l_count);
    let mut agg = rel.aggregate(&alpha, &gamma)?;
    let projection = Projected {
        seed: &jl_seed,
        level,
        omega: &omega,
        kept: None,
    };
    complete(ring, &mut agg, &projection, &shape, &beta, &bpp)?;

    let bytes = take(input, sizes.u2)?;
    transcript.absorb(bytes);
    let u2 = decode_residues(q, bytes, params.kappa2 * d)?;

    let c_seed = squeeze_seed(transcript);
    let challenges = (0..shape.r)
        .map(|i| ctx.set.sample(&c_seed, i as u32, &ctx.eta_ring))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Next {
        ring: ring.clone(),
        pp_seed: ctx.pp_seed,
        level,
        params,
        phi_z: combine(ring, &challenge_hats(ring, &challenges), &agg.phi, shape.n),
        layout,
        challenges,
        u1,
        u2,
        quad: agg.quad,
        constant: agg.constant,
    })
}

/// The polynomials of vector `i` of the next witness that carry values: all $`n'`$ of the
/// parts of $`z_0`$ and $`z_1`$, and those of $`v`$ before its zero padding.
fn final_length(layout: &Layout, nu: usize, i: usize) -> usize {
    if i < 2 * nu {
        layout.n_next
    } else {
        let start = (i - 2 * nu) * layout.n_next;
        layout.n_next.min(layout.m - start)
    }
}

/// Whether vector `i` of the final witness is a part of $`z_1`$, whose coefficients are close
/// to a discrete Gaussian, rather than of $`z_0`$ or $`v`$, whose coefficients are centred
/// digits, close to uniform.
fn gaussian_part(nu: usize, i: usize) -> bool {
    (nu..2 * nu).contains(&i)
}

/// The bits of `x` in the Gaussian code of [`BitWriter::gaussian`] with parameter `p`: the
/// unary quotient, its end bit and $`p+1`$ remainder bits.
pub(crate) fn gaussian_bits(x: i128, p: u32) -> u64 {
    // The scale is a power of two: the remainder is a mask, and the quotient of the multiple
    // x - low an exact shift.
    let scale = 1i128 << (p + 1);
    let r = x & (scale - 1);
    let low = if r >= scale / 2 { r - scale } else { r };
    let quotient = (x - low) >> (p + 1);
    let n = if quotient > 0 {
        2 * quotient - 1
    } else {
        -2 * quotient
    };
    n as u64 + 1 + u64::from(p + 1)
}

/// The parameter of the Gaussian code that writes `values` in the fewest bits, the least on
/// ties. From the least $`p`$ with every value in $`[-2^p,2^p)`$ on, every quotient is zero
/// and the length grows with $`p`$, so the parameters up to that one are compared.
fn gaussian_parameter(values: &[i128]) -> u32 {
    let max = values.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0);
    (0..=128 - max.leading_zeros())
        .min_by_key(|p| {
            let bits: u64 = values.iter().map(|x| gaussian_bits(*x, *p)).sum();
            (bits, *p)
        })
        .expect("a nonempty range")
}

/// The longest unary quotient of a value of magnitude at most `half` in the Gaussian code with
/// parameter `p`: $`|x|\le h`$ gives a quotient of magnitude at most
/// $`\lfloor h/2^{p+1}\rfloor+1`$, whose code has at most twice as many one bits.
fn unary_limit(half: u128, p: u32) -> usize {
    usize::try_from(2 * ((half >> (p + 1)) + 1)).unwrap_or(usize::MAX)
}

/// The number of digits of one chunk of the radix code in base `radix` at least 3, the
/// largest $`k`$ with $`R^k<2^{128}`$.
pub(crate) fn radix_chunk(radix: u128) -> usize {
    let (mut k, mut power) = (1, radix);
    while let Some(next) = power.checked_mul(radix) {
        power = next;
        k += 1;
    }
    k
}

/// Values in $`[-h,h]`$, $`h\ge1`$, in the radix code: the digits $`x+h`$ in base
/// $`R=2h+1`$, in chunks of [`radix_chunk`] digits (the last one shorter), each chunk the
/// number $`\sum_j(x_j+h)R^j`$ in the canonical code of $`[0,R^k)`$ ([`BitWriter::uniform`]).
/// Uniform digits take $`\log_2R`$ bits each, up to the rounding of one chunk.
fn write_radix(writer: &mut BitWriter, values: &[i128], h: u128) -> Result<(), Error> {
    let radix = 2 * h + 1;
    for chunk in values.chunks(radix_chunk(radix)) {
        let (mut value, mut power) = (0u128, 1u128);
        for x in chunk.iter().rev() {
            let digit = x.checked_add(h as i128).ok_or(Error::Encoding)?;
            if !(0..radix as i128).contains(&digit) {
                return Err(Error::Encoding);
            }
            value = value * radix + digit as u128;
            power *= radix;
        }
        writer.uniform(value, power)?;
    }
    Ok(())
}

/// The inverse of [`write_radix`] for `count` values: a chunk at or above $`R^k`$ is refused.
fn read_radix(reader: &mut BitReader<'_>, count: usize, h: u128) -> Result<Vec<i128>, Error> {
    let radix = 2 * h + 1;
    let k = radix_chunk(radix);
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let len = k.min(count - out.len());
        let power = (0..len).fold(1u128, |p, _| p * radix);
        let value = reader.uniform(power)?;
        push_radix_digits(&mut out, value, len, radix, h);
    }
    Ok(out)
}

/// The `len` digits of `value` $`<R^{len}`$ in base $`R=`$ `radix` from the least significant
/// one, each less `h`. Split by $`R^m<2^{64}`$ first, so that most digits come from 64-bit
/// divisions: the same digits as dividing `value` by $`R`$ `len` times.
fn push_radix_digits(out: &mut Vec<i128>, mut value: u128, len: usize, radix: u128, h: u128) {
    let (radix64, h) = (radix as u64, h as i128);
    if u128::from(radix64) != radix {
        for _ in 0..len {
            out.push((value % radix) as i128 - h);
            value /= radix;
        }
        return;
    }
    let (mut m, mut piece) = (1usize, radix);
    while let Some(next) = piece
        .checked_mul(radix)
        .filter(|x| *x <= u128::from(u64::MAX))
    {
        piece = next;
        m += 1;
    }
    let mut left = len;
    while left > 0 {
        let take = m.min(left);
        let mut low = if take == m {
            let low = value % piece;
            value /= piece;
            low as u64
        } else {
            value as u64
        };
        for _ in 0..take {
            out.push(i128::from(low % radix64) - h);
            low /= radix64;
        }
        left -= take;
    }
}

/// The final witness, one bit string zero padded to a byte: for each vector, the coefficients
/// of its polynomials that carry values ([`final_length`]), centred, in the code of its part.
/// The padding of $`v`$ is not sent.
///
/// - The parts of $`z_0`$ and of $`v`$, centred digits: one byte holding the least signed width
///   $`w`$ for their largest magnitude $`h`$, then $`h`$ in $`w-1`$ bits, then the values in
///   the radix code of base $`2h+1`$ ([`write_radix`], nothing when $`h=0`$).
/// - The parts of $`z_1`$: one byte holding the parameter $`p`$ of the Gaussian code with the
///   fewest bits for them, the least on ties ([`gaussian_parameter`]), then the values in that
///   code ([`BitWriter::gaussian`]).
///
/// Each value, and so each witness, has exactly one encoding, which [`decode_final`] checks.
pub(crate) fn encode_final(
    ring: &Rq,
    layout: &Layout,
    nu: usize,
    witness: &[Zeroizing<Vec<u64>>],
) -> Result<Vec<u8>, Error> {
    let d = ring.degree();
    let half = u128::from(ring.modulus() / 2);
    let mut writer = BitWriter::new();
    for (i, v) in witness.iter().enumerate() {
        let len = final_length(layout, nu, i) * d;
        debug_assert!(v[len..].iter().all(|x| *x == 0));
        let values: Vec<i128> = v[..len]
            .iter()
            .map(|x| i128::from(ring.center(*x)))
            .collect();
        if gaussian_part(nu, i) {
            let p = gaussian_parameter(&values);
            writer.unsigned(u128::from(p), 8)?;
            let limit = unary_limit(half, p);
            for x in &values {
                writer.gaussian(*x, p, limit)?;
            }
        } else {
            let h = values.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0);
            let w = signed_width(h);
            writer.unsigned(u128::from(w), 8)?;
            writer.unsigned(h, w - 1)?;
            if h > 0 {
                write_radix(&mut writer, &values, h)?;
            }
        }
    }
    let pad = (8 - writer.bit_len() % 8) % 8;
    writer.unsigned(0, pad as u32)?;
    writer.into_bytes()
}

/// The inverse of [`encode_final`], which consumes its bytes from `input`: refuses a width or
/// a Gaussian parameter that is not the one [`encode_final`] chooses for the decoded values, a
/// largest magnitude that no value reaches, a coefficient outside $`[-(q-1)/2,(q-1)/2]`$, a
/// chunk of the radix code out of range, and nonzero padding.
pub(crate) fn decode_final(
    ring: &Rq,
    layout: &Layout,
    nu: usize,
    input: &mut &[u8],
) -> Result<Vec<Zeroizing<Vec<u64>>>, Error> {
    let d = ring.degree();
    let half = u128::from(ring.modulus() / 2);
    let max_width = signed_width(half);
    let mut reader = BitReader::new(input);
    let mut out = Vec::with_capacity(layout.r_next);
    for i in 0..layout.r_next {
        let count = final_length(layout, nu, i) * d;
        let values = if gaussian_part(nu, i) {
            let p = reader.unsigned(8)? as u32;
            // The parameter chosen for magnitudes up to `half` is at most its bit length.
            if p > 128 - half.leading_zeros() {
                return Err(Error::Encoding);
            }
            let limit = unary_limit(half, p);
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                let x = reader.gaussian(p, limit)?;
                if x.unsigned_abs() > half {
                    return Err(Error::Encoding);
                }
                values.push(x);
            }
            if gaussian_parameter(&values) != p {
                return Err(Error::Encoding);
            }
            values
        } else {
            let w = reader.unsigned(8)? as u32;
            if !(2..=max_width).contains(&w) {
                return Err(Error::Encoding);
            }
            let h = reader.unsigned(w - 1)?;
            if h > half || signed_width(h) != w {
                return Err(Error::Encoding);
            }
            let values = match h {
                0 => vec![0; count],
                _ => read_radix(&mut reader, count, h)?,
            };
            if values.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0) != h {
                return Err(Error::Encoding);
            }
            values
        };
        let mut v: Vec<u64> = values.iter().map(|x| ring.lift_i128(*x)).collect();
        v.resize(layout.n_next * d, 0);
        out.push(Zeroizing::new(v));
    }
    let pad = (8 - reader.position() % 8) % 8;
    if reader.unsigned(pad as u32)? != 0 {
        return Err(Error::Encoding);
    }
    *input = &input[reader.position() / 8..];
    Ok(out)
}

#[cfg(test)]
mod tests;
