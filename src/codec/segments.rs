//! Byte strings held in bounded chunks, for the equation encodings that transcripts absorb.
//!
//! The encoding of an equation (`QuadEq::to_bytes`) can take tens of megabytes, most of it the
//! zero codes of zero polynomials. [`Segments`] holds the same bytes in chunks of at most
//! [`CHUNK`] bytes, each allocated once with that capacity, and keeps long runs of zero bytes
//! as counts. `Transcript::absorb_segments` hashes them as `Transcript::absorb` hashes the
//! joined bytes.

#[cfg(test)]
mod tests;

/// The capacity that every chunk is allocated with, and so the largest buffer of a
/// [`Segments`] apart from its list of chunks. Chunks never grow;
/// [`Segments::shrink_to_fit`] shrinks the last one to its length.
///
/// 32 KiB is the buffer of a degree-1024 polynomial with 256-bit coefficients, a size class
/// that statements over such rings already use, and is far below the sizes that allocators
/// treat as large. On macOS, blocks above 2 MiB were observed to get regions of their own,
/// which the allocator may keep charged to the process after the block is freed; glibc's
/// allocator maps blocks from 128 KiB by default, a threshold that it raises with use up to
/// 32 MiB. Beyond its bytes, a chunk costs 24 bytes in the list of chunks, and a record its
/// header.
pub(crate) const CHUNK: usize = 32 * 1024;
/// Zero runs of at least this many bytes are kept as counts, shorter ones as bytes: a count
/// costs a record header, and from this length on it saves at least seven eighths of the run.
pub(crate) const MIN_ZERO_RUN: u64 = 64;
/// A record's header: the number of its literal bytes, which follow the header, and the number
/// of zero bytes that follow them in the string, as `LE32` each.
const HEADER: usize = 8;
/// The zero bytes that the pieces of zero runs are slices of.
static ZEROS: [u8; 4096] = [0; 4096];

/// A byte string, held as a sequence of records in chunks of at most [`CHUNK`] bytes.
///
/// Each record is a header, then literal bytes, which stand for themselves, then nothing: the
/// header's count of zero bytes stands for the run of zero bytes after the literal bytes. The
/// bytes of the string are those of the records in order.
#[derive(Debug, Default)]
pub(crate) struct Segments {
    chunks: Vec<Vec<u8>>,
    /// The offset of the last record's header in the last chunk.
    last: usize,
    /// The length of the byte string.
    len: u64,
}
impl Segments {
    /// The length of the byte string.
    pub(crate) fn len(&self) -> u64 {
        self.len
    }
    /// The literal and zero byte counts of the last record, if there is one.
    fn last_header(&self) -> Option<(u32, u32)> {
        let header = self.chunks.last()?.get(self.last..self.last + HEADER)?;
        let (literal, zeros) = header.split_at(4);
        Some((
            u32::from_le_bytes(literal.try_into().expect("4 bytes")),
            u32::from_le_bytes(zeros.try_into().expect("4 bytes")),
        ))
    }
    fn set_last_header(&mut self, literal: u32, zeros: u32) {
        let chunk = self.chunks.last_mut().expect("a record");
        chunk[self.last..self.last + 4].copy_from_slice(&literal.to_le_bytes());
        chunk[self.last + 4..self.last + HEADER].copy_from_slice(&zeros.to_le_bytes());
    }
    /// Start an empty record: in the last chunk if it has room for the header and one more
    /// byte, otherwise in a new chunk.
    fn start_record(&mut self) {
        if self
            .chunks
            .last()
            .is_none_or(|chunk| chunk.capacity() - chunk.len() <= HEADER)
        {
            self.chunks.push(Vec::with_capacity(CHUNK));
        }
        let chunk = self.chunks.last_mut().expect("a chunk");
        self.last = chunk.len();
        chunk.extend_from_slice(&[0; HEADER]);
    }
    /// Append `bytes`.
    pub(crate) fn push_literal(&mut self, mut bytes: &[u8]) {
        self.len += bytes.len() as u64;
        while !bytes.is_empty() {
            // The last record takes more literal bytes while no zero bytes follow its literal
            // bytes and its chunk has room; a chunk is never filled beyond its capacity.
            let room = self
                .chunks
                .last()
                .map_or(0, |chunk| chunk.capacity() - chunk.len());
            let literal = match self.last_header() {
                Some((literal, 0)) if room > 0 => literal,
                _ => {
                    self.start_record();
                    continue;
                }
            };
            let n = room.min(bytes.len());
            self.chunks
                .last_mut()
                .expect("a record")
                .extend_from_slice(&bytes[..n]);
            self.set_last_header(literal + n as u32, 0);
            bytes = &bytes[n..];
        }
    }
    /// Append `count` zero bytes, kept as a count.
    pub(crate) fn push_zeros(&mut self, mut count: u64) {
        self.len += count;
        while count > 0 {
            let (literal, zeros) = match self.last_header() {
                Some((literal, zeros)) if zeros < u32::MAX => (literal, zeros),
                _ => {
                    self.start_record();
                    continue;
                }
            };
            let n = count.min(u64::from(u32::MAX - zeros));
            self.set_last_header(literal, zeros + n as u32);
            count -= n;
        }
    }
    /// Shrink the last chunk to its length, once the string is complete.
    pub(crate) fn shrink_to_fit(&mut self) {
        if let Some(chunk) = self.chunks.last_mut() {
            chunk.shrink_to_fit();
        }
    }
    /// Pass the byte string to `f` in pieces, in order: the literal bytes of each record, then
    /// its zero bytes in slices of a static buffer of zeros. Empty pieces are left out.
    pub(crate) fn for_each_piece(&self, mut f: impl FnMut(&[u8])) {
        for chunk in &self.chunks {
            let mut at = 0;
            while at < chunk.len() {
                let (literal, zeros) = chunk[at..at + HEADER].split_at(4);
                let literal = u32::from_le_bytes(literal.try_into().expect("4 bytes")) as usize;
                let mut zeros = u32::from_le_bytes(zeros.try_into().expect("4 bytes")) as usize;
                at += HEADER;
                if literal > 0 {
                    f(&chunk[at..at + literal]);
                }
                at += literal;
                while zeros > 0 {
                    let n = zeros.min(ZEROS.len());
                    f(&ZEROS[..n]);
                    zeros -= n;
                }
            }
        }
    }
    /// The byte string, joined.
    #[cfg(test)]
    pub(crate) fn to_vec(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.for_each_piece(|piece| out.extend_from_slice(piece));
        out
    }
    /// The length and the capacity of each chunk.
    #[cfg(test)]
    pub(crate) fn chunk_shapes(&self) -> Vec<(usize, usize)> {
        self.chunks
            .iter()
            .map(|chunk| (chunk.len(), chunk.capacity()))
            .collect()
    }
}
