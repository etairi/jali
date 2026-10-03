//! Segments against the joined bytes: random appends of literal bytes and zero runs give the same
//! string, in chunks allocated once with capacity `CHUNK` and packed to within a header of it; a
//! chunk exactly full, one byte short and one byte over; zero runs too long for one record.
use super::*;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

/// While a string is built, every chunk has the capacity it was allocated with.
fn assert_allocated_once(s: &Segments) {
    assert!(s.chunk_shapes().iter().all(|(_, cap)| *cap == CHUNK));
}

/// The chunks of a complete string: every chunk but the last has capacity `CHUNK` and was left
/// with at most a header's room; the last was shrunk to its length.
fn assert_complete(s: &Segments) {
    let shapes = s.chunk_shapes();
    if let Some(((last_len, last_cap), full)) = shapes.split_last() {
        for (len, cap) in full {
            assert_eq!(*cap, CHUNK);
            assert!((CHUNK - HEADER..=CHUNK).contains(len), "{len}");
        }
        assert_eq!(last_len, last_cap);
        assert!(*last_cap <= CHUNK);
    }
}

#[test]
fn random_appends_give_the_joined_bytes() {
    let mut rng = Rng(41);
    let mut lengths = [0u64; 4];
    for case in 0..250 {
        let mut s = Segments::default();
        let mut joined = Vec::new();
        for _ in 0..rng.below(40) {
            match rng.below(5) {
                0 | 1 => {
                    let n = match rng.below(4) {
                        0 => rng.below(16),
                        1 => rng.below(700),
                        2 => (CHUNK - HEADER) as u64 - 2 + rng.below(5),
                        _ => rng.below(3 * CHUNK as u64),
                    } as usize;
                    // Literal zero bytes stay literal.
                    let bytes = if rng.below(4) == 0 {
                        vec![0; n]
                    } else {
                        rng.bytes(n)
                    };
                    s.push_literal(&bytes);
                    joined.extend_from_slice(&bytes);
                    lengths[0] += n as u64;
                }
                2 | 3 => {
                    let n = match rng.below(3) {
                        0 => rng.below(MIN_ZERO_RUN),
                        1 => rng.below(1000),
                        _ => rng.below(3 * CHUNK as u64),
                    };
                    s.push_zeros(n);
                    joined.resize(joined.len() + n as usize, 0);
                    lengths[1] += n;
                }
                _ => {
                    let n = 1 + rng.below(8) as usize;
                    let bytes = rng.bytes(n);
                    s.push_literal(&bytes);
                    joined.extend_from_slice(&bytes);
                    lengths[2] += 1;
                }
            }
            assert_allocated_once(&s);
            assert_eq!(s.len(), joined.len() as u64, "case {case}");
        }
        lengths[3] += s.chunk_shapes().len() as u64;
        s.shrink_to_fit();
        assert_complete(&s);
        assert_eq!(s.to_vec(), joined, "case {case}");
        s.for_each_piece(|piece| assert!(!piece.is_empty()));
    }
    // Many literal and zero bytes, short literals and chunks were appended.
    assert!(lengths[0] > 1 << 24 && lengths[1] > 1 << 24, "{lengths:?}");
    assert!(lengths[2] > 500 && lengths[3] > 500, "{lengths:?}");
}

#[test]
fn chunks_fill_exactly_and_overflow_by_one_record() {
    let byte = |n: usize| vec![0xa5u8; n];
    // (literal bytes before a zero run, the chunk shapes at the end): a run needs no room, so it
    // joins the last record, and the literal byte after it needs a new record.
    for (first, shapes) in [
        // Exactly full: the next record starts a new chunk.
        (
            CHUNK - HEADER,
            vec![(CHUNK, CHUNK), (HEADER + 1, HEADER + 1)],
        ),
        // One byte short: room for no header.
        (
            CHUNK - HEADER - 1,
            vec![(CHUNK - 1, CHUNK), (HEADER + 1, HEADER + 1)],
        ),
        // A header's room: still no room for a header and a byte.
        (
            CHUNK - 2 * HEADER,
            vec![(CHUNK - HEADER, CHUNK), (HEADER + 1, HEADER + 1)],
        ),
        // Room for a header and a byte: the chunk ends exactly full.
        (CHUNK - 2 * HEADER - 1, vec![(CHUNK, CHUNK)]),
        // One byte over: the second chunk takes it with a header of its own.
        (
            CHUNK - HEADER + 1,
            vec![(CHUNK, CHUNK), (2 * HEADER + 2, 2 * HEADER + 2)],
        ),
    ] {
        let mut s = Segments::default();
        s.push_literal(&byte(first));
        s.push_zeros(1000);
        s.push_literal(&byte(1));
        assert_allocated_once(&s);
        s.shrink_to_fit();
        assert_eq!(s.chunk_shapes(), shapes, "first {first}");
        let mut joined = byte(first);
        joined.resize(first + 1000, 0);
        joined.extend_from_slice(&byte(1));
        assert_eq!(s.len(), joined.len() as u64);
        assert_eq!(s.to_vec(), joined, "first {first}");
    }
    // One literal piece across three chunks, each with its own header.
    let mut s = Segments::default();
    let bytes = Rng(3).bytes(2 * CHUNK);
    s.push_literal(&bytes);
    s.shrink_to_fit();
    assert_eq!(
        s.chunk_shapes(),
        [(CHUNK, CHUNK), (CHUNK, CHUNK), (3 * HEADER, 3 * HEADER)]
    );
    assert_eq!(s.to_vec(), bytes);
    // The empty string has no chunk and no piece.
    let mut s = Segments::default();
    s.push_literal(&[]);
    s.push_zeros(0);
    s.shrink_to_fit();
    assert_eq!((s.len(), s.chunk_shapes()), (0, vec![]));
    s.for_each_piece(|_| panic!("a piece of the empty string"));
}

#[test]
fn zero_runs_beyond_a_record_take_several_records() {
    let mut s = Segments::default();
    s.push_literal(&[1, 2, 3]);
    let run = u64::from(u32::MAX) + 10;
    s.push_zeros(run);
    s.push_literal(&[4]);
    let header = |literal: u32, zeros: u32| [literal.to_le_bytes(), zeros.to_le_bytes()].concat();
    let expected = [
        header(3, u32::MAX),
        vec![1, 2, 3],
        header(0, 10),
        header(1, 0),
        vec![4],
    ]
    .concat();
    assert_eq!(s.chunks, [expected]);
    assert_eq!(s.len(), run + 4);
    // The pieces: the literal bytes, then the run in slices of the zero buffer, then the last
    // byte. The run is only measured, not read.
    let (mut pieces, mut total) = (Vec::new(), 0u64);
    s.for_each_piece(|piece| {
        total += piece.len() as u64;
        if piece.as_ptr() != ZEROS.as_ptr() {
            pieces.push(piece.to_vec());
        }
    });
    assert_eq!(total, run + 4);
    assert_eq!(pieces, [vec![1, 2, 3], vec![4]]);
}
