//! Absorbing segments equals absorbing their joined bytes, whatever the pieces: literal bytes
//! across chunks, zero runs held as counts, the empty string, and labels of every length.
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
}

#[test]
fn absorbing_segments_equals_absorbing_their_bytes() {
    let mut rng = Rng(5);
    let long_label = [7u8; 300];
    let labels: [&[u8]; 3] = [b"", b"equation", &long_label];
    for case in 0..300usize {
        let mut segments = Segments::default();
        for _ in 0..rng.below(12) {
            if rng.below(2) == 0 {
                let max = if rng.below(4) == 0 { 80_000 } else { 300 };
                let bytes: Vec<u8> = (0..rng.below(max)).map(|_| rng.next() as u8).collect();
                segments.push_literal(&bytes);
            } else {
                let max = if rng.below(4) == 0 { 80_000 } else { 300 };
                segments.push_zeros(rng.below(max));
            }
        }
        segments.shrink_to_fit();
        let bytes = segments.to_vec();
        let label = labels[case % 3];
        let mut joined = Transcript::new(b"t", b"parameters", &[case as u8; 32], b"statement");
        let mut pieces = joined.clone();
        joined.absorb(label, &bytes);
        pieces.absorb_segments(label, &segments);
        assert_eq!(joined, pieces, "case {case}");
        // The chain goes on alike.
        joined.absorb(b"next", &bytes[..bytes.len().min(5)]);
        pieces.absorb(b"next", &bytes[..bytes.len().min(5)]);
        assert_eq!(joined.challenge_seed(b"c"), pieces.challenge_seed(b"c"));
    }
}

#[test]
fn absorbing_empty_segments_equals_absorbing_no_bytes() {
    let start = Transcript::new(b"t", b"parameters", &[3; 32], b"statement");
    for label in [&b""[..], b"equation"] {
        let mut joined = start.clone();
        let mut pieces = start.clone();
        joined.absorb(label, b"");
        pieces.absorb_segments(label, &Segments::default());
        assert_eq!(joined, pieces);
    }
}
