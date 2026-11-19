//! The sponge against the `shake` crate: every length up to four blocks and some longer, the
//! input in random pieces, the output read in random pieces across blocks, and clones taken in
//! the middle of absorbing.
use super::*;
use shake::{ExtendableOutput, Update, XofReader};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn reference(input: &[u8], len: usize) -> Vec<u8> {
    let mut h = shake::Shake128::default();
    h.update(input);
    let mut out = vec![0u8; len];
    h.finalize_xof().read(&mut out);
    out
}

/// `data` absorbed in pieces of random lengths (empty ones included).
fn absorb_in_pieces(rng: &mut Rng, h: &mut Shake128, mut data: &[u8]) {
    while !data.is_empty() {
        let take = rng.below(data.len().min(400) + 1);
        h.update(&data[..take]);
        data = &data[take..];
    }
}

/// `len` bytes read in pieces of random lengths.
fn read_in_pieces(rng: &mut Rng, r: &mut Shake128Reader, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    let mut k = 0;
    while k < len {
        let take = 1 + rng.below((len - k).min(300));
        r.read(&mut out[k..k + take]);
        k += take;
    }
    out
}

#[test]
fn equals_the_shake_crate() {
    let mut rng = Rng(0x5a5a_0001);
    let lengths = (0..=4 * RATE).chain([1000, 4096, 10_000, 65_536 + 7]);
    for len in lengths {
        let data: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        let out_len = [0, 1, 31, 32, 167, 168, 169, 500, 2000][rng.below(9)];
        let expect = reference(&data, out_len);
        let mut h = Shake128::default();
        absorb_in_pieces(&mut rng, &mut h, &data);
        assert_eq!(
            read_in_pieces(&mut rng, &mut h.finalize_xof(), out_len),
            expect,
            "len {len}"
        );
        // At once, as the transcript and the key derivation do.
        let mut h = Shake128::default();
        h.update(&data);
        let mut out = vec![0u8; out_len];
        h.finalize_xof().read(&mut out);
        assert_eq!(out, expect, "len {len}");
    }
}

#[test]
fn a_clone_continues_on_its_own() {
    let mut rng = Rng(0x5a5a_0002);
    for _ in 0..200 {
        let first: Vec<u8> = (0..rng.below(700)).map(|_| rng.next() as u8).collect();
        let second: Vec<u8> = (0..rng.below(700)).map(|_| rng.next() as u8).collect();
        let mut h = Shake128::default();
        absorb_in_pieces(&mut rng, &mut h, &first);
        let snapshot = h.clone();
        absorb_in_pieces(&mut rng, &mut h, &second);
        let both = [first.as_slice(), second.as_slice()].concat();
        assert_eq!(
            read_in_pieces(&mut rng, &mut h.finalize_xof(), 400),
            reference(&both, 400)
        );
        assert_eq!(
            read_in_pieces(&mut rng, &mut snapshot.finalize_xof(), 400),
            reference(&first, 400)
        );
    }
}
