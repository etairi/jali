//! Test helpers. `SeededRng` is a seeded `rand_core::CryptoRng` for the tests of the RNG
//! variants: SHAKE128 of a fixed label and seed, so every run draws the same bytes.
//! `SplitMix64` draws test data, `params` fits parameters to a statement shape and `ring`
//! holds ring helpers. For tests only; each test binary uses a different subset.
#![allow(dead_code)]
use shake::{ExtendableOutput, Shake128, Shake128Reader, Update, XofReader};

pub mod params;
pub mod ring;

pub struct SeededRng(Shake128Reader);
impl SeededRng {
    pub fn new(seed: &[u8]) -> Self {
        let mut h = Shake128::default();
        h.update(b"jali/tests/seeded-rng");
        h.update(seed);
        Self(h.finalize_xof())
    }
}
impl rand_core::TryRng for SeededRng {
    type Error = core::convert::Infallible;
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        let mut bytes = [0; 4];
        self.0.read(&mut bytes);
        Ok(u32::from_le_bytes(bytes))
    }
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        let mut bytes = [0; 8];
        self.0.read(&mut bytes);
        Ok(u64::from_le_bytes(bytes))
    }
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        self.0.read(dst);
        Ok(())
    }
}
impl rand_core::TryCryptoRng for SeededRng {}

/// SplitMix64: deterministic, non-cryptographic test data. `below` has a modulo bias of at
/// most $`n/2^{64}`$, which no test depends on.
pub struct SplitMix64(pub u64);
impl SplitMix64 {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// Coefficients in $`\{-1,0,1\}`$.
    pub fn ternary(&mut self, d: usize) -> Vec<i128> {
        (0..d).map(|_| self.below(3) as i128 - 1).collect()
    }
    /// Coefficients in $`\{0,1\}`$.
    pub fn binary(&mut self, d: usize) -> Vec<i128> {
        (0..d).map(|_| self.below(2) as i128).collect()
    }
    /// Coefficients in $`[0,q)`$, for $`q<2^{64}`$.
    pub fn uniform(&mut self, d: usize, q: i128) -> Vec<i128> {
        (0..d).map(|_| self.below(q as u64) as i128).collect()
    }
}

/// A 256-bit value that fits `u128`, as `u128`. It reads bytes, not limbs, so that the tests
/// also run with 32-bit limbs (`RUSTFLAGS='--cfg cpubits="32"'`); so does the next helper.
pub fn narrow(x: &jali::math::U256) -> u128 {
    let bytes = x.to_le_bytes();
    assert!(bytes[16..].iter().all(|b| *b == 0), "value above 2^128");
    u128::from_le_bytes(bytes[..16].try_into().unwrap())
}
/// A 256-bit value from four 64-bit words, least significant first.
pub fn from_u64_words(words: [u64; 4]) -> jali::math::U256 {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    jali::math::U256::from_le_slice(&bytes)
}
/// A modulus below $`2^{127}`$ as `i128`.
pub fn modulus(ring: &jali::math::Ring) -> i128 {
    i128::try_from(narrow(&ring.modulus())).unwrap()
}
/// Centred coefficients of a polynomial over a modulus below $`2^{127}`$.
pub fn values(p: &jali::math::Poly) -> Vec<i128> {
    p.coefficients_i128().unwrap().to_vec()
}
