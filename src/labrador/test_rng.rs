//! A small deterministic generator for test data (SplitMix64). Not for keys or protocol
//! randomness.

/// SplitMix64 over a 64-bit state.
pub(crate) struct TestRng(u64);

impl TestRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A value in $`[0,m)`$ (slightly biased, which test data tolerates).
    pub(crate) fn below(&mut self, m: u64) -> u64 {
        ((u128::from(self.next_u64()) * u128::from(m)) >> 64) as u64
    }

    /// A value in $`[-b,b]`$.
    pub(crate) fn small(&mut self, b: u64) -> i64 {
        self.below(2 * b + 1) as i64 - b as i64
    }
}
