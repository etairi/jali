//! 256-bit test values from 64-bit words and back, through bytes, so that tests read the same
//! with `crypto-bigint`'s 64-bit and 32-bit limbs.

/// A 256-bit value from four 64-bit words, least significant first, through its bytes: the
/// same value whether `crypto-bigint`'s limbs have 64 or 32 bits.
pub(crate) fn u256_from_u64s(words: [u64; 4]) -> crate::math::U256 {
    let mut bytes = [0u8; 32];
    for (chunk, w) in bytes.as_chunks_mut::<8>().0.iter_mut().zip(words) {
        *chunk = w.to_le_bytes();
    }
    crate::math::U256::from_le_slice(&bytes)
}

/// The low and high 128 bits of a 256-bit value, through its bytes.
pub(crate) fn u256_halves(x: &crate::math::U256) -> (u128, u128) {
    let bytes = x.to_le_bytes();
    let half = |b: &[u8]| u128::from_le_bytes(b.try_into().expect("16 bytes"));
    (half(&bytes[..16]), half(&bytes[16..]))
}
