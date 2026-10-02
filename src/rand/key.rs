//! Secret keys derived from a caller's seed and the inputs of one call, in the manner of
//! deterministic signature nonces (EdDSA, RFC 6979). Calls that differ in any input read
//! different keys; identical calls read identical ones.
use shake::{ExtendableOutput, Shake128, Update, XofReader};
use zeroize::{Zeroize, Zeroizing};

/// Version tag, the first field of every derivation.
const PREFIX: &[u8] = b"jali/key/v1";

/// The first 32 bytes of SHAKE128 over $`f(\texttt{jali/key/v1})\,\|\,f(label)\,\|\,f(seed)\,\|
/// \,f(part_1)\,\|\cdots\|\,f(part_n)`$, where $`f(x)=\mathrm{LE64}(|x|)\,\|\,x`$.
///
/// Every field carries its length, so distinct field sequences give distinct inputs.
pub(crate) fn derive_key(label: &[u8], seed: &[u8; 32], parts: &[&[u8]]) -> Zeroizing<[u8; 32]> {
    let mut h = Shake128::default();
    for field in [PREFIX, label, seed.as_slice()]
        .into_iter()
        .chain(parts.iter().copied())
    {
        h.update(&(field.len() as u64).to_le_bytes());
        h.update(field);
    }
    let mut key = Zeroizing::new([0u8; 32]);
    h.finalize_xof().read(&mut *key);
    #[cfg(test)]
    super::stream_log::derived(seed, &key);
    key
}

/// Move a seed argument into a buffer that is wiped on drop, and wipe the argument.
pub(crate) fn take_seed(seed: &mut [u8; 32]) -> Zeroizing<[u8; 32]> {
    let out = Zeroizing::new(*seed);
    seed.zeroize();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_field_changes_the_key_and_boundaries_are_unambiguous() {
        let seed = [7; 32];
        let base = derive_key(b"label", &seed, &[b"ab", b"c"]);
        assert_eq!(*base, *derive_key(b"label", &seed, &[b"ab", b"c"]));
        for other in [
            derive_key(b"label2", &seed, &[b"ab", b"c"]),
            derive_key(b"label", &[8; 32], &[b"ab", b"c"]),
            derive_key(b"label", &seed, &[b"a", b"bc"]),
            derive_key(b"label", &seed, &[b"abc"]),
            derive_key(b"label", &seed, &[b"ab", b"c", b""]),
            derive_key(b"labelab", &seed, &[b"c"]),
        ] {
            assert_ne!(*base, *other);
        }
    }

    #[test]
    fn derivation_matches_an_independent_implementation_of_its_encoding() {
        // Python: f = lambda x: len(x).to_bytes(8, "little") + x; hashlib.shake_128(
        //   f(b"jali/key/v1") + f(b"label") + f(bytes([7] * 32)) + f(b"part one") + f(b"")
        // ).hexdigest(32)
        assert_eq!(
            hex::encode(*derive_key(b"label", &[7; 32], &[b"part one", b""])),
            "e3432c917d8998225fd16cb3800e9d04c4e5cdf101f160af497e41a805b6309f"
        );
    }
}
