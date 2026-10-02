//! Canonical LSB-first integer codes; hints use the prefix-free code of LNP22 Table 3.
//! Readers bound unary runs and reject noncanonical padding and trailing data.
mod bits;
pub mod proof;
pub use bits::{BitReader, BitWriter};

use crate::Error;

/// The longest proof encoding the decoders read: 16 MiB.
pub(crate) const MAX_PROOF_BYTES: usize = 16 * 1024 * 1024;

/// A reader over a proof encoding that refuses one longer than `MAX_PROOF_BYTES` before any
/// parsing. Both proof decoders start with it.
pub(crate) fn proof_reader(bytes: &[u8]) -> Result<BitReader<'_>, Error> {
    if bytes.len() > MAX_PROOF_BYTES {
        return Err(Error::Encoding);
    }
    Ok(BitReader::new(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_proof_reader_refuses_inputs_above_16_mib() {
        assert_eq!(MAX_PROOF_BYTES, 1 << 24);
        let bytes = vec![0xa5u8; MAX_PROOF_BYTES + 1];
        assert_eq!(proof_reader(&bytes).err(), Some(Error::Encoding));
        // At the cap the reader starts at bit zero of the input.
        let mut reader = proof_reader(&bytes[..MAX_PROOF_BYTES]).unwrap();
        assert_eq!(reader.unsigned(8), Ok(0xa5));
        assert!(proof_reader(&[]).is_ok());
    }
}
