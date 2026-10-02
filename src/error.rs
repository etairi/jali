//! Errors never include witness values or randomness.

/// A rejected input or failed operation.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// Invalid public parameters.
    #[error("invalid parameter: {0}")]
    Parameter(&'static str),
    /// Operands belong to different rings.
    #[error("incompatible rings")]
    RingMismatch,
    /// An input has the wrong shape.
    #[error("incompatible dimensions")]
    Dimension,
    /// A bound cannot be represented by the arithmetic backend.
    #[error("arithmetic capacity exceeded")]
    Overflow,
    /// A byte string is not a canonical encoding.
    #[error("invalid encoding")]
    Encoding,
    /// An index is outside its declared space.
    #[error("index out of bounds")]
    Index,
    /// The source of randomness failed or exhausted its stream.
    #[error("randomness source failed")]
    Randomness,
    /// The witness does not satisfy its declared relation or bound.
    #[error("invalid witness")]
    Witness,
    /// A proof failed verification.
    #[error("invalid proof")]
    InvalidProof,
    /// A bounded rejection loop exhausted its allowed attempts.
    #[error("prover restart limit exceeded")]
    RestartLimit,
}
