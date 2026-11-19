//! Deterministic PRGs and local-state samplers.
mod bernoulli;
mod cdf;
pub(crate) mod challenge;
mod gauss;
mod key;
mod prg;
pub mod reject;
mod uniform;
pub use challenge::{
    CHALLENGE_BYTES, MAX_CHALLENGE_DRAWS, challenge, challenge_from_bytes, eta_norm_power,
    within_eta,
};
pub use gauss::gaussian;

/// Largest Gaussian exponent $`t`$, for widths $`1.55\cdot2^t`$: the sampler, its variance, the
/// Gaussian code and the parameter check share this limit.
pub const MAX_LOG_SIGMA: u32 = 100;
pub(crate) use key::{derive_key, take_seed};
pub(crate) use prg::secret;
#[cfg(test)]
pub(crate) use prg::stream_log;
pub use prg::{AesPrg, ByteStream, ShakePrg, domain};
pub use uniform::{autostable, binomial, bounded, uniform, uniform_u256};
pub(crate) use uniform::{bits, uniform_poly};
