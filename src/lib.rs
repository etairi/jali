#![doc = include_str!("../README.md")]
// Without the `simd` feature, or off x86-64, no unsafe code is compiled at all. With it on
// x86-64, the one module `simd` holds the kernels and is the only place allowed unsafe code.
#![cfg_attr(
    not(all(feature = "simd", target_arch = "x86_64")),
    forbid(unsafe_code)
)]
#![cfg_attr(all(feature = "simd", target_arch = "x86_64"), deny(unsafe_code))]
#![deny(missing_docs)]

pub mod abdlop;
pub mod codec;
pub mod dcompress;
mod error;
pub mod labrador;
// The duplex sponge and codecs of draft-irtf-cfrg-fiat-shamir-04, under the transcript.
mod fiat_shamir;
#[cfg(test)]
mod golden_tests;
#[cfg(feature = "serde")]
mod json;
pub mod lin;
mod linhc;
pub mod lnp;
pub mod math;
mod par;
pub mod params;
pub mod quad;
pub mod quad_eval;
pub mod quad_many;
pub mod rand;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod simd;
mod sponge;
pub mod statement;
pub mod tbox;
#[cfg(feature = "test-utils")]
pub mod test_utils;
#[cfg(test)]
mod test_words;
mod transcript;
#[cfg(test)]
mod wide_tests;
mod wipe;

pub use error::Error;
pub use lnp::{Prover, Verifier};
