#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod abdlop;
pub mod codec;
pub mod dcompress;
mod error;
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
pub mod statement;
pub mod tbox;
#[cfg(feature = "test-utils")]
pub mod test_utils;
mod transcript;
#[cfg(test)]
mod wide_tests;

pub use error::Error;
pub use lnp::{Prover, Verifier};
