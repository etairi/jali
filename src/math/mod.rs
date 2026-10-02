//! Exact arithmetic in $`\mathbb Z_q[X]/(X^d+1)`$.
mod containers;
pub mod int;
pub mod iso;
pub mod ntt;
mod poly;
mod ring;
mod rns;
pub(crate) mod terms;

pub use containers::{PolyMat, PolyVec, SparsePolyMat, SparsePolyVec};
/// The 256-bit integer types of the coefficient and modulus API, from `crypto-bigint`.
pub use crypto_bigint::{I256, U256};
pub use poly::Poly;
pub use ring::Ring;
pub use rns::{PolyNtt, RnsProduct};
