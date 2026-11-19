//! Exact arithmetic in $`\mathbb Z_q[X]/(X^d+1)`$.
mod constant;
mod containers;
pub mod int;
pub mod iso;
pub(crate) mod mod128;
pub mod ntt;
mod poly;
mod ring;
mod rns;
pub(crate) mod terms;

pub(crate) use constant::ConstantSum;
pub use containers::{PolyMat, PolyVec, SparsePolyMat, SparsePolyVec};
/// The 256-bit integer types of the coefficient and modulus API, from `crypto-bigint`.
pub use crypto_bigint::{I256, U256};
pub(crate) use poly::Coeffs;
pub use poly::Poly;
pub use ring::Ring;
pub(crate) use ring::SmallCrt;
pub use rns::{PolyNtt, RnsProduct};
