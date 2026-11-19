//! Constant coefficients of sums of products, without computing the products: in
//! $`\mathbb Z_q[X]/(X^d+1)`$, $`\mathrm{ct}(ab)=a_0b_0-\sum_{j=1}^{d-1}a_jb_{d-j}`$, a dot
//! product of length $`d`$ instead of a transformed product.
use super::{
    Poly, Ring, int,
    mod128::{Mod128, add_mod128, sub_mod128},
    poly::Coeffs,
};
use crypto_bigint::U256;

/// A sum of constant coefficients, kept exactly and reduced modulo $`q`$ by `finish`.
pub(crate) struct ConstantSum<'a> {
    ring: &'a Ring,
    state: State,
}
enum State {
    /// $`q<2^{64}`$: the positive and the negative terms as 128-bit sums of products below
    /// $`q^2<2^{128}`$, each with the count of its carries out of 128 bits.
    Small {
        q: u64,
        positive: (u128, u64),
        negative: (u128, u64),
    },
    /// $`2^{64}\le q<2^{126}`$: a canonical sum of Barrett products.
    Wide { m: Mod128, sum: u128 },
    /// Larger moduli: a canonical sum of multi-limb products.
    General { sum: U256 },
}

#[inline(always)]
fn add_carry(acc: &mut (u128, u64), x: u128) {
    let (sum, carry) = acc.0.overflowing_add(x);
    acc.0 = sum;
    acc.1 += u64::from(carry);
}

/// $`(c2^{128}+s)\bmod q`$ for a 64-bit modulus. Both terms are reduced first, so their sum is
/// below $`2q<2^{65}`$ and is reduced in 128 bits.
fn reduce_carried((s, c): (u128, u64), q: u64) -> u64 {
    let q = u128::from(q);
    let two64 = (1u128 << 64) % q;
    let two128 = two64 * two64 % q;
    ((u128::from(c) % q * two128 % q + s % q) % q) as u64
}

/// The terms of $`\mathrm{ct}(ab)`$ below $`2^{64}`$: $`a_0b_0`$ to the positive sum, the
/// products $`a_jb_{d-j}`$ to the negative one, each below $`2^{128}`$.
#[inline(always)]
fn small_product<T>(
    positive: &mut (u128, u64),
    negative: &mut (u128, u64),
    a: &[T],
    b: &[T],
    low: impl Fn(&T) -> u128,
) {
    add_carry(positive, low(&a[0]) * low(&b[0]));
    for (x, y) in a[1..].iter().zip(b[1..].iter().rev()) {
        add_carry(negative, low(x) * low(y));
    }
}

impl<'a> ConstantSum<'a> {
    pub(crate) fn new(ring: &'a Ring) -> Self {
        let state = match (int::to_u64(&ring.q), ring.wide) {
            (Some(q), _) => State::Small {
                q,
                positive: (0, 0),
                negative: (0, 0),
            },
            (None, Some(m)) => State::Wide { m, sum: 0 },
            (None, None) => State::General { sum: U256::ZERO },
        };
        Self { ring, state }
    }
    /// Add a canonical value.
    pub(crate) fn add(&mut self, x: &U256) {
        match &mut self.state {
            State::Small { positive, .. } => add_carry(positive, u128::from(int::low_u64(x))),
            State::Wide { m, sum } => {
                *sum = add_mod128(*sum, int::low_u128(x), m.modulus());
            }
            State::General { sum } => *sum = self.ring.add(sum, x),
        }
    }
    /// Add $`\mathrm{ct}(ab)`$ for polynomials of the ring, with canonical coefficients.
    pub(crate) fn add_product(&mut self, a: &Poly, b: &Poly) {
        debug_assert_eq!(a.ring().degree(), b.ring().degree());
        match (&mut self.state, a.coeffs(), b.coeffs()) {
            (
                State::Small {
                    positive, negative, ..
                },
                Coeffs::Word(a),
                Coeffs::Word(b),
            ) => small_product(positive, negative, a, b, |x| u128::from(*x)),
            // 2^63 <= q < 2^64: two-word coefficients whose values fit a word.
            (
                State::Small {
                    positive, negative, ..
                },
                Coeffs::Double(a),
                Coeffs::Double(b),
            ) => small_product(positive, negative, a, b, |x| *x),
            (State::Wide { m, sum }, Coeffs::Double(a), Coeffs::Double(b)) => {
                let q = m.modulus();
                let mut s = add_mod128(*sum, m.mul(a[0], b[0]), q);
                for (x, y) in a[1..].iter().zip(b[1..].iter().rev()) {
                    s = sub_mod128(s, m.mul(*x, *y), q);
                }
                *sum = s;
            }
            (State::General { sum }, Coeffs::Wide(a), Coeffs::Wide(b)) => {
                let ring = self.ring;
                let mut s = ring.add(sum, &ring.mul(&a[0], &b[0]));
                for (x, y) in a[1..].iter().zip(b[1..].iter().rev()) {
                    s = ring.sub(&s, &ring.mul(x, y));
                }
                *sum = s;
            }
            _ => unreachable!("the ring decides the state and the coefficient width"),
        }
    }
    /// The canonical sum.
    pub(crate) fn finish(self) -> U256 {
        match self.state {
            State::Small {
                q,
                positive,
                negative,
            } => {
                let (p, n) = (reduce_carried(positive, q), reduce_carried(negative, q));
                U256::from_u64(if p >= n { p - n } else { p + (q - n) })
            }
            State::Wide { sum, .. } => U256::from_u128(sum),
            State::General { sum } => sum,
        }
    }
}

#[cfg(test)]
mod tests;
