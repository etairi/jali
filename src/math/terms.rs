//! Exact sums of products with fewer transforms: polynomials classified by shape, and sums of
//! transformed products reduced once per [`NADDS`] products.
//!
//! A product with a zero, scalar or monomial factor is computed in the coefficient domain
//! (`Poly::add_scaled_assign`, `Poly::add_monomial_product_assign`); other products are summed
//! as transformed products and reduced together. Every step is exact modulo $`q`$, as
//! [`Poly::mul`] and [`Poly::add`] are, so a sum computed this way is the canonical value that
//! a fold of `add` and `mul` returns, in whatever order its terms come.
use super::{Poly, PolyNtt, Ring, RnsProduct};
use crate::{Error, params::moduli::NADDS};
use crypto_bigint::U256;
use std::sync::Arc;
use zeroize::Zeroize;

/// The shape of a polynomial for products: zero, a scalar $`c`$, a monomial $`cX^k`$ with
/// $`1\le k<d`$, or general.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Shape {
    Zero,
    Scalar(U256),
    Monomial(usize, U256),
    General,
}
impl Shape {
    /// Classify the canonical coefficients of `p`, in variable time.
    pub(crate) fn of(p: &Poly) -> Self {
        let mut found = None;
        for (i, x) in p.coefficients().iter().enumerate() {
            if !x.is_zero_vartime() {
                if found.is_some() {
                    return Shape::General;
                }
                found = Some((i, *x));
            }
        }
        match found {
            None => Shape::Zero,
            Some((0, c)) => Shape::Scalar(c),
            Some((k, c)) => Shape::Monomial(k, c),
        }
    }
    /// `out += self * x`, where `self` is the shape of a polynomial $`a`$: exact for every
    /// shape but `General`, which returns `false` and leaves `out` to the caller.
    pub(crate) fn add_product_to(&self, out: &mut Poly, x: &Poly) -> Result<bool, Error> {
        match self {
            Shape::Zero => {}
            Shape::Scalar(c) => out.add_scaled_assign(x, c)?,
            Shape::Monomial(k, c) => out.add_monomial_product_assign(x, *k, c)?,
            Shape::General => return Ok(false),
        }
        Ok(true)
    }
}
/// The shapes of secret vectors hold coefficients, so buffers of them are wiped like `Poly`'s.
impl Zeroize for Shape {
    fn zeroize(&mut self) {
        match self {
            Shape::Scalar(c) => c.zeroize(),
            Shape::Monomial(k, c) => {
                k.zeroize();
                c.zeroize();
            }
            Shape::Zero | Shape::General => {}
        }
    }
}

/// A polynomial classified once and, if general, transformed once, for repeated products.
#[derive(Clone, Debug)]
pub(crate) enum Term {
    Cheap(Shape),
    General(PolyNtt),
}
impl Term {
    pub(crate) fn new(p: &Poly) -> Self {
        match Shape::of(p) {
            Shape::General => Term::General(p.to_ntt()),
            shape => Term::Cheap(shape),
        }
    }
}

/// A sum of transformed products, reduced into a polynomial whenever [`NADDS`] products are
/// pending, so that every reduction is exact (`Ring::with_modulus` sizes the RNS for that).
pub(crate) struct Accumulator {
    ring: Arc<Ring>,
    sum: Option<RnsProduct>,
}
impl Accumulator {
    pub(crate) fn new(ring: &Arc<Ring>) -> Self {
        Self {
            ring: ring.clone(),
            sum: None,
        }
    }
    /// Add $`ab`$ to the pending sum; when it already holds [`NADDS`] products, they are
    /// reduced and added to `spill` first.
    pub(crate) fn mac(&mut self, a: &PolyNtt, b: &PolyNtt, spill: &mut Poly) -> Result<(), Error> {
        let sum = self.sum.get_or_insert_with(|| RnsProduct::zero(&self.ring));
        if sum.count() == NADDS {
            spill.add_assign(&sum.reduce()?)?;
            *sum = RnsProduct::zero(&self.ring);
        }
        sum.mac(a, b)
    }
    /// Reduce the pending products, if any, and add them to `out`.
    pub(crate) fn finish_into(&mut self, out: &mut Poly) -> Result<(), Error> {
        if let Some(sum) = self.sum.take() {
            out.add_assign(&sum.reduce()?)?;
        }
        Ok(())
    }
}
