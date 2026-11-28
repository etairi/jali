//! A quadratic relation with automorphisms: LNP22 Fig. 6, Eqs. 29–31.
//!
//! On its own the proof runs the transcript `JALI-V02-LNP22-QUAD-C1-SHAKE128`, whose instance
//! is the parameters, the public seed, the commitment and the equation. Inside the many-equation
//! and evaluation proofs it continues their transcript after the folding challenge. Either way
//! its prover message is $`(t,v,w_1)`$, with the LinHC mask commitment before $`w_1`$ in the
//! extractable mode, and its verifier message the challenge (`docs/encodings.md`).
use crate::wipe::Secret;
use crate::{
    Error,
    abdlop::{self, Abdlop, Caller, Commitment, Opening, OpeningProof},
    codec::{BitWriter, CodeWriter, SegmentWriter, Segments, fields},
    linhc,
    math::{
        ConstantSum, Poly, PolyNtt, PolyVec, Ring, SparsePolyMat, SparsePolyVec, U256, int,
        terms::{Accumulator, Shape, Term},
    },
    rand::take_seed,
    transcript::{Protocol, Transcript},
};
use std::{collections::BTreeMap, sync::Arc};

#[cfg(test)]
mod constant_tests;
#[cfg(test)]
pub(crate) mod encoding_tests;
#[cfg(test)]
mod expansion_tests;
#[cfg(test)]
mod extraction_tests;
#[cfg(test)]
mod in_place_tests;
#[cfg(test)]
mod mask_tests;
#[cfg(test)]
mod substitute_tests;

/// A weight of [`QuadEq::combination`]: classified once and, if general, transformed once.
struct Weight<'a> {
    poly: &'a Poly,
    shape: Shape,
    ntt: Option<PolyNtt>,
}
impl<'a> Weight<'a> {
    fn new(poly: &'a Poly) -> Self {
        let shape = Shape::of(poly);
        let ntt = (shape == Shape::General).then(|| poly.to_ntt());
        Self { poly, shape, ntt }
    }
}

/// $`\sum_iw_ip_i`$ over the terms $`(i,p_i)`$, exactly: a product with a cheap factor in the
/// coefficient domain, the general ones summed transformed ([`Accumulator`]).
fn combine_terms<'p>(
    ring: &Arc<Ring>,
    weights: &[Weight<'_>],
    terms: impl Iterator<Item = (usize, &'p Poly)>,
) -> Result<Poly, Error> {
    let mut out = Poly::zero(ring.clone());
    let mut sum = Accumulator::new(ring);
    for (i, p) in terms {
        let w = &weights[i];
        if w.shape.add_product_to(&mut out, p)? {
            continue;
        }
        match Shape::of(p) {
            Shape::General => {
                let b = w.ntt.as_ref().expect("general weights are transformed");
                sum.mac(&p.to_ntt(), b, &mut out)?;
            }
            cheap => {
                cheap.add_product_to(&mut out, w.poly)?;
            }
        }
    }
    sum.finish_into(&mut out)?;
    Ok(out)
}

/// Every key of the sorted maps `maps`, the `i`-th weighted by `weights[i]`, with
/// [`combine_terms`] of its polynomials there, or `None` where each map that has the key holds
/// an explicit zero.
fn combine_maps<'p, K: Ord + Copy + Send + Sync>(
    ring: &Arc<Ring>,
    weights: &[Weight<'_>],
    maps: impl Iterator<Item = impl Iterator<Item = (K, Option<&'p Poly>)>>,
) -> Result<Vec<(K, Option<Poly>)>, Error> {
    let mut slots: Vec<(K, usize, Option<&Poly>)> = maps
        .enumerate()
        .flat_map(|(i, map)| map.map(move |(key, p)| (key, i, p)))
        .collect();
    // A stable sort merges the sorted runs of the maps.
    slots.sort_by_key(|slot| slot.0);
    let groups: Vec<&[(K, usize, Option<&Poly>)]> = slots.chunk_by(|x, y| x.0 == y.0).collect();
    crate::par::try_map_min(groups.len(), 16, |g| {
        let group = groups[g];
        let terms = group.iter().filter_map(|&(_, i, p)| p.map(|p| (i, p)));
        let value = if group.iter().all(|slot| slot.2.is_none()) {
            None
        } else {
            Some(combine_terms(ring, weights, terms)?)
        };
        Ok((group[0].0, value))
    })
}

/// The term of the constant of `map[k]`, made on first use.
fn map_constant<'t>(constants: &'t mut [Option<Term>], map: &[Affine], k: usize) -> &'t Term {
    constants[k].get_or_insert_with(|| Term::new(&map[k].constant))
}

/// $`xy`$ exactly, the value of [`Poly::mul`]: two general factors multiply transformed, from
/// `ty` and the term of `x` in `tx` (made here if it is not yet), a cheap one in the coefficient
/// domain.
fn product_of(x: &Poly, tx: &mut Option<Term>, y: &Poly, ty: &Term) -> Result<Poly, Error> {
    if let Term::General(b) = ty
        && let Term::General(a) = tx.get_or_insert_with(|| Term::new(x))
    {
        return a.product(b)?.reduce();
    }
    x.mul(y)
}

/// `out += xy` exactly: in place when `y` is zero, a scalar or a monomial, else by
/// [`Poly::mul`].
fn add_product(out: &mut Poly, x: &Poly, y: &Poly) -> Result<(), Error> {
    if !Shape::of(y).add_product_to(out, x)? {
        out.add_assign(&x.mul(y)?)?;
    }
    Ok(())
}

/// `out += xy` exactly, `ty` the term of `y`: a cheap factor in place, two general ones summed
/// transformed into `sum`, which adds its products to `out` when it is full.
fn add_to_sum(
    sum: &mut Accumulator,
    out: &mut Poly,
    x: &Poly,
    y: &Poly,
    ty: &Term,
) -> Result<(), Error> {
    match ty {
        Term::Cheap(shape) => {
            shape.add_product_to(out, x)?;
            Ok(())
        }
        Term::General(b) => match Shape::of(x) {
            Shape::General => sum.mac(&x.to_ntt(), b, out),
            shape => {
                shape.add_product_to(out, y)?;
                Ok(())
            }
        },
    }
}

/// An affine form $`\sum_uc_uy_u+C`$ that [`QuadEq::substitute`] puts in place of a variable:
/// `terms` lists the coordinates $`u`$ with their coefficients $`c_u`$.
#[derive(Clone, Debug)]
pub(crate) struct Affine {
    pub(crate) terms: Vec<(u16, Poly)>,
    pub(crate) constant: Poly,
}

/// An upper-triangular quadratic form in the interleaved variable/conjugate space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuadEq {
    /// Quadratic coefficients.
    pub r2: SparsePolyMat,
    /// Linear coefficients.
    pub r1: SparsePolyVec,
    /// Constant term.
    pub r0: Poly,
}
impl QuadEq {
    /// Product of two affine forms; rejects forms that already contain quadratic terms.
    pub fn product_affine(&self, rhs: &Self) -> Result<Self, Error> {
        let ring = self.r0.ring().clone();
        let dim = self.r2.dimension();
        self.check(&ring, dim)?;
        rhs.check(&ring, dim)?;
        if self.r2.entries().any(|(_, p)| !p.is_zero())
            || rhs.r2.entries().any(|(_, p)| !p.is_zero())
        {
            return Err(Error::Parameter("expression degree exceeds two"));
        }
        let mut matrix = BTreeMap::new();
        for (i, a) in self.r1.entries() {
            for (j, b) in rhs.r1.entries() {
                let entry = matrix
                    .entry((i.min(j), i.max(j)))
                    .or_insert_with(|| Poly::zero(ring.clone()));
                *entry = entry.add(&a.mul(b)?)?;
            }
        }
        let mut out = Self::zero(ring.clone(), dim)?;
        out.r2 = SparsePolyMat::new(
            ring,
            dim,
            matrix.into_iter().map(|((i, j), p)| (i, j, p)).collect(),
        )?;
        out = out.add(&self.scale(&rhs.r0)?)?.add(&rhs.scale(&self.r0)?)?;
        out.r0 = self.r0.mul(&rhs.r0)?;
        Ok(out)
    }
    /// Remap witness indices, adding terms that collide under the map.
    pub fn remap(&self, indices: &[usize], dimension: usize) -> Result<Self, Error> {
        let ring = self.r0.ring().clone();
        self.check(&ring, indices.len())?;
        if indices
            .iter()
            .any(|i| *i >= dimension || *i > u16::MAX as usize)
        {
            return Err(Error::Index);
        }
        let mut matrix = BTreeMap::new();
        let mut vector = BTreeMap::new();
        for ((i, j), p) in self.r2.entries() {
            let (i, j) = (
                indices[usize::from(i)] as u16,
                indices[usize::from(j)] as u16,
            );
            let entry = matrix
                .entry((i.min(j), i.max(j)))
                .or_insert_with(|| Poly::zero(ring.clone()));
            *entry = entry.add(p)?;
        }
        for (i, p) in self.r1.entries() {
            let entry = vector
                .entry(indices[usize::from(i)] as u16)
                .or_insert_with(|| Poly::zero(ring.clone()));
            *entry = entry.add(p)?;
        }
        Ok(Self {
            r2: SparsePolyMat::new(
                ring.clone(),
                dimension,
                matrix.into_iter().map(|((i, j), p)| (i, j, p)).collect(),
            )?,
            r1: SparsePolyVec::new(ring, dimension, vector.into_iter().collect())?,
            r0: self.r0.clone(),
        })
    }
    /// Put the affine form `map[i]` in place of variable $`i`$ and expand, in a space of
    /// `dimension` coordinates. For $`x_i=\sum_uc_uy_u+C_i`$, an r1 entry $`(i,a)`$ gives
    /// $`a\,c_u`$ at every term $`u`$ and $`a\,C_i`$ to the constant; an r2 entry $`(i,j,a)`$
    /// gives $`a\,c_uc_v`$ at $`(\min(u,v),\max(u,v))`$ for every term $`u`$ of $`x_i`$ and $`v`$
    /// of $`x_j`$ (so a square's cross terms land twice on one key), $`a\,c_uC_j`$ at $`u`$,
    /// $`a\,c_vC_i`$ at $`v`$ and $`a\,C_iC_j`$ to the constant. Colliding terms add up. A zero
    /// constant creates no entry, so a map of single unit terms gives [`QuadEq::remap`]'s form.
    pub(crate) fn substitute(&self, map: &[Affine], dimension: usize) -> Result<Self, Error> {
        let ring = self.r0.ring().clone();
        self.check(&ring, map.len())?;
        for affine in map {
            if affine.constant.ring() != &ring
                || affine.terms.iter().any(|(_, c)| c.ring() != &ring)
            {
                return Err(Error::RingMismatch);
            }
            if affine
                .terms
                .iter()
                .any(|(u, _)| usize::from(*u) >= dimension)
            {
                return Err(Error::Index);
            }
        }
        let mut matrix = BTreeMap::new();
        let mut vector = BTreeMap::new();
        let mut constant = self.r0.clone();
        let zero = || Poly::zero(ring.clone());
        // Every product is exact modulo q, so the sums below are the canonical values that a
        // fold of `mul` and `add` gives. The constants of the map meet every entry that names
        // their variable: each is classified and, if general, transformed once, on first use.
        // The products that land on the constant term are summed transformed and reduced
        // together, and a product with a cheap factor is added in place.
        let mut constants: Vec<Option<Term>> = (0..map.len()).map(|_| None).collect();
        let mut constant_sum = Accumulator::new(&ring);
        for ((i, j), a) in self.r2.entries() {
            let (x, y) = (&map[usize::from(i)], &map[usize::from(j)]);
            for (u, cu) in &x.terms {
                let acu = a.mul(cu)?;
                for (v, cv) in &y.terms {
                    add_product(
                        matrix
                            .entry(((*u).min(*v), (*u).max(*v)))
                            .or_insert_with(zero),
                        &acu,
                        cv,
                    )?;
                }
            }
            // The term of `a`, made when a general constant first needs it.
            let mut ta = None;
            if !y.constant.is_zero() {
                let ty = map_constant(&mut constants, map, usize::from(j));
                let ay = product_of(a, &mut ta, &y.constant, ty)?;
                for (u, cu) in &x.terms {
                    add_product(vector.entry(*u).or_insert_with(zero), &ay, cu)?;
                }
                if !x.constant.is_zero() {
                    let tx = map_constant(&mut constants, map, usize::from(i));
                    add_to_sum(&mut constant_sum, &mut constant, &ay, &x.constant, tx)?;
                }
            }
            if !x.constant.is_zero() {
                let tx = map_constant(&mut constants, map, usize::from(i));
                let ax = product_of(a, &mut ta, &x.constant, tx)?;
                for (v, cv) in &y.terms {
                    add_product(vector.entry(*v).or_insert_with(zero), &ax, cv)?;
                }
            }
        }
        for (i, a) in self.r1.entries() {
            let x = &map[usize::from(i)];
            for (u, cu) in &x.terms {
                add_product(vector.entry(*u).or_insert_with(zero), a, cu)?;
            }
            if !x.constant.is_zero() {
                let tx = map_constant(&mut constants, map, usize::from(i));
                add_to_sum(&mut constant_sum, &mut constant, a, &x.constant, tx)?;
            }
        }
        constant_sum.finish_into(&mut constant)?;
        Ok(Self {
            r2: SparsePolyMat::new(
                ring.clone(),
                dimension,
                matrix.into_iter().map(|((i, j), p)| (i, j, p)).collect(),
            )?,
            r1: SparsePolyVec::new(ring, dimension, vector.into_iter().collect())?,
            r0: constant,
        })
    }
    /// Zero form in an explicitly allocated variable space.
    pub fn zero(ring: Arc<Ring>, dimension: usize) -> Result<Self, Error> {
        Ok(Self {
            r2: SparsePolyMat::new(ring.clone(), dimension, vec![])?,
            r1: SparsePolyVec::new(ring.clone(), dimension, vec![])?,
            r0: Poly::zero(ring),
        })
    }
    /// Multiply every coefficient by a public ring element.
    pub fn scale(&self, weight: &Poly) -> Result<Self, Error> {
        let ring = self.r0.ring().clone();
        let dim = self.r2.dimension();
        self.check(&ring, dim)?;
        Ok(Self {
            r2: SparsePolyMat::new(
                ring.clone(),
                dim,
                self.r2
                    .entries()
                    .map(|((r, c), p)| Ok((r, c, p.mul(weight)?)))
                    .collect::<Result<_, Error>>()?,
            )?,
            r1: SparsePolyVec::new(
                ring,
                dim,
                self.r1
                    .entries()
                    .map(|(i, p)| Ok((i, p.mul(weight)?)))
                    .collect::<Result<_, Error>>()?,
            )?,
            r0: self.r0.mul(weight)?,
        })
    }
    /// Add forms, summing entries with the same sparse indices.
    pub fn add(&self, rhs: &Self) -> Result<Self, Error> {
        let ring = self.r0.ring().clone();
        let dim = self.r2.dimension();
        self.check(&ring, dim)?;
        rhs.check(&ring, dim)?;
        let mut matrix: BTreeMap<_, _> = self.r2.entries().map(|(i, p)| (i, p.clone())).collect();
        let mut vector: BTreeMap<_, _> = self.r1.entries().map(|(i, p)| (i, p.clone())).collect();
        for (index, p) in rhs.r2.entries() {
            let entry = matrix
                .entry(index)
                .or_insert_with(|| Poly::zero(ring.clone()));
            *entry = entry.add(p)?;
        }
        for (index, p) in rhs.r1.entries() {
            let entry = vector
                .entry(index)
                .or_insert_with(|| Poly::zero(ring.clone()));
            *entry = entry.add(p)?;
        }
        Ok(Self {
            r2: SparsePolyMat::new(
                ring.clone(),
                dim,
                matrix.into_iter().map(|((r, c), p)| (r, c, p)).collect(),
            )?,
            r1: SparsePolyVec::new(ring, dim, vector.into_iter().collect())?,
            r0: self.r0.add(&rhs.r0)?,
        })
    }
    /// `self += rhs`, in place: the keys and values of [`QuadEq::add`].
    pub(crate) fn add_assign(&mut self, rhs: &Self) -> Result<(), Error> {
        let ring = self.r0.ring().clone();
        let dim = self.r2.dimension();
        self.check(&ring, dim)?;
        rhs.check(&ring, dim)?;
        for ((r, c), p) in rhs.r2.entries() {
            self.r2.add_at(r, c, p)?;
        }
        for (i, p) in rhs.r1.entries() {
            self.r1.add_at(i, p)?;
        }
        self.r0.add_assign(&rhs.r0)
    }
    /// `self += w rhs` for a scalar $`w`$, in place: the keys and values of
    /// `self.add(&rhs.scale(&Poly::constant_u256(ring, w))?)`. Every key of `rhs` is kept, also
    /// where its entry is zero; zero entries are not multiplied.
    pub(crate) fn add_scaled_scalar_assign(&mut self, rhs: &Self, w: &U256) -> Result<(), Error> {
        let ring = self.r0.ring().clone();
        let dim = self.r2.dimension();
        self.check(&ring, dim)?;
        rhs.check(&ring, dim)?;
        let w = ring.reduce(w);
        for ((r, c), p) in rhs.r2.entries() {
            if p.is_zero() {
                self.r2.insert_key(r, c)?;
            } else {
                self.r2.entry_mut(r, c)?.add_scaled_assign(p, &w)?;
            }
        }
        for (i, p) in rhs.r1.entries() {
            if p.is_zero() {
                self.r1.insert_key(i)?;
            } else {
                self.r1.entry_mut(i)?.add_scaled_assign(p, &w)?;
            }
        }
        self.r0.add_scaled_assign(&rhs.r0, &w)
    }
    /// `self += weight rhs`, in place: the keys and values of `self.add(&rhs.scale(weight)?)`.
    /// The weight is classified and, if general, transformed once; every key of `rhs` is kept.
    /// The keys are inserted first, then the products are added, each entry on its own, in
    /// parallel with the `parallel` feature when there are at least 64 entries.
    pub(crate) fn add_scaled_assign(&mut self, rhs: &Self, weight: &Poly) -> Result<(), Error> {
        let ring = self.r0.ring().clone();
        let dim = self.r2.dimension();
        self.check(&ring, dim)?;
        rhs.check(&ring, dim)?;
        if weight.ring() != &ring {
            return Err(Error::RingMismatch);
        }
        let shape = Shape::of(weight);
        let transformed = (shape == Shape::General).then(|| weight.to_ntt());
        let add_product = |entry: &mut Poly, p: &Poly| -> Result<(), Error> {
            if shape.add_product_to(entry, p)? {
                return Ok(());
            }
            match Shape::of(p) {
                Shape::General => {
                    let w = transformed.as_ref().expect("general weight is transformed");
                    entry.add_assign(&p.to_ntt().product(w)?.reduce()?)
                }
                cheap => cheap.add_product_to(entry, weight).map(|_| ()),
            }
        };
        let mut pairs = self.r2.pairs_mut(&rhs.r2)?;
        pairs.extend(self.r1.pairs_mut(&rhs.r1)?);
        pairs.push((&mut self.r0, &rhs.r0));
        crate::par::try_for_each_min(pairs, 64, |(entry, p)| add_product(entry, p))
    }
    /// $`\sum_iw_ie_i`$ over `equations` and `weights`, with the keys and values that
    /// [`QuadEq::add_scaled_assign`] of each pair in turn gives a zero equation: every key of
    /// any equation, explicit zeros included. Each weight is classified once and, if general,
    /// transformed once, and each key's general products are summed transformed and reduced
    /// once per [`NADDS`](crate::params::moduli::NADDS) products instead of once each.
    pub(crate) fn combination(
        ring: &Arc<Ring>,
        dim: usize,
        equations: &[Self],
        weights: &[Poly],
    ) -> Result<Self, Error> {
        if equations.len() != weights.len() {
            return Err(Error::Dimension);
        }
        for equation in equations {
            equation.check(ring, dim)?;
        }
        if weights.iter().any(|w| w.ring() != ring) {
            return Err(Error::RingMismatch);
        }
        let weights: Vec<Weight<'_>> = weights.iter().map(Weight::new).collect();
        let r2 = combine_maps(ring, &weights, equations.iter().map(|e| e.r2.slots()))?;
        let r1 = combine_maps(ring, &weights, equations.iter().map(|e| e.r1.slots()))?;
        let constants = equations.iter().enumerate().map(|(i, e)| (i, &e.r0));
        Ok(Self {
            r2: SparsePolyMat::from_sorted(ring.clone(), dim, r2)?,
            r1: SparsePolyVec::from_sorted(ring.clone(), dim, r1)?,
            r0: combine_terms(ring, &weights, constants)?,
        })
    }
    /// `self += a.product_affine(b)?`, in place, with the same keys, values and refusals: every
    /// pair of linear keys and every key of either form is kept, also where the product is
    /// zero, and zero products are not computed.
    pub(crate) fn add_product_affine(&mut self, a: &Self, b: &Self) -> Result<(), Error> {
        let ring = a.r0.ring().clone();
        let dim = a.r2.dimension();
        a.check(&ring, dim)?;
        b.check(&ring, dim)?;
        if a.r2.entries().any(|(_, p)| !p.is_zero()) || b.r2.entries().any(|(_, p)| !p.is_zero()) {
            return Err(Error::Parameter("expression degree exceeds two"));
        }
        let own = self.r0.ring().clone();
        self.check(&own, self.r2.dimension())?;
        a.check(&own, self.r2.dimension())?;
        // Each coefficient is classified once and, if general, transformed at most once.
        let classify = |form: &Self| -> Vec<(u16, Shape)> {
            form.r1.entries().map(|(i, p)| (i, Shape::of(p))).collect()
        };
        let (a_shapes, b_shapes) = (classify(a), classify(b));
        let a_values: Vec<&Poly> = a.r1.entries().map(|(_, p)| p).collect();
        let b_values: Vec<&Poly> = b.r1.entries().map(|(_, p)| p).collect();
        let mut a_ntt = vec![None; a_values.len()];
        let mut b_ntt = vec![None; b_values.len()];
        // Every pair's key, by one merge per row; then the products of the nonzero pairs.
        let keys = |shapes: &[(u16, Shape)]| -> Vec<u16> { shapes.iter().map(|s| s.0).collect() };
        self.r2
            .insert_outer_keys(&keys(&a_shapes), &keys(&b_shapes))?;
        let nonzero = |shapes: &[(u16, Shape)]| -> Vec<usize> {
            (0..shapes.len())
                .filter(|&x| shapes[x].1 != Shape::Zero)
                .collect()
        };
        let (a_nonzero, b_nonzero) = (nonzero(&a_shapes), nonzero(&b_shapes));
        for &x in &a_nonzero {
            let (i, sa) = a_shapes[x];
            for &y in &b_nonzero {
                let (j, sb) = b_shapes[y];
                let (r, c) = (i.min(j), i.max(j));
                match (sa, sb) {
                    (Shape::Zero, _) | (_, Shape::Zero) => {}
                    (Shape::General, Shape::General) => {
                        let pa = a_ntt[x].get_or_insert_with(|| a_values[x].to_ntt());
                        let pb = b_ntt[y].get_or_insert_with(|| b_values[y].to_ntt());
                        let product = pa.product(pb)?.reduce()?;
                        self.r2.entry_mut(r, c)?.add_assign(&product)?;
                    }
                    (Shape::General, cheap) => {
                        cheap.add_product_to(self.r2.entry_mut(r, c)?, a_values[x])?;
                    }
                    (cheap, _) => {
                        cheap.add_product_to(self.r2.entry_mut(r, c)?, b_values[y])?;
                    }
                }
            }
        }
        // The zero quadratic entries of both forms, whose keys `scale` keeps.
        for ((r, c), _) in a.r2.entries().chain(b.r2.entries()) {
            self.r2.insert_key(r, c)?;
        }
        // The linear terms a_i b_0 and b_j a_0, and a_0 b_0.
        for (form, other) in [(a, &b.r0), (b, &a.r0)] {
            let other_zero = other.is_zero();
            for (i, p) in form.r1.entries() {
                if !other_zero && !p.is_zero() {
                    self.r1.entry_mut(i)?.add_assign(&p.mul(other)?)?;
                } else {
                    self.r1.insert_key(i)?;
                }
            }
        }
        if !a.r0.is_zero() && !b.r0.is_zero() {
            self.r0.add_assign(&a.r0.mul(&b.r0)?)?;
        }
        Ok(())
    }
    /// Multiply every coefficient by a canonical scalar, in place: the keys and values of
    /// `scale` by that constant.
    fn scale_scalar_assign(&mut self, w: &U256) {
        for p in self.r2.values_mut().chain(self.r1.values_mut()) {
            if !p.is_zero() {
                p.scale_assign(w);
            }
        }
        self.r0.scale_assign(w);
    }
    /// [`QuadEq::resized`] in place.
    pub(crate) fn resize(&mut self, dimension: usize) -> Result<(), Error> {
        if dimension < self.r2.dimension() {
            return Err(Error::Dimension);
        }
        let ring = self.r0.ring().clone();
        self.check(&ring, self.r2.dimension())?;
        self.r2.set_dimension(dimension)?;
        self.r1.set_dimension(dimension)
    }
    /// Extend the ambient variable space with new coordinates.
    pub fn resized(&self, dimension: usize) -> Result<Self, Error> {
        if dimension < self.r2.dimension() {
            return Err(Error::Dimension);
        }
        let ring = self.r0.ring().clone();
        self.check(&ring, self.r2.dimension())?;
        Ok(Self {
            r2: SparsePolyMat::new(
                ring.clone(),
                dimension,
                self.r2
                    .entries()
                    .map(|((r, c), p)| (r, c, p.clone()))
                    .collect(),
            )?,
            r1: SparsePolyVec::new(
                ring,
                dimension,
                self.r1.entries().map(|(i, p)| (i, p.clone())).collect(),
            )?,
            r0: self.r0.clone(),
        })
    }
    /// Conjugate all coefficients and swap each variable/conjugate pair (Lemma 4.8).
    pub fn conjugate(&self) -> Result<Self, Error> {
        let ring = self.r0.ring().clone();
        let dim = self.r2.dimension();
        self.check(&ring, dim)?;
        if !dim.is_multiple_of(2) {
            return Err(Error::Dimension);
        }
        // Explicit zero entries stay explicit zeros, without a polynomial of their own.
        Ok(Self {
            r2: self.r2.conjugated()?,
            r1: self.r1.conjugated()?,
            r0: self.r0.auto(),
        })
    }
    /// Trace map of a form, valid on interleaved witnesses.
    pub fn trace(&self) -> Result<Self, Error> {
        self.clone().into_trace()
    }
    /// [`QuadEq::trace`] of an owned form: `(self + conjugate) (q+1)/2`, the sum and the
    /// scaling in place.
    pub(crate) fn into_trace(mut self) -> Result<Self, Error> {
        let ring = self.r0.ring().clone();
        if !bool::from(ring.modulus().is_odd()) {
            return Err(Error::Parameter("trace requires invertible two"));
        }
        // (q+1)/2 without overflow for odd q.
        let half = ring.half.wrapping_add(&U256::ONE);
        let conjugate = self.conjugate()?;
        self.add_assign(&conjugate)?;
        self.scale_scalar_assign(&ring.reduce(&half));
        Ok(self)
    }
    /// Check all component rings and dimensions against the caller's variable space.
    pub fn check(&self, ring: &Arc<Ring>, dimension: usize) -> Result<(), Error> {
        if self.r2.dimension() != dimension || self.r1.dimension() != dimension {
            return Err(Error::Dimension);
        }
        if self.r2.ring() != ring || self.r1.ring() != ring || self.r0.ring() != ring {
            return Err(Error::RingMismatch);
        }
        Ok(())
    }
    /// Evaluate the full ring equation on an interleaved witness.
    pub fn evaluate(&self, witness: &PolyVec) -> Result<Poly, Error> {
        self.check(witness.ring(), witness.len())?;
        self.r2
            .bilinear(witness, witness)?
            .add(&self.r1.dot(witness)?)?
            .add(&self.r0)
    }
    /// [`QuadEq::evaluate_constant`] with the witness products of `pairs`: each quadratic
    /// entry contributes $`\mathrm{ct}(R_{rc}\cdot s_rs_c)`$, a dot product, and no transform.
    pub(crate) fn evaluate_constant_with(
        &self,
        witness: &PolyVec,
        pairs: &PairProducts,
    ) -> Result<U256, Error> {
        self.check(witness.ring(), witness.len())?;
        let mut sum = ConstantSum::new(self.r0.ring());
        for ((r, c), a) in self.r2.entries() {
            if a.is_zero() {
                continue;
            }
            match pairs.products.get(&(r, c)) {
                Some(product) => sum.add_product(a, product),
                None => {
                    let w = witness.entries();
                    let product = w[usize::from(r)].mul(&w[usize::from(c)])?;
                    sum.add_product(a, &product);
                }
            }
        }
        self.r1.dot_constant(witness, &mut sum)?;
        sum.add(&self.r0.coefficient(0));
        Ok(sum.finish())
    }
    /// The constant coefficient of [`QuadEq::evaluate`], exactly, with the same refusals:
    /// each product contributes only its constant coefficient, a dot product of length
    /// $`d`$, so no product by a witness polynomial is computed in full (see
    /// `SparsePolyMat::bilinear_constant`).
    pub(crate) fn evaluate_constant(&self, witness: &PolyVec) -> Result<U256, Error> {
        self.check(witness.ring(), witness.len())?;
        let mut sum = ConstantSum::new(self.r0.ring());
        self.r2.bilinear_constant(witness, &mut sum)?;
        self.r1.dot_constant(witness, &mut sum)?;
        sum.add(&self.r0.coefficient(0));
        Ok(sum.finish())
    }
    /// Canonical, dimension-bound encoding for statement hashing.
    ///
    /// The transcripts absorb these bytes, prefixed with their length, as part of an instance,
    /// without building them in one buffer: they hash them in pieces from chunks of at most
    /// 32 KiB, which hold long runs of zero bytes, such as the codes of zero polynomials, as
    /// counts. The transcript states are those of absorbing the bytes returned here.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut w = BitWriter::new();
        self.write_codes(&mut w)?;
        Ok(w.finish())
    }
    /// The bytes of [`QuadEq::to_bytes`] as [`Segments`], for
    /// `Transcript::absorb_varlen_segments`: no buffer of them exceeds 32 KiB, apart from their list
    /// of chunks, and a run of at least 64 whole zero bytes takes only a record header. A zero
    /// polynomial whose codes take 576 bits or more (degree 64 and $`q>2^8`$, for example)
    /// always leaves such a run.
    pub(crate) fn to_segments(&self) -> Result<Segments, Error> {
        let mut w = SegmentWriter::new();
        self.write_codes(&mut w)?;
        Ok(w.finish())
    }
    /// The codes of [`QuadEq::to_bytes`] before its end bit: the dimension, the number of
    /// matrix entries, each entry's row, column and coefficients, the number of vector entries,
    /// each one's index and coefficients, and the constant's coefficients.
    fn write_codes(&self, w: &mut impl CodeWriter) -> Result<(), Error> {
        self.check(self.r0.ring(), self.r2.dimension())?;
        w.unsigned(self.r2.dimension() as u128, 32)?;
        w.unsigned(self.r2.key_count() as u128, 32)?;
        for ((r, c), p) in self.r2.entries() {
            w.unsigned(r.into(), 16)?;
            w.unsigned(c.into(), 16)?;
            encode_poly(w, p)?;
        }
        w.unsigned(self.r1.key_count() as u128, 32)?;
        for (i, p) in self.r1.entries() {
            w.unsigned(i.into(), 16)?;
            encode_poly(w, p)?;
        }
        encode_poly(w, &self.r0)
    }
}
/// The products $`s_rs_c`$ of witness polynomials for the pairs that the quadratic parts of a
/// batch of equations use, each computed once. A quadratic entry then contributes
/// $`\mathrm{ct}(R_{rc}\cdot s_rs_c)`$ to an equation's constant coefficient, a dot product
/// of length $`d`$, instead of transformed products and a reduced row sum: the self-check of a
/// Kyber1024 statement has 33,313 quadratic entries on 99 distinct pairs (counted 2026-11-16).
/// The products are canonical, so every constant coefficient is the one of
/// [`QuadEq::evaluate_constant`]. They are secret, and `Poly` wipes them on drop.
pub(crate) struct PairProducts {
    products: BTreeMap<(u16, u16), Poly>,
}
impl PairProducts {
    /// The products for `eqs`, or `None` when an equation does not fit the witness (the
    /// per-equation evaluation then reports it, in order) or when the entries do not outnumber
    /// the distinct pairs twice, so that the products hold at most half the memory of the
    /// equations' own entries.
    pub(crate) fn for_equations(eqs: &[QuadEq], witness: &PolyVec) -> Result<Option<Self>, Error> {
        let mut products = Self::empty();
        Ok(products.add_for(eqs, witness)?.then_some(products))
    }
    /// No products, for [`PairProducts::add_for`].
    pub(crate) fn empty() -> Self {
        Self {
            products: BTreeMap::new(),
        }
    }
    /// Add the products of the pairs of `eqs` that are not held yet, when
    /// [`PairProducts::for_equations`] would return products for `eqs` on their own: `true`
    /// when every pair of `eqs` is then held (and [`QuadEq::evaluate_constant_with`] computes
    /// no product for them), `false` when nothing was added. Equations that arrive in chunks
    /// share the products of their common pairs, each computed once.
    pub(crate) fn add_for(&mut self, eqs: &[QuadEq], witness: &PolyVec) -> Result<bool, Error> {
        let ring = witness.ring();
        let mut pairs = BTreeMap::new();
        let mut entries = 0usize;
        for eq in eqs {
            if eq.check(ring, witness.len()).is_err() {
                return Ok(false);
            }
            for ((r, c), a) in eq.r2.entries() {
                if !a.is_zero() {
                    entries += 1;
                    pairs.insert((r, c), ());
                }
            }
        }
        if pairs.is_empty() || entries < 2 * pairs.len() {
            return Ok(false);
        }
        let keys: Vec<(u16, u16)> = pairs
            .into_keys()
            .filter(|key| !self.products.contains_key(key))
            .collect();
        let w = witness.entries();
        let values = crate::par::try_map(keys.len(), |k| {
            let (r, c) = keys[k];
            w[usize::from(r)].mul(&w[usize::from(c)])
        })?;
        self.products.extend(keys.into_iter().zip(values));
        Ok(true)
    }
}
fn encode_poly(w: &mut impl CodeWriter, p: &Poly) -> Result<(), Error> {
    // Each coefficient is a uniform code of bits(q - 1) bits; those of a zero polynomial are
    // all zero, written as one run.
    if p.is_zero() {
        w.zeros(p.ring().degree() as u64 * u64::from(p.ring().coefficient_bits()));
        return Ok(());
    }
    let q = p.ring().modulus();
    match int::to_u128(&q) {
        Some(m) => w.uniform_each(p.values_u128(), m)?,
        None => {
            for x in p.values() {
                w.uniform_u256(&x, &q)?;
            }
        }
    }
    Ok(())
}
/// Interleave each variable with its automorphism image.
pub fn interleave(s1: &PolyVec, m: &PolyVec) -> Result<PolyVec, Error> {
    PolyVec::new(
        s1.ring().clone(),
        s1.entries()
            .iter()
            .chain(m.entries())
            .flat_map(|p| [p.clone(), p.auto()])
            .collect(),
    )
}

/// A compressed quadratic proof with its additional BDLOP commitment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuadProof {
    /// Commitment to the linear cross-term $`g_1`$.
    pub t: Poly,
    /// Shared opening responses and compression hints.
    pub opening: OpeningProof,
}

/// The variable space of the equations of `scheme`: $`2(m_1'+\ell)`$ coordinates, each
/// variable and its conjugate.
pub(crate) fn dimension(scheme: &Abdlop) -> usize {
    2 * (scheme.bounded_len() + scheme.message_len())
}
/// The transcript of the quadratic proof on its own, after its instance: the scheme, the
/// commitment's serialization, and the equation's encoding ([`QuadEq::to_bytes`]) as
/// `SerializeVarLenString`. Checks the equation's shape.
fn transcript(
    scheme: &Abdlop,
    commitment: &Commitment,
    equation: &QuadEq,
    context: &[u8],
) -> Result<Transcript, Error> {
    equation.check(scheme.ring(), dimension(scheme))?;
    let mut transcript = Transcript::new(Protocol::Quadratic, context)?;
    scheme.absorb_scheme(&mut transcript)?;
    transcript.absorb(&scheme.commitment_bytes(commitment)?);
    transcript.absorb_varlen_segments(&equation.to_segments()?)?;
    Ok(transcript)
}
/// The challenge after the prover message $`(t,v,w_1)`$: on a copy of `transcript`, absorb the
/// message's serialization, $`t`$ and $`v`$ modulo $`q`$, the LinHC mask commitment of the
/// extractable mode when given (modulo $`q'`$), then $`w_1`$ modulo $`m`$, and squeeze the
/// challenge.
fn challenge_ext(
    scheme: &Abdlop,
    transcript: &Transcript,
    t: &Poly,
    v: &Poly,
    w1: &PolyVec,
    mask: Option<linhc::MaskCommitment<'_>>,
) -> Result<Poly, Error> {
    let ring = scheme.ring();
    let mut w = BitWriter::new();
    fields::write_below(
        &mut w,
        &PolyVec::new(ring.clone(), vec![t.clone(), v.clone()])?,
        &ring.q,
    )?;
    if let Some((w1_mask, w2_mask)) = mask {
        linhc::write_pair(&mut w, w1_mask, w2_mask)?;
    }
    fields::write_below(&mut w, w1, &scheme.compression.hint_modulus())?;
    let mut transcript = transcript.clone();
    transcript.absorb(&w.into_bytes()?);
    scheme.squeeze_challenge(&mut transcript)
}
/// Row $`\ell`$ of the matrix $`B`$ with one more row, the row of the commitment to $`t`$.
/// Entries of the public matrices are independent streams, so the row is expanded alone.
fn extra_row(scheme: &Abdlop) -> Result<PolyVec, Error> {
    let cols = scheme.a2.cols();
    let end = abdlop::matrix_size(scheme.message_len() + 1, cols)?;
    PolyVec::new(
        scheme.ring.clone(),
        abdlop::matrix_entries(&scheme.ring, end - cols..end, &scheme.seed, 2)?,
    )
}

/// Prove one quadratic equation using paper Rej_2 for the randomness response.
///
/// The seed must be secret and uniformly random, and it may be reused, also for the commitment
/// being proven. The masks are read under a key derived from the seed, the transcript before
/// the first message (parameters, public seed, context, commitment and equation) and the
/// opening: identical calls return identical proofs, and calls that differ in any input read
/// independent masks (assuming SHAKE128 and AES-256 behave as pseudorandom functions).
///
/// With a reused seed, zero-knowledge holds only relative to the equality pattern of the
/// inputs: identical inputs give identical proofs. Uses that need multi-theorem
/// zero-knowledge, and settings where faults can be injected (a fault that changes a challenge
/// but not the key can reveal the witness), need a fresh seed per call, drawn from a
/// `rand_core::CryptoRng`.
pub fn prove_with_seed(
    scheme: &Abdlop,
    commitment: &Commitment,
    opening: &Opening,
    equation: &QuadEq,
    context: &[u8],
    mut seed: [u8; 32],
) -> Result<QuadProof, Error> {
    prove_seeded(
        scheme,
        commitment,
        opening,
        equation,
        context,
        &take_seed(&mut seed),
    )
}
pub(crate) fn prove_seeded(
    scheme: &Abdlop,
    commitment: &Commitment,
    opening: &Opening,
    equation: &QuadEq,
    context: &[u8],
    seed: &Secret<[u8; 32]>,
) -> Result<QuadProof, Error> {
    let transcript = transcript(scheme, commitment, equation, context)?;
    Ok(prove_in(
        scheme,
        commitment,
        opening,
        equation,
        &transcript,
        seed,
        None,
    )?
    .0)
}
/// The quadratic proof in `transcript`, which binds everything its challenge depends on
/// besides the prover's message, with the extractable commitment beside the opening when `ext`
/// is given, returning the accepted attempt's LinHC opening too. The masks' key binds the
/// transcript through `Transcript::binding`.
pub(crate) fn prove_in(
    scheme: &Abdlop,
    commitment: &Commitment,
    opening: &Opening,
    equation: &QuadEq,
    transcript: &Transcript,
    seed: &Secret<[u8; 32]>,
    ext: Option<&linhc::ProverExt<'_>>,
) -> Result<(QuadProof, Option<linhc::LinhcOpening>), Error> {
    equation.check(scheme.ring(), dimension(scheme))?;
    let s = interleave(&opening.s1, &opening.m)?;
    if !equation.evaluate(&s)?.is_zero() {
        return Err(Error::Witness);
    }
    let (proof, linhc_opening, ()) = prove_core_ext(
        scheme,
        commitment,
        opening,
        equation,
        &transcript.binding(),
        seed,
        ext,
        |t, v, w1, mask| Ok((challenge_ext(scheme, transcript, t, v, w1, mask)?, ())),
    )?;
    Ok((proof, linhc_opening))
}
/// The quadratic proof with a caller-defined challenge of the first message `(t, v, w1)`. It
/// does not check the witness against the equation. `binding` is as in `Abdlop::prove_core`.
/// The challenge function returns side data with the challenge, and the proof is returned
/// with that of its accepted attempt; the function may be called for further attempts, in no
/// particular order (`Abdlop::prove_core`).
#[cfg(test)]
pub(crate) fn prove_core<U: Send>(
    scheme: &Abdlop,
    commitment: &Commitment,
    opening: &Opening,
    equation: &QuadEq,
    binding: &[u8; 32],
    seed: &Secret<[u8; 32]>,
    challenge: impl Fn(&Poly, &Poly, &PolyVec) -> Result<(Poly, U), Error> + Sync,
) -> Result<(QuadProof, U), Error> {
    let (proof, _, side) = prove_core_ext(
        scheme,
        commitment,
        opening,
        equation,
        binding,
        seed,
        None,
        |t, v, w1, _| challenge(t, v, w1),
    )?;
    Ok((proof, side))
}
/// The part of $`g_1`$ linear in an attempt's mask $`y`$, with coefficients that depend on
/// the witness alone, computed once per proof instead of once per attempt:
/// $`\mathrm{bilinear}(s,y)+\mathrm{bilinear}(y,s)+r_1\cdot y=\sum_ky_kw_k`$ with
/// $`w_k=\sum_{r:(r,k)}s_rR_{rk}+\sum_{c:(k,c)}R_{kc}s_c+r_{1,k}`$, collecting each
/// $`y_k`$'s terms (products commute). Each $`w_k`$ is exact, so $`g_1`$ is the canonical value
/// of the three-term sum. The coefficients come from the witness and are wiped as `Poly`s are.
struct MaskLinear {
    ring: Arc<Ring>,
    dim: usize,
    /// The nonzero $`w_k`$, each classified and, if general, transformed once.
    terms: Vec<(usize, Poly, Term)>,
}
impl MaskLinear {
    fn new(equation: &QuadEq, quadratic: &Quadratic<'_>, s: &PolyVec) -> Result<Self, Error> {
        let ring = s.ring().clone();
        let dim = s.len();
        equation.check(&ring, dim)?;
        let mut w: Vec<Poly> = (0..dim).map(|_| Poly::zero(ring.clone())).collect();
        for (i, p) in equation.r1.entries() {
            w[usize::from(i)].add_assign(p)?;
        }
        let mut sums: Vec<Accumulator> = (0..dim).map(|_| Accumulator::new(&ring)).collect();
        let s_terms: Vec<Term> = s.entries().iter().map(Term::new).collect();
        // out += a b, for a and b with their terms.
        let add_product = |out: &mut Poly,
                           sum: &mut Accumulator,
                           (a, at): (&Poly, &Term),
                           (b, bt): (&Poly, &Term)|
         -> Result<(), Error> {
            match (at, bt) {
                (Term::General(x), Term::General(y)) => sum.mac(x, y, out),
                (Term::Cheap(shape), _) => shape.add_product_to(out, b).map(|_| ()),
                (_, Term::Cheap(shape)) => shape.add_product_to(out, a).map(|_| ()),
            }
        };
        for &(r, c, p, ref pt) in &quadratic.entries {
            let (sr, sc) = (&s.entries()[r], &s.entries()[c]);
            // w_c += s_r R_rc and w_r += R_rc s_c.
            add_product(&mut w[c], &mut sums[c], (sr, &s_terms[r]), (p, pt))?;
            add_product(&mut w[r], &mut sums[r], (p, pt), (sc, &s_terms[c]))?;
        }
        let mut terms = Vec::new();
        for (k, (mut wk, mut sum)) in w.into_iter().zip(sums).enumerate() {
            sum.finish_into(&mut wk)?;
            if !wk.is_zero() {
                let term = Term::new(&wk);
                terms.push((k, wk, term));
            }
        }
        Ok(Self { ring, dim, terms })
    }
    /// $`\sum_ky_kw_k`$, with the refusals of the products it replaces.
    fn apply(&self, y: &PolyVec) -> Result<Poly, Error> {
        if y.len() != self.dim {
            return Err(Error::Dimension);
        }
        if y.ring() != &self.ring {
            return Err(Error::RingMismatch);
        }
        let mut out = Poly::zero(self.ring.clone());
        let mut sum = Accumulator::new(&self.ring);
        for (k, wk, term) in &self.terms {
            let yk = &y.entries()[*k];
            match term {
                Term::Cheap(shape) => {
                    shape.add_product_to(&mut out, yk)?;
                }
                Term::General(x) => match Shape::of(yk) {
                    Shape::General => sum.mac(x, &yk.to_ntt(), &mut out)?,
                    cheap => {
                        cheap.add_product_to(&mut out, wk)?;
                    }
                },
            }
        }
        sum.finish_into(&mut out)?;
        Ok(out)
    }
}

/// The nonzero quadratic coefficients of an equation, each classified and, if general,
/// transformed once per proof, for [`MaskLinear::new`] and every attempt's
/// [`Quadratic::evaluate`] instead of once per use.
struct Quadratic<'a> {
    ring: Arc<Ring>,
    dim: usize,
    /// $`(r,c,R_{rc},t_{rc})`$ for the nonzero entries, by row then column.
    entries: Vec<(usize, usize, &'a Poly, Term)>,
}
impl<'a> Quadratic<'a> {
    fn new(equation: &'a QuadEq) -> Self {
        let entries = equation
            .r2
            .entries()
            .filter(|(_, p)| !p.is_zero())
            .map(|((r, c), p)| (usize::from(r), usize::from(c), p, Term::new(p)))
            .collect();
        Self {
            ring: equation.r0.ring().clone(),
            dim: equation.r2.dimension(),
            entries,
        }
    }
    /// `bilinear(y, y)` of the quadratic coefficients, computed as it computes it: each row's
    /// sum $`\sum_cR_{rc}y_c`$, then its product by $`y_r`$, every step exact.
    fn evaluate(&self, y: &PolyVec) -> Result<Poly, Error> {
        if y.len() != self.dim {
            return Err(Error::Dimension);
        }
        if y.ring() != &self.ring {
            return Err(Error::RingMismatch);
        }
        let ring = &self.ring;
        let mut shapes: Secret<Vec<Option<Shape>>> = Secret::new(vec![None; y.len()]);
        let mut transforms: Vec<Option<PolyNtt>> = vec![None; y.len()];
        let mut out = Poly::zero(ring.clone());
        let mut outer = Accumulator::new(ring);
        for row in self.entries.chunk_by(|x, z| x.0 == z.0) {
            let r = row[0].0;
            let mut inner = Poly::zero(ring.clone());
            let mut inner_sum = Accumulator::new(ring);
            for &(_, c, p, ref term) in row {
                let yc = &y.entries()[c];
                let shape = *shapes[c].get_or_insert_with(|| Shape::of(yc));
                match (term, shape) {
                    (Term::General(a), Shape::General) => {
                        let b = transforms[c].get_or_insert_with(|| yc.to_ntt());
                        inner_sum.mac(a, b, &mut inner)?;
                    }
                    (Term::General(_), cheap) => {
                        cheap.add_product_to(&mut inner, p)?;
                    }
                    (Term::Cheap(cheap), _) => {
                        cheap.add_product_to(&mut inner, yc)?;
                    }
                }
            }
            inner_sum.finish_into(&mut inner)?;
            let yr = &y.entries()[r];
            let shape = *shapes[r].get_or_insert_with(|| Shape::of(yr));
            match (shape, Shape::of(&inner)) {
                (Shape::Zero, _) | (_, Shape::Zero) => {}
                (Shape::General, Shape::General) => {
                    let a = transforms[r].get_or_insert_with(|| yr.to_ntt());
                    outer.mac(a, &inner.to_ntt(), &mut out)?;
                }
                (Shape::General, cheap) => {
                    cheap.add_product_to(&mut out, yr)?;
                }
                (cheap, _) => {
                    cheap.add_product_to(&mut out, &inner)?;
                }
            }
        }
        outer.finish_into(&mut out)?;
        Ok(out)
    }
}

/// [`prove_core`] with the extractable commitment when `ext` is given: the challenge function
/// also receives each attempt's LinHC mask commitment, and must absorb it.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn prove_core_ext<U: Send>(
    scheme: &Abdlop,
    commitment: &Commitment,
    opening: &Opening,
    equation: &QuadEq,
    binding: &[u8; 32],
    seed: &Secret<[u8; 32]>,
    ext: Option<&linhc::ProverExt<'_>>,
    challenge: impl Fn(
        &Poly,
        &Poly,
        &PolyVec,
        Option<linhc::MaskCommitment<'_>>,
    ) -> Result<(Poly, U), Error>
    + Sync,
) -> Result<(QuadProof, Option<linhc::LinhcOpening>, U), Error> {
    let s = interleave(&opening.s1, &opening.m)?;
    let quadratic = Quadratic::new(equation);
    let linear = MaskLinear::new(equation, &quadratic, &s)?;
    let b = extra_row(scheme)?;
    let s21 = abdlop::part(&opening.s2, 0, scheme.a2.cols())?;
    let bs = b.dot(&s21)?;
    // Each attempt's t is its side data, so that the proof takes the accepted attempt's.
    let (_, proof, linhc_opening, (t, side)) = scheme.prove_core_ext(
        commitment,
        opening,
        Caller::Quadratic,
        binding,
        seed,
        crate::rand::secret::MAX_ATTEMPTS,
        crate::par::speculation_width(),
        ext,
        |y1, y21, w1, mask| {
            let by = scheme.b.mul(y21)?;
            let minus_by = PolyVec::new(
                by.ring().clone(),
                by.entries().iter().map(Poly::neg).collect(),
            )?;
            let y = interleave(y1, &minus_by)?;
            let g1 = linear.apply(&y)?;
            let t = bs.add(&g1)?;
            let v = quadratic.evaluate(&y)?.add(&b.dot(y21)?)?;
            let (c, side) = challenge(&t, &v, w1, mask)?;
            Ok((c, (t, side)))
        },
    )?;
    Ok((QuadProof { t, opening: proof }, linhc_opening, side))
}

/// Verify the check of Fig. 6 (Eq. (31), rewritten) and both response norms, rebuilding the
/// statement transcript.
pub fn verify(
    scheme: &Abdlop,
    commitment: &Commitment,
    equation: &QuadEq,
    proof: &QuadProof,
    context: &[u8],
) -> Result<(), Error> {
    let transcript = transcript(scheme, commitment, equation, context)?;
    verify_in(scheme, commitment, equation, proof, &transcript, None)
}
/// [`verify`] in `transcript`, as [`prove_in`] proves, with the extractable commitment's checks
/// when `ext` is given.
pub(crate) fn verify_in(
    scheme: &Abdlop,
    commitment: &Commitment,
    equation: &QuadEq,
    proof: &QuadProof,
    transcript: &Transcript,
    ext: Option<&linhc::VerifierExt<'_>>,
) -> Result<(), Error> {
    equation.check(scheme.ring(), dimension(scheme))?;
    verify_core_ext(scheme, commitment, equation, proof, ext, |v, w1, mask| {
        challenge_ext(scheme, transcript, &proof.t, v, w1, mask)
    })
}
/// The check of Fig. 6 (Eq. (31), rewritten) with a caller-defined challenge of the
/// reconstructed `(v, w1)`. The caller must have checked the equation's shape, as `verify_in`
/// does.
#[cfg(test)]
pub(crate) fn verify_core(
    scheme: &Abdlop,
    commitment: &Commitment,
    equation: &QuadEq,
    proof: &QuadProof,
    challenge: impl FnOnce(&Poly, &PolyVec) -> Result<Poly, Error>,
) -> Result<(), Error> {
    verify_core_ext(scheme, commitment, equation, proof, None, |v, w1, _| {
        challenge(v, w1)
    })
}
/// [`verify_core`] with the extractable commitment's checks when `ext` is given: the challenge
/// function also receives the recomputed LinHC mask commitment.
pub(crate) fn verify_core_ext(
    scheme: &Abdlop,
    commitment: &Commitment,
    equation: &QuadEq,
    proof: &QuadProof,
    ext: Option<&linhc::VerifierExt<'_>>,
    challenge: impl FnOnce(&Poly, &PolyVec, Option<linhc::MaskCommitment<'_>>) -> Result<Poly, Error>,
) -> Result<(), Error> {
    if proof.t.ring() != scheme.ring() {
        return Err(Error::RingMismatch);
    }
    let b = extra_row(scheme)?;
    let c = &proof.opening.challenge;
    let masked_m = abdlop::sub(
        &abdlop::scale(&commitment.t_b, c)?,
        &scheme.b.mul(&proof.opening.z21)?,
    )?;
    let z = interleave(&proof.opening.z1, &masked_m)?;
    let f = c.mul(&proof.t)?.sub(&b.dot(&proof.opening.z21)?)?;
    let v = equation
        .r2
        .bilinear(&z, &z)?
        .add(&c.mul(&equation.r1.dot(&z)?)?)?
        .add(&c.mul(c)?.mul(&equation.r0)?)?
        .sub(&f)?;
    scheme.verify_core_ext(commitment, &proof.opening, ext, |w1, mask| {
        challenge(&v, w1, mask)
    })
}
