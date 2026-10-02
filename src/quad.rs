//! A quadratic relation with automorphisms: LNP22 Fig. 6, Eqs. 29–31.
use crate::{
    Error,
    abdlop::{self, Abdlop, Caller, Commitment, Opening, OpeningProof},
    codec::BitWriter,
    math::{Poly, PolyVec, Ring, SparsePolyMat, SparsePolyVec, U256, int, terms::Shape},
    rand::take_seed,
    transcript::Transcript,
};
use std::{collections::BTreeMap, sync::Arc};
use zeroize::Zeroizing;

#[cfg(test)]
mod encoding_tests;
#[cfg(test)]
mod expansion_tests;
#[cfg(test)]
mod extraction_tests;
#[cfg(test)]
mod in_place_tests;
#[cfg(test)]
mod substitute_tests;

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
        let add = |entry: &mut Poly, value: &Poly| -> Result<(), Error> {
            *entry = entry.add(value)?;
            Ok(())
        };
        let zero = || Poly::zero(ring.clone());
        for ((i, j), a) in self.r2.entries() {
            let (x, y) = (&map[usize::from(i)], &map[usize::from(j)]);
            for (u, cu) in &x.terms {
                let acu = a.mul(cu)?;
                for (v, cv) in &y.terms {
                    add(
                        matrix
                            .entry(((*u).min(*v), (*u).max(*v)))
                            .or_insert_with(zero),
                        &acu.mul(cv)?,
                    )?;
                }
            }
            if !y.constant.is_zero() {
                let ay = a.mul(&y.constant)?;
                for (u, cu) in &x.terms {
                    add(vector.entry(*u).or_insert_with(zero), &ay.mul(cu)?)?;
                }
                if !x.constant.is_zero() {
                    add(&mut constant, &ay.mul(&x.constant)?)?;
                }
            }
            if !x.constant.is_zero() {
                let ax = a.mul(&x.constant)?;
                for (v, cv) in &y.terms {
                    add(vector.entry(*v).or_insert_with(zero), &ax.mul(cv)?)?;
                }
            }
        }
        for (i, a) in self.r1.entries() {
            let x = &map[usize::from(i)];
            for (u, cu) in &x.terms {
                add(vector.entry(*u).or_insert_with(zero), &a.mul(cu)?)?;
            }
            if !x.constant.is_zero() {
                add(&mut constant, &a.mul(&x.constant)?)?;
            }
        }
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
        for (x, &(i, sa)) in a_shapes.iter().enumerate() {
            for (y, &(j, sb)) in b_shapes.iter().enumerate() {
                let (r, c) = (i.min(j), i.max(j));
                match (sa, sb) {
                    (Shape::Zero, _) | (_, Shape::Zero) => self.r2.insert_key(r, c)?,
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
    /// Canonical, dimension-bound encoding for statement hashing.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        self.check(self.r0.ring(), self.r2.dimension())?;
        let mut w = BitWriter::new();
        w.unsigned(self.r2.dimension() as u128, 32)?;
        let matrix: Vec<_> = self.r2.entries().collect();
        let vector: Vec<_> = self.r1.entries().collect();
        w.unsigned(matrix.len() as u128, 32)?;
        for ((r, c), p) in matrix {
            w.unsigned(r.into(), 16)?;
            w.unsigned(c.into(), 16)?;
            encode_poly(&mut w, p)?;
        }
        w.unsigned(vector.len() as u128, 32)?;
        for (i, p) in vector {
            w.unsigned(i.into(), 16)?;
            encode_poly(&mut w, p)?;
        }
        encode_poly(&mut w, &self.r0)?;
        Ok(w.finish())
    }
}
fn encode_poly(w: &mut BitWriter, p: &Poly) -> Result<(), Error> {
    // Each coefficient is a uniform code of bits(q - 1) bits; those of a zero polynomial are
    // all zero, written as one run.
    if p.is_zero() {
        w.zeros(p.coefficients().len() as u64 * u64::from(p.ring().coefficient_bits()));
        return Ok(());
    }
    let q = p.ring().modulus();
    match int::to_u128(&q) {
        Some(m) => {
            for x in p.coefficients() {
                w.uniform(int::low_u128(x), m)?;
            }
        }
        None => {
            for x in p.coefficients() {
                w.uniform_u256(x, &q)?;
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

fn prefix(
    scheme: &Abdlop,
    commitment: &Commitment,
    equation: &QuadEq,
    context: &[u8],
) -> Result<Transcript, Error> {
    equation.check(
        scheme.ring(),
        2 * (scheme.bounded_len() + scheme.message_len()),
    )?;
    let mut prefix = scheme.prefix(commitment, context)?;
    prefix.absorb(b"quadratic-equation", &equation.to_bytes()?);
    Ok(prefix)
}
fn challenge(
    scheme: &Abdlop,
    prefix: &Transcript,
    t: &Poly,
    v: &Poly,
    w1: &PolyVec,
) -> Result<Poly, Error> {
    let mut writer = BitWriter::new();
    encode_poly(&mut writer, t)?;
    encode_poly(&mut writer, v)?;
    let mut prefix = prefix.clone();
    prefix.absorb(b"quadratic-first-message", &writer.finish());
    scheme.challenge(&prefix, w1)
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
    seed: &Zeroizing<[u8; 32]>,
) -> Result<QuadProof, Error> {
    let prefix = prefix(scheme, commitment, equation, context)?;
    let s = interleave(&opening.s1, &opening.m)?;
    if !equation.evaluate(&s)?.is_zero() {
        return Err(Error::Witness);
    }
    let (proof, ()) = prove_core(
        scheme,
        commitment,
        opening,
        equation,
        &prefix.digest(),
        seed,
        |t, v, w1| Ok((challenge(scheme, &prefix, t, v, w1)?, ())),
    )?;
    Ok(proof)
}
/// The quadratic proof with a caller-defined challenge of the first message `(t, v, w1)`. It
/// does not check the witness against the equation; `binding` is as in `Abdlop::prove_core`.
/// The challenge function returns side data with the challenge, and the proof is returned
/// with that of its accepted attempt; the function may be called for further attempts, in no
/// particular order (`Abdlop::prove_core`).
pub(crate) fn prove_core<U: Send>(
    scheme: &Abdlop,
    commitment: &Commitment,
    opening: &Opening,
    equation: &QuadEq,
    binding: &[u8; 32],
    seed: &Zeroizing<[u8; 32]>,
    challenge: impl Fn(&Poly, &Poly, &PolyVec) -> Result<(Poly, U), Error> + Sync,
) -> Result<(QuadProof, U), Error> {
    let s = interleave(&opening.s1, &opening.m)?;
    let b = extra_row(scheme)?;
    let s21 = abdlop::part(&opening.s2, 0, scheme.a2.cols())?;
    let bs = b.dot(&s21)?;
    // Each attempt's t is its side data, so that the proof takes the accepted attempt's.
    let (proof, (t, side)) = scheme.prove_core(
        commitment,
        opening,
        Caller::Quadratic,
        binding,
        seed,
        |y1, y21, w1| {
            let by = scheme.b.mul(y21)?;
            let minus_by = PolyVec::new(
                by.ring().clone(),
                by.entries().iter().map(Poly::neg).collect(),
            )?;
            let y = interleave(y1, &minus_by)?;
            let g1 = equation
                .r2
                .bilinear(&s, &y)?
                .add(&equation.r2.bilinear(&y, &s)?)?
                .add(&equation.r1.dot(&y)?)?;
            let t = bs.add(&g1)?;
            let v = equation.r2.bilinear(&y, &y)?.add(&b.dot(y21)?)?;
            let (c, side) = challenge(&t, &v, w1)?;
            Ok((c, (t, side)))
        },
    )?;
    Ok((QuadProof { t, opening: proof }, side))
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
    let prefix = prefix(scheme, commitment, equation, context)?;
    verify_core(scheme, commitment, equation, proof, |v, w1| {
        challenge(scheme, &prefix, &proof.t, v, w1)
    })
}
/// The check of Fig. 6 (Eq. (31), rewritten) with a caller-defined challenge of the
/// reconstructed `(v, w1)`; the caller must have checked the equation's shape, as `prefix`
/// does.
pub(crate) fn verify_core(
    scheme: &Abdlop,
    commitment: &Commitment,
    equation: &QuadEq,
    proof: &QuadProof,
    challenge: impl FnOnce(&Poly, &PolyVec) -> Result<Poly, Error>,
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
    scheme.verify_core(commitment, &proof.opening, |w1| challenge(&v, w1))
}
