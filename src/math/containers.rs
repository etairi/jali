use super::{
    Poly, PolyNtt, Ring,
    constant::ConstantSum,
    terms::{Accumulator, Shape, Term},
};
use crate::wipe::Secret;
use crate::{Error, params::moduli::NADDS};
use crypto_bigint::U256;
use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock},
};

/// A ring-checked vector of polynomials, including empty vectors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolyVec {
    ring: Arc<Ring>,
    entries: Vec<Poly>,
}
impl zeroize::Zeroize for PolyVec {
    /// Overwrites every entry with the zero polynomial ([`Poly`]'s `Zeroize`), keeping the
    /// length.
    fn zeroize(&mut self) {
        for p in self.entries.iter_mut() {
            p.zeroize();
        }
    }
}
impl PolyVec {
    /// Validate the ring of every entry.
    pub fn new(ring: Arc<Ring>, entries: Vec<Poly>) -> Result<Self, Error> {
        if entries.iter().any(|p| p.ring() != &ring) {
            return Err(Error::RingMismatch);
        }
        Ok(Self { ring, entries })
    }
    /// Construct a zero vector.
    pub fn zero(ring: Arc<Ring>, len: usize) -> Self {
        Self {
            entries: vec![Poly::zero(ring.clone()); len],
            ring,
        }
    }
    /// Vector entries.
    pub fn entries(&self) -> &[Poly] {
        &self.entries
    }
    /// Shared ring.
    pub fn ring(&self) -> &Arc<Ring> {
        &self.ring
    }
    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    /// Whether the vector has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Exact dot product, reducing every 128 products.
    pub fn dot(&self, rhs: &Self) -> Result<Poly, Error> {
        if self.ring != rhs.ring {
            return Err(Error::RingMismatch);
        }
        if self.len() != rhs.len() {
            return Err(Error::Dimension);
        }
        let mut result = Poly::zero(self.ring.clone());
        for (a, b) in self.entries.chunks(NADDS).zip(rhs.entries.chunks(NADDS)) {
            let mut acc = a[0].to_ntt().product(&b[0].to_ntt())?;
            for (a, b) in a[1..].iter().zip(&b[1..]) {
                acc.add_assign(&a.to_ntt().product(&b.to_ntt())?)?;
            }
            result = result.add(&acc.reduce()?)?;
        }
        Ok(result)
    }
    /// $`T(a,b)=\sum_i\sigma(a_i)b_i`$; its constant term is an integer inner product modulo q.
    pub fn inner_product(&self, rhs: &Self) -> Result<Poly, Error> {
        Self::new(
            self.ring.clone(),
            self.entries.iter().map(Poly::auto).collect(),
        )?
        .dot(rhs)
    }
    /// Sum of squared Euclidean norms: exact below $`2^{256}`$, and `U256::MAX` otherwise, as
    /// [`Poly::norm_squared`]. Never fails; the `Result` is kept for compatibility.
    pub fn norm_squared(&self) -> Result<U256, Error> {
        Ok(self
            .entries
            .iter()
            .fold(U256::ZERO, |acc, p| acc.saturating_add(&p.norm_squared())))
    }
}

/// Dense row-major matrix with checked shape and ring membership.
///
/// The selectors and blocks of compiled statements are built from their nonzero entries, and
/// such a matrix holds only those, since most of their entries are zero and each would
/// otherwise hold a polynomial buffer of its own. It reads the same: [`PolyMat::get`] returns
/// a shared zero for the other entries, and [`PolyMat::entries`] builds the row-major entries
/// on its first call.
#[derive(Clone)]
pub struct PolyMat {
    ring: Arc<Ring>,
    rows: usize,
    cols: usize,
    /// The entries, which cannot change after construction: clones share them.
    entries: Arc<Entries>,
    /// The listed entries ([`Entries`]) classified and, if general, transformed, on the first
    /// product of any clone. Clones share it, also clones made before it. It is never stale,
    /// since the entries cannot change. Not part of the value: equality and `Debug` ignore it.
    terms: Arc<OnceLock<Vec<Term>>>,
}
/// The entries of a [`PolyMat`].
enum Entries {
    /// Every entry, row-major. All of them are listed.
    Dense(Vec<Poly>),
    /// The listed entries with their row-major positions, increasing: the nonzero ones. Every
    /// other entry is `zero`. The row-major entries are built on the first call of
    /// [`PolyMat::entries`] only.
    Sparse {
        listed: Vec<(usize, Poly)>,
        zero: Poly,
        dense: OnceLock<Vec<Poly>>,
    },
}
impl Entries {
    /// The number of listed entries.
    fn len(&self) -> usize {
        match self {
            Entries::Dense(entries) => entries.len(),
            Entries::Sparse { listed, .. } => listed.len(),
        }
    }
    /// Listed entry `k`: its row-major position and value.
    #[inline]
    fn listed(&self, k: usize) -> (usize, &Poly) {
        match self {
            Entries::Dense(entries) => (k, &entries[k]),
            Entries::Sparse { listed, .. } => (listed[k].0, &listed[k].1),
        }
    }
    /// The listed entries of the positions `range`, as indices of listed entries.
    fn range(&self, range: core::ops::Range<usize>) -> core::ops::Range<usize> {
        match self {
            Entries::Dense(_) => range,
            Entries::Sparse { listed, .. } => {
                listed.partition_point(|(p, _)| *p < range.start)
                    ..listed.partition_point(|(p, _)| *p < range.end)
            }
        }
    }
    /// The entry at a row-major position.
    fn at(&self, position: usize) -> &Poly {
        match self {
            Entries::Dense(entries) => &entries[position],
            Entries::Sparse { listed, zero, .. } => {
                match listed.binary_search_by_key(&position, |(p, _)| *p) {
                    Ok(k) => &listed[k].1,
                    Err(_) => zero,
                }
            }
        }
    }
}
impl PartialEq for PolyMat {
    fn eq(&self, rhs: &Self) -> bool {
        self.ring == rhs.ring
            && self.rows == rhs.rows
            && self.cols == rhs.cols
            && match (&*self.entries, &*rhs.entries) {
                (Entries::Dense(a), Entries::Dense(b)) => a == b,
                (a, b) => (0..self.rows * self.cols).all(|p| a.at(p) == b.at(p)),
            }
    }
}
impl Eq for PolyMat {}
impl core::fmt::Debug for PolyMat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut out = f.debug_struct("PolyMat");
        out.field("ring", &self.ring)
            .field("rows", &self.rows)
            .field("cols", &self.cols);
        match &*self.entries {
            Entries::Dense(entries) => out.field("entries", entries),
            Entries::Sparse { listed, .. } => out.field("nonzero", listed),
        };
        out.finish()
    }
}
impl PolyMat {
    /// Construct a matrix; empty shapes are supported.
    pub fn new(
        ring: Arc<Ring>,
        rows: usize,
        cols: usize,
        entries: Vec<Poly>,
    ) -> Result<Self, Error> {
        if rows.checked_mul(cols) != Some(entries.len()) {
            return Err(Error::Dimension);
        }
        if entries.iter().any(|p| p.ring() != &ring) {
            return Err(Error::RingMismatch);
        }
        Ok(Self {
            ring,
            rows,
            cols,
            entries: Arc::new(Entries::Dense(entries)),
            terms: Arc::new(OnceLock::new()),
        })
    }
    /// The matrix with the entries `(row, column, value)`, at most one per position, and
    /// zero entries elsewhere. Only the nonzero entries are held.
    pub(crate) fn sparse(
        ring: Arc<Ring>,
        rows: usize,
        cols: usize,
        entries: Vec<(usize, usize, Poly)>,
    ) -> Result<Self, Error> {
        rows.checked_mul(cols).ok_or(Error::Dimension)?;
        let mut listed = Vec::with_capacity(entries.len());
        for (row, col, value) in entries {
            if row >= rows || col >= cols {
                return Err(Error::Index);
            }
            if value.ring() != &ring {
                return Err(Error::RingMismatch);
            }
            listed.push((row * cols + col, value));
        }
        listed.sort_by_key(|(p, _)| *p);
        if listed.windows(2).any(|w| w[0].0 == w[1].0) {
            return Err(Error::Parameter("duplicate sparse entry"));
        }
        listed.retain(|(_, value)| !value.is_zero());
        listed.shrink_to_fit();
        Ok(Self {
            entries: Arc::new(Entries::Sparse {
                listed,
                zero: Poly::zero(ring.clone()),
                dense: OnceLock::new(),
            }),
            ring,
            rows,
            cols,
            terms: Arc::new(OnceLock::new()),
        })
    }
    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }
    /// Shared ring.
    pub fn ring(&self) -> &Arc<Ring> {
        &self.ring
    }
    /// Row-major entries. A matrix built from its nonzero entries builds them on the first
    /// call, and keeps them.
    pub fn entries(&self) -> &[Poly] {
        match &*self.entries {
            Entries::Dense(entries) => entries,
            Entries::Sparse {
                listed,
                zero,
                dense,
            } => dense.get_or_init(|| {
                let mut out = vec![zero.clone(); self.rows * self.cols];
                for (position, value) in listed {
                    out[*position] = value.clone();
                }
                out
            }),
        }
    }
    /// The number of entries that the entries' buffer has room for: the listed entries.
    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        match &*self.entries {
            Entries::Dense(entries) => entries.capacity(),
            Entries::Sparse { listed, .. } => listed.capacity(),
        }
    }
    /// The listed entries of `row` with their columns, in increasing order: every entry of a
    /// dense matrix, the nonzero ones of a matrix built from them ([`PolyMat::sparse`]).
    pub(crate) fn listed_in_row(&self, row: usize) -> impl Iterator<Item = (usize, &Poly)> {
        let entries = &*self.entries;
        let start = row * self.cols;
        entries.range(start..start + self.cols).map(move |k| {
            let (position, value) = entries.listed(k);
            (position - start, value)
        })
    }
    /// Checked entry access.
    pub fn get(&self, row: usize, col: usize) -> Result<&Poly, Error> {
        if row >= self.rows || col >= self.cols {
            return Err(Error::Index);
        }
        Ok(self.entries.at(row * self.cols + col))
    }
    /// Matrix-vector multiplication with bounded RNS accumulation.
    ///
    /// The entries are classified and transformed once per matrix (on the first product) and
    /// the vector once per product; zero entries are skipped, products with a scalar or
    /// monomial factor are taken in the coefficient domain, and each row sums its other
    /// products transformed, reducing once per [`NADDS`] products. Every step is exact modulo
    /// $`q`$, so each row is the exact dot product of the row with the vector. A matrix built
    /// from its nonzero entries visits those only.
    pub fn mul(&self, vector: &PolyVec) -> Result<PolyVec, Error> {
        self.check_vector(vector)?;
        let shapes: Secret<Vec<Shape>> =
            Secret::new(vector.entries.iter().map(Shape::of).collect());
        let transformed = transform_where(vector, &shapes, &self.meets_general());
        self.rows_with(vector, &shapes, &transformed)
    }
    /// The products of `self` and `other` with one vector, which is classified and transformed
    /// once for both: each product takes the arithmetic of [`PolyMat::mul`], so the results
    /// are those of two calls.
    pub(crate) fn mul_pair(
        &self,
        other: &Self,
        vector: &PolyVec,
    ) -> Result<(PolyVec, PolyVec), Error> {
        self.check_vector(vector)?;
        other.check_vector(vector)?;
        let shapes: Secret<Vec<Shape>> =
            Secret::new(vector.entries.iter().map(Shape::of).collect());
        let meets: Vec<bool> = self
            .meets_general()
            .iter()
            .zip(other.meets_general())
            .map(|(a, b)| *a || b)
            .collect();
        let transformed = transform_where(vector, &shapes, &meets);
        Ok((
            self.rows_with(vector, &shapes, &transformed)?,
            other.rows_with(vector, &shapes, &transformed)?,
        ))
    }
    fn check_vector(&self, vector: &PolyVec) -> Result<(), Error> {
        if vector.len() != self.cols {
            return Err(Error::Dimension);
        }
        if vector.ring != self.ring {
            return Err(Error::RingMismatch);
        }
        Ok(())
    }
    /// The terms of the listed entries, computed on the first product of any clone.
    fn terms(&self) -> &[Term] {
        let entries = &*self.entries;
        self.terms.get_or_init(|| {
            (0..entries.len())
                .map(|k| Term::new(entries.listed(k).1))
                .collect()
        })
    }
    /// The columns with a general entry, where a general vector entry is worth transforming.
    fn meets_general(&self) -> Vec<bool> {
        let mut meets = vec![false; self.cols];
        for (k, term) in self.terms().iter().enumerate() {
            if matches!(term, Term::General(_)) {
                meets[self.entries.listed(k).0 % self.cols] = true;
            }
        }
        meets
    }
    /// The rows of the product with `vector`, whose entries have the shapes `shapes` and, where
    /// `transformed` holds them, transforms.
    fn rows_with(
        &self,
        vector: &PolyVec,
        shapes: &[Shape],
        transformed: &[Option<PolyNtt>],
    ) -> Result<PolyVec, Error> {
        let entries = &*self.entries;
        let terms = self.terms();
        let compute_row = |i: usize| -> Result<Poly, Error> {
            let mut out = Poly::zero(self.ring.clone());
            let mut sum = Accumulator::new(&self.ring);
            for k in entries.range(i * self.cols..(i + 1) * self.cols) {
                let (position, entry) = entries.listed(k);
                let j = position - i * self.cols;
                match (&terms[k], transformed[j].as_ref()) {
                    (Term::Cheap(shape), _) => {
                        shape.add_product_to(&mut out, &vector.entries[j])?;
                    }
                    (Term::General(a), Some(b)) => sum.mac(a, b, &mut out)?,
                    (Term::General(_), None) => {
                        shapes[j].add_product_to(&mut out, entry)?;
                    }
                }
            }
            sum.finish_into(&mut out)?;
            Ok(out)
        };
        // The rows in parallel with the `parallel` feature, in order.
        let out = crate::par::try_map(self.rows, compute_row)?;
        PolyVec::new(self.ring.clone(), out)
    }
}

/// The transforms of the general entries of `vector` in the columns that `needed` marks.
fn transform_where(vector: &PolyVec, shapes: &[Shape], needed: &[bool]) -> Vec<Option<PolyNtt>> {
    (0..vector.entries.len())
        .map(|j| (shapes[j] == Shape::General && needed[j]).then(|| vector.entries[j].to_ntt()))
        .collect()
}

/// An entry of a sparse map. An explicit zero entry holds no polynomial of its own: reads of
/// it return the map's shared zero. A value may be zero too, after a cancellation.
#[derive(Clone)]
enum Slot {
    Zero,
    Value(Poly),
}
impl Slot {
    /// An explicit zero for a zero polynomial, whose buffer is dropped.
    fn new(p: Poly) -> Self {
        if p.is_zero() {
            Slot::Zero
        } else {
            Slot::Value(p)
        }
    }
    /// The entry's own polynomial, a new zero if it is an explicit zero.
    fn get_mut(&mut self, ring: &Arc<Ring>) -> &mut Poly {
        if let Slot::Zero = self {
            *self = Slot::Value(Poly::zero(ring.clone()));
        }
        match self {
            Slot::Value(p) => p,
            Slot::Zero => unreachable!("an explicit zero was just given a value"),
        }
    }
}

/// The zero polynomial that the explicit zero entries of one map share, allocated when first
/// read. Clones start without it.
#[derive(Default)]
struct SharedZero(OnceLock<Poly>);
impl Clone for SharedZero {
    fn clone(&self) -> Self {
        Self::default()
    }
}
impl SharedZero {
    fn get(&self, ring: &Arc<Ring>) -> &Poly {
        self.0.get_or_init(|| Poly::zero(ring.clone()))
    }
}

/// Canonically sorted sparse upper-triangular matrix. Duplicate entries are rejected.
///
/// Explicit zero entries are keys like any other, but share one zero polynomial instead of
/// holding one each. Equality and `Debug` see only the keys and values.
#[derive(Clone)]
pub struct SparsePolyMat {
    ring: Arc<Ring>,
    dim: usize,
    entries: BTreeMap<(u16, u16), Slot>,
    zero: SharedZero,
}
impl PartialEq for SparsePolyMat {
    fn eq(&self, rhs: &Self) -> bool {
        self.ring == rhs.ring && self.dim == rhs.dim && self.entries().eq(rhs.entries())
    }
}
impl Eq for SparsePolyMat {}
impl core::fmt::Debug for SparsePolyMat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SparsePolyMat")
            .field("ring", &self.ring)
            .field("dim", &self.dim)
            .field("entries", &self.entries().collect::<BTreeMap<_, _>>())
            .finish()
    }
}
impl SparsePolyMat {
    /// Construct from `(row, column, coefficient)` tuples.
    pub fn new(ring: Arc<Ring>, dim: usize, entries: Vec<(u16, u16, Poly)>) -> Result<Self, Error> {
        if dim > u16::MAX as usize + 1 {
            return Err(Error::Dimension);
        }
        let mut map = BTreeMap::new();
        for (r, c, p) in entries {
            if r > c || usize::from(c) >= dim {
                return Err(Error::Index);
            }
            if p.ring() != &ring {
                return Err(Error::RingMismatch);
            }
            if map.insert((r, c), Slot::new(p)).is_some() {
                return Err(Error::Parameter("duplicate sparse entry"));
            }
        }
        Ok(Self {
            ring,
            dim,
            entries: map,
            zero: SharedZero::default(),
        })
    }
    /// Dimension of the square form.
    pub fn dimension(&self) -> usize {
        self.dim
    }
    /// Shared ring.
    pub fn ring(&self) -> &Arc<Ring> {
        &self.ring
    }
    /// The number of [`SparsePolyMat::entries`].
    pub(crate) fn key_count(&self) -> usize {
        self.entries.len()
    }
    /// Sorted entries (zeros, if supplied, are preserved).
    pub fn entries(&self) -> impl Iterator<Item = ((u16, u16), &Poly)> {
        self.entries.iter().map(|(idx, slot)| {
            let p = match slot {
                Slot::Zero => self.zero.get(&self.ring),
                Slot::Value(p) => p,
            };
            (*idx, p)
        })
    }
    fn check_key(&self, r: u16, c: u16) -> Result<(), Error> {
        if r > c || usize::from(c) >= self.dim {
            return Err(Error::Index);
        }
        Ok(())
    }
    /// The entry at $`(r,c)`$, inserted as zero if absent, for in-place accumulation. An
    /// explicit zero gets a polynomial of its own.
    pub(crate) fn entry_mut(&mut self, r: u16, c: u16) -> Result<&mut Poly, Error> {
        self.check_key(r, c)?;
        Ok(self
            .entries
            .entry((r, c))
            .or_insert(Slot::Zero)
            .get_mut(&self.ring))
    }
    /// Insert the key $`(r,c)`$ as an explicit zero if it is absent.
    pub(crate) fn insert_key(&mut self, r: u16, c: u16) -> Result<(), Error> {
        self.check_key(r, c)?;
        self.entries.entry((r, c)).or_insert(Slot::Zero);
        Ok(())
    }
    /// The sorted keys, each with its polynomial or `None` for an explicit zero.
    pub(crate) fn slots(&self) -> impl Iterator<Item = ((u16, u16), Option<&Poly>)> {
        self.entries.iter().map(|(key, slot)| {
            let p = match slot {
                Slot::Zero => None,
                Slot::Value(p) => Some(p),
            };
            (*key, p)
        })
    }
    /// The map of strictly increasing keys, each with its polynomial or `None` for an explicit
    /// zero (as a zero polynomial is held too), built without a search per key.
    pub(crate) fn from_sorted(
        ring: Arc<Ring>,
        dim: usize,
        entries: Vec<((u16, u16), Option<Poly>)>,
    ) -> Result<Self, Error> {
        if dim > u16::MAX as usize + 1 {
            return Err(Error::Dimension);
        }
        let mut out = Self {
            ring,
            dim,
            entries: BTreeMap::new(),
            zero: SharedZero::default(),
        };
        for (k, ((r, c), p)) in entries.iter().enumerate() {
            out.check_key(*r, *c)?;
            if k > 0 && entries[k - 1].0 >= (*r, *c) {
                return Err(Error::Parameter("unsorted sparse entries"));
            }
            if p.as_ref().is_some_and(|p| p.ring() != &out.ring) {
                return Err(Error::RingMismatch);
            }
        }
        out.entries = entries
            .into_iter()
            .map(|(key, p)| (key, p.map_or(Slot::Zero, Slot::new)))
            .collect();
        Ok(out)
    }
    /// Insert as explicit zeros the absent keys $`(\min(i,j),\max(i,j))`$ for $`i`$ in `a` and
    /// $`j`$ in `b`, both strictly increasing: the keys that `insert_key` on every pair inserts,
    /// found by merging the columns of each row with that row's keys instead of a search per
    /// pair.
    pub(crate) fn insert_outer_keys(&mut self, a: &[u16], b: &[u16]) -> Result<(), Error> {
        for list in [a, b] {
            if list.windows(2).any(|w| w[0] >= w[1]) {
                return Err(Error::Parameter("unsorted sparse keys"));
            }
            if list.last().is_some_and(|&c| usize::from(c) >= self.dim) {
                return Err(Error::Index);
            }
        }
        let mut missing = Vec::new();
        let (mut x, mut y) = (0, 0);
        // The rows are the keys of either list, in increasing order: row r takes the columns
        // c >= r of `b` if r is in `a`, and those of `a` if r is in `b`.
        while let Some(r) = match (a.get(x), b.get(y)) {
            (Some(&i), Some(&j)) => Some(i.min(j)),
            (Some(&i), None) => Some(i),
            (None, Some(&j)) => Some(j),
            (None, None) => None,
        } {
            let (in_a, in_b) = (a.get(x) == Some(&r), b.get(y) == Some(&r));
            let from_b: &[u16] = if in_a { &b[y..] } else { &[] };
            let from_a: &[u16] = if in_b { &a[x..] } else { &[] };
            let mut present = self
                .entries
                .range((r, r)..=(r, u16::MAX))
                .map(|(key, _)| key.1)
                .peekable();
            let (mut p, mut q) = (0, 0);
            loop {
                let c = match (from_a.get(p), from_b.get(q)) {
                    (Some(&u), Some(&v)) => {
                        p += usize::from(u <= v);
                        q += usize::from(v <= u);
                        u.min(v)
                    }
                    (Some(&u), None) => {
                        p += 1;
                        u
                    }
                    (None, Some(&v)) => {
                        q += 1;
                        v
                    }
                    (None, None) => break,
                };
                while present.next_if(|&e| e < c).is_some() {}
                if present.peek() != Some(&c) {
                    missing.push((r, c));
                }
            }
            x += usize::from(in_a);
            y += usize::from(in_b);
        }
        self.insert_absent_zeros(missing);
        Ok(())
    }
    /// Insert explicit zeros at the strictly increasing keys `missing`, none of them present:
    /// one at a time when they are few against the map, by rebuilding it from the merged sorted
    /// keys otherwise.
    fn insert_absent_zeros(&mut self, missing: Vec<(u16, u16)>) {
        if missing.len() * 8 < self.entries.len() {
            for key in missing {
                self.entries.insert(key, Slot::Zero);
            }
            return;
        }
        let old = core::mem::take(&mut self.entries);
        let mut new = missing.into_iter().peekable();
        let mut merged = Vec::with_capacity(old.len() + new.len());
        for (key, slot) in old {
            while let Some(k) = new.next_if(|k| *k < key) {
                merged.push((k, Slot::Zero));
            }
            merged.push((key, slot));
        }
        merged.extend(new.map(|k| (k, Slot::Zero)));
        self.entries = merged.into_iter().collect();
    }
    /// Add `p` at $`(r,c)`$, inserting a copy if the entry is absent.
    pub(crate) fn add_at(&mut self, r: u16, c: u16, p: &Poly) -> Result<(), Error> {
        self.check_key(r, c)?;
        if p.ring() != &self.ring {
            return Err(Error::RingMismatch);
        }
        if p.is_zero() {
            return self.insert_key(r, c);
        }
        match self.entries.get_mut(&(r, c)) {
            Some(slot) => slot.get_mut(&self.ring).add_assign(p),
            None => {
                self.entries.insert((r, c), Slot::Value(p.clone()));
                Ok(())
            }
        }
    }
    /// The entries that hold a polynomial, in order; explicit zeros are skipped.
    pub(crate) fn values_mut(&mut self) -> impl Iterator<Item = &mut Poly> {
        self.entries.values_mut().filter_map(|slot| match slot {
            Slot::Value(p) => Some(p),
            Slot::Zero => None,
        })
    }
    /// Insert the keys of `rhs` that are absent, as explicit zeros, and pair each entry of
    /// `rhs` that holds a polynomial with the entry at its key (given a polynomial of its own),
    /// in key order. The explicit zeros of `rhs` add nothing and are not paired.
    pub(crate) fn pairs_mut<'a>(
        &'a mut self,
        rhs: &'a Self,
    ) -> Result<Vec<(&'a mut Poly, &'a Poly)>, Error> {
        for (r, c) in rhs.entries.keys() {
            self.insert_key(*r, *c)?;
        }
        Ok(pair(
            &self.ring,
            self.entries.iter_mut(),
            rhs.entries.iter(),
        ))
    }
    /// The quadratic part of [`QuadEq::conjugate`](crate::quad::QuadEq::conjugate): entry
    /// $`(r,c)`$ moves to the key of $`(r\oplus1,c\oplus1)`$ with its automorphism image, and
    /// explicit zeros stay explicit zeros. The dimension must be even.
    pub(crate) fn conjugated(&self) -> Result<Self, Error> {
        let mut map = BTreeMap::new();
        for (&(r, c), slot) in &self.entries {
            let (r, c) = ((r ^ 1).min(c ^ 1), (r ^ 1).max(c ^ 1));
            self.check_key(r, c)?;
            let slot = match slot {
                Slot::Zero => Slot::Zero,
                Slot::Value(p) => Slot::new(p.auto()),
            };
            if map.insert((r, c), slot).is_some() {
                return Err(Error::Parameter("duplicate sparse entry"));
            }
        }
        Ok(Self {
            ring: self.ring.clone(),
            dim: self.dim,
            entries: map,
            zero: SharedZero::default(),
        })
    }
    /// Enlarge the variable space, keeping the entries.
    pub(crate) fn set_dimension(&mut self, dim: usize) -> Result<(), Error> {
        if dim < self.dim || dim > u16::MAX as usize + 1 {
            return Err(Error::Dimension);
        }
        self.dim = dim;
        Ok(())
    }
    /// Evaluate $`x^T R_2 y`$.
    ///
    /// Computed as $`\sum_rx_r\big(\sum_cR_{rc}y_c\big)`$: the entries of a row are summed
    /// first (zero entries skipped, scalar and monomial products in the coefficient domain,
    /// the others as transformed products, each $`y_c`$ transformed at most once), the inner
    /// sum is reduced, and each row takes one product by $`x_r`$. Every step is exact modulo
    /// $`q`$, so the value is the sum of the products $`x_rR_{rc}y_c`$.
    pub fn bilinear(&self, x: &PolyVec, y: &PolyVec) -> Result<Poly, Error> {
        if x.len() != self.dim || y.len() != self.dim {
            return Err(Error::Dimension);
        }
        if x.ring != self.ring || y.ring != self.ring {
            return Err(Error::RingMismatch);
        }
        let ring = &self.ring;
        // With x = y (an evaluation), x_r shares the transforms of y.
        let same = std::ptr::eq(x, y);
        let mut out = Poly::zero(ring.clone());
        let mut outer = Accumulator::new(ring);
        self.for_each_row_sum(y, |r, inner, y_ntt| {
            // The row is complete: out += x_r (sum_c R_rc y_c).
            let xr = &x.entries[r];
            match (Shape::of(xr), Shape::of(inner)) {
                (Shape::Zero, _) | (_, Shape::Zero) => {}
                (Shape::General, Shape::General) => {
                    let own;
                    let a = if same {
                        &*y_ntt[r].get_or_insert_with(|| xr.to_ntt())
                    } else {
                        own = xr.to_ntt();
                        &own
                    };
                    outer.mac(a, &inner.to_ntt(), &mut out)?;
                }
                (Shape::General, cheap) => {
                    cheap.add_product_to(&mut out, xr)?;
                }
                (cheap, _) => {
                    cheap.add_product_to(&mut out, inner)?;
                }
            }
            Ok(())
        })?;
        outer.finish_into(&mut out)?;
        Ok(out)
    }
    /// The constant coefficient of [`SparsePolyMat::bilinear`] with $`x=y`$, exactly: each row's
    /// inner sum as `bilinear` computes it, then only the constant coefficient of its product by
    /// $`y_r`$, a dot product, instead of the product.
    pub(crate) fn bilinear_constant(
        &self,
        y: &PolyVec,
        sum: &mut ConstantSum<'_>,
    ) -> Result<(), Error> {
        if y.len() != self.dim {
            return Err(Error::Dimension);
        }
        if y.ring != self.ring {
            return Err(Error::RingMismatch);
        }
        self.for_each_row_sum(y, |r, inner, _| {
            let yr = &y.entries[r];
            if !yr.is_zero() && !inner.is_zero() {
                sum.add_product(yr, inner);
            }
            Ok(())
        })
    }
    /// For each row $`r`$ that has an entry, in order, `f(r, inner, y_ntt)` with
    /// $`\mathit{inner}=\sum_cR_{rc}y_c`$: zero entries skipped, scalar and monomial products in
    /// the coefficient domain, the others as transformed products, each $`y_c`$ transformed at
    /// most once into `y_ntt`, which `f` may fill and use as well.
    fn for_each_row_sum(
        &self,
        y: &PolyVec,
        mut f: impl FnMut(usize, &Poly, &mut [Option<PolyNtt>]) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let ring = &self.ring;
        let mut y_shapes: Secret<Vec<Option<Shape>>> = Secret::new(vec![None; y.len()]);
        let mut y_ntt: Vec<Option<PolyNtt>> = vec![None; y.len()];
        let mut inner = Poly::zero(ring.clone());
        let mut inner_sum = Accumulator::new(ring);
        let mut entries = self.entries.iter().peekable();
        while let Some((&(r, c), slot)) = entries.next() {
            let (r, c) = (usize::from(r), usize::from(c));
            let yc = &y.entries[c];
            if let Slot::Value(p) = slot
                && let shape = Shape::of(p)
                && shape != Shape::Zero
            {
                let y_shape = *y_shapes[c].get_or_insert_with(|| Shape::of(yc));
                match (shape, y_shape) {
                    (Shape::General, Shape::General) => {
                        let b = y_ntt[c].get_or_insert_with(|| yc.to_ntt());
                        inner_sum.mac(&p.to_ntt(), b, &mut inner)?;
                    }
                    (Shape::General, cheap) => {
                        cheap.add_product_to(&mut inner, p)?;
                    }
                    (cheap, _) => {
                        cheap.add_product_to(&mut inner, yc)?;
                    }
                }
            }
            if entries
                .peek()
                .is_some_and(|((next, _), _)| usize::from(*next) == r)
            {
                continue;
            }
            inner_sum.finish_into(&mut inner)?;
            f(r, &inner, &mut y_ntt)?;
            inner = Poly::zero(ring.clone());
        }
        Ok(())
    }
}

/// Canonically sorted sparse vector with checked indices and ring membership. Explicit zero
/// entries share one zero polynomial, as in [`SparsePolyMat`].
#[derive(Clone)]
pub struct SparsePolyVec {
    ring: Arc<Ring>,
    dim: usize,
    entries: BTreeMap<u16, Slot>,
    zero: SharedZero,
}
impl PartialEq for SparsePolyVec {
    fn eq(&self, rhs: &Self) -> bool {
        self.ring == rhs.ring && self.dim == rhs.dim && self.entries().eq(rhs.entries())
    }
}
impl Eq for SparsePolyVec {}
impl core::fmt::Debug for SparsePolyVec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SparsePolyVec")
            .field("ring", &self.ring)
            .field("dim", &self.dim)
            .field("entries", &self.entries().collect::<BTreeMap<_, _>>())
            .finish()
    }
}
impl SparsePolyVec {
    /// Construct from `(index, coefficient)` pairs.
    pub fn new(ring: Arc<Ring>, dim: usize, entries: Vec<(u16, Poly)>) -> Result<Self, Error> {
        if dim > u16::MAX as usize + 1 {
            return Err(Error::Dimension);
        }
        let mut map = BTreeMap::new();
        for (i, p) in entries {
            if usize::from(i) >= dim {
                return Err(Error::Index);
            }
            if p.ring() != &ring {
                return Err(Error::RingMismatch);
            }
            if map.insert(i, Slot::new(p)).is_some() {
                return Err(Error::Parameter("duplicate sparse entry"));
            }
        }
        Ok(Self {
            ring,
            dim,
            entries: map,
            zero: SharedZero::default(),
        })
    }
    /// Vector dimension.
    pub fn dimension(&self) -> usize {
        self.dim
    }
    /// Shared ring.
    pub fn ring(&self) -> &Arc<Ring> {
        &self.ring
    }
    /// The number of [`SparsePolyVec::entries`].
    pub(crate) fn key_count(&self) -> usize {
        self.entries.len()
    }
    /// Sorted entries.
    pub fn entries(&self) -> impl Iterator<Item = (u16, &Poly)> {
        self.entries.iter().map(|(i, slot)| {
            let p = match slot {
                Slot::Zero => self.zero.get(&self.ring),
                Slot::Value(p) => p,
            };
            (*i, p)
        })
    }
    fn check_key(&self, i: u16) -> Result<(), Error> {
        if usize::from(i) >= self.dim {
            return Err(Error::Index);
        }
        Ok(())
    }
    /// The entry at $`i`$, inserted as zero if absent, for in-place accumulation. An explicit
    /// zero gets a polynomial of its own.
    pub(crate) fn entry_mut(&mut self, i: u16) -> Result<&mut Poly, Error> {
        self.check_key(i)?;
        Ok(self
            .entries
            .entry(i)
            .or_insert(Slot::Zero)
            .get_mut(&self.ring))
    }
    /// Insert the key $`i`$ as an explicit zero if it is absent.
    pub(crate) fn insert_key(&mut self, i: u16) -> Result<(), Error> {
        self.check_key(i)?;
        self.entries.entry(i).or_insert(Slot::Zero);
        Ok(())
    }
    /// The sorted keys, each with its polynomial or `None` for an explicit zero.
    pub(crate) fn slots(&self) -> impl Iterator<Item = (u16, Option<&Poly>)> {
        self.entries.iter().map(|(key, slot)| {
            let p = match slot {
                Slot::Zero => None,
                Slot::Value(p) => Some(p),
            };
            (*key, p)
        })
    }
    /// The vector of strictly increasing keys, each with its polynomial or `None` for an
    /// explicit zero (as a zero polynomial is held too), built without a search per key.
    pub(crate) fn from_sorted(
        ring: Arc<Ring>,
        dim: usize,
        entries: Vec<(u16, Option<Poly>)>,
    ) -> Result<Self, Error> {
        if dim > u16::MAX as usize + 1 {
            return Err(Error::Dimension);
        }
        let mut out = Self {
            ring,
            dim,
            entries: BTreeMap::new(),
            zero: SharedZero::default(),
        };
        for (k, (i, p)) in entries.iter().enumerate() {
            out.check_key(*i)?;
            if k > 0 && entries[k - 1].0 >= *i {
                return Err(Error::Parameter("unsorted sparse entries"));
            }
            if p.as_ref().is_some_and(|p| p.ring() != &out.ring) {
                return Err(Error::RingMismatch);
            }
        }
        out.entries = entries
            .into_iter()
            .map(|(key, p)| (key, p.map_or(Slot::Zero, Slot::new)))
            .collect();
        Ok(out)
    }
    /// Add `p` at $`i`$, inserting a copy if the entry is absent.
    pub(crate) fn add_at(&mut self, i: u16, p: &Poly) -> Result<(), Error> {
        self.check_key(i)?;
        if p.ring() != &self.ring {
            return Err(Error::RingMismatch);
        }
        if p.is_zero() {
            return self.insert_key(i);
        }
        match self.entries.get_mut(&i) {
            Some(slot) => slot.get_mut(&self.ring).add_assign(p),
            None => {
                self.entries.insert(i, Slot::Value(p.clone()));
                Ok(())
            }
        }
    }
    /// The entries that hold a polynomial, in order; explicit zeros are skipped.
    pub(crate) fn values_mut(&mut self) -> impl Iterator<Item = &mut Poly> {
        self.entries.values_mut().filter_map(|slot| match slot {
            Slot::Value(p) => Some(p),
            Slot::Zero => None,
        })
    }
    /// [`SparsePolyMat::pairs_mut`] for vectors.
    pub(crate) fn pairs_mut<'a>(
        &'a mut self,
        rhs: &'a Self,
    ) -> Result<Vec<(&'a mut Poly, &'a Poly)>, Error> {
        for i in rhs.entries.keys() {
            self.insert_key(*i)?;
        }
        Ok(pair(
            &self.ring,
            self.entries.iter_mut(),
            rhs.entries.iter(),
        ))
    }
    /// The linear part of [`QuadEq::conjugate`](crate::quad::QuadEq::conjugate): entry $`i`$
    /// moves to $`i\oplus1`$ with its automorphism image; explicit zeros stay explicit zeros.
    /// The dimension must be even.
    pub(crate) fn conjugated(&self) -> Result<Self, Error> {
        let mut map = BTreeMap::new();
        for (&i, slot) in &self.entries {
            self.check_key(i ^ 1)?;
            let slot = match slot {
                Slot::Zero => Slot::Zero,
                Slot::Value(p) => Slot::new(p.auto()),
            };
            if map.insert(i ^ 1, slot).is_some() {
                return Err(Error::Parameter("duplicate sparse entry"));
            }
        }
        Ok(Self {
            ring: self.ring.clone(),
            dim: self.dim,
            entries: map,
            zero: SharedZero::default(),
        })
    }
    /// Enlarge the variable space, keeping the entries.
    pub(crate) fn set_dimension(&mut self, dim: usize) -> Result<(), Error> {
        if dim < self.dim || dim > u16::MAX as usize + 1 {
            return Err(Error::Dimension);
        }
        self.dim = dim;
        Ok(())
    }
    /// Sparse dot product: zero entries skipped, products with a scalar or monomial factor in
    /// the coefficient domain, the others as transformed products reduced together; exact.
    pub fn dot(&self, x: &PolyVec) -> Result<Poly, Error> {
        if x.len() != self.dim {
            return Err(Error::Dimension);
        }
        if x.ring != self.ring {
            return Err(Error::RingMismatch);
        }
        let mut out = Poly::zero(self.ring.clone());
        let mut sum = Accumulator::new(&self.ring);
        for (i, slot) in &self.entries {
            let Slot::Value(p) = slot else {
                continue;
            };
            let xi = &x.entries[usize::from(*i)];
            match Shape::of(p) {
                Shape::Zero => {}
                Shape::General => match Shape::of(xi) {
                    Shape::General => sum.mac(&p.to_ntt(), &xi.to_ntt(), &mut out)?,
                    cheap => {
                        cheap.add_product_to(&mut out, p)?;
                    }
                },
                cheap => {
                    cheap.add_product_to(&mut out, xi)?;
                }
            }
        }
        sum.finish_into(&mut out)?;
        Ok(out)
    }
    /// The constant coefficient of [`SparsePolyVec::dot`], added to `sum`: a dot product of
    /// length $`d`$ per nonzero entry instead of a product.
    pub(crate) fn dot_constant(&self, x: &PolyVec, sum: &mut ConstantSum<'_>) -> Result<(), Error> {
        if x.len() != self.dim {
            return Err(Error::Dimension);
        }
        if x.ring != self.ring {
            return Err(Error::RingMismatch);
        }
        for (i, slot) in &self.entries {
            let Slot::Value(p) = slot else {
                continue;
            };
            let xi = &x.entries[usize::from(*i)];
            if !p.is_zero() && !xi.is_zero() {
                sum.add_product(p, xi);
            }
        }
        Ok(())
    }
}

/// Pair each entry of `rhs` that holds a polynomial with the entry of `mine` at its key, given
/// a polynomial of its own: every key of `rhs` must be a key of `mine`. Both iterate in key
/// order.
fn pair<'a, K: Ord + 'a>(
    ring: &Arc<Ring>,
    mine: impl Iterator<Item = (&'a K, &'a mut Slot)>,
    rhs: impl Iterator<Item = (&'a K, &'a Slot)>,
) -> Vec<(&'a mut Poly, &'a Poly)> {
    let mut mine = mine;
    rhs.filter_map(|(key, slot)| match slot {
        Slot::Zero => None,
        Slot::Value(p) => {
            let entry = mine
                .find(|(k, _)| *k == key)
                .expect("every key of rhs is a key of self")
                .1;
            Some((entry.get_mut(ring), p))
        }
    })
    .collect()
}

#[cfg(test)]
mod oracle_tests;
#[cfg(test)]
mod zero_tests;
