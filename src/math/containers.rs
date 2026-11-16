use super::{
    Poly, PolyNtt, Ring,
    constant::ConstantSum,
    terms::{Accumulator, Shape, Term},
};
use crate::{Error, params::moduli::NADDS};
use crypto_bigint::U256;
use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock},
};
use zeroize::Zeroizing;

/// A ring-checked vector of polynomials, including empty vectors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolyVec {
    ring: Arc<Ring>,
    entries: Vec<Poly>,
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
/// A matrix built from its nonzero entries ([`PolyMat::sparse`]) holds only those: a selector
/// or a block of a statement has mostly zero entries, each of which would otherwise hold a
/// polynomial buffer of its own.
#[derive(Clone)]
pub struct PolyMat {
    ring: Arc<Ring>,
    rows: usize,
    cols: usize,
    /// The entries, which cannot change after construction: clones share them.
    entries: Arc<Entries>,
    /// The listed entries ([`Entries`]) classified and, if general, transformed, on the first
    /// product of any clone; clones share it, also clones made before it. It is never stale,
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
        if vector.len() != self.cols {
            return Err(Error::Dimension);
        }
        if vector.ring != self.ring {
            return Err(Error::RingMismatch);
        }
        let entries = &*self.entries;
        let terms = self.terms.get_or_init(|| {
            (0..entries.len())
                .map(|k| Term::new(entries.listed(k).1))
                .collect()
        });
        let shapes: Zeroizing<Vec<Shape>> =
            Zeroizing::new(vector.entries.iter().map(Shape::of).collect());
        // Transform a vector entry only if it is general and meets a general entry.
        let mut meets_general = vec![false; self.cols];
        for (k, term) in terms.iter().enumerate() {
            if matches!(term, Term::General(_)) {
                meets_general[entries.listed(k).0 % self.cols] = true;
            }
        }
        let transformed: Vec<Option<PolyNtt>> = (0..self.cols)
            .map(|j| {
                let needed = shapes[j] == Shape::General && meets_general[j];
                needed.then(|| vector.entries[j].to_ntt())
            })
            .collect();
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
        let mut y_shapes: Zeroizing<Vec<Option<Shape>>> = Zeroizing::new(vec![None; y.len()]);
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
