//! Products of the containers against reference formulas:
//! `SparsePolyMat::bilinear` and `SparsePolyVec::dot` against the sums of `mul` products of
//! every entry, with `x = y`, empty forms, zero, scalar, monomial and general entries and
//! vectors, and rows and sums of more than `NADDS` general products at the largest centred
//! magnitudes; `PolyMat::mul` with its cache against row-wise `PolyVec::dot`, including more
//! than `NADDS` columns, repeated products, clones, and equality and `Debug` that ignore the
//! cache. With the `parallel` feature the rows are computed in parallel.
use super::*;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// A uniform 256-bit value, from bytes, so that it does not depend on the limb width.
    fn uniform(&mut self) -> U256 {
        let words = [self.next(), self.next(), self.next(), self.next()];
        U256::from_le_slice(&words.map(u64::to_le_bytes).concat())
    }
    fn value(&mut self, ring: &Arc<Ring>) -> U256 {
        match self.below(5) {
            0 => U256::ONE,
            1 => ring.modulus().wrapping_sub(&U256::ONE),
            2 => ring.half,
            3 => ring.half.wrapping_add(&U256::ONE),
            _ => ring.reduce(&self.uniform()),
        }
    }
    /// Zero, a scalar, a monomial or dense.
    pub(super) fn poly(&mut self, ring: &Arc<Ring>) -> Poly {
        let d = ring.degree();
        let mut coeffs = vec![U256::ZERO; d];
        match self.below(4) {
            0 => {}
            1 => coeffs[0] = self.value(ring),
            2 => coeffs[1 + self.below(d as u64 - 1) as usize] = self.value(ring),
            _ => coeffs.iter_mut().for_each(|x| *x = self.value(ring)),
        }
        Poly::from_u256(ring.clone(), coeffs).unwrap()
    }
    fn vector(&mut self, ring: &Arc<Ring>, n: usize) -> PolyVec {
        PolyVec::new(ring.clone(), (0..n).map(|_| self.poly(ring)).collect()).unwrap()
    }
}

/// Reference `bilinear`: the sum of x_r R_rc y_c over the entries.
fn bilinear_by_entries(m: &SparsePolyMat, x: &PolyVec, y: &PolyVec) -> Result<Poly, Error> {
    if x.len() != m.dimension() || y.len() != m.dimension() {
        return Err(Error::Dimension);
    }
    if x.ring() != m.ring() || y.ring() != m.ring() {
        return Err(Error::RingMismatch);
    }
    m.entries()
        .try_fold(Poly::zero(m.ring().clone()), |sum, ((r, c), p)| {
            sum.add(
                &x.entries()[usize::from(r)]
                    .mul(p)?
                    .mul(&y.entries()[usize::from(c)])?,
            )
        })
}

/// Reference `dot`: the sum of p_i x_i over the entries.
fn dot_by_entries(v: &SparsePolyVec, x: &PolyVec) -> Result<Poly, Error> {
    if x.len() != v.dimension() {
        return Err(Error::Dimension);
    }
    if x.ring() != v.ring() {
        return Err(Error::RingMismatch);
    }
    v.entries()
        .try_fold(Poly::zero(v.ring().clone()), |sum, (i, p)| {
            sum.add(&p.mul(&x.entries()[usize::from(i)])?)
        })
}

pub(super) fn rings() -> Vec<Arc<Ring>> {
    let wide = U256::ONE
        .shl_vartime(240)
        .wrapping_add(&U256::from_u16(325));
    vec![
        Ring::new(3, 64).unwrap(),
        Ring::new(1099511627917, 64).unwrap(),
        Ring::new(1099511627917, 128).unwrap(),
        Ring::new((1 << 62) - 57, 64).unwrap(),
        Ring::new(i128::MAX, 64).unwrap(),
        Ring::with_modulus(wide, 64).unwrap(),
    ]
}

#[test]
fn bilinear_forms_and_dot_products_equal_the_entrywise_sums() {
    let mut rng = Rng(21);
    for ring in rings() {
        for case in 0..60 {
            let n = 1 + rng.below(14) as usize;
            let mut matrix = BTreeMap::new();
            for _ in 0..rng.below(3 * n as u64 + 1) {
                let (i, j) = (rng.below(n as u64) as u16, rng.below(n as u64) as u16);
                matrix.insert((i.min(j), i.max(j)), rng.poly(&ring));
            }
            let m = SparsePolyMat::new(
                ring.clone(),
                n,
                matrix.into_iter().map(|((i, j), p)| (i, j, p)).collect(),
            )
            .unwrap();
            let mut vector = BTreeMap::new();
            for _ in 0..rng.below(n as u64 + 1) {
                vector.insert(rng.below(n as u64) as u16, rng.poly(&ring));
            }
            let v = SparsePolyVec::new(ring.clone(), n, vector.into_iter().collect()).unwrap();
            let (x, y) = (rng.vector(&ring, n), rng.vector(&ring, n));
            assert_eq!(
                m.bilinear(&x, &y),
                bilinear_by_entries(&m, &x, &y),
                "case {case}"
            );
            assert_eq!(m.bilinear(&y, &x), bilinear_by_entries(&m, &y, &x));
            // x = y, by reference (shared transforms) and as an equal copy.
            assert_eq!(m.bilinear(&x, &x), bilinear_by_entries(&m, &x, &x));
            assert_eq!(m.bilinear(&x, &x.clone()), bilinear_by_entries(&m, &x, &x));
            assert_eq!(v.dot(&x), dot_by_entries(&v, &x));
        }
    }
}

#[test]
fn empty_forms_and_mismatches_behave_as_the_reference_formulas() {
    let ring = Ring::new(1099511627917, 64).unwrap();
    let mut rng = Rng(22);
    let x = rng.vector(&ring, 5);
    let m = SparsePolyMat::new(ring.clone(), 5, vec![]).unwrap();
    let v = SparsePolyVec::new(ring.clone(), 5, vec![]).unwrap();
    assert_eq!(m.bilinear(&x, &x), Ok(Poly::zero(ring.clone())));
    assert_eq!(v.dot(&x), Ok(Poly::zero(ring.clone())));
    let empty = PolyVec::zero(ring.clone(), 0);
    let m0 = SparsePolyMat::new(ring.clone(), 0, vec![]).unwrap();
    assert_eq!(m0.bilinear(&empty, &empty), Ok(Poly::zero(ring.clone())));
    let short = rng.vector(&ring, 4);
    let other = rng.vector(&Ring::new(13, 64).unwrap(), 5);
    for (a, b) in [(&x, &short), (&short, &x), (&x, &other), (&other, &x)] {
        assert_eq!(m.bilinear(a, b), bilinear_by_entries(&m, a, b));
    }
    for a in [&short, &other] {
        assert_eq!(v.dot(a), dot_by_entries(&v, a));
    }
}

#[test]
fn rows_and_sums_of_more_than_nadds_general_products_are_exact() {
    for ring in rings() {
        let n = 2 * NADDS + 5;
        let half = Poly::from_u256(ring.clone(), vec![ring.half; ring.degree()]).unwrap();
        let low = half.neg();
        // Row 0 holds n general entries; rows 1.. hold one each, so that the outer sum also
        // takes more than NADDS products. All at the largest centred magnitudes.
        let mut entries: Vec<(u16, u16, Poly)> = (0..n)
            .map(|c| {
                (
                    0,
                    c as u16,
                    if c % 3 == 0 {
                        low.clone()
                    } else {
                        half.clone()
                    },
                )
            })
            .collect();
        entries.extend((1..n).map(|r| (r as u16, r as u16, half.clone())));
        let m = SparsePolyMat::new(ring.clone(), n, entries).unwrap();
        let x = PolyVec::new(
            ring.clone(),
            (0..n)
                .map(|i| {
                    if i % 2 == 0 {
                        half.clone()
                    } else {
                        low.clone()
                    }
                })
                .collect(),
        )
        .unwrap();
        let y = PolyVec::new(ring.clone(), vec![half.clone(); n]).unwrap();
        assert_eq!(m.bilinear(&x, &y), bilinear_by_entries(&m, &x, &y));
        assert_eq!(m.bilinear(&x, &x), bilinear_by_entries(&m, &x, &x));
        let v = SparsePolyVec::new(
            ring.clone(),
            n,
            (0..n).map(|i| (i as u16, half.clone())).collect(),
        )
        .unwrap();
        assert_eq!(v.dot(&x), dot_by_entries(&v, &x));
    }
}

/// Reference `PolyMat::mul`: each row's `PolyVec::dot` with the vector.
fn mul_by_rows(m: &PolyMat, v: &PolyVec) -> Result<PolyVec, Error> {
    if v.len() != m.cols() {
        return Err(Error::Dimension);
    }
    if v.ring() != m.ring() {
        return Err(Error::RingMismatch);
    }
    let rows = (0..m.rows())
        .map(|i| {
            PolyVec::new(
                m.ring().clone(),
                m.entries()[i * m.cols()..(i + 1) * m.cols()].to_vec(),
            )?
            .dot(v)
        })
        .collect::<Result<Vec<_>, Error>>()?;
    PolyVec::new(m.ring().clone(), rows)
}

#[test]
fn cached_matrix_products_equal_row_dot_products() {
    let mut rng = Rng(23);
    for ring in rings() {
        for case in 0..30 {
            let (rows, cols) = match case % 6 {
                0 => (3, 2 * NADDS + 3),
                1 => (0, 4),
                2 => (3, 0),
                _ => (1 + rng.below(6) as usize, 1 + rng.below(10) as usize),
            };
            let entries = (0..rows * cols).map(|_| rng.poly(&ring)).collect();
            let m = PolyMat::new(ring.clone(), rows, cols, entries).unwrap();
            let fresh = m.clone();
            let debug = format!("{m:?}");
            for _ in 0..3 {
                let v = rng.vector(&ring, cols);
                let expected = mul_by_rows(&m, &v).unwrap();
                assert_eq!(m.mul(&v).unwrap(), expected, "case {case}");
                // A clone made after the first product shares the cache.
                assert_eq!(m.clone().mul(&v).unwrap(), expected);
            }
            assert!(m.terms.get().is_some());
            assert!(fresh.terms.get().is_none());
            assert_eq!(m, fresh, "equality ignores the cache");
            assert_eq!(format!("{m:?}"), debug, "Debug ignores the cache");
        }
    }
}

#[test]
fn matrix_products_at_the_largest_magnitudes_and_mismatches() {
    for ring in rings() {
        let half = Poly::from_u256(ring.clone(), vec![ring.half; ring.degree()]).unwrap();
        let low = half.neg();
        let cols = 2 * NADDS + 1;
        let entries = (0..2 * cols)
            .map(|i| {
                if i % 5 == 0 {
                    low.clone()
                } else {
                    half.clone()
                }
            })
            .collect();
        let m = PolyMat::new(ring.clone(), 2, cols, entries).unwrap();
        let v = PolyVec::new(
            ring.clone(),
            (0..cols)
                .map(|i| {
                    if i % 2 == 0 {
                        half.clone()
                    } else {
                        low.clone()
                    }
                })
                .collect(),
        )
        .unwrap();
        assert_eq!(m.mul(&v), mul_by_rows(&m, &v));
    }
    let ring = Ring::new(1099511627917, 64).unwrap();
    let mut rng = Rng(24);
    let m = PolyMat::new(
        ring.clone(),
        2,
        3,
        (0..6).map(|_| rng.poly(&ring)).collect(),
    )
    .unwrap();
    for v in [
        rng.vector(&ring, 2),
        rng.vector(&ring, 4),
        rng.vector(&Ring::new(13, 64).unwrap(), 3),
    ] {
        assert_eq!(m.mul(&v), mul_by_rows(&m, &v));
    }
}
