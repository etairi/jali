//! Projection rows from prepared forms against a reference computation:
//! `Prepared::projection` and `projection` against the sum of `f_j.scale(sigma(r_j))` by
//! copying additions, for random forms with zero, scalar, monomial and general coefficients
//! and more than `NADDS` general terms in a column, and `projection_equations` against a
//! reference by copies on random statements, with both range blocks, either one or none.
use super::*;
use crate::lnp::{AffineBlock, L2Block};
use crate::math::{PolyMat, SparsePolyMat};

pub(super) struct Rng(pub(super) u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub(super) fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// A uniform 256-bit value, from bytes, so that it does not depend on the limb width.
    fn uniform(&mut self) -> U256 {
        let words = [self.next(), self.next(), self.next(), self.next()];
        U256::from_le_slice(&words.map(u64::to_le_bytes).concat())
    }
    fn value(&mut self, ring: &Arc<Ring>) -> U256 {
        match self.below(4) {
            0 => U256::ONE,
            1 => ring.modulus().wrapping_sub(&U256::ONE),
            2 => ring.half,
            _ => ring.reduce(&self.uniform()),
        }
    }
    /// Zero, a scalar, a monomial or dense, each a quarter of the time.
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
    /// A linear form with entries at some of `keys`.
    fn affine(&mut self, ring: &Arc<Ring>, dim: usize, keys: &[u16]) -> QuadEq {
        let mut entries = BTreeMap::new();
        for _ in 0..self.below(keys.len() as u64 + 1) {
            entries.insert(
                keys[self.below(keys.len() as u64) as usize],
                self.poly(ring),
            );
        }
        let mut form = QuadEq::zero(ring.clone(), dim).unwrap();
        form.r1 = SparsePolyVec::new(ring.clone(), dim, entries.into_iter().collect()).unwrap();
        form.r0 = self.poly(ring);
        form
    }
    /// A projection row: Bin_1 coefficients under sigma, as `projection_equations` makes them.
    fn row(&mut self, ring: &Arc<Ring>) -> Poly {
        let coeffs = (0..ring.degree())
            .map(|_| [-1, 0, 0, 1][self.below(4) as usize])
            .collect();
        Poly::new(ring.clone(), coeffs).unwrap().auto()
    }
    fn matrix(&mut self, ring: &Arc<Ring>, rows: usize, cols: usize) -> PolyMat {
        let entries = (0..rows * cols).map(|_| self.poly(ring)).collect();
        PolyMat::new(ring.clone(), rows, cols, entries).unwrap()
    }
}

/// The reference sum: copying additions of `f_j.scale(sigma(r_j))`.
fn projection_by_copies(ring: &Arc<Ring>, dim: usize, affines: &[QuadEq], rows: &[Poly]) -> QuadEq {
    let mut projection = QuadEq::zero(ring.clone(), dim).unwrap();
    for (affine, row) in affines.iter().zip(rows) {
        projection = projection.add(&affine.scale(row).unwrap()).unwrap();
    }
    projection
}

/// Reference `projection_equations`, by copying additions.
fn projection_equations_by_copies(
    scheme: &Abdlop,
    forms: &Forms,
    seed: &[u8; 32],
    ze: &PolyVec,
    zd: &PolyVec,
) -> Result<Vec<QuadEq>, Error> {
    let dim = 2 * (scheme.bounded_len() + scheme.message_len());
    let ring = scheme.ring();
    let mut equations = forms.evals.clone();
    for (exact, affines, sign, offset, z) in [
        (true, &forms.exact, &forms.sign_e, forms.mask_e, ze),
        (false, &forms.approx, &forms.sign_d, forms.mask_d, zd),
    ] {
        if affines.is_empty() {
            if !z.is_empty() {
                return Err(Error::Dimension);
            }
            continue;
        }
        if z.len() != 256 / ring.degree() || z.ring() != ring {
            return Err(Error::Dimension);
        }
        let response = abdlop::flatten(z)?;
        for i in 0..256 {
            let r = row(seed, exact, i, affines.len() * ring.degree())?;
            let mut projection = QuadEq::zero(ring.clone(), dim)?;
            for (affine, coeffs) in affines.iter().zip(r.chunks_exact(ring.degree())) {
                projection = projection
                    .add(&affine.scale(&Poly::new(ring.clone(), coeffs.to_vec())?.auto())?)?;
            }
            let mask = variable(scheme, dim, offset + i / ring.degree())?
                .scale(&Poly::constant(ring.clone(), 1).rotate(-((i % ring.degree()) as i64)))?;
            let mut equation = sign.product_affine(&projection)?.add(&mask)?;
            equation.r0 = equation
                .r0
                .sub(&Poly::constant(ring.clone(), response[i]))?;
            equations.push(equation);
        }
    }
    Ok(equations)
}

fn rings() -> Vec<Arc<Ring>> {
    let wide = U256::ONE
        .shl_vartime(240)
        .wrapping_add(&U256::from_u16(325));
    vec![
        Ring::new(1099511627917, 64).unwrap(),
        Ring::new(1099511627917, 128).unwrap(),
        Ring::new(i128::MAX, 64).unwrap(),
        Ring::with_modulus(wide, 64).unwrap(),
    ]
}

#[test]
fn prepared_rows_equal_the_sums_of_scaled_forms() {
    let mut rng = Rng(11);
    let dim = 24;
    let keys: Vec<u16> = (0..dim as u16).collect();
    for ring in rings() {
        for _ in 0..40 {
            let count = 1 + rng.below(12) as usize;
            // Few keys, so that columns collect terms of several forms.
            let some: Vec<u16> = (0..4)
                .map(|_| keys[rng.below(dim as u64) as usize])
                .collect();
            let affines: Vec<QuadEq> = (0..count).map(|_| rng.affine(&ring, dim, &some)).collect();
            let rows: Vec<Poly> = (0..count)
                .map(|_| {
                    if rng.below(2) == 0 {
                        rng.row(&ring)
                    } else {
                        rng.poly(&ring)
                    }
                })
                .collect();
            let expected = projection_by_copies(&ring, dim, &affines, &rows);
            let prepared = Prepared::new(&ring, dim, &affines).unwrap().unwrap();
            assert_eq!(prepared.projection(&ring, dim, &rows).unwrap(), expected);
            assert_eq!(projection(&ring, dim, &affines, &rows).unwrap(), expected);
        }
    }
}

#[test]
fn columns_with_more_than_nadds_general_terms_are_reduced_in_time() {
    use crate::params::moduli::NADDS;
    for ring in rings() {
        let dim = 8;
        let half = ring.half;
        // The largest centred magnitudes: every product sums d products of floor(q/2)^2,
        // so that NADDS + 1 of them unreduced would exceed what the primes represent.
        let extreme = Poly::from_u256(ring.clone(), vec![half; ring.degree()]).unwrap();
        let count = 2 * NADDS + 3;
        let affines: Vec<QuadEq> = (0..count)
            .map(|j| {
                let mut form = QuadEq::zero(ring.clone(), dim).unwrap();
                let entries = vec![
                    (0, extreme.clone()),
                    (
                        2,
                        if j % 3 == 0 {
                            Poly::zero(ring.clone())
                        } else {
                            extreme.clone()
                        },
                    ),
                    (5, Poly::constant(ring.clone(), (j % 5) as i128)),
                ];
                form.r1 = SparsePolyVec::new(ring.clone(), dim, entries).unwrap();
                form.r0 = extreme.clone();
                form
            })
            .collect();
        let rows: Vec<Poly> = (0..count)
            .map(|j| {
                if j % 2 == 0 {
                    extreme.clone()
                } else {
                    extreme.neg()
                }
            })
            .collect();
        let expected = projection_by_copies(&ring, dim, &affines, &rows);
        let prepared = Prepared::new(&ring, dim, &affines).unwrap().unwrap();
        assert_eq!(prepared.projection(&ring, dim, &rows).unwrap(), expected);
    }
}

#[test]
fn forms_with_quadratic_entries_take_the_generic_path() {
    let mut rng = Rng(12);
    let ring = Ring::new(1099511627917, 64).unwrap();
    let dim = 10;
    let keys: Vec<u16> = (0..dim as u16).collect();
    for zero in [false, true] {
        let mut affines: Vec<QuadEq> = (0..5).map(|_| rng.affine(&ring, dim, &keys)).collect();
        // Explicit zero quadratic entries are kept as keys; nonzero ones are summed.
        let entry = if zero {
            Poly::zero(ring.clone())
        } else {
            rng.poly(&ring)
        };
        affines[3].r2 = SparsePolyMat::new(ring.clone(), dim, vec![(1, 4, entry)]).unwrap();
        let rows: Vec<Poly> = (0..5).map(|_| rng.row(&ring)).collect();
        assert!(Prepared::new(&ring, dim, &affines).unwrap().is_none());
        assert_eq!(
            projection(&ring, dim, &affines, &rows).unwrap(),
            projection_by_copies(&ring, dim, &affines, &rows)
        );
    }
    // A form of another dimension is refused, as the copying sum refuses it.
    let other = rng.affine(&ring, dim + 2, &keys);
    assert_eq!(
        Prepared::new(&ring, dim, &[other]).err(),
        Some(Error::Dimension)
    );
    assert!(Prepared::new(&ring, dim, &[]).unwrap().is_some());
}

/// A statement on `toy_d64` with random block maps and offsets.
pub(super) fn statement(rng: &mut Rng, scheme: &Abdlop) -> Statement {
    let p = &scheme.parameters;
    let ring = scheme.ring().clone();
    let mut block = |rows: usize| AffineBlock {
        rows,
        s: (rng.below(4) != 0).then(|| rng.matrix(&ring, rows, p.m1)),
        m: (rng.below(2) != 0).then(|| rng.matrix(&ring, rows, p.l)),
        offset: (rng.below(2) != 0).then(|| {
            PolyVec::new(ring.clone(), (0..rows).map(|_| rng.poly(&ring)).collect()).unwrap()
        }),
    };
    Statement {
        quadratic: vec![],
        evaluation: vec![],
        binary: Some(block(p.n_bin)),
        arp: Some(block(p.n_prime)),
        l2: p
            .l2_rows
            .iter()
            .zip(&p.l2_bounds_squared)
            .map(|(rows, bound)| L2Block {
                map: block(*rows),
                bound_squared: *bound,
            })
            .collect(),
    }
}

#[test]
fn projection_equations_equal_the_reference() {
    let mut rng = Rng(13);
    let scheme = Abdlop::new([7; 32], crate::params::toy_d64()).unwrap();
    let ring = scheme.ring().clone();
    let extended = scheme.extend_messages(9).unwrap();
    let response = |rng: &mut Rng| {
        let entries = (0..256 / ring.degree())
            .map(|_| {
                let c = (0..ring.degree())
                    .map(|_| rng.below(2001) as i128 - 1000)
                    .collect();
                Poly::new(ring.clone(), c).unwrap()
            })
            .collect();
        PolyVec::new(ring.clone(), entries).unwrap()
    };
    let empty = PolyVec::zero(ring.clone(), 0);
    for case in 0..8 {
        let statement = statement(&mut rng, &scheme);
        let mut forms = forms(&scheme, &extended, &statement).unwrap();
        let seed = [case as u8; 32];
        let (mut ze, mut zd) = (response(&mut rng), response(&mut rng));
        // Both blocks, the exact block alone, the approximate block alone, neither.
        match case % 4 {
            1 => {
                forms.approx.clear();
                zd = empty.clone();
            }
            2 => {
                forms.exact.clear();
                ze = empty.clone();
            }
            3 => {
                forms.exact.clear();
                forms.approx.clear();
                (ze, zd) = (empty.clone(), empty.clone());
            }
            _ => {}
        }
        assert_eq!(
            projection_equations(&extended, &forms, &seed, &ze, &zd).unwrap(),
            projection_equations_by_copies(&extended, &forms, &seed, &ze, &zd).unwrap(),
            "case {case}"
        );
        // Responses of the wrong shape: refused alike.
        let short = PolyVec::new(ring.clone(), response(&mut rng).entries()[..1].to_vec()).unwrap();
        for (ze, zd) in [
            (empty.clone(), zd.clone()),
            (ze.clone(), short.clone()),
            (short, empty.clone()),
        ] {
            assert_eq!(
                projection_equations(&extended, &forms, &seed, &ze, &zd),
                projection_equations_by_copies(&extended, &forms, &seed, &ze, &zd)
            );
        }
    }
}
