//! In-place accumulation against the copying operations it fuses:
//! `add_assign` ≡ `add`, `add_scaled_scalar_assign(w)` ≡ `add(scale(constant w))`,
//! `add_scaled_assign(p)` ≡ `add(scale(p))`, `add_product_affine(a, b)` ≡
//! `add(a.product_affine(b))`, `into_trace` ≡ a trace by copies and `resize` ≡ `resized`,
//! comparing keys and values, on random sparse forms with overlapping and disjoint keys, zero
//! entries and every coefficient shape, at moduli on both sides of $`2^{63}`$ and $`2^{128}`$;
//! refusals included, and the polynomial helpers against the operations they fuse.
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
    /// A canonical value: 0, 1, q-1, q-2, floor(q/2), floor(q/2)+1 or uniform.
    fn value(&mut self, ring: &Arc<Ring>) -> U256 {
        let q = ring.modulus();
        match self.below(8) {
            0 => U256::ZERO,
            1 => U256::ONE,
            2 => q.wrapping_sub(&U256::ONE),
            3 => q.wrapping_sub(&U256::from_u8(2)),
            4 => ring.half,
            5 => ring.half.wrapping_add(&U256::ONE),
            _ => ring.reduce(&self.uniform()),
        }
    }
    /// Zero, a scalar, a monomial (also at k = 1 and d - 1), a binomial or dense.
    fn poly(&mut self, ring: &Arc<Ring>) -> Poly {
        let d = ring.degree();
        let mut coeffs = vec![U256::ZERO; d];
        match self.below(6) {
            0 => {}
            1 => coeffs[0] = self.value(ring),
            2 => {
                let k = [1, d - 1, 1 + self.below(d as u64 - 1) as usize][self.below(3) as usize];
                coeffs[k] = self.value(ring);
            }
            3 => {
                coeffs[0] = self.value(ring);
                coeffs[self.below(d as u64) as usize] = self.value(ring);
            }
            _ => coeffs.iter_mut().for_each(|x| *x = self.value(ring)),
        }
        Poly::from_u256(ring.clone(), coeffs).unwrap()
    }
    /// A form in `n` variables with up to `entries` linear and, if `quadratic`, quadratic
    /// entries; few variables, so keys of two forms overlap often.
    fn form(&mut self, ring: &Arc<Ring>, n: usize, entries: u64, quadratic: bool) -> QuadEq {
        let mut linear = BTreeMap::new();
        for _ in 0..self.below(entries + 1) {
            linear.insert(self.below(n as u64) as u16, self.poly(ring));
        }
        let mut matrix = BTreeMap::new();
        if quadratic {
            for _ in 0..self.below(entries + 1) {
                let (i, j) = (self.below(n as u64) as u16, self.below(n as u64) as u16);
                matrix.insert((i.min(j), i.max(j)), self.poly(ring));
            }
        }
        QuadEq {
            r2: SparsePolyMat::new(
                ring.clone(),
                n,
                matrix.into_iter().map(|((i, j), p)| (i, j, p)).collect(),
            )
            .unwrap(),
            r1: SparsePolyVec::new(ring.clone(), n, linear.into_iter().collect()).unwrap(),
            r0: if self.below(3) == 0 {
                Poly::zero(ring.clone())
            } else {
                self.poly(ring)
            },
        }
    }
    /// An affine form, sometimes with explicit zero quadratic entries (accepted by
    /// `product_affine`, and kept as keys).
    fn affine(&mut self, ring: &Arc<Ring>, n: usize) -> QuadEq {
        let mut form = self.form(ring, n, 6, false);
        if self.below(4) == 0 {
            let (i, j) = (self.below(n as u64) as u16, self.below(n as u64) as u16);
            form.r2 = SparsePolyMat::new(
                ring.clone(),
                n,
                vec![(i.min(j), i.max(j), Poly::zero(ring.clone()))],
            )
            .unwrap();
        }
        form
    }
}

/// Moduli: odd and even, below 2^63 (Shoup scalar products), between 2^63 and 2^128, and wide.
fn rings() -> Vec<Arc<Ring>> {
    let mut rings: Vec<Arc<Ring>> = [3i128, 12, 1099511627917, (1 << 62) - 57, i128::MAX]
        .into_iter()
        .map(|q| Ring::new(q, 64).unwrap())
        .collect();
    rings.push(Ring::new(1099511627917, 128).unwrap());
    let wide = U256::ONE
        .shl_vartime(240)
        .wrapping_add(&U256::from_u16(325));
    rings.push(Ring::with_modulus(wide, 64).unwrap());
    rings
}

/// The reference trace, by copying operations.
fn trace_by_copies(form: &QuadEq) -> Result<QuadEq, Error> {
    let ring = form.r0.ring();
    if !bool::from(ring.modulus().is_odd()) {
        return Err(Error::Parameter("trace requires invertible two"));
    }
    let half = ring.half.wrapping_add(&U256::ONE);
    form.add(&form.conjugate()?)?
        .scale(&Poly::constant_u256(ring.clone(), &half))
}

#[test]
fn polynomial_helpers_equal_the_operations_they_fuse() {
    let mut rng = Rng(1);
    for ring in rings() {
        for _ in 0..200 {
            let (a, b) = (rng.poly(&ring), rng.poly(&ring));
            let c = rng.value(&ring);
            let mut x = a.clone();
            x.add_assign(&b).unwrap();
            assert_eq!(x, a.add(&b).unwrap());
            let mut x = a.clone();
            x.sub_assign(&b).unwrap();
            assert_eq!(x, a.sub(&b).unwrap());
            let mut x = a.clone();
            x.add_scaled_assign(&b, &c).unwrap();
            assert_eq!(x, a.add(&b.scale_u256(&c).unwrap()).unwrap());
            let k = rng.below(ring.degree() as u64) as usize;
            let mut x = a.clone();
            x.add_monomial_product_assign(&b, k, &c).unwrap();
            let mut monomial = Poly::zero(ring.clone());
            monomial.coeffs[k] = c;
            assert_eq!(x, a.add(&monomial.mul(&b).unwrap()).unwrap());
            assert_eq!(
                x,
                a.add(&b.rotate(k as i64).scale_u256(&c).unwrap()).unwrap()
            );
            let mut x = a.clone();
            x.scale_assign(&c);
            assert_eq!(x, a.scale_u256(&c).unwrap());
            assert_eq!(
                a.is_scalar(),
                a.coefficients()[1..].iter().all(|x| *x == U256::ZERO)
            );
        }
    }
    let (r1, r2) = (Ring::new(13, 64).unwrap(), Ring::new(17, 64).unwrap());
    let mut x = Poly::zero(r1.clone());
    let y = Poly::constant(r2, 1);
    assert_eq!(x.add_assign(&y), Err(Error::RingMismatch));
    assert_eq!(
        x.add_scaled_assign(&y, &U256::ONE),
        Err(Error::RingMismatch)
    );
    assert_eq!(
        x.add_monomial_product_assign(&y, 1, &U256::ONE),
        Err(Error::RingMismatch)
    );
    assert_eq!(x, Poly::zero(r1));
}

#[test]
fn in_place_sums_equal_the_copying_ones() {
    let mut rng = Rng(2);
    let n = 10;
    for ring in rings() {
        for _ in 0..150 {
            let a = rng.form(&ring, n, 6, true);
            let b = rng.form(&ring, n, 6, true);
            let mut x = a.clone();
            x.add_assign(&b).unwrap();
            assert_eq!(x, a.add(&b).unwrap());

            let w = rng.value(&ring);
            let mut x = a.clone();
            x.add_scaled_scalar_assign(&b, &w).unwrap();
            assert_eq!(
                x,
                a.add(&b.scale(&Poly::constant_u256(ring.clone(), &w)).unwrap())
                    .unwrap()
            );
            // An unreduced scalar is reduced, as `constant_u256` reduces it.
            let unreduced = w.wrapping_add(&ring.modulus());
            if unreduced > w {
                let mut y = a.clone();
                y.add_scaled_scalar_assign(&b, &unreduced).unwrap();
                assert_eq!(y, x);
            }

            let weight = rng.poly(&ring);
            let mut x = a.clone();
            x.add_scaled_assign(&b, &weight).unwrap();
            assert_eq!(x, a.add(&b.scale(&weight).unwrap()).unwrap());
        }
    }
}

#[test]
fn in_place_products_equal_product_affine() {
    let mut rng = Rng(3);
    let n = 10;
    for ring in rings() {
        for _ in 0..150 {
            let acc = rng.form(&ring, n, 6, true);
            let (a, b) = (rng.affine(&ring, n), rng.affine(&ring, n));
            let mut x = acc.clone();
            x.add_product_affine(&a, &b).unwrap();
            assert_eq!(x, acc.add(&a.product_affine(&b).unwrap()).unwrap());
            // A square, as the norm forms use it.
            let conjugate = a.conjugate().unwrap();
            let mut x = acc.clone();
            x.add_product_affine(&conjugate, &a).unwrap();
            assert_eq!(x, acc.add(&conjugate.product_affine(&a).unwrap()).unwrap());
        }
    }
}

#[test]
fn in_place_products_refuse_what_product_affine_refuses() {
    let mut rng = Rng(4);
    let ring = Ring::new(1099511627917, 64).unwrap();
    let n = 8;
    for _ in 0..100 {
        let acc = rng.form(&ring, n, 4, true);
        let a = rng.affine(&ring, n);
        let mut quadratic = rng.form(&ring, n, 4, true);
        if quadratic.r2.entries().all(|(_, p)| p.is_zero()) {
            quadratic.r2 = SparsePolyMat::new(
                ring.clone(),
                n,
                vec![(0, 1, Poly::constant(ring.clone(), 1))],
            )
            .unwrap();
        }
        for (x, y) in [(&a, &quadratic), (&quadratic, &a)] {
            let expected = x.product_affine(y).unwrap_err();
            assert_eq!(expected, Error::Parameter("expression degree exceeds two"));
            let mut z = acc.clone();
            assert_eq!(z.add_product_affine(x, y), Err(expected));
            assert_eq!(z, acc, "a refused product leaves the sum unchanged");
        }
    }
    // Dimension and ring mismatches, refused before anything changes.
    let other_dim = rng.affine(&ring, n + 2);
    let other_ring = QuadEq::zero(Ring::new(13, 64).unwrap(), n).unwrap();
    let acc = rng.form(&ring, n, 4, true);
    let a = rng.affine(&ring, n);
    for (x, y, err) in [
        (&a, &other_dim, Error::Dimension),
        (&other_dim, &a, Error::Dimension),
        (&a, &other_ring, Error::RingMismatch),
    ] {
        let mut z = acc.clone();
        assert_eq!(z.add_product_affine(x, y), Err(err));
        assert_eq!(z, acc);
    }
    let mut z = acc.clone();
    assert_eq!(
        z.add_product_affine(&other_dim, &other_dim),
        Err(Error::Dimension)
    );
    assert_eq!(z.add_assign(&other_dim), Err(Error::Dimension));
    assert_eq!(z.add_assign(&other_ring), Err(Error::RingMismatch));
    assert_eq!(
        z.add_scaled_scalar_assign(&other_dim, &U256::ONE),
        Err(Error::Dimension)
    );
    assert_eq!(
        z.add_scaled_assign(&other_ring, &Poly::constant(ring.clone(), 1)),
        Err(Error::RingMismatch)
    );
    assert_eq!(
        z.add_scaled_assign(&a, &Poly::constant(Ring::new(13, 64).unwrap(), 1)),
        Err(Error::RingMismatch)
    );
    assert_eq!(z, acc);
}

#[test]
fn trace_and_resize_equal_the_copying_ones() {
    let mut rng = Rng(5);
    for ring in rings() {
        for _ in 0..100 {
            let form = rng.form(&ring, 10, 6, true);
            assert_eq!(form.trace(), trace_by_copies(&form));
            assert_eq!(form.clone().into_trace(), trace_by_copies(&form));
            let odd = rng.form(&ring, 9, 4, true);
            assert_eq!(odd.trace(), trace_by_copies(&odd));
            let mut x = form.clone();
            x.resize(14).unwrap();
            assert_eq!(x, form.resized(14).unwrap());
            assert_eq!(x.clone().resize(13), Err(Error::Dimension));
            assert_eq!(x.clone().resize(65537), Err(Error::Dimension));
        }
    }
}
