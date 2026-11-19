//! In-place accumulation against the copying operations it fuses:
//! `add_assign` ≡ `add`, `add_scaled_scalar_assign(w)` ≡ `add(scale(constant w))`,
//! `add_scaled_assign(p)` ≡ `add(scale(p))`, `add_product_affine(a, b)` ≡
//! `add(a.product_affine(b))`, `combination` ≡ a fold of `add(scale(w))`, `insert_outer_keys`
//! ≡ `insert_key` on every pair, `into_trace` ≡ a trace by copies and `resize` ≡ `resized`,
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
            let mut values = vec![U256::ZERO; ring.degree()];
            values[k] = c;
            let monomial = Poly::from_canonical(ring.clone(), values);
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

/// A form with `entries` linear keys among `n`, most of them explicit zeros, as the forms of
/// large compiled statements have them.
fn mostly_zero(rng: &mut Rng, ring: &Arc<Ring>, n: usize, entries: usize) -> QuadEq {
    let mut linear = BTreeMap::new();
    while linear.len() < entries {
        let p = if rng.below(8) == 0 {
            rng.poly(ring)
        } else {
            Poly::zero(ring.clone())
        };
        linear.insert(rng.below(n as u64) as u16, p);
    }
    let mut form = QuadEq::zero(ring.clone(), n).unwrap();
    form.r1 = SparsePolyVec::new(ring.clone(), n, linear.into_iter().collect()).unwrap();
    if rng.below(2) == 0 {
        form.r0 = rng.poly(ring);
    }
    form
}

#[test]
fn wide_products_equal_product_affine() {
    // Forms of up to 60 keys, mostly explicit zeros, into accumulators that hold none, some or
    // all of the product's keys: both ways of inserting the keys.
    let mut rng = Rng(11);
    let n = 128;
    for ring in rings().into_iter().take(4) {
        for case in 0..24 {
            let (sa, sb) = (1 + rng.below(60) as usize, 1 + rng.below(60) as usize);
            let a = mostly_zero(&mut rng, &ring, n, sa);
            let b = if case % 3 == 0 {
                a.clone()
            } else {
                mostly_zero(&mut rng, &ring, n, sb)
            };
            let product = a.product_affine(&b).unwrap();
            let accumulators = [
                QuadEq::zero(ring.clone(), n).unwrap(),
                rng.form(&ring, n, 40, true),
                product.clone(),
            ];
            for acc in accumulators {
                let mut x = acc.clone();
                x.add_product_affine(&a, &b).unwrap();
                assert_eq!(x, acc.add(&product).unwrap(), "case {case}");
            }
        }
    }
}

#[test]
fn outer_keys_equal_a_key_per_pair() {
    let mut rng = Rng(12);
    let ring = Ring::new(1099511627917, 64).unwrap();
    let n = 300u64;
    let sorted = |rng: &mut Rng, len: u64| -> Vec<u16> {
        let set: std::collections::BTreeSet<u16> = (0..len).map(|_| rng.below(n) as u16).collect();
        set.into_iter().collect()
    };
    for case in 0..200 {
        let len = rng.below(70);
        let a = sorted(&mut rng, len);
        let b = match case % 4 {
            0 => a.clone(),
            1 => Vec::new(),
            _ => {
                let len = rng.below(70);
                sorted(&mut rng, len)
            }
        };
        // A map with values and explicit zeros, with few or many keys against the pairs.
        let mut present = Vec::new();
        for _ in 0..[0, 3, 40, 4000][case % 4] {
            let (i, j) = (rng.below(n) as u16, rng.below(n) as u16);
            present.push((i.min(j), i.max(j)));
        }
        present.sort_unstable();
        present.dedup();
        let mut entries = Vec::new();
        for (k, &(r, c)) in present.iter().enumerate() {
            let p = if k % 2 == 0 {
                rng.poly(&ring)
            } else {
                Poly::zero(ring.clone())
            };
            entries.push((r, c, p));
        }
        let map = SparsePolyMat::new(ring.clone(), n as usize, entries).unwrap();
        let mut expected = map.clone();
        for &i in &a {
            for &j in &b {
                expected.insert_key(i.min(j), i.max(j)).unwrap();
            }
        }
        let mut x = map.clone();
        x.insert_outer_keys(&a, &b).unwrap();
        assert_eq!(x, expected, "case {case}");
        let mut y = map.clone();
        y.insert_outer_keys(&b, &a).unwrap();
        assert_eq!(y, expected, "case {case}");
    }
    // Two absent keys against a large map: inserted one at a time.
    let entries = (0..200u16).map(|c| (0, c, rng.poly(&ring))).collect();
    let map = SparsePolyMat::new(ring.clone(), n as usize, entries).unwrap();
    let mut x = map.clone();
    x.insert_outer_keys(&[1], &[1, 250]).unwrap();
    let mut expected = map.clone();
    expected.insert_key(1, 1).unwrap();
    expected.insert_key(1, 250).unwrap();
    assert_eq!(x, expected);
    assert_eq!(x.key_count(), 202);
    // Unsorted, repeated and out-of-range keys are refused.
    let mut map = SparsePolyMat::new(ring.clone(), 10, Vec::new()).unwrap();
    assert!(map.insert_outer_keys(&[3, 2], &[1]).is_err());
    assert!(map.insert_outer_keys(&[1], &[2, 2]).is_err());
    assert_eq!(map.insert_outer_keys(&[1], &[10]), Err(Error::Index));
    assert_eq!(map.key_count(), 0);
}

/// The fold by copying operations: `add(scale(w))` of each equation in turn.
fn combination_by_copies(
    ring: &Arc<Ring>,
    n: usize,
    equations: &[QuadEq],
    weights: &[Poly],
) -> QuadEq {
    let mut out = QuadEq::zero(ring.clone(), n).unwrap();
    for (e, w) in equations.iter().zip(weights) {
        out = out.add(&e.scale(w).unwrap()).unwrap();
    }
    out
}

#[test]
fn combinations_equal_the_folds_they_replace() {
    let mut rng = Rng(13);
    let n = 10;
    for ring in rings() {
        for case in 0..40 {
            let count = rng.below(9) as usize;
            let equations: Vec<QuadEq> = (0..count).map(|_| rng.form(&ring, n, 8, true)).collect();
            let weights: Vec<Poly> = (0..count).map(|_| rng.poly(&ring)).collect();
            let x = QuadEq::combination(&ring, n, &equations, &weights).unwrap();
            assert_eq!(
                x,
                combination_by_copies(&ring, n, &equations, &weights),
                "case {case}"
            );
            let mut y = QuadEq::zero(ring.clone(), n).unwrap();
            for (e, w) in equations.iter().zip(&weights) {
                y.add_scaled_assign(e, w).unwrap();
            }
            assert_eq!(x, y, "case {case}");
        }
    }
    // More general products on one key than an accumulator holds before it reduces.
    let mut rng = Rng(14);
    for ring in rings().into_iter().take(4) {
        let count = crate::params::moduli::NADDS + 12;
        let dense = |rng: &mut Rng| loop {
            let p = rng.poly(&ring);
            if Shape::of(&p) == Shape::General {
                break p;
            }
        };
        let mut equations = Vec::new();
        for _ in 0..count {
            let mut e = QuadEq::zero(ring.clone(), 4).unwrap();
            e.r2 = SparsePolyMat::new(ring.clone(), 4, vec![(1, 2, dense(&mut rng))]).unwrap();
            e.r1 = SparsePolyVec::new(ring.clone(), 4, vec![(3, dense(&mut rng))]).unwrap();
            e.r0 = dense(&mut rng);
            equations.push(e);
        }
        let weights: Vec<Poly> = (0..count).map(|_| dense(&mut rng)).collect();
        let x = QuadEq::combination(&ring, 4, &equations, &weights).unwrap();
        let mut y = QuadEq::zero(ring.clone(), 4).unwrap();
        for (e, w) in equations.iter().zip(&weights) {
            y.add_scaled_assign(e, w).unwrap();
        }
        assert_eq!(x, y);
    }
    // Refusals: a weight per equation, the ring and the dimension of every equation.
    let ring = Ring::new(1099511627917, 64).unwrap();
    let other = Ring::new(3, 64).unwrap();
    let e = QuadEq::zero(ring.clone(), 4).unwrap();
    let w = Poly::constant(ring.clone(), 2);
    assert_eq!(
        QuadEq::combination(&ring, 4, std::slice::from_ref(&e), &[]).err(),
        Some(Error::Dimension)
    );
    assert!(QuadEq::combination(&ring, 5, std::slice::from_ref(&e), &[w]).is_err());
    assert_eq!(
        QuadEq::combination(&ring, 4, &[e], &[Poly::constant(other, 2)]).err(),
        Some(Error::RingMismatch)
    );
}

#[test]
fn sorted_constructors_build_what_new_builds() {
    let ring = Ring::new(1099511627917, 64).unwrap();
    let p = Poly::constant(ring.clone(), 5);
    let zero = || Poly::zero(ring.clone());
    let m = SparsePolyMat::from_sorted(
        ring.clone(),
        4,
        vec![
            ((0, 1), Some(p.clone())),
            ((0, 3), None),
            ((2, 2), Some(zero())),
        ],
    )
    .unwrap();
    let reference = SparsePolyMat::new(
        ring.clone(),
        4,
        vec![(0, 1, p.clone()), (0, 3, zero()), (2, 2, zero())],
    )
    .unwrap();
    assert_eq!(m, reference);
    for bad in [
        vec![((0, 1), None), ((0, 1), None)],
        vec![((0, 2), None), ((0, 1), None)],
        vec![((1, 0), None)],
        vec![((0, 4), None)],
    ] {
        assert!(SparsePolyMat::from_sorted(ring.clone(), 4, bad).is_err());
    }
    let v =
        SparsePolyVec::from_sorted(ring.clone(), 4, vec![(0, None), (3, Some(p.clone()))]).unwrap();
    let reference = SparsePolyVec::new(ring.clone(), 4, vec![(0, zero()), (3, p.clone())]).unwrap();
    assert_eq!(v, reference);
    for bad in [
        vec![(1, None), (1, None)],
        vec![(2, None), (1, None)],
        vec![(4, None)],
    ] {
        assert!(SparsePolyVec::from_sorted(ring.clone(), 4, bad).is_err());
    }
    let other = Ring::new(3, 64).unwrap();
    let foreign = vec![(0, Some(Poly::constant(other, 1)))];
    assert_eq!(
        SparsePolyVec::from_sorted(ring.clone(), 4, foreign).err(),
        Some(Error::RingMismatch)
    );
}
