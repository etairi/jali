//! `QuadEq::substitute`: with a single unit term per variable it is `remap`, zero entries and
//! collisions included; with affine forms its expansion evaluates, on every witness, to the
//! form evaluated at the substituted values (squares, products and constants on seeded random
//! forms).
use super::*;

const Q: i128 = 1099511627917;

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
    /// A polynomial that is zero, a constant, or dense, each a third of the time.
    fn poly(&mut self, ring: &Arc<Ring>) -> Poly {
        let d = ring.degree();
        let values = match self.below(3) {
            0 => vec![0; d],
            1 => {
                let mut c = vec![0; d];
                c[0] = self.below(Q as u64) as i128;
                c
            }
            _ => (0..d).map(|_| self.below(Q as u64) as i128).collect(),
        };
        Poly::new(ring.clone(), values).unwrap()
    }
    /// A random form in `n` variables, with up to six linear and six quadratic entries.
    fn form(&mut self, ring: &Arc<Ring>, n: usize) -> QuadEq {
        let mut form = QuadEq::zero(ring.clone(), n).unwrap();
        let mut linear = BTreeMap::new();
        for _ in 0..self.below(7) {
            linear.insert(self.below(n as u64) as u16, self.poly(ring));
        }
        let mut quadratic = BTreeMap::new();
        for _ in 0..self.below(7) {
            let (i, j) = (self.below(n as u64) as u16, self.below(n as u64) as u16);
            quadratic.insert((i.min(j), i.max(j)), self.poly(ring));
        }
        form.r1 = SparsePolyVec::new(ring.clone(), n, linear.into_iter().collect()).unwrap();
        form.r2 = SparsePolyMat::new(
            ring.clone(),
            n,
            quadratic.into_iter().map(|((i, j), p)| (i, j, p)).collect(),
        )
        .unwrap();
        form.r0 = self.poly(ring);
        form
    }
}

#[test]
fn unit_terms_give_remap() {
    let ring = Ring::new(Q, 64).unwrap();
    let mut rng = Rng(0x2567_2301);
    for _ in 0..300 {
        let n = 1 + rng.below(8) as usize;
        let dimension = n + rng.below(8) as usize;
        let form = rng.form(&ring, n);
        // Any map, collisions included.
        let indices: Vec<usize> = (0..n)
            .map(|_| rng.below(dimension as u64) as usize)
            .collect();
        let map: Vec<Affine> = indices
            .iter()
            .map(|i| Affine {
                terms: vec![(*i as u16, Poly::constant(ring.clone(), 1))],
                constant: Poly::zero(ring.clone()),
            })
            .collect();
        assert_eq!(
            form.substitute(&map, dimension).unwrap(),
            form.remap(&indices, dimension).unwrap()
        );
    }
}

#[test]
fn the_expansion_evaluates_as_the_substituted_form() {
    let ring = Ring::new(Q, 64).unwrap();
    let mut rng = Rng(0x2567_2302);
    for _ in 0..200 {
        let n = 1 + rng.below(6) as usize;
        let dimension = 1 + rng.below(10) as usize;
        let form = rng.form(&ring, n);
        let map: Vec<Affine> = (0..n)
            .map(|_| Affine {
                terms: (0..rng.below(4))
                    .map(|_| (rng.below(dimension as u64) as u16, rng.poly(&ring)))
                    .collect(),
                constant: rng.poly(&ring),
            })
            .collect();
        let y = PolyVec::new(
            ring.clone(),
            (0..dimension).map(|_| rng.poly(&ring)).collect(),
        )
        .unwrap();
        // x_i = sum_u c_u y_u + C_i.
        let x = PolyVec::new(
            ring.clone(),
            map.iter()
                .map(|a| {
                    a.terms.iter().fold(a.constant.clone(), |acc, (u, c)| {
                        acc.add(&c.mul(&y.entries()[usize::from(*u)]).unwrap())
                            .unwrap()
                    })
                })
                .collect(),
        )
        .unwrap();
        let substituted = form.substitute(&map, dimension).unwrap();
        assert_eq!(
            substituted.evaluate(&y).unwrap(),
            form.evaluate(&x).unwrap()
        );
    }
}

#[test]
fn out_of_range_terms_and_other_rings_are_refused() {
    let ring = Ring::new(Q, 64).unwrap();
    let form = QuadEq::zero(ring.clone(), 1).unwrap();
    let one = Poly::constant(ring.clone(), 1);
    let unit = |u: u16, c: Poly| Affine {
        terms: vec![(u, c)],
        constant: Poly::zero(ring.clone()),
    };
    assert_eq!(
        form.substitute(&[unit(2, one.clone())], 2).err(),
        Some(Error::Index)
    );
    let other = Ring::new(13, 64).unwrap();
    assert_eq!(
        form.substitute(&[unit(0, Poly::constant(other, 1))], 2)
            .err(),
        Some(Error::RingMismatch)
    );
    assert_eq!(
        form.substitute(&[unit(0, one.clone()), unit(1, one)], 2)
            .err(),
        Some(Error::Dimension)
    );
}

/// The expansion of 0.3.3, one `mul` and one `add` per product: the oracle of the expansion
/// that transforms the map's constants once and sums the products on the constant term.
fn substitute_reference(form: &QuadEq, map: &[Affine], dimension: usize) -> QuadEq {
    let ring = form.r0.ring().clone();
    let mut matrix = BTreeMap::new();
    let mut vector = BTreeMap::new();
    let mut constant = form.r0.clone();
    let add = |entry: &mut Poly, value: &Poly| *entry = entry.add(value).unwrap();
    let zero = || Poly::zero(ring.clone());
    for ((i, j), a) in form.r2.entries() {
        let (x, y) = (&map[usize::from(i)], &map[usize::from(j)]);
        for (u, cu) in &x.terms {
            let acu = a.mul(cu).unwrap();
            for (v, cv) in &y.terms {
                let key = ((*u).min(*v), (*u).max(*v));
                add(
                    matrix.entry(key).or_insert_with(zero),
                    &acu.mul(cv).unwrap(),
                );
            }
        }
        if !y.constant.is_zero() {
            let ay = a.mul(&y.constant).unwrap();
            for (u, cu) in &x.terms {
                add(vector.entry(*u).or_insert_with(zero), &ay.mul(cu).unwrap());
            }
            if !x.constant.is_zero() {
                add(&mut constant, &ay.mul(&x.constant).unwrap());
            }
        }
        if !x.constant.is_zero() {
            let ax = a.mul(&x.constant).unwrap();
            for (v, cv) in &y.terms {
                add(vector.entry(*v).or_insert_with(zero), &ax.mul(cv).unwrap());
            }
        }
    }
    for (i, a) in form.r1.entries() {
        let x = &map[usize::from(i)];
        for (u, cu) in &x.terms {
            add(vector.entry(*u).or_insert_with(zero), &a.mul(cu).unwrap());
        }
        if !x.constant.is_zero() {
            add(&mut constant, &a.mul(&x.constant).unwrap());
        }
    }
    QuadEq {
        r2: SparsePolyMat::new(
            ring.clone(),
            dimension,
            matrix.into_iter().map(|((i, j), p)| (i, j, p)).collect(),
        )
        .unwrap(),
        r1: SparsePolyVec::new(ring, dimension, vector.into_iter().collect()).unwrap(),
        r0: constant,
    }
}

/// A polynomial of each shape the expansion tells apart: zero, a scalar, a monomial or dense,
/// with coefficients below `q` from four random words.
fn shaped(rng: &mut Rng, ring: &Arc<Ring>, general_share: u64) -> Poly {
    let d = ring.degree();
    let q = ring.modulus();
    let value = |rng: &mut Rng| {
        let words = [rng.next(), rng.next(), rng.next(), rng.next()];
        U256::from_le_slice(&words.map(u64::to_le_bytes).concat())
            .rem_vartime(&crypto_bigint::NonZero::new(q).unwrap())
    };
    let mut values = vec![U256::ZERO; d];
    match rng.below(4 + general_share) {
        0 => {}
        1 => values[0] = value(rng),
        2 => values[1 + rng.below(d as u64 - 1) as usize] = value(rng),
        _ => values.iter_mut().for_each(|x| *x = value(rng)),
    }
    Poly::from_canonical(ring.clone(), values)
}

#[test]
fn transformed_constants_give_the_reference_expansion() {
    // One-word, two-word and 256-bit coefficients, both degrees, forms and maps of every shape
    // (cheap and general terms, constants and coefficients), and keys that collide.
    let moduli = [
        U256::from_u64(1099511627917),
        U256::from_u128(800602353498606932557),
        U256::ONE.shl_vartime(127).wrapping_add(&U256::from_u8(45)),
    ];
    let mut rng = Rng(0x2567_2303);
    for q in moduli {
        for d in [64, 128] {
            let ring = Ring::with_modulus(q, d).unwrap();
            for _ in 0..12 {
                let n = 1 + rng.below(8) as usize;
                let dimension = 1 + rng.below(10) as usize;
                let mut form = QuadEq::zero(ring.clone(), n).unwrap();
                let mut linear = BTreeMap::new();
                for _ in 0..rng.below(7) {
                    linear.insert(rng.below(n as u64) as u16, shaped(&mut rng, &ring, 4));
                }
                let mut quadratic = BTreeMap::new();
                for _ in 0..rng.below(7) {
                    let (i, j) = (rng.below(n as u64) as u16, rng.below(n as u64) as u16);
                    quadratic.insert((i.min(j), i.max(j)), shaped(&mut rng, &ring, 4));
                }
                form.r1 =
                    SparsePolyVec::new(ring.clone(), n, linear.into_iter().collect()).unwrap();
                form.r2 = SparsePolyMat::new(
                    ring.clone(),
                    n,
                    quadratic.into_iter().map(|((i, j), p)| (i, j, p)).collect(),
                )
                .unwrap();
                form.r0 = shaped(&mut rng, &ring, 1);
                let map: Vec<Affine> = (0..n)
                    .map(|_| Affine {
                        terms: (0..rng.below(4))
                            .map(|_| {
                                (
                                    rng.below(dimension as u64) as u16,
                                    shaped(&mut rng, &ring, 1),
                                )
                            })
                            .collect(),
                        constant: shaped(&mut rng, &ring, 4),
                    })
                    .collect();
                assert_eq!(
                    form.substitute(&map, dimension).unwrap(),
                    substitute_reference(&form, &map, dimension),
                    "q {q} d {d}"
                );
            }
        }
    }
}

#[test]
fn more_constant_products_than_one_sum_holds() {
    // 200 linear and 100 quadratic entries whose general products all land on the constant term:
    // past NADDS = 128 the sum spills into the constant and starts again.
    let ring = Ring::with_modulus(U256::from_u128(800602353498606932557), 64).unwrap();
    let mut rng = Rng(0x2567_2304);
    let n = 200;
    let dense = |rng: &mut Rng| loop {
        let p = shaped(rng, &ring, 1000);
        if crate::math::terms::Shape::of(&p) == crate::math::terms::Shape::General {
            break p;
        }
    };
    let mut form = QuadEq::zero(ring.clone(), n).unwrap();
    form.r1 = SparsePolyVec::new(
        ring.clone(),
        n,
        (0..n as u16).map(|i| (i, dense(&mut rng))).collect(),
    )
    .unwrap();
    form.r2 = SparsePolyMat::new(
        ring.clone(),
        n,
        (0..100u16).map(|i| (i, i + 100, dense(&mut rng))).collect(),
    )
    .unwrap();
    form.r0 = dense(&mut rng);
    let map: Vec<Affine> = (0..n)
        .map(|_| Affine {
            terms: vec![],
            constant: dense(&mut rng),
        })
        .collect();
    let expanded = form.substitute(&map, 1).unwrap();
    assert_eq!(expanded, substitute_reference(&form, &map, 1));
    assert!(expanded.r1.entries().next().is_none() && expanded.r2.entries().next().is_none());
}
