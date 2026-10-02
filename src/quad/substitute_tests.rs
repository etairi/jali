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
