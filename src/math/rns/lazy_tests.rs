//! [`LazySum`] against the product-by-product sums of [`RnsProduct::mac`], residue for
//! residue, and [`NttPlan::reduce_wide`](crate::math::ntt::NttPlan::reduce_wide) against `%`.
use super::*;
use crate::math::Ring;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

/// Rings with two primes (q below 2^63), three (q near 2^70) and nine (q near 2^240).
fn rings() -> Vec<Arc<Ring>> {
    let wide = U256::ONE
        .shl_vartime(240)
        .wrapping_add(&U256::from_u16(325));
    vec![
        Ring::new(1099511627917, 64).unwrap(),
        Ring::new((1 << 70) - 35, 64).unwrap(),
        Ring::with_modulus(wide, 64).unwrap(),
    ]
}

/// A transform with uniform residues below each prime.
fn uniform(ring: &Arc<Ring>, rng: &mut Rng) -> PolyNtt {
    let residues = ring
        .plans
        .iter()
        .map(|plan| (0..ring.d).map(|_| rng.next() % plan.modulus()).collect())
        .collect();
    PolyNtt {
        ring: ring.clone(),
        residues: Residues(residues),
    }
}

/// The transform whose every residue is p - 1, the largest product.
fn largest(ring: &Arc<Ring>) -> PolyNtt {
    let residues = ring
        .plans
        .iter()
        .map(|plan| vec![plan.modulus() - 1; ring.d])
        .collect();
    PolyNtt {
        ring: ring.clone(),
        residues: Residues(residues),
    }
}

#[test]
fn lazy_sums_equal_the_sums_of_montgomery_products() {
    let mut rng = Rng(21);
    for ring in rings() {
        for count in [0, 1, 2, 14, 15, 16, 17, 30, 31, 45, 46, 100, NADDS] {
            for extreme in [false, true] {
                let mut lazy = LazySum::new(&ring);
                let mut reference = RnsProduct::zero(&ring);
                for _ in 0..count {
                    let (a, b) = if extreme {
                        (largest(&ring), largest(&ring))
                    } else {
                        (uniform(&ring, &mut rng), uniform(&ring, &mut rng))
                    };
                    lazy.mac(&a, &b).unwrap();
                    reference.mac(&a, &b).unwrap();
                }
                assert_eq!(lazy.count(), reference.count());
                assert_eq!(lazy.residues().0, reference.residues.0, "count {count}");
                assert_eq!(lazy.reduce().unwrap(), reference.reduce().unwrap());
            }
        }
        // A product past NADDS is refused and leaves the sum as it was.
        let mut lazy = LazySum::new(&ring);
        let a = uniform(&ring, &mut rng);
        for _ in 0..NADDS {
            lazy.mac(&a, &a).unwrap();
        }
        let before = lazy.residues().0.clone();
        assert_eq!(lazy.mac(&a, &a), Err(Error::Overflow));
        assert_eq!(lazy.residues().0, before);
    }
    let (r1, r2) = (&rings()[0], &rings()[1]);
    let mut lazy = LazySum::new(r1);
    let mut rng = Rng(22);
    let foreign = uniform(r2, &mut rng);
    let own = uniform(r1, &mut rng);
    assert_eq!(lazy.mac(&own, &foreign), Err(Error::RingMismatch));
    assert_eq!(lazy.count(), 0);
}

#[test]
fn wide_reductions_equal_remainders() {
    let mut rng = Rng(23);
    let ring = &rings()[2];
    for plan in &ring.plans {
        let p = u128::from(plan.modulus());
        let mut values = vec![
            0,
            1,
            p - 1,
            p,
            p + 1,
            2 * p,
            3 * p - 1,
            u128::MAX,
            u128::MAX - p,
        ];
        values.extend([(p - 1) * (p - 1), (p - 1) * (p - 1) * 15 + p - 1]);
        values.extend((0..2000).map(|_| (u128::from(rng.next()) << 64) | u128::from(rng.next())));
        values.extend((0..2000).map(|_| u128::from(rng.next()) * u128::from(rng.next() >> 2)));
        for x in values {
            assert_eq!(u128::from(plan.reduce_wide(x)), x % p, "x {x} p {p}");
        }
    }
}
