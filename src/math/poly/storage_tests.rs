//! The polynomial operations in each coefficient width, one word below $`2^{63}`$, two words up
//! to $`2^{126}`$ and 256-bit values above, against big-integer arithmetic, at moduli on both
//! sides of each switch. Also: the 256-bit copy of `coefficients` follows every change.
use super::*;
use num_bigint::{BigInt, BigUint};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, q: &BigUint) -> BigUint {
        let words = [self.next(), self.next(), self.next(), self.next()];
        let x = BigUint::from_bytes_le(
            &words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        x % q
    }
}
fn big(x: &U256) -> BigUint {
    BigUint::from_bytes_le(&x.to_le_bytes())
}
fn u256(x: &BigUint) -> U256 {
    let mut bytes = x.to_bytes_le();
    bytes.resize(32, 0);
    U256::from_le_slice(&bytes)
}
fn two(k: u32) -> BigUint {
    BigUint::from(1u32) << k
}

/// Moduli around the width switches: one word up to $`2^{63}-25`$, two words from $`2^{63}`$
/// to $`2^{126}-1`$, 256-bit values from $`2^{126}`$ on.
fn moduli() -> Vec<BigUint> {
    vec![
        BigUint::from(3u32),
        BigUint::from(3329u32),
        BigUint::from(1099511627917u64),
        two(63) - 25u32,
        two(63),
        two(63) + 1u32,
        two(64) - 59u32,
        two(64) + 13u32,
        BigUint::from(800602353498606932557u128),
        two(126) - 1u32,
        two(126),
        two(127) + 45u32,
        two(200) - 75u32,
    ]
}

/// The polynomial of canonical big-integer coefficients.
fn poly(ring: &Arc<Ring>, values: &[BigUint]) -> Poly {
    Poly::from_canonical(ring.clone(), values.iter().map(u256).collect())
}
fn coefficients(p: &Poly) -> Vec<BigUint> {
    p.values().map(|x| big(&x)).collect()
}
/// The centred representative in $`(-q/2,q/2]`$.
fn centred(x: &BigUint, q: &BigUint) -> BigInt {
    if x > &(q >> 1u32) {
        BigInt::from(x.clone()) - BigInt::from(q.clone())
    } else {
        BigInt::from(x.clone())
    }
}
fn modulo(x: &BigInt, q: &BigUint) -> BigUint {
    let q = BigInt::from(q.clone());
    (((x % &q) + &q) % &q).to_biguint().unwrap()
}

#[test]
fn every_width_equals_big_integer_arithmetic() {
    let mut rng = Rng(31);
    for q in moduli() {
        let ring = Ring::with_modulus(u256(&q), 64).unwrap();
        let d = 64;
        let half = &q >> 1u32;
        let edges = [
            BigUint::from(0u32),
            BigUint::from(1u32),
            half.clone(),
            &half + 1u32,
            &q - 1u32,
        ];
        let values = |rng: &mut Rng| -> Vec<BigUint> {
            let mut v: Vec<BigUint> = edges.to_vec();
            v.extend((v.len()..d).map(|_| rng.below(&q)));
            v.truncate(d);
            v
        };
        let (av, bv) = (values(&mut rng), values(&mut rng));
        let (a, b) = (poly(&ring, &av), poly(&ring, &bv));
        let case = format!("q {q}");
        // The width follows the ring.
        let expect_width = if q < two(63) {
            0
        } else if q < two(126) {
            1
        } else {
            2
        };
        let width = match a.coeffs() {
            Coeffs::Word(_) => 0,
            Coeffs::Double(_) => 1,
            Coeffs::Wide(_) => 2,
        };
        assert_eq!(width, expect_width, "{case}");
        assert_eq!(coefficients(&a), av, "{case}");
        // Additions, negation, scaling.
        let sum: Vec<BigUint> = av.iter().zip(&bv).map(|(x, y)| (x + y) % &q).collect();
        let difference: Vec<BigUint> = av.iter().zip(&bv).map(|(x, y)| (x + &q - y) % &q).collect();
        assert_eq!(coefficients(&a.add(&b).unwrap()), sum, "{case}");
        assert_eq!(coefficients(&a.sub(&b).unwrap()), difference, "{case}");
        let mut x = a.clone();
        x.add_assign(&b).unwrap();
        assert_eq!(coefficients(&x), sum, "{case}");
        let mut x = a.clone();
        x.sub_assign(&b).unwrap();
        assert_eq!(coefficients(&x), difference, "{case}");
        let negated: Vec<BigUint> = av.iter().map(|x| (&q - x) % &q).collect();
        assert_eq!(coefficients(&a.neg()), negated, "{case}");
        for c in edges.iter().chain([rng.below(&q), rng.below(&q)].iter()) {
            let scaled: Vec<BigUint> = av.iter().map(|x| x * c % &q).collect();
            assert_eq!(
                coefficients(&a.scale_u256(&u256(c)).unwrap()),
                scaled,
                "{case} c {c}"
            );
            let mut x = a.clone();
            x.scale_assign(&u256(c));
            assert_eq!(coefficients(&x), scaled, "{case} c {c}");
        }
        // The negacyclic product.
        let mut product = vec![BigInt::from(0); d];
        for (i, x) in av.iter().enumerate() {
            for (j, y) in bv.iter().enumerate() {
                let t = BigInt::from(x * y);
                if i + j < d {
                    product[i + j] += t;
                } else {
                    product[i + j - d] -= t;
                }
            }
        }
        let product: Vec<BigUint> = product.iter().map(|x| modulo(x, &q)).collect();
        assert_eq!(coefficients(&a.mul(&b).unwrap()), product, "{case}");
        // Rotations and the automorphism.
        for j in [-129i64, -65, -1, 0, 1, 5, 63, 64, 127, 200] {
            let shift = j.rem_euclid(2 * d as i64) as usize;
            let mut rotated = vec![BigUint::from(0u32); d];
            for (i, x) in av.iter().enumerate() {
                let k = i + shift;
                rotated[k % d] = if (k / d).is_multiple_of(2) {
                    x.clone()
                } else {
                    (&q - x) % &q
                };
            }
            assert_eq!(coefficients(&a.rotate(j)), rotated, "{case} j {j}");
        }
        let auto: Vec<BigUint> = (0..d)
            .map(|i| {
                if i == 0 {
                    av[0].clone()
                } else {
                    (&q - &av[d - i]) % &q
                }
            })
            .collect();
        assert_eq!(coefficients(&a.auto()), auto, "{case}");
        // Norms and centred values.
        let magnitudes: Vec<BigUint> = av
            .iter()
            .map(|x| centred(x, &q).magnitude().clone())
            .collect();
        let max = magnitudes.iter().max().unwrap();
        assert_eq!(big(&a.norm_infinity()), *max, "{case}");
        let squares: BigUint = magnitudes.iter().map(|m| m * m).sum();
        let expect = if squares < two(256) {
            squares
        } else {
            two(256) - 1u32
        };
        assert_eq!(big(&a.norm_squared()), expect, "{case}");
        for (i, x) in av.iter().enumerate() {
            assert_eq!(big(&a.magnitude(i)), magnitudes[i], "{case} i {i}");
            let c = centred(x, &q);
            assert_eq!(
                a.centred(i).map(BigInt::from),
                i128::try_from(c.clone()).ok().map(BigInt::from),
                "{case} i {i}"
            );
        }
        // Reduction of i128 values and single coefficients.
        let mut xs = vec![
            0i128,
            1,
            -1,
            i128::MAX,
            i128::MIN,
            i128::MIN + 1,
            i64::MAX as i128,
            i64::MIN as i128,
        ];
        if let Ok(v) = i128::try_from(q.clone()) {
            xs.extend([v, -v, v - 1, 1 - v, v + 1, -v - 1]);
        }
        while xs.len() < d {
            xs.push(((u128::from(rng.next()) << 64) | u128::from(rng.next())) as i128);
        }
        let reduced: Vec<BigUint> = xs.iter().map(|x| modulo(&BigInt::from(*x), &q)).collect();
        let p = Poly::new(ring.clone(), xs.clone()).unwrap();
        assert_eq!(coefficients(&p), reduced, "{case}");
        let mut p = Poly::zero(ring.clone());
        for (i, x) in xs.iter().enumerate() {
            p.set_coefficient(i, *x).unwrap();
        }
        assert_eq!(coefficients(&p), reduced, "{case}");
        assert_eq!(
            coefficients(&Poly::constant(ring.clone(), -5)),
            {
                let mut v = vec![BigUint::from(0u32); d];
                v[0] = modulo(&BigInt::from(-5), &q);
                v
            },
            "{case}"
        );
        // Shapes.
        assert!(Poly::zero(ring.clone()).is_zero() && !a.is_zero(), "{case}");
        assert_eq!(
            Poly::zero(ring.clone()).sole_nonzero(),
            Some(None),
            "{case}"
        );
        let mut monomial = Poly::zero(ring.clone());
        monomial.set_coefficient(7, 2).unwrap();
        assert_eq!(monomial.monomial(), Some((7, U256::from_u8(2))), "{case}");
        assert!(!monomial.is_scalar(), "{case}");
        assert_eq!(a.sole_nonzero(), None, "{case}");
    }
}

#[test]
fn the_wide_copy_follows_each_change() {
    // The copy that `coefficients` returns is made on the first call, and every method that
    // changes the coefficients drops it, so a later call sees the new values.
    for q in moduli() {
        let ring = Ring::with_modulus(u256(&q), 64).unwrap();
        let mut rng = Rng(32);
        let values: Vec<BigUint> = (0..64).map(|_| rng.below(&q)).collect();
        let other = poly(&ring, &(0..64).map(|_| rng.below(&q)).collect::<Vec<_>>());
        let c = u256(&rng.below(&q));
        let mut p = poly(&ring, &values);
        let check = |p: &Poly, case: &str| {
            let copy: Vec<U256> = p.coefficients().to_vec();
            assert_eq!(copy, p.values().collect::<Vec<_>>(), "q {q} {case}");
        };
        check(&p, "initial");
        p.add_assign(&other).unwrap();
        check(&p, "add_assign");
        p.sub_assign(&other).unwrap();
        check(&p, "sub_assign");
        p.add_scaled_assign(&other, &c).unwrap();
        check(&p, "add_scaled_assign");
        p.add_monomial_product_assign(&other, 3, &c).unwrap();
        check(&p, "add_monomial_product_assign");
        p.scale_assign(&c);
        check(&p, "scale_assign");
        p.set_coefficient(9, -1).unwrap();
        check(&p, "set_coefficient");
        let cloned = p.clone();
        check(&cloned, "clone");
        assert_eq!(cloned, p, "q {q}");
    }
}
