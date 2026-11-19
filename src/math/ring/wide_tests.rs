//! The ring operations that have native paths for $`q<2^{128}`$ (products, Shoup multipliers,
//! sign tests, magnitudes, centred values, residues, reduction) against big integers, on moduli
//! at both sides of $`2^{63}`$, $`2^{64}`$, $`2^{126}`$ and $`2^{128}`$.
use super::*;
use crate::math::Poly;
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
    fn u256(&mut self) -> U256 {
        crate::test_words::u256_from_u64s([self.next(), self.next(), self.next(), self.next()])
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

fn moduli() -> Vec<U256> {
    let two = |k: u32| U256::ONE.shl_vartime(k);
    vec![
        two(63).wrapping_sub(&U256::from_u8(25)),
        two(63),
        two(63).wrapping_add(&U256::ONE),
        two(64).wrapping_sub(&U256::from_u8(59)),
        two(64),
        two(64).wrapping_add(&U256::from_u8(13)),
        two(100).wrapping_sub(&U256::from_u8(15)),
        U256::from_u128(800602353498606932557),
        two(125).wrapping_add(&U256::from_u8(2)),
        two(126).wrapping_sub(&U256::ONE),
        two(126),
        two(127).wrapping_add(&U256::from_u8(45)),
        two(128).wrapping_sub(&U256::ONE),
        two(128).wrapping_add(&U256::from_u8(165)),
        two(200).wrapping_sub(&U256::from_u8(75)),
    ]
}

/// Canonical values: edges and random ones below $`q`$.
fn values(q: &U256, rng: &mut Rng) -> Vec<U256> {
    let half = q.shr_vartime(1);
    let mut out = vec![
        U256::ZERO,
        U256::ONE,
        half,
        half.wrapping_add(&U256::ONE),
        q.wrapping_sub(&U256::ONE),
    ];
    let qb = big(q);
    out.extend((0..40).map(|_| u256(&(big(&rng.u256()) % &qb))));
    out
}

#[test]
fn native_ring_operations_equal_big_integer_ones() {
    let mut rng = Rng(21);
    for q in moduli() {
        let ring = Ring::with_modulus(q, 64).unwrap();
        let qb = big(&q);
        let half = &qb >> 1u32;
        let xs = values(&q, &mut rng);
        for a in &xs {
            let ab = big(a);
            // sign, magnitude and centred value
            let negative = ab > half;
            assert_eq!(ring.is_negative(a), negative, "q {q} x {a}");
            let magnitude = if negative { &qb - &ab } else { ab.clone() };
            assert_eq!(big(&ring.magnitude(a)), magnitude, "q {q} x {a}");
            let centred = if negative {
                -BigInt::from(magnitude.clone())
            } else {
                BigInt::from(magnitude.clone())
            };
            assert_eq!(
                ring.centred_i128(a),
                i128::try_from(centred.clone()).ok(),
                "q {q} x {a}"
            );
            // residues modulo every selected prime
            for (i, plan) in ring.plans.iter().enumerate() {
                let p = BigUint::from(plan.modulus());
                assert_eq!(BigUint::from(ring.residue(a, i)), &ab % &p, "q {q} x {a}");
            }
            // products, and the scaled sums of polynomials in the ring's coefficient width
            let expect: Vec<U256> = xs.iter().map(|b| u256(&((&ab * big(b)) % &qb))).collect();
            for (b, e) in xs.iter().zip(&expect) {
                assert_eq!(ring.mul(a, b), *e, "q {q} a {a} b {b}");
            }
            let cycle = |v: &[U256]| -> Vec<U256> { (0..64).map(|i| v[i % v.len()]).collect() };
            let x = Poly::from_canonical(ring.clone(), cycle(&xs));
            let products = cycle(&expect);
            assert_eq!(
                x.scale_u256(a).unwrap().coefficients(),
                &products[..],
                "q {q} w {a}"
            );
            let mut in_place = x.clone();
            in_place.scale_assign(a);
            assert_eq!(in_place.coefficients(), &products[..], "q {q} w {a}");
            // out + x a and out + X^k x a, from an arbitrary canonical out.
            let start: Vec<U256> = cycle(&xs).into_iter().rev().collect();
            let mut added = Poly::from_canonical(ring.clone(), start.clone());
            added.add_scaled_assign(&x, a).unwrap();
            for ((s0, e), p) in start.iter().zip(&products).zip(added.coefficients()) {
                assert_eq!(*p, ring.add(s0, e), "q {q} w {a}");
            }
            let k = 5;
            let mut shifted = Poly::from_canonical(ring.clone(), start.clone());
            shifted.add_monomial_product_assign(&x, k, a).unwrap();
            for (i, s) in shifted.coefficients().iter().enumerate() {
                let e = if i >= k {
                    ring.add(&start[i], &products[i - k])
                } else {
                    ring.sub(&start[i], &products[64 + i - k])
                };
                assert_eq!(*s, e, "q {q} w {a} i {i}");
            }
        }
        // reduction of arbitrary 256-bit values
        for _ in 0..40 {
            let y = rng.u256();
            assert_eq!(big(&ring.reduce(&y)), big(&y) % &qb, "q {q} y {y}");
        }
    }
}

#[test]
fn reduce_i128_equals_big_integer_remainders() {
    let mut moduli: Vec<U256> = vec![
        U256::from_u8(2),
        U256::from_u8(13),
        U256::from_u64(1099511627917),
        U256::from_u64((1 << 62) - 1),
        U256::from_u64(i64::MAX as u64),
        U256::from_u64(1 << 63),
        U256::from_u64(u64::MAX - 58),
    ];
    moduli.extend(moduli_list_wide());
    let mut rng = Rng(22);
    for q in moduli {
        let ring = Ring::with_modulus(q, 64).unwrap();
        let qb = BigInt::from(big(&q));
        let mut xs = vec![
            0i128,
            1,
            -1,
            i64::MAX as i128,
            i64::MIN as i128,
            i64::MAX as i128 + 1,
            i64::MIN as i128 - 1,
            i128::MAX,
            i128::MIN,
            i128::MIN + 1,
        ];
        if let Some(v) = int::to_u128(&q).and_then(|v| i128::try_from(v).ok()) {
            xs.extend([v, -v, v - 1, 1 - v, v + 1, -v - 1]);
        }
        xs.extend((0..40).map(|_| (u128::from(rng.next()) << 64 | u128::from(rng.next())) as i128));
        xs.extend((0..40).map(|_| rng.next() as i64 as i128));
        for x in xs {
            let r = ((BigInt::from(x) % &qb) + &qb) % &qb;
            assert_eq!(BigInt::from(big(&ring.reduce_i128(x))), r, "q {q} x {x}");
        }
    }
}

/// The wide moduli of `moduli()` from 2^64 on.
fn moduli_list_wide() -> Vec<U256> {
    moduli()
        .into_iter()
        .filter(|q| int::to_u64(q).is_none())
        .collect()
}
