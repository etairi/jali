//! `ConstantSum` against the constant coefficient of products computed in full, in each of its
//! three states, with sums long enough to carry out of 128 bits below $`2^{64}`$.
use super::*;
use crate::math::Poly;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

fn random_poly(ring: &std::sync::Arc<Ring>, rng: &mut Rng) -> Poly {
    let values = (0..ring.d)
        .map(|_| {
            crate::test_words::u256_from_u64s([rng.next(), rng.next(), rng.next(), rng.next()])
        })
        .collect();
    Poly::from_u256(ring.clone(), values).unwrap()
}

/// The polynomial with every coefficient $`q-1`$: the largest products.
fn top_poly(ring: &std::sync::Arc<Ring>) -> Poly {
    Poly::from_u256(ring.clone(), vec![ring.q.wrapping_sub(&U256::ONE); ring.d]).unwrap()
}

#[test]
fn constant_sums_equal_constant_coefficients_of_full_products() {
    let two = |k: u32| U256::ONE.shl_vartime(k);
    let moduli = [
        U256::from_u8(13),
        U256::from_u64(1099511627917),
        two(62).wrapping_sub(&U256::ONE),
        two(63).wrapping_add(&U256::from_u8(29)),
        two(64).wrapping_sub(&U256::from_u8(59)),
        two(64).wrapping_add(&U256::from_u8(13)),
        U256::from_u128(800602353498606932557),
        two(125).wrapping_add(&U256::from_u8(2)),
        two(128).wrapping_add(&U256::from_u8(165)),
        two(200).wrapping_sub(&U256::from_u8(75)),
    ];
    let mut rng = Rng(31);
    for q in moduli {
        for d in [64, 256] {
            let ring = Ring::with_modulus(q, d).unwrap();
            // 40 random products, then 40 of the largest, then a constant.
            let mut pairs: Vec<(Poly, Poly)> = (0..40)
                .map(|_| (random_poly(&ring, &mut rng), random_poly(&ring, &mut rng)))
                .collect();
            pairs.extend((0..40).map(|_| (top_poly(&ring), top_poly(&ring))));
            let extra = random_poly(&ring, &mut rng).coefficients()[0];
            let mut sum = ConstantSum::new(&ring);
            let mut full = Poly::zero(ring.clone());
            for (a, b) in &pairs {
                sum.add_product(a, b);
                full = full.add(&a.mul(b).unwrap()).unwrap();
            }
            sum.add(&extra);
            let want = ring.add(&full.coefficients()[0], &extra);
            assert_eq!(sum.finish(), want, "q {q} d {d}");
        }
    }
}

#[test]
fn small_sums_carry_out_of_128_bits() {
    // (q - 1)^2 is just below 2^128, so every term of the top products carries.
    let q = U256::ONE.shl_vartime(64).wrapping_sub(&U256::from_u8(59));
    let ring = Ring::with_modulus(q, 64).unwrap();
    let top = top_poly(&ring);
    let mut sum = ConstantSum::new(&ring);
    for _ in 0..3 {
        sum.add_product(&top, &top);
    }
    if let State::Small {
        positive, negative, ..
    } = &sum.state
    {
        assert!(positive.1 >= 1 && negative.1 >= 60);
    } else {
        panic!("a modulus below 2^64 takes the small state");
    }
    let want = top.mul(&top).unwrap().coefficients()[0];
    let want = ring.add(&ring.add(&want, &want), &want);
    assert_eq!(sum.finish(), want);
}
