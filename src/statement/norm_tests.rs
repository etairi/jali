//! The word-sized norms of [`exact_norm_squared`] and [`l1_norm`] against the 1024-bit sums
//! they replaced, for one- and two-word moduli (two words from $`2^{63}`$ to just below
//! $`2^{126}`$) and wider ones, which keep the 1024-bit sums, at every coefficient's largest
//! magnitude and on random polynomials.
use super::*;
use crate::math::{Ring, U256};

/// The sums as they were computed before: every magnitude widened to [`Bound`].
fn reference(p: &Poly) -> (Bound, Bound) {
    let (mut squared, mut l1) = (Bound::ZERO, Bound::ZERO);
    for i in 0..p.ring().degree() {
        let m = p.magnitude(i).resize::<{ Bound::LIMBS }>();
        squared = squared.wrapping_add(&m.wrapping_mul(&m));
        l1 = l1.wrapping_add(&m);
    }
    (squared, l1)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

#[test]
fn word_norms_equal_the_wide_sums() {
    let mut rng = Rng(0x5eed0fa01ce);
    let moduli = [
        U256::from_u64(65537),
        U256::from_u64(1099511627917),
        U256::from_u64((1 << 63) - 25),
        U256::from_u128(800602353498606932557),
        U256::ONE.shl_vartime(126).wrapping_sub(&U256::from_u8(1)),
        U256::ONE.shl_vartime(127).wrapping_add(&U256::from_u8(45)),
        U256::from_u128(u128::MAX - 158),
    ];
    for q in moduli {
        for d in [64usize, 128] {
            let Ok(ring) = Ring::with_modulus(q, d) else {
                continue;
            };
            let half = q.shr_vartime(1);
            let random = |rng: &mut Rng| {
                let words = [rng.next(), rng.next(), rng.next(), rng.next()];
                U256::from_le_slice(&words.map(u64::to_le_bytes).concat())
                    .rem_vartime(&crypto_bigint::NonZero::new(q).unwrap())
            };
            let polys = [
                // Every coefficient at the largest magnitude, on both sides.
                vec![half; d],
                vec![q.wrapping_sub(&half); d],
                (0..d)
                    .map(|i| {
                        if i % 2 == 0 {
                            half
                        } else {
                            q.wrapping_sub(&half)
                        }
                    })
                    .collect(),
                vec![U256::ZERO; d],
                (0..d).map(|_| random(&mut rng)).collect::<Vec<_>>(),
                (0..d).map(|_| random(&mut rng)).collect::<Vec<_>>(),
            ];
            for values in polys {
                let p = Poly::from_canonical(ring.clone(), values);
                let narrow = q < U256::ONE.shl_vartime(126);
                assert_eq!(p.fold_magnitudes((), |(), _| ()).is_some(), narrow, "q {q}");
                let (squared, l1) = reference(&p);
                assert_eq!(exact_norm_squared(&p).unwrap(), squared, "q {q} d {d}");
                assert_eq!(l1_norm(&p).unwrap(), l1, "q {q} d {d}");
            }
        }
    }
}

/// $`m^2`$ in two words against the 256-bit product, at the ends of its domain.
#[test]
fn squares_of_two_words_are_exact() {
    let mut rng = Rng(77);
    let edges = [
        0u128,
        1,
        u64::MAX.into(),
        1 << 64,
        (1 << 127) - 1,
        (1 << 126) + 12345,
    ];
    for m in edges
        .into_iter()
        .chain((0..1000).map(|_| ((u128::from(rng.next()) << 64) | u128::from(rng.next())) >> 1))
    {
        let (low, high) = square_u128(m);
        let wide = U256::from_u128(m).wrapping_mul(&U256::from_u128(m));
        let expect = U256::from_u128(low).wrapping_add(&U256::from_u128(high).shl_vartime(128));
        assert_eq!(wide, expect, "m {m}");
    }
}
