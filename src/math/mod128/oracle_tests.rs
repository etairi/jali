//! `Mod128` against `crypto-bigint`'s multi-limb arithmetic, for every bit length of the modulus
//! from 64 to 126, odd and even moduli, edge and random operands.
use super::*;

/// SplitMix64, for reproducible operands.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn word(&mut self) -> u128 {
        (u128::from(self.next()) << 64) | u128::from(self.next())
    }
    /// Uniform in $`[0,n)`$, by rejection, for $`n\ge1`$.
    fn below(&mut self, n: u128) -> u128 {
        let mask = u128::MAX >> n.leading_zeros();
        loop {
            let x = self.word() & mask;
            if x < n {
                return x;
            }
        }
    }
}

fn product_mod(a: u128, b: u128, q: u128) -> u128 {
    let q = NonZero::new(U256::from_u128(q)).unwrap();
    int::low_u128(&U256::from_u128(a).mul_mod_vartime(&U256::from_u128(b), &q))
}

/// The moduli: for each bit length 64 to 126 an odd and an even random one, then edges and a
/// prime of 69.44 bits.
fn moduli(rng: &mut Rng) -> Vec<u128> {
    let mut out = Vec::new();
    for s in 64..=126u32 {
        let low = 1u128 << (s - 1);
        let x = low + rng.below(low);
        out.push(x | 1);
        out.push(x & !1);
    }
    out.extend([
        1 << 63,
        (1 << 63) + 1,
        (1 << 64) - 59,
        1 << 64,
        (1 << 64) + 1,
        (1 << 100) - 15,
        1 << 125,
        (1 << 126) - 1,
        800602353498606932557,
    ]);
    out
}

fn operands(rng: &mut Rng, q: u128, random: usize) -> Vec<u128> {
    let mut out = vec![0, 1, 2, q / 2, q / 2 + 1, q - 2, q - 1];
    out.extend((0..random).map(|_| rng.below(q)));
    out
}

#[test]
fn wide_products_equal_multi_limb_products() {
    let mut rng = Rng(1);
    let edges = [0, 1, 2, u64::MAX as u128, 1 << 64, u128::MAX - 1, u128::MAX];
    let mut values = edges.to_vec();
    values.extend((0..200).map(|_| rng.word()));
    for a in &values {
        for b in &values {
            let (low, high) = mul_wide(*a, *b);
            // Below 2^256, so the wrapping 256-bit product is exact.
            let product = U256::from_u128(*a).wrapping_mul(&U256::from_u128(*b));
            assert_eq!(
                (low, high),
                crate::test_words::u256_halves(&product),
                "a {a} b {b}"
            );
        }
    }
}

#[test]
fn barrett_products_equal_multi_limb_products() {
    let mut rng = Rng(2);
    let mut checked = 0;
    for q in moduli(&mut rng) {
        let m = Mod128::new(&U256::from_u128(q)).expect("in range");
        assert_eq!(m.modulus(), q);
        let values = operands(&mut rng, q, 40);
        for a in &values {
            for b in &values {
                assert_eq!(m.mul(*a, *b), product_mod(*a, *b, q), "q {q} a {a} b {b}");
                checked += 1;
            }
        }
    }
    // 63 bit lengths with an odd and an even modulus, nine more, 47 operands each.
    assert_eq!(checked, (63 * 2 + 9) * 47 * 47);
}

#[test]
fn barrett_reduces_every_value_below_its_bound() {
    // reduce_wide accepts x < 2^(2s), not only products of canonical values.
    let mut rng = Rng(3);
    for q in moduli(&mut rng) {
        let m = Mod128::new(&U256::from_u128(q)).unwrap();
        let s = 128 - q.leading_zeros();
        let divisor = NonZero::new(U256::from_u128(q)).unwrap();
        let top = U256::ONE.shl_vartime(2 * s).wrapping_sub(&U256::ONE);
        let mut xs = vec![U256::ZERO, U256::ONE, top, top.shr_vartime(1)];
        for _ in 0..300 {
            let x =
                crate::test_words::u256_from_u64s([rng.next(), rng.next(), rng.next(), rng.next()]);
            xs.push(x.bitand(&top));
        }
        for x in xs {
            let (low, high) = crate::test_words::u256_halves(&x);
            assert_eq!(
                m.reduce_wide(low, high),
                int::low_u128(&x.rem_vartime(&divisor)),
                "q {q}"
            );
        }
    }
}

#[test]
fn shoup_constants_equal_quotients() {
    let mut rng = Rng(4);
    for q in moduli(&mut rng) {
        let m = Mod128::new(&U256::from_u128(q)).unwrap();
        let divisor = NonZero::new(U256::from_u128(q)).unwrap();
        for w in operands(&mut rng, q, 60) {
            let expect = U256::from_u128(w)
                .shl_vartime(128)
                .div_rem_vartime(&divisor)
                .0;
            assert_eq!(m.shoup(w), int::low_u128(&expect), "q {q} w {w}");
        }
    }
}

#[test]
fn shoup_products_equal_multi_limb_products_for_every_word() {
    // shoup_mul128 accepts any x < 2^128, not only canonical values.
    let mut rng = Rng(5);
    for q in moduli(&mut rng) {
        let m = Mod128::new(&U256::from_u128(q)).unwrap();
        let divisor = NonZero::new(U256::from_u128(q)).unwrap();
        let mut xs = operands(&mut rng, q, 30);
        xs.extend([q, q + 1, u128::MAX, u128::MAX - 1]);
        xs.extend((0..30).map(|_| rng.word()));
        for w in operands(&mut rng, q, 12) {
            let w_shoup = m.shoup(w);
            for x in &xs {
                let expect = U256::from_u128(*x)
                    .wrapping_mul(&U256::from_u128(w))
                    .rem_vartime(&divisor);
                assert_eq!(
                    shoup_mul128(*x, w, w_shoup, q),
                    int::low_u128(&expect),
                    "q {q} w {w} x {x}"
                );
            }
        }
    }
}

#[test]
fn sums_and_differences_stay_canonical() {
    let mut rng = Rng(6);
    for q in moduli(&mut rng) {
        for a in operands(&mut rng, q, 8) {
            for b in operands(&mut rng, q, 8) {
                let (sum, diff) = (add_mod128(a, b, q), sub_mod128(a, b, q));
                assert!(sum < q && diff < q);
                assert_eq!((sum + q - a) % q, b);
                assert_eq!((diff + b) % q, a);
            }
        }
    }
}

#[test]
fn the_context_covers_exactly_two_to_the_63_to_two_to_the_126() {
    for q in [
        2u128,
        13,
        (1 << 63) - 1,
        1 << 126,
        (1 << 126) + 1,
        u128::MAX,
    ] {
        assert!(Mod128::new(&U256::from_u128(q)).is_none(), "q {q}");
    }
    assert!(Mod128::new(&U256::ONE.shl_vartime(128)).is_none());
    assert!(Mod128::new(&U256::MAX).is_none());
    for q in [1u128 << 63, (1 << 126) - 1] {
        assert!(Mod128::new(&U256::from_u128(q)).is_some(), "q {q}");
    }
}
