//! `QuadEq::evaluate_constant` against the constant coefficient of `QuadEq::evaluate`, for
//! equations with zero, scalar, monomial and general entries in both forms, at moduli in each
//! state of `ConstantSum`.
use super::*;

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

fn word(rng: &mut Rng) -> U256 {
    crate::test_words::u256_from_u64s([rng.next(), rng.next(), rng.next(), rng.next()])
}

/// A polynomial of each shape in turn: zero, a scalar, a monomial, a general one.
fn shaped(ring: &Arc<Ring>, rng: &mut Rng, kind: u64) -> Poly {
    let d = ring.degree();
    let mut values = vec![U256::ZERO; d];
    match kind % 4 {
        0 => {}
        1 => values[0] = word(rng),
        2 => {
            let k = 1 + (rng.next() as usize) % (d - 1);
            values[k] = word(rng);
        }
        _ => {
            for v in values.iter_mut() {
                *v = word(rng);
            }
        }
    }
    Poly::from_u256(ring.clone(), values).unwrap()
}

#[test]
fn constant_evaluations_equal_full_evaluations() {
    let two = |k: u32| U256::ONE.shl_vartime(k);
    let moduli = [
        U256::from_u8(13),
        U256::from_u64(1099511627917),
        two(64).wrapping_sub(&U256::from_u8(59)),
        U256::from_u128(800602353498606932557),
        two(128).wrapping_add(&U256::from_u8(165)),
    ];
    let mut rng = Rng(41);
    for q in moduli {
        for d in [64, 128] {
            let ring = Ring::with_modulus(q, d).unwrap();
            let dim = 6;
            for round in 0..8u64 {
                let witness = PolyVec::new(
                    ring.clone(),
                    (0..dim)
                        .map(|i| shaped(&ring, &mut rng, i as u64 + round))
                        .collect(),
                )
                .unwrap();
                let mut eq = QuadEq::zero(ring.clone(), dim).unwrap();
                let mut r2 = Vec::new();
                for r in 0..dim as u16 {
                    for c in r..dim as u16 {
                        if !rng.next().is_multiple_of(3) {
                            let kind = rng.next();
                            r2.push((r, c, shaped(&ring, &mut rng, kind)));
                        }
                    }
                }
                eq.r2 = SparsePolyMat::new(ring.clone(), dim, r2).unwrap();
                let mut r1 = Vec::new();
                for i in 0..dim as u16 {
                    let kind = rng.next();
                    r1.push((i, shaped(&ring, &mut rng, kind)));
                }
                eq.r1 = SparsePolyVec::new(ring.clone(), dim, r1).unwrap();
                eq.r0 = shaped(&ring, &mut rng, round);
                let full = eq.evaluate(&witness).unwrap();
                assert_eq!(
                    eq.evaluate_constant(&witness).unwrap(),
                    full.coefficients()[0],
                    "q {q} d {d} round {round}"
                );
            }
            // The refusals of `evaluate`: a witness of another length.
            let eq = QuadEq::zero(ring.clone(), 3).unwrap();
            let short = PolyVec::new(ring.clone(), vec![Poly::zero(ring.clone()); 2]).unwrap();
            assert_eq!(
                eq.evaluate_constant(&short).err(),
                eq.evaluate(&short).err()
            );
        }
    }
}
