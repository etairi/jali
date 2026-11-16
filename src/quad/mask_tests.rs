//! `MaskLinear` against the three products it replaces in the quadratic proof's attempts:
//! $`\mathrm{bilinear}(s,y)+\mathrm{bilinear}(y,s)+r_1\cdot y`$, for equations with zero,
//! scalar, monomial and general entries and masks of every shape, at small, PCS-size and
//! large moduli.
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

fn shaped(ring: &Arc<Ring>, rng: &mut Rng, kind: u64) -> Poly {
    let d = ring.degree();
    let mut values = vec![U256::ZERO; d];
    let word = |rng: &mut Rng| {
        crate::test_words::u256_from_u64s([rng.next(), rng.next(), rng.next(), rng.next()])
    };
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
fn the_hoisted_form_equals_the_three_products() {
    let two = |k: u32| U256::ONE.shl_vartime(k);
    let moduli = [
        U256::from_u8(13),
        U256::from_u64(1099511627917),
        U256::from_u128(800602353498606932557),
        two(200).wrapping_sub(&U256::from_u8(75)),
    ];
    let mut rng = Rng(71);
    for q in moduli {
        let ring = Ring::with_modulus(q, 64).unwrap();
        let dim = 5;
        for round in 0..6u64 {
            let s = PolyVec::new(
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
            let linear = MaskLinear::new(&eq, &s).unwrap();
            for attempt in 0..4u64 {
                let y = PolyVec::new(
                    ring.clone(),
                    (0..dim)
                        .map(|i| shaped(&ring, &mut rng, i as u64 + attempt + round))
                        .collect(),
                )
                .unwrap();
                let want = eq
                    .r2
                    .bilinear(&s, &y)
                    .unwrap()
                    .add(&eq.r2.bilinear(&y, &s).unwrap())
                    .unwrap()
                    .add(&eq.r1.dot(&y).unwrap())
                    .unwrap();
                assert_eq!(linear.apply(&y).unwrap(), want, "q {q} round {round}");
            }
            // The refusals of the products: a mask of another length.
            let short =
                PolyVec::new(ring.clone(), vec![Poly::zero(ring.clone()); dim - 1]).unwrap();
            assert_eq!(linear.apply(&short), Err(Error::Dimension));
        }
    }
}
