//! The accumulator's sums against sums of exact products, below and past its spill at
//! `NADDS` products, and its refusal of products from another ring.
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

fn general(ring: &Arc<Ring>, rng: &mut Rng, top: bool) -> Poly {
    let values = (0..ring.degree())
        .map(|_| {
            if top {
                ring.q.wrapping_sub(&U256::ONE)
            } else {
                crate::test_words::u256_from_u64s([rng.next(), rng.next(), rng.next(), rng.next()])
            }
        })
        .collect();
    Poly::from_u256(ring.clone(), values).unwrap()
}

#[test]
fn accumulated_sums_equal_sums_of_exact_products() {
    let two = |k: u32| U256::ONE.shl_vartime(k);
    let moduli = [
        U256::from_u8(13),
        U256::from_u64((1 << 62) - 57),
        U256::from_u128(800602353498606932557),
        two(200).wrapping_sub(&U256::from_u8(75)),
    ];
    let mut rng = Rng(61);
    for q in moduli {
        let ring = Ring::with_modulus(q, 64).unwrap();
        // 1, 3, 4, 129 and 300 (spills) products, the largest residues in half of them.
        for count in [1usize, 3, 4, 129, 300] {
            let mut sum = Accumulator::new(&ring);
            let mut out = general(&ring, &mut rng, false);
            let mut want = out.clone();
            for k in 0..count {
                let a = general(&ring, &mut rng, k % 2 == 0);
                let b = general(&ring, &mut rng, k % 3 == 0);
                sum.mac(&a.to_ntt(), &b.to_ntt(), &mut out).unwrap();
                want = want.add(&a.mul(&b).unwrap()).unwrap();
            }
            sum.finish_into(&mut out).unwrap();
            assert_eq!(out, want, "q {q} count {count}");
        }
    }
}

#[test]
fn products_of_another_ring_are_refused() {
    let ring = Ring::new(13, 64).unwrap();
    let other = Ring::new(17, 64).unwrap();
    let mut rng = Rng(62);
    let mut sum = Accumulator::new(&ring);
    let mut out = Poly::zero(ring.clone());
    let (a, b) = (
        general(&ring, &mut rng, false),
        general(&other, &mut rng, false),
    );
    assert_eq!(
        sum.mac(&a.to_ntt(), &b.to_ntt(), &mut out),
        Err(Error::RingMismatch)
    );
}
