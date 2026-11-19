//! The AVX-512 transforms against the scalar ones: every NTT prime, every degree from 16 to
//! 1024, random residues and the edges 0 and p - 1, forward, inverse and one after the other.
//! On a processor without AVX-512 the test says so and passes, the scalar code being the only
//! code that runs there.
use crate::math::ntt::{NttPlan, pow_mod};
use crate::params::moduli::NTT_PRIMES;

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

#[test]
fn avx512_transforms_equal_the_scalar_ones() {
    if !super::avx512() {
        eprintln!("no AVX-512 here (or JALI_SCALAR set): only the scalar transforms run");
        return;
    }
    let mut rng = Rng(41);
    for (p, root) in NTT_PRIMES {
        for d in [16usize, 32, 64, 128, 256, 512, 1024] {
            let plan = NttPlan::new(p, d, pow_mod(root, (1024 / d) as u64, p)).unwrap();
            for case in 0..5 {
                let a: Vec<u64> = (0..d)
                    .map(|i| match (case, i % 7) {
                        (0, _) => 0,
                        (1, _) => p - 1,
                        (2, 0) => p - 1,
                        (2, 1) => 0,
                        _ => rng.next() % p,
                    })
                    .collect();
                let (mut scalar, mut vector) = (a.clone(), a.clone());
                plan.forward_mont_scalar(&mut scalar);
                assert!(super::forward_mont(&plan.simd_tables(), &mut vector));
                assert_eq!(vector, scalar, "forward p {p} d {d} case {case}");
                let (mut scalar, mut vector) = (a.clone(), a.clone());
                plan.inverse_mont_scalar(&mut scalar);
                assert!(super::inverse_mont(&plan.simd_tables(), &mut vector));
                assert_eq!(vector, scalar, "inverse p {p} d {d} case {case}");
                let mut x = a.clone();
                plan.forward_mont(&mut x);
                plan.inverse_mont(&mut x);
                assert_eq!(x, a, "round trip p {p} d {d} case {case}");
            }
        }
    }
}
