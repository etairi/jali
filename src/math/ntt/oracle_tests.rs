//! The transform against a reference transform, kept below as the oracle: equal
//! outputs for all nine NTT primes and every degree from 2 to 1024, on boundary and random
//! residues; the Montgomery form against the standard one; Shoup and Montgomery products against
//! remainders, at the NTT primes and at primes just below $`2^{63}`$.
use super::*;
use crate::params::moduli::NTT_PRIMES;

/// The reference plan: a remainder per product, the twiddles advanced
/// by a product per butterfly, each stage's step and the inverse's constants recomputed.
struct Reference {
    p: u64,
    d: usize,
    twist: Vec<u64>,
    untwist: Vec<u64>,
    root: u64,
}
impl Reference {
    fn new(p: u64, d: usize, psi: u64) -> Self {
        let inv_psi = pow_mod(psi, p - 2, p);
        let mut twist = vec![1; d];
        let mut untwist = vec![1; d];
        for i in 1..d {
            twist[i] = mul_mod(twist[i - 1], psi, p);
            untwist[i] = mul_mod(untwist[i - 1], inv_psi, p);
        }
        Self {
            p,
            d,
            twist,
            untwist,
            root: mul_mod(psi, psi, p),
        }
    }
    fn cyclic(&self, a: &mut [u64], root: u64) {
        let mut j = 0;
        for i in 1..self.d {
            let mut bit = self.d >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                a.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= self.d {
            let step = pow_mod(root, (self.d / len) as u64, self.p);
            for block in a.chunks_exact_mut(len) {
                let mut w = 1;
                for j in 0..len / 2 {
                    let u = block[j];
                    let v = mul_mod(block[j + len / 2], w, self.p);
                    block[j] = (u + v) % self.p;
                    block[j + len / 2] = (u + self.p - v) % self.p;
                    w = mul_mod(w, step, self.p);
                }
            }
            len *= 2;
        }
    }
    fn forward(&self, a: &mut [u64]) {
        for (x, t) in a.iter_mut().zip(&self.twist) {
            *x = mul_mod(*x, *t, self.p);
        }
        self.cyclic(a, self.root);
    }
    fn inverse(&self, a: &mut [u64]) {
        self.cyclic(a, pow_mod(self.root, self.p - 2, self.p));
        let inv_d = pow_mod(self.d as u64, self.p - 2, self.p);
        for (x, t) in a.iter_mut().zip(&self.untwist) {
            *x = mul_mod(mul_mod(*x, inv_d, self.p), *t, self.p);
        }
    }
}

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

/// Boundary vectors (zero, one, $`p-1`$, alternating, unit vectors at both ends) and random
/// ones, of length `d` over $`[0,p)`$.
fn inputs(p: u64, d: usize, rng: &mut Rng) -> Vec<Vec<u64>> {
    let mut out = vec![vec![0; d], vec![1; d], vec![p - 1; d]];
    out.push((0..d).map(|i| if i % 2 == 0 { p - 1 } else { 1 }).collect());
    for i in [0, d - 1] {
        let mut unit = vec![0; d];
        unit[i] = p - 1;
        out.push(unit);
    }
    for _ in 0..3 {
        out.push((0..d).map(|_| rng.next() % p).collect());
    }
    out
}

#[test]
fn transforms_equal_the_reference_for_every_prime_and_degree() {
    let mut rng = Rng(1);
    for (p, root) in NTT_PRIMES {
        let r = ((1u128 << 64) % u128::from(p)) as u64;
        for log_d in 1..=10 {
            let d = 1usize << log_d;
            let psi = pow_mod(root, (1024 / d) as u64, p);
            let plan = NttPlan::new(p, d, psi).unwrap();
            let slow = Reference::new(p, d, psi);
            for a in inputs(p, d, &mut rng) {
                let (mut new, mut expected) = (a.clone(), a.clone());
                plan.forward(&mut new).unwrap();
                slow.forward(&mut expected);
                assert_eq!(new, expected, "forward p {p} d {d}");
                // The Montgomery form is the standard one times R.
                let mut mont = a.clone();
                plan.forward_mont(&mut mont);
                let scaled: Vec<u64> = new.iter().map(|x| mul_mod(*x, r, p)).collect();
                assert_eq!(mont, scaled, "forward_mont p {p} d {d}");
                plan.inverse_mont(&mut mont);
                assert_eq!(mont, a, "inverse_mont p {p} d {d}");
                // The inverse of an arbitrary vector, and the inverse of the transform.
                let (mut new_inv, mut expected_inv) = (a.clone(), a.clone());
                plan.inverse(&mut new_inv).unwrap();
                slow.inverse(&mut expected_inv);
                assert_eq!(new_inv, expected_inv, "inverse p {p} d {d}");
                plan.inverse(&mut new).unwrap();
                assert_eq!(new, a, "round trip p {p} d {d}");
            }
        }
    }
}

#[test]
fn montgomery_and_shoup_products_equal_remainders() {
    let mut rng = Rng(2);
    let reference = |x: u64, w: u64, p: u64| (u128::from(x) * u128::from(w) % u128::from(p)) as u64;
    let values = |p: u64, rng: &mut Rng| {
        let mut v = vec![0, 1, 2, p / 2, p / 2 + 1, p - 2, p - 1];
        v.extend((0..40).map(|_| rng.next() % p));
        v
    };
    // Shoup products need only p < 2^63: the NTT primes and the three largest primes below
    // 2^63, where the bound 2p < 2^64 is tightest. Any x < 2^64 is admitted.
    let wide: Vec<u64> = (1..1000u64)
        .map(|k| (1u64 << 63) - k)
        .filter(|p| is_prime(*p))
        .take(3)
        .collect();
    assert_eq!(wide[0], (1 << 63) - 25);
    for p in NTT_PRIMES.iter().map(|(p, _)| *p).chain(wide) {
        let values = values(p, &mut rng);
        for &w in &values {
            let w_shoup = shoup(w, p);
            for x in values
                .iter()
                .copied()
                .chain([p, 2 * p - 1, u64::MAX - 1, u64::MAX])
            {
                assert_eq!(
                    shoup_mul(x, w, w_shoup, p),
                    reference(x, w, p),
                    "p {p} x {x} w {w}"
                );
            }
        }
    }
    // Montgomery products and the sums of the butterflies, at each NTT prime.
    for (p, root) in NTT_PRIMES {
        let plan = NttPlan::new(p, 2, pow_mod(root, 512, p)).unwrap();
        assert_eq!(p.wrapping_mul(plan.neg_inv.wrapping_neg()), 1);
        let r_inv = pow_mod(((1u128 << 64) % u128::from(p)) as u64, p - 2, p);
        let values = values(p, &mut rng);
        for &a in &values {
            for &b in &values {
                assert_eq!(plan.mont_mul(a, b), reference(reference(a, b, p), r_inv, p));
                assert_eq!(
                    u128::from(add_mod(a, b, p)),
                    (u128::from(a) + u128::from(b)) % u128::from(p)
                );
                assert_eq!(
                    u128::from(sub_mod(a, b, p)),
                    (u128::from(a) + u128::from(p - b)) % u128::from(p)
                );
            }
        }
    }
}
