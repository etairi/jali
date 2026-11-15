//! The ring arithmetic against an `i128` schoolbook oracle: products, accumulated inner
//! products, sparse products and $`\sigma_{-1}`$, at every supported degree and at moduli from
//! just above $`2^{16}`$ to just below $`2^{48}`$, on random and extreme operands.
use super::*;
use crate::labrador::test_rng::TestRng;

/// Moduli $`\equiv5\pmod8`$, prime: the smallest above $`2^{16}`$, the largest below
/// $`2^{20}`$, $`2^{32}`$, $`2^{40}`$, $`2^{47}`$ and $`2^{48}`$, and $`2^{36}-243`$,
/// $`2^{38}-107`$.
const MODULI: [u64; 8] = [
    65557,
    1048573,
    4294967197,
    68719476493,
    274877906837,
    1099511627581,
    140737488355213,
    281474976710597,
];

fn schoolbook(ring: &Rq, a: &[u64], b: &[u64]) -> Vec<i128> {
    let d = ring.degree();
    let mut acc = vec![0i128; d];
    for i in 0..d {
        for j in 0..d {
            let p = i128::from(ring.center(a[i])) * i128::from(ring.center(b[j]));
            if i + j < d {
                acc[i + j] += p;
            } else {
                acc[i + j - d] -= p;
            }
        }
    }
    acc
}

fn reduce(ring: &Rq, v: &[i128]) -> Vec<u64> {
    v.iter().map(|x| ring.lift_i128(*x)).collect()
}

fn random_poly(ring: &Rq, rng: &mut TestRng) -> Vec<u64> {
    (0..ring.degree())
        .map(|_| rng.below(ring.modulus()))
        .collect()
}

#[test]
fn construction_accepts_exactly_the_supported_moduli_and_degrees() {
    for q in MODULI {
        for d in [64, 128, 256] {
            let ring = Rq::new(q, d).unwrap();
            assert_eq!((ring.modulus(), ring.degree()), (q, d));
            // Two primes up to 2^40, three above.
            assert_eq!(ring.primes(), if q < 1 << 40 { 2 } else { 3 });
        }
    }
    // Composite, 1 or 3 mod 8, out of range, and unsupported degrees.
    for q in [65557 * 3, 65537, 4294967291, 13, 1 << 48, (1 << 48) + 5] {
        assert_eq!(
            Rq::new(q, 64).err(),
            Some(Error::Parameter("LaBRADOR modulus")),
            "{q}"
        );
    }
    for d in [0, 16, 32, 96, 512] {
        assert_eq!(
            Rq::new(4294967197, d).err(),
            Some(Error::Parameter("LaBRADOR ring degree"))
        );
    }
}

#[test]
fn products_match_the_schoolbook_oracle() {
    let mut rng = TestRng::new(1);
    for q in MODULI {
        for d in [64, 128, 256] {
            let ring = Rq::new(q, d).unwrap();
            for _ in 0..4 {
                let (a, b) = (random_poly(&ring, &mut rng), random_poly(&ring, &mut rng));
                assert_eq!(
                    ring.poly_mul(&a, &b),
                    reduce(&ring, &schoolbook(&ring, &a, &b))
                );
            }
            // Extreme operands: every coefficient (q-1)/2, (q+1)/2 or q-1.
            for (x, y) in [(q / 2, q / 2), (q / 2 + 1, q / 2), (q - 1, q / 2 + 1)] {
                let (a, b) = (vec![x; d], vec![y; d]);
                assert_eq!(
                    ring.poly_mul(&a, &b),
                    reduce(&ring, &schoolbook(&ring, &a, &b))
                );
            }
        }
    }
}

#[test]
fn accumulated_worst_case_products_reconstruct_exactly() {
    // 3000 products of all-(q-1)/2 polynomials: the accumulated value is far beyond q, and the
    // CRT must still return it modulo q.
    for q in [4294967197, 281474976710597] {
        let ring = Rq::new(q, 256).unwrap();
        let a = vec![q / 2; 256];
        let one = schoolbook(&ring, &a, &a);
        let count = 3000usize;
        let hat = ring.ntt_many(&a.repeat(count));
        let got = ring.inner_hat(&hat, &hat);
        let want: Vec<i128> = one.iter().map(|x| x * count as i128).collect();
        assert_eq!(got, reduce(&ring, &want));
    }
}

#[test]
fn inner_products_and_transform_sums_match_the_oracle() {
    let mut rng = TestRng::new(2);
    let ring = Rq::new(4294967197, 64).unwrap();
    let n = 37;
    let a: Vec<u64> = (0..n).flat_map(|_| random_poly(&ring, &mut rng)).collect();
    let b: Vec<u64> = (0..n).flat_map(|_| random_poly(&ring, &mut rng)).collect();
    let mut want = vec![0i128; 64];
    let d = ring.degree();
    for (x, y) in a.chunks_exact(d).zip(b.chunks_exact(d)) {
        for (w, v) in want.iter_mut().zip(schoolbook(&ring, x, y)) {
            *w += v;
        }
    }
    assert_eq!(
        ring.inner_hat(&ring.ntt_many(&a), &ring.ntt_many(&b)),
        reduce(&ring, &want)
    );
    // Sums and differences in the transform domain are those of the coefficients.
    let (x, y) = (&a[..64], &b[..64]);
    let mut sum = ring.ntt(x);
    ring.hat_add_assign(&mut sum, &ring.ntt(y));
    let mut expected = x.to_vec();
    ring.add_assign(&mut expected, y);
    assert_eq!(ring.intt(sum), expected);
    let mut diff = ring.ntt(x);
    ring.hat_sub_assign(&mut diff, &ring.ntt(y));
    let mut expected = x.to_vec();
    ring.sub_assign(&mut expected, y);
    assert_eq!(ring.intt(diff), expected);
}

#[test]
fn sigma_minus_one_turns_constant_coefficients_into_dot_products() {
    let mut rng = TestRng::new(3);
    for d in [64, 256] {
        let ring = Rq::new(1099511627581, d).unwrap();
        let (a, b) = (random_poly(&ring, &mut rng), random_poly(&ring, &mut rng));
        let mut sa = vec![0; d];
        ring.sigma_minus_one(&a, &mut sa);
        let dot: i128 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| i128::from(ring.center(*x)) * i128::from(ring.center(*y)))
            .sum();
        assert_eq!(ring.poly_mul(&sa, &b)[0], ring.lift_i128(dot));
        // It is an involution and a ring automorphism.
        let mut back = vec![0; d];
        ring.sigma_minus_one(&sa, &mut back);
        assert_eq!(back, a);
        let (mut sb, mut sab) = (vec![0; d], vec![0; d]);
        ring.sigma_minus_one(&b, &mut sb);
        ring.sigma_minus_one(&ring.poly_mul(&a, &b), &mut sab);
        assert_eq!(sab, ring.poly_mul(&sa, &sb));
    }
}

#[test]
fn sparse_products_match_dense_ones() {
    let mut rng = TestRng::new(4);
    let ring = Rq::new(4294967197, 64).unwrap();
    for _ in 0..8 {
        let terms: Vec<(usize, i64)> = (0..10)
            .map(|i| (i * 6 + rng.below(6) as usize, rng.below(5) as i64 - 2))
            .collect();
        let c = Sparse { terms };
        let a = random_poly(&ring, &mut rng);
        let mut out = random_poly(&ring, &mut rng);
        let mut want = out.clone();
        ring.add_assign(&mut want, &ring.poly_mul(&c.dense(&ring), &a));
        ring.mul_sparse_add(&c, &a, &mut out);
        assert_eq!(out, want);
    }
}

#[test]
fn scalars_norms_and_centring() {
    let ring = Rq::new(65557, 64).unwrap();
    let q = 65557;
    assert_eq!(ring.center(q / 2), (q / 2) as i64);
    assert_eq!(ring.center(q / 2 + 1), -((q / 2) as i64));
    assert_eq!(ring.lift(-1), q - 1);
    let mut a = vec![0u64; 64];
    a[3] = q - 5;
    a[7] = 4;
    assert_eq!(ring.norm_sq(&a), 41);
    assert_eq!(ring.linf(&a), 5);
    let mut b = a.clone();
    ring.scale_assign(&mut b, 3);
    assert_eq!((ring.center(b[3]), ring.center(b[7])), (-15, 12));
    let mut c = a.clone();
    ring.add_scaled_assign(&mut c, &a, q - 1);
    assert!(c.iter().all(|x| *x == 0));
}
