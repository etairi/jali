//! Exactness of the RNS arithmetic (Shoup and Montgomery residues, native CRT
//! below $`2^{63}`$) against num-bigint schoolbook products, at moduli from 2 to
//! $`2^{256}-435`$ on both sides of every native-arithmetic boundary ($`2^{62}`$, $`2^{63}`$,
//! $`2^{64}`$, $`2^{128}`$): random and boundary operands, round trips, and 128 accumulated
//! products of extreme centred values, the CRT capacity, with the 129th refused.
use jali::{
    Error,
    math::{Poly, Ring, U256},
};
use num_bigint::{BigInt, Sign};
use std::sync::Arc;

mod common;
use common::{SplitMix64, from_u64_words};

fn big(x: &U256) -> BigInt {
    BigInt::from_bytes_le(Sign::Plus, &x.to_le_bytes())
}
/// The canonical representative of `x` modulo `q`.
fn canonical(x: &BigInt, q: &BigInt) -> U256 {
    let r = ((x % q) + q) % q;
    let (_, mut bytes) = r.to_bytes_le();
    bytes.resize(32, 0);
    U256::from_le_slice(&bytes)
}
/// Schoolbook product in $`\mathbb Z_q[X]/(X^d+1)`$, canonical, times `times`.
fn schoolbook(a: &Poly, b: &Poly, times: u32) -> Vec<U256> {
    let q = big(&a.ring().modulus());
    let d = a.ring().degree();
    let (a, b): (Vec<BigInt>, Vec<BigInt>) = (
        a.coefficients().iter().map(big).collect(),
        b.coefficients().iter().map(big).collect(),
    );
    let mut out = vec![BigInt::from(0); d];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            if i + j < d {
                out[i + j] += x * y;
            } else {
                out[i + j - d] -= x * y;
            }
        }
    }
    out.iter().map(|x| canonical(&(x * times), &q)).collect()
}

/// The moduli of the test plan: tiny ones, odd and even; about $`2^{40}`$; both sides of
/// $`2^{62}`$, $`2^{63}`$ and $`2^{64}`$; and wide ones up to the largest prime below
/// $`2^{256}`$ that is $`5\bmod8`$.
fn moduli() -> Vec<U256> {
    let two = |k: u32| U256::ONE.shl_vartime(k);
    let minus = |k: u32, c: u64| two(k).wrapping_sub(&U256::from_u64(c));
    let plus = |k: u32, c: u64| two(k).wrapping_add(&U256::from_u64(c));
    vec![
        U256::from_u64(2),
        U256::from_u64(3),
        U256::from_u64(12),
        U256::from_u64(13),
        U256::from_u64(1099511627917),
        minus(62, 57),
        plus(62, 3),
        minus(63, 1),
        two(63),
        plus(63, 1),
        minus(64, 59),
        two(64),
        minus(100, 15),
        minus(128, 1),
        plus(128, 165),
        plus(240, 325),
        U256::MAX.wrapping_sub(&U256::from_u64(434)),
    ]
}

fn random_poly(ring: &Arc<Ring>, rng: &mut SplitMix64) -> Poly {
    Poly::from_u256(
        ring.clone(),
        (0..ring.degree())
            .map(|_| from_u64_words([rng.next(), rng.next(), rng.next(), rng.next()]))
            .collect(),
    )
    .unwrap()
}
/// The product of the constant-coefficient vectors $`(\alpha,\dots)`$ and $`(\beta,\dots)`$,
/// times `times`: coefficient $`k`$ is $`\alpha\beta((k+1)-(d-1-k))`$.
fn constant_product(alpha: &U256, beta: &U256, ring: &Ring, times: u32) -> Vec<U256> {
    let q = big(&ring.modulus());
    let d = ring.degree() as i64;
    let ab = big(alpha) * big(beta) * times;
    (0..d)
        .map(|k| canonical(&(&ab * (2 * k + 2 - d)), &q))
        .collect()
}
/// Constant-coefficient polynomials at the extremes, and a few structured ones.
fn boundary_polys(ring: &Arc<Ring>) -> Vec<Poly> {
    let q = ring.modulus();
    let d = ring.degree();
    let half = q.shr_vartime(1);
    let fill = |x: U256| Poly::from_u256(ring.clone(), vec![x; d]).unwrap();
    let mut out = vec![
        fill(U256::ONE),
        fill(half),
        fill(q.wrapping_sub(&half)),
        fill(q.wrapping_sub(&U256::ONE)),
    ];
    let mut monomial = vec![U256::ZERO; d];
    monomial[d - 1] = q.wrapping_sub(&U256::ONE);
    out.push(Poly::from_u256(ring.clone(), monomial).unwrap());
    out.push(
        Poly::from_u256(
            ring.clone(),
            (0..d)
                .map(|i| {
                    if i % 2 == 0 {
                        half
                    } else {
                        q.wrapping_sub(&half)
                    }
                })
                .collect(),
        )
        .unwrap(),
    );
    out
}

#[test]
fn rns_products_equal_schoolbook_products_at_every_modulus_width() {
    let mut rng = SplitMix64(25);
    for q in moduli() {
        for d in [64, 128] {
            let ring = Ring::with_modulus(q, d).unwrap();
            let mut polys = boundary_polys(&ring);
            polys.extend((0..3).map(|_| random_poly(&ring, &mut rng)));
            for a in &polys {
                assert_eq!(
                    &a.to_ntt().to_poly().unwrap(),
                    a,
                    "round trip, q {q}, d {d}"
                );
                for b in &polys {
                    let want = schoolbook(a, b, 1);
                    let product = a.to_ntt().product(&b.to_ntt()).unwrap();
                    assert_eq!(
                        product.reduce().unwrap().coefficients(),
                        want,
                        "q {q}, d {d}"
                    );
                    assert_eq!(a.mul(b).unwrap().coefficients(), want, "mul, q {q}, d {d}");
                }
            }
        }
    }
}

#[test]
fn accumulated_extreme_products_reach_the_capacity_and_the_next_is_refused() {
    for q in moduli() {
        for d in [64, 1024] {
            let ring = Ring::with_modulus(q, d).unwrap();
            let half = q.shr_vartime(1);
            // All coefficients at +floor(q/2), or at -floor(q/2): coefficient d - 1 of a product
            // is then d floor(q/2)^2 in absolute value, the largest any product can reach.
            let plus = Poly::from_u256(ring.clone(), vec![half; d]).unwrap();
            let minus = Poly::from_u256(ring.clone(), vec![q.wrapping_sub(&half); d]).unwrap();
            for (a, b) in [(&plus, &plus), (&plus, &minus), (&minus, &minus)] {
                let product = a.to_ntt().product(&b.to_ntt()).unwrap();
                let mut sum = a.to_ntt().product(&b.to_ntt()).unwrap();
                for _ in 1..128 {
                    sum.add_assign(&product).unwrap();
                }
                let (alpha, beta) = (a.coefficients()[0], b.coefficients()[0]);
                let want = constant_product(&alpha, &beta, &ring, 128);
                if d == 64 {
                    assert_eq!(want, schoolbook(a, b, 128), "closed form, q {q}");
                }
                assert_eq!(sum.reduce().unwrap().coefficients(), want, "q {q}, d {d}");
                assert_eq!(sum.add_assign(&product), Err(Error::Overflow));
                assert_eq!(
                    sum.reduce().unwrap().coefficients(),
                    want,
                    "after refusal, q {q}"
                );
            }
        }
    }
}

/// $`a\cdot b`$ through the transforms alone, without the fast paths of `Poly::mul`.
fn ntt_product(a: &Poly, b: &Poly) -> Poly {
    a.to_ntt().product(&b.to_ntt()).unwrap().reduce().unwrap()
}

#[test]
fn monomial_and_scalar_products_equal_ntt_products() {
    let mut rng = SplitMix64(9);
    for q in moduli() {
        let ring = Ring::with_modulus(q, 64).unwrap();
        let d = ring.degree();
        let half = q.shr_vartime(1);
        let minus = |k: u64| q.wrapping_sub(&U256::from_u64(k));
        // Scalars: small, the negative ones near q (q - 1 has its own path), around q/2, and
        // random; then unreduced ones, which scale_u256 reduces first.
        let mut scalars = vec![U256::ZERO, U256::ONE, U256::from_u64(2), minus(1)];
        if q > U256::from_u64(4) {
            scalars.extend([minus(2), minus(3)]);
        }
        scalars.extend([half, half.wrapping_add(&U256::ONE)]);
        scalars.retain(|x| x < &q);
        scalars.push(random_poly(&ring, &mut rng).coefficients()[0]);
        let unreduced = [q, q.wrapping_add(&U256::ONE), U256::MAX];
        let a = random_poly(&ring, &mut rng);
        for c in scalars.iter().chain(&unreduced) {
            let constant = Poly::constant_u256(ring.clone(), c);
            let want = ntt_product(&a, &constant);
            assert_eq!(a.scale_u256(c).unwrap(), want, "q {q}, scalar {c}");
            assert_eq!(a.mul(&constant).unwrap(), want, "q {q}, scalar {c}");
            assert_eq!(constant.mul(&a).unwrap(), want, "q {q}, scalar {c}");
            if c >= &q {
                continue;
            }
            for k in [1, 2, d / 2, d - 1] {
                let mut coefficients = vec![U256::ZERO; d];
                coefficients[k] = *c;
                let monomial = Poly::from_u256(ring.clone(), coefficients).unwrap();
                let want = ntt_product(&a, &monomial);
                assert_eq!(a.mul(&monomial).unwrap(), want, "q {q}, {c} X^{k}");
                assert_eq!(monomial.mul(&a).unwrap(), want, "q {q}, {c} X^{k}");
            }
        }
        // Two nonzero coefficients are not a monomial.
        let mut coefficients = vec![U256::ZERO; d];
        coefficients[3] = U256::ONE;
        coefficients[d - 1] = minus(1);
        let binomial = Poly::from_u256(ring.clone(), coefficients).unwrap();
        assert_eq!(
            a.mul(&binomial).unwrap(),
            ntt_product(&a, &binomial),
            "q {q}"
        );
        assert!(Poly::zero(ring.clone()).is_zero() && !binomial.is_zero());
    }
}
