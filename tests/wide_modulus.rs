//! Moduli from $`2^{100}`$ to $`2^{256}-1`$ against num-bigint oracles: ring arithmetic, CRT
//! capacity, norms, compression, codecs and the uniform sampler. The primes are congruent to 5
//! modulo 8 and were proven prime in Sage (`is_prime(proof=True)`): $`2^{100}-99`$,
//! $`2^{128}-275`$, $`2^{200}-75`$ and $`2^{256}-435`$, the largest such primes below their
//! powers of two, $`2^{128}+165`$ and $`2^{240}+325`$, the smallest above theirs,
//! $`2^{100}+1533`$, whose $`q-1`$ has a 33-bit divisor made of primes below $`10^6`$, and
//! $`2^{255}-19`$. Ring arithmetic does not need a prime, so even and composite moduli appear
//! too.
use jali::{
    Error,
    codec::{BitReader, BitWriter},
    dcompress::Compression,
    math::{I256, Poly, PolyVec, Ring, U256},
    params::moduli::NTT_PRIMES,
    rand::{ByteStream, gaussian, uniform, uniform_u256},
};
use num_bigint::{BigInt, Sign};
use proptest::{prelude::*, test_runner::RngSeed};
use std::sync::Arc;

mod common;
use common::{
    SplitMix64, from_u64_words, narrow,
    params::{power_minus, power_plus},
};

fn big(x: &U256) -> BigInt {
    BigInt::from_bytes_le(Sign::Plus, &x.to_le_bytes())
}
/// The canonical representative of `x` modulo `q`.
fn wide(x: &BigInt, q: &BigInt) -> U256 {
    let r = ((x % q) + q) % q;
    let (_, mut bytes) = r.to_bytes_le();
    bytes.resize(32, 0);
    U256::from_le_slice(&bytes)
}
fn centred(x: &BigInt, q: &BigInt) -> BigInt {
    let r = ((x % q) + q) % q;
    if &r * 2 > *q { r - q } else { r }
}
fn random_u256(rng: &mut SplitMix64) -> U256 {
    from_u64_words([rng.next(), rng.next(), rng.next(), rng.next()])
}
fn random_poly(ring: &Arc<Ring>, rng: &mut SplitMix64) -> Poly {
    Poly::from_u256(
        ring.clone(),
        (0..ring.degree()).map(|_| random_u256(rng)).collect(),
    )
    .unwrap()
}
fn values(p: &Poly) -> Vec<BigInt> {
    p.coefficients().iter().map(big).collect()
}
/// Schoolbook product in $`\mathbb Z_q[X]/(X^d+1)`$, canonical.
fn negacyclic(a: &[BigInt], b: &[BigInt], q: &BigInt) -> Vec<BigInt> {
    let d = a.len();
    let mut out = vec![BigInt::from(0); d];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            let t = x * y;
            if i + j < d {
                out[i + j] += t;
            } else {
                out[i + j - d] -= t;
            }
        }
    }
    out.iter().map(|x| ((x % q) + q) % q).collect()
}
fn canonical(v: &[BigInt], q: &BigInt) -> Vec<BigInt> {
    v.iter().map(|x| ((x % q) + q) % q).collect()
}

/// Proven primes 5 mod 8, the maximum modulus, and even and composite moduli.
fn moduli() -> Vec<(&'static str, U256)> {
    vec![
        ("2^100-99", power_minus(100, 99)),
        ("2^100+1533", power_plus(100, 1533)),
        ("2^127-1", power_minus(127, 1)),
        ("2^128-275", power_minus(128, 275)),
        ("2^128", power_plus(128, 0)),
        ("2^128+165", power_plus(128, 165)),
        ("2^200-75", power_minus(200, 75)),
        ("2^240+325", power_plus(240, 325)),
        ("2^255", power_plus(255, 0)),
        ("2^255-19", power_minus(255, 19)),
        ("2^256-435", power_minus(256, 435)),
        ("2^256-1", U256::MAX),
    ]
}

/// Products, sums, differences, negation, scaling, the automorphism, rotation, the trace and
/// the NTT round trip, at degrees 64 and 128 for every modulus.
fn check_ring(label: &str, q: U256, d: usize, rng: &mut SplitMix64) {
    let ring = Ring::with_modulus(q, d).unwrap();
    let qb = big(&q);
    let a = random_poly(&ring, rng);
    let b = random_poly(&ring, rng);
    let (va, vb) = (values(&a), values(&b));
    let context = format!("{label} d={d}");
    assert_eq!(
        values(&a.mul(&b).unwrap()),
        negacyclic(&va, &vb, &qb),
        "{context}"
    );
    let sum: Vec<BigInt> = va.iter().zip(&vb).map(|(x, y)| x + y).collect();
    let difference: Vec<BigInt> = va.iter().zip(&vb).map(|(x, y)| x - y).collect();
    let negation: Vec<BigInt> = va.iter().map(|x| -x).collect();
    assert_eq!(
        values(&a.add(&b).unwrap()),
        canonical(&sum, &qb),
        "{context}"
    );
    assert_eq!(
        values(&a.sub(&b).unwrap()),
        canonical(&difference, &qb),
        "{context}"
    );
    assert_eq!(values(&a.neg()), canonical(&negation, &qb), "{context}");
    // Scalars: random, the special cases 0, 1, q - 1, and the i128 extremes.
    let s = random_u256(rng);
    let scaled: Vec<BigInt> = va.iter().map(|x| x * big(&s)).collect();
    assert_eq!(
        values(&a.scale_u256(&s).unwrap()),
        canonical(&scaled, &qb),
        "{context}"
    );
    for scalar in [0, 1, -1, i128::MIN, i128::MAX, 7] {
        let scaled: Vec<BigInt> = va.iter().map(|x| x * BigInt::from(scalar)).collect();
        assert_eq!(
            values(&a.scale(scalar).unwrap()),
            canonical(&scaled, &qb),
            "{context} {scalar}"
        );
        assert_eq!(
            a.scale(scalar).unwrap(),
            a.mul(&Poly::constant(ring.clone(), scalar)).unwrap(),
            "{context} {scalar}"
        );
    }
    let minus_one = q.wrapping_sub(&U256::ONE);
    assert_eq!(a.scale_u256(&minus_one).unwrap(), a.neg(), "{context}");
    // sigma(a)_0 = a_0 and sigma(a)_i = -a_{d-i}.
    let auto: Vec<BigInt> = (0..d)
        .map(|i| {
            if i == 0 {
                va[0].clone()
            } else {
                -va[d - i].clone()
            }
        })
        .collect();
    assert_eq!(values(&a.auto()), canonical(&auto, &qb), "{context}");
    assert_eq!(a.auto().auto(), a, "{context}");
    assert_eq!(
        a.auto().mul(&b.auto()).unwrap(),
        a.mul(&b).unwrap().auto(),
        "{context}"
    );
    // X^13 a: a shift with sign changes on wrap-around.
    let rotated: Vec<BigInt> = (0..d)
        .map(|i| {
            if i >= 13 {
                va[i - 13].clone()
            } else {
                -va[i + d - 13].clone()
            }
        })
        .collect();
    assert_eq!(values(&a.rotate(13)), canonical(&rotated, &qb), "{context}");
    assert_eq!(a.rotate(d as i64), a.neg(), "{context}");
    assert_eq!(a.rotate(-13).rotate(13), a, "{context}");
    if bool::from(q.is_odd()) {
        // Trace (a + sigma(a))/2: with 2^{-1} = (q + 1)/2.
        let inverse_two = (&qb + 1) / 2;
        let trace: Vec<BigInt> = va
            .iter()
            .zip(&auto)
            .map(|(x, y)| (x + y) * &inverse_two)
            .collect();
        assert_eq!(
            values(&a.trace().unwrap()),
            canonical(&trace, &qb),
            "{context}"
        );
    } else {
        assert!(a.trace().is_err());
    }
    // CRT round trip and accumulation of products.
    assert_eq!(a.to_ntt().to_poly().unwrap(), a, "{context}");
    let c = random_poly(&ring, rng);
    let mut sum = a.to_ntt().product(&b.to_ntt()).unwrap();
    sum.add_assign(&c.to_ntt().product(&a.to_ntt()).unwrap())
        .unwrap();
    assert_eq!(
        sum.reduce().unwrap(),
        a.mul(&b).unwrap().add(&c.mul(&a).unwrap()).unwrap(),
        "{context}"
    );
}

#[test]
fn ring_arithmetic_matches_big_integers_at_degrees_64_and_128() {
    let mut rng = SplitMix64(0x2567_2401);
    for (label, q) in moduli() {
        for d in [64, 128] {
            check_ring(label, q, d, &mut rng);
        }
    }
}

#[test]
fn ring_arithmetic_matches_big_integers_at_degree_1024() {
    let mut rng = SplitMix64(0x2567_1024);
    for (label, q) in [
        ("2^100+1533", power_plus(100, 1533)),
        ("2^128+165", power_plus(128, 165)),
        ("2^200-75", power_minus(200, 75)),
        ("2^240+325", power_plus(240, 325)),
        ("2^256-435", power_minus(256, 435)),
    ] {
        check_ring(label, q, 1024, &mut rng);
    }
}

/// A value of exactly `bits` bits, from random words.
fn exact_bits(words: [u64; 4], bits: u32) -> U256 {
    let x = from_u64_words(words);
    let x = match bits {
        256 => x,
        _ => x.bitand(&U256::ONE.shl_vartime(bits).wrapping_sub(&U256::ONE)),
    };
    x.bitor(&U256::ONE.shl_vartime(bits - 1))
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 48,
        failure_persistence: None,
        rng_seed: RngSeed::Fixed(41),
        ..ProptestConfig::default()
    })]
    /// Random moduli of 100 to 256 bits, odd and even, at degrees 64 and 128: products, sums,
    /// differences, scaling and the NTT round trip against the schoolbook oracle.
    #[test]
    fn products_match_big_integers_at_random_wide_moduli(
        bits in 100u32..=256,
        words in any::<[u64; 4]>(),
        seed in any::<u64>(),
        wide_degree in any::<bool>(),
    ) {
        let d = if wide_degree { 128 } else { 64 };
        let q = exact_bits(words, bits);
        let ring = Ring::with_modulus(q, d).unwrap();
        let qb = big(&q);
        let mut rng = SplitMix64(seed);
        let a = random_poly(&ring, &mut rng);
        let b = random_poly(&ring, &mut rng);
        let (va, vb) = (values(&a), values(&b));
        prop_assert_eq!(values(&a.mul(&b).unwrap()), negacyclic(&va, &vb, &qb));
        let sum: Vec<BigInt> = va.iter().zip(&vb).map(|(x, y)| x + y).collect();
        prop_assert_eq!(values(&a.add(&b).unwrap()), canonical(&sum, &qb));
        let difference: Vec<BigInt> = va.iter().zip(&vb).map(|(x, y)| x - y).collect();
        prop_assert_eq!(values(&a.sub(&b).unwrap()), canonical(&difference, &qb));
        let s = random_u256(&mut rng);
        let scaled: Vec<BigInt> = va.iter().map(|x| x * big(&s)).collect();
        prop_assert_eq!(values(&a.scale_u256(&s).unwrap()), canonical(&scaled, &qb));
        prop_assert_eq!(a.to_ntt().to_poly().unwrap(), a);
    }
}

#[test]
fn rns_prime_counts_follow_the_capacity_rule() {
    // P > (q - 1)^2 d 128, computed in Python from the nine NTT primes.
    for (q, counts) in [
        (power_minus(100, 99), [4, 4, 4, 4, 4]),
        (power_minus(115, 1), [4, 4, 4, 4, 4]),
        (power_plus(116, 0), [4, 4, 4, 5, 5]),
        (power_minus(126, 203), [5, 5, 5, 5, 5]),
        (power_plus(128, 165), [5, 5, 5, 5, 5]),
        (power_minus(200, 75), [7, 7, 7, 7, 7]),
        (power_plus(240, 325), [8, 8, 8, 9, 9]),
        (power_minus(256, 435), [9, 9, 9, 9, 9]),
        (U256::MAX, [9, 9, 9, 9, 9]),
    ] {
        for (d, count) in [64, 128, 256, 512, 1024].into_iter().zip(counts) {
            assert_eq!(Ring::with_modulus(q, d).unwrap().rns_primes(), count);
        }
    }
}

#[test]
fn accumulation_at_the_largest_moduli_holds_its_capacity() {
    // 128 products of all-(q-1)/2 polynomials at degree 1024 reach the limit of 128 products,
    // not the capacity bound: their centred values leave about 30 bits of the CRT range unused
    // (see the next test for the tight case). The 129th addition is refused and leaves the sum
    // unchanged.
    for q in [power_minus(256, 435), U256::MAX] {
        let ring = Ring::with_modulus(q, 1024).unwrap();
        let half = q.shr_vartime(1);
        let a = Poly::from_u256(ring.clone(), vec![half; 1024]).unwrap();
        let product = a.to_ntt().product(&a.to_ntt()).unwrap();
        let mut sum = a.to_ntt().product(&a.to_ntt()).unwrap();
        for _ in 1..128 {
            sum.add_assign(&product).unwrap();
        }
        let want = a.mul(&a).unwrap().scale(128).unwrap();
        assert_eq!(sum.reduce().unwrap(), want);
        assert_eq!(sum.add_assign(&product), Err(Error::Overflow));
        assert_eq!(sum.reduce().unwrap(), want);
        // The oracle for one coefficient: sum_j (q-1)^2/4 * sign, times 128.
        let qb = big(&q);
        let h = big(&half);
        let va = vec![h.clone(); 1024];
        let square = negacyclic(&va, &va, &qb);
        let scaled: Vec<BigInt> = square.iter().map(|x| x * 128).collect();
        assert_eq!(values(&want), canonical(&scaled, &qb));
    }
}

/// The largest modulus that selects `k` primes at degree `d`. Rings take primes until
/// $`P_k>(q-1)^2\cdot128d+1`$, so this is $`q=1+\lfloor\sqrt{(P_k-2)/(128d)}\rfloor`$.
fn largest_modulus_with_primes(k: usize, d: usize) -> BigInt {
    let product: BigInt = NTT_PRIMES[..k]
        .iter()
        .map(|(p, _)| BigInt::from(*p))
        .product();
    let quotient: BigInt = (product - 2) / BigInt::from(128 * d);
    quotient.sqrt() + 1
}

#[test]
fn accumulation_is_exact_where_the_capacity_is_tight() {
    // At the largest modulus of each prime count, 128 products of centred values reach at most
    // about P/4, one bit inside the reconstruction range |x| <= (P - 1)/2; canonical values
    // would reach about P. The all-(q-1) polynomial is -1 centred: its square has coefficient
    // 2j + 2 - d at j, and 128 squares have 128 (2j + 2 - d) modulo q, through the accumulator
    // and through `PolyVec::dot`.
    let mut rings = 0;
    for d in [64, 128, 256, 512, 1024] {
        for k in 1..=8 {
            let qb = largest_modulus_with_primes(k, d);
            let q = wide(&qb, &(BigInt::from(1) << 256));
            let ring = Ring::with_modulus(q, d).unwrap();
            assert_eq!(ring.rns_primes(), k, "q={qb} d={d}");
            let next = Ring::with_modulus(q.wrapping_add(&U256::ONE), d).unwrap();
            assert_eq!(next.rns_primes(), k + 1, "q={qb} d={d}");
            let a = Poly::from_u256(ring.clone(), vec![q.wrapping_sub(&U256::ONE); d]).unwrap();
            let expected: Vec<BigInt> = (0..d as i64)
                .map(|j| big(&wide(&BigInt::from(128 * (2 * j + 2 - d as i64)), &qb)))
                .collect();
            let product = a.to_ntt().product(&a.to_ntt()).unwrap();
            let mut sum = a.to_ntt().product(&a.to_ntt()).unwrap();
            for _ in 1..128 {
                sum.add_assign(&product).unwrap();
            }
            assert_eq!(values(&sum.reduce().unwrap()), expected, "q={qb} d={d}");
            let vector = PolyVec::new(ring.clone(), vec![a; 128]).unwrap();
            assert_eq!(
                values(&vector.dot(&vector).unwrap()),
                expected,
                "q={qb} d={d}"
            );
            rings += 1;
        }
    }
    assert_eq!(rings, 40);
}

#[test]
fn constructors_and_narrow_accessors() {
    let q = power_plus(240, 325);
    let qb = big(&q);
    let ring = Ring::with_modulus(q, 64).unwrap();
    // Values at or above q are reduced, and i128 extremes land on their residues.
    let mut inputs = vec![U256::ZERO; 64];
    inputs[0] = q;
    inputs[1] = U256::MAX;
    inputs[2] = q.wrapping_add(&U256::from_u8(5));
    let p = Poly::from_u256(ring.clone(), inputs).unwrap();
    assert_eq!(p.coefficients()[0], U256::ZERO);
    assert_eq!(p.coefficients()[1], wide(&big(&U256::MAX), &qb));
    assert_eq!(p.coefficients()[2], U256::from_u8(5));
    let mut small = vec![0i128; 64];
    small[0] = i128::MIN;
    small[1] = i128::MAX;
    small[2] = -1;
    let p = Poly::new(ring.clone(), small).unwrap();
    assert_eq!(p.coefficients()[0], wide(&BigInt::from(i128::MIN), &qb));
    assert_eq!(p.coefficients()[1], wide(&BigInt::from(i128::MAX), &qb));
    assert_eq!(p.coefficients()[2], q.wrapping_sub(&U256::ONE));
    // Centred values that fit i128 come back; others fail with Overflow.
    let values = p.coefficients_i128().unwrap();
    assert_eq!(
        (values[0], values[1], values[2]),
        (i128::MIN, i128::MAX, -1)
    );
    // -2^127 is i128::MIN, which fits; 2^127 does not.
    let mut exact = Poly::zero(ring.clone());
    exact.set_coefficient(0, i128::MIN).unwrap();
    assert_eq!(exact.coefficient_i128(0), Ok(i128::MIN));
    let big_value = Poly::constant_u256(ring.clone(), &U256::ONE.shl_vartime(127));
    assert_eq!(big_value.coefficient_i128(0), Err(Error::Overflow));
    assert_eq!(big_value.coefficients_i128().err(), Some(Error::Overflow));
    assert_eq!(big_value.coefficient_i128(1), Ok(0));
    assert_eq!(big_value.coefficient_i128(64), Err(Error::Index));
    let negative = Poly::constant_u256(ring.clone(), &q.wrapping_sub(&U256::ONE.shl_vartime(127)));
    assert_eq!(negative.coefficient_i128(0), Ok(i128::MIN));
    assert_eq!(
        Poly::from_u256(ring.clone(), vec![U256::ZERO; 63]).err(),
        Some(Error::Dimension)
    );
    // Below 2^127 every centred value fits.
    let narrow_ring = Ring::with_modulus(power_minus(127, 1), 64).unwrap();
    let mut rng = SplitMix64(9);
    let p = random_poly(&narrow_ring, &mut rng);
    let qn = big(&narrow_ring.modulus());
    let expected: Vec<i128> = values_of(&p)
        .iter()
        .map(|x| i128::try_from(centred(x, &qn)).unwrap())
        .collect();
    assert_eq!(p.coefficients_i128().unwrap().to_vec(), expected);
}
fn values_of(p: &Poly) -> Vec<BigInt> {
    values(p)
}

#[test]
fn norms_are_exact_below_2_256_and_saturate_above() {
    let q = power_minus(256, 435);
    let qb = big(&q);
    let ring = Ring::with_modulus(q, 1024).unwrap();
    let half = q.shr_vartime(1);
    let full = Poly::from_u256(ring.clone(), vec![half; 1024]).unwrap();
    assert_eq!(full.norm_squared(), U256::MAX);
    assert_eq!(full.norm_infinity(), half);
    let pair = PolyVec::new(ring.clone(), vec![full.clone(), full.neg()]).unwrap();
    assert_eq!(pair.norm_squared(), Ok(U256::MAX));
    assert_eq!(full.neg().norm_infinity(), half);
    // A coefficient of 2^128 alone has norm 2^256; one below 2^128 has an exact norm.
    let at = Poly::constant_u256(ring.clone(), &U256::ONE.shl_vartime(128));
    assert_eq!(at.norm_squared(), U256::MAX);
    let below = Poly::constant_u256(ring.clone(), &U256::from_u128(u128::MAX)).neg();
    let square = BigInt::from(u128::MAX) * BigInt::from(u128::MAX);
    assert_eq!(big(&below.norm_squared()), square);
    // Two such coefficients sum to above 2^256.
    let two = below.add(&below.rotate(1)).unwrap();
    assert_eq!(two.norm_squared(), U256::MAX);
    // Squares of magnitudes below 2^64 are taken in u128: magnitudes on either side of 2^64
    // and of 2^65, of either sign, alone and summed in one polynomial.
    let mut all = Poly::zero(ring.clone());
    let mut sum = BigInt::from(0);
    for (i, m) in [u128::from(u64::MAX), 1 << 64, (1 << 65) - 1, 1 << 65]
        .into_iter()
        .enumerate()
    {
        let x = Poly::constant_u256(ring.clone(), &U256::from_u128(m));
        for (j, x) in [x.clone(), x.neg()].into_iter().enumerate() {
            assert_eq!(big(&x.norm_squared()), BigInt::from(m).pow(2), "{m}");
            assert_eq!(x.norm_infinity(), U256::from_u128(m), "{m}");
            all = all.add(&x.rotate(2 * i as i64 + j as i64)).unwrap();
            sum += BigInt::from(m).pow(2);
        }
    }
    assert_eq!(big(&all.norm_squared()), sum);
    // Random short vectors: exact against the oracle.
    let mut rng = SplitMix64(11);
    let short: Vec<i128> = (0..1024)
        .map(|_| (rng.next() >> 1) as i128 - (1i128 << 62))
        .collect();
    let p = Poly::new(ring.clone(), short.clone()).unwrap();
    let norm: BigInt = short
        .iter()
        .map(|x| BigInt::from(*x) * BigInt::from(*x))
        .sum();
    assert_eq!(big(&p.norm_squared()), norm);
    let max = short.iter().map(|x| x.unsigned_abs()).max().unwrap();
    assert_eq!(p.norm_infinity(), U256::from_u128(max));
    // A random polynomial: the centred norm exceeds 2^256 almost surely; compare with the
    // oracle's saturation.
    let r = random_poly(&ring, &mut rng);
    let norm: BigInt = values(&r).iter().map(|x| centred(x, &qb).pow(2)).sum();
    let expected = if norm < BigInt::from(1) << 256 {
        wide(&norm, &(BigInt::from(1) << 256))
    } else {
        U256::MAX
    };
    assert_eq!(r.norm_squared(), expected);
}

/// A signed 256-bit value from a big integer of magnitude below 2^255.
fn to_i256(x: &BigInt) -> I256 {
    let magnitude = *wide(
        &BigInt::from(x.magnitude().clone()),
        &(BigInt::from(1) << 256),
    )
    .as_int();
    if x.sign() == Sign::Minus {
        magnitude.wrapping_neg()
    } else {
        magnitude
    }
}
/// A big integer from a signed 256-bit value.
fn signed(x: &I256) -> BigInt {
    let (magnitude, negative) = x.abs_sign();
    let m = big(&magnitude);
    if bool::from(negative) { -m } else { m }
}

/// Power2Round, Decompose (with the wrap of Fig. 17), MakeHint and UseHint against their
/// definitions over the integers, on random inputs and boundary values, for one modulus,
/// divisor and rounding exponent.
fn check_compression(q: U256, gamma: u64, d_bits: u32, seed: u64) {
    let qb = big(&q);
    let context = format!("q={qb} gamma={gamma} D={d_bits}");
    let c = Compression::with_modulus(q, U256::from_u64(gamma), d_bits).unwrap();
    let m = big(&c.hint_modulus());
    assert_eq!(&m * BigInt::from(gamma), &qb - 1, "{context}");
    let g = BigInt::from(gamma);
    let power = BigInt::from(1) << d_bits;
    let mut rng = SplitMix64(seed);
    let mut inputs: Vec<U256> = (0..200).map(|_| random_u256(&mut rng)).collect();
    // Boundaries: 0, q - 1, the wrap region below q - 1, multiples of gamma and 2^D.
    for r in [
        BigInt::from(0),
        &qb - 1,
        &qb - 1 - &g / 2,
        &qb - &g / 2,
        &qb - 2,
        g.clone() * 7 + &g / 2,
        g.clone() * 7 + &g / 2 + 1,
        &power / 2,
        &power / 2 + 1,
        power.clone() * 3 - 1,
    ] {
        inputs.push(wide(&r, &(BigInt::from(1) << 256)));
    }
    for r in &inputs {
        let rb = big(r) % &qb;
        let (high, low) = c.power2round(r);
        let (high, low) = (big(&high), signed(&low));
        assert_eq!(&high * &power + &low, rb, "power2round {context}");
        let half_power: BigInt = &power / 2;
        assert!(low > -half_power.clone() || d_bits == 0, "{context}");
        assert!(low <= half_power, "{context}");
        let (high, low) = c.decompose(r);
        let (high, low) = (big(&high), signed(&low));
        // gamma high + low = r modulo q, also for the wrap of Fig. 17, which gives
        // (0, low - 1) where r - low = q - 1.
        let half_gamma: BigInt = &g / 2;
        assert!(
            high < m && low <= half_gamma && low >= -half_gamma.clone(),
            "{context}"
        );
        assert_eq!(
            centred(&(&high * &g + &low - &rb), &qb),
            BigInt::from(0),
            "{context}"
        );
    }
    // MakeHint and UseHint: UseHint(MakeHint(z, r), r) = HighBits(r + z).
    for pair in inputs.chunks(2) {
        let (z, r) = (&pair[0], &pair[1]);
        let hint = c.make_hint(z, r);
        let sum = wide(&(big(z) + big(r)), &qb);
        assert_eq!(
            c.use_hint(&hint, r).unwrap(),
            c.decompose(&sum).0,
            "{context}"
        );
        let h = signed(&hint);
        assert!(&h * 2 <= m && &h * 2 > -&m, "{context}");
    }
    // UseHint refuses hints outside (-m/2, m/2].
    let half: BigInt = &m / 2;
    for (hint, ok) in [
        (half.clone(), true),
        (&half + 1, false),
        (&half + 1 - &m, true),
        (&half - &m, false),
    ] {
        assert_eq!(
            c.use_hint(&to_i256(&hint), &U256::ZERO).is_ok(),
            ok,
            "{context} hint {hint}"
        );
    }
}

#[test]
fn compression_matches_its_definition_at_241_bits() {
    let q = power_plus(240, 325);
    // An even divisor of q - 1 = 2^2 5^2 13 2393 2593 ... and several rounding exponents.
    let gamma = 2 * 5 * 13 * 2393 * 2593u64;
    for d_bits in [0u32, 1, 7, 17, 64, 200, 240] {
        check_compression(q, gamma, d_bits, u64::from(d_bits) + 40);
    }
    // Parameter refusals: even q, odd gamma, gamma not dividing q - 1, 2^D >= q.
    let even = power_plus(240, 326);
    assert!(Compression::with_modulus(even, U256::from_u64(gamma), 1).is_err());
    assert!(Compression::with_modulus(q, U256::from_u64(gamma + 1), 1).is_err());
    assert!(Compression::with_modulus(q, U256::from_u64(2 * 3 * 7), 1).is_err());
    assert!(Compression::with_modulus(q, U256::from_u64(gamma), 241).is_err());
    assert!(Compression::with_modulus(q, U256::from_u64(gamma), 256).is_err());
    Compression::with_modulus(q, U256::from_u64(gamma), 240).unwrap();
    // At the top of the range: q = 2^256 - 435 and D = 255.
    let top = power_minus(256, 435);
    let c = Compression::with_modulus(top, U256::from_u64(4), 255).unwrap();
    let (high, low) = c.power2round(&top.wrapping_sub(&U256::ONE));
    assert_eq!(
        big(&high) * (BigInt::from(1) << 255) + signed(&low),
        big(&top) - 1
    );
}

#[test]
fn compression_matches_its_definition_at_every_wide_prime() {
    // For each prime, gamma is the product of the factors of q - 1 below 10^6 (Sage, trial
    // division); `check_compression` confirms that gamma m = q - 1. Rounding exponents from 0 to
    // the largest with 2^D < q.
    for (q, gamma) in [
        (power_minus(100, 99), 6376312846088724),
        (power_plus(100, 1533), 6822801252),
        (power_minus(128, 275), 21020),
        (power_plus(128, 165), 75418140),
        (power_minus(200, 75), 326517756279300),
        (power_plus(240, 325), 806656370),
        (power_minus(255, 19), 781764),
        (power_minus(256, 435), 325500),
    ] {
        let top = q.bits_vartime() - 1;
        for d_bits in [0, 1, 13, 64, top] {
            check_compression(q, gamma, d_bits, u64::from(top) * 1000 + u64::from(d_bits));
        }
        assert!(Compression::with_modulus(q, U256::from_u64(gamma), top + 1).is_err());
        // 3 gamma does not divide q - 1, whose cofactor (q - 1)/gamma has no prime factor below
        // 10^6, and is refused.
        assert!(Compression::with_modulus(q, U256::from_u64(3 * gamma), 1).is_err());
    }
}

#[test]
fn wide_codes_round_trip_and_refuse_values_outside_their_range() {
    let mut rng = SplitMix64(21);
    for modulus in [
        power_plus(128, 0),
        power_plus(128, 165),
        power_plus(240, 325),
        power_minus(256, 435),
        U256::MAX,
    ] {
        let bits = modulus.wrapping_sub(&U256::ONE).bits_vartime();
        let mut values: Vec<U256> = (0..64)
            .map(|_| random_u256(&mut rng).rem_vartime(&modulus.to_nz().unwrap()))
            .collect();
        values.extend([U256::ZERO, modulus.wrapping_sub(&U256::ONE)]);
        let mut w = BitWriter::new();
        for v in &values {
            w.uniform_u256(v, &modulus).unwrap();
        }
        assert_eq!(w.uniform_u256(&modulus, &modulus), Err(Error::Encoding));
        let bytes = w.finish();
        assert_eq!(bytes.len(), (values.len() * bits as usize + 1).div_ceil(8));
        let mut r = BitReader::new(&bytes);
        for v in &values {
            assert_eq!(r.uniform_u256(&modulus).unwrap(), *v);
        }
        r.finish().unwrap();
        // The code of the modulus itself, where it fits the width, is refused by the reader.
        if modulus.bits_vartime() <= bits {
            let mut w = BitWriter::new();
            w.unsigned_u256(&modulus, bits).unwrap();
            let bytes = w.finish();
            assert_eq!(
                BitReader::new(&bytes).uniform_u256(&modulus),
                Err(Error::Encoding)
            );
        }
    }
    // unsigned_u256: exactly `bits` bits, up to 256.
    let mut w = BitWriter::new();
    w.unsigned_u256(&U256::MAX, 256).unwrap();
    assert_eq!(w.unsigned_u256(&U256::ONE, 257), Err(Error::Encoding));
    assert_eq!(w.unsigned_u256(&U256::from_u8(4), 2), Err(Error::Encoding));
    let bytes = w.finish();
    let mut r = BitReader::new(&bytes);
    assert_eq!(r.unsigned_u256(256).unwrap(), U256::MAX);
    assert_eq!(r.unsigned_u256(257), Err(Error::Encoding));
    assert!(BitReader::new(&[]).uniform_u256(&U256::ONE).is_err());
    // Below 2^128 the 128-bit and 256-bit codes write the same bits.
    for m in [2u128, 13, 1 << 64, u128::MAX] {
        let mut a = BitWriter::new();
        let mut b = BitWriter::new();
        for _ in 0..50 {
            let x = (u128::from(rng.next()) << 64 | u128::from(rng.next())) % m;
            a.uniform(x, m).unwrap();
            b.uniform_u256(&U256::from_u128(x), &U256::from_u128(m))
                .unwrap();
        }
        assert_eq!(a.finish(), b.finish());
    }
}

/// A stream that serves fixed bytes and counts what it serves.
struct Fixed {
    bytes: Vec<u8>,
    read: usize,
}
impl ByteStream for Fixed {
    fn fill(&mut self, output: &mut [u8]) -> Result<(), Error> {
        let end = self.read + output.len();
        output.copy_from_slice(self.bytes.get(self.read..end).ok_or(Error::Randomness)?);
        self.read = end;
        Ok(())
    }
}

#[test]
fn the_wide_sampler_rejects_words_at_or_above_the_modulus() {
    // m = 3 * 2^254: 256-bit words, kept below m. The first pass reads 64 bytes for two
    // words: all ones (refused) and 5 (kept). The second pass reads 32 bytes for one word:
    // m itself (refused). The third reads m - 1 (kept).
    let m = U256::from_u8(3).shl_vartime(254);
    let mut bytes = vec![0xff; 32];
    bytes.extend(U256::from_u8(5).to_le_bytes().iter());
    bytes.extend(m.to_le_bytes().iter());
    bytes.extend(m.wrapping_sub(&U256::ONE).to_le_bytes().iter());
    let mut stream = Fixed { bytes, read: 0 };
    let values = uniform_u256(&mut stream, &m, 2).unwrap();
    assert_eq!(values, vec![U256::from_u8(5), m.wrapping_sub(&U256::ONE)]);
    assert_eq!(stream.read, 128);
    // 129-bit words pack across byte boundaries: two words take 33 bytes.
    let m = power_plus(128, 165);
    let mut stream = Fixed {
        bytes: vec![0; 33],
        read: 0,
    };
    assert_eq!(
        uniform_u256(&mut stream, &m, 2).unwrap(),
        vec![U256::ZERO; 2]
    );
    assert_eq!(stream.read, 33);
    // Below 2^128 the wide sampler reads what `uniform` reads.
    let mut a = Fixed {
        bytes: (0..4096).map(|i| (i * 37 % 251) as u8).collect(),
        read: 0,
    };
    let mut b = Fixed {
        bytes: a.bytes.clone(),
        read: 0,
    };
    let narrow_values: Vec<U256> = uniform(&mut a, 1 << 100, 40)
        .unwrap()
        .into_iter()
        .map(U256::from_u128)
        .collect();
    assert_eq!(
        uniform_u256(&mut b, &U256::ONE.shl_vartime(100), 40).unwrap(),
        narrow_values
    );
    assert_eq!(a.read, b.read);
    // An exhausted stream fails without a value.
    let mut empty = Fixed {
        bytes: vec![],
        read: 0,
    };
    assert_eq!(
        uniform_u256(&mut empty, &m, 1).err(),
        Some(Error::Randomness)
    );
}

#[test]
fn the_wide_sampler_is_uniform_on_a_fixed_seed() {
    // m = 3 * 2^254: the top two bits of a uniform value in [0, m) are 00, 01 or 10, each
    // with probability 1/3, and the value is odd with probability 1/2. Chi-square statistics
    // on 30,000 values from one fixed seed; the thresholds have p-values below 1e-5 (2 and
    // 1 degrees of freedom: 23.0 and 19.5). A regression check of the rejection rule, not a
    // proof of uniformity.
    let m = U256::from_u8(3).shl_vartime(254);
    let values = uniform_u256(&mut jali::rand::AesPrg::new(&[5; 32], 1), &m, 30_000).unwrap();
    let mut top = [0f64; 4];
    let mut odd = 0f64;
    for v in &values {
        assert!(v < &m);
        top[usize::from(v.shr_vartime(254).to_le_bytes()[0])] += 1.0;
        odd += f64::from(u8::from(bool::from(v.is_odd())));
    }
    assert_eq!(top[3], 0.0);
    let expected = 10_000.0;
    let chi: f64 = top[..3]
        .iter()
        .map(|o| (o - expected).powi(2) / expected)
        .sum();
    assert!(chi < 23.0, "top bits {top:?}, chi-square {chi}");
    let chi = (odd - 15_000.0).powi(2) / 15_000.0 * 2.0;
    assert!(chi < 19.5, "odd {odd}, chi-square {chi}");
}

#[test]
fn gaussians_up_to_the_width_cap_round_trip_through_their_code() {
    // Widths 1.55 * 2^t with t = 64, 72, 100 (the cap): 256 samples each round-trip through
    // the Gaussian code; t = 101 is refused by the sampler and both codec directions.
    for t in [64u32, 72, 100] {
        let samples =
            gaussian(&mut jali::rand::AesPrg::new(&[6; 32], u64::from(t)), t, 256).unwrap();
        let mut w = BitWriter::new();
        for x in &samples {
            w.gaussian(*x, t, 1 << 20).unwrap();
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for x in &samples {
            assert_eq!(r.gaussian(t, 1 << 20).unwrap(), *x);
        }
        r.finish().unwrap();
        assert!(samples.iter().any(|x| x.unsigned_abs() > 1u128 << (t - 1)));
    }
    assert!(gaussian(&mut jali::rand::AesPrg::new(&[6; 32], 0), 101, 1).is_err());
    assert!(BitWriter::new().gaussian(0, 101, 1).is_err());
    assert!(BitReader::new(&[0xff]).gaussian(101, 1).is_err());
    assert!(jali::rand::reject::Variance::gaussian(101).is_err());
    jali::rand::reject::Variance::gaussian(100).unwrap();
}

#[test]
fn centred_values_of_wide_rings_convert_back_and_forth() {
    // The centred representative of every canonical value, through the i128 accessor where it
    // fits and through negation symmetry otherwise.
    let mut rng = SplitMix64(31);
    for (_, q) in moduli() {
        let qb = big(&q);
        let ring = Ring::with_modulus(q, 64).unwrap();
        let p = random_poly(&ring, &mut rng);
        for (i, x) in values(&p).iter().enumerate() {
            let c = centred(x, &qb);
            match p.coefficient_i128(i) {
                Ok(v) => assert_eq!(BigInt::from(v), c),
                Err(e) => {
                    assert_eq!(e, Error::Overflow);
                    assert!(c.bits() > 127);
                }
            }
            // -x centres to -c, except the even-modulus value q/2, which is its own negative.
            let negated = centred(&big(&p.neg().coefficients()[i]), &qb);
            if &c * 2 == qb {
                assert_eq!(negated, c);
            } else {
                assert_eq!(negated, -c);
            }
        }
        assert_eq!(narrow(&U256::from_u64(7)), 7);
    }
}
