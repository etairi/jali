use jali::{
    Error,
    dcompress::Compression,
    math::{
        I256, Poly, PolyVec, Ring, SparsePolyMat, SparsePolyVec, U256, int::is_prime, iso,
        ntt::NttPlan,
    },
    params::{TboxParams, moduli::NTT_PRIMES, toy_d64},
    quad::QuadEq,
    statement,
};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use proptest::prelude::*;

mod common;
use common::{
    SplitMix64, narrow,
    params::{next_prime_5_mod_8, prev_prime_5_mod_8},
    ring::poly,
    values,
};

fn reference(a: &[i128], b: &[i128], q: i128) -> Vec<i128> {
    let d = a.len();
    let mut out = vec![BigInt::from(0); d];
    for i in 0..d {
        for j in 0..d {
            let term = BigInt::from(a[i]) * BigInt::from(b[j]);
            if i + j < d {
                out[i + j] += term;
            } else {
                out[i + j - d] -= term;
            }
        }
    }
    out.into_iter()
        .map(|x| {
            let v = ((x % q + q) % q).to_i128().unwrap();
            if v > q / 2 { v - q } else { v }
        })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    #[test]
    fn rns_products_match_big_integer_oracle(a in prop::collection::vec(any::<i128>(),64), b in prop::collection::vec(any::<i128>(),64), bits in 30u32..100) {
        let q = (1i128<<bits)-1; let ring = Ring::new(q,64).unwrap();
        let a = Poly::new(ring.clone(),a).unwrap(); let b = Poly::new(ring,b).unwrap();
        prop_assert_eq!(values(&a.mul(&b).unwrap()),reference(&values(&a),&values(&b),q));
        prop_assert_eq!(a.to_ntt().to_poly().unwrap(),a.clone());
        prop_assert_eq!(a.auto().auto(),a.clone());
        prop_assert_eq!(a.auto().mul(&b.auto()).unwrap(),a.mul(&b).unwrap().auto());
        prop_assert_eq!(a.trace().unwrap().coefficients()[0],a.coefficients()[0]);
        prop_assert_eq!(a.trace().unwrap().coefficient_i128(32),Ok(0));
        prop_assert_eq!(a.rotate(64),a.neg());
        prop_assert_eq!(a.rotate(-13).rotate(13),a);
    }
}

#[test]
fn accumulation_enforces_its_integer_capacity() {
    let q = (1i128 << 99) - 17;
    let ring = Ring::new(q, 128).unwrap();
    let a = Poly::new(ring, vec![q / 2; 128]).unwrap();
    let product = a.to_ntt().product(&a.to_ntt()).unwrap();
    let mut sum = a.to_ntt().product(&a.to_ntt()).unwrap();
    for _ in 1..128 {
        sum.add_assign(&product).unwrap();
    }
    assert_eq!(
        sum.reduce().unwrap(),
        a.mul(&a).unwrap().scale(128).unwrap()
    );
    assert_eq!(sum.add_assign(&product), Err(Error::Overflow));
    assert_eq!(
        sum.reduce().unwrap(),
        a.mul(&a).unwrap().scale(128).unwrap()
    );
}

#[test]
fn scalar_fast_paths_agree_with_ntt_and_big_integer_products() {
    for q in [
        13,
        1i128 << 63,
        (1i128 << 64) - 1,
        1i128 << 64,
        (1i128 << 99) - 17,
    ] {
        let ring = Ring::new(q, 64).unwrap();
        let a = Poly::new(ring.clone(), (0..64).map(|i| q / 2 - i).collect()).unwrap();
        for scalar in [0, 1, -1, q / 2, i128::MIN, i128::MAX] {
            let b = Poly::constant(ring.clone(), scalar);
            let want = a.to_ntt().product(&b.to_ntt()).unwrap().reduce().unwrap();
            assert_eq!(a.mul(&b).unwrap(), want);
            assert_eq!(b.mul(&a).unwrap(), want);
            assert_eq!(a.scale(scalar).unwrap(), want);
            assert_eq!(values(&want), reference(&values(&a), &values(&b), q));
        }
    }
}

#[test]
fn ntt_all_prime_and_degree_combinations() {
    fn pow(mut a: u64, mut e: u64, p: u64) -> u64 {
        let mut v = 1;
        while e > 0 {
            if e & 1 == 1 {
                v = (v as u128 * a as u128 % p as u128) as u64;
            }
            a = (a as u128 * a as u128 % p as u128) as u64;
            e >>= 1;
        }
        v
    }
    for (p, root) in NTT_PRIMES {
        for d in [64, 128, 256, 512, 1024] {
            let plan = NttPlan::new(p, d, pow(root, (1024 / d) as u64, p)).unwrap();
            let a: Vec<u64> = (0..d).map(|i| (i as u64 * 137 + 91) % p).collect();
            let mut transformed = a.clone();
            plan.forward(&mut transformed).unwrap();
            plan.inverse(&mut transformed).unwrap();
            assert_eq!(transformed, a);
            let mut x = vec![0; d];
            x[d - 1] = 1;
            let mut y = vec![0; d];
            y[1] = 1;
            plan.forward(&mut x).unwrap();
            plan.forward(&mut y).unwrap();
            for (a, b) in x.iter_mut().zip(y) {
                *a = (*a as u128 * b as u128 % p as u128) as u64;
            }
            plan.inverse(&mut x).unwrap();
            assert_eq!(x[0], p - 1);
            assert!(x[1..].iter().all(|x| *x == 0));
        }
    }
}

#[test]
fn quadratic_lowering_matches_direct_evaluation_including_square_terms() {
    for d in [128, 256, 512] {
        let source = Ring::new(12289, d).unwrap();
        let target = Ring::new(12289, 64).unwrap();
        let a = Poly::new(source.clone(), (0..d).map(|i| i as i128 * 3 - 60).collect()).unwrap();
        let b = Poly::new(
            source.clone(),
            (0..d).map(|i| (i as i128 * 7) % 19 - 9).collect(),
        )
        .unwrap();
        let coeff = a.rotate(17);
        let x = PolyVec::new(source.clone(), vec![a.clone(), b.clone()]).unwrap();
        let form = SparsePolyMat::new(
            source,
            2,
            vec![
                (0, 0, coeff.clone()),
                (0, 1, coeff.neg()),
                (1, 1, coeff.clone()),
            ],
        )
        .unwrap();
        let mut parts = iso::split(&a, target.clone()).unwrap();
        parts.extend(iso::split(&b, target.clone()).unwrap());
        let lower_x = PolyVec::new(target.clone(), parts).unwrap();
        let expected = iso::split(&form.bilinear(&x, &x).unwrap(), target.clone()).unwrap();
        let forms = iso::quadratic(&form, target.clone()).unwrap();
        for (f, want) in forms.iter().zip(expected) {
            assert_eq!(f.bilinear(&lower_x, &lower_x).unwrap(), want);
        }
        let lower_a =
            PolyVec::new(target.clone(), iso::split(&a, target.clone()).unwrap()).unwrap();
        let matrix = iso::multiplication_matrix(&coeff, target.clone()).unwrap();
        assert_eq!(
            matrix.mul(&lower_a).unwrap().entries(),
            iso::split(&coeff.mul(&a).unwrap(), target.clone()).unwrap()
        );
        assert_eq!(iso::join(lower_a.entries(), a.ring().clone()).unwrap(), a);
    }
}

#[test]
fn quadratic_lowering_from_every_source_degree_to_both_proof_degrees() {
    // Dense random coefficients and a sigma image, for every source degree up to 1024 and both
    // proof degrees, including k = 1; `statement::lower` end to end; and ct(source) =
    // ct(component 0), the identity that `const_coeff_zero` relies on.
    let q = 12289;
    for d in [128, 256, 512, 1024] {
        for target_degree in [64, 128] {
            let source = Ring::new(q, d).unwrap();
            let target = Ring::new(q, target_degree).unwrap();
            let k = d / target_degree;
            let mut rng = SplitMix64((d + target_degree) as u64);
            let a = poly(&source, rng.uniform(d, q));
            let b = poly(&source, rng.ternary(d));
            let coeff = poly(&source, rng.uniform(d, q));
            let x = PolyVec::new(source.clone(), vec![a.clone(), b.clone()]).unwrap();
            let form = SparsePolyMat::new(
                source.clone(),
                2,
                vec![
                    (0, 0, coeff.clone()),
                    (0, 1, coeff.neg()),
                    (1, 1, coeff.auto()),
                ],
            )
            .unwrap();
            let mut parts = iso::split(&a, target.clone()).unwrap();
            parts.extend(iso::split(&b, target.clone()).unwrap());
            let lower_x = PolyVec::new(target.clone(), parts).unwrap();
            let expected = iso::split(&form.bilinear(&x, &x).unwrap(), target.clone()).unwrap();
            let forms = iso::quadratic(&form, target.clone()).unwrap();
            assert_eq!(forms.len(), k);
            for (f, want) in forms.iter().zip(expected) {
                assert_eq!(f.bilinear(&lower_x, &lower_x).unwrap(), want);
            }
            let lower_a =
                PolyVec::new(target.clone(), iso::split(&a, target.clone()).unwrap()).unwrap();
            let matrix = iso::multiplication_matrix(&coeff, target.clone()).unwrap();
            assert_eq!(
                matrix.mul(&lower_a).unwrap().entries(),
                iso::split(&coeff.mul(&a).unwrap(), target.clone()).unwrap()
            );
            let mut equation = QuadEq::zero(source.clone(), 2).unwrap();
            equation.r2 = form;
            equation.r1 = SparsePolyVec::new(
                source.clone(),
                2,
                vec![(0, poly(&source, rng.uniform(d, q))), (1, coeff.rotate(3))],
            )
            .unwrap();
            equation.r0 = poly(&source, rng.uniform(d, q));
            let value = equation.evaluate(&x).unwrap();
            let lowered = statement::lower(&equation, target.clone()).unwrap();
            assert_eq!(lowered.len(), k);
            let values: Vec<Poly> = lowered
                .iter()
                .map(|l| l.evaluate(&lower_x).unwrap())
                .collect();
            assert_eq!(iso::join(&values, source.clone()).unwrap(), value);
            assert_eq!(values[0].coefficients()[0], value.coefficients()[0]);
        }
    }
}

#[test]
fn trace_inner_product_identity() {
    let ring = Ring::new(1099511627917, 64).unwrap();
    let a = Poly::new(ring.clone(), (0..64).map(|x| x - 30).collect()).unwrap();
    let b = a.rotate(11);
    let want: i128 = values(&a).iter().zip(values(&b)).map(|(a, b)| a * b).sum();
    let a = PolyVec::new(ring.clone(), vec![a]).unwrap();
    let b = PolyVec::new(ring, vec![b]).unwrap();
    assert_eq!(a.inner_product(&b).unwrap().coefficient_i128(0), Ok(want));
}

#[test]
fn exhaustive_hint_identity_and_rounding_boundaries() {
    // The functions reduce their inputs modulo q, so r and z range over two periods.
    let signed = |x: &I256| {
        let (magnitude, negative) = x.abs_sign();
        let magnitude = narrow(&magnitude) as i128;
        if bool::from(negative) {
            -magnitude
        } else {
            magnitude
        }
    };
    for q in [13, 29, 101] {
        let reduce = |x: i128| U256::from_u128(x.rem_euclid(q) as u128);
        for gamma in (2..q).step_by(2).filter(|g| (q - 1) % g == 0) {
            let c = Compression::new(q, gamma, 1).unwrap();
            for r in -q..q {
                for z in -q..q {
                    let h = c.make_hint(&reduce(z), &reduce(r));
                    assert_eq!(
                        c.use_hint(&h, &reduce(r)).unwrap(),
                        c.decompose(&reduce(r + z)).0
                    );
                    let (high, low) = c.decompose(&reduce(r));
                    let (high, low) = (narrow(&high) as i128, signed(&low));
                    assert_eq!((gamma * high + low - r).rem_euclid(q), 0);
                    assert!(low >= -gamma / 2 && low <= gamma / 2);
                }
            }
        }
    }
    let c = Compression::new(1099511627917, 65202, 7).unwrap();
    let pair = |(high, low): (U256, I256)| (narrow(&high), signed(&low));
    assert_eq!(pair(c.power2round(&U256::from_u8(64))), (0, 64));
    assert_eq!(pair(c.power2round(&U256::from_u8(65))), (1, -63));
    assert_eq!(pair(c.decompose(&U256::from_u64(1099511627916))), (0, -1));
}

#[test]
fn pinned_reference_parameters_and_mutations() {
    let p = toy_d64();
    #[cfg(feature = "serde")]
    assert_eq!(
        jali::params::TboxParams::from_json(include_str!("../src/params/sets/toy-d64.json"))
            .unwrap(),
        p
    );
    let c = p.check().unwrap();
    assert_eq!((c.lambda, c.n_ex, c.l_ext), (4, 7, 12));
    assert_eq!(
        (
            c.b_squared,
            c.z3_bound_squared,
            c.z4_bound,
            c.estimated_proof_bytes
        ),
        (3487433255883, 1734566565, 50790, 17896)
    );
    // Euclidean bounds of the range vectors: 128 + 64 + (2 + 2) * 64 and 2 * 64 * 4^2. Sized
    // from the second, sigma4 = 1.55 * 2^11 has a rejection constant of 2; sized from the bound
    // 4 alone, sigma4 = 1.55 * 2^8 would need 9.
    assert_eq!(
        (
            c.exact_alpha_squared,
            c.approx_alpha_squared,
            c.approx_extraction_bound
        ),
        (448, 2048, 2 * 50790)
    );
    assert_eq!(jali::params::range_rejection_constant(11, 2048), Ok(2));
    assert_eq!(jali::params::range_rejection_constant(8, 2048), Ok(9));
    assert_eq!(jali::params::range_rejection_constant(8, 16), Ok(2));
    // A range width far below the Euclidean bound would need M >= 2^16: check() refuses it.
    let mut narrow = p.clone();
    narrow.log_sigma[3] = 0;
    assert_eq!(
        narrow.check().err(),
        Some(Error::Parameter("range rejection M"))
    );
    for mutation in 0..12 {
        let mut p = p.clone();
        match mutation {
            0 => p.prime_factors[0] = p.prime_factors[0].wrapping_add(&U256::from_u8(8)),
            1 => p.gamma += 2,
            2 => p.degree = 32,
            3 => p.m1 = 1,
            4 => p.m2 = 12,
            5 => p.mlwe_rank = 1,
            6 => p.mlwe_delta = f64::NAN,
            7 => p.mlwe_delta = 1.005,
            8 => p.log_sigma[2] = 30,
            9 => p.l2_rows[0] = 0,
            10 => p.n_prime = 0,
            _ => p.estimator.clear(),
        }
        assert!(p.check().is_err(), "mutation {mutation}");
    }
}

/// The first even divisor of $`q-1`$ above 256 and at most $`2^{16}`$, if any.
fn small_even_divisor(q: u64) -> Option<u64> {
    (258..=65536)
        .step_by(2)
        .find(|g| (q - 1).is_multiple_of(*g))
}
/// `p` over the prime `q` with D = 0 and a small compression divisor, so that the rounding and
/// MSIS checks pass whatever the divisors of q - 1 are.
fn with_prime(p: &TboxParams, window: impl Fn(u64) -> bool, start: u64) -> TboxParams {
    let mut q = next_prime_5_mod_8(start);
    while window(q) {
        if let Some(gamma) = small_even_divisor(q) {
            let mut p = p.clone();
            p.prime_factors = vec![U256::from_u64(q)];
            p.gamma = gamma;
            p.d_bits = 0;
            return p;
        }
        q = next_prime_5_mod_8(q + 1);
    }
    panic!("no prime in the window");
}

#[test]
fn every_parameter_check_rejects_one_perturbation() {
    // Each case perturbs toy_d64 so that exactly the named check of `TboxParams::check`
    // fires first; where one field cannot reach a check, the case says which others it sets.
    let base = toy_d64();
    let checked = base.check().unwrap();
    let error = |p: &TboxParams| p.check().err();
    let parameter = |s: &'static str| Some(Error::Parameter(s));
    let case = |f: &dyn Fn(&mut TboxParams)| {
        let mut p = base.clone();
        f(&mut p);
        p
    };
    for (name, p) in [
        ("empty id", case(&|p| p.id.clear())),
        ("long id", case(&|p| p.id = "i".repeat(257))),
        ("empty estimator", case(&|p| p.estimator.clear())),
        ("long estimator", case(&|p| p.estimator = "e".repeat(4097))),
    ] {
        assert_eq!(error(&p), parameter("parameter provenance"), "{name}");
    }
    let q = base.prime_factors[0];
    let u = U256::from_u64;
    for (name, p) in [
        ("no factor", case(&|p| p.prime_factors.clear())),
        (
            "three factors",
            case(&|p| p.prime_factors = vec![u(13), u(29), q]),
        ),
        ("1 mod 8", case(&|p| p.prime_factors = vec![u(17)])),
        ("composite 21", case(&|p| p.prime_factors = vec![u(21)])),
        ("descending", case(&|p| p.prime_factors = vec![q, u(13)])),
        ("repeated", case(&|p| p.prime_factors = vec![u(13), u(13)])),
    ] {
        assert_eq!(error(&p), parameter("proof modulus factors"), "{name}");
    }
    // Two primes above 2^128 (2^128 + 165 and 2^128 + 421, both 5 mod 8, proven prime in
    // Sage): the product does not fit the 256-bit modulus type. The product of two 64-bit
    // primes fits.
    let above = |k: u64| U256::ONE.shl_vartime(128).wrapping_add(&U256::from_u64(k));
    assert_eq!(
        error(&case(&|p| p.prime_factors = vec![above(165), above(421)])),
        Some(Error::Overflow)
    );
    assert_eq!(error(&case(&|p| p.degree = 32)), parameter("proof degree"));
    // The ring capacity is the modulus type itself, so no product of accepted factors
    // reaches `Ring::with_modulus`'s own refusal; two primes of about 64 bits pass the modulus
    // checks and fail on the compression divisor instead.
    let top = prev_prime_5_mod_8(u64::MAX);
    let below = prev_prime_5_mod_8(top - 1);
    assert_eq!(
        error(&case(&|p| p.prime_factors = vec![u(below), u(top)])),
        parameter("compression")
    );
    for (name, p) in [
        ("gamma not dividing q - 1", case(&|p| p.gamma += 2)),
        ("odd gamma", case(&|p| p.gamma = 3)),
        ("2^D >= q", case(&|p| p.d_bits = 41)),
    ] {
        assert_eq!(error(&p), parameter("compression"), "{name}");
    }
    // q = 2^k - j with j = 3 mod 8 and j < 2^(D-1): Power2Round(q - 1) rounds up to 2^(k-D),
    // one bit wider than the packed field. The divisor 4 divides q - 1 = 2^k - j - 1.
    let wide = (30..=40)
        .rev()
        .flat_map(|k| (3..64).step_by(8).map(move |j| (1u64 << k) - j))
        .find(|q| is_prime(*q))
        .unwrap();
    assert_eq!(
        error(&case(&|p| {
            p.prime_factors = vec![U256::from_u64(wide)];
            p.gamma = 4;
        })),
        parameter("compressed commitment bit width")
    );
    for (name, p) in [
        ("rows and bounds differ", case(&|p| p.l2_rows.push(1))),
        ("zero rows", case(&|p| p.l2_rows[0] = 0)),
        ("zero bound", case(&|p| p.l2_bounds_squared[1] = 0)),
        (
            "1025 blocks",
            case(&|p| {
                p.l2_rows = vec![1; 1025];
                p.l2_bounds_squared = vec![1; 1025];
            }),
        ),
    ] {
        assert_eq!(error(&p), parameter("exact norm blocks"), "{name}");
    }
    for (name, p) in [
        ("m1 above 65535", case(&|p| p.m1 = 65536)),
        ("n_msis zero", case(&|p| p.n_msis = 0)),
        ("m2 not above n_msis", case(&|p| p.m2 = p.n_msis)),
        ("alpha zero", case(&|p| p.alpha_squared = 0)),
    ] {
        assert_eq!(error(&p), parameter("dimensions"), "{name}");
    }
    assert_eq!(
        error(&case(&|p| p.n_prime = 0)),
        parameter("approximate block")
    );
    assert_eq!(
        error(&case(&|p| p.linf_bound = 0)),
        parameter("approximate block")
    );
    // The Gaussian cap is MAX_LOG_SIGMA = 100.
    for i in 0..4 {
        assert_eq!(
            error(&case(&|p| p.log_sigma[i] = 101)),
            parameter("Gaussian width capacity"),
            "log_sigma[{i}]"
        );
    }
    assert_eq!(
        jali::params::range_rejection_constant(101, 1),
        Err(Error::Parameter("Gaussian width capacity"))
    );
    // Two primes with the divisor and exponent of the two-prime test fixture.
    assert_eq!(
        error(&case(&|p| {
            p.prime_factors = vec![u(1048589), u(4194389)];
            p.gamma = 123280;
            p.d_bits = 9;
        })),
        parameter("binary, exact-norm and range blocks require a prime modulus")
    );
    assert_eq!(
        error(&case(&|p| p.m1 = 1)),
        parameter("completeness dimensions")
    );
    for delta in [1, -1i64] {
        assert_eq!(
            error(&case(
                &|p| p.mlwe_rank = (p.mlwe_rank as i64 + delta) as usize
            )),
            parameter("simulatability rank")
        );
    }
    for delta in [f64::NAN, f64::INFINITY, 1.0045, 0.999] {
        assert_eq!(
            error(&case(&|p| p.mlwe_delta = delta)),
            parameter("MLWE estimate"),
            "{delta}"
        );
    }
    // A 60-bit prime with D = 55: eta 2^(D-1) sqrt(nd) alone exceeds 2^64.
    let prime60 = next_prime_5_mod_8(3 << 58);
    assert_eq!(
        error(&case(&|p| {
            p.prime_factors = vec![u(prime60)];
            p.gamma = 4;
            p.d_bits = 55;
        })),
        parameter("compressed response bound capacity")
    );
    // Half the MSIS rank, with the MLWE rank raised to keep m2.
    assert_eq!(
        error(&case(&|p| {
            p.n_msis = 8;
            p.mlwe_rank += 8;
        })),
        parameter("MSIS estimate")
    );
    // gamma = 4 divides q - 1 but is below 2^(D-1) omega d = 32768.
    assert_eq!(error(&case(&|p| p.gamma = 4)), parameter("rounding bound"));
    // linf_bound^2 * n'd does not fit 128 bits.
    assert_eq!(
        error(&case(&|p| p.linf_bound = u64::MAX)),
        Some(Error::Overflow)
    );
    assert_eq!(
        error(&case(&|p| p.log_sigma[2] = 0)),
        parameter("range rejection M")
    );
    assert_eq!(
        error(&case(&|p| p.log_sigma[3] = 0)),
        parameter("range rejection M")
    );
    // The extraction and lifting inequalities of the exact-range block. The ARP bound grows
    // with sigma_3; at log_sigma[2] = 22 the modulus bound fails, at 17 the binary lifting
    // bound, whose arp_bound^2 then exceeds q.
    assert_eq!(
        error(&case(&|p| p.log_sigma[2] = 22)),
        parameter("ARP modulus bound")
    );
    assert_eq!(
        error(&case(&|p| p.log_sigma[2] = 17)),
        parameter("binary lifting bound")
    );
    // At log_sigma[2] = 16, arp_bound^2 is just below q. Without binary rows, a prime in
    // (arp^2, arp^2 + arp sqrt(d)] fails only the slack lifting bound; with them, a prime above
    // arp^2 + arp sqrt(n_bin d) and a Euclidean bound B with 3B + arp^2 >= q fails only the
    // exact-norm lifting bound. B stays small enough for the range rejection constant.
    let wide_range = case(&|p| p.log_sigma[2] = 16);
    wide_range.check().unwrap();
    let arp = 2.0 * (256.0f64 / 26.0).sqrt() * 1.64 * 1.55 * 2f64.powi(16);
    let d = base.degree as f64;
    let mut slack = wide_range.clone();
    slack.n_bin = 0;
    let slack = with_prime(
        &slack,
        |q| (q as f64) <= arp * arp + arp * d.sqrt(),
        (arp * arp) as u64 + 1,
    );
    assert_eq!(error(&slack), parameter("slack lifting bound"));
    let mut exact = wide_range.clone();
    exact.l2_bounds_squared[0] = 600_000_000;
    exact.check().unwrap();
    let bound = arp * arp + 3.0 * 600_000_000.0;
    let exact = with_prime(
        &exact,
        |q| (q as f64) <= bound,
        (arp * arp + arp * (2.0 * d).sqrt()) as u64 + 1,
    );
    assert_eq!(error(&exact), parameter("exact norm lifting bound"));
    assert!(checked.arp_bound < arp);
}

#[test]
fn incompatible_inputs_return_errors() {
    let a = Ring::new(13, 64).unwrap();
    let b = Ring::new(29, 64).unwrap();
    assert!(matches!(
        Poly::zero(a.clone()).mul(&Poly::zero(b)),
        Err(Error::RingMismatch)
    ));
    // Moduli are the integers from 2 to 2^256 - 1.
    for q in [i128::MIN, -13, 0, 1] {
        assert_eq!(
            Ring::new(q, 64).err(),
            Some(Error::Parameter("ring modulus"))
        );
    }
    assert_eq!(
        Ring::with_modulus(U256::ONE, 64).err(),
        Some(Error::Parameter("ring modulus"))
    );
    Ring::new(2, 64).unwrap();
    Ring::new(1i128 << 101, 64).unwrap();
    Ring::new(i128::MAX, 64).unwrap();
    Ring::with_modulus(U256::MAX, 64).unwrap();
    assert!(Ring::new(13, 63).is_err());
    assert!(SparsePolyMat::new(a.clone(), 1, vec![(0, 1, Poly::zero(a))]).is_err());
}

#[test]
fn even_statement_rings_keep_canonical_half_ties() {
    for q in [2, 16, 256] {
        let ring = Ring::new(q, 64).unwrap();
        let a = Poly::new(ring.clone(), vec![q / 2; 64]).unwrap();
        assert_eq!(a.neg(), a);
        assert_eq!(a.auto().auto(), a);
        assert_eq!(a.rotate(64), a);
        assert!(a.trace().is_err());
        assert_eq!(
            values(&a.mul(&a).unwrap()),
            reference(&values(&a), &values(&a), q)
        );
        assert!(
            values(&a.mul(&a).unwrap())
                .iter()
                .all(|x| *x > -q / 2 && *x <= q / 2)
        );
    }
}

#[test]
#[ignore = "extended independent NTT oracle run: 10,000 pairs per prime and degree"]
fn ntt_ten_thousand_pairs_per_prime() {
    fn power(mut a: u64, mut e: u64, p: u64) -> u64 {
        let mut r = 1;
        while e > 0 {
            if e & 1 == 1 {
                r = (r as u128 * a as u128 % p as u128) as u64;
            }
            a = (a as u128 * a as u128 % p as u128) as u64;
            e >>= 1;
        }
        r
    }
    let mut state = 0x923abed567882345u64;
    for (p, root) in NTT_PRIMES {
        for d in [64, 128, 256] {
            let plan = NttPlan::new(p, d, power(root, (1024 / d) as u64, p)).unwrap();
            for _ in 0..10000 {
                let mut sample = || {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (state >> 32) as i128 % 2001 - 1000
                };
                let a: Vec<i128> = (0..d).map(|_| sample()).collect();
                let b: Vec<i128> = (0..d).map(|_| sample()).collect();
                let mut expected = vec![0i128; d];
                for (i, a) in a.iter().enumerate() {
                    for (j, b) in b.iter().enumerate() {
                        expected[(i + j) % d] += if i + j < d { a * b } else { -a * b };
                    }
                }
                let mut x: Vec<u64> = a.iter().map(|x| x.rem_euclid(p as i128) as u64).collect();
                let mut y: Vec<u64> = b.iter().map(|x| x.rem_euclid(p as i128) as u64).collect();
                plan.forward(&mut x).unwrap();
                plan.forward(&mut y).unwrap();
                for (x, y) in x.iter_mut().zip(y) {
                    *x = (*x as u128 * y as u128 % p as u128) as u64;
                }
                plan.inverse(&mut x).unwrap();
                assert!(
                    x.iter()
                        .zip(expected)
                        .all(|(x, y)| *x == y.rem_euclid(p as i128) as u64)
                );
            }
        }
    }
}
