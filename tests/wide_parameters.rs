//! Parameter sets above $`2^{64}`$: the primality test, the checks on prime factors and
//! Gaussian widths, and the JSON number rule. Prime lists and factorizations are from Sage
//! (`is_prime(proof=True)`, `factor`); the composites were checked in Python.
use jali::{
    Error,
    math::{U256, int::is_prime_u256},
    params::TboxParams,
    statement::Requirements,
};
use num_bigint::{BigInt, BigUint};

mod common;
use common::params::{FACTORS_128, FACTORS_240, even_divisors, fit_wide, power_minus, power_plus};

fn decimal(text: &str) -> U256 {
    let mut x = U256::ZERO;
    for b in text.bytes() {
        x = x
            .wrapping_mul(&U256::from_u8(10))
            .wrapping_add(&U256::from_u8(b - b'0'));
    }
    x
}

#[test]
fn baillie_psw_agrees_with_proven_prime_lists_above_2_64() {
    // Every prime in [2^64, 2^64 + 2000), [2^128, 2^128 + 2000), [2^255 - 2000, 2^255) and
    // [2^256 - 2000, 2^256), as offsets; Sage proved each and found no others.
    let ranges: [(u32, i64, &[i64]); 4] = [
        (
            64,
            0,
            &[
                13, 37, 51, 81, 93, 141, 307, 331, 393, 493, 541, 597, 637, 651, 717, 741, 745,
                757, 805, 807, 885, 925, 961, 981, 997, 1005, 1081, 1113, 1243, 1285, 1341, 1353,
                1407, 1413, 1417, 1483, 1521, 1555, 1675, 1785, 1795, 1831, 1885, 1917, 1927, 1945,
                1981,
            ],
        ),
        (
            128,
            0,
            &[
                51, 81, 165, 273, 385, 421, 463, 573, 625, 757, 817, 847, 1173, 1287, 1783, 1875,
                1975,
            ],
        ),
        (
            255,
            -2000,
            &[
                -1941, -1839, -1831, -1645, -1351, -1311, -1285, -949, -921, -765, -735, -475, -31,
                -19,
            ],
        ),
        (
            256,
            -2000,
            &[
                -1883, -1539, -1299, -1053, -923, -617, -587, -435, -357, -189,
            ],
        ),
    ];
    for (k, start, primes) in ranges {
        for offset in start..start + 2000 {
            let n = if offset >= 0 {
                power_plus(k, offset as u64)
            } else {
                power_minus(k, offset.unsigned_abs())
            };
            assert_eq!(
                is_prime_u256(&n),
                primes.contains(&offset),
                "2^{k} + {offset}"
            );
        }
    }
    // Below 2^64 the deterministic test answers.
    for n in [2u64, 3, 13, 1099511627917, u64::MAX - 58] {
        assert!(is_prime_u256(&U256::from_u64(n)));
    }
    for n in [0u64, 1, 4, 21, u64::MAX] {
        assert!(!is_prime_u256(&U256::from_u64(n)));
    }
}

/// Composites that fixed-base Miller-Rabin tests can miss, and Carmichael numbers.
const COMPOSITES: [(&str, &str); 9] = [
    // Strong pseudoprimes to the first 12 and 13 prime bases, 79 and 82 bits, 5 mod 8.
    ("psi12", "318665857834031151167461"),
    ("psi13", "3317044064679887385961981"),
    // Carmichael numbers 5 mod 8: 1921021 * 6723571 * 4305373695331, 1838341 * 6434191 *
    // 438082856191 and 1140311 * 5929613 * 7526047 (Korselt's criterion checked).
    ("carmichael86", "55608727969335720486208021"),
    ("carmichael83", "5181747899977148295508021"),
    ("carmichael66", "50888141443830911221"),
    // Chernick's Carmichael numbers (6k+1)(12k+1)(18k+1), 1 mod 8, of 65, 128 and 248 bits.
    ("chernick65", "23353635090086210521"),
    ("chernick128", "215334935796707021262036014826227070049"),
    (
        "chernick248",
        "286229224494098261424511362641918196375936449341437280168789685274018136729",
    ),
    // (2^63 - 375)(2^62 - 171), a 125-bit semiprime 5 mod 8.
    ("semiprime125", "42535295865117304626342950716533963389"),
];

#[test]
fn baillie_psw_refuses_pseudoprimes_and_carmichael_numbers() {
    for (name, n) in COMPOSITES {
        assert!(!is_prime_u256(&decimal(n)), "{name}");
    }
    // 2^126 - 201 = 163 * 599 * 607 * 1435411217316163072882453763957, and products of two
    // proven primes above 2^100.
    assert!(!is_prime_u256(&power_minus(126, 201)));
    assert!(!is_prime_u256(
        &power_minus(100, 99).wrapping_mul(&power_plus(100, 277))
    ));
    assert!(!is_prime_u256(
        &power_plus(128, 165).wrapping_mul(&U256::from_u64(1099511627917))
    ));
}

/// Whether `n` passes the strong probable-prime test to base 2 (one Miller-Rabin round).
fn strong_probable_prime_to_base_2(n: &BigUint) -> bool {
    let one = BigUint::from(1u8);
    let minus_one = n - &one;
    let s = minus_one.trailing_zeros().unwrap();
    let mut x = BigUint::from(2u8).modpow(&(&minus_one >> s), n);
    if x == one || x == minus_one {
        return true;
    }
    for _ in 1..s {
        x = &x * &x % n;
        if x == minus_one {
            return true;
        }
    }
    false
}

#[test]
fn baillie_psw_refuses_base_2_strong_pseudoprimes_up_to_251_bits() {
    // Every composite Mersenne number 2^p - 1 with p prime is a strong pseudoprime to base 2:
    // 2 has order p modulo it, and p divides (2^p - 2)/2 by Fermat. So is every composite
    // Fermat number F_k = 2^(2^k) + 1: squaring 2 k times gives 2^(2^k) = -1 modulo F_k. The
    // Miller-Rabin round of Baillie-PSW passes all of them, and only its Lucas test can refuse
    // them. From 2^64 on that is the test that runs; below, the deterministic Miller-Rabin test
    // with seven bases does. Sage proved M_p prime for the listed exponents and composite for
    // the other 42 primes p below 256, 33 of them above 64.
    const MERSENNE: [u32; 12] = [2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127];
    let mut composite_above_2_64 = 0;
    for p in (2u32..256).filter(|p| (2..*p).all(|d| p % d != 0)) {
        let n = power_minus(p, 1);
        let big = (BigUint::from(1u8) << p) - 1u8;
        assert!(strong_probable_prime_to_base_2(&big), "M_{p}");
        assert_eq!(is_prime_u256(&n), MERSENNE.contains(&p), "M_{p}");
        composite_above_2_64 += usize::from(p > 64 && !MERSENNE.contains(&p));
    }
    assert_eq!(composite_above_2_64, 33);
    // F_5, F_6 and F_7 (33, 65 and 129 bits) are composite.
    for k in [5u32, 6, 7] {
        let big = (BigUint::from(1u8) << (1u32 << k)) + 1u8;
        assert!(strong_probable_prime_to_base_2(&big), "F_{k}");
        assert!(!is_prime_u256(&power_plus(1 << k, 1)), "F_{k}");
    }
    // The two strong pseudoprimes of `COMPOSITES` pass the base-2 round too.
    for (name, n) in &COMPOSITES[..2] {
        assert!(
            strong_probable_prime_to_base_2(&n.parse().unwrap()),
            "{name}"
        );
    }
}

/// The toolbox shape of `tests/toolbox.rs` at degree 64.
fn requirements() -> Requirements {
    Requirements {
        m1: 10,
        l: 2,
        alpha_squared: 640,
        n_bin: 2,
        l2_rows: vec![2, 1],
        l2_bounds_squared: vec![128, 64],
        n_prime: 2,
        linf_bound: 4,
        max_integer_coefficient: U256::ZERO,
        approx_alpha_squared: None,
        lifted_moduli: Vec::new(),
        linf: None,
    }
}
fn wide_set() -> TboxParams {
    let q = power_plus(240, 325);
    fit_wide(
        &requirements(),
        "wide-parameters-test-only",
        q,
        &even_divisors(&q, &FACTORS_240),
        64,
    )
    .unwrap()
}

#[test]
fn parameter_checks_accept_wide_primes_and_refuse_composites() {
    let base = wide_set();
    let checked = base.check().unwrap();
    assert_eq!(checked.q, power_plus(240, 325));
    let q128 = power_plus(128, 165);
    fit_wide(
        &requirements(),
        "wide-parameters-128-test-only",
        q128,
        &even_divisors(&q128, &FACTORS_128),
        64,
    )
    .unwrap()
    .check()
    .unwrap();
    let factors = Some(Error::Parameter("proof modulus factors"));
    let with = |p: Vec<U256>| {
        let mut set = base.clone();
        set.prime_factors = p;
        set.check().err()
    };
    // Composites 5 mod 8 pass the congruence filter; only the primality test refuses them.
    for (name, n) in COMPOSITES {
        assert_eq!(with(vec![decimal(n)]), factors, "{name}");
    }
    assert_eq!(with(vec![power_minus(126, 201)]), factors);
    // A prime 1 mod 8 above 2^64 (2^64 + 81) and one 7 mod 8 (2^127 - 1).
    assert_eq!(with(vec![power_plus(64, 81)]), factors);
    assert_eq!(with(vec![power_minus(127, 1)]), factors);
    // Two primes above 2^128 whose product exceeds 2^256.
    assert_eq!(
        with(vec![power_plus(128, 165), power_plus(128, 421)]),
        Some(Error::Overflow)
    );
    // Two primes of 100 bits: the product is accepted as a ring modulus, but range and norm
    // blocks refuse a composite modulus, as below 2^64. gamma = 2 and D = 0 pass the
    // compression checks for any odd modulus.
    let mut two = base.clone();
    two.prime_factors = vec![power_minus(100, 99), power_plus(100, 277)];
    two.gamma = 2;
    two.d_bits = 0;
    assert_eq!(
        two.check().err(),
        Some(Error::Parameter(
            "binary, exact-norm and range blocks require a prime modulus"
        ))
    );
}

#[test]
fn gaussian_widths_up_to_the_cap_with_exact_response_bounds() {
    let base = wide_set();
    let set = |i: usize, t: u32| {
        let mut p = base.clone();
        p.log_sigma[i] = t;
        p
    };
    // sigma_4 up to 2^100: z4_bound = floor(124 * 2^t / 5) exactly.
    for t in [41, 64, 68, 100] {
        let c = set(3, t).check().unwrap();
        assert_eq!(
            BigInt::from(c.z4_bound),
            (BigInt::from(124) << t) / 5,
            "log_sigma[3] = {t}"
        );
        assert_eq!(c.approx_extraction_bound, 2 * c.z4_bound);
    }
    for i in 0..4 {
        assert_eq!(
            set(i, 101).check().err(),
            Some(Error::Parameter("Gaussian width capacity"))
        );
    }
    // z3_bound_squared = floor(656^2 961 4^t / 250000). At t = 50 an unchecked u128
    // evaluation overflows its intermediate; t = 58 is the last width whose value fits.
    for t in [50, 58] {
        let c = set(2, t).check().unwrap();
        assert_eq!(
            BigInt::from(c.z3_bound_squared),
            (BigInt::from(656 * 656 * 961) << (2 * t)) / 250000,
            "log_sigma[2] = {t}"
        );
    }
    assert_eq!(
        set(2, 59).check().err(),
        Some(Error::Parameter("response bound capacity"))
    );
}

#[cfg(feature = "serde")]
mod json {
    use super::*;
    use serde_json::{Value, json};

    const FIXTURES: [&str; 6] = [
        include_str!("../src/params/sets/toy-d64.json"),
        include_str!("../examples/fixtures/possession.json"),
        include_str!("fixtures/params/tag-preimage-crt-d64.json"),
        include_str!("fixtures/params/tag-preimage-crt-d128.json"),
        include_str!("fixtures/params/dense-blocks-7.json"),
        include_str!("fixtures/params/decoder-ranges.json"),
    ];

    #[test]
    fn existing_files_load_to_the_same_json_values() {
        for text in FIXTURES {
            let params = TboxParams::from_json(text).unwrap();
            let written = serde_json::to_value(&params).unwrap();
            assert_eq!(written, serde_json::from_str::<Value>(text).unwrap());
            // Every integer is still a JSON number.
            assert!(written["prime_factors"][0].is_u64());
        }
    }

    #[test]
    fn wide_factors_are_decimal_strings_and_readers_accept_both_forms() {
        let params = wide_set();
        let text = serde_json::to_string_pretty(&params).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            value["prime_factors"],
            json!(["1766847064778384329583297500742918515827483896875618958121606201292620101"])
        );
        assert!(value["gamma"].is_u64() && value["alpha_squared"].is_u64());
        assert_eq!(TboxParams::from_json(&text).unwrap(), params);
        // Decimal strings for any wide field, including values below 2^64.
        let base: Value = serde_json::from_str(FIXTURES[0]).unwrap();
        let reference = TboxParams::from_json(FIXTURES[0]).unwrap();
        for (field, value) in [
            ("prime_factors", json!(["1099511627917"])),
            ("alpha_squared", json!("5760")),
            ("l2_bounds_squared", json!(["128", 64])),
            ("linf_bound", json!("4")),
            ("gamma", json!("65202")),
        ] {
            let mut v = base.clone();
            v[field] = value;
            assert_eq!(
                TboxParams::from_json(&v.to_string()).unwrap(),
                reference,
                "{field}"
            );
        }
        // Refused: noncanonical strings, signs, floats, JSON numbers from 2^64 on, values
        // above the field, 2^256, 10^78 and 10^80.
        let ten = |k: usize| json!([format!("1{}", "0".repeat(k))]);
        for (field, value) in [
            ("prime_factors", ten(78)),
            ("prime_factors", ten(80)),
            ("prime_factors", json!(["01099511627917"])),
            ("prime_factors", json!(["+1099511627917"])),
            ("prime_factors", json!([""])),
            ("prime_factors", json!(["1099511627917.0"])),
            ("prime_factors", json!([1.5])),
            ("prime_factors", json!([-13])),
            (
                "prime_factors",
                serde_json::from_str::<Value>("[18446744073709551629]").unwrap(),
            ),
            (
                "prime_factors",
                json!([
                    "115792089237316195423570985008687907853269984665640564039457584007913129639936"
                ]),
            ),
            ("gamma", json!("18446744073709551616")),
            ("alpha_squared", json!("-1")),
            ("linf_bound", json!(" 4")),
        ] {
            let mut v = base.clone();
            v[field] = value.clone();
            assert_eq!(
                TboxParams::from_json(&v.to_string()).err(),
                Some(Error::Encoding),
                "{field} = {value}"
            );
        }
        // 2^256 - 1 parses; it is not prime, so the check refuses it.
        let mut v = base.clone();
        v["prime_factors"] = json!([
            "115792089237316195423570985008687907853269984665640564039457584007913129639935"
        ]);
        assert_eq!(
            TboxParams::from_json(&v.to_string()).err(),
            Some(Error::Parameter("proof modulus factors"))
        );
    }

    #[test]
    fn requirements_write_wide_bounds_as_strings() {
        let mut req = requirements();
        req.max_integer_coefficient = power_plus(200, 3);
        req.linf_bound = 1 << 70;
        let value = serde_json::to_value(&req).unwrap();
        assert_eq!(
            value["max_integer_coefficient"],
            json!("1606938044258990275541962092341162602522202993782792835301379")
        );
        assert_eq!(value["linf_bound"], json!("1180591620717411303424"));
        assert!(value["alpha_squared"].is_u64());
        let back: Requirements = serde_json::from_value(value).unwrap();
        assert_eq!(back, req);
    }

    #[test]
    fn decimal_strings_from_2_256_on_are_refused() {
        // A reader whose value wrapped modulo 2^256 would accept most of these: only a value
        // within one digit of a multiple of 2^256, such as 2^256 itself, overflows in the last
        // checked addition.
        let read = |text: &str| {
            let mut value = serde_json::to_value(requirements()).unwrap();
            value["max_integer_coefficient"] = json!(text);
            serde_json::from_value::<Requirements>(value).map(|r| r.max_integer_coefficient)
        };
        let ten = |k: usize| format!("1{}", "0".repeat(k));
        assert_eq!(read(&ten(77)).unwrap(), decimal(&ten(77)));
        assert!(read(&ten(78)).is_err() && read(&ten(80)).is_err());
        // Random values of 257 to 512 bits.
        let mut rng = common::SplitMix64(0x2567_0512);
        for _ in 0..64 {
            let bits = 257 + rng.below(256);
            let bytes: Vec<u8> = (0..64).map(|_| rng.next() as u8).collect();
            let x = (BigUint::from_bytes_le(&bytes) >> (512 - bits))
                | (BigUint::from(1u8) << (bits - 1));
            assert_eq!(x.bits(), bits);
            assert!(read(&x.to_string()).is_err(), "{x}");
        }
    }
}

/// postcard does not describe itself, so it cannot read the number-or-string form of JSON; in
/// such formats the wide fields take fixed forms.
#[cfg(feature = "serde")]
mod binary {
    use super::*;

    #[test]
    fn parameters_and_requirements_round_trip_through_postcard() {
        for params in [jali::params::toy_d64(), wide_set()] {
            let bytes = postcard::to_allocvec(&params).unwrap();
            assert_eq!(postcard::from_bytes::<TboxParams>(&bytes).unwrap(), params);
            // After the identifier (length, then bytes) come the factor count and each factor
            // as 32 little-endian bytes.
            let start = 1 + params.id.len();
            assert_eq!(bytes[start], 1);
            assert_eq!(
                bytes[start + 1..start + 33],
                params.prime_factors[0].to_le_bytes()[..]
            );
            assert!(postcard::from_bytes::<TboxParams>(&bytes[..bytes.len() - 1]).is_err());
        }
        let mut req = requirements();
        req.alpha_squared = 1 << 100;
        req.l2_bounds_squared = vec![u64::MAX, 3];
        req.linf_bound = u128::MAX;
        req.max_integer_coefficient = power_minus(256, 435);
        let bytes = postcard::to_allocvec(&req).unwrap();
        assert_eq!(postcard::from_bytes::<Requirements>(&bytes).unwrap(), req);
        // The 32 bytes of max_integer_coefficient, then the absent approx_alpha_squared, the
        // empty lifted_moduli and the absent linf, one byte each, which postcard always gets
        // (tests/per_slot_range.rs, tests/multi_modulus.rs and tests/linf.rs round-trip present
        // ones).
        let end = bytes.len() - 3;
        assert_eq!(
            bytes[end - 32..end],
            req.max_integer_coefficient.to_le_bytes()[..]
        );
        assert_eq!(bytes[end..], [0, 0, 0]);
        // JSON keeps its form.
        let value = serde_json::to_value(&req).unwrap();
        assert_eq!(
            value["linf_bound"],
            serde_json::json!(u128::MAX.to_string())
        );
        assert_eq!(serde_json::from_value::<Requirements>(value).unwrap(), req);
    }
}
