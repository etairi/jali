//! The encodings that moduli above $`2^{64}`$ and $`2^{128}`$ add, byte for byte: prime factors
//! in the parameter transcript, the modulus in the scheme's parameter bytes and centred secret
//! coefficients in key derivation. `golden_tests` pins the narrow forms by digest. Also the
//! rounding of wide values to `f64`, which parameter checks use.
use crate::{
    abdlop::{Abdlop, secret_bytes},
    math::{Poly, PolyVec, Ring, U256, int},
    params::{TboxParams, toy_d64},
};

fn power(k: u32) -> U256 {
    U256::ONE.shl_vartime(k)
}
fn le(x: u64) -> Vec<u8> {
    x.to_le_bytes().to_vec()
}

/// The set that `fit_wide` in `tests/common/params.rs` derives at $`q=2^{240}+325`$ for the
/// toolbox shape of `tests/toolbox.rs`. Its MLWE metadata is synthetic.
fn wide_set() -> TboxParams {
    TboxParams {
        id: "wide-parameters-test-only".into(),
        prime_factors: vec![power(240).wrapping_add(&U256::from_u64(325))],
        degree: 64,
        m1: 10,
        m2: 42,
        l: 2,
        n_msis: 3,
        alpha_squared: 640,
        n_bin: 2,
        l2_rows: vec![2, 1],
        l2_bounds_squared: vec![128, 64],
        n_prime: 2,
        linf_bound: 4,
        log_sigma: [15, 12, 10, 11],
        gamma: 1685450,
        d_bits: 12,
        mlwe_rank: 26,
        mlwe_delta: 1.0044,
        estimator: "test-only synthetic MLWE metadata; not an estimate".into(),
    }
}

/// The transcript bytes after the identifier and the estimator, which come first.
fn after_strings(p: &TboxParams) -> Vec<u8> {
    p.transcript_bytes()[16 + p.id.len() + p.estimator.len()..].to_vec()
}

#[test]
fn prime_factors_take_the_narrow_form_below_2_64_and_are_length_prefixed_above() {
    let mut p = toy_d64();
    let narrow = after_strings(&p);
    // Narrow form: LE64(count), then each factor as LE64.
    assert_eq!(narrow[..16], [le(1), le(1099511627917)].concat()[..]);
    let rest = &narrow[16..];
    p.prime_factors = vec![U256::from_u64(13), U256::from_u64(29)];
    assert_eq!(
        after_strings(&p),
        [le(2), le(13), le(29), rest.to_vec()].concat()
    );
    // Wide form: LE64(count + 2^63), then per factor LE64(k) and its k low-order bytes, with
    // k = ceil(bits / 8); the fields after the factors do not change.
    p.prime_factors = vec![power(64).wrapping_add(&U256::from_u64(13))];
    let one_wide = [
        le(1 | 1 << 63),
        le(9),
        vec![13, 0, 0, 0, 0, 0, 0, 0, 1],
        rest.to_vec(),
    ]
    .concat();
    assert_eq!(after_strings(&p), one_wide);
    // Any factor from 2^64 on switches every factor to the wide form.
    let big = power(240).wrapping_add(&U256::from_u64(325));
    p.prime_factors = vec![U256::from_u64(13), big];
    let mut two_wide = [le(2 | 1 << 63), le(1), vec![13], le(31)].concat();
    two_wide.extend_from_slice(&big.to_le_bytes()[..31]);
    two_wide.extend_from_slice(rest);
    assert_eq!(after_strings(&p), two_wide);
    assert_eq!(big.to_le_bytes()[30], 1);
    // The top bit of the first word tells the forms apart, and the length prefix makes the
    // wide form self-delimiting: moving a byte between the factor and the next field changes
    // the encoding.
    p.prime_factors = vec![power(255).wrapping_sub(&U256::from_u64(19))];
    let top = after_strings(&p);
    assert_eq!(top[..16], [le(1 | 1 << 63), le(32)].concat()[..]);
    assert_eq!(top.len(), 16 + 32 + rest.len());
    p.prime_factors = vec![power(255).wrapping_sub(&U256::from_u64(21))];
    assert_ne!(after_strings(&p), top);
}

/// A scheme at modulus `q` for the byte layout only: `toy_d64` with its factor and divisor
/// replaced and its checked values kept, since no parameter set lies between $`2^{64}`$ and
/// $`2^{128}`$. Four divides $`q-1`$ for every modulus used here.
fn scheme_at(q: U256) -> Abdlop {
    let mut parameters = toy_d64();
    parameters.prime_factors = vec![q];
    parameters.gamma = 4;
    let mut checked = toy_d64().check().unwrap();
    checked.q = q;
    Abdlop::with_checked([0; 32], parameters, checked).unwrap()
}

#[test]
fn the_modulus_takes_16_parameter_bytes_below_2_128_and_32_above() {
    let mut schemes = vec![
        (Abdlop::new([0; 32], toy_d64()).unwrap(), 16),
        (Abdlop::new([0; 32], wide_set()).unwrap(), 32),
    ];
    for (q, width) in [
        (power(64).wrapping_sub(&U256::from_u64(59)), 16),
        (power(64).wrapping_add(&U256::from_u64(13)), 16),
        (power(128).wrapping_sub(&U256::from_u64(159)), 16),
        (power(128).wrapping_add(&U256::from_u64(165)), 32),
        (U256::MAX.wrapping_sub(&U256::from_u64(434)), 32),
    ] {
        schemes.push((scheme_at(q), width));
    }
    for (scheme, width) in schemes {
        let transcript = scheme.parameters.transcript_bytes();
        let bytes = scheme.parameter_bytes();
        let n = transcript.len();
        assert_eq!(bytes[..n], transcript[..]);
        let q = scheme.ring().modulus().to_le_bytes();
        assert_eq!(bytes[n..n + width], q[..width], "width {width}");
        // The degree follows as a 16-byte integer, then 13 more.
        assert_eq!(bytes[n + width..n + width + 16], 64u128.to_le_bytes());
        assert_eq!(bytes.len(), n + width + 14 * 16);
    }
}

#[test]
fn to_f64_rounds_to_nearest_with_ties_to_even() {
    // Doubles from 2^64 to 2^65 are 2^12 apart. The tie 2^64 + 2^11 goes to the even 2^64; one
    // more unit, below the 64 bits that are kept, rounds up; so does the tie above 2^64 + 2^12.
    let above = |c: u64| int::to_f64(&power(64).wrapping_add(&U256::from_u64(c)));
    let two64 = 2f64.powi(64);
    assert_eq!(above(2047), two64);
    assert_eq!(above(2048), two64);
    assert_eq!(above(2049), two64 + 4096.0);
    assert_eq!(above(4096 + 2048), two64 + 8192.0);
    // The same 190 places further up, and at the top of the range.
    let tie = power(254).wrapping_add(&power(201));
    assert_eq!(int::to_f64(&tie), 2f64.powi(254));
    let past = tie.wrapping_add(&U256::ONE);
    assert_eq!(int::to_f64(&past), 2f64.powi(254) + 2f64.powi(202));
    assert_eq!(int::to_f64(&U256::MAX), 2f64.powi(256));
    // Below 2^128 the cast from u128 rounds the same way.
    let mut x = 0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c834u128;
    for _ in 0..1000 {
        x = x.rotate_left(17).wrapping_mul(0x2545_f491_4f6c_dd1d);
        for v in [x, x >> 40, x | 1 << 64] {
            assert_eq!(int::to_f64(&U256::from_u128(v)), v as f64, "{v}");
        }
    }
}

#[test]
fn secret_coefficients_take_16_bytes_below_2_128_and_32_above() {
    for (q, width) in [
        (U256::from_u64(1099511627917), 16),
        (power(128).wrapping_sub(&U256::from_u64(159)), 16),
        (power(128).wrapping_add(&U256::from_u64(165)), 32),
        (U256::MAX, 32),
    ] {
        let ring = Ring::with_modulus(q, 64).unwrap();
        // (q + 1) / 2 is the most negative centred value, -(q - 1)/2.
        let lowest = q.shr_vartime(1).wrapping_add(&U256::ONE);
        let v = PolyVec::new(
            ring.clone(),
            vec![
                Poly::constant(ring.clone(), -1),
                Poly::constant_u256(ring.clone(), &lowest),
            ],
        )
        .unwrap();
        let bytes = secret_bytes(&v);
        assert_eq!(bytes.len(), 2 * 64 * width);
        // -1 in two's complement, then 63 zero coefficients.
        assert!(bytes[..width].iter().all(|b| *b == 0xff));
        assert!(bytes[width..64 * width].iter().all(|b| *b == 0));
        // -(q - 1)/2 in two's complement: the negation of (q - 1)/2 in `width` bytes.
        let magnitude = q.shr_vartime(1).to_le_bytes();
        let mut negated = vec![0u8; width];
        let mut carry = 1u16;
        for (out, m) in negated.iter_mut().zip(magnitude[..].iter()) {
            let sum = u16::from(!m) + carry;
            *out = sum as u8;
            carry = sum >> 8;
        }
        assert_eq!(bytes[64 * width..65 * width], negated[..]);
    }
}
