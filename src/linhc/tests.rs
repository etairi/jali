use super::*;
use crate::rand::{ByteStream, challenge};

/// Degree 64, rank 4, eight message polynomials, messages up to 2^20 in absolute value,
/// LaZer's degree-64 challenges (omega 8, eta 140), phi = 53.
fn params() -> LinhcParams {
    derive(64, 4, 8, 1 << 20, 140, 53).unwrap()
}

/// A ring of the toy proof modulus, where the challenges are drawn before they are lifted.
fn proof_ring() -> Arc<Ring> {
    Ring::new((1i128 << 40) + 141, 64).unwrap()
}

fn sample_challenge(ring: &Arc<Ring>, seed: u8, i: u32) -> Poly {
    let c = challenge(
        &mut AesPrg::new(&[seed; 32], domain(7, i)),
        proof_ring(),
        8,
        140,
    )
    .unwrap();
    lift_poly(ring, &c).unwrap()
}

fn uniform_vec(ring: &Arc<Ring>, len: usize, bound: i128, seed: u8, dom: u32) -> PolyVec {
    let entries = (0..len)
        .map(|i| {
            let values = bounded(
                &mut AesPrg::new(&[seed; 32], domain(dom, i as u32)),
                -bound,
                bound,
                ring.degree(),
            )
            .unwrap();
            Poly::new(ring.clone(), values).unwrap()
        })
        .collect();
    PolyVec::new(ring.clone(), entries).unwrap()
}

fn coin(seed: u8, i: u32) -> [u8; COIN_BYTES] {
    let mut out = [0u8; COIN_BYTES];
    AesPrg::new(&[seed; 32], domain(9, i))
        .fill(&mut out)
        .unwrap();
    out
}

fn coefficients(v: &PolyVec) -> Vec<Vec<i128>> {
    v.entries()
        .iter()
        .map(|p| p.coefficients_i128().unwrap().to_vec())
        .collect()
}

/// The rejection constant M as an f64, from its scaled form.
fn m_value(checked: &LinhcChecked) -> f64 {
    let bytes = checked.m_scaled.to_le_bytes();
    let top = u128::from_le_bytes(bytes[16..32].try_into().unwrap());
    let low = u128::from_le_bytes(bytes[..16].try_into().unwrap());
    top as f64 + low as f64 / 2f64.powi(128)
}

#[test]
fn derived_parameters_are_minimal_and_consistent() {
    let params = params();
    let checked = params.check().unwrap();
    // p = 4B + 1 and every message bound below p/4.
    assert_eq!(params.plaintext, 4 * params.message_bound + 1);
    // q' is the least prime above 2(pN + B): the prime below it fails the check.
    let floor = 2 * (params.plaintext * checked.noise_bound + params.message_bound);
    assert!(params.modulus > floor);
    let mut below = params.modulus - 2;
    while !is_prime(below) {
        below -= 2;
    }
    let smaller = LinhcParams {
        modulus: below,
        ..params.clone()
    };
    assert_eq!(
        smaller.check(),
        Err(Error::Parameter("LinHC modulus below the decryption bound"))
    );
    // sigma' >= 53 T, and one step smaller would not be.
    let sigma = 1.55 * 2f64.powi(params.log_sigma as i32);
    let t = (checked.shift_bound_squared as f64).sqrt();
    assert!(sigma >= 53.0 * t && sigma / 2.0 < 53.0 * t);
    // phi = 53 gives M close to exp(sqrt(384 ln 2)/gamma), about 1.17 to 1.37.
    let m = m_value(&checked);
    assert!((1.15..1.37).contains(&m), "M = {m}");
}

#[test]
fn honest_openings_recompute_the_mask_commitment() {
    let params = params();
    let checked = params.check().unwrap();
    let ring = params.ring().unwrap();
    let key = LinhcKey::expand(&ring, &params, &[1; 32]).unwrap();
    let e = uniform_vec(&ring, params.len, 1, 2, 0);
    let (commitment, secret) = commit(&key, &params, &e, &[3; 32], 0).unwrap();
    let mut accepted = 0;
    for attempt in 0..24u32 {
        let c = sample_challenge(&ring, 4, attempt);
        let r = uniform_vec(&ring, params.len, 1 << 19, 5, attempt);
        let mask = mask(&key, &params, &r, &[6; 32], 3 * attempt).unwrap();
        let z = abdlop::add(&abdlop::scale(&e, &c).unwrap(), &r).unwrap();
        if let Some(opening) = open(&checked, &c, &secret, &mask, &coin(7, attempt)).unwrap() {
            accepted += 1;
            let (w1, w2) =
                recompute(&key, &params, &checked, &c, &commitment, &z, &opening).unwrap();
            assert_eq!(w1, mask.w1);
            assert_eq!(w2, mask.w2);
        }
    }
    assert!(accepted >= 12, "accepted {accepted} of 24");
}

#[test]
fn acceptance_rate_is_one_over_m() {
    let params = derive(64, 2, 3, 1 << 20, 140, 53).unwrap();
    let checked = params.check().unwrap();
    let ring = params.ring().unwrap();
    let key = LinhcKey::expand(&ring, &params, &[11; 32]).unwrap();
    let e = uniform_vec(&ring, params.len, 1, 12, 0);
    let (_, secret) = commit(&key, &params, &e, &[13; 32], 0).unwrap();
    let r = uniform_vec(&ring, params.len, 0, 14, 0);
    let trials = 600u32;
    let mut accepted = 0u32;
    for i in 0..trials {
        let c = sample_challenge(&ring, 15, i);
        let mask = mask(&key, &params, &r, &[16; 32], 3 * i).unwrap();
        if open(&checked, &c, &secret, &mask, &coin(17, i))
            .unwrap()
            .is_some()
        {
            accepted += 1;
        }
    }
    // E[accept] = 1/M exactly when the test never clips, so the rate is binomial with mean 1/M.
    let rate = f64::from(accepted) / f64::from(trials);
    let expected = 1.0 / m_value(&checked);
    let sd = (expected * (1.0 - expected) / f64::from(trials)).sqrt();
    assert!(
        (rate - expected).abs() < 5.0 * sd,
        "rate {rate}, expected {expected}, sd {sd}"
    );
}

#[test]
fn the_trapdoor_decrypts_every_combination_of_an_honest_commitment() {
    let params = params();
    params.check().unwrap();
    let ring = params.ring().unwrap();
    let (key, trapdoor) = LinhcKey::with_trapdoor(&ring, &params, &[21; 32]).unwrap();
    let e = uniform_vec(&ring, params.len, 1, 22, 0);
    let r = uniform_vec(&ring, params.len, 1 << 19, 23, 0);
    let (commitment, _) = commit(&key, &params, &e, &[24; 32], 0).unwrap();
    let mask = mask(&key, &params, &r, &[25; 32], 0).unwrap();
    for i in 0..12 {
        let c = sample_challenge(&ring, 26, i);
        let v1 = abdlop::add(&abdlop::scale(&commitment.t1, &c).unwrap(), &mask.w1).unwrap();
        let v2 = abdlop::add(&abdlop::scale(&commitment.t2, &c).unwrap(), &mask.w2).unwrap();
        let z = abdlop::add(&abdlop::scale(&e, &c).unwrap(), &r).unwrap();
        assert_eq!(
            decrypt(&trapdoor, params.plaintext, &v1, &v2).unwrap(),
            coefficients(&z)
        );
    }
}

/// A vector of the norm `bound_squared` allows whose product with `row` has the largest
/// constant coefficient: coefficient 0 of $`ab`$ is $`a_0b_0-\sum_{l\ge1}a_lb_{d-l}`$, so
/// $`b_0=\lambda\,\mathrm{sgn}(a_0)`$ and $`b_{d-l}=-\lambda\,\mathrm{sgn}(a_l)`$ give
/// $`\lambda\|a\|_1`$, scaled by `sign`.
fn aligned(ring: &Arc<Ring>, row: &[&Poly], bound_squared: u128, sign: i128) -> PolyVec {
    let rows: Vec<Vec<i128>> = row
        .iter()
        .map(|a| a.coefficients_i128().unwrap().to_vec())
        .collect();
    let nonzero = rows.iter().flatten().filter(|x| **x != 0).count() as u128;
    let lambda = ceil_sqrt(bound_squared / nonzero + 1) - 1;
    assert!(lambda * lambda * nonzero <= bound_squared);
    let d = ring.degree();
    let entries = rows
        .iter()
        .map(|a| {
            let mut b = vec![0i128; d];
            b[0] = sign * lambda as i128 * a[0].signum();
            for l in 1..d {
                b[d - l] = -sign * lambda as i128 * a[l].signum();
            }
            Poly::new(ring.clone(), b).unwrap()
        })
        .collect();
    PolyVec::new(ring.clone(), entries).unwrap()
}

/// An opening of $`(v_1,v_2)=(p(Az'_1+z'_2),p(Bz'_1+z'_3)+z)`$ aligned with row 0 of the
/// trapdoor, the constant coefficient of its noise, and the vectors.
fn worst_case(
    params: &LinhcParams,
    checked: &LinhcChecked,
    seed: &[u8; 32],
) -> (Trapdoor, PolyVec, PolyVec, PolyVec, i128) {
    let ring = params.ring().unwrap();
    let (key, trapdoor) = LinhcKey::with_trapdoor(&ring, params, seed).unwrap();
    fn row(m: &PolyMat) -> Vec<&Poly> {
        (0..m.cols()).map(|j| m.get(0, j).unwrap()).collect()
    }
    let z1 = aligned(
        &ring,
        &row(&trapdoor.d2),
        checked.block_bounds_squared[0],
        1,
    );
    let z2 = aligned(
        &ring,
        &row(&trapdoor.d1),
        checked.block_bounds_squared[1],
        -1,
    );
    let mut z3 = PolyVec::zero(ring.clone(), params.len).entries().to_vec();
    let beta3 = ceil_sqrt(checked.block_bounds_squared[2] + 1) - 1;
    z3[0] = Poly::constant(ring.clone(), beta3 as i128);
    let z3 = PolyVec::new(ring.clone(), z3).unwrap();
    for (z, bound) in [&z1, &z2, &z3].iter().zip(checked.block_bounds_squared) {
        assert!(z.norm_squared().unwrap() <= U256::from_u128(bound));
    }
    // The message: B at the attacked coefficient, small elsewhere.
    let mut message = uniform_vec(&ring, params.len, 3, 31, 0).entries().to_vec();
    message[0]
        .set_coefficient(0, params.message_bound as i128)
        .unwrap();
    let message = PolyVec::new(ring.clone(), message).unwrap();
    let (v1, v2) = combine(&key, params, [&z1, &z2, &z3], &message).unwrap();
    // The noise D_2 z'_1 - D_1 z'_2 + z'_3 at coefficient 0 of row 0, over the integers.
    let noise = trapdoor.d2.mul(&z1).unwrap().entries()[0]
        .sub(&trapdoor.d1.mul(&z2).unwrap().entries()[0])
        .unwrap()
        .add(&z3.entries()[0])
        .unwrap()
        .coefficient_i128(0)
        .unwrap();
    (trapdoor, v1, v2, message, noise)
}

#[test]
fn decryption_holds_at_an_aligned_worst_case_opening() {
    let params = params();
    let checked = params.check().unwrap();
    let (trapdoor, v1, v2, message, noise) = worst_case(&params, &checked, &[30; 32]);
    // The alignment reaches a constant share of the Cauchy–Schwarz bound N (the ternary rows
    // have about two thirds of their entries nonzero), so the check is not vacuous.
    assert!(noise > 0 && (noise as u128) <= checked.noise_bound);
    assert!(
        noise as f64 >= 0.6 * checked.noise_bound as f64,
        "noise {noise}, bound {}",
        checked.noise_bound
    );
    assert_eq!(
        decrypt(&trapdoor, params.plaintext, &v1, &v2).unwrap(),
        coefficients(&message)
    );
}

#[test]
fn decryption_fails_once_the_modulus_is_below_the_noise() {
    let params = params();
    let checked = params.check().unwrap();
    let (_, _, _, _, noise) = worst_case(&params, &checked, &[30; 32]);
    // A prime modulus below 2(p noise + B): the same opening wraps around q'/2.
    let mut below = 2 * (params.plaintext * noise as u128 + params.message_bound) - 1;
    while !is_prime(below) {
        below -= 2;
    }
    let small = LinhcParams {
        modulus: below,
        ..params.clone()
    };
    assert!(small.check().is_err());
    let (trapdoor, v1, v2, message, small_noise) = worst_case(&small, &checked, &[30; 32]);
    // The trapdoor and the aligned opening are the same integers over the smaller modulus.
    assert_eq!(small_noise, noise);
    let decrypted = decrypt(&trapdoor, small.plaintext, &v1, &v2).unwrap();
    assert_ne!(
        decrypted[0][0],
        message.entries()[0].coefficient_i128(0).unwrap()
    );
}

#[test]
fn tampering_is_caught() {
    let params = params();
    let checked = params.check().unwrap();
    let ring = params.ring().unwrap();
    let key = LinhcKey::expand(&ring, &params, &[41; 32]).unwrap();
    let e = uniform_vec(&ring, params.len, 1, 42, 0);
    let (commitment, secret) = commit(&key, &params, &e, &[43; 32], 0).unwrap();
    let r = uniform_vec(&ring, params.len, 1 << 19, 44, 0);
    let (c, mask, opening) = (0..64u32)
        .find_map(|i| {
            let c = sample_challenge(&ring, 45, i);
            let mask = mask(&key, &params, &r, &[46; 32], 3 * i).unwrap();
            let opening = open(&checked, &c, &secret, &mask, &coin(47, i)).unwrap()?;
            Some((c, mask, opening))
        })
        .unwrap();
    let z = abdlop::add(&abdlop::scale(&e, &c).unwrap(), &r).unwrap();
    // Another message by one in one coefficient: w_2 moves by exactly that one.
    let mut other = z.entries().to_vec();
    let x = other[2].coefficient_i128(5).unwrap();
    other[2].set_coefficient(5, x + 1).unwrap();
    let other = PolyVec::new(ring.clone(), other).unwrap();
    let (w1, w2) = recompute(&key, &params, &checked, &c, &commitment, &other, &opening).unwrap();
    assert_eq!(w1, mask.w1);
    assert_ne!(w2, mask.w2);
    // Another challenge changes both.
    let c2 = sample_challenge(&ring, 48, 0);
    let (w1, _) = recompute(&key, &params, &checked, &c2, &commitment, &z, &opening).unwrap();
    assert_ne!(w1, mask.w1);
    // A block above its bound is refused.
    let mut big = opening.clone();
    let beta = ceil_sqrt(checked.block_bounds_squared[2]) as i128;
    let mut entries = big.z[2].entries().to_vec();
    entries[0] = Poly::constant(ring.clone(), beta + 1);
    big.z[2] = PolyVec::new(ring.clone(), entries).unwrap();
    assert_eq!(
        recompute(&key, &params, &checked, &c, &commitment, &z, &big),
        Err(Error::InvalidProof)
    );
    // Wrong shapes are refused.
    let short = PolyVec::new(ring.clone(), z.entries()[1..].to_vec()).unwrap();
    assert_eq!(
        recompute(&key, &params, &checked, &c, &commitment, &short, &opening),
        Err(Error::Dimension)
    );
}

#[test]
fn the_check_refuses_invalid_parameters() {
    let good = params();
    good.check().unwrap();
    let refuse = |p: LinhcParams, why: &'static str| {
        assert_eq!(p.check(), Err(Error::Parameter(why)), "{p:?}");
    };
    refuse(
        LinhcParams {
            plaintext: good.plaintext + 1,
            ..good.clone()
        },
        "LinHC plaintext modulus",
    );
    refuse(
        LinhcParams {
            plaintext: 4 * good.message_bound - 1,
            ..good.clone()
        },
        "LinHC plaintext modulus",
    );
    // A composite modulus, even one above the decryption bound.
    refuse(
        LinhcParams {
            modulus: good.modulus * 3,
            ..good.clone()
        },
        "LinHC modulus",
    );
    // A larger prime modulus is fine.
    let larger = LinhcParams {
        modulus: next_prime(good.modulus + 2).unwrap(),
        ..good.clone()
    };
    larger.check().unwrap();
    refuse(
        LinhcParams {
            modulus: 15,
            ..good.clone()
        },
        "LinHC modulus",
    );
    refuse(
        LinhcParams {
            degree: 32,
            ..good.clone()
        },
        "LinHC degree",
    );
    refuse(
        LinhcParams {
            rank: 0,
            ..good.clone()
        },
        "LinHC dimensions",
    );
    refuse(
        LinhcParams {
            log_sigma: 41,
            ..good.clone()
        },
        "LinHC mask exponent",
    );
    // A width close to T makes M at least 2^16.
    let narrow = LinhcParams {
        log_sigma: 1,
        ..good.clone()
    };
    assert!(matches!(narrow.check(), Err(Error::Parameter(_))));
    assert_eq!(
        rejection_constant(1, narrow.bounds().unwrap().0),
        Err(Error::Parameter("LinHC rejection constant"))
    );
}

/// Prints the LinHC parameters of the shipped extraction sets and of three larger shapes, for
/// `docs/parameters.md`. Run with `--ignored --nocapture`.
#[test]
#[ignore = "diagnostic output"]
fn report_parameters_for_known_shapes() {
    use crate::params::{self, ExtractionParams};
    let report = |name: &str, params: &LinhcParams| {
        let checked = params.check().unwrap();
        let (n, m, d) = (params.rank, params.len, params.degree);
        let com_bits = 2 * m * d * (128 - params.modulus.leading_zeros()) as usize;
        let gbits = (1.55f64.log2() + params.log_sigma as f64 + 2.5).ceil() as usize;
        let op_bits = (n + 2 * m) * d * gbits;
        eprintln!(
            "{name}: n = {n}, m' = {m}, t' = {}, log2 p = {:.2}, log2 q' = {:.2}, \
             N = 2^{:.2}, M = {:.4}, com {:.1} kB, op {:.1} kB",
            params.log_sigma,
            (params.plaintext as f64).log2(),
            (params.modulus as f64).log2(),
            (checked.noise_bound as f64).log2(),
            m_value(&checked),
            com_bits as f64 / 8000.0,
            op_bits as f64 / 8000.0,
        );
    };
    type Set = (
        &'static str,
        fn() -> params::TboxParams,
        fn() -> ExtractionParams,
    );
    let sets: [Set; 4] = [
        ("toy_d64", params::toy_d64, params::toy_d64_extraction),
        (
            "kyber1024_d64",
            params::kyber1024_d64,
            params::kyber1024_d64_extraction,
        ),
        (
            "kyber1024_d128",
            params::kyber1024_d128,
            params::kyber1024_d128_extraction,
        ),
        ("demo_d64", params::demo_d64, params::demo_d64_extraction),
    ];
    for (name, set, extraction) in sets {
        let set = set();
        let (params, _) = extraction().linhc(&set, &set.check().unwrap()).unwrap();
        report(name, &params);
    }
    // Larger shapes at degree 64: name, degree, LinHC rank, m', B (the larger of the z1 bound
    // and the joint z2 bound).
    for (name, d, n, m, b) in [
        ("m149", 64usize, 37usize, 149usize, 18_696_498u128),
        ("m101", 64, 36, 101, 13_791_067),
        ("m52", 64, 34, 52, 3_981_138),
    ] {
        report(name, &derive(d, n, m, b, 140, 53).unwrap());
    }
}
