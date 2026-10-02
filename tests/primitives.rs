use crypto_bigint::{I256, U256};
use jali::{
    Error,
    codec::{BitReader, BitWriter},
    math::Ring,
    rand::{
        AesPrg, ByteStream, ShakePrg, autostable, binomial, gaussian,
        reject::{self, Policy, Variance},
        uniform, uniform_u256,
    },
    transcript::Transcript,
};
use serde_json::Value;

mod common;

fn vectors() -> Value {
    if let Ok(path) = std::env::var("JALI_KAT_PATH") {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    } else {
        serde_json::from_str(include_str!("../kat/primitives.json")).unwrap()
    }
}
fn uint(v: &Value) -> U256 {
    U256::from_str_radix_vartime(v.as_str().unwrap(), 10).unwrap()
}
fn signed(v: &Value) -> I256 {
    let s = v.as_str().unwrap();
    let negative = s.starts_with('-');
    let magnitude = U256::from_str_radix_vartime(s.trim_start_matches('-'), 10).unwrap();
    let value = *magnitude.as_int();
    if negative {
        value.wrapping_neg()
    } else {
        value
    }
}
#[test]
fn python_prg_sampler_and_codec_known_answers() {
    let kat = vectors();
    let seed: [u8; 32] = hex::decode(kat["seed"].as_str().unwrap())
        .unwrap()
        .try_into()
        .unwrap();
    for item in kat["prgs"].as_array().unwrap() {
        let d = item["domain"].as_str().unwrap().parse().unwrap();
        let mut aes = AesPrg::new(&seed, d);
        let mut shake = ShakePrg::new(&seed, d);
        for (stream, name) in [
            (&mut aes as &mut dyn ByteStream, "aes"),
            (&mut shake as &mut dyn ByteStream, "shake"),
        ] {
            let mut out = [0u8; 103];
            stream.fill(&mut out[..7]).unwrap();
            stream.fill(&mut out[7..33]).unwrap();
            stream.fill(&mut out[33..]).unwrap();
            assert_eq!(hex::encode(out), item[name].as_str().unwrap());
        }
    }
    // Moduli up to 2^256 - 1: `uniform_u256` everywhere, and `uniform` where the modulus fits
    // u128, with the same values.
    let wide = |text: &str| {
        let n = num_bigint::BigUint::parse_bytes(text.as_bytes(), 10).unwrap();
        let mut bytes = n.to_bytes_le();
        bytes.resize(32, 0);
        U256::from_le_slice(&bytes)
    };
    for item in kat["samplers"].as_array().unwrap() {
        let text = item["modulus"].as_str().unwrap();
        let expected: Vec<U256> = item["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| wide(x.as_str().unwrap()))
            .collect();
        assert_eq!(
            uniform_u256(&mut AesPrg::new(&seed, 7), &wide(text), 137).unwrap(),
            expected,
            "{text}"
        );
        if let Ok(q) = text.parse::<u128>() {
            let narrow: Vec<U256> = uniform(&mut AesPrg::new(&seed, 7), q, 137)
                .unwrap()
                .into_iter()
                .map(U256::from_u128)
                .collect();
            assert_eq!(narrow, expected, "{text}");
        }
    }
    let expected: Vec<i128> = kat["binomial"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| i128::from(x.as_i64().unwrap()))
        .collect();
    assert_eq!(
        binomial(&mut AesPrg::new(&seed, 7), 3, 137).unwrap(),
        expected
    );
    for item in kat["gaussians"].as_array().unwrap() {
        let t = item["t"].as_u64().unwrap() as u32;
        let expected: Vec<i128> = item["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().parse().unwrap())
            .collect();
        assert_eq!(
            gaussian(&mut AesPrg::new(&seed, 7), t, 64).unwrap(),
            expected
        );
    }
    let c = &kat["codec"];
    let mut writer = BitWriter::new();
    for x in c["uniform"].as_array().unwrap() {
        writer.uniform(x.as_u64().unwrap().into(), 13).unwrap();
    }
    for x in c["gaussian"].as_array().unwrap() {
        writer.gaussian(x.as_i64().unwrap().into(), 4, 100).unwrap();
    }
    for x in c["hints"].as_array().unwrap() {
        writer.hint(x.as_i64().unwrap().into(), 100).unwrap();
    }
    let bytes = writer.finish();
    assert_eq!(hex::encode(&bytes), c["encoded"].as_str().unwrap());
    let mut reader = BitReader::new(&bytes);
    for x in c["uniform"].as_array().unwrap() {
        assert_eq!(reader.uniform(13).unwrap(), u128::from(x.as_u64().unwrap()));
    }
    for x in c["gaussian"].as_array().unwrap() {
        assert_eq!(
            reader.gaussian(4, 100).unwrap(),
            i128::from(x.as_i64().unwrap())
        );
    }
    for x in c["hints"].as_array().unwrap() {
        assert_eq!(
            reader.hint(16, 100).unwrap(),
            i128::from(x.as_i64().unwrap())
        );
    }
    reader.finish().unwrap();
}

#[test]
fn fixed_point_rejection_matches_300_bit_mpmath_including_boundaries() {
    let kat = vectors();
    for item in kat["exponentials"].as_array().unwrap() {
        let expected = uint(&item["scaled"]);
        let actual = reject::exp_negative(uint(&item["n"]), uint(&item["d"])).unwrap();
        let error = if actual > expected {
            actual.wrapping_sub(&expected)
        } else {
            expected.wrapping_sub(&actual)
        };
        assert!(error < U256::ONE.shl_vartime(22), "exp {item}");
    }
    // Integer variances, then fractional ones such as the sampler's 961*4^t/400.
    let integer = kat["rejection"].as_array().unwrap().iter().map(|item| {
        let variance = Variance {
            numerator: uint(&item["variance"]),
            denominator: 1,
        };
        (item, variance)
    });
    let fractional = kat["rational_rejection"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| {
            let variance = Variance {
                numerator: uint(&item["variance_numerator"]),
                denominator: item["variance_denominator"]
                    .as_str()
                    .unwrap()
                    .parse()
                    .unwrap(),
            };
            (item, variance)
        });
    for (item, variance) in integer.chain(fractional) {
        let policy = [Policy::Standard, Policy::Rej2, Policy::Bimodal]
            [item["policy"].as_u64().unwrap() as usize];
        let actual = reject::accept(
            policy,
            signed(&item["dot"]),
            uint(&item["norm"]),
            variance,
            uint(&item["m_scaled"]),
            item["u"].as_str().unwrap().parse().unwrap(),
        )
        .unwrap();
        assert_eq!(actual, item["accept"].as_bool().unwrap(), "{item}");
    }
}

#[test]
fn rejection_at_the_sampler_variance_outputs_the_target_gaussian() {
    // Accepted responses follow D_sigma exactly only if the acceptance test uses the variance
    // the mask was sampled with, 961*4^t/400, which is never an integer. Recover each
    // acceptance probability from accept() by bisection over u and check that
    // proposal(z) * acceptance(z) / rho_sigma(z) does not depend on z. It is constant to about
    // 1e-15; with the variance rounded to an integer it drifts by 4e-2 at t = 2 and 5e-6 at
    // t = 8.
    let m_scaled = U256::from(4u8).shl_vartime(128);
    for t in [2u32, 3, 8] {
        let variance = Variance::gaussian(t).unwrap();
        let s2 = 961.0 * 4f64.powi(t as i32) / 400.0;
        let rho = |x: f64| (-x * x / (2.0 * s2)).exp();
        let v = (0.8 * s2.sqrt()).round() as i128;
        for policy in [Policy::Standard, Policy::Rej2, Policy::Bimodal] {
            let mut ratios = Vec::new();
            for step in -40i128..=40 {
                let z = step * (3.0 * s2.sqrt()).round() as i128 / 40;
                let (dot, norm) = reject::moments(&[z], &[v]).unwrap();
                let accepts = |u| reject::accept(policy, dot, norm, variance, m_scaled, u).unwrap();
                if !accepts(0) || accepts(u128::MAX) {
                    continue; // never accepted (Rej2, z*v < 0) or always accepted (capped at 1)
                }
                let (mut low, mut high) = (0u128, u128::MAX);
                while high - low > 1 {
                    let mid = low + (high - low) / 2;
                    if accepts(mid) { low = mid } else { high = mid }
                }
                let acceptance = (low as f64 + 1.0) / 2f64.powi(128);
                let (z, v) = (z as f64, v as f64);
                let proposal = match policy {
                    Policy::Bimodal => (rho(z - v) + rho(z + v)) / 2.0,
                    _ => rho(z - v),
                };
                ratios.push(proposal * acceptance / rho(z));
            }
            assert!(ratios.len() > 20, "t = {t}, {policy:?}: too few points");
            let (min, max) = ratios
                .iter()
                .fold((f64::MAX, 0f64), |(a, b), r| (a.min(*r), b.max(*r)));
            assert!(
                max / min - 1.0 < 1e-10,
                "t = {t}, {policy:?}: ratio {min} to {max}"
            );
        }
    }
}

#[test]
fn rejection_with_a_scaled_variance_decides_alike_at_extreme_inputs() {
    // accept() multiplies both moments by the variance denominator, so n/1 and (n*k)/k must give
    // the same decision, including at the extremes of every input, where the scaled moments
    // approach 2^321. Extreme fractions must also decide without panicking.
    let dots = [I256::MIN, I256::MINUS_ONE, I256::ZERO, I256::ONE, I256::MAX];
    let norms = [U256::ZERO, U256::ONE, U256::MAX];
    let numerators = [
        U256::ONE,
        U256::from(961u16).shl_vartime(150),
        U256::MAX.shr_vartime(64),
    ];
    let ms = [
        U256::ONE.shl_vartime(128),
        U256::ONE.shl_vartime(144).wrapping_sub(&U256::ONE),
    ];
    let policies = [Policy::Standard, Policy::Rej2, Policy::Bimodal];
    for policy in policies {
        for dot in dots {
            for norm in norms {
                for m in ms {
                    for u in [0, 1, u128::MAX / 2, u128::MAX] {
                        let decide = |variance| reject::accept(policy, dot, norm, variance, m, u);
                        for numerator in numerators {
                            let reference = decide(Variance {
                                numerator,
                                denominator: 1,
                            })
                            .unwrap();
                            for k in [2u64, 400, u64::MAX] {
                                let scaled = Variance {
                                    numerator: numerator.wrapping_mul(&U256::from(k)),
                                    denominator: k,
                                };
                                assert_eq!(decide(scaled).unwrap(), reference, "{scaled:?}");
                            }
                        }
                        for (numerator, denominator) in
                            [(U256::MAX, 1), (U256::MAX, u64::MAX), (U256::ONE, u64::MAX)]
                        {
                            decide(Variance {
                                numerator,
                                denominator,
                            })
                            .unwrap();
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn malformed_encodings_fail_closed() {
    assert_eq!(BitReader::new(&[15]).uniform(13), Err(Error::Encoding));
    assert_eq!(BitReader::new(&[]).unsigned(1), Err(Error::Encoding));
    assert_eq!(
        BitReader::new(&[0xff; 100]).gaussian(4, 10),
        Err(Error::Encoding)
    );
    assert_eq!(
        BitReader::new(&[3, 0, 0]).hint(100, 10),
        Err(Error::Encoding)
    );
    assert_eq!(BitReader::new(&[0]).finish(), Err(Error::Encoding));
    assert_eq!(BitReader::new(&[3]).finish(), Err(Error::Encoding));
    assert_eq!(BitReader::new(&[1, 0]).finish(), Err(Error::Encoding));
    assert_eq!(BitReader::new(&[1]).finish(), Ok(()));
    for value in -1000..=1000 {
        let mut writer = BitWriter::new();
        writer.gaussian(value, 3, 1000).unwrap();
        writer.hint(value, 2000).unwrap();
        let encoded = writer.finish();
        let mut reader = BitReader::new(&encoded);
        assert_eq!(reader.gaussian(3, 1000).unwrap(), value);
        assert_eq!(reader.hint(1000, 2000).unwrap(), value);
        reader.finish().unwrap();
    }
    // A canonical hint code above max_abs is refused, one at max_abs read: both one-bit-pair
    // codes (±1) and unary codes of either sign.
    for value in [1i128, -1, 2, -2, 5, -5] {
        let mut writer = BitWriter::new();
        writer.hint(value, 100).unwrap();
        let encoded = writer.finish();
        let bound = value.unsigned_abs();
        assert_eq!(BitReader::new(&encoded).hint(bound, 100), Ok(value));
        assert_eq!(
            BitReader::new(&encoded).hint(bound - 1, 100),
            Err(Error::Encoding),
            "{value}"
        );
    }
}

/// A proof of the right shape whose every value is zero.
fn zero_proof(scheme: &jali::abdlop::Abdlop, p: &jali::params::TboxParams) -> jali::tbox::Proof {
    use jali::{
        abdlop::{Commitment, OpeningProof},
        math::{Poly, PolyVec},
        quad::QuadProof,
        quad_eval::EvalProof,
    };
    let c = p.check().unwrap();
    let ring = scheme.ring().clone();
    let e = if c.n_ex > 0 { 256 / p.degree } else { 0 };
    let d = if p.n_prime > 0 { 256 / p.degree } else { 0 };
    let g = c.lambda / 2;
    let zero = |n| PolyVec::zero(ring.clone(), n);
    jali::tbox::Proof {
        commitment: Commitment {
            t_a: zero(p.n_msis),
            t_b: zero(p.l + e + d + 1),
        },
        z_exact: zero(e),
        z_approx: zero(d),
        evaluation: EvalProof {
            garbage_commitments: zero(g),
            h: zero(g),
            quadratic: QuadProof {
                t: Poly::zero(ring.clone()),
                opening: OpeningProof {
                    challenge: Poly::zero(ring.clone()),
                    z1: zero(scheme.bounded_len()),
                    z21: zero(p.m2 - p.n_msis),
                    hint: zero(p.n_msis),
                },
            },
        },
    }
}
/// A hand-written encoding of a zero proof. Polynomials always hold coefficients reduced into
/// $`(-q/2,q/2]`$, so values outside the decoders' ranges exist only as bit strings: `t_a0`
/// replaces the first coefficient of $`t_A`$ and `z` the first Gaussian response coefficient,
/// of $`z^{(d)}`$ for a toolbox proof and of $`z_1`$ for an opening proof. The field order is
/// that of `codec::proof::encode` and `Abdlop::encode_proof`.
fn zero_bytes(
    scheme: &jali::abdlop::Abdlop,
    p: &jali::params::TboxParams,
    toolbox: bool,
    t_a0: u128,
    z: i128,
) -> Vec<u8> {
    let c = p.check().unwrap();
    let ring = scheme.ring();
    let (q, d) = (common::narrow(&ring.modulus()), p.degree);
    let e = if c.n_ex > 0 { 256 / d } else { 0 };
    let ds = if p.n_prime > 0 { 256 / d } else { 0 };
    let g = c.lambda / 2;
    let mut w = BitWriter::new();
    let mut gaussians = vec![
        (scheme.bounded_len(), p.log_sigma[0]),
        (p.m2 - p.n_msis, p.log_sigma[1]),
    ];
    if toolbox {
        for _ in 0..(p.l + e + ds + 1 + g + 1 + g) * d {
            w.uniform(0, q).unwrap();
        }
        let high_mod = 1u128 << (ring.coefficient_bits() - p.d_bits);
        w.uniform(t_a0, high_mod).unwrap();
        for _ in 1..p.n_msis * d {
            w.uniform(0, high_mod).unwrap();
        }
        gaussians.extend([(e, p.log_sigma[2]), (ds, p.log_sigma[3])]);
    }
    let omega = c.omega as u128;
    for _ in 0..d {
        w.uniform(omega, 2 * omega + 1).unwrap();
    }
    for _ in 0..p.n_msis * d {
        w.hint(0, 1 << 20).unwrap();
    }
    // The coefficient to replace: the first of z_approx, or the first of z1.
    let target = if toolbox { 3 } else { 0 };
    for (i, (len, t)) in gaussians.into_iter().enumerate() {
        for j in 0..len * d {
            let value = if i == target && j == 0 { z } else { 0 };
            w.gaussian(value, t, 1 << 20).unwrap();
        }
    }
    w.finish()
}

#[test]
fn proof_decoders_enforce_value_ranges() {
    use jali::{
        abdlop::Abdlop,
        codec::proof::{decode, encode},
        dcompress::Compression,
        math::PolyVec,
        statement::Requirements,
    };
    // Wide Gaussians (log_sigma[0] = 21 and [3] = 29 at a 40-bit q) give a coefficient just
    // above q/2 a Gaussian code within the unary limit, so the decoders' range checks, not that
    // limit, refuse it. Fitted by the test port of the parameter tool.
    let req = Requirements {
        m1: 10,
        l: 1,
        alpha_squared: 1 << 22,
        n_bin: 0,
        l2_rows: vec![],
        l2_bounds_squared: vec![],
        n_prime: 1,
        linf_bound: 1 << 20,
        max_integer_coefficient: jali::math::U256::ZERO,
        approx_alpha_squared: None,
        lifted_moduli: Vec::new(),
        linf: None,
    };
    let (_, p) = common::params::fit_upward(&req, "decoder-ranges-test-only", 64, 1099511627917);
    assert_eq!((p.log_sigma[0], p.log_sigma[3]), (21, 29));
    let scheme = Abdlop::new([3; 32], p.clone()).unwrap();
    let ring = scheme.ring().clone();
    let q = common::modulus(&ring);
    let omega = p.check().unwrap().omega;
    let high_max = common::narrow(
        &Compression::new(q, i128::from(p.gamma), p.d_bits)
            .unwrap()
            .power2round(&U256::from_u128((q - 1) as u128))
            .0,
    );
    // The hand-written encoder agrees with both encoders on zero proofs.
    let proof = zero_proof(&scheme, &p);
    assert_eq!(
        zero_bytes(&scheme, &p, true, 0, 0),
        encode(&scheme, &proof).unwrap()
    );
    let opening = proof.evaluation.quadratic.opening.clone();
    assert_eq!(
        zero_bytes(&scheme, &p, false, 0, 0),
        scheme.encode_proof(&opening).unwrap()
    );
    // t_A: high_max decodes, high_max + 1 does not.
    let decoded = decode(&scheme, &zero_bytes(&scheme, &p, true, high_max, 0)).unwrap();
    assert_eq!(
        decoded.commitment.t_a.entries()[0].coefficient_i128(0),
        Ok(high_max as i128)
    );
    assert_eq!(
        decode(&scheme, &zero_bytes(&scheme, &p, true, high_max + 1, 0)).err(),
        Some(Error::Encoding)
    );
    // Responses: |z| <= (q - 1) / 2 decodes, one more does not, in either decoder.
    let half = q / 2;
    for (z, ok) in [
        (half, true),
        (-half, true),
        (half + 1, false),
        (-half - 1, false),
    ] {
        let toolbox = decode(&scheme, &zero_bytes(&scheme, &p, true, 0, z));
        let opening = scheme.decode_proof(&zero_bytes(&scheme, &p, false, 0, z));
        if ok {
            assert_eq!(
                toolbox.unwrap().z_approx.entries()[0].coefficient_i128(0),
                Ok(z)
            );
            assert_eq!(opening.unwrap().z1.entries()[0].coefficient_i128(0), Ok(z));
        } else {
            assert_eq!(toolbox.err(), Some(Error::Encoding), "{z}");
            assert_eq!(opening.err(), Some(Error::Encoding), "{z}");
        }
    }
    // Inputs above 16 MiB are refused. Parsing would refuse this one too, so the cap itself
    // is tested at its boundary by the unit test of `codec::proof_reader`, which both
    // decoders call.
    let huge = vec![0u8; 16 * 1024 * 1024 + 1];
    assert_eq!(decode(&scheme, &huge).err(), Some(Error::Encoding));
    assert_eq!(scheme.decode_proof(&huge).err(), Some(Error::Encoding));
    // Encoders refuse t_A above high_max and challenge coefficients above omega.
    let mut bad = proof.clone();
    let mut t_a = bad.commitment.t_a.entries().to_vec();
    t_a[0].set_coefficient(0, high_max as i128 + 1).unwrap();
    bad.commitment.t_a = PolyVec::new(ring.clone(), t_a).unwrap();
    assert_eq!(encode(&scheme, &bad).err(), Some(Error::Encoding));
    let mut bad = proof.clone();
    let challenge = &mut bad.evaluation.quadratic.opening.challenge;
    challenge.set_coefficient(0, omega).unwrap();
    encode(&scheme, &bad).unwrap();
    let challenge = &mut bad.evaluation.quadratic.opening.challenge;
    challenge.set_coefficient(0, -omega - 1).unwrap();
    assert_eq!(encode(&scheme, &bad).err(), Some(Error::Encoding));
    assert_eq!(
        scheme.encode_proof(&bad.evaluation.quadratic.opening).err(),
        Some(Error::Encoding)
    );
}

#[test]
fn gaussians_are_local_deterministic_and_have_expected_moments() {
    for t in [0, 1, 3, 6, 24, 29, 60] {
        let samples = gaussian(&mut AesPrg::new(&[7; 32], 19), t, 20000).unwrap();
        let sigma = 1.55 * (2.0f64).powi(t as i32);
        let mean = samples.iter().map(|x| *x as f64 / sigma).sum::<f64>() / samples.len() as f64;
        let variance = samples
            .iter()
            .map(|x| (*x as f64 / sigma).powi(2))
            .sum::<f64>()
            / samples.len() as f64;
        assert!(mean.abs() < 0.04, "t={t} mean={mean}");
        assert!((variance - 1.0).abs() < 0.06, "t={t} variance={variance}");
        assert!(samples.iter().any(|x| *x != samples[0]));
        gaussian(&mut AesPrg::new(&[9; 32], 33), t, 11).unwrap();
        assert_eq!(
            gaussian(&mut AesPrg::new(&[7; 32], 19), t, 20000).unwrap(),
            samples
        );
    }
}

#[test]
fn challenge_symmetry_and_transcript_binding() {
    for d in [64, 128] {
        let c = autostable(
            &mut AesPrg::new(&[12; 32], 0),
            Ring::new(1099511627917, d).unwrap(),
            8,
        )
        .unwrap();
        assert_eq!(c.auto(), c);
        assert_eq!(c.coefficient_i128(d / 2), Ok(0));
        assert!(c.norm_infinity() <= U256::from_u8(8));
    }
    let mut a = Transcript::new(b"quad", b"params", &[0; 32], b"statement");
    let mut b = a.clone();
    a.absorb(b"h", b"abc");
    b.absorb(b"ha", b"bc");
    assert_ne!(a.challenge_seed(b"mu"), b.challenge_seed(b"mu"));
    assert_ne!(a.challenge_seed(b"mu"), a.challenge_seed(b"gamma"));
}

#[test]
#[cfg(feature = "test-utils")]
fn sampler_tapes_replay_exactly_and_fail_on_exhaustion() {
    use jali::test_utils::{RecordingStream, ReplayStream};
    let mut recording = RecordingStream::new(AesPrg::new(&[111; 32], 123));
    let expected = gaussian(&mut recording, 24, 64).unwrap();
    let mut replay = ReplayStream::new(recording.tape());
    assert_eq!(gaussian(&mut replay, 24, 64).unwrap(), expected);
    assert_eq!(replay.remaining(), 0);
    assert_eq!(replay.fill(&mut [0; 1]), Err(Error::Randomness));
    let mut replay = ReplayStream::new(&[3, 4]);
    assert_eq!(replay.fill(&mut [0; 3]), Err(Error::Randomness));
    assert_eq!(replay.remaining(), 2);
}
