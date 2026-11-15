use crypto_bigint::{I256, U256, U1024};
use jali::{
    Error,
    codec::{BitReader, BitWriter},
    math::Ring,
    rand::{
        AesPrg, ByteStream, ShakePrg, autostable, binomial, challenge, eta_norm_power, gaussian,
        reject::{self, Policy, Variance},
        uniform, uniform_u256, within_eta,
    },
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
fn python_challenge_known_answers() {
    // Draws of `autostable` from the stream (seed, domain) with the exact operator-norm test:
    // the l1 norm of (sigma_{-1}(c) c)^32 of every draw, rejected ones included, is the Python
    // oracle's, every draw but the last exceeds eta^64, and `challenge` returns the last. Per
    // degree: domains 0 and 1, and the first two domains whose first draw is rejected.
    let kat = vectors();
    let seed: [u8; 32] = hex::decode(kat["seed"].as_str().unwrap())
        .unwrap()
        .try_into()
        .unwrap();
    let items = kat["challenges"].as_array().unwrap();
    let mut redrawn = 0;
    for item in items {
        let d = item["degree"].as_u64().unwrap() as usize;
        let omega = i128::from(item["omega"].as_i64().unwrap());
        let eta = item["eta"].as_u64().unwrap();
        let domain: u64 = item["domain"].as_str().unwrap().parse().unwrap();
        let ring = Ring::new(1099511627917, d).unwrap();
        let norms: Vec<U1024> = item["norms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| U1024::from_str_radix_vartime(x.as_str().unwrap(), 10).unwrap())
            .collect();
        let mut stream = AesPrg::new(&seed, domain);
        for (i, norm) in norms.iter().enumerate() {
            let c = autostable(&mut stream, ring.clone(), omega).unwrap();
            assert_eq!(eta_norm_power(&c).unwrap(), *norm, "{d} {domain} {i}");
            assert_eq!(within_eta(&c, eta).unwrap(), i + 1 == norms.len());
        }
        let expected: Vec<i128> = item["challenge"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| i128::from(x.as_i64().unwrap()))
            .collect();
        let c = challenge(&mut AesPrg::new(&seed, domain), ring, omega, eta).unwrap();
        assert_eq!(*c.coefficients_i128().unwrap(), expected, "{d} {domain}");
        redrawn += usize::from(norms.len() > 1);
    }
    assert_eq!((items.len(), redrawn), (8, 4));
}

#[test]
fn python_challenge_from_bytes_known_answers() {
    // `challenge_from_bytes` on 4032 bytes of SHAKE128(b"jali challenge kat" || seed ||
    // LE32(index)), as the Python oracle decodes them: the challenge, whose l1 norm of
    // (sigma_{-1}(c) c)^32 is the oracle's last norm. Per degree: indices 0 and 1, the first two
    // indices whose first draw is rejected, and the fallback to 1 on bytes 0xff.
    use shake::{ExtendableOutput, Shake128, Update, XofReader};
    let kat = vectors();
    let seed = hex::decode(kat["seed"].as_str().unwrap()).unwrap();
    let items = kat["challenges_from_bytes"].as_array().unwrap();
    let (mut redrawn, mut fallbacks) = (0, 0);
    for item in items {
        let d = item["degree"].as_u64().unwrap() as usize;
        let omega = i128::from(item["omega"].as_i64().unwrap());
        let eta = item["eta"].as_u64().unwrap();
        let bytes = match item["index"].as_u64() {
            Some(index) => {
                let mut h = Shake128::default();
                h.update(b"jali challenge kat");
                h.update(&seed);
                h.update(&(index as u32).to_le_bytes());
                let mut out = vec![0u8; jali::rand::CHALLENGE_BYTES];
                h.finalize_xof().read(&mut out);
                out
            }
            None => vec![0xff; jali::rand::CHALLENGE_BYTES],
        };
        let norms = item["norms"].as_array().unwrap();
        let ring = Ring::new(1099511627917, d).unwrap();
        let c = jali::rand::challenge_from_bytes(&bytes, ring, omega, eta).unwrap();
        let expected: Vec<i128> = item["challenge"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| i128::from(x.as_i64().unwrap()))
            .collect();
        assert_eq!(*c.coefficients_i128().unwrap(), expected, "{item}");
        match norms.last() {
            Some(norm) => assert_eq!(
                eta_norm_power(&c).unwrap(),
                U1024::from_str_radix_vartime(norm.as_str().unwrap(), 10).unwrap()
            ),
            None => fallbacks += 1,
        }
        redrawn += usize::from(norms.len() > 1);
    }
    assert_eq!((items.len(), redrawn, fallbacks), (10, 4, 2));
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
            uint(&item["u"]),
        )
        .unwrap();
        assert_eq!(actual, item["accept"].as_bool().unwrap(), "{item}");
    }
}

/// The largest coin that `accepts` takes, for a test that takes 0 and refuses `U256::MAX`, by
/// bisection: a rejection test takes exactly the coins up to a threshold.
fn largest_accepted(accepts: impl Fn(U256) -> bool) -> U256 {
    let (mut low, mut high) = (U256::ZERO, U256::MAX);
    while high.wrapping_sub(&low) > U256::ONE {
        let mid = low.wrapping_add(&high.wrapping_sub(&low).shr_vartime(1));
        if accepts(mid) {
            low = mid;
        } else {
            high = mid;
        }
    }
    low
}

/// `x` in floating point, for ratios that need about 50 bits.
fn to_f64(x: &U256) -> f64 {
    x.to_le_bytes()[..]
        .iter()
        .rev()
        .fold(0.0, |acc, b| acc * 256.0 + f64::from(*b))
}

#[test]
fn rejection_decides_with_every_bit_of_its_256_bit_coin() {
    // A test takes exactly the coins up to its threshold, and the threshold is resolved to one
    // unit of 2^-256: it lies anywhere modulo 2^128, so coins with the same high 128 bits are
    // decided differently on either side of it. A test that read 128 bits of its coin, or
    // compared only the coin's high half, would take whole blocks of 2^128 coins, and every
    // threshold + 1 would be a multiple of 2^128 (for a genuine 256-bit test, with probability
    // about 2^-128 per input). The lowest bit counts too: for each policy some threshold + 1
    // is odd, which a test that dropped any low bits of its coin would never give. The
    // bisection follows the test's own decisions, so the exact comparison (<=) is pinned by
    // rejection_takes_exactly_the_coins_up_to_its_threshold instead.
    let low_half = U256::ONE.shl_vartime(128).wrapping_sub(&U256::ONE);
    let mut thresholds = 0;
    let mut odd = [false; 3];
    for t in [1u32, 8, 16] {
        let variance = Variance::gaussian(t).unwrap();
        let s = (961.0 * 4f64.powi(t as i32) / 400.0).sqrt();
        let v = (0.8 * s).round() as i128;
        for m in [2u8, 3, 5] {
            let m_scaled = U256::from(m).shl_vartime(128);
            for policy in [Policy::Standard, Policy::Rej2, Policy::Bimodal] {
                for step in -8i128..=8 {
                    let z = (step as f64 * s / 4.0).round() as i128;
                    let (dot, norm) = reject::moments(&[z], &[v]).unwrap();
                    let accepts =
                        |u: U256| reject::accept(policy, dot, norm, variance, m_scaled, u).unwrap();
                    if !accepts(U256::ZERO) || accepts(U256::MAX) {
                        continue; // never accepted (Rej2, z*v < 0) or always accepted
                    }
                    let threshold = largest_accepted(accepts);
                    let next = threshold.wrapping_add(&U256::ONE);
                    let block = threshold.bitand(&low_half.not());
                    assert!(
                        accepts(threshold) && !accepts(next),
                        "{t} {m} {policy:?} {z}"
                    );
                    assert!(
                        next.bitand(&low_half) != U256::ZERO,
                        "{t} {m} {policy:?} {z}"
                    );
                    // The block of 2^128 coins around the threshold: the start is taken, the
                    // end refused.
                    assert!(accepts(block), "{t} {m} {policy:?} {z}");
                    assert!(!accepts(block.bitor(&low_half)), "{t} {m} {policy:?} {z}");
                    odd[policy as usize] |= next.bitand(&U256::ONE) == U256::ONE;
                    thresholds += 1;
                }
            }
        }
    }
    assert!(thresholds > 200, "{thresholds}");
    assert_eq!(odd, [true; 3]);
}

#[test]
fn rejection_takes_exactly_the_coins_up_to_its_threshold() {
    // Where 2^256 p is an integer, a test must take that coin and refuse the next, odd one: this
    // pins the comparison (<=, not <) and the coin's lowest bit. With M = 2 and the exponent
    // -k/(2n) (norm k, dot k, variance n/1), Rej_1 and Rej_2 accept exactly when
    // u 2^129 <= e 2^192, with e = exp_negative(k, 2n): the threshold is e 2^63. With norm and
    // dot 0, every policy accepts with probability 1/2 (a positive-exponent branch): the
    // threshold is 2^255. The bimodal test's negative branch has an exact threshold where its
    // denominator 1 + exp(-2|dot|/s^2) is exactly 1, which the 192-bit exponential gives from
    // 2|dot|/s^2 >= 256 on: with variance 1, dot 128 and norm 254, p = 2 e^-1/M and the
    // threshold is e 2^64, with e = exp_negative(2, 2).
    let m_scaled = U256::from(2u8).shl_vartime(128);
    let mut cases = Vec::new();
    for (k, n) in [
        (1u64, 7u64),
        (5, 3),
        (1000, 999),
        (12345, 6789),
        (3, 1 << 40),
    ] {
        let e = reject::exp_negative(U256::from(k), U256::from(2 * n)).unwrap();
        let variance = Variance {
            numerator: U256::from(n),
            denominator: 1,
        };
        for policy in [Policy::Standard, Policy::Rej2] {
            let moments = (I256::from(k as i64), U256::from(k));
            cases.push((policy, moments, variance, e.shl_vartime(63)));
        }
    }
    for t in [0u32, 1, 8] {
        let variance = Variance::gaussian(t).unwrap();
        for policy in [Policy::Standard, Policy::Rej2, Policy::Bimodal] {
            let moments = (I256::ZERO, U256::ZERO);
            cases.push((policy, moments, variance, U256::ONE.shl_vartime(255)));
        }
    }
    let e = reject::exp_negative(U256::from(2u8), U256::from(2u8)).unwrap();
    let variance = Variance {
        numerator: U256::ONE,
        denominator: 1,
    };
    let moments = (I256::from(128i64), U256::from(254u16));
    cases.push((Policy::Bimodal, moments, variance, e.shl_vartime(64)));
    for (policy, (dot, norm), variance, threshold) in cases {
        let accepts = |u: U256| reject::accept(policy, dot, norm, variance, m_scaled, u).unwrap();
        let case = format!("{policy:?} {dot:?} {norm:?} {variance:?}");
        assert!(accepts(threshold), "{case}");
        assert!(!accepts(threshold.wrapping_add(&U256::ONE)), "{case}");
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
                let accepts =
                    |u: U256| reject::accept(policy, dot, norm, variance, m_scaled, u).unwrap();
                if !accepts(U256::ZERO) || accepts(U256::MAX) {
                    continue; // never accepted (Rej2, z*v < 0) or always accepted (capped at 1)
                }
                let low = largest_accepted(accepts);
                let acceptance = to_f64(&low.wrapping_add(&U256::ONE)) / 2f64.powi(256);
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
                    for u in [
                        U256::ZERO,
                        U256::ONE,
                        U256::from_u128(u128::MAX / 2),
                        U256::from_u128(u128::MAX),
                        U256::MAX.shr_vartime(1),
                        U256::MAX,
                    ] {
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
/// Raw codes that a hand-written proof puts in place of the first coefficient of a field.
#[derive(Clone, Copy, Default)]
struct Codes {
    /// The first coefficient of $`t_A`$, in $`\mathrm{bitlen}(q-1)-D`$ bits.
    t_a0: u128,
    /// The code of the first response coefficient, of $`z^{(d)}`$ for a toolbox proof and of
    /// $`z_1`$ for an opening proof, in place of the code of zero.
    z: Option<u128>,
    /// The code of the challenge's first coefficient, in place of that of zero.
    c0: Option<u128>,
    /// The code of the first hint, in place of that of zero.
    h0: Option<u128>,
}
/// A hand-written encoding of a zero proof in the NARG layout of `docs/encodings.md`.
/// Polynomials always hold coefficients reduced into $`(-q/2,q/2]`$, so values outside the
/// decoders' ranges exist only as bit strings: `codes` replaces the first coefficient of some
/// fields. The field order is that of `codec::proof::encode` and `Abdlop::encode_proof`.
fn zero_bytes(
    scheme: &jali::abdlop::Abdlop,
    p: &jali::params::TboxParams,
    toolbox: bool,
    codes: Codes,
) -> Vec<u8> {
    let c = p.check().unwrap();
    let ring = scheme.ring();
    let d = p.degree;
    let k_q = 128 - (common::narrow(&ring.modulus()) - 1).leading_zeros();
    let e = if c.n_ex > 0 { 256 / d } else { 0 };
    let ds = if p.n_prime > 0 { 256 / d } else { 0 };
    let g = c.lambda / 2;
    let mut w = BitWriter::new();
    // The signed code of zero in `width` bits, or the replacement of the first coefficient.
    let signed = |w: &mut BitWriter, n: usize, width: u32, first: Option<u128>| {
        let zero = (1u128 << (width - 1)) - 1;
        for j in 0..n {
            w.unsigned(if j == 0 { first.unwrap_or(zero) } else { zero }, width)
                .unwrap();
        }
    };
    if toolbox {
        let k_a = ring.coefficient_bits() - p.d_bits;
        w.unsigned(codes.t_a0, k_a).unwrap();
        for _ in 1..p.n_msis * d {
            w.unsigned(0, k_a).unwrap();
        }
        for _ in 0..(p.l + e + ds + 1) * d {
            w.unsigned(0, k_q).unwrap();
        }
        signed(&mut w, e * d, p.log_sigma[2] + 4, None);
        signed(&mut w, ds * d, p.log_sigma[3] + 4, codes.z);
        for _ in 0..(2 * g + 1) * d {
            w.unsigned(0, k_q).unwrap();
        }
    }
    let omega = c.omega as u128;
    let k_c = 128 - (2 * omega).leading_zeros();
    for j in 0..d / 2 {
        w.unsigned(
            if j == 0 {
                codes.c0.unwrap_or(omega)
            } else {
                omega
            },
            k_c,
        )
        .unwrap();
    }
    let z1 = if toolbox { None } else { codes.z };
    signed(&mut w, scheme.bounded_len() * d, p.log_sigma[0] + 4, z1);
    signed(&mut w, (p.m2 - p.n_msis) * d, p.log_sigma[1] + 4, None);
    signed(&mut w, p.n_msis * d, c.hint_width, codes.h0);
    w.into_bytes().unwrap()
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
    // A set with wide Gaussians (log_sigma[0] = 21 and [3] = 29 at a 40-bit q), whose response
    // codes of 25 and 33 bits stay below q/2, fitted by the test port of the parameter tool.
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
    let checked = p.check().unwrap();
    let omega = checked.omega;
    let high_max = common::narrow(
        &Compression::new(q, i128::from(p.gamma), p.d_bits)
            .unwrap()
            .power2round(&U256::from_u128((q - 1) as u128))
            .0,
    );
    // The hand-written encoder agrees with both encoders on zero proofs, and the encoding's
    // length is the set's exact proof size.
    let proof = zero_proof(&scheme, &p);
    let zero = zero_bytes(&scheme, &p, true, Codes::default());
    assert_eq!(zero, encode(&scheme, &proof).unwrap());
    assert_eq!(zero.len(), checked.estimated_proof_bytes);
    let opening = proof.evaluation.quadratic.opening.clone();
    assert_eq!(
        zero_bytes(&scheme, &p, false, Codes::default()),
        scheme.encode_proof(&opening).unwrap()
    );
    // t_A: high_max decodes, high_max + 1 does not.
    let codes = |t_a0| Codes {
        t_a0,
        ..Codes::default()
    };
    let decoded = decode(&scheme, &zero_bytes(&scheme, &p, true, codes(high_max))).unwrap();
    assert_eq!(
        decoded.commitment.t_a.entries()[0].coefficient_i128(0),
        Ok(high_max as i128)
    );
    assert_eq!(
        decode(&scheme, &zero_bytes(&scheme, &p, true, codes(high_max + 1))).err(),
        Some(Error::Encoding)
    );
    // Responses: the codes 0 and 2b of a width w, b = 2^(w-1) - 1, decode to b and -b, and the
    // last code 2b + 1 does not decode, in either decoder.
    for (toolbox, t) in [(true, p.log_sigma[3]), (false, p.log_sigma[0])] {
        let b = (1u128 << (t + 3)) - 1;
        assert!(b < (q / 2) as u128, "the width keeps responses below q/2");
        for (code, value) in [
            (0, Some(b as i128)),
            (2 * b, Some(-(b as i128))),
            (2 * b + 1, None),
        ] {
            let bytes = zero_bytes(
                &scheme,
                &p,
                toolbox,
                Codes {
                    z: Some(code),
                    ..Codes::default()
                },
            );
            let first = if toolbox {
                decode(&scheme, &bytes).map(|x| x.z_approx.entries()[0].clone())
            } else {
                scheme
                    .decode_proof(&bytes)
                    .map(|x| x.z1.entries()[0].clone())
            };
            match value {
                Some(value) => assert_eq!(first.unwrap().coefficient_i128(0), Ok(value)),
                None => assert_eq!(first.err(), Some(Error::Encoding), "{toolbox}"),
            }
        }
    }
    // The challenge: the code 2 omega decodes to -omega, 2 omega + 1 does not. Hints: the code
    // 2b of the hint width decodes, 2b + 1 does not.
    let omega_code = 2 * omega as u128;
    let hint_code = 2 * ((1u128 << (checked.hint_width - 1)) - 1);
    for (c0, h0, ok) in [
        (Some(omega_code), None, true),
        (Some(omega_code + 1), None, false),
        (None, Some(hint_code), true),
        (None, Some(hint_code + 1), false),
    ] {
        let bytes = zero_bytes(
            &scheme,
            &p,
            false,
            Codes {
                c0,
                h0,
                ..Codes::default()
            },
        );
        assert_eq!(scheme.decode_proof(&bytes).is_ok(), ok, "{c0:?} {h0:?}");
    }
    // Exact length: a byte more or a byte less does not decode.
    let mut longer = zero.clone();
    longer.push(0);
    assert_eq!(decode(&scheme, &longer).err(), Some(Error::Encoding));
    assert_eq!(
        decode(&scheme, &zero[..zero.len() - 1]).err(),
        Some(Error::Encoding)
    );
    // Inputs above 16 MiB are refused. Parsing would refuse this one too, so the cap itself
    // is tested at its boundary by the unit test of `codec::proof_reader`, which both
    // decoders call.
    let huge = vec![0u8; 16 * 1024 * 1024 + 1];
    assert_eq!(decode(&scheme, &huge).err(), Some(Error::Encoding));
    assert_eq!(scheme.decode_proof(&huge).err(), Some(Error::Encoding));
    // Encoders refuse t_A above high_max, challenge coefficients above omega, and responses
    // beyond their width.
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
    let mut bad = proof.clone();
    let mut z1 = bad.evaluation.quadratic.opening.z1.entries().to_vec();
    z1[0].set_coefficient(0, 1 << (p.log_sigma[0] + 3)).unwrap();
    bad.evaluation.quadratic.opening.z1 = PolyVec::new(ring.clone(), z1).unwrap();
    assert_eq!(encode(&scheme, &bad).err(), Some(Error::Encoding));
}

#[test]
fn gaussians_are_local_deterministic_and_have_expected_moments() {
    for t in [0, 1, 2, 3, 6, 24, 29, 60] {
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
fn gaussians_follow_the_discrete_gaussian_at_small_widths() {
    // Chi-square against D_{Z,sigma} at the widths whose paths differ: t = 0 (no offset and no
    // Bernoulli test), t = 1 (no offset bits), t = 2 (one offset bit) and t = 3. One bin per
    // value with an expected count of at least 5, the tails merged into the outermost bins.
    // The streams are fixed, so the outcome is deterministic: |z| < 4 for
    // z = (chi^2 - dof) / sqrt(2 dof). A wrong width exponent, offset or table changes the
    // variance by a factor of 2 or more and fails this by far.
    let n = 100_000;
    for t in 0..=3u32 {
        let samples = gaussian(&mut AesPrg::new(&[17; 32], u64::from(t)), t, n).unwrap();
        let s2 = 961.0 * 4f64.powi(t as i32) / 400.0;
        let rho = |x: i128| (-(x * x) as f64 / (2.0 * s2)).exp();
        let reach = (20.0 * s2.sqrt()) as i128;
        let total: f64 = (-reach..=reach).map(rho).sum();
        let expected = |x: i128| n as f64 * rho(x) / total;
        // The outermost bins are (-inf, -m] and [m, inf), m the last value with 5 expected.
        let m = (0..reach)
            .take_while(|x| expected(*x) >= 5.0)
            .last()
            .unwrap();
        let bin = |x: i128| (x.clamp(-m, m) + m) as usize;
        let mut observed = vec![0f64; 2 * m as usize + 1];
        let mut want = vec![0f64; 2 * m as usize + 1];
        for x in &samples {
            observed[bin(*x)] += 1.0;
        }
        for x in -reach..=reach {
            want[bin(x)] += expected(x);
        }
        let chi2: f64 = observed
            .iter()
            .zip(&want)
            .map(|(o, e)| (o - e).powi(2) / e)
            .sum();
        let dof = (want.len() - 1) as f64;
        let z = (chi2 - dof) / (2.0 * dof).sqrt();
        assert!(
            z.abs() < 4.0,
            "t = {t}: chi^2 = {chi2} with {dof} degrees of freedom"
        );
    }
}

#[test]
fn challenge_symmetry() {
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
