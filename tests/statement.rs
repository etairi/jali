use jali::{
    Error, lin,
    math::{Poly, PolyMat, PolyVec, Ring, U256},
    params::toy_d64,
    quad,
    statement::{Norm, Statement},
};
use std::{collections::BTreeMap, sync::Arc};

mod common;

fn params() -> jali::params::TboxParams {
    let mut p = toy_d64();
    // tools/params/lnp_params.py possession.request.json toy-d64.report.json
    p.id = "quadratic-possession-test-only".into();
    p.m1 = 20;
    p.m2 = 55;
    p.n_msis = 15;
    p.l = 2;
    p.alpha_squared = 192;
    p.n_bin = 0;
    p.l2_rows = vec![16, 4];
    p.n_prime = 2;
    p.linf_bound = 13;
    // sigma4 is sized from the Euclidean bound sqrt(n'd)*linf_bound of the carry vector.
    p.log_sigma = [14, 12, 10, 13];
    p.d_bits = 6;
    #[cfg(feature = "serde")]
    assert_eq!(
        p,
        jali::params::TboxParams::from_json(include_str!("../examples/fixtures/possession.json"))
            .unwrap()
    );
    p
}
/// `(s_0 - 1)(5 - x_0) = 1` over a degree-128 statement ring with the given modulus.
fn possession(source: &Arc<Ring>) -> Statement {
    let mut statement = Statement::new(source.clone());
    statement.var("s", 8, Norm::L2Squared(128)).unwrap();
    statement.var("x", 2, Norm::L2Squared(64)).unwrap();
    let one = statement
        .constant(Poly::constant(source.clone(), 1))
        .unwrap();
    let five = statement
        .constant(Poly::constant(source.clone(), 5))
        .unwrap();
    let s = statement.variable("s", 0).unwrap();
    let x = statement.variable("x", 0).unwrap();
    let left = s
        .add(&one.scale(&Poly::constant(source.clone(), -1)).unwrap())
        .unwrap();
    let right = five
        .add(&x.scale(&Poly::constant(source.clone(), -1)).unwrap())
        .unwrap();
    let equation = left
        .product_affine(&right)
        .unwrap()
        .add(&one.scale(&Poly::constant(source.clone(), -1)).unwrap())
        .unwrap();
    statement.eq_mod_p(equation).unwrap();
    statement
}

#[test]
fn lifting_check_never_admits_a_modulus_that_can_wrap_around() {
    // Extraction bounds a carry by approx_extraction_bound = 2*z4_bound (LNP22 Lemma 2.7), so
    // the lifted relation holds over the integers when q > max_f + p*approx_extraction_bound.
    // Find the largest statement modulus the compiler admits and check that condition there.
    let params = params();
    let checked = params.check().unwrap();
    assert_eq!(checked.approx_extraction_bound, 2 * checked.z4_bound);
    let q = common::narrow(&checked.q);
    let bound = checked.approx_extraction_bound;
    let max_f = common::narrow(
        &possession(&Ring::new(13, 128).unwrap())
            .requirements(64, checked.q)
            .unwrap()
            .max_integer_coefficient,
    );
    let compile = |p: u128| possession(&Ring::new(p as i128, 128).unwrap()).compile(params.clone());
    // Every p above `unsafe_from` can wrap around; the compiler must refuse it.
    let unsafe_from = (q - 1 - max_f) / bound + 1;
    assert!(max_f + unsafe_from * bound >= q);
    let (mut admitted, mut refused) = (16u128, unsafe_from);
    assert!(compile(admitted).is_ok());
    assert_eq!(
        compile(refused).err(),
        Some(Error::Parameter("modulus lifting bound"))
    );
    while refused - admitted > 1 {
        let mid = (admitted + refused) / 2;
        match compile(mid) {
            Ok(_) => admitted = mid,
            Err(e) => {
                assert_eq!(e, Error::Parameter("modulus lifting bound"));
                refused = mid;
            }
        }
    }
    assert!(
        max_f + admitted * bound < q,
        "largest admitted modulus {admitted} can wrap around"
    );
}

/// The witness `s_0 = 5`, `x_0 = -5` of `possession`, all other polynomials zero.
fn possession_witness(source: &Arc<Ring>) -> BTreeMap<String, Vec<Poly>> {
    let mut s = vec![Poly::zero(source.clone()); 8];
    s[0] = Poly::constant(source.clone(), 5);
    let mut x = vec![Poly::zero(source.clone()); 2];
    x[0] = Poly::constant(source.clone(), -5);
    BTreeMap::from([("s".into(), s), ("x".into(), x)])
}

#[test]
fn named_quadratic_possession_with_degree_lowering_and_committed_carries() {
    let source = Ring::new(13, 128).unwrap();
    let statement = possession(&source);
    let required = statement
        .requirements(64, U256::from_u64(1099511627917))
        .unwrap();
    assert_eq!((required.m1, required.l, required.n_prime), (20, 2, 2));
    assert_eq!(required.l2_rows, vec![16, 4]);
    assert!(required.linf_bound <= 13);
    let compiled = statement.compile(params()).unwrap();
    let mut witness = possession_witness(&source);
    let (bounded, messages) = compiled.map_witness(&witness).unwrap();
    // (5-1)*(5-(-5))-1=39=13*3 over the integers. Interleaving degree 128→64
    // puts carry 3 in the first component and zero in the second.
    assert_eq!(messages.entries()[0].coefficient_i128(0), Ok(3));
    assert!(messages.entries()[1].is_zero());
    let internal = quad::interleave(&bounded, &messages).unwrap();
    for equation in &compiled.statement().quadratic {
        assert!(equation.evaluate(&internal).unwrap().is_zero());
    }
    let proof = compiled
        .prove_with_seed([33; 32], &witness, b"example/possession", [44; 32])
        .unwrap();
    compiled
        .verify([33; 32], &proof, b"example/possession")
        .unwrap();
    let scheme = jali::abdlop::Abdlop::new([33; 32], params()).unwrap();
    let bytes = jali::codec::proof::encode(&scheme, &proof).unwrap();
    // These checks come before the pinned fingerprint, so a changed fingerprint cannot hide
    // their failure.
    compiled
        .verify_bytes([33; 32], &bytes, b"example/possession")
        .unwrap();
    assert!(
        compiled
            .verify_bytes([33; 32], &bytes, b"another context")
            .is_err()
    );
    assert!(
        compiled
            .verify([34; 32], &proof, b"example/possession")
            .is_err()
    );
    witness.get_mut("x").unwrap()[0] = Poly::constant(source, -4);
    assert!(compiled.map_witness(&witness).is_err());
    // Fixed wire fingerprint also runs in CI across architectures and feature sets.
    // Generated by the standalone possession example without the parallel feature.
    use shake::{ExtendableOutput, Shake128, Update, XofReader};
    let mut hash = Shake128::default();
    hash.update(&bytes);
    let mut digest = [0u8; 32];
    hash.finalize_xof().read(&mut digest);
    assert_eq!(bytes.len(), 18370);
    assert_eq!(
        hex::encode(digest),
        "e8e5b346c49763c5bcd924129c55501a4786a65e170865f8f0e2017885134b56"
    );
}

#[test]
fn compiled_rng_proofs_are_fresh_and_bound_to_their_context() {
    // `prove` and `prove_bytes` draw a fresh seed from the RNG per call.
    let source = Ring::new(13, 128).unwrap();
    let compiled = possession(&source).compile(params()).unwrap();
    let witness = possession_witness(&source);
    let mut rng = common::SeededRng::new(b"statement");
    let proof = compiled
        .prove([33; 32], &witness, b"context A", &mut rng)
        .unwrap();
    compiled.verify([33; 32], &proof, b"context A").unwrap();
    assert!(compiled.verify([33; 32], &proof, b"context B").is_err());
    let bytes = compiled
        .prove_bytes([33; 32], &witness, b"context A", &mut rng)
        .unwrap();
    compiled
        .verify_bytes([33; 32], &bytes, b"context A")
        .unwrap();
    for other in [b"context B".as_slice(), b""] {
        assert!(compiled.verify_bytes([33; 32], &bytes, other).is_err());
    }
    let scheme = jali::abdlop::Abdlop::new([33; 32], params()).unwrap();
    let second = jali::codec::proof::decode(&scheme, &bytes).unwrap();
    assert_ne!(second.commitment.t_a, proof.commitment.t_a);
}

#[test]
fn linear_front_end_uses_no_committed_carries_and_rejects_invalid_partition() {
    let ring = Ring::new(13, 128).unwrap();
    let mut entries = vec![Poly::zero(ring.clone()); 10];
    entries[0] = Poly::constant(ring.clone(), 3);
    let a = PolyMat::new(ring.clone(), 1, 10, entries).unwrap();
    let t = PolyVec::new(ring.clone(), vec![Poly::constant(ring.clone(), -2)]).unwrap();
    let blocks = [
        lin::Block {
            name: "s".into(),
            length: 8,
            norm: Norm::L2Squared(128),
        },
        lin::Block {
            name: "x".into(),
            length: 2,
            norm: Norm::L2Squared(64),
        },
    ];
    let mut p = params();
    p.l = 0;
    p.mlwe_rank += 2;
    let compiled = lin::compile(&a, &t, &blocks, p.clone()).unwrap();
    let mut witness = BTreeMap::new();
    let mut s = vec![Poly::zero(ring.clone()); 8];
    s[0] = Poly::constant(ring.clone(), 5);
    witness.insert("s".into(), s);
    witness.insert("x".into(), vec![Poly::zero(ring); 2]);
    let (s, m) = compiled.map_witness(&witness).unwrap();
    assert!(m.is_empty());
    let arp = compiled
        .statement()
        .arp
        .as_ref()
        .unwrap()
        .evaluate(&s, &m)
        .unwrap();
    assert_eq!(arp.entries()[0].coefficient_i128(0), Ok(1));
    assert!(lin::compile(&a, &t, &blocks[..1], p).is_err());
}

/// `a (x_0 y_0) + b x_0 - c = 0` over the given degree-128 statement ring, with a binary `x`
/// and a Euclidean-bounded `y`. Four and two polynomials give the compiled witness the length
/// that completeness needs ((m_1 + Z) d >= 640); the other components are free.
fn wide_relation(source: &Arc<Ring>, a: &Poly, b: &Poly, c: &Poly) -> Statement {
    let mut statement = Statement::new(source.clone());
    statement.var("x", 4, Norm::Binary).unwrap();
    statement.var("y", 2, Norm::L2Squared(64)).unwrap();
    let x = statement.variable("x", 0).unwrap();
    let y = statement.variable("y", 0).unwrap();
    let equation = x
        .product_affine(&y)
        .unwrap()
        .scale(a)
        .unwrap()
        .add(&x.scale(b).unwrap())
        .unwrap()
        .add(&statement.constant(c.neg()).unwrap())
        .unwrap();
    statement.eq_mod_p(equation).unwrap();
    statement
}

#[test]
fn a_quadratic_relation_modulo_a_200_bit_prime_lifts_to_a_241_bit_proof_modulus() {
    // Statement modulus p = 2^200 - 75 at degree 128, proof modulus q = 2^240 + 325 at degree
    // 64 with test-only parameters (both prime, 5 mod 8, proven in Sage). The public
    // polynomials a and b are uniform modulo p, so their centred coefficients and the integer
    // value of the relation (about 2^211) exceed i128: the compiler evaluates it in 512 bits.
    use common::params::{FACTORS_240, even_divisors, fit_wide, power_minus, power_plus};
    use num_bigint::{BigInt, Sign};
    const CTX: &[u8] = b"jali-test/wide-statement";
    let p = power_minus(200, 75);
    let q = power_plus(240, 325);
    let source = Ring::with_modulus(p, 128).unwrap();
    let mut rng = common::SplitMix64(0x2567_0200);
    let mut uniform = || {
        let values = (0..128)
            .map(|_| common::from_u64_words([0; 4].map(|_: u64| rng.next())))
            .collect();
        Poly::from_u256(source.clone(), values).unwrap()
    };
    let (a, b) = (uniform(), uniform());
    let mut rng = common::SplitMix64(7);
    let x = Poly::new(
        source.clone(),
        (0..128).map(|_| (rng.next() % 2) as i128).collect(),
    )
    .unwrap();
    let y = Poly::new(
        source.clone(),
        (0..128)
            .map(|i| {
                if i % 3 == 0 {
                    (rng.next() % 3) as i128 - 1
                } else {
                    0
                }
            })
            .collect(),
    )
    .unwrap();
    assert!(y.norm_squared() <= U256::from_u8(64));
    let c = a
        .mul(&x.mul(&y).unwrap())
        .unwrap()
        .add(&b.mul(&x).unwrap())
        .unwrap();
    let statement = wide_relation(&source, &a, &b, &c);
    let req = statement.requirements(64, q).unwrap();
    // The integer bound exceeds 2^128, and the quotient bound is its ceiling over p.
    let big = |v: &U256| BigInt::from_bytes_le(Sign::Plus, &v.to_le_bytes());
    let (f, pb) = (big(&req.max_integer_coefficient), big(&p));
    assert!(f.bits() > 200 && f.bits() <= 256, "F has {} bits", f.bits());
    assert_eq!(BigInt::from(req.linf_bound), (&f + &pb - 1) / &pb);
    assert_eq!((req.l, req.n_prime), (2, 2));
    let params = fit_wide(
        &req,
        "wide-statement-test-only",
        q,
        &even_divisors(&q, &FACTORS_240),
        64,
    )
    .unwrap();
    let compiled = statement.compile(params.clone()).unwrap();
    let zero = Poly::zero(source.clone());
    let witness = BTreeMap::from([
        (
            "x".into(),
            vec![x.clone(), zero.clone(), zero.clone(), zero.clone()],
        ),
        ("y".into(), vec![y.clone(), zero]),
    ]);
    let (bounded, messages) = compiled.map_witness(&witness).unwrap();
    // The carry f / p, from the integer evaluation on centred representatives, computed here
    // with big integers, and split into the two degree-64 components.
    let centred = |poly: &Poly| -> Vec<BigInt> {
        poly.coefficients()
            .iter()
            .map(|v| {
                let v = big(v);
                if &v * 2 > pb { v - &pb } else { v }
            })
            .collect()
    };
    let negacyclic = |u: &[BigInt], v: &[BigInt]| {
        let mut out = vec![BigInt::from(0); 128];
        for (i, s) in u.iter().enumerate() {
            for (j, t) in v.iter().enumerate() {
                if i + j < 128 {
                    out[i + j] += s * t;
                } else {
                    out[i + j - 128] -= s * t;
                }
            }
        }
        out
    };
    let (ca, cb, cc, cx, cy) = (
        centred(&a),
        centred(&b),
        centred(&c),
        centred(&x),
        centred(&y),
    );
    let value: Vec<BigInt> = negacyclic(&ca, &negacyclic(&cx, &cy))
        .iter()
        .zip(negacyclic(&cb, &cx))
        .zip(&cc)
        .map(|((u, v), w)| u + v - w)
        .collect();
    assert!(value.iter().any(|v| v.bits() > 128));
    for (component, carry) in messages.entries().iter().enumerate() {
        for (j, expected) in value.iter().skip(component).step_by(2).enumerate() {
            assert_eq!(expected % &pb, BigInt::from(0));
            assert_eq!(
                BigInt::from(carry.coefficient_i128(j).unwrap()),
                expected / &pb,
                "component {component}, coefficient {j}"
            );
        }
    }
    let internal = quad::interleave(&bounded, &messages).unwrap();
    for equation in &compiled.statement().quadratic {
        assert!(equation.evaluate(&internal).unwrap().is_zero());
    }
    let proof = compiled
        .prove_with_seed([51; 32], &witness, CTX, [52; 32])
        .unwrap();
    compiled.verify([51; 32], &proof, CTX).unwrap();
    assert!(
        compiled
            .verify([51; 32], &proof, b"another context")
            .is_err()
    );
    // Pinned bytes. No other pinned proof has a modulus above 2^63, where the wide CRT runs, so
    // this fingerprint pins it and the samplers at a 241-bit modulus end to end.
    let scheme = jali::abdlop::Abdlop::new([51; 32], params.clone()).unwrap();
    let bytes = jali::codec::proof::encode(&scheme, &proof).unwrap();
    compiled.verify_bytes([51; 32], &bytes, CTX).unwrap();
    use shake::{ExtendableOutput, Shake128, Update, XofReader};
    let mut hash = Shake128::default();
    hash.update(&bytes);
    let mut digest = [0u8; 32];
    hash.finalize_xof().read(&mut digest);
    assert_eq!(bytes.len(), 40284);
    assert_eq!(
        hex::encode(digest),
        "7327d379269e58bd0b2d7837d94a1f9dcd2ecb9add47fe5ab9e629a426c1a637"
    );
    // A flipped bit of x breaks the relation modulo p, and the witness map refuses it.
    let mut wrong = witness.clone();
    let flipped = 1 - x.coefficient_i128(9).unwrap();
    wrong.get_mut("x").unwrap()[0]
        .set_coefficient(9, flipped)
        .unwrap();
    assert_eq!(compiled.map_witness(&wrong).err(), Some(Error::Witness));
    // The same shape modulo 2^230 - 5 needs a proof modulus above 2^250: the lifting check,
    // computed exactly in 1024 bits, refuses q.
    let large = Ring::with_modulus(power_minus(230, 5), 128).unwrap();
    let lift = |poly: &Poly| Poly::from_u256(large.clone(), poly.coefficients().to_vec()).unwrap();
    assert_eq!(
        wide_relation(&large, &lift(&a), &lift(&b), &lift(&c))
            .compile(params)
            .err(),
        Some(Error::Parameter("modulus lifting bound"))
    );
}

#[test]
fn integer_bounds_with_coefficients_above_2_128_match_big_integers() {
    // For `a x + b (x y) + c` with ||x||^2 <= bx and ||y||^2 <= by, the bound on the integer
    // value is F = |c|_inf + ceil(sqrt(||a||^2 bx)) + |b|_1 ceil(sqrt(bx by)) on centred
    // representatives. Squares of coefficients from 2^128 on need more than 256 bits. F from
    // 2^256 on is refused.
    use common::params::{power_minus, power_plus};
    use num_bigint::{BigInt, Sign};
    let big = |v: &U256| BigInt::from_bytes_le(Sign::Plus, &v.to_le_bytes());
    let ceil_sqrt = |x: BigInt| {
        let r = x.sqrt();
        if &r * &r == x { r } else { r + 1 }
    };
    let mut rng = common::SplitMix64(0x2567_0f00);
    let (mut exact, mut refused) = (0, 0);
    for p in [
        power_minus(100, 99),
        power_minus(200, 75),
        power_minus(255, 19),
    ] {
        let (pb, bits) = (big(&p), p.bits_vartime());
        let ring = Ring::with_modulus(p, 64).unwrap();
        let centred = |v: &BigInt| if v * 2 > pb { v - &pb } else { v.clone() };
        for trial in 0..8u64 {
            let mut below = || {
                let words: Vec<BigInt> = (0..5).map(|_| BigInt::from(rng.next())).collect();
                words.iter().fold(BigInt::from(0), |acc, w| (acc << 64) + w) % &pb
            };
            // Extremes, random values and values just above 2^129 (2^(bits - 2) for small p).
            let a: Vec<BigInt> = (0..64)
                .map(|i| match i % 4 {
                    0 => (&pb - 1) / 2,
                    1 => (&pb + 1) / 2,
                    2 => below(),
                    _ => (BigInt::from(1) << 129.min(bits - 2)) + i,
                })
                .collect();
            let b: Vec<BigInt> = (0..64)
                .map(|i| if i % 7 == 0 { below() } else { BigInt::from(0) })
                .collect();
            let c: Vec<BigInt> = (0..64).map(|_| below()).collect();
            let poly = |v: &[BigInt]| {
                let values = v
                    .iter()
                    .map(|x| {
                        let mut bytes = x.to_bytes_le().1;
                        bytes.resize(32, 0);
                        U256::from_le_slice(&bytes)
                    })
                    .collect();
                Poly::from_u256(ring.clone(), values).unwrap()
            };
            let (bx, by) = (1000 + trial, 77 + 3 * trial);
            let mut statement = Statement::new(ring.clone());
            statement.var("x", 1, Norm::L2Squared(bx)).unwrap();
            statement.var("y", 1, Norm::L2Squared(by)).unwrap();
            let x = statement.variable("x", 0).unwrap();
            let y = statement.variable("y", 0).unwrap();
            let form = x
                .scale(&poly(&a))
                .unwrap()
                .add(&x.product_affine(&y).unwrap().scale(&poly(&b)).unwrap())
                .unwrap()
                .add(&statement.constant(poly(&c)).unwrap())
                .unwrap();
            statement.eq_mod_p(form).unwrap();
            let magnitude = |v: &BigInt| BigInt::from(centred(v).magnitude().clone());
            let norm: BigInt = a.iter().map(|v| centred(v).pow(2)).sum();
            let l1: BigInt = b.iter().map(magnitude).sum();
            let inf = c.iter().map(magnitude).max().unwrap();
            let f = inf + ceil_sqrt(norm * bx) + l1 * ceil_sqrt(BigInt::from(bx) * by);
            let req = statement.requirements(64, power_plus(240, 325));
            if f.bits() > 256 {
                assert_eq!(req.err(), Some(Error::Overflow), "p={pb} trial {trial}");
                refused += 1;
                continue;
            }
            let req = req.unwrap();
            assert_eq!(big(&req.max_integer_coefficient), f, "p={pb} trial {trial}");
            assert_eq!(
                BigInt::from(req.linf_bound),
                (&f + &pb - 1) / &pb,
                "p={pb} trial {trial}"
            );
            exact += 1;
        }
    }
    assert_eq!((exact, refused), (16, 8));
}
