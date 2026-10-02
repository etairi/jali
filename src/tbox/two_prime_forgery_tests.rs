//! End-to-end forgeries over a two-prime modulus, with only the prime-modulus refusal of
//! `TboxParams::check` switched off. They show that the refusal prevents a real soundness break:
//! the unmodified verifier accepts proofs that a committed message is short or binary although
//! it is neither. The cheating prover follows `prove_with_seed` except that it uses the CRT
//! value u = (1 mod q1, -1 mod q2) as a sign, which squares to one, and projects the short
//! vector u*x instead of the committed long vector x.
use super::*;
use crate::{
    lnp::{AffineBlock, L2Block},
    math::PolyMat,
};

const Q1: i128 = 1048589;
const Q2: i128 = 4194389;
const REFUSAL: Error =
    Error::Parameter("binary, exact-norm and range blocks require a prime modulus");

fn params() -> crate::params::TboxParams {
    let mut p = crate::params::toy_d64();
    p.id = "two-prime-forgery-test-only".into();
    p.prime_factors = vec![U256::from_u64(Q1 as u64), U256::from_u64(Q2 as u64)];
    p.degree = 128;
    p.m2 = 45;
    p.n_msis = 7;
    p.log_sigma = [15, 11, 11, 8];
    p.gamma = 123280;
    p.d_bits = 9;
    p.n_bin = 0;
    p.l2_rows.clear();
    p.l2_bounds_squared.clear();
    p.n_prime = 0;
    p.linf_bound = 0;
    // m2 - n_msis - l - l_ext, with l_ext = 2 range slots + 1 sign + lambda/2 + 1 = 8.
    p.mlwe_rank = 28;
    p
}
/// The CRT value (1 mod q1, -1 mod q2): a square root of one that is not a sign.
fn crt_sign() -> i128 {
    (1..Q2)
        .map(|k| 1 + k * Q1)
        .find(|x| (x + 1) % Q2 == 0)
        .unwrap()
}
fn centred(x: i128, q: i128) -> i128 {
    let r = x.rem_euclid(q);
    if r > q / 2 { r - q } else { r }
}
/// The statement `block(m[0])` with a one-row selector of the first message polynomial.
fn select_first_message(scheme: &Abdlop) -> AffineBlock {
    let ring = scheme.ring().clone();
    let mut row = vec![Poly::zero(ring.clone()); scheme.message_len()];
    row[0] = Poly::constant(ring.clone(), 1);
    AffineBlock {
        rows: 1,
        s: None,
        m: Some(PolyMat::new(ring, 1, scheme.message_len(), row).unwrap()),
        offset: None,
    }
}

/// Run the toolbox with a forged packed sign, projecting `u * x` in place of the committed `x`.
/// With `reject`, the range response goes through honest rejection sampling; without it the
/// cheating prover only has to meet the verifier's norm bound.
fn forge(
    scheme: &Abdlop,
    statement: &Statement,
    bounded: &PolyVec,
    m: &PolyVec,
    exact: bool,
    reject: bool,
) -> Proof {
    let p = &scheme.parameters;
    let ring = scheme.ring().clone();
    let q = ring.modulus_i128();
    let u = crt_sign();
    let seed = [11u8; 32];
    let (_, opening) = scheme
        .commit_with_seed(bounded.clone(), m.clone(), seed)
        .unwrap();
    let slots = 256 / p.degree;
    let extended = scheme.extend_messages(slots + 1).unwrap();
    let forms = forms(scheme, &extended, statement).unwrap();
    let mut random = AesPrg::new(&seed, domain(crate::rand::secret::TOOLBOX, 0));
    for attempt in 0..4096u32 {
        let y = abdlop::sample_gaussian(
            ring.clone(),
            slots,
            p.log_sigma[if exact { 2 } else { 3 }],
            &seed,
            crate::rand::secret::RANGE_MASKS + 2 * attempt + u32::from(!exact),
        )
        .unwrap();
        // Packed signs b_e - X^(d/2) b_d: the forged sign u on the block being attacked.
        let (be, bd) = if exact { (u, 1) } else { (1, u) };
        let packed = Poly::constant(ring.clone(), be)
            .sub(&Poly::constant(ring.clone(), bd).rotate((p.degree / 2) as i64))
            .unwrap();
        let mut messages = m.entries().to_vec();
        messages.extend(y.values.entries().iter().cloned());
        messages.push(packed);
        let messages = PolyVec::new(ring.clone(), messages).unwrap();
        let witness = quad::interleave(bounded, &messages).unwrap();
        let (commitment, full_opening) = extended
            .commit_with_randomness(bounded.clone(), messages, opening.s2.clone())
            .unwrap();
        let mut prefix = round_prefix(&extended, &commitment, &forms, b"").unwrap();
        let projection_seed = prefix.challenge_seed(b"range-matrices");
        let affines = if exact { &forms.exact } else { &forms.approx };
        // sign * R * x = u * R * x = R * (u x) modulo q: project the short vector u x.
        let short: Vec<i128> = affines
            .iter()
            .flat_map(|f| {
                f.evaluate(&witness)
                    .unwrap()
                    .coefficients_i128()
                    .unwrap()
                    .to_vec()
            })
            .map(|x| centred(x * u % q, q))
            .collect();
        let response = if reject {
            range_response(scheme, &projection_seed, exact, &short, &y, 1, &mut random).unwrap()
        } else {
            let v = projections(&projection_seed, exact, &short).unwrap();
            let z: Vec<i128> = abdlop::flatten(&y.values)
                .unwrap()
                .iter()
                .zip(&v)
                .map(|(a, b)| a + b)
                .collect();
            (crate::math::int::squared_norm(&z).unwrap()
                <= U256::from(scheme.checked.z3_bound_squared))
            .then(|| {
                PolyVec::new(
                    ring.clone(),
                    z.chunks_exact(p.degree)
                        .map(|c| Poly::new(ring.clone(), c.to_vec()).unwrap())
                        .collect(),
                )
                .unwrap()
            })
        };
        let Some(z) = response else {
            continue;
        };
        let empty = PolyVec::zero(ring.clone(), 0);
        let (ze, zd) = if exact { (z, empty) } else { (empty, z) };
        let evals = projection_equations(&extended, &forms, &projection_seed, &ze, &zd).unwrap();
        bind_responses(&mut prefix, &ze, &zd).unwrap();
        let mut proof_seed = [0u8; 32];
        random.fill(&mut proof_seed).unwrap();
        let evaluation = quad_eval::prove_with_seed(
            &extended,
            &commitment,
            &full_opening,
            &forms.eqs,
            &evals,
            &prefix.digest(),
            proof_seed,
        )
        .unwrap();
        return Proof {
            commitment,
            z_exact: ze,
            z_approx: zd,
            evaluation,
        };
    }
    panic!("forgery attempts exhausted");
}
/// Encode, decode and verify with the unmodified verifier.
fn accepted(scheme: &Abdlop, statement: &Statement, proof: &Proof) -> bool {
    let bytes = crate::codec::proof::encode(scheme, proof).unwrap();
    let decoded = crate::codec::proof::decode(scheme, &bytes).unwrap();
    verify(scheme, statement, &decoded, b"").is_ok()
}

#[test]
fn two_prime_range_block_accepts_a_long_message() {
    let mut p = params();
    p.n_prime = 1;
    p.linf_bound = 4;
    assert_eq!(p.check().err(), Some(REFUSAL));
    let scheme = Abdlop::new_allowing_composite_range_blocks([7; 32], p.clone()).unwrap();
    let ring = scheme.ring().clone();
    let q = ring.modulus_i128();
    let statement = Statement {
        arp: Some(select_first_message(&scheme)),
        ..Statement::default()
    };
    // m[0] = u (1 + X): the statement claims |m[0]|_inf <= 4.
    let mut short = vec![0i128; p.degree];
    short[0] = 1;
    short[1] = 1;
    let long = Poly::new(ring.clone(), short)
        .unwrap()
        .scale(crt_sign())
        .unwrap();
    assert!(long.norm_infinity() > U256::from_u128((q / 3) as u128));
    let mut messages = vec![Poly::zero(ring.clone()); scheme.message_len()];
    messages[0] = long;
    let m = PolyVec::new(ring.clone(), messages).unwrap();
    let s1 = PolyVec::zero(ring, p.m1);
    assert_eq!(
        prove_with_seed(&scheme, &statement, &s1, &m, b"", [9; 32]).err(),
        Some(Error::Witness)
    );
    let proof = forge(&scheme, &statement, &s1, &m, false, true);
    assert!(accepted(&scheme, &statement, &proof));
}

#[test]
fn two_prime_exact_norm_block_accepts_a_long_message() {
    let mut p = params();
    p.l2_rows = vec![1];
    p.l2_bounds_squared = vec![64];
    assert_eq!(p.check().err(), Some(REFUSAL));
    let scheme = Abdlop::new_allowing_composite_range_blocks([7; 32], p.clone()).unwrap();
    let ring = scheme.ring().clone();
    let q = ring.modulus_i128();
    let statement = Statement {
        l2: vec![L2Block {
            map: select_first_message(&scheme),
            bound_squared: 64,
        }],
        ..Statement::default()
    };
    // m[0] = 8u: its square is 64 modulo q, but its coefficient is about q/2.4.
    let long = Poly::constant(ring.clone(), 8 * crt_sign());
    assert!(long.norm_infinity() > U256::from_u128((q / 3) as u128));
    let mut messages = vec![Poly::zero(ring.clone()); scheme.message_len()];
    messages[0] = long;
    let m = PolyVec::new(ring.clone(), messages).unwrap();
    assert_eq!(
        prove_with_seed(
            &scheme,
            &statement,
            &PolyVec::zero(ring.clone(), p.m1),
            &m,
            b"",
            [9; 32]
        )
        .err(),
        Some(Error::Witness)
    );
    // The slack polynomial of 64 - 8^2 is zero.
    let bounded = PolyVec::zero(ring, scheme.bounded_len());
    let proof = forge(&scheme, &statement, &bounded, &m, true, true);
    assert!(accepted(&scheme, &statement, &proof));
}

#[test]
fn two_prime_binary_block_accepts_a_nonbinary_message() {
    let mut p = params();
    p.n_bin = 1;
    assert_eq!(p.check().err(), Some(REFUSAL));
    let scheme = Abdlop::new_allowing_composite_range_blocks([7; 32], p.clone()).unwrap();
    let ring = scheme.ring().clone();
    let q = ring.modulus_i128();
    let statement = Statement {
        binary: Some(select_first_message(&scheme)),
        ..Statement::default()
    };
    // e' with sum e'^2 = 8388745 = 8 q1 + 33 = 2 q2 - 33 and sum e' = 33, so
    // <u e', u e' - 1> = sum e'^2 - u sum e' = 0 modulo q although u e' is not binary.
    let mut e = vec![256i128; 64];
    e.extend(vec![-256i128; 64]);
    e[0] -= 40;
    e[64] += 21;
    e[1] += 52;
    assert_eq!(
        (
            e.iter().map(|x| x * x).sum::<i128>(),
            e.iter().sum::<i128>()
        ),
        (8388745, 33)
    );
    let long = Poly::new(ring.clone(), e.iter().map(|v| v * crt_sign()).collect()).unwrap();
    assert!(long.coefficients().iter().any(|x| *x > U256::ONE));
    assert!(long.norm_infinity() > U256::from_u128((q / 3) as u128));
    let mut messages = vec![Poly::zero(ring.clone()); scheme.message_len()];
    messages[0] = long;
    let m = PolyVec::new(ring.clone(), messages).unwrap();
    let s1 = PolyVec::zero(ring, p.m1);
    assert_eq!(
        prove_with_seed(&scheme, &statement, &s1, &m, b"", [9; 32]).err(),
        Some(Error::Witness)
    );
    // The projected vector e' is not short enough for honest rejection sampling; a cheating
    // prover only needs the verifier's bound on the response.
    let proof = forge(&scheme, &statement, &s1, &m, true, false);
    assert!(accepted(&scheme, &statement, &proof));
}
