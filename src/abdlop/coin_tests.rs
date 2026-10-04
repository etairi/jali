//! The rejection coins of the opening proof: attempt $`a`$ reads bytes $`64a`$ to $`64a+63`$
//! of its coin stream, the 256-bit coin of Rej_1 for $`z_1`$ and then that of Rej_2 for
//! $`z_2`$, and each test decides on every bit of its coin.
use super::*;
use crate::{par::Abandon, rand::ByteStream};

type Fixture = (Abdlop, Commitment, Opening, Transcript);

/// An opening of the restart tests' shape and its transcript prefix.
fn fixture() -> Fixture {
    let scheme = Abdlop::new([91; 32], crate::params::toy_d64()).unwrap();
    let ring = scheme.ring().clone();
    let s1 = PolyVec::new(
        ring.clone(),
        vec![Poly::constant(ring.clone(), 1); scheme.bounded_len()],
    )
    .unwrap();
    let m = PolyVec::new(
        ring.clone(),
        vec![Poly::constant(ring, 3); scheme.message_len()],
    )
    .unwrap();
    let (commitment, opening) = scheme.commit_with_seed(s1, m, [92; 32]).unwrap();
    let prefix = scheme.prefix(&commitment, b"coins").unwrap();
    (scheme, commitment, opening, prefix)
}

/// The largest coin that a test takes, for a test that takes 0 and refuses `U256::MAX`.
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

/// The two rejection tests of attempt `a`, recomputed as `attempt` runs them, each as the
/// function of its coin.
fn tests_of_attempt(
    (scheme, _, opening, prefix): &Fixture,
    key: &[u8; 32],
    a: u32,
) -> [impl Fn(U256) -> bool; 2] {
    let constants = scheme.rejection_constants().unwrap();
    let mask = |len, t, word| sample_gaussian(scheme.ring.clone(), len, t, key, word).unwrap();
    let p = &scheme.parameters;
    let y1 = mask(
        scheme.bounded_len(),
        p.log_sigma[0],
        secret::OPENING_MASKS + 2 * a,
    );
    let y2 = mask(p.m2, p.log_sigma[1], secret::OPENING_MASKS + 2 * a + 1);
    let y21 = part(&y2.values, 0, scheme.a2.cols()).unwrap();
    let y22 = part(&y2.values, scheme.a2.cols(), y2.values.len()).unwrap();
    let w = add(
        &add(
            &scheme.a1.mul(&y1.values).unwrap(),
            &scheme.a2.mul(&y21).unwrap(),
        )
        .unwrap(),
        &y22,
    )
    .unwrap();
    let w1 = map(&w, |x| scheme.compression.decompose(x).0).unwrap();
    let c = scheme.challenge(prefix, &w1).unwrap();
    let test = |y: &Mask, s: &PolyVec, m: U256, policy: Policy| {
        let v = scale(s, &c).unwrap();
        let z = add(&y.values, &v).unwrap();
        let (dot, norm) = reject::moments(&flatten(&z).unwrap(), &flatten(&v).unwrap()).unwrap();
        let variance = y.variance;
        move |u| reject::accept(policy, dot, norm, variance, m, u).unwrap()
    };
    [
        test(&y1, &opening.s1, constants[0], Policy::Standard),
        test(&y2, &opening.s2, constants[1], Policy::Rej2),
    ]
}

/// Coin bytes with `first` and `second` little-endian, in that order.
fn coins(first: &U256, second: &U256) -> [u8; OPENING_COIN_BYTES] {
    let mut bytes = [0u8; OPENING_COIN_BYTES];
    bytes[..reject::COIN_BYTES].copy_from_slice(&first.to_le_bytes()[..]);
    bytes[reject::COIN_BYTES..].copy_from_slice(&second.to_le_bytes()[..]);
    bytes
}

#[test]
fn an_attempt_decides_with_two_whole_256_bit_coins_in_order() {
    let f = fixture();
    let (scheme, _, opening, prefix) = &f;
    let key = Abdlop::opening_key(opening, Caller::Opening, &prefix.digest(), &[5; 32]);
    let constants = scheme.rejection_constants().unwrap();
    let challenge = |_: &PolyVec, _: &PolyVec, w1: &PolyVec| -> Result<(Poly, ()), Error> {
        Ok((scheme.challenge(prefix, w1)?, ()))
    };
    let run = |a: u32, bytes: &[u8; OPENING_COIN_BYTES]| {
        scheme
            .attempt(
                opening,
                &key,
                &constants,
                a,
                bytes,
                &challenge,
                &Abandon::never(),
            )
            .unwrap()
            .map(|(proof, ())| proof)
    };
    let low_half = U256::ONE.shl_vartime(128).wrapping_sub(&U256::ONE);
    let mut checked = 0;
    for a in 0..64 {
        // An attempt that passes its bound checks. A coin of 0 passes every test that can pass,
        // and Rej_2 cannot for a negative inner product.
        let Some(proof) = run(a, &coins(&U256::ZERO, &U256::ZERO)) else {
            continue;
        };
        let [first, second] = tests_of_attempt(&f, &key, a);
        if first(U256::MAX) || second(U256::MAX) {
            continue; // a test that takes every coin has no threshold
        }
        let (t1, t2) = (largest_accepted(&first), largest_accepted(&second));
        let one = U256::ONE;
        assert_ne!(t1.wrapping_add(&one).bitand(&low_half), U256::ZERO, "{a}");
        assert_ne!(t2.wrapping_add(&one).bitand(&low_half), U256::ZERO, "{a}");
        // Both coins at their thresholds: accepted, with the same proof. One unit above either
        // threshold, in the coin's low half: refused.
        assert_eq!(run(a, &coins(&t1, &t2)), Some(proof), "{a}");
        assert_eq!(run(a, &coins(&t1.wrapping_add(&one), &t2)), None, "{a}");
        assert_eq!(run(a, &coins(&t1, &t2.wrapping_add(&one))), None, "{a}");
        checked += 1;
    }
    assert!(checked >= 3, "{checked}");
}

#[test]
fn attempt_a_reads_bytes_64a_to_64a_plus_63_of_the_coin_stream() {
    let f = fixture();
    let (scheme, commitment, opening, prefix) = &f;
    let challenge = |_: &PolyVec, _: &PolyVec, w1: &PolyVec| -> Result<(Poly, ()), Error> {
        Ok((scheme.challenge(prefix, w1)?, ()))
    };
    let constants = scheme.rejection_constants().unwrap();
    let mut later = 0;
    for seed in 0u8..16 {
        let (index, proof, ()) = scheme
            .prove_core_speculative(
                commitment,
                opening,
                Caller::Opening,
                &prefix.digest(),
                &Zeroizing::new([seed; 32]),
                secret::MAX_ATTEMPTS,
                1,
                challenge,
            )
            .unwrap();
        // The sequential loop, reading 64 coin bytes per attempt.
        let key = Abdlop::opening_key(opening, Caller::Opening, &prefix.digest(), &[seed; 32]);
        let mut stream = AesPrg::new(&key, domain(secret::OPENING_COINS, 0));
        let accepted = (0..)
            .find_map(|a| {
                let mut bytes = [0u8; OPENING_COIN_BYTES];
                stream.fill(&mut bytes).unwrap();
                scheme
                    .attempt(
                        opening,
                        &key,
                        &constants,
                        a,
                        &bytes,
                        &challenge,
                        &Abandon::never(),
                    )
                    .unwrap()
                    .map(|(proof, ())| (a, proof))
            })
            .unwrap();
        assert_eq!(accepted, (index, proof), "seed {seed}");
        later += usize::from(index > 0);
    }
    // Some seeds accept after a rejected attempt, so later attempts' offsets are checked too.
    assert!(later >= 8, "{later}");
}
