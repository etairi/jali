//! Two claims of the enclosure's proof that the sampler's outputs do not pin: steps 1 and 2,
//! $`X^-\le x2^{64}\le X^+`$, checked exactly with big integers on sampler inputs, and the
//! decision rule of [`decide`], which takes exactly the decisions the proof covers. With
//! $`X^-`$ computed with $`K^+`$, or with a reject threshold of $`h\ge2U`$, every other test
//! passes (checked by mutation), as no tested decision changes, but the proof would no longer
//! cover the result.
use super::{decide, exp_bounds, x_bounds};
use num_bigint::BigUint;

/// The operands $`(a,v)`$ of the trial for remainder `u`, base sample `k` and sign `b`, as the
/// sampler forms them.
fn operands(u: u128, k: i128, b: i128, t: u32) -> (u128, u128) {
    let k = if b == 1 { k + 1 } else { -k };
    let scale = 1i128 << t;
    (
        (k * scale - u as i128).unsigned_abs(),
        ((k - b) * scale).unsigned_abs(),
    )
}

/// Sampler inputs at width exponent `t`: every base sample (0 to 20) and sign, with every
/// remainder when $`t\le`$ `all`, and otherwise with the remainders at the ends, the middle and
/// `random` random ones.
fn inputs(t: u32, all: u32, random: usize, state: &mut u64) -> Vec<(u128, u128)> {
    let top = (1u128 << t) - 1;
    let remainders: Vec<u128> = if t <= all {
        (0..=top).collect()
    } else {
        let mut next = || {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *state
        };
        let mut r = vec![0, top, top / 2 + 1];
        r.extend((0..random).map(|_| (u128::from(next()) << 64 | u128::from(next())) & top));
        r
    };
    let mut out = Vec::new();
    for k in 0..=20 {
        for b in 0..=1 {
            out.extend(remainders.iter().map(|u| operands(*u, k, b, t)));
        }
    }
    out
}

#[test]
fn x_bounds_enclose_x_exactly() {
    let mut state = 1;
    for t in 0..=crate::rand::MAX_LOG_SIGMA {
        for (a, v) in inputs(t, 10, 300, &mut state) {
            let (low, high) = x_bounds(a, v, t).expect("sampler inputs");
            // x 2^64 = 200 (a - v)(a + v) 2^64 / (961 4^t), compared cross-multiplied.
            let scaled = (BigUint::from(200u32) * (a - v) * (a + v)) << 64u32;
            let denominator = BigUint::from(961u32) << (2 * t);
            assert!(
                BigUint::from(low) * &denominator <= scaled,
                "X- at t {t}, a {a}, v {v}"
            );
            assert!(
                scaled <= BigUint::from(high) * &denominator,
                "X+ at t {t}, a {a}, v {v}"
            );
        }
    }
}

/// Accept exactly when $`h+2\le2L`$ and reject exactly when $`h\ge2U+1`$, at and around both
/// thresholds and at the ends of the range: outside these the proof covers no decision, and
/// inside them a missed decision runs the exact test more often than for the documented
/// $`2(U-L)+2`$ top words.
#[test]
fn decide_takes_exactly_the_decisions_the_proof_covers() {
    let mut state = 2;
    for t in 0..=crate::rand::MAX_LOG_SIGMA {
        for (a, v) in inputs(t, 6, 40, &mut state) {
            let [lower, upper] = exp_bounds(a, v, t).expect("sampler inputs");
            let mut words = vec![0, 1, 2, u128::from(u64::MAX)];
            for threshold in [2 * lower, 2 * upper] {
                words.extend((0..4).flat_map(|d| [threshold.saturating_sub(d), threshold + d]));
            }
            for h in words.into_iter().filter(|h| *h <= u128::from(u64::MAX)) {
                let covered = if h + 2 <= 2 * lower {
                    Some(true)
                } else if h > 2 * upper {
                    Some(false)
                } else {
                    None
                };
                assert_eq!(
                    decide(a, v, t, h as u64),
                    covered,
                    "t {t}, a {a}, v {v}, h {h}"
                );
            }
        }
    }
}
