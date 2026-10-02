//! The bit weights of [`Norm::LinfExact`]: their subset sums are exactly $`[0,2\beta]`$, and the
//! greedy digits reach every value there (the proof is at `weights`). Exhaustive for
//! $`2\beta\le2^{12}`$; for random $`\beta<2^{63}`$, the facts the proof uses, which imply it,
//! and greedy digits of random values.
use super::*;

/// Every subset sum of `weights`, as a table of reachable values from 0 to their sum.
fn subset_sums(weights: &[u64]) -> Vec<bool> {
    let total = weights.iter().sum::<u64>() as usize;
    let mut reachable = vec![false; total + 1];
    reachable[0] = true;
    for w in weights {
        for x in (*w as usize..=total).rev() {
            reachable[x] |= reachable[x - *w as usize];
        }
    }
    reachable
}

/// The facts of the proof: the weights have the definition's value
/// $`\lfloor(B+2^b)/2^{b+1}\rfloor`$ (computed in `u128`), sum to $`B`$, and satisfy
/// $`1\le c_b\le1+\sum_{b'>b}c_{b'}`$; there are $`\mathrm{bitlen}(B)`$ of them.
fn check_facts(beta: u64) -> Vec<u64> {
    let b = 2 * u128::from(beta);
    let w = weights(beta);
    assert_eq!(
        w.len(),
        (u128::BITS - b.leading_zeros()) as usize,
        "beta {beta}"
    );
    assert_eq!(bit_count(beta), w.len());
    for (i, c) in w.iter().enumerate() {
        assert_eq!(
            u128::from(*c),
            (b + (1 << i)) >> (i + 1),
            "beta {beta}, weight {i}"
        );
        let after: u128 = w[i + 1..].iter().map(|x| u128::from(*x)).sum();
        assert!(
            *c >= 1 && u128::from(*c) <= after + 1,
            "beta {beta}, weight {i}"
        );
    }
    assert_eq!(w.iter().map(|x| u128::from(*x)).sum::<u128>(), b);
    w
}

/// The greedy digits of `x` are bits whose weighted sum is `x`.
fn check_digits(x: u64, w: &[u64]) {
    let mut bits = vec![7; w.len()];
    digits(x, w, &mut bits);
    assert!(bits.iter().all(|b| *b <= 1), "x {x}");
    let sum: u128 = bits.iter().zip(w).map(|(b, c)| u128::from(b * c)).sum();
    assert_eq!(sum, u128::from(x));
}

#[test]
fn the_subset_sums_are_exactly_zero_to_two_beta_up_to_2_to_the_12() {
    for beta in 1..=2048u64 {
        let w = check_facts(beta);
        let sums = subset_sums(&w);
        assert_eq!(sums.len() as u64, 2 * beta + 1);
        assert!(sums.iter().all(|x| *x), "beta {beta}");
        for x in 0..=2 * beta {
            check_digits(x, &w);
        }
    }
}

#[test]
fn the_facts_and_greedy_digits_hold_for_random_beta_below_2_to_the_63() {
    let mut state = 0x2567_2300u64;
    let mut next = || {
        // SplitMix64.
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };
    let mut betas = vec![1, 2, 3, (1 << 62) - 1, 1 << 62, (1 << 63) - 1];
    for _ in 0..100_000 {
        // A width w uniform in 1..=63, then a value uniform below 2^w (at least 1).
        let bits = next() % 63 + 1;
        betas.push((next() >> (64 - bits)).max(1));
    }
    for beta in betas {
        let w = check_facts(beta);
        for x in [0, 1, 2 * beta - 1, 2 * beta, next() % (2 * beta + 1)] {
            check_digits(x, &w);
        }
    }
}

#[test]
fn known_weights_and_what_powers_of_two_would_reach() {
    assert_eq!(weights(1), [1, 1]);
    assert_eq!(weights(5), [5, 3, 1, 1]);
    let mut w: Vec<u64> = (0..10).rev().map(|i| 1 << i).collect();
    w.push(1);
    assert_eq!(weights(512), w);
    assert_eq!(weights((1 << 63) - 1).len(), 64);
    // Plain powers of two for B = 10 reach 11 to 15, beyond beta = 5.
    let sums = subset_sums(&[1, 2, 4, 8]);
    assert_eq!(sums.len(), 16);
    assert!(sums.iter().all(|x| *x));
    assert_eq!(subset_sums(&weights(5)).len(), 11);
}
