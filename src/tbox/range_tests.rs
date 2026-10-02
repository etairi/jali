//! LNP22 Lemma 2.7 (from LNS21a), on which extraction from the approximate range proof rests:
//! for $`w\in\mathbb Z_q^m`$ and $`y\in\mathbb Z_q^k`$,
//! $`\Pr_{R\leftarrow\mathrm{Bin}_1^{k\times m}}[\|Rw+y\|_\infty<\|w\|_\infty/2]\le2^{-k}`$.
//! For one row the bound is 1/2, and it is tight for a vector with one nonzero coefficient,
//! which a row misses exactly when its coefficient there is zero. Extraction gives
//! `approx_extraction_bound` $`=2\cdot`$`z4_bound`: a vector with a coefficient above it
//! projects within `z4_bound` in a row with probability at most 1/2, and in all 256 rows with
//! probability at most $`2^{-256}`$. This test measures that frequency for the prover's `row`.
use super::*;

/// Frequency, over `seeds` projection seeds and 256 rows each, of $`|\langle r,w\rangle+y|\le`$
/// `z4_bound`, with $`y`$ a Gaussian mask coefficient at width `log_sigma[3]`.
fn frequency_below(scheme: &Abdlop, w: &[i128], seeds: u8) -> (f64, usize) {
    let z4 = scheme.checked.z4_bound as i128;
    let mut below = 0usize;
    let mut most = 0usize;
    for s in 0..seeds {
        let seed = [s; 32];
        let y = crate::rand::gaussian(
            &mut AesPrg::new(&[s ^ 0x55; 32], 0),
            scheme.parameters.log_sigma[3],
            256,
        )
        .unwrap();
        let v = projections(&seed, false, w).unwrap();
        let count = v
            .iter()
            .zip(&y)
            .filter(|(v, y)| (*v + *y).abs() <= z4)
            .count();
        below += count;
        most = most.max(count);
    }
    (below as f64 / (256.0 * f64::from(seeds)), most)
}

#[test]
fn a_projection_row_misses_a_long_vector_with_probability_at_most_one_half() {
    let scheme = Abdlop::new([1; 32], crate::params::toy_d64()).unwrap();
    let z4 = scheme.checked.z4_bound as i128;
    assert_eq!(
        scheme.checked.approx_extraction_bound,
        2 * scheme.checked.z4_bound
    );
    // n'd = 128 coefficients, as the approximate block of toy_d64 projects.
    let m = scheme.parameters.n_prime * scheme.parameters.degree;
    let seeds = 40;
    // 10,240 rows: five standard deviations of a frequency near 1/2 are 0.025.
    let slack = 5.0 * (0.25f64 / (256.0 * f64::from(seeds))).sqrt();
    let mut single = vec![0i128; m];
    single[17] = 2 * z4 + 1;
    let (frequency, most) = frequency_below(&scheme, &single, seeds);
    assert!(frequency <= 0.5 + slack, "single coefficient: {frequency}");
    // Tight: the rows are Bin_1, zero with probability exactly 1/2.
    assert!(frequency >= 0.5 - slack, "single coefficient: {frequency}");
    assert!(most < 256);
    // Several long coefficients are missed less often than one.
    let mut several = vec![0i128; m];
    for (i, x) in several.iter_mut().enumerate().step_by(13) {
        *x = if i % 2 == 0 { 2 * z4 + 1 } else { -3 * z4 };
    }
    let (frequency, _) = frequency_below(&scheme, &several, seeds);
    assert!(
        frequency <= 0.5 + slack,
        "several coefficients: {frequency}"
    );
    // A vector within z4_bound / 256 per coefficient always projects within z4_bound up to
    // the mask, so the test above is not vacuous: short vectors do pass.
    let short = vec![z4 / 512; m];
    let (frequency, _) = frequency_below(&scheme, &short, 4);
    assert!(frequency > 0.99, "short vector: {frequency}");
}
