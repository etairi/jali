//! The public randomness: ranges, determinism, the projection distribution, and the challenge
//! set, whose operator-norm bound is checked against an independent floating-point evaluation
//! at the primitive $`2d`$-th roots of unity.
use super::*;

const Q: u64 = 4294967197;

fn eta_ring() -> Arc<Ring> {
    Ring::new(i128::from(Q), 64).unwrap()
}

#[test]
fn uniform_residues_are_in_range_and_deterministic_per_domain() {
    let seed = [7u8; 32];
    for q in [65557u64, Q, 281474976710597] {
        let mut a = Uniform::new(&seed, purpose::MATRIX_A, 3, q);
        let mut b = Uniform::new(&seed, purpose::MATRIX_A, 3, q);
        let mut c = Uniform::new(&seed, purpose::MATRIX_A, 4, q);
        let (mut x, mut y, mut z) = (vec![0; 9000], vec![0; 9000], vec![0; 9000]);
        a.fill(&mut x).unwrap();
        b.fill(&mut y).unwrap();
        c.fill(&mut z).unwrap();
        assert_eq!(x, y);
        assert_ne!(x, z);
        assert!(x.iter().all(|v| *v < q));
        // Both halves of the range occur.
        assert!(x.iter().any(|v| *v < q / 2) && x.iter().any(|v| *v > q / 2));
    }
}

#[test]
fn projection_entries_follow_the_binomial_law() {
    let mut p = Projection::new(&[1u8; 32], 0);
    let mut counts = [0u64; 3];
    let mut column = [0i8; 256];
    for _ in 0..2000 {
        p.column(&mut column).unwrap();
        for e in column {
            counts[(e + 1) as usize] += 1;
        }
    }
    let total = 2000.0 * 256.0;
    // 512000 draws: each frequency is within 0.5% of its probability (about 7 standard
    // deviations).
    for (count, prob) in counts.iter().zip([0.25, 0.5, 0.25]) {
        assert!((*count as f64 / total - prob).abs() < 0.005, "{counts:?}");
    }
    // Deterministic per seed and level.
    let (mut a, mut b) = (
        Projection::new(&[2u8; 32], 5),
        Projection::new(&[2u8; 32], 5),
    );
    let (mut x, mut y) = ([0i8; 256], [0i8; 256]);
    a.column(&mut x).unwrap();
    b.column(&mut y).unwrap();
    assert_eq!(x, y);
}

/// The largest $`|c(\zeta)|`$ over the primitive $`2d`$-th roots of unity, in `f64`: the
/// operator norm of multiplication by $`c`$.
fn operator_norm(c: &Sparse, d: usize) -> f64 {
    (0..d)
        .map(|m| {
            let angle = std::f64::consts::PI * (2 * m + 1) as f64 / d as f64;
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for &(pos, value) in &c.terms {
                re += value as f64 * (angle * pos as f64).cos();
                im += value as f64 * (angle * pos as f64).sin();
            }
            (re * re + im * im).sqrt()
        })
        .fold(0.0, f64::max)
}

#[test]
fn challenges_have_the_shape_and_operator_norm_of_the_set() {
    let set = ChallengeSet::for_degree(64).unwrap();
    assert_eq!(set.l2_squared(), 71);
    let ring = eta_ring();
    let mut draws = 0u64;
    let samples = 300;
    for i in 0..samples {
        let (c, n) = set.sample_counting(&[3u8; 32], i, &ring).unwrap();
        draws += u64::from(n);
        let ones = c.terms.iter().filter(|t| t.1.abs() == 1).count();
        let twos = c.terms.iter().filter(|t| t.1.abs() == 2).count();
        assert_eq!((ones, twos), (31, 10));
        let mut positions: Vec<usize> = c.terms.iter().map(|t| t.0).collect();
        positions.dedup();
        assert_eq!(positions.len(), 41);
        assert!(positions.iter().all(|p| *p < 64));
        let op = operator_norm(&c, 64);
        assert!(op <= 15.0 + 1e-9, "{op}");
        // Deterministic.
        assert_eq!(set.sample(&[3u8; 32], i, &ring).unwrap(), c);
    }
    // The filter accepts a fair fraction of the draws (it is a sufficient condition, so it
    // rejects more than an exact operator norm would). Recorded, not a security property.
    let mean = draws as f64 / f64::from(samples);
    assert!(mean < 40.0, "mean draws per challenge {mean}");
    eprintln!("challenge draws per accepted challenge: {mean:.2}");
}

#[test]
fn only_degree_64_has_a_challenge_set() {
    for d in [128, 256] {
        assert!(ChallengeSet::for_degree(d).is_err());
    }
}

#[test]
fn projection_blocks_start_where_the_stream_reaches_them() {
    let seed = [5u8; 32];
    let mut whole = Projection::new(&seed, 3);
    let mut columns = Vec::new();
    let mut column = [0u8; 64];
    for _ in 0..5 * PROJECTION_BLOCK + 17 {
        whole.column_bytes(&mut column).unwrap();
        columns.push(column);
    }
    for block in [0usize, 1, 2, 5] {
        let mut part = Projection::at(&seed, 3, block).unwrap();
        for want in columns
            .iter()
            .skip(block * PROJECTION_BLOCK)
            .take(PROJECTION_BLOCK + 17)
        {
            part.column_bytes(&mut column).unwrap();
            assert_eq!(&column, want, "block {block}");
        }
    }
}
