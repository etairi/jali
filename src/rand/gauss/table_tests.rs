//! The base table (`cdf`), which `tools/kat/half_gaussian_cdf.py` writes from its definition:
//! a pinned digest, so that `cargo test` notices any change to an entry, and the entries
//! recomputed from the definition in `f64`.
use super::CDF;
use shake::{ExtendableOutput, Shake128, Update, XofReader};

#[test]
fn the_table_is_the_generated_one() {
    // SHAKE128 of the entries as 16-byte little-endian integers, 32 bytes; after regenerating
    // the table, `half_gaussian_cdf.py --digest` prints the new value.
    let mut h = Shake128::default();
    for entry in CDF {
        h.update(&entry.to_le_bytes());
    }
    let mut out = [0u8; 32];
    h.finalize_xof().read(&mut out);
    assert_eq!(
        hex::encode(out),
        "15f0b78dd3d891593add0c119dfee29541ed29f05f42b7a8c591d49f3818eb77"
    );
}

#[test]
fn the_entries_follow_their_definition() {
    // Entry j is 2^128 Pr[K > j] rounded, for Pr[K = i] proportional to exp(-200 i^2 / 961) on
    // i >= 0; the terms from i = 64 on sum to less than 2^-1000. In f64 the sums carry a
    // relative error of about 1e-14 at most, far below the tolerance.
    let rho: Vec<f64> = (0..64i32)
        .map(|i| (-200.0 * f64::from(i * i) / 961.0).exp())
        .collect();
    let total: f64 = rho.iter().sum();
    for (j, &entry) in CDF.iter().enumerate() {
        let expected = rho[j + 1..].iter().sum::<f64>() / total * 2f64.powi(128);
        assert!(
            (entry as f64 - expected).abs() <= 0.5 + 1e-12 * expected,
            "entry {j}: {entry}, expected about {expected}"
        );
    }
    assert_eq!(CDF.last(), Some(&0));
}
