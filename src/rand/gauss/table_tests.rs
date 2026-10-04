//! The base tables (`cdf`), which `tools/kat/half_gaussian_cdf.py` writes from their
//! definition: a pinned digest, so that `cargo test` notices any change to an entry, the
//! entries recomputed from the definition in `f64`, the order the sampler relies on, and the
//! range of Bernoulli operands that the enclosure's proof covers.
use super::{HALF_1_55, HALF_3_1};
use crypto_bigint::U256;
use shake::{ExtendableOutput, Shake128, Update, XofReader};

/// The tables with their names and the exponent factor $`n`$ of their definition: entry
/// $`j`$ is $`2^{256}\Pr[K>j]`$ rounded, for $`\Pr[K=i]`$ proportional to
/// $`\exp(-ni^2/961)`$.
const TABLES: [(&str, &[U256], f64); 2] = [
    ("HALF_3_1", &HALF_3_1, 50.0),
    ("HALF_1_55", &HALF_1_55, 200.0),
];

#[test]
fn the_tables_are_the_generated_ones() {
    // SHAKE128, 32 bytes, of each table in turn: its length as 8 little-endian bytes, then
    // the entries as 32-byte little-endian integers. After regenerating the tables,
    // `half_gaussian_cdf.py --digest` prints the new value.
    let mut h = Shake128::default();
    for (_, table, _) in TABLES {
        h.update(&(table.len() as u64).to_le_bytes());
        for entry in table {
            h.update(&entry.to_le_bytes());
        }
    }
    let mut out = [0u8; 32];
    h.finalize_xof().read(&mut out);
    assert_eq!(
        hex::encode(out),
        "e05107a8027292259444057dcd6b84212cca92ec60d0c69ac91efd58f8fa5239"
    );
    assert_eq!((HALF_3_1.len(), HALF_1_55.len()), (59, 30));
}

/// An entry as `f64`, from its little-endian bytes (not from limb words, whose width depends
/// on the platform).
fn to_f64(entry: &U256) -> f64 {
    entry
        .to_le_bytes()
        .iter()
        .rev()
        .fold(0.0, |x, byte| x * 256.0 + f64::from(*byte))
}

#[test]
fn the_entries_follow_their_definition() {
    // Pr[K = i] proportional to exp(-n i^2 / 961) on i >= 0; the terms from i = 160 on are
    // below 2^-1900 of the first. In f64 the sums carry a relative error of about 1e-14 at
    // most, far below the tolerance.
    for (name, table, n) in TABLES {
        let rho: Vec<f64> = (0..160i32)
            .map(|i| (-n * f64::from(i * i) / 961.0).exp())
            .collect();
        let total: f64 = rho.iter().sum();
        for (j, entry) in table.iter().enumerate() {
            let expected = rho[j + 1..].iter().sum::<f64>() / total * 2f64.powi(256);
            let entry = to_f64(entry);
            assert!(
                (entry - expected).abs() <= 0.5 + 1e-12 * expected,
                "{name} entry {j}: {entry}, expected about {expected}"
            );
        }
    }
}

#[test]
fn the_entries_decrease_strictly_to_a_final_zero() {
    // Then K = #{j : V < T_j} is the length of the run of entries above V, which the sampler
    // counts, and Pr[K > j] = T_j / 2^256. The last entry is 0 and the one before is not: a
    // table is as long as its rounding allows, no longer.
    for (name, table, _) in TABLES {
        assert!(table.windows(2).all(|w| w[0] > w[1]), "{name}");
        assert_eq!(table.last(), Some(&U256::ZERO), "{name}");
        assert!(table[table.len() - 2] > U256::ZERO, "{name}");
    }
}

#[test]
fn bernoulli_operands_stay_within_the_proven_range() {
    // The largest exponent of the Bernoulli test is x = 50 (2K + 1) / 961, for the largest
    // base sample K with sign 1 and offset 0 (see `bernoulli`). The enclosure's proof and its
    // assumption (A) need x < 16, and its documentation states x <= 8200/961; with 59 entries
    // K is at most 58 and x at most 5850/961.
    let largest = HALF_3_1.len() as u64 - 1;
    assert_eq!(largest, 58);
    assert!(50 * (2 * largest + 1) <= 8200);
}
