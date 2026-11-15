//! The Kyber-1024 key relation of the README, proved in the straight-line extractable mode
//! (Katsumata's transform) and in the plain mode, with both proofs' lengths and times.
//!
//! ```sh
//! cargo run --release --example extractable
//! ```
use jali::{
    Error,
    math::{Poly, Ring},
    params::{kyber1024_d64, kyber1024_d64_extraction},
    statement::{Norm, Statement},
};
use std::{collections::BTreeMap, time::Instant};

fn main() -> Result<(), Error> {
    let ring = Ring::new(3329, 256)?;
    // A deterministic toy instance: ternary w, pseudorandom A, t = -Aw.
    let mut x = 1u64;
    let mut sample = |m: u64| {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((x >> 33) % m) as i128
    };
    let w = (0..8)
        .map(|_| Poly::new(ring.clone(), (0..256).map(|_| sample(3) - 1).collect()))
        .collect::<Result<Vec<_>, _>>()?;
    let mut statement = Statement::new(ring.clone());
    statement.var("w", 8, Norm::L2Squared(2950))?;
    for _ in 0..4 {
        let row = (0..8)
            .map(|_| Poly::new(ring.clone(), (0..256).map(|_| sample(3329)).collect()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut t = Poly::zero(ring.clone());
        for (a, wj) in row.iter().zip(&w) {
            t = t.sub(&a.mul(wj)?)?;
        }
        let mut form = statement.constant(t)?;
        for (j, a) in row.iter().enumerate() {
            form = form.add(&statement.variable("w", j)?.scale(a)?)?;
        }
        statement.eq_mod_p(form)?;
    }
    let compiled = statement.compile(kyber1024_d64())?;
    let witness = BTreeMap::from([("w".to_string(), w)]);
    let public_seed = [7; 32];
    // The extraction set that goes with the parameter set: the LinHC's rank and modulus.
    let extraction = kyber1024_d64_extraction();
    let context = b"example/extractable";

    let start = Instant::now();
    let proof = compiled.prove_extractable_bytes_with_seed(
        public_seed,
        &extraction,
        &witness,
        context,
        [9; 32],
    )?;
    let proved = start.elapsed();
    let start = Instant::now();
    compiled.verify_extractable_bytes(public_seed, &extraction, &proof, context)?;
    let verified = start.elapsed();
    println!(
        "extractable: {} bytes, prove {:.1} ms, verify {:.1} ms",
        proof.len(),
        proved.as_secs_f64() * 1e3,
        verified.as_secs_f64() * 1e3
    );

    let start = Instant::now();
    let plain = compiled.prove_bytes_with_seed(public_seed, &witness, context, [9; 32])?;
    let proved = start.elapsed();
    let start = Instant::now();
    compiled.verify_bytes(public_seed, &plain, context)?;
    let verified = start.elapsed();
    println!(
        "plain:       {} bytes, prove {:.1} ms, verify {:.1} ms",
        plain.len(),
        proved.as_secs_f64() * 1e3,
        verified.as_secs_f64() * 1e3
    );
    // Neither proof verifies in the other mode.
    assert!(
        compiled
            .verify_extractable_bytes(public_seed, &extraction, &plain, context)
            .is_err()
    );
    assert!(compiled.verify_bytes(public_seed, &proof, context).is_err());
    Ok(())
}
