//! A small quadratic possession relation, with degree lowering and committed carries.
//! Seeds and parameters are test fixtures, not application randomness or fresh estimates.
use jali::{
    Error,
    math::{Poly, Ring, U256},
    params::TboxParams,
    statement::{Norm, Statement},
};
use shake::{ExtendableOutput, Shake128, Update, XofReader};
use std::{collections::BTreeMap, time::Instant};

fn main() -> Result<(), Error> {
    let source = Ring::new(13, 128)?;
    let mut statement = Statement::new(source.clone());
    statement.var("s", 8, Norm::L2Squared(128))?;
    statement.var("x", 2, Norm::L2Squared(64))?;
    let constant = |n| statement.constant(Poly::constant(source.clone(), n));
    let left = statement.variable("s", 0)?.add(&constant(-1)?)?;
    let right = constant(5)?.add(
        &statement
            .variable("x", 0)?
            .scale(&Poly::constant(source.clone(), -1))?,
    )?;
    let equation = left.product_affine(&right)?.add(&constant(-1)?)?;
    statement.eq_mod_p(equation)?;
    let requirements = statement.requirements(64, U256::from_u64(1099511627917))?;
    println!("{}", serde_json::to_string_pretty(&requirements).unwrap());
    // tools/params/lnp_params.py possession.request.json toy-d64.report.json
    let params = TboxParams::from_json(include_str!("fixtures/possession.json"))?;
    let compiled = statement.compile(params)?;
    let mut s = vec![Poly::zero(source.clone()); 8];
    s[0] = Poly::constant(source.clone(), 5);
    let mut x = vec![Poly::zero(source.clone()); 2];
    x[0] = Poly::constant(source, -5);
    let witness = BTreeMap::from([("s".into(), s), ("x".into(), x)]);
    // (5 - 1) * (5 - (-5)) - 1 = 39 = 13 * 3.
    let start = Instant::now();
    let bytes =
        compiled.prove_bytes_with_seed([33; 32], &witness, b"example/possession", [44; 32])?;
    println!("prove: {:?}; proof: {} bytes", start.elapsed(), bytes.len());
    let mut hash = Shake128::default();
    hash.update(&bytes);
    let mut digest = [0u8; 32];
    hash.finalize_xof().read(&mut digest);
    println!("proof SHAKE128-256: {}", hex::encode(digest));
    let start = Instant::now();
    compiled.verify_bytes([33; 32], &bytes, b"example/possession")?;
    println!("verify: {:?}", start.elapsed());
    Ok(())
}
