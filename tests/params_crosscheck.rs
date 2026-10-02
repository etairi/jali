//! The test port of the parameter tool (`common::params::fit`) against the tool itself: for
//! every request and MLWE report checked in with an output of `tools/params/lnp_params.py`,
//! the port derives the same parameter set, and where the tool refuses a prime, so does the
//! port. Regenerate the outputs with the tool, as `tools/README.md` shows. The
//! `tag-preimage-crt` fixtures describe the CRT form of `tests/tag_preimage.rs` with one carry
//! polynomial per tag clause (`l` and `n_prime` 12 at degree 128 and 20 at degree 64, where the
//! compiler packs the tag carries into 9 and 17); they cross-check the port and the tool, and
//! no test compiles a statement with them.
#![cfg(feature = "serde")]
use jali::{params::TboxParams, statement::Requirements};
use serde_json::Value;

mod common;
use common::params::fit;

/// Requirements, id, prime factors and degree of a request.
fn request(text: &str) -> (Requirements, String, Vec<u64>, usize) {
    let v: Value = serde_json::from_str(text).unwrap();
    let n = |k: &str| v[k].as_u64().unwrap();
    let list = |k: &str| -> Vec<u64> {
        v[k].as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap())
            .collect()
    };
    let req = Requirements {
        m1: n("m1") as usize,
        l: n("l") as usize,
        alpha_squared: u128::from(n("alpha_squared")),
        n_bin: n("n_bin") as usize,
        l2_rows: list("l2_rows").into_iter().map(|x| x as usize).collect(),
        l2_bounds_squared: list("l2_bounds_squared"),
        n_prime: n("n_prime") as usize,
        linf_bound: u128::from(n("linf_bound")),
        max_integer_coefficient: jali::math::U256::ZERO,
        approx_alpha_squared: None,
        lifted_moduli: Vec::new(),
        linf: None,
    };
    (
        req,
        v["id"].as_str().unwrap().into(),
        list("prime_factors"),
        n("degree") as usize,
    )
}
/// The port's output under a report: its rank, delta and provenance replace the synthetic ones.
fn derive(request_text: &str, report_text: &str) -> Result<TboxParams, String> {
    let (req, id, factors, degree) = request(request_text);
    let report: Value = serde_json::from_str(report_text).unwrap();
    assert_eq!(report["degree"].as_u64().unwrap() as usize, degree);
    let modulus: u128 = factors.iter().map(|p| u128::from(*p)).product();
    assert_eq!(u128::from(report["modulus"].as_u64().unwrap()), modulus);
    let rank = report["rank"].as_u64().unwrap() as usize;
    let mut params = fit(&req, &id, factors, degree, rank)?;
    params.mlwe_delta = report["delta"].as_f64().unwrap();
    params.estimator = report["provenance"].as_str().unwrap().into();
    Ok(params)
}

macro_rules! fixture {
    ($name:literal) => {
        (
            include_str!(concat!("fixtures/params/", $name, ".request.json")),
            include_str!(concat!("fixtures/params/", $name, ".report.json")),
            include_str!(concat!("fixtures/params/", $name, ".json")),
        )
    };
}

#[test]
fn the_port_derives_every_checked_in_tool_output() {
    let report = include_str!("../tools/params/toy-d64.report.json");
    let toy = (
        include_str!("../tools/params/toy-d64.request.json"),
        report,
        include_str!("../src/params/sets/toy-d64.json"),
    );
    let possession = (
        include_str!("../tools/params/possession.request.json"),
        report,
        include_str!("../examples/fixtures/possession.json"),
    );
    for (name, (request, report, output)) in [
        ("toy-d64", toy),
        ("possession", possession),
        ("tag-preimage-crt-d128", fixture!("tag-preimage-crt-d128")),
        ("tag-preimage-crt-d64", fixture!("tag-preimage-crt-d64")),
        ("dense-blocks-7", fixture!("dense-blocks-7")),
        ("decoder-ranges", fixture!("decoder-ranges")),
    ] {
        let tool = TboxParams::from_json(output).unwrap();
        assert_eq!(derive(request, report).unwrap(), tool, "{name}");
    }
}

#[test]
fn the_port_refuses_a_prime_that_the_tool_refuses() {
    assert_eq!(
        derive(
            include_str!("fixtures/params/tag-preimage-crt-d128-no-divisor.request.json"),
            include_str!("fixtures/params/tag-preimage-crt-d128-no-divisor.report.json"),
        )
        .err()
        .as_deref(),
        Some("no suitable divisor; choose another prime and rerun the estimator")
    );
}
