#!/usr/bin/env python3
"""Regenerate the Rust reference data of the parameter tool's tests. Needs cargo.

    python3 tools/params/rust_reference.py [--crate DIR] [--work DIR] [--keep] [--data DIR]
                                           [--no-prove] [--check]

Copies the crate (default: this repository) into a fresh directory under `--work`
(default: the system's temporary directory), adds the verifier `VERIFIER` below as an example,
builds it in release mode with the `serde` feature and its own `CARGO_TARGET_DIR`, and writes

- `tests/data/rust_check_cases.json`: the cases of `cases()` with Rust's result for each:
  `TboxParams::from_json` (an encoding error or the error of `check`), the checked values, and
  for cases with a statement `lin::compile` of the worst-case statement $`Aw+t=0`$ (4 x 8 over
  degree 256, every public coefficient $`\\lfloor p/2\\rfloor`$);
- `tests/data/rust_reference.json`: for each set under `sets/`, the checked values, the
  worst-case requirements and compile result, the same for the previous prime of the set's
  $`\\gamma`$ class, and one proof and verification of a seeded random instance; and for the
  statement kinds of the verifier (two constraint moduli, $`\\ell_\\infty`$ variables, a block
  in the BDLOP part, 65 constant-coefficient clauses whose carries fill two packed
  polynomials, subring variables, variables bounded exactly by bits, and the refusals of
  their compilation), Rust's requirements and blocks, the set the tool derives from them,
  Rust's compilation, proof and verification of it, and Rust's decision on mutations of that
  set (`statement_cases`).

With `--check` nothing is written: the fresh results are compared with the checked-in data
(provenance aside), every case, statement case, set and statement kind that differs is named,
and the exit status is 1 if anything differs, that is, if the crate's checks no longer decide
these cases as recorded or the tool no longer derives the recorded sets. The f64 outputs come
from the platform's libm (`log2`, `exp`, `powf`), so on another platform than the recorded one
they may differ in the last places: `f64_close` then allows 4 ULP. The repository's crate is
never modified, and the fresh directory, build included, is removed at the end unless `--keep`
is given.
`tests/test_rustcheck.py` checks that the emulation in `jali_params.rustcheck` agrees with the
checked-in data.

Integers follow the number rule of `jali_params.jsonio`: a JSON number below $`2^{64}`$, a
decimal string otherwise; `load` accepts both. A case whose text must hold a JSON number of
$`2^{64}`$ or more (to test that Rust refuses it) is stored as its `raw` text.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TOOL = Path(__file__).resolve().parent
JALI = TOOL.parents[1]
sys.path.insert(0, str(TOOL))
from jali_params import cli, derive, hardness, jsonio, modulus, request  # noqa: E402

DATA = TOOL / "tests" / "data"
PLATFORM = f"{platform.system()} {platform.machine()}"
# Rust sources whose logic `rustcheck` emulates; their hashes go into the provenance.
EMULATED = ["src/params/mod.rs", "src/statement.rs", "src/lin.rs", "src/lnp.rs", "src/abdlop.rs",
            "src/json.rs", "src/dcompress.rs", "src/math/int.rs", "src/math/ring.rs",
            "src/params/moduli.rs", "src/rand/mod.rs"]
SETS = {"kyber1024-d64": (3329, 2950), "kyber1024-d128": (3329, 2950),
        "demo": (4294962689, 2048)}
BASE_FILES = {
    "toy-d64": "src/params/sets/toy-d64.json",
    "possession": "examples/fixtures/possession.json",
    "tag-preimage-crt-d128": "tests/fixtures/params/tag-preimage-crt-d128.json",
    "tag-preimage-crt-d64": "tests/fixtures/params/tag-preimage-crt-d64.json",
    "dense-blocks-7": "tests/fixtures/params/dense-blocks-7.json",
    "decoder-ranges": "tests/fixtures/params/decoder-ranges.json",
    "kyber1024-d64": "tools/params/sets/kyber1024-d64.json",
    "kyber1024-d128": "tools/params/sets/kyber1024-d128.json",
    "demo": "tools/params/sets/demo.json",
}
TOY_NO_RANGE = dict(m1=10, l=2, alpha_squared=5760, n_bin=2, l2_rows=[2, 1],
                      l2_bounds_squared=[128, 64], n_prime=0, linf_bound=0)
# The statement kinds of the verifier (`kind` in VERIFIER) with their statement moduli. The tool
# derives a set from the requirements and blocks that Rust exports for each kind of DERIVED at
# PROBE; each kind of OTHERS is exported at the modulus of the set of the kind it names and
# compiled with that set (the statement modulus None is twice that modulus).
DERIVED = {"two-moduli": 156, "linf": 156, "placed": 12289, "packed": 156, "subring": 156,
           "linf-exact": 156}
OTHERS = {"linf-limit": ("linf", 1009), "placed-ajtai": ("placed", 12289),
          "noninvertible-constraint": ("two-moduli", 156),
          "noninvertible-statement": ("two-moduli", None)}
PROBE = 1099511627917

VERIFIER = r'''//! A verifier written by tools/params/rust_reference.py; not part of the crate.
use jali::{
    Error,
    lin::{self, Block},
    math::{Poly, PolyMat, PolyVec, Ring, U256},
    params::{CheckedParams, TboxParams},
    quad::QuadEq,
    statement::{Compiled, Norm, Placement, Statement},
};
use std::{collections::BTreeMap, sync::Arc};

type Witness = BTreeMap<String, Vec<Poly>>;
const CONTEXT: &[u8] = b"jali tools/params/rust_reference.py";

fn derived(c: &CheckedParams) -> String {
    format!(
        concat!(
            "{{\"q_hex\":\"{:x}\",\"lambda\":{},\"omega\":{},\"eta\":{},\"n_ex\":{},\"l_ext\":{},",
            "\"b_squared\":\"{}\",\"z1_bound_squared\":\"{}\",\"z3_bound_squared\":\"{}\",",
            "\"z4_bound\":\"{}\",\"exact_alpha_squared\":\"{}\",\"approx_alpha_squared\":\"{}\",",
            "\"approx_extraction_bound\":\"{}\",\"arp_bound\":\"{:?}\",",
            "\"msis_delta\":\"{:?}\",\"estimated_proof_bytes\":{}}}"
        ),
        c.q, c.lambda, c.omega, c.eta, c.n_ex, c.l_ext, c.b_squared, c.z1_bound_squared,
        c.z3_bound_squared, c.z4_bound, c.exact_alpha_squared, c.approx_alpha_squared,
        c.approx_extraction_bound, c.arp_bound, c.msis_delta,
        c.estimated_proof_bytes
    )
}

fn quote(s: &str) -> String {
    serde_json::to_string(s).unwrap()
}

fn worst(p: i128) -> (Arc<Ring>, Vec<Vec<Poly>>, Vec<Poly>) {
    let ring = Ring::new(p, 256).unwrap();
    let half = p / 2;
    let a = (0..4)
        .map(|_| (0..8).map(|_| Poly::new(ring.clone(), vec![half; 256]).unwrap()).collect())
        .collect();
    let t = (0..4).map(|_| Poly::new(ring.clone(), vec![half; 256]).unwrap()).collect();
    (ring, a, t)
}

/// The statement lin::compile builds from A w + t = 0, mirrored for its requirements.
fn statement(ring: &Arc<Ring>, bsq: u64, a: &[Vec<Poly>], t: &[Poly]) -> Result<Statement, Error> {
    let mut s = Statement::new(ring.clone());
    s.var("w", 8, Norm::L2Squared(bsq))?;
    for row in 0..4 {
        let mut eq = s.constant(t[row].clone())?;
        for j in 0..8 {
            eq = eq.add(&s.variable("w", j)?.scale(&a[row][j])?)?;
        }
        s.eq_mod_p(eq)?;
    }
    Ok(s)
}

fn compile_lin(ring: &Arc<Ring>, bsq: u64, a: &[Vec<Poly>], t: &[Poly], params: TboxParams)
    -> Result<Compiled, Error> {
    let am = PolyMat::new(ring.clone(), 4, 8, a.iter().flatten().cloned().collect())?;
    let tv = PolyVec::new(ring.clone(), t.to_vec())?;
    let blocks = [Block { name: "w".into(), length: 8, norm: Norm::L2Squared(bsq) }];
    lin::compile(&am, &tv, &blocks, params)
}

fn outcome<T>(r: Result<T, Error>) -> String {
    match r {
        Ok(_) => "ok".into(),
        Err(e) => format!("{e:?}"),
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

/// set <params.json> <p> <bound squared> [prove]
fn set_mode(args: &[String]) -> Result<(), Error> {
    let text = std::fs::read_to_string(&args[2]).unwrap();
    let p: i128 = args[3].parse().unwrap();
    let bsq: u64 = args[4].parse().unwrap();
    let params = TboxParams::from_json(&text)?;
    let c = params.check()?;
    let (ring, a, t) = worst(p);
    let req = statement(&ring, bsq, &a, &t)?.requirements(params.degree, c.q)?;
    let mirrored = outcome(statement(&ring, bsq, &a, &t)?.compile(params.clone()));
    let via_lin = outcome(compile_lin(&ring, bsq, &a, &t, params.clone()));
    print!(
        concat!(
            "{{\"checked\":{},\"requirements\":{},\"worst_case_compile\":{},",
            "\"worst_case_lin_compile\":{}"
        ),
        derived(&c), serde_json::to_string(&req).unwrap(), quote(&mirrored), quote(&via_lin)
    );
    if args.get(5).map(|s| s == "prove").unwrap_or(false) {
        let mut rng = Rng(2026);
        let half = p / 2;
        let w: Vec<Poly> = (0..8)
            .map(|_| {
                let c = (0..256).map(|_| (rng.next() % 3) as i128 - 1).collect();
                Poly::new(ring.clone(), c).unwrap()
            })
            .collect();
        let a: Vec<Vec<Poly>> = (0..4)
            .map(|_| {
                (0..8)
                    .map(|_| {
                        let c = (0..256).map(|_| (rng.next() % p as u64) as i128 - half).collect();
                        Poly::new(ring.clone(), c).unwrap()
                    })
                    .collect()
            })
            .collect();
        let mut t = Vec::new();
        for row in &a {
            let mut acc = Poly::zero(ring.clone());
            for (aij, wj) in row.iter().zip(&w) {
                acc = acc.add(&aij.mul(wj)?)?;
            }
            t.push(Poly::zero(ring.clone()).sub(&acc)?);
        }
        let compiled = compile_lin(&ring, bsq, &a, &t, params)?;
        let witness = BTreeMap::from([("w".to_string(), w)]);
        let context = CONTEXT;
        let bytes = compiled.prove_bytes_with_seed([7; 32], &witness, context, [9; 32])?;
        let verified = compiled.verify_bytes([7; 32], &bytes, context).is_ok();
        let mut tampered = bytes.clone();
        let mid = tampered.len() / 2;
        tampered[mid] ^= 1;
        print!(
            concat!(
                ",\"random_instance\":{{\"proof_bytes\":{},\"verified\":{},",
                "\"tampered_rejected\":{},\"other_context_rejected\":{}}}"
            ),
            bytes.len(),
            verified,
            compiled.verify_bytes([7; 32], &tampered, context).is_err(),
            compiled.verify_bytes([7; 32], &bytes, b"another context").is_err()
        );
    }
    println!("}}");
    Ok(())
}

/// check <cases.jsonl>: from_json, and lin::compile of the worst case for cases with "statement".
fn check_mode(args: &[String]) {
    let text = std::fs::read_to_string(&args[2]).unwrap();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let case: serde_json::Value = serde_json::from_str(line).unwrap();
        let name = quote(case["name"].as_str().unwrap());
        match TboxParams::from_json(case["params"].as_str().unwrap()) {
            Ok(params) => {
                let c = params.check().expect("from_json checks");
                let compile = case.get("statement").map(|st| {
                    let (ring, a, t) = worst(st["p"].as_i64().unwrap() as i128);
                    let bsq = st["bsq"].as_u64().unwrap();
                    let result = outcome(compile_lin(&ring, bsq, &a, &t, params.clone()));
                    format!(",\"compile\":{}", quote(&result))
                });
                let compile = compile.unwrap_or_default();
                println!("{{\"name\":{name},\"ok\":true,\"derived\":{}{compile}}}", derived(&c));
            }
            Err(e) => {
                let error = quote(&format!("{e:?}"));
                println!("{{\"name\":{name},\"ok\":false,\"error\":{error}}}");
            }
        }
    }
}

/// A polynomial over `ring` from (index, coefficient) pairs.
fn sparse(ring: &Arc<Ring>, c: &[(usize, i128)]) -> Poly {
    let mut v = vec![0i128; ring.degree()];
    for (i, x) in c {
        v[*i] = *x;
    }
    Poly::new(ring.clone(), v).unwrap()
}

/// Over `ring` (degree 128): s (4, ||s||^2 <= 64) and a binary x. A: 3 s0 + 5 x0 = 17 + 3X +
/// 5X^3 modulo the modulus of `ring_a` (implicit quotients), B: ct(s1 x0 + 3 s0 + 1) = 0
/// modulo 12 (a committed carry), C: s2 = X^2 over the statement ring.
fn two_moduli(ring: Arc<Ring>, ring_a: Arc<Ring>) -> Result<(Statement, Witness), Error> {
    let r12 = Ring::new(12, 128)?;
    let mut st = Statement::new(ring.clone());
    st.var("s", 4, Norm::L2Squared(64))?;
    st.var("x", 1, Norm::Binary)?;
    let a = st
        .variable_in(&ring_a, "s", 0)?
        .scale(&Poly::constant(ring_a.clone(), 3))?
        .add(&st.variable_in(&ring_a, "x", 0)?.scale(&Poly::constant(ring_a.clone(), 5))?)?
        .add(&st.constant_in(sparse(&ring_a, &[(0, -17), (1, -3), (3, -5)]))?)?;
    st.eq_mod_p(a)?;
    let b = st
        .variable_in(&r12, "s", 1)?
        .product_affine(&st.variable_in(&r12, "x", 0)?)?
        .add(&st.variable_in(&r12, "s", 0)?.scale(&Poly::constant(r12.clone(), 3))?)?
        .add(&st.constant_in(Poly::constant(r12.clone(), 1))?)?;
    st.const_coeff_zero(b)?;
    let c = st.variable("s", 2)?.add(&st.constant(sparse(&ring, &[(2, -1)]))?)?;
    st.eq_mod_p(c)?;
    let s = vec![
        sparse(&ring, &[(0, 4), (1, 1)]),
        sparse(&ring, &[(0, -1)]),
        sparse(&ring, &[(2, 1)]),
        Poly::zero(ring.clone()),
    ];
    let x = vec![sparse(&ring, &[(0, 1), (3, 1)])];
    Ok((st, BTreeMap::from([("s".into(), s), ("x".into(), x)])))
}

/// Over `ring` (degree 128): s (4, ||s||^2 <= 64) and y (linf <= 2). D: ct(3 y s0 - 5) = 0
/// modulo 13 (a committed carry), E: 5 y + s1 + 2 + 5X - 5X^2 - X^5 = 0 modulo 12 (implicit
/// quotients).
fn linf(ring: Arc<Ring>) -> Result<(Statement, Witness), Error> {
    let (r12, r13) = (Ring::new(12, 128)?, Ring::new(13, 128)?);
    let mut st = Statement::new(ring.clone());
    st.var("s", 4, Norm::L2Squared(64))?;
    st.var("y", 1, Norm::Linf(2))?;
    let d = st
        .variable_in(&r13, "y", 0)?
        .product_affine(&st.variable_in(&r13, "s", 0)?)?
        .scale(&Poly::constant(r13.clone(), 3))?
        .add(&st.constant_in(Poly::constant(r13.clone(), -5))?)?;
    st.const_coeff_zero(d)?;
    let e = st
        .variable_in(&r12, "y", 0)?
        .scale(&Poly::constant(r12.clone(), 5))?
        .add(&st.variable_in(&r12, "s", 1)?)?
        .add(&st.constant_in(sparse(&r12, &[(0, 2), (1, 5), (2, -5), (5, -1)]))?)?;
    st.eq_mod_p(e)?;
    let s = vec![
        sparse(&ring, &[(0, 3), (1, -1)]),
        sparse(&ring, &[(5, 1)]),
        Poly::zero(ring.clone()),
        Poly::zero(ring.clone()),
    ];
    let y = vec![sparse(&ring, &[(0, 2), (1, -1), (2, 1)])];
    Ok((st, BTreeMap::from([("s".into(), s), ("y".into(), y)])))
}

/// Over Z_12289 (degree 128): w (1, ||w||^2 <= 2^24) in the given part and s (10,
/// ||s||^2 <= 1000), with a w + sum b_i s_i + t = 0 for a = 3 + X^7, b_i = 1 - 2X^(i+1).
fn placed(placement: Placement) -> Result<(Statement, Witness), Error> {
    let ring = Ring::new(12289, 128)?;
    let mut st = Statement::new(ring.clone());
    st.var_placed("w", 1, Norm::L2Squared(1 << 24), placement)?;
    st.var("s", 10, Norm::L2Squared(1000))?;
    let w = vec![sparse(&ring, &[(0, 4000), (5, -800)])];
    let s: Vec<Poly> = (0..10)
        .map(|i| sparse(&ring, &[(i, 1), (i + 20, -1)]))
        .collect();
    let a = sparse(&ring, &[(0, 3), (7, 1)]);
    let mut form = st.variable("w", 0)?.scale(&a)?;
    let mut value = a.mul(&w[0])?;
    for (i, si) in s.iter().enumerate() {
        let b = sparse(&ring, &[(0, 1), (i + 1, -2)]);
        form = form.add(&st.variable("s", i)?.scale(&b)?)?;
        value = value.add(&b.mul(si)?)?;
    }
    st.eq_mod_p(form.add(&st.constant(value.neg())?)?)?;
    Ok((st, BTreeMap::from([("w".into(), w), ("s".into(), s)])))
}

/// Over `ring` (degree 128): s (4, ||s||^2 <= 64) and a binary x, with 65 clauses
/// ct(sigma(a_j) s_(j mod 4) + x_0 + c_j) = 0 modulo 13 for even j and 12 for odd j, where
/// a_j has coefficients (7 i + 3 j) mod p_j and c_j makes the witness satisfy the clause: at
/// proof degree 64 their carries fill two packed polynomials.
fn packed(ring: Arc<Ring>) -> Result<(Statement, Witness), Error> {
    let mut st = Statement::new(ring.clone());
    st.var("s", 4, Norm::L2Squared(64))?;
    st.var("x", 1, Norm::Binary)?;
    let s: Vec<Poly> = (0..4)
        .map(|i| sparse(&ring, &[(i, 1), (i + 5, -1), (3 * i + 20, 1), (40 + i, 1), (90, -1)]))
        .collect();
    let x = vec![sparse(&ring, &[(0, 1), (7, 1), (100, 1)])];
    for j in 0..65usize {
        let p: i128 = if j % 2 == 0 { 13 } else { 12 };
        let r = Ring::new(p, 128)?;
        let a = Poly::new(r.clone(), (0..128).map(|i| ((7 * i + 3 * j) as i128) % p).collect())?;
        let form = st
            .variable_in(&r, "s", j % 4)?
            .scale(&a.auto())?
            .add(&st.variable_in(&r, "x", 0)?)?;
        let w: Vec<Poly> = s
            .iter()
            .chain(&x)
            .map(|v| Poly::new(r.clone(), v.coefficients_i128().unwrap().to_vec()))
            .collect::<Result<_, _>>()?;
        let v = form.evaluate(&PolyVec::new(r.clone(), w)?)?.coefficient_i128(0)?;
        st.const_coeff_zero(form.add(&st.constant_in(Poly::constant(r, -v))?)?)?;
    }
    Ok((st, BTreeMap::from([("s".into(), s), ("x".into(), x)])))
}

/// `form` minus its value at the witness `w` (statement-ring polynomials in variable order,
/// read in the form's ring through their centred coefficients): it then holds.
fn settle(st: &Statement, form: QuadEq, w: &[Poly]) -> Result<QuadEq, Error> {
    let r = form.r0.ring().clone();
    let values = w
        .iter()
        .map(|v| Poly::new(r.clone(), v.coefficients_i128().unwrap().to_vec()))
        .collect::<Result<_, _>>()?;
    let value = form.evaluate(&PolyVec::new(r, values)?)?;
    form.add(&st.constant_in(value.neg())?)
}

/// Over `ring` (degree 128): s (4, ||s||^2 <= 64), and in the subring of degree 64 (even
/// coefficients only) a binary x and y (linf <= 2). F: (1 - X) s0 x + (2 + X^3) y s1 + c = 0
/// modulo 13 (committed carries); G_j: ct(X^-2j y + (j + 1) s2) = t_j modulo 12 for j < 3,
/// reading coefficient j of y (one packed carry); y's own range rows.
fn subring(ring: Arc<Ring>) -> Result<(Statement, Witness), Error> {
    let (r12, r13) = (Ring::new(12, 128)?, Ring::new(13, 128)?);
    let mut st = Statement::new(ring.clone());
    st.var("s", 4, Norm::L2Squared(64))?;
    st.var_subring("x", 1, 64, Norm::Binary, Placement::Ajtai)?;
    st.var_subring("y", 1, 64, Norm::Linf(2), Placement::Ajtai)?;
    let s = vec![
        sparse(&ring, &[(0, 2), (3, -1)]),
        sparse(&ring, &[(1, 1), (7, 1)]),
        sparse(&ring, &[(0, -1), (2, 1)]),
        Poly::zero(ring.clone()),
    ];
    let x = vec![sparse(&ring, &[(0, 1), (4, 1), (10, 1)])];
    let y = vec![sparse(&ring, &[(0, 2), (2, -1), (6, -2), (8, 1)])];
    let w: Vec<Poly> = s.iter().chain(&x).chain(&y).cloned().collect();
    let f = st
        .variable_in(&r13, "s", 0)?
        .product_affine(&st.variable_in(&r13, "x", 0)?)?
        .scale(&sparse(&r13, &[(0, 1), (1, -1)]))?
        .add(
            &st.variable_in(&r13, "y", 0)?
                .product_affine(&st.variable_in(&r13, "s", 1)?)?
                .scale(&sparse(&r13, &[(0, 2), (3, 1)]))?,
        )?;
    st.eq_mod_p(settle(&st, f, &w)?)?;
    for j in 0..3 {
        let g = st
            .coefficient_in(&r12, "y", 0, j)?
            .add(&st.variable_in(&r12, "s", 2)?.scale(&Poly::constant(r12.clone(), j as i128 + 1))?)?;
        st.const_coeff_zero(settle(&st, g, &w)?)?;
    }
    Ok((st, BTreeMap::from([("s".into(), s), ("x".into(), x), ("y".into(), y)])))
}

/// Over `ring` (degree 128): s (4, ||s||^2 <= 64), u (|u_i| <= 1 exactly) and, in the BDLOP
/// part, v (|v_i| <= 5 exactly, subring of degree 64). H: (1 + X) u s0 + 3 v + c = 0 modulo 13
/// (committed carries); I: ct(X^-2 v + (2 - X^5) u) = t modulo 12, coefficient 1 of v (a packed
/// carry); J: 2 u + s1 + e = 0 modulo 12 (implicit quotients).
fn linf_exact(ring: Arc<Ring>) -> Result<(Statement, Witness), Error> {
    let (r12, r13) = (Ring::new(12, 128)?, Ring::new(13, 128)?);
    let mut st = Statement::new(ring.clone());
    st.var("s", 4, Norm::L2Squared(64))?;
    st.var("u", 1, Norm::LinfExact(1))?;
    st.var_subring("v", 1, 64, Norm::LinfExact(5), Placement::Bdlop)?;
    let s = vec![
        sparse(&ring, &[(0, 1), (5, -2)]),
        sparse(&ring, &[(2, 1), (9, -1)]),
        Poly::zero(ring.clone()),
        sparse(&ring, &[(4, 1)]),
    ];
    let u = vec![sparse(&ring, &[(0, 1), (1, -1), (3, 1), (70, -1)])];
    let v = vec![sparse(&ring, &[(0, 5), (2, -5), (4, 3), (100, -2)])];
    let w: Vec<Poly> = s.iter().chain(&u).chain(&v).cloned().collect();
    let h = st
        .variable_in(&r13, "u", 0)?
        .product_affine(&st.variable_in(&r13, "s", 0)?)?
        .scale(&sparse(&r13, &[(0, 1), (1, 1)]))?
        .add(&st.variable_in(&r13, "v", 0)?.scale(&Poly::constant(r13.clone(), 3))?)?;
    st.eq_mod_p(settle(&st, h, &w)?)?;
    let i = st
        .coefficient_in(&r12, "v", 0, 1)?
        .add(&st.variable_in(&r12, "u", 0)?.scale(&sparse(&r12, &[(0, 2), (5, -1)]))?)?;
    st.const_coeff_zero(settle(&st, i, &w)?)?;
    let j = st
        .variable_in(&r12, "u", 0)?
        .scale(&Poly::constant(r12.clone(), 2))?
        .add(&st.variable_in(&r12, "s", 1)?)?;
    st.eq_mod_p(settle(&st, j, &w)?)?;
    Ok((st, BTreeMap::from([("s".into(), s), ("u".into(), u), ("v".into(), v)])))
}

/// A statement kind of the reference data and its witness. `q` is the proof modulus, from
/// which the "noninvertible" kinds build a modulus.
fn kind(name: &str, q: &U256) -> Result<(Statement, Witness), Error> {
    let ring = |m: i128| Ring::new(m, 128);
    let times = |k: u8| Ring::with_modulus(q.wrapping_mul(&U256::from_u8(k)), 128);
    match name {
        "two-moduli" => two_moduli(ring(156)?, ring(13)?),
        "noninvertible-constraint" => two_moduli(ring(156)?, times(3)?),
        "noninvertible-statement" => two_moduli(times(2)?, ring(13)?),
        "linf" => linf(ring(156)?),
        "linf-limit" => linf(ring(1009)?),
        "placed" => placed(Placement::Bdlop),
        "placed-ajtai" => placed(Placement::Ajtai),
        "packed" => packed(ring(156)?),
        "subring" => subring(ring(156)?),
        "linf-exact" => linf_exact(ring(156)?),
        other => panic!("unknown statement kind {other}"),
    }
}

/// statement <kind> <proof degree> <proof modulus>: Statement::requirements and
/// Statement::blocks of a kind.
fn statement_mode(args: &[String]) -> Result<(), Error> {
    let degree: usize = args[3].parse().unwrap();
    let q = U256::from_u128(args[4].parse().unwrap());
    let (st, _) = kind(&args[2], &q)?;
    println!(
        "{{\"requirements\":{},\"blocks\":{}}}",
        serde_json::to_string(&st.requirements(degree, q)?).unwrap(),
        serde_json::to_string(&st.blocks(degree)?).unwrap()
    );
    Ok(())
}

/// compile <kind> <params.json> [prove]: compile a kind; prove and verify its witness.
fn compile_mode(args: &[String]) -> Result<(), Error> {
    let params = TboxParams::from_json(&std::fs::read_to_string(&args[3]).unwrap())?;
    let q = params.check()?.q;
    let (st, witness) = kind(&args[2], &q)?;
    match st.compile(params) {
        Err(e) => println!("{{\"compile\":{}}}", quote(&format!("{e:?}"))),
        Ok(compiled) if args.get(4).map(|s| s == "prove").unwrap_or(false) => {
            let bytes = compiled.prove_bytes_with_seed([7; 32], &witness, CONTEXT, [9; 32])?;
            let mut tampered = bytes.clone();
            let mid = tampered.len() / 2;
            tampered[mid] ^= 1;
            println!(
                concat!(
                    "{{\"compile\":\"ok\",\"random_instance\":{{\"proof_bytes\":{},",
                    "\"verified\":{},\"tampered_rejected\":{},\"other_context_rejected\":{}}}}}"
                ),
                bytes.len(),
                compiled.verify_bytes([7; 32], &bytes, CONTEXT).is_ok(),
                compiled.verify_bytes([7; 32], &tampered, CONTEXT).is_err(),
                compiled.verify_bytes([7; 32], &bytes, b"another context").is_err()
            );
        }
        Ok(_) => println!("{{\"compile\":\"ok\"}}"),
    }
    Ok(())
}

/// cases <cases.jsonl>: from_json and compile of each line's kind with its parameter text.
fn cases_mode(args: &[String]) {
    let text = std::fs::read_to_string(&args[2]).unwrap();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let case: serde_json::Value = serde_json::from_str(line).unwrap();
        let result = TboxParams::from_json(case["params"].as_str().unwrap()).and_then(|p| {
            let q = p.check()?.q;
            kind(case["kind"].as_str().unwrap(), &q)?.0.compile(p)
        });
        println!(
            "{{\"name\":{},\"compile\":{}}}",
            quote(case["name"].as_str().unwrap()),
            quote(&outcome(result))
        );
    }
}

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().collect();
    match args[1].as_str() {
        "set" => set_mode(&args),
        "check" => {
            check_mode(&args);
            Ok(())
        }
        "statement" => statement_mode(&args),
        "compile" => compile_mode(&args),
        "cases" => {
            cases_mode(&args);
            Ok(())
        }
        other => panic!("unknown mode {other}"),
    }
}
'''


def case_text(base, case):
    """The exact JSON text of a case: `raw`, or the base with fields deleted and replaced."""
    if "raw" in case:
        return case["raw"]
    raw = dict(base)
    for key in case.get("delete", []):
        raw.pop(key)
    raw.update(case.get("set", {}))
    return json.dumps(raw, indent=2) + "\n"


def _primes(lo, count, gamma):
    out = []
    while len(out) < count:
        q, _ = modulus.construct_prime(lo, gamma)
        out.append(q)
        lo = q + 1
    return out


def wide_base(bits):
    """A synthetic set with a q of `bits` bits: the tool's search on the shape of toy-d64
    without its range block (test data only, not an estimate)."""
    shape = derive.Shape(TOY_NO_RANGE, 64)
    params, _, _, _, _ = derive.search(shape, hardness.Policy("delta"), None,
                                       window_bits=0, min_bits=bits)
    params["id"] = f"test-wide-{bits}"
    params["estimator"] = "synthetic test parameters for the emulation cross-check; not an estimate"
    return json.loads(jsonio.dump_params(params))


def cases():
    """The bases and cases: every refusal of `check` and `compile` targeted once, JSON decoding
    edge cases, moduli up to 255 bits, and 160 seeded random mutations."""
    bases = {k: json.loads((JALI / v).read_text()) for k, v in BASE_FILES.items()}
    for bits in (100, 129, 241, 255):
        bases[f"wide{bits}"] = wide_base(bits)
    out = []

    def add(name, base, **kw):
        out.append(dict(name=name, base=base, **kw))

    for b in bases:
        add(f"{b}/as-is", b)
    t = "toy-d64"
    q1 = bases[t]["prime_factors"][0]
    # JSON decoding (Rust: Error::Encoding) and the number rule.
    add("enc/prime-as-string", t, set={"prime_factors": [str(q1)]})
    add("enc/gamma-as-string", t, set={"gamma": "65202"})
    add("enc/bounds-as-strings", t, set={"l2_bounds_squared": ["128", "64"],
                                         "alpha_squared": "5760", "linf_bound": "4"})
    add("enc/prime-leading-zero", t, set={"prime_factors": ["0" + str(q1)]})
    add("enc/prime-number-2^64", t, set={"prime_factors": [2 ** 64 + 13]})
    add("enc/prime-string-2^256", t, set={"prime_factors": [str(2 ** 256 + 1)]})
    add("enc/prime-negative", t, set={"prime_factors": [-q1]})
    add("enc/prime-float", t, set={"prime_factors": [float(q1)]})
    add("enc/prime-scalar", t, set={"prime_factors": q1})
    add("enc/gamma-string-2^64", t, set={"gamma": str(2 ** 64)})
    add("enc/gamma-number-2^64", t, set={"gamma": 2 ** 64})
    add("enc/alpha-u64-max", t, set={"alpha_squared": 2 ** 64 - 1})
    add("enc/alpha-2^64", t, set={"alpha_squared": str(2 ** 64)})
    add("enc/log-sigma-three", t, set={"log_sigma": [16, 12, 10]})
    add("enc/log-sigma-string", t, set={"log_sigma": [16, 12, "10", 8]})
    add("enc/log-sigma-2^32", t, set={"log_sigma": [16, 12, 10, 2 ** 32]})
    add("enc/d-bits-negative", t, set={"d_bits": -1})
    add("enc/m1-string", t, set={"m1": "10"})
    add("enc/m1-float", t, set={"m1": 10.0})
    add("enc/m1-negative", t, set={"m1": -10})
    add("enc/m1-bool", t, set={"m1": True})
    add("enc/delta-int", t, set={"mlwe_delta": 1})
    add("enc/delta-string", t, set={"mlwe_delta": "1.0044"})
    add("enc/id-number", t, set={"id": 5})
    add("enc/unknown-field", t, set={"extra": 1})
    add("enc/missing-field", t, delete=["gamma"])
    raw = json.dumps(bases[t], indent=2)
    add("enc/duplicate-key", t,
        raw=raw.replace('"gamma": 65202,', '"gamma": 65202,\n  "gamma": 65202,'))
    delta = f'"mlwe_delta": {bases[t]["mlwe_delta"]!r}'
    assert raw.count(delta) == 1
    add("enc/nan-delta", t, raw=raw.replace(delta, '"mlwe_delta": NaN'))
    add("enc/not-an-object", t, raw="[1, 2]\n")
    # Each refusal of check_with.
    add("chk/provenance-empty-estimator", t, set={"estimator": ""})
    add("chk/provenance-long-id", t, set={"id": "x" * 257})
    add("chk/provenance-id-256", t, set={"id": "x" * 256})
    add("chk/provenance-long-estimator", t, set={"estimator": "e" * 4097})
    add("chk/provenance-estimator-4096", t, set={"estimator": "e" * 4096})
    add("chk/provenance-multibyte-id", t, set={"id": "é" * 129})
    add("chk/factors-not-5-mod-8", t, set={"prime_factors": [q1 + 8]})
    add("chk/factors-composite", t, set={"prime_factors": [q1 * 17]})  # 5 * 1 = 5 mod 8
    add("chk/factors-descending", t, set={"prime_factors": [q1, 13]})
    add("chk/factors-three", t, set={"prime_factors": [5, 13, 29]})
    add("chk/factors-empty", t, set={"prime_factors": []})
    add("chk/factors-wide-prime", t, set={"prime_factors": [2 ** 64 + 13]})
    add("chk/factors-wide-composite", t, set={"prime_factors": [2 ** 64 + 21]})
    add("chk/factors-wide-prime-string", t, set={"prime_factors": [str(2 ** 64 + 13)]})
    add("chk/factors-wide-composite-string", t, set={"prime_factors": [str(2 ** 64 + 21)]})
    # 3277 = 29 * 113, a strong pseudoprime to base 2 and 5 mod 8.
    add("chk/factors-psp2", t, set={"prime_factors": [3277]})
    add("chk/factors-product-2^256", "wide255",
        set={"prime_factors": [13] + bases["wide255"]["prime_factors"]})
    add("chk/degree-256", t, set={"degree": 256})
    add("chk/degree-32", t, set={"degree": 32})
    add("chk/compression-odd-gamma", t, set={"gamma": 65203})
    add("chk/compression-gamma-8", t, set={"gamma": 8})
    add("chk/compression-gamma-0", t, set={"gamma": 0})
    add("chk/compression-d-bits-40", t, set={"d_bits": 40})
    add("chk/compression-d-bits-41", t, set={"d_bits": 41})
    add("chk/compression-d-bits-256", t, set={"d_bits": 256})
    # q = 2^42 - 11 is prime and 5 mod 8, and q - 1 has bits 7..41 set and low 7 bits above
    # 64: power2round with D = 7 carries past the bit width (after `compression`, which
    # gamma = 4 passes).
    q = 2 ** 42 - 11
    assert modulus.is_prime(q) and q % 8 == 5 and (q - 1) % 128 > 64
    add("chk/commitment-bit-width", t, set={"prime_factors": [q], "gamma": 4})
    add("chk/rounding-gamma-4", t, set={"gamma": 4})
    add("chk/exact-blocks-zero-rows", t, set={"l2_rows": [2, 0]})
    add("chk/exact-blocks-length", t, set={"l2_bounds_squared": [128]})
    add("chk/dimensions-m2", t, set={"m2": 70000})
    add("chk/dimensions-alpha-zero", t, set={"alpha_squared": 0})
    add("chk/dimensions-n-msis-zero", t, set={"n_msis": 0})
    add("chk/approximate-block", t, set={"linf_bound": 0})
    add("chk/width-101", t, set={"log_sigma": [101, 12, 10, 8]})
    add("chk/width-100", t, set={"log_sigma": [100, 12, 10, 8]})
    add("chk/two-primes-range", t, set={"prime_factors": [13, q1], "gamma": 2, "d_bits": 0})
    add("chk/completeness", t, set={"m1": 0, "l2_rows": [2], "l2_bounds_squared": [128]})
    add("chk/rank", t, set={"mlwe_rank": 25})
    add("chk/delta-high", t, set={"mlwe_delta": 1.0045})
    add("chk/delta-low", t, set={"mlwe_delta": 0.999})
    add("chk/delta-edge", t, set={"mlwe_delta": 1.0044})
    add("chk/msis", t, set={"n_msis": 10, "m2": 50, "mlwe_rank": 26})
    add("chk/range-m", t, set={"log_sigma": [16, 12, 3, 8]})
    add("chk/range-m4", t, set={"log_sigma": [16, 12, 10, 2]})
    add("chk/response-capacity", t, set={"log_sigma": [16, 60, 10, 8]})
    add("chk/response-bound-capacity", t, set={"log_sigma": [100, 12, 10, 8]})
    add("chk/arp-modulus", "kyber1024-d64", set={"log_sigma": [16, 12, 35, 23]})
    # The hint allowance of the size estimate above sigma_2/gamma = 0.96875: smaller even
    # divisors gamma of q - 1, with D lowered to meet the rounding bound.
    for g, dbits in ((2028, 2), (676, 1), (338, 0)):
        add(f"est/hints-gamma-{g}", t, set={"gamma": g, "d_bits": dbits})
    for g, dbits in ((6342, 4), (434, 0)):
        add(f"est/hints-demo-gamma-{g}", "demo", set={"gamma": g, "d_bits": dbits})
    # The checks of compile against the worst-case lin statement.
    k64 = bases["kyber1024-d64"]
    add("cmp/norm-budget", "kyber1024-d64", statement={"p": 3329, "bsq": 2951})
    add("cmp/exact-block-bound", "kyber1024-d64", statement={"p": 3329, "bsq": 2949})
    add("cmp/two-exact-blocks", "kyber1024-d64",
        set={"l2_rows": [16, 16], "l2_bounds_squared": [2950, 2950], "alpha_squared": 5900})
    add("cmp/range-rows", "kyber1024-d64", set={"n_prime": 12})
    add("cmp/quotient-bound", "kyber1024-d64", set={"linf_bound": 3475})
    add("cmp/lifting-slack", "kyber1024-d64", set={"log_sigma": [16, 12, 12, 72]})
    add("cmp/lifting-psi-below-u64", "kyber1024-d64", set={"log_sigma": [16, 12, 12, 70]})
    add("cmp/lifting-bound-sigma4", "kyber1024-d64", set={"log_sigma": [16, 12, 12, 24]})
    add("cmp/witness-dimensions", "kyber1024-d64", set={"l": 1, "m2": k64["m2"] + 1})
    add("cmp/other-statement-modulus", "kyber1024-d64", statement={"p": 7681, "bsq": 2950})
    # Random mutations of one to three fields in plausible ranges.
    rng = random.Random(20260929)
    fields = ["m1", "m2", "l", "n_msis", "n_bin", "n_prime", "linf_bound", "alpha_squared",
              "log_sigma", "gamma", "d_bits", "mlwe_rank", "prime_factors", "l2", "mlwe_delta"]
    names = list(bases)
    for i in range(160):
        b = rng.choice(names)
        base = bases[b]
        change = {}
        for f in rng.sample(fields, rng.randint(1, 3)):
            if f in ("m1", "m2", "l", "n_msis", "n_bin", "n_prime", "mlwe_rank", "d_bits"):
                change[f] = max(0, base[f] + rng.randint(-3, 3))
            elif f == "linf_bound":
                change[f] = rng.choice([0, 1, base[f], base[f] * 2, rng.randrange(1, 2 ** 40)])
            elif f == "alpha_squared":
                change[f] = rng.choice([1, base[f], base[f] * 4, rng.randrange(1, 2 ** 63)])
            elif f == "log_sigma":
                ls = list(base[f])
                ls[rng.randrange(4)] = max(0, ls[rng.randrange(4)] + rng.randint(-4, 4))
                change[f] = ls
            elif f == "gamma":
                g = int(base[f])
                change[f] = rng.choice([g // 2, g * 2, g - 2, g + 2, 2 * rng.randrange(1, 2 ** 20)])
                if rng.random() < 0.3:
                    change[f] = 4
            elif f == "prime_factors":
                q, g = int(base[f][0]), int(base["gamma"])
                if modulus.v2(g) not in (1, 2):
                    continue
                lo = q - rng.randrange(1, 2 ** 20) * 8 * g if q > 2 ** 30 else q
                v = rng.choice(_primes(max(lo, 2 ** 20), 3, g))
                change[f] = [v if v < 2 ** 64 else str(v)]
            elif f == "l2" and base["l2_rows"]:
                bs = list(base["l2_bounds_squared"])
                j = rng.randrange(len(bs))
                bs[j] = max(1, int(bs[j]) * rng.choice([1, 2, 1000, 10 ** 6]))
                change["l2_bounds_squared"] = bs
            elif f == "mlwe_delta":
                change[f] = rng.choice([1.0, 1.0043, 1.0044, 1.00440001, 1.0042736680871425])
        add(f"rnd/{i:03d}/{b}", b, set=change)
    for c in out:
        if c["base"] in SETS and "statement" not in c and "raw" not in c:
            p, bsq = SETS[c["base"]]
            c.setdefault("statement", {"p": p, "bsq": bsq})
    for c in out:
        # The number rule for this file: a change that writes a JSON number of 2^64 or more
        # into the case text is kept as that text (the same bytes, so the same SHA-256).
        if _has_wide_number(c.get("set", {})):
            c["raw"] = case_text(bases[c["base"]], c)
            c.pop("set")
            c.pop("delete", None)
    return bases, out


def statement_request(kind, exported, p):
    """The version-2 request of a statement kind: the requirements and blocks Rust exported,
    with `w` movable for the kind "placed", and the search's default window of 4 bits."""
    statement = {"source": "requirements", "requirements": exported["requirements"],
                 "statement_modulus": jsonio.encode_int(p), "blocks": exported["blocks"]}
    if kind == "placed":
        statement["movable"] = ["w"]
    return {"schema": request.SCHEMA, "id": f"rust-reference-{kind}", "degree": 64,
            "modulus": {"search": {"factors": 1, "window_bits": 4}},
            "hardness": {"policy": "delta"}, "statement": statement}


def statement_cases(kind, params, seed):
    """Mutations of the set derived for a kind, each for Rust to decide against that kind: the
    previous prime of its gamma class, wider and narrower range widths, other range bounds, a
    smaller budget, and seeded random changes of one or two fields."""
    out = []

    def add(name, **change):
        out.append({"name": f"{kind}/{name}", "kind": kind, "base": kind, "set": change})
    q, g = params["prime_factors"][0], params["gamma"]
    step = g * (2 if modulus.v2(g) == 2 else 4)
    prev = q - step
    while not modulus.is_prime(prev):
        prev -= step
    ls, linf = params["log_sigma"], params["linf_bound"]
    add("as-derived")
    add("previous-prime", prime_factors=[prev])
    for t in (-1, 1, 2, 5, 9, 10, 12, 16):
        add(f"log-sigma3-{t:+d}", log_sigma=ls[:3] + [ls[3] + t])
    for name, value in (("1", 1), ("half", max(1, linf // 2)), ("double", 2 * linf)):
        add(f"linf-bound-{name}", linf_bound=value)
    add("alpha-minus-1", alpha_squared=params["alpha_squared"] - 1)
    add("m1-plus-1", m1=params["m1"] + 1)
    rng = random.Random(seed)
    for i in range(24):
        change = {}
        for f in rng.sample(["log_sigma", "linf_bound", "prime_factors", "alpha_squared",
                             "n_prime", "l"], rng.randint(1, 2)):
            if f == "log_sigma":
                change[f] = ls[:3] + [max(0, ls[3] + rng.randint(-3, 12))]
            elif f == "linf_bound":
                change[f] = rng.choice([1, 2, linf - 1, linf + 1, rng.randrange(1, 2 ** 20)])
            elif f == "prime_factors":
                change[f] = [rng.choice(_primes(max(q >> rng.randint(0, 3), 2 ** 20), 2, g))]
            elif f == "alpha_squared":
                change[f] = rng.choice([params["alpha_squared"] // 2, 2 * params["alpha_squared"]])
            elif f == "n_prime":
                change[f] = max(0, params["n_prime"] + rng.choice([-1, 1]))
            else:  # l, keeping the MLWE rank
                change.update(l=params["l"] + 1, m2=params["m2"] + 1)
        add(f"random-{i:02d}", **change)
    return out


def _run_json(args):
    return json.loads(subprocess.run([str(x) for x in args], capture_output=True, text=True,
                                     check=True).stdout)


def statements(exe, work, no_prove):
    """The statement kinds: Rust's requirements and blocks, the tool's set, Rust's compilation
    and seeded proof, and Rust's decision on each case of `statement_cases`."""
    out, cases = {}, []
    for i, (kind, p) in enumerate(DERIVED.items()):
        exported = _run_json([exe, "statement", kind, 64, PROBE])
        req = statement_request(kind, exported, p)
        text, rep = cli.derive_request(json.dumps(req), what_if=False, compare=False)
        compiled = kind
        if kind == "placed" and "w" not in rep["placement"]["chosen"]["bdlop"]:
            compiled = "placed-ajtai"
        path = work / f"{kind}.params.json"
        path.write_text(text)
        result = _run_json([exe, "compile", compiled, path] + ([] if no_prove else ["prove"]))
        out[kind] = {"statement_modulus": p, "requirements": exported["requirements"],
                     "blocks": exported["blocks"], "request": req, "params": json.loads(text),
                     "params_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "compiled_kind": compiled, **result}
        if "placement" in rep:
            out[kind]["placement"] = rep["placement"]["chosen"]
        cases += statement_cases(kind, json.loads(text), 20260930 + i)
    for kind, (base, p) in OTHERS.items():
        q = out[base]["params"]["prime_factors"][0]
        p = 2 * q if p is None else p
        exported = _run_json([exe, "statement", kind, 64, q])
        out[kind] = {"statement_modulus": jsonio.encode_int(p),
                     "requirements": exported["requirements"], "blocks": exported["blocks"],
                     "request": statement_request(kind, exported, p), "base": base}
        cases.append({"name": f"{kind}/{base}-set", "kind": kind, "base": base, "set": {}})
    # The tool refuses linf-limit at every modulus, with Rust's message.
    try:
        cli.derive_request(json.dumps(out["linf-limit"]["request"]), what_if=False,
                           compare=False)
        out["linf-limit"]["tool"] = "derived"
    except (derive.DeriveError, request.RequestError) as e:
        out["linf-limit"]["tool"] = str(e)
    lines = [json.dumps({"name": c["name"], "kind": c["kind"],
                         "params": case_text(out[c["base"]]["params"], c)}) for c in cases]
    (work / "statement-cases.jsonl").write_text("\n".join(lines) + "\n")
    decided = {r["name"]: r["compile"] for r in
               map(json.loads, subprocess.run([str(exe), "cases",
                                               str(work / "statement-cases.jsonl")],
                                              capture_output=True, text=True,
                                              check=True).stdout.splitlines())}
    for c in cases:
        text = case_text(out[c["base"]]["params"], c)
        c["sha256"] = hashlib.sha256(text.encode()).hexdigest()
        c["rust"] = decided[c["name"]]
    return out, cases


def _has_wide_number(o):
    if isinstance(o, dict):
        return any(_has_wide_number(v) for v in o.values())
    if isinstance(o, list):
        return any(_has_wide_number(v) for v in o)
    return isinstance(o, int) and not isinstance(o, bool) and abs(o) >= jsonio.LIMIT


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checked(d):
    """Rust's derived values: integers by the number rule, f64 values as strings (shortest
    round trip)."""
    out = {"q": jsonio.encode_int(int(d["q_hex"], 16))}
    for k, v in d.items():
        if k in ("q_hex", "arp_bound", "msis_delta"):
            continue
        out[k] = jsonio.encode_int(int(v))
    return out, {k: d[k] for k in ("arp_bound", "msis_delta")}


def _int(v):
    """An integer as the number rule writes it; Python's json reads a wider JSON number
    exactly, so the data written before the rule applied here also loads."""
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 \
        else jsonio.decode_int(v)


def _decode_checked(checked):
    return {k: _int(v) for k, v in checked.items()}


def load(path):
    """A data file with its integers decoded: numbers or decimal strings (the number rule)."""
    return decode(json.loads(Path(path).read_text()))


def decode(data):
    """`load` on parsed data (changed in place and returned)."""
    for c in data.get("cases", []):
        if "checked" in c.get("rust", {}):
            c["rust"]["checked"] = _decode_checked(c["rust"]["checked"])
    for entry in data.get("sets", {}).values():
        entry["checked"] = _decode_checked(entry["checked"])
        prev = entry["previous_prime_same_gamma_class"]
        prev["q"] = _int(prev["q"])
        prev["checked"] = _decode_checked(prev["checked"])
        req = entry["requirements"]
        for k in ("alpha_squared", "linf_bound", "max_integer_coefficient"):
            req[k] = _int(req[k])
        req["l2_bounds_squared"] = [_int(v) for v in req["l2_bounds_squared"]]
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--crate", default=str(JALI))
    ap.add_argument("--work", help="where to create the build directory (default: the system's "
                    "temporary directory)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the build directory, which is otherwise removed at the end")
    ap.add_argument("--no-prove", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--data", default=str(DATA), help="where the data files are (tests/data)")
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="jali-rust-reference-", dir=a.work))
    try:
        run(a, work)
    finally:
        if a.keep:
            print(f"the build is kept in {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


def run(a, work):
    """Build the verifier in `work`, run it, and write the data or (`--check`) compare it."""
    data_dir = Path(a.data)
    crate = Path(a.crate).resolve()
    copy = work / "jali"
    shutil.copytree(crate, copy, ignore=shutil.ignore_patterns("target", ".git", "__pycache__"))
    (copy / "examples" / "verify_sets.rs").write_text(VERIFIER)
    manifest = (copy / "Cargo.toml").read_text()
    if 'name = "verify_sets"' not in manifest:
        (copy / "Cargo.toml").write_text(
            manifest + '\n[[example]]\nname = "verify_sets"\nrequired-features = ["serde"]\n')
    env = dict(os.environ, CARGO_TARGET_DIR=str(work / "target"))
    build = ["cargo", "build", "--locked", "--release", "--features", "serde", "--example",
             "verify_sets"]
    subprocess.run(build, cwd=copy, env=env, check=True)
    exe = work / "target" / "release" / "examples" / "verify_sets"
    rustc = subprocess.run(["rustc", "--version"], cwd=copy, capture_output=True, text=True,
                           check=True).stdout.strip()
    provenance = {
        "generated_by": "python3 tools/params/rust_reference.py",
        "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "rustc": rustc,
        "platform": PLATFORM,
        "build": " ".join(build),
        "crate_sources_sha256": {p: _sha(crate / p) for p in EMULATED},
    }
    # The cases.
    bases, all_cases = cases()
    lines = []
    for c in all_cases:
        line = {"name": c["name"], "params": case_text(bases[c["base"]], c)}
        if "statement" in c:
            line["statement"] = c["statement"]
        lines.append(json.dumps(line))
    (work / "cases.jsonl").write_text("\n".join(lines) + "\n")
    res = subprocess.run([str(exe), "check", str(work / "cases.jsonl")], capture_output=True,
                         text=True, check=True).stdout
    results = {r["name"]: r for r in map(json.loads, res.splitlines())}
    for c in all_cases:
        r = results[c["name"]]
        c["sha256"] = hashlib.sha256(case_text(bases[c["base"]], c).encode()).hexdigest()
        if r["ok"]:
            checked, f64 = _checked(r["derived"])
            c["rust"] = {"checked": checked, "f64": f64}
            if "compile" in r:
                c["rust"]["compile"] = r["compile"]
        else:
            c["rust"] = {"error": r["error"]}
    data = {"provenance": dict(provenance, what=(
        "TboxParams::from_json and lin::compile of the worst-case statement, per case; the "
        "case text is case_text(bases[base], case) of tools/params/rust_reference.py, whose "
        "SHA-256 is recorded")), "bases": bases, "cases": all_cases}
    # The sets.
    ref = {"provenance": dict(provenance, what=(
        "for each set: TboxParams::check, Statement::requirements and compile of the "
        "worst-case lin statement (every public coefficient floor(p/2)), the same for the "
        "previous prime of the set's gamma class, and one proof of a seeded random instance "
        "with a ternary witness; for each statement kind of the verifier, its requirements and "
        "blocks, the tool's set, its compilation and one seeded proof, and compile of every "
        "statement case")),
        "toy-d64": {"b_squared": 3487433255883, "z3_bound_squared": 1734566565,
                    "z4_bound": 50790, "estimated_proof_bytes": 17896,
                    "source": "tests/arithmetic.rs, the pinned CheckedParams of toy_d64"},
        "sets": {}}
    for name, (p, bsq) in SETS.items():
        path = TOOL / "sets" / f"{name}.json"
        params = jsonio.load_params(path.read_text())
        q, g = params["prime_factors"][0], params["gamma"]
        step = g * (2 if modulus.v2(g) == 2 else 4)
        prev = q - step
        while not modulus.is_prime(prev):
            prev -= step
        prev_path = work / f"{name}.prev.json"
        prev_path.write_text(jsonio.dump_params(dict(params, prime_factors=[prev])))
        args = [str(exe), "set", str(path), str(p), str(bsq)] + ([] if a.no_prove else ["prove"])
        now = json.loads(subprocess.run(args, capture_output=True, text=True,
                                        check=True).stdout)
        before = json.loads(subprocess.run([str(exe), "set", str(prev_path), str(p), str(bsq)],
                                           capture_output=True, text=True, check=True).stdout)
        checked, f64 = _checked(now["checked"])
        entry = {"params_sha256": _sha(path), "statement_modulus": p, "bound_squared": bsq,
                 "checked": checked, "checked_f64": f64, "requirements": now["requirements"],
                 "worst_case_compile": now["worst_case_compile"],
                 "worst_case_lin_compile": now["worst_case_lin_compile"],
                 "previous_prime_same_gamma_class": {
                     "q": jsonio.encode_int(prev), "checked": _checked(before["checked"])[0],
                     "worst_case_compile": before["worst_case_compile"],
                     "worst_case_lin_compile": before["worst_case_lin_compile"]}}
        if "random_instance" in now:
            entry["random_instance"] = now["random_instance"]
        ref["sets"][name] = entry
    ref["statements"], ref["statement_cases"] = statements(exe, work, a.no_prove)
    outputs = {"rust_check_cases.json": (data, 1), "rust_reference.json": (ref, 2)}
    if a.check:
        sys.exit(_compare(outputs, data_dir, a.no_prove))
    for name, (obj, indent) in outputs.items():
        (data_dir / name).write_text(json.dumps(obj, indent=indent) + "\n")
    print(f"{len(all_cases)} cases, {len(ref['sets'])} sets")


def f64_close(a, b, exact):
    """Rust f64 outputs as recorded (strings): equal, or within 4 ULP unless `exact`."""
    a, b = float(a), float(b)
    return a == b or (not exact and abs(a - b) <= 4 * math.ulp(b))


def _same(new, old, exact, key=None):
    if key in ("f64", "checked_f64") and isinstance(new, dict) and isinstance(old, dict):
        return set(new) == set(old) and all(f64_close(new[k], old[k], exact) for k in new)
    if isinstance(new, dict) and isinstance(old, dict):
        return set(new) == set(old) and all(_same(new[k], old[k], exact, k) for k in new)
    if isinstance(new, list) and isinstance(old, list):
        return len(new) == len(old) and all(_same(a, b, exact) for a, b in zip(new, old))
    return new == old


def _comparable(data):
    """Decoded integers, and each case identified by the SHA-256 of its text rather than by
    how the file spells it (`set`, `delete` or `raw`)."""
    for c in data.get("cases", []):
        for key in ("set", "delete", "raw"):
            c.pop(key, None)
    return data


def _compare(outputs, data_dir, no_prove):
    """Differences between fresh results and the checked-in data, provenance aside. Either
    spelling of an integer (number or decimal string) compares equal."""
    differ = 0
    for name, (obj, _) in outputs.items():
        new = _comparable(decode(json.loads(json.dumps(obj))))
        old = _comparable(load(data_dir / name))
        exact = old["provenance"].get("platform") == PLATFORM
        for x in (new, old):
            x.pop("provenance")
            entries = list(x.get("sets", {}).values()) + list(x.get("statements", {}).values())
            for entry in entries:
                if no_prove:
                    entry.pop("random_instance", None)
        if name == "rust_check_cases.json":
            for key in ("bases",):
                if new[key] != old[key]:
                    print(f"{name}: {key} differ")
                    differ += 1
            olds = {c["name"]: c for c in old["cases"]}
            for c in new["cases"]:
                if not _same(c, olds.get(c["name"]), exact):
                    print(f"{name}: case {c['name']} differs")
                    differ += 1
            if len(olds) != len(new["cases"]):
                print(f"{name}: {len(olds)} recorded cases, {len(new['cases'])} now")
                differ += 1
        elif not _same(new, old, exact):
            for key in sorted(set(new) | set(old)):
                if not _same(new.get(key), old.get(key), exact, key):
                    for what in differences(key, new.get(key), old.get(key), exact):
                        print(f"{name}: {what}")
                        differ += 1
    print("the checked-in Rust reference data is current" if not differ else
          f"{differ} differences: rerun tools/params/rust_reference.py and check the emulation")
    return 1 if differ else 0


def differences(key, new, old, exact):
    """What differs under one key of `rust_reference.json`: each statement case by name, and
    each set or statement kind with the fields that differ; otherwise the key."""
    if key == "statement_cases" and isinstance(new, list) and isinstance(old, list):
        news, olds = {c["name"]: c for c in new}, {c["name"]: c for c in old}
        out = [f"statement case {n} differs" for n in news
               if n in olds and not _same(news[n], olds[n], exact)]
        out += [f"statement case {n} is new" for n in news if n not in olds]
        out += [f"statement case {n} is missing" for n in olds if n not in news]
        return out or [f"{key}: the same cases in another order"]
    if key in ("sets", "statements") and isinstance(new, dict) and isinstance(old, dict):
        out = []
        for n in sorted(set(new) | set(old)):
            a, b = new.get(n), old.get(n)
            if _same(a, b, exact):
                continue
            if not (isinstance(a, dict) and isinstance(b, dict)):
                out.append(f"{key} {n} is {'new' if b is None else 'missing'}")
                continue
            fields = sorted(k for k in set(a) | set(b) if not _same(a.get(k), b.get(k), exact, k))
            out.append(f"{key} {n} differs in {', '.join(fields)}")
        return out
    return [f"{key} differs"]


if __name__ == "__main__":
    main()
