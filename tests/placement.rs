//! Placement of bounded blocks in the BDLOP part of the commitment (`Statement::var_placed`,
//! `lin::compile_placed`). The requirements move a placed block's rows from `m1` to `l` and its
//! share out of `alpha_squared`, and change nothing else, which `Statement::blocks` lets a tool
//! recompute; the default placement compiles exactly as `var` does; exact-norm, binary and
//! $`\ell_\infty`$ blocks prove and verify from the BDLOP part, with lifted constraints of two
//! moduli, while their norms are still enforced; parameters for one placement do not compile
//! the other; and moving a block with a large bound shrinks the proof. Parameters come from the
//! test port of the parameter tool (`common::params`); each test fixes its seeds.
use jali::{
    Error,
    lin::{self, Block},
    math::{Poly, PolyMat, PolyVec, Ring, U256, iso},
    params::TboxParams,
    quad::QuadEq,
    statement::{BlockRequirement, Norm, Placement, Requirements, Statement},
};
use std::{collections::BTreeMap, sync::Arc};

mod common;
use common::{
    SplitMix64,
    params::{RANK, fit, fit_upward, lifting_threshold, next_prime_5_mod_8, range_width},
    ring::{integer_value, poly},
};

/// The prime of `toy_d64`, where every search for a proof modulus starts.
const Q: u64 = 1099511627917;
/// $`13\cdot101`$: the statement modulus, which the second constraint's modulus 13 divides.
const P: i128 = 1313;
const AJTAI: Placement = Placement::Ajtai;
const BDLOP: Placement = Placement::Bdlop;

type Witness = BTreeMap<String, Vec<Poly>>;

/// A sparse public polynomial over `ring`: four coefficients in $`[-6,6]`$.
fn sparse(rng: &mut SplitMix64, ring: &Arc<Ring>) -> Poly {
    let mut c = vec![0i128; ring.degree()];
    for _ in 0..4 {
        c[rng.below(ring.degree() as u64) as usize] = rng.below(13) as i128 - 6;
    }
    poly(ring, c)
}

/// The integer witness of `placed`, in declaration order: $`a_0..a_4`$ and $`s_0,s_1`$ with 16
/// ternary coefficients each, a binary $`x`$ and $`y`$ in $`[-2,2]`$, with $`\pm2`$ in its first
/// two coefficients.
fn integers(seed: u64) -> Vec<Vec<i128>> {
    let mut rng = SplitMix64(seed);
    let mut w = Vec::new();
    for _ in 0..7 {
        let mut c = vec![0i128; 128];
        for _ in 0..16 {
            c[rng.below(128) as usize] = rng.below(3) as i128 - 1;
        }
        w.push(c);
    }
    w.push(rng.binary(128));
    let mut y: Vec<i128> = (0..128).map(|_| rng.below(5) as i128 - 2).collect();
    (y[0], y[1]) = (2, -2);
    w.push(y);
    w
}

/// Over $`\mathbb Z_{1313}`$, degree 128: `a` (five polynomials, $`\|a\|^2\le640`$, in the
/// Ajtai part), and `s` (two, $`\|s\|^2\le64`$), a binary `x` and `y` ($`\ell_\infty\le2`$),
/// placed as `placement` says. Two constraints, with constants that make `w` satisfy them:
/// $`g\,s_0+h\,x+i\,y+j\,a_0+c=0`$ modulo 1313 (implicit quotients) and
/// $`\mathrm{ct}(3s_1y+k\,x+c')=0`$ modulo 13 (one committed carry). Returns the statement, the
/// two forms and the named witness.
fn placed(placement: [Placement; 3], w: &[Vec<i128>]) -> (Statement, [QuadEq; 2], Witness) {
    let ring = Ring::new(P, 128).unwrap();
    let r13 = Ring::new(13, 128).unwrap();
    let mut st = Statement::new(ring.clone());
    st.var("a", 5, Norm::L2Squared(640)).unwrap();
    st.var_placed("s", 2, Norm::L2Squared(64), placement[0])
        .unwrap();
    st.var_placed("x", 1, Norm::Binary, placement[1]).unwrap();
    st.var_placed("y", 1, Norm::Linf(2), placement[2]).unwrap();
    let mut rng = SplitMix64(0x2567_0b17);
    let v = |r: &Arc<Ring>, name: &str, i: usize| st.variable_in(r, name, i).unwrap();
    let over = |r: &Arc<Ring>| {
        PolyVec::new(r.clone(), w.iter().map(|c| poly(r, c.clone())).collect()).unwrap()
    };
    let linear = [("s", 0), ("x", 0), ("y", 0), ("a", 0)]
        .into_iter()
        .map(|(name, i)| v(&ring, name, i).scale(&sparse(&mut rng, &ring)).unwrap())
        .reduce(|x, y| x.add(&y).unwrap())
        .unwrap();
    let value = linear.evaluate(&over(&ring)).unwrap();
    let first = linear.add(&st.constant(value.neg()).unwrap()).unwrap();
    let clause = v(&r13, "s", 1)
        .product_affine(&v(&r13, "y", 0))
        .unwrap()
        .scale(&Poly::constant(r13.clone(), 3))
        .unwrap()
        .add(&v(&r13, "x", 0).scale(&sparse(&mut rng, &r13)).unwrap())
        .unwrap();
    let value = clause
        .evaluate(&over(&r13))
        .unwrap()
        .coefficient_i128(0)
        .unwrap();
    let second = clause
        .add(&st.constant_in(Poly::constant(r13, -value)).unwrap())
        .unwrap();
    st.eq_mod_p(first.clone()).unwrap();
    st.const_coeff_zero(second.clone()).unwrap();
    let names = [("a", 0..5), ("s", 5..7), ("x", 7..8), ("y", 8..9)];
    let witness = names
        .into_iter()
        .map(|(name, range)| {
            let polys = w[range].iter().map(|c| poly(&ring, c.clone())).collect();
            (name.to_string(), polys)
        })
        .collect();
    (st, [first, second], witness)
}

/// Parameters at the first prime above the lifting threshold that the port recomputes.
fn fitted(req: &Requirements, id: &str) -> TboxParams {
    let threshold = lifting_threshold(req, P as u128, range_width(req, 64));
    fit_upward(req, id, 64, (threshold as u64 + 1).max(Q)).1
}

/// Every placement of `s`, `x` and `y`.
fn placements() -> Vec<[Placement; 3]> {
    (0..8)
        .map(|bits: u8| [0, 1, 2].map(|i| if bits >> i & 1 == 1 { BDLOP } else { AJTAI }))
        .collect()
}

#[test]
fn requirements_and_blocks_follow_the_placement() {
    let w = integers(0x2567_0b01);
    let all_ajtai = placed([AJTAI; 3], &w)
        .0
        .requirements(64, U256::from_u64(Q))
        .unwrap();
    // One carry row, from the clause modulo 13.
    let carries = all_ajtai.l;
    assert_eq!(carries, 1);
    for placement in placements() {
        let st = placed(placement, &w).0;
        let req = st.requirements(64, U256::from_u64(Q)).unwrap();
        let blocks = st.blocks(64).unwrap();
        let names: Vec<&str> = blocks.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(names, ["a", "s", "x", "y"]);
        assert_eq!(
            blocks.iter().map(|b| (b.rows, b.norm)).collect::<Vec<_>>(),
            [
                (10, Norm::L2Squared(640)),
                (4, Norm::L2Squared(64)),
                (2, Norm::Binary),
                (2, Norm::Linf(2)),
            ]
        );
        assert_eq!(blocks[0].placement, AJTAI);
        assert_eq!(
            blocks[1..].iter().map(|b| b.placement).collect::<Vec<_>>(),
            placement
        );
        // What a tool recomputes from the blocks: rows and shares of the Ajtai part.
        let share = |b: &BlockRequirement| match b.norm {
            Norm::L2Squared(bound) => u128::from(bound),
            Norm::Binary | Norm::LinfExact(_) => b.rows as u128 * 64,
            Norm::Linf(beta) => b.rows as u128 * 64 * u128::from(beta).pow(2),
            Norm::Unbounded => 0,
        };
        let (ajtai, bdlop): (Vec<_>, Vec<_>) = blocks.iter().partition(|b| b.placement == AJTAI);
        assert_eq!(req.m1, ajtai.iter().map(|b| b.rows).sum::<usize>());
        assert_eq!(req.l, carries + bdlop.iter().map(|b| b.rows).sum::<usize>());
        assert_eq!(
            req.alpha_squared,
            ajtai.iter().map(|b| share(b)).sum::<u128>()
        );
        // Nothing else depends on the placement.
        let mut same = req.clone();
        (same.m1, same.l, same.alpha_squared) =
            (all_ajtai.m1, all_ajtai.l, all_ajtai.alpha_squared);
        assert_eq!(same, all_ajtai, "{placement:?}");
    }
    assert_eq!(
        (all_ajtai.m1, all_ajtai.alpha_squared),
        (18, 640 + 64 + 128 + 128 * 4)
    );
    // The degree checks of `requirements`.
    let st = placed([AJTAI; 3], &w).0;
    assert_eq!(st.blocks(32).err(), Some(Error::Parameter("target ring")));
    assert_eq!(st.blocks(256).err(), Some(Error::Parameter("target ring")));
    assert_eq!(st.blocks(128).unwrap()[0].rows, 5);
}

#[test]
fn declarations() {
    let mut st = Statement::new(Ring::new(P, 128).unwrap());
    // An unbounded block has no place in the Ajtai part, whose norm bounds everything.
    assert_eq!(
        st.var_placed("u", 1, Norm::Unbounded, AJTAI).err(),
        Some(Error::Parameter("variable declaration"))
    );
    assert_eq!(st.var_placed("u", 1, Norm::Unbounded, BDLOP), Ok(0));
    assert_eq!(st.var_placed("v", 2, Norm::Binary, BDLOP), Ok(1));
    // The checks of `var` apply: a repeated name, and beta 0.
    assert!(st.var_placed("v", 1, Norm::L2Squared(1), AJTAI).is_err());
    assert!(st.var_placed("z", 1, Norm::Linf(0), BDLOP).is_err());
    for norm in [Norm::L2Squared(5), Norm::Binary, Norm::Linf(3)] {
        assert_eq!(Placement::default_for(norm), AJTAI);
    }
    assert_eq!(Placement::default_for(Norm::Unbounded), BDLOP);
    let blocks = st.blocks(64).unwrap();
    assert_eq!(
        (blocks[0].placement, blocks[1].placement, blocks[1].rows),
        (BDLOP, BDLOP, 4)
    );
}

#[test]
fn the_default_placement_compiles_as_var_did() {
    // `var` is `var_placed` with `Placement::default_for`: the same requirements, the same
    // compiled statement and the same witness map, hence the same proofs.
    let w = integers(0x2567_0b02);
    let (st, forms, witness) = placed([AJTAI; 3], &w);
    let ring = Ring::new(P, 128).unwrap();
    let mut by_var = Statement::new(ring);
    by_var.var("a", 5, Norm::L2Squared(640)).unwrap();
    by_var.var("s", 2, Norm::L2Squared(64)).unwrap();
    by_var.var("x", 1, Norm::Binary).unwrap();
    by_var.var("y", 1, Norm::Linf(2)).unwrap();
    by_var.eq_mod_p(forms[0].clone()).unwrap();
    by_var.const_coeff_zero(forms[1].clone()).unwrap();
    let req = st.requirements(64, U256::from_u64(Q)).unwrap();
    assert_eq!(by_var.requirements(64, U256::from_u64(Q)).unwrap(), req);
    let params = fitted(&req, "placement-default-test-only");
    let (a, b) = (
        st.compile(params.clone()).unwrap(),
        by_var.compile(params).unwrap(),
    );
    assert_eq!(a.statement(), b.statement());
    assert_eq!(a.map_witness(&witness), b.map_witness(&witness));
}

#[test]
fn bounded_blocks_prove_from_the_bdlop_part() {
    const CTX: &[u8] = b"jali-test/placement";
    let w = integers(0x2567_0b03);
    let (st, forms, witness) = placed([BDLOP; 3], &w);
    let req = st.requirements(64, U256::from_u64(Q)).unwrap();
    assert_eq!((req.m1, req.l, req.alpha_squared), (10, 9, 640));
    let params = fitted(&req, "placement-bdlop-test-only");
    let compiled = st.compile(params.clone()).unwrap();
    // The Ajtai part holds a; the BDLOP part s, x and y, in declaration order, then the carry.
    let (s1, m) = compiled.map_witness(&witness).unwrap();
    let q = common::narrow(&params.prime_factors[0]) as i128;
    let (proof_ring, lifted) = (Ring::new(q, 64).unwrap(), Ring::new(q, 128).unwrap());
    let split = |c: &Vec<i128>| iso::split(&poly(&lifted, c.clone()), proof_ring.clone()).unwrap();
    let parts = |range: std::ops::Range<usize>| w[range].iter().flat_map(split).collect::<Vec<_>>();
    assert_eq!(s1.entries(), parts(0..5).as_slice());
    let carry = integer_value(&forms[1], &w)[0];
    assert_eq!(carry % 13, 0);
    let mut expected = parts(5..9);
    expected.push(Poly::constant(proof_ring.clone(), carry / 13));
    assert_eq!(m.entries(), expected.as_slice());
    let proof = compiled
        .prove_with_seed([3; 32], &witness, CTX, [4; 32])
        .unwrap();
    compiled.verify([3; 32], &proof, CTX).unwrap();
    assert!(compiled.verify([3; 32], &proof, b"other").is_err());
    // The norms of the placed blocks are still enforced.
    let ring = Ring::new(P, 128).unwrap();
    for (name, index, value) in [("s", 0, 9), ("x", 0, 2), ("y", 0, 3)] {
        let mut bad = witness.clone();
        let mut c = w[match name {
            "s" => 5,
            "x" => 7,
            _ => 8,
        }]
        .clone();
        c[index] = value;
        bad.get_mut(name).unwrap()[0] = poly(&ring, c);
        assert_eq!(
            compiled.map_witness(&bad).err(),
            Some(Error::Witness),
            "{name}"
        );
    }
    // Parameters of one placement do not compile the other.
    let all_ajtai = placed([AJTAI; 3], &w).0;
    let other = fitted(
        &all_ajtai.requirements(64, U256::from_u64(Q)).unwrap(),
        "placement-ajtai-test-only",
    );
    assert_eq!(
        st.compile(other).err(),
        Some(Error::Parameter("compiled witness dimensions"))
    );
    assert_eq!(
        all_ajtai.compile(params).err(),
        Some(Error::Parameter("bounded witness norm budget"))
    );
}

/// Parameters at the first prime from $`2^{40}`$ up, in steps of one bit, that the port fits:
/// the Ajtai part's bound sets how large $`q`$ must be for MSIS.
fn fit_anywhere(req: &Requirements, id: &str) -> TboxParams {
    for bits in 40..62 {
        let mut q = next_prime_5_mod_8(1 << bits);
        for _ in 0..20 {
            if let Ok(params) = fit(req, id, vec![q], 64, RANK) {
                return params;
            }
            q = next_prime_5_mod_8(q + 1);
        }
    }
    panic!("no prime fits {id}");
}

#[test]
fn a_large_bound_in_the_bdlop_part_shrinks_the_proof() {
    const CTX: &[u8] = b"jali-test/placement-size";
    // A w + s + t = 0 modulo 12289, degree 128: w one polynomial with ||w||^2 <= 2^24, s ten
    // with ||s||^2 <= 1000. In the Ajtai part, w's bound sets sigma_1 for all 22 polynomials.
    let ring = Ring::new(12289, 128).unwrap();
    let blocks = [
        Block {
            name: "w".into(),
            length: 1,
            norm: Norm::L2Squared(1 << 24),
        },
        Block {
            name: "s".into(),
            length: 10,
            norm: Norm::L2Squared(1000),
        },
    ];
    let mut rng = SplitMix64(0x2567_0b04);
    let mut wc = vec![0i128; 128];
    (wc[0], wc[5]) = (4000, -800);
    let w = vec![poly(&ring, wc)];
    let s: Vec<Poly> = (0..10)
        .map(|_| {
            let mut c = vec![0i128; 128];
            for _ in 0..10 {
                c[rng.below(128) as usize] = rng.below(3) as i128 - 1;
            }
            poly(&ring, c)
        })
        .collect();
    let entries: Vec<Poly> = (0..11).map(|_| sparse(&mut rng, &ring)).collect();
    let mut t = Poly::zero(ring.clone());
    for (a, x) in entries.iter().zip(w.iter().chain(&s)) {
        t = t.sub(&a.mul(x).unwrap()).unwrap();
    }
    let a = PolyMat::new(ring.clone(), 1, 11, entries).unwrap();
    let t = PolyVec::new(ring.clone(), vec![t]).unwrap();
    let witness = Witness::from([("w".into(), w), ("s".into(), s)]);
    // The requirements of both placements, from the named builder of the same statement.
    let requirements = |placement: Placement| {
        let mut st = Statement::new(ring.clone());
        st.var_placed("w", 1, Norm::L2Squared(1 << 24), placement)
            .unwrap();
        st.var("s", 10, Norm::L2Squared(1000)).unwrap();
        let mut eq = st.constant(t.entries()[0].clone()).unwrap();
        for (j, (name, i)) in [("w", 0)]
            .into_iter()
            .chain((0..10).map(|i| ("s", i)))
            .enumerate()
        {
            let term = st.variable(name, i).unwrap().scale(a.get(0, j).unwrap());
            eq = eq.add(&term.unwrap()).unwrap();
        }
        st.eq_mod_p(eq).unwrap();
        st.requirements(64, U256::from_u64(Q)).unwrap()
    };
    let (in_ajtai, in_bdlop) = (requirements(AJTAI), requirements(BDLOP));
    assert_eq!((in_ajtai.m1, in_ajtai.l), (22, 0));
    assert_eq!((in_bdlop.m1, in_bdlop.l), (20, 2));
    let at = |req: &Requirements, id| {
        let params = fit_anywhere(req, id);
        let threshold = lifting_threshold(req, 12289, params.log_sigma[3]);
        assert!(params.check().unwrap().q > U256::from_u128(threshold));
        params
    };
    let ajtai_params = at(&in_ajtai, "placement-size-ajtai-test-only");
    let bdlop_params = at(&in_bdlop, "placement-size-bdlop-test-only");
    let size = |p: &TboxParams| p.check().unwrap().estimated_proof_bytes;
    assert!(size(&bdlop_params) < size(&ajtai_params));
    assert!(bdlop_params.log_sigma[0] < ajtai_params.log_sigma[0]);
    // lin::compile_placed realizes the placement; each set compiles only its own.
    let default = lin::compile(&a, &t, &blocks, ajtai_params.clone()).unwrap();
    let moved = lin::compile_placed(&a, &t, &blocks, &["w"], bdlop_params.clone()).unwrap();
    let mut lengths = Vec::new();
    for (compiled, seed) in [(&default, 5), (&moved, 6)] {
        let bytes = compiled
            .prove_bytes_with_seed([seed; 32], &witness, CTX, [seed + 1; 32])
            .unwrap();
        compiled.verify_bytes([seed; 32], &bytes, CTX).unwrap();
        lengths.push(bytes.len());
    }
    assert!(lengths[1] < lengths[0], "{lengths:?}");
    // In the Ajtai part, w's bound exceeds the other set's budget, which compile checks first.
    assert_eq!(
        lin::compile(&a, &t, &blocks, bdlop_params.clone()).err(),
        Some(Error::Parameter("bounded witness norm budget"))
    );
    assert_eq!(
        lin::compile_placed(&a, &t, &blocks, &["w"], ajtai_params).err(),
        Some(Error::Parameter("compiled witness dimensions"))
    );
    assert_eq!(
        lin::compile_placed(&a, &t, &blocks, &["v"], bdlop_params).err(),
        Some(Error::Index)
    );
}

/// The serde forms: `Norm` and `Placement` as the tool's requests write them, and the blocks.
#[cfg(feature = "serde")]
mod serde_forms {
    use super::*;
    use serde_json::json;

    #[test]
    fn norms_placements_and_blocks() {
        for (norm, value) in [
            (Norm::L2Squared(2950), json!({"l2_squared": 2950})),
            (Norm::Binary, json!("binary")),
            (Norm::Linf(u64::MAX), json!({"linf": u64::MAX})),
            (Norm::Unbounded, json!("unbounded")),
            (Norm::LinfExact(512), json!({"linf_exact": 512})),
        ] {
            assert_eq!(serde_json::to_value(norm).unwrap(), value);
            assert_eq!(serde_json::from_value::<Norm>(value).unwrap(), norm);
        }
        assert_eq!(serde_json::to_value(AJTAI).unwrap(), json!("ajtai"));
        assert_eq!(serde_json::to_value(BDLOP).unwrap(), json!("bdlop"));
        let w = integers(0x2567_0b05);
        let st = placed([BDLOP, AJTAI, BDLOP], &w).0;
        let blocks = st.blocks(64).unwrap();
        let text = serde_json::to_string(&blocks).unwrap();
        assert_eq!(
            text,
            concat!(
                r#"[{"name":"a","rows":10,"norm":{"l2_squared":640},"placement":"ajtai"},"#,
                r#"{"name":"s","rows":4,"norm":{"l2_squared":64},"placement":"bdlop"},"#,
                r#"{"name":"x","rows":2,"norm":"binary","placement":"ajtai"},"#,
                r#"{"name":"y","rows":2,"norm":{"linf":2},"placement":"bdlop"}]"#
            )
        );
        assert_eq!(
            serde_json::from_str::<Vec<BlockRequirement>>(&text).unwrap(),
            blocks
        );
        // Strict: an unknown field or an unknown placement is refused.
        let mut value = serde_json::to_value(&blocks[0]).unwrap();
        value["extra"] = json!(1);
        assert!(serde_json::from_value::<BlockRequirement>(value).is_err());
        let mut value = serde_json::to_value(&blocks[0]).unwrap();
        value["placement"] = json!("search");
        assert!(serde_json::from_value::<BlockRequirement>(value).is_err());
        let bytes = postcard::to_allocvec(&blocks).unwrap();
        assert_eq!(
            postcard::from_bytes::<Vec<BlockRequirement>>(&bytes).unwrap(),
            blocks
        );
        // Postcard writes a norm's variant index: LinfExact came last, so the others kept
        // theirs.
        for (norm, bytes) in [
            (Norm::L2Squared(5), vec![0, 5]),
            (Norm::Binary, vec![1]),
            (Norm::Linf(5), vec![2, 5]),
            (Norm::Unbounded, vec![3]),
            (Norm::LinfExact(5), vec![4, 5]),
        ] {
            assert_eq!(postcard::to_allocvec(&norm).unwrap(), bytes);
        }
    }
}
