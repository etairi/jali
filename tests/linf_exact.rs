//! Exact $`\ell_\infty`$ bounds ([`Norm::LinfExact`]): each coefficient $`v`$ is
//! $`\sum_bc_bx_b-\beta`$ over bits $`x_b`$ in the binary block, substituted into every
//! constraint. The tests check the rows, shares and integer bounds that `requirements` exports
//! (bits in `n_bin`, $`\beta`$ in $`F`$ both honestly and at extraction, for each exact factor of
//! a product, nothing in `n_prime` or `linf_bound`, no `linf` section of their own), the range
//! condition $`2\beta<p`$, strict for an even $`p`$, the witness map ($`\pm\beta`$ admitted,
//! $`\beta+1`$ refused, the bits it writes), a statement that mixes exact bounds with every
//! other norm, subring and BDLOP blocks, lifted quadratic, linear and constant-coefficient
//! constraints, which proves at both proof degrees, `lin::compile` with an exact block, and that
//! a value beyond $`\beta`$, which a [`Norm::Linf`] variable admits up to the extraction bound
//! (`tests/linf.rs`), has no binary encoding and the prover refuses it. Parameters come from the
//! test port of the parameter tool (`common::params`).
use jali::{
    Error, lin,
    lnp::Prover,
    math::{Poly, PolyMat, PolyVec, Ring, U256, iso},
    params::TboxParams,
    quad::QuadEq,
    statement::{Extraction, Norm, Placement, Requirements, Statement},
};
use std::{collections::BTreeMap, sync::Arc};

mod common;
use common::{
    SplitMix64, narrow,
    params::{extraction_bound, fit_upward, lifting_threshold, linf_f, range_width},
    ring::{
        VarBound::{self, LinfIn, Squared},
        f_bound_linf, integer_value, poly,
    },
};

const Q: u64 = 1099511627917;
const CTX: &[u8] = b"jali-test/linf-exact";
type Witness = BTreeMap<String, Vec<Poly>>;

/// The weights $`c_b=\lfloor(2\beta+2^b)/2^{b+1}\rfloor`$ for $`b<\mathrm{bitlen}(2\beta)`$, from
/// their definition.
fn weights(beta: u64) -> Vec<i128> {
    let b = 2 * u128::from(beta);
    (0..u128::BITS - b.leading_zeros())
        .map(|i| ((b + (1 << i)) >> (i + 1)) as i128)
        .collect()
}

/// The port's parameters at the first prime above the lifting threshold, $`E`$ included.
fn fitted(req: &Requirements, p: u128, degree: usize, id: &str) -> TboxParams {
    let threshold = lifting_threshold(req, p, range_width(req, degree));
    fit_upward(req, id, degree, (threshold as u64 + 1).max(Q)).1
}

/// A statement over $`\mathbb Z_{1092}[X]/(X^{256}+1)`$, whose modulus $`7\cdot156`$ both
/// constraint moduli divide and which holds $`\pm512`$, with the integer witness per variable
/// index and the constraint forms:
///
/// - `s`: two polynomials, $`\|s\|^2\le64`$; `x`: binary; `u`: $`|u_i|\le1`$ exactly; `w`: two
///   elements of the subring of degree 128, $`|w_i|\le5`$ exactly; `y`: $`\ell_\infty\le3`$
///   ([`Norm::Linf`]); `v`: the subring of degree 128, $`|v_i|\le512`$ exactly, in the BDLOP
///   part;
/// - A: $`a\,us_0+b\,w_0x+c\,uw_1+e\,uy+f=0`$ modulo 13 (committed carries);
/// - B: $`\mathrm{ct}(X^{-6}v+\sigma(g)w_0)=t`$ modulo 12, reading $`v_3`$ (a packed carry);
/// - C: $`h\,u+i\,v+j=0`$ modulo 12 (implicit quotients);
///
/// each made true on the witness, which has every bound reached in both signs.
struct Built {
    st: Statement,
    forms: Vec<QuadEq>,
    w: Vec<Vec<i128>>,
    witness: Witness,
}
fn build(seed: u64) -> Built {
    let ring = Ring::new(1092, 256).unwrap();
    let (r12, r13) = (Ring::new(12, 256).unwrap(), Ring::new(13, 256).unwrap());
    let mut rng = SplitMix64(seed);
    let mut st = Statement::new(ring.clone());
    st.var("s", 2, Norm::L2Squared(64)).unwrap();
    st.var("x", 1, Norm::Binary).unwrap();
    st.var("u", 1, Norm::LinfExact(1)).unwrap();
    st.var_subring("w", 2, 128, Norm::LinfExact(5), Placement::Ajtai)
        .unwrap();
    st.var("y", 1, Norm::Linf(3)).unwrap();
    st.var_subring("v", 1, 128, Norm::LinfExact(512), Placement::Bdlop)
        .unwrap();
    let mut w = vec![vec![0i128; 256]; 8];
    for _ in 0..32 {
        let (i, j) = (rng.below(2) as usize, rng.below(256) as usize);
        w[i][j] = rng.below(3) as i128 - 1;
    }
    w[2] = rng.binary(256);
    w[3] = rng.ternary(256);
    (w[3][0], w[3][1]) = (1, -1);
    for j in 0..128 {
        w[4][2 * j] = rng.below(11) as i128 - 5;
        w[5][2 * j] = rng.below(11) as i128 - 5;
        w[7][2 * j] = rng.below(1025) as i128 - 512;
    }
    (w[4][0], w[5][2], w[7][4], w[7][6]) = (5, -5, 512, -512);
    w[6] = (0..256).map(|_| rng.below(7) as i128 - 3).collect();
    let names = [
        ("s", 0),
        ("s", 1),
        ("x", 0),
        ("u", 0),
        ("w", 0),
        ("w", 1),
        ("y", 0),
        ("v", 0),
    ];
    let v = |r: &Arc<Ring>, i: usize| st.variable_in(r, names[i].0, names[i].1).unwrap();
    let mut sparse = |r: &Arc<Ring>| {
        let mut c = vec![0i128; 256];
        for _ in 0..3 {
            c[rng.below(256) as usize] = rng.below(7) as i128 - 3;
        }
        poly(r, c)
    };
    let settle = |form: QuadEq| {
        let r = form.r0.ring().clone();
        let value = poly(&r, integer_value(&form, &w));
        form.add(&st.constant_in(value.neg()).unwrap()).unwrap()
    };
    let product = |r: &Arc<Ring>, i: usize, j: usize, c: Poly| {
        v(r, i).product_affine(&v(r, j)).unwrap().scale(&c).unwrap()
    };
    let (a, b, c, e) = (sparse(&r13), sparse(&r13), sparse(&r13), sparse(&r13));
    let (g, h, i) = (sparse(&r12), sparse(&r12), sparse(&r12));
    let forms = vec![
        settle(
            product(&r13, 3, 0, a)
                .add(&product(&r13, 4, 2, b))
                .unwrap()
                .add(&product(&r13, 3, 5, c))
                .unwrap()
                .add(&product(&r13, 3, 6, e))
                .unwrap(),
        ),
        settle(
            st.coefficient_in(&r12, "v", 0, 3)
                .unwrap()
                .add(&v(&r12, 4).scale(&g.auto()).unwrap())
                .unwrap(),
        ),
        settle(
            v(&r12, 3)
                .scale(&h)
                .unwrap()
                .add(&v(&r12, 7).scale(&i).unwrap())
                .unwrap(),
        ),
    ];
    st.eq_mod_p(forms[0].clone()).unwrap();
    st.const_coeff_zero(forms[1].clone()).unwrap();
    st.eq_mod_p(forms[2].clone()).unwrap();
    let block = |range: std::ops::Range<usize>| range.map(|i| poly(&ring, w[i].clone())).collect();
    let witness = Witness::from([
        ("s".into(), block(0..2)),
        ("x".into(), block(2..3)),
        ("u".into(), block(3..4)),
        ("w".into(), block(4..6)),
        ("y".into(), block(6..7)),
        ("v".into(), block(7..8)),
    ]);
    Built {
        st,
        forms,
        w,
        witness,
    }
}

/// The oracle's bounds per variable index, with `y` at `e`.
fn bounds(e: u128) -> [VarBound; 8] {
    [
        Squared(64),
        Squared(64),
        Squared(256),
        LinfIn(1, 256),
        LinfIn(5, 128),
        LinfIn(5, 128),
        LinfIn(e, 256),
        LinfIn(512, 128),
    ]
}

#[test]
fn requirements_count_bits_and_bound_by_beta() {
    let b = build(0x2567_2210);
    let f: Vec<u128> = b
        .forms
        .iter()
        .map(|f| f_bound_linf(f, &bounds(3)))
        .collect();
    for degree in [64, 128] {
        let k = 256 / degree;
        let req = b.st.requirements(degree, U256::from_u64(Q)).unwrap();
        // Bits: u 2 per component, w 4 (2 beta = 10), v 11 (2 beta = 1024), of their committed
        // components (w and v: k/2 per element).
        let (u, w, v) = (2 * k, 2 * (k / 2) * 4, (k / 2) * 11);
        assert_eq!(req.n_bin, k + u + w + v, "degree {degree}");
        assert_eq!(req.m1, 2 * k + k + u + w + k);
        // v in BDLOP, A's k carries and one packed polynomial.
        assert_eq!(req.l, v + k + 1);
        // A's carries, the packed row, C's quotients and y's rows; no rows for the bits.
        assert_eq!(req.n_prime, 3 * k + 1);
        // Shares: s, x's 256 coefficients, bits of u (256 * 2) and w (2 * 128 * 4), y 256 * 9.
        assert_eq!(req.alpha_squared, 64 + 256 + 512 + 1024 + 2304);
        let blocks = b.st.blocks(degree).unwrap();
        assert_eq!(
            blocks.iter().map(|b| b.rows).collect::<Vec<_>>(),
            [2 * k, k, u, w, k, v]
        );
        // F with beta for the exact variables and 3 for y; linf_bound does not see 512.
        let f13 = f[0];
        let f12 = f[1].max(f[2]);
        assert_eq!(
            req.lifted_moduli
                .iter()
                .map(|m| (narrow(&m.modulus), narrow(&m.max_integer_coefficient)))
                .collect::<Vec<_>>(),
            [(12, f12), (13, f13)]
        );
        assert_eq!(
            req.linf_bound,
            f13.div_ceil(13).max(f12.div_ceil(12)).max(3)
        );
        // Only A reads the Linf variable y: its terms with y at E, the others exact.
        let linf = req.linf.as_ref().unwrap();
        assert_eq!(linf.lifting.len(), 1);
        for e in [3, 1 << 30] {
            assert_eq!(
                linf_f(&linf.lifting[0], e),
                f_bound_linf(&b.forms[0], &bounds(e))
            );
        }
        assert_eq!(narrow(&linf.lifting[0].quadratic), 0);
        // The per-slot bound: A's and C's 256 slots, B's one and y's 256 coefficients at 3.
        let per_slot = 256 * f[0].div_ceil(13).pow(2)
            + f[1].div_ceil(12).pow(2)
            + 256 * f[2].div_ceil(12).pow(2)
            + 256 * 9;
        assert_eq!(req.approx_alpha_squared, Some(per_slot));
    }
    // Exact variables alone: no linf section, and linf_bound 1, not beta.
    let mut st = Statement::new(Ring::new(13, 128).unwrap());
    st.var("u", 2, Norm::LinfExact(6)).unwrap();
    let r12 = Ring::new(12, 128).unwrap();
    let clause = st.variable_in(&r12, "u", 1).unwrap();
    st.const_coeff_zero(clause).unwrap();
    let req = st.requirements(64, U256::from_u64(Q)).unwrap();
    assert_eq!(req.linf, None);
    // 2 * 6 = 12 has 4 bits; one clause of bound 6 over 12.
    assert_eq!((req.n_bin, req.n_prime, req.linf_bound), (2 * 2 * 4, 1, 1));
}

#[test]
fn product_bounds_carry_the_beta_of_each_exact_factor() {
    // In `build` the first exact factor of every product is u, bounded by 1, so a weight that
    // leaves out that factor's beta goes unnoticed there. Here it is w (two elements of the
    // subring of degree 128, bounded by 5 exactly) in both products of
    // (2 - X^7) w_0 w_1 + 3 X^3 w_0 y + 1 = 0 modulo 13, with y bounded by 3 in Norm::Linf.
    let r13 = Ring::new(13, 256).unwrap();
    let mut st = Statement::new(Ring::new(1092, 256).unwrap());
    st.var_subring("w", 2, 128, Norm::LinfExact(5), Placement::Ajtai)
        .unwrap();
    st.var("y", 1, Norm::Linf(3)).unwrap();
    let v = |name: &str, i: usize| st.variable_in(&r13, name, i).unwrap();
    let c = |values: &[(usize, i128)]| {
        let mut x = vec![0i128; 256];
        for (i, a) in values {
            x[*i] = *a;
        }
        poly(&r13, x)
    };
    let form = v("w", 0)
        .product_affine(&v("w", 1))
        .unwrap()
        .scale(&c(&[(0, 2), (7, -1)]))
        .unwrap()
        .add(
            &v("w", 0)
                .product_affine(&v("y", 0))
                .unwrap()
                .scale(&c(&[(3, 3)]))
                .unwrap(),
        )
        .unwrap()
        .add(&st.constant_in(Poly::constant(r13.clone(), 1)).unwrap())
        .unwrap();
    st.eq_mod_p(form.clone()).unwrap();
    let bounds = |e: u128| [LinfIn(5, 128), LinfIn(5, 128), LinfIn(e, 256)];
    let req = st.requirements(64, U256::from_u64(Q)).unwrap();
    // The honest F, with y at its beta.
    assert_eq!(
        narrow(&req.lifted_moduli[0].max_integer_coefficient),
        f_bound_linf(&form, &bounds(3))
    );
    // w_0 w_1 weighs ||2 - X^7||_1 ceil(sqrt(128 * 128)) 5 * 5 in `exact`, and w_0 y weighs
    // ||3 X^3||_1 ceil(sqrt(128 * 256)) 5 per unit of E in `linear`.
    let linf = req.linf.as_ref().unwrap();
    assert_eq!(narrow(&linf.lifting[0].exact), 1 + 3 * 128 * 25);
    assert_eq!(narrow(&linf.lifting[0].linear), 3 * 182 * 5);
    for e in [3, 1 << 30] {
        assert_eq!(linf_f(&linf.lifting[0], e), f_bound_linf(&form, &bounds(e)));
    }
}

#[test]
fn the_range_condition_needs_two_beta_below_the_statement_modulus() {
    // A clause modulo 12, which does not divide 13, reads u: 2 beta < 13.
    let refused = Some(Error::Parameter("variable range above statement modulus"));
    for (beta, admitted) in [(6, true), (7, false), (1 << 40, false)] {
        let mut st = Statement::new(Ring::new(13, 128).unwrap());
        st.var("u", 1, Norm::LinfExact(beta)).unwrap();
        let r12 = Ring::new(12, 128).unwrap();
        let clause = st.variable_in(&r12, "u", 0).unwrap();
        st.const_coeff_zero(clause).unwrap();
        let result = st.requirements(64, U256::from_u64(Q));
        assert_eq!(result.is_ok(), admitted, "beta {beta}");
        if !admitted {
            assert_eq!(result.err(), refused);
        }
    }
    // Modulo 13 itself there is no condition.
    let mut st = Statement::new(Ring::new(13, 128).unwrap());
    st.var("u", 1, Norm::LinfExact(1 << 40)).unwrap();
    let clause = st.variable("u", 0).unwrap();
    st.const_coeff_zero(clause).unwrap();
    st.requirements(64, U256::from_u64(Q)).unwrap();
    // An even p: -p/2 and p/2 are one residue, which a witness reads as p/2, but bits of bound
    // p/2 also encode -p/2, so 2 beta = p is refused. A clause modulo 7, which divides none of
    // these p, reads u.
    for (p, beta, admitted) in [
        (12, 5, true),
        (12, 6, false),
        (16, 7, true),
        (16, 8, false),
        (156, 77, true),
        (156, 78, false),
    ] {
        let mut st = Statement::new(Ring::new(p, 128).unwrap());
        st.var("u", 1, Norm::LinfExact(beta)).unwrap();
        let r7 = Ring::new(7, 128).unwrap();
        let clause = st.variable_in(&r7, "u", 0).unwrap();
        st.const_coeff_zero(clause).unwrap();
        let result = st.requirements(64, U256::from_u64(Q));
        assert_eq!(result.is_ok(), admitted, "p {p}, beta {beta}");
        if !admitted {
            assert_eq!(result.err(), refused, "p {p}, beta {beta}");
            // compile refuses through requirements, before it reads the dimensions.
            assert_eq!(st.compile(jali::params::toy_d64()).err(), refused);
        }
    }
}

/// For each committed component of the statement polynomial `values` (stride `stride` of `k`
/// components), its bits from `bits` onwards decode to the values: $`\sum_bc_bx_b-\beta`$.
fn check_bits(bits: &[Poly], values: &[i128], k: usize, stride: usize, beta: u64) {
    let w = weights(beta);
    let d = bits[0].ring().degree();
    let mut at = 0;
    for c in (0..k).step_by(stride) {
        for t in 0..d {
            let x: i128 = w
                .iter()
                .enumerate()
                .map(|(b, c)| {
                    let bit = bits[at + b].coefficient_i128(t).unwrap();
                    assert!(bit == 0 || bit == 1);
                    bit * c
                })
                .sum();
            assert_eq!(
                x - beta as i128,
                values[k * t + c],
                "component {c}, position {t}"
            );
        }
        at += w.len();
    }
    assert_eq!(at, bits.len());
}

#[test]
fn exact_variables_prove_at_both_degrees() {
    let b = build(0x2567_2211);
    for degree in [64, 128] {
        let k = 256 / degree;
        let req = b.st.requirements(degree, U256::from_u64(Q)).unwrap();
        let params = fitted(&req, 1092, degree, "linf-exact-test-only");
        let compiled = b.st.compile(params.clone()).unwrap();
        for (name, want) in [
            ("u", Extraction::LinfExact(1)),
            ("w", Extraction::LinfExact(5)),
            ("v", Extraction::LinfExact(512)),
            ("y", Extraction::Linf(extraction_bound(params.log_sigma[3]))),
        ] {
            assert_eq!(compiled.extraction_bound(name), Ok(want));
        }
        let (s1, m) = compiled.map_witness(&b.witness).unwrap();
        // The bits in their places: u after s and x, w after u, v first among the messages.
        let (u, w, v) = (2 * k, 2 * (k / 2) * 4, (k / 2) * 11);
        check_bits(&s1.entries()[3 * k..3 * k + u], &b.w[3], k, 1, 1);
        check_bits(
            &s1.entries()[3 * k + u..3 * k + u + w / 2],
            &b.w[4],
            k,
            2,
            5,
        );
        check_bits(
            &s1.entries()[3 * k + u + w / 2..3 * k + u + w],
            &b.w[5],
            k,
            2,
            5,
        );
        check_bits(&m.entries()[..v], &b.w[7], k, 2, 512);
        // The binary block: x, then the bits of u, w and v.
        let binary = compiled.statement().binary.as_ref().unwrap();
        let rows = binary.evaluate(&s1, &m).unwrap();
        let mut want: Vec<Poly> = s1.entries()[2 * k..3 * k + u + w].to_vec();
        want.extend_from_slice(&m.entries()[..v]);
        assert_eq!(rows.entries(), &want[..]);
        let bytes = compiled
            .prove_bytes_with_seed([3; 32], &b.witness, CTX, [4; 32])
            .unwrap();
        compiled.verify_bytes([3; 32], &bytes, CTX).unwrap();
        assert!(compiled.verify_bytes([3; 32], &bytes, b"other").is_err());
        // Beyond beta, the witness map refuses before any constraint (the honest witness has
        // +-beta in every exact block).
        for (name, element, index, value) in [
            ("u", 0, 5, 2),
            ("u", 0, 5, -2),
            ("w", 1, 8, 6),
            ("v", 0, 10, -513),
            ("w", 0, 3, 1),
        ] {
            let mut edited = b.witness.clone();
            edited.get_mut(name).unwrap()[element]
                .set_coefficient(index, value)
                .unwrap();
            assert_eq!(
                compiled.map_witness(&edited).err(),
                Some(Error::Witness),
                "{name}[{element}][{index}] = {value}"
            );
        }
    }
}

#[test]
fn a_value_beyond_beta_has_no_binary_encoding_and_the_prover_refuses_it() {
    // tests/linf.rs proves ct(u - 5) = 0 over Z_13 from u_0 = 5 for u bounded by 1 in
    // Norm::Linf, once the prover skips the witness map. Bounded exactly, u_0 = 5 needs
    // x = 6 = c_0 x_0 + c_1 x_1 with the weights (1, 1): no binary bits reach it. With the
    // "bits" (3, 3) every compiled equation holds, and the prover's witness check refuses them.
    let r13 = Ring::new(13, 128).unwrap();
    let mut st = Statement::new(r13.clone());
    st.var("s", 4, Norm::L2Squared(64)).unwrap();
    st.var("u", 1, Norm::LinfExact(1)).unwrap();
    let clause = st
        .variable("u", 0)
        .unwrap()
        .add(&st.constant(Poly::constant(r13.clone(), -5)).unwrap())
        .unwrap();
    st.const_coeff_zero(clause).unwrap();
    let req = st.requirements(64, U256::from_u64(Q)).unwrap();
    assert_eq!((req.m1, req.n_bin, req.l, req.n_prime), (12, 4, 1, 1));
    let params = fitted(&req, 13, 64, "linf-exact-relaxed-test-only");
    let compiled = st.compile(params.clone()).unwrap();
    let q = narrow(&params.prime_factors[0]) as i128;
    let proof_ring = Ring::new(q, 64).unwrap();
    // s zero; u's two components, each as its two bits, all encoding 0 but position 0.
    let witness = |x0: i128, x1: i128| {
        let mut polys = vec![Poly::zero(proof_ring.clone()); 12];
        for component in 0..2 {
            polys[8 + 2 * component] = Poly::new(proof_ring.clone(), vec![1; 64]).unwrap();
        }
        polys[8].set_coefficient(0, x0).unwrap();
        polys[9].set_coefficient(0, x1).unwrap();
        PolyVec::new(proof_ring.clone(), polys).unwrap()
    };
    let carry = PolyVec::new(proof_ring.clone(), vec![Poly::zero(proof_ring.clone())]).unwrap();
    let statement = compiled.statement();
    let equation = &statement.evaluation[0];
    let at = |s1: &PolyVec| {
        let full = jali::quad::interleave(s1, &carry).unwrap();
        equation
            .evaluate(&full)
            .unwrap()
            .coefficient_i128(0)
            .unwrap()
    };
    // With carry 0 the equation reads x_0 + x_1 - 1 - 5 at position 0: never 0 for bits.
    for (x0, x1) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        assert_eq!(at(&witness(x0, x1)), x0 + x1 - 6);
    }
    let s1 = witness(3, 3);
    assert_eq!(at(&s1), 0);
    let mut prover = Prover::new([5; 32], Arc::new(params)).unwrap();
    prover.set_statement(statement.clone()).unwrap();
    prover.set_witness(&s1, &carry).unwrap();
    assert_eq!(
        prover.prove_with_seed(CTX, [6; 32]).err(),
        Some(Error::Witness)
    );
    // The honest witness u_0 = 0 would need 5 = 13 c: none exists either.
    let named = Witness::from([
        ("s".into(), vec![Poly::zero(r13.clone()); 4]),
        ("u".into(), vec![Poly::zero(r13.clone())]),
    ]);
    assert_eq!(compiled.map_witness(&named).err(), Some(Error::Witness));
}

#[test]
fn lin_compile_takes_an_exact_block() {
    // A w + t = 0 over Z_3329 of degree 256 with w (three polynomials) bounded by 2 exactly
    // and one exact-norm block, at proof degree 64.
    let ring = Ring::new(3329, 256).unwrap();
    let mut rng = SplitMix64(0x2567_2212);
    let w: Vec<Poly> = (0..3)
        .map(|_| poly(&ring, (0..256).map(|_| rng.below(5) as i128 - 2).collect()))
        .collect();
    let z = vec![poly(&ring, rng.ternary(256))];
    let a: Vec<Poly> = (0..8)
        .map(|_| poly(&ring, rng.uniform(256, 3329)))
        .collect();
    let t: Vec<Poly> = (0..2)
        .map(|row| {
            let mut acc = Poly::zero(ring.clone());
            for (j, v) in w.iter().chain(&z).enumerate() {
                acc = acc.add(&a[row * 4 + j].mul(v).unwrap()).unwrap();
            }
            acc.neg()
        })
        .collect();
    let am = PolyMat::new(ring.clone(), 2, 4, a).unwrap();
    let tv = PolyVec::new(ring.clone(), t).unwrap();
    let blocks = [
        lin::Block {
            name: "w".into(),
            length: 3,
            norm: Norm::LinfExact(2),
        },
        lin::Block {
            name: "z".into(),
            length: 1,
            norm: Norm::L2Squared(256),
        },
    ];
    // The requirements of the statement lin::compile builds.
    let mut st = Statement::new(ring.clone());
    st.var("w", 3, Norm::LinfExact(2)).unwrap();
    st.var("z", 1, Norm::L2Squared(256)).unwrap();
    for row in 0..2 {
        let mut eq = st.constant(tv.entries()[row].clone()).unwrap();
        for (j, (name, i)) in [("w", 0), ("w", 1), ("w", 2), ("z", 0)].iter().enumerate() {
            eq = eq
                .add(
                    &st.variable(name, *i)
                        .unwrap()
                        .scale(am.get(row, j).unwrap())
                        .unwrap(),
                )
                .unwrap();
        }
        st.eq_mod_p(eq).unwrap();
    }
    let req = st.requirements(64, U256::from_u64(Q)).unwrap();
    // Three bits per coefficient (2 beta = 4); implicit quotients only.
    assert_eq!(
        (req.n_bin, req.m1, req.l, req.n_prime),
        (3 * 4 * 3, 36 + 4, 0, 8)
    );
    let params = fitted(&req, 3329, 64, "linf-exact-lin-test-only");
    let compiled = lin::compile(&am, &tv, &blocks, params).unwrap();
    let witness = Witness::from([("w".into(), w), ("z".into(), z)]);
    let bytes = compiled
        .prove_bytes_with_seed([7; 32], &witness, CTX, [8; 32])
        .unwrap();
    compiled.verify_bytes([7; 32], &bytes, CTX).unwrap();
    assert_eq!(compiled.extraction_bound("w"), Ok(Extraction::LinfExact(2)));
    // An embedded subring element is a statement-ring polynomial like any other.
    let small = Ring::new(3329, 128).unwrap();
    let e = iso::embed(&Poly::constant(small, 2), ring).unwrap();
    assert_eq!(e.coefficient_i128(0), Ok(2));
}
