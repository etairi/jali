//! Variables bounded in $`\ell_\infty`$ ([`Norm::Linf`]), proven approximately: their
//! coefficients join the one approximate range proof, with the carries and quotients, under its
//! common bound `linf_bound`. A statement that mixes them with exact-norm and binary variables
//! and with carries proves and verifies at both proof degrees, also with constraints of their
//! own moduli. The witness map refuses a coefficient above $`\beta`$, but a proof bounds it only
//! by the extraction bound $`E=2\cdot`$`z4_bound`, which `Compiled::extraction_bound` reports: a
//! statement with no witness within $`\beta`$ proves from one within $`E`$. The lifting check
//! and the range condition use $`E`$, exactly at their boundaries, and the requirements export
//! what a parameter tool needs to reproduce both. Parameters come from the test port of the
//! parameter tool (`common::params`); each test fixes its seeds.
use jali::{
    Error, lin,
    lnp::{Prover, Verifier},
    math::{Poly, PolyMat, PolyVec, Ring, U256, iso},
    params::TboxParams,
    quad::QuadEq,
    statement::{Extraction, Norm, Requirements, Statement},
};
use std::{collections::BTreeMap, sync::Arc};

mod common;
use common::{
    SplitMix64, narrow,
    params::{extraction_bound, fit_downward, fit_upward, lifting_threshold, linf_f, range_width},
    ring::{
        VarBound::{self, Linf, Squared},
        f_bound_linf, integer_value, poly,
    },
};

/// The prime of `toy_d64`, where every search for a proof modulus starts.
const Q: u64 = 1099511627917;
const SEED: u64 = 0x2567_1f00;

type Witness = BTreeMap<String, Vec<Poly>>;

/// The refusal of a statement that violates the range condition of the witness relation.
const OUT_OF_RANGE: Option<Error> =
    Some(Error::Parameter("variable range above statement modulus"));

/// The exact quotients of an integer value by `p`, after checking that `p` divides it.
fn quotients(f: &[i128], p: i128) -> Vec<i128> {
    assert!(f.iter().all(|x| x % p == 0), "not a multiple of {p}");
    f.iter().map(|x| x / p).collect()
}

/// Parameters at the first prime above the lifting threshold that the port recomputes from the
/// requirements, $`E`$ included.
fn fitted(req: &Requirements, statement_modulus: u128, degree: usize, id: &str) -> TboxParams {
    let threshold = lifting_threshold(req, statement_modulus, range_width(req, degree));
    fit_upward(req, id, degree, (threshold as u64 + 1).max(Q)).1
}

/// A sparse public polynomial: `count` coefficients in $`[-6,6]`$ at random positions.
fn sparse(rng: &mut SplitMix64, ring: &Arc<Ring>, count: usize) -> Poly {
    let mut c = vec![0i128; ring.degree()];
    for _ in 0..count {
        let i = rng.below(ring.degree() as u64) as usize;
        c[i] = rng.below(13) as i128 - 6;
    }
    poly(ring, c)
}

/// The declared bounds of `mixed` per variable index ($`s_0,s_1,x,y,z`$): $`\|s\|^2\le64`$,
/// binary $`x`$ (128 coefficients), $`\|y\|_\infty\le1`$ and $`\|z\|_\infty\le3`$; with `e`,
/// both $`\ell_\infty`$ variables at $`e`$ instead.
fn mixed_bounds(e: Option<u128>) -> [VarBound; 5] {
    let (y, z) = e.map_or((1, 3), |e| (e, e));
    [Squared(64), Squared(64), Squared(128), Linf(y), Linf(z)]
}

/// The integer witness of `mixed`: $`s_0,s_1`$ ternary on 16 coefficients each, $`x`$ binary,
/// $`y`$ ternary and $`z`$ in $`[-3,3]`$, with $`y`$ and $`z`$ at $`\pm\beta`$ in their first
/// two coefficients; then `edit`.
fn mixed_witness(rng: &mut SplitMix64, edit: impl Fn(&mut [Vec<i128>])) -> Vec<Vec<i128>> {
    let mut sparse_ternary = || {
        let mut c = vec![0i128; 128];
        for _ in 0..16 {
            c[rng.below(128) as usize] = rng.below(3) as i128 - 1;
        }
        c
    };
    let (s0, s1) = (sparse_ternary(), sparse_ternary());
    let x = rng.binary(128);
    let mut y = rng.ternary(128);
    let mut z: Vec<i128> = (0..128).map(|_| rng.below(7) as i128 - 3).collect();
    (y[0], y[1], z[0], z[1]) = (1, -1, 3, -3);
    let mut w = vec![s0, s1, x, y, z];
    edit(&mut w);
    w
}

/// Over the degree-128 statement ring `ring`: `s` (two polynomials, $`\|s\|^2\le64`$), `x`
/// (binary), `y` ($`\ell_\infty\le1`$) and `z` ($`\ell_\infty\le3`$), and four constraints,
/// the $`j`$-th over `rings[j]`:
///
/// 0. $`3s_0y+b\,z+c\,x+d=0`$: committed carries, an $`\ell_\infty`$ times an exact-norm
///    variable;
/// 1. $`yz+e\,s_1+f=0`$: committed carries, two $`\ell_\infty`$ variables;
/// 2. $`g\,y+h\,s_0+i=0`$: implicit quotients;
/// 3. $`\mathrm{ct}(j\,z+k\,x+l)=0`$: one committed carry;
///
/// with sparse $`b,c,e,g,h,j,k`$ and constants that make the integer witness of
/// `mixed_witness` satisfy each constraint modulo its modulus. Returns the statement, the forms,
/// the named witness over `ring` and the integer witness.
fn mixed(
    ring: &Arc<Ring>,
    rings: [&Arc<Ring>; 4],
    seed: u64,
    edit: impl Fn(&mut [Vec<i128>]),
) -> (Statement, Vec<QuadEq>, Witness, Vec<Vec<i128>>) {
    let mut rng = SplitMix64(seed);
    let w = mixed_witness(&mut rng, edit);
    let mut st = Statement::new(ring.clone());
    st.var("s", 2, Norm::L2Squared(64)).unwrap();
    st.var("x", 1, Norm::Binary).unwrap();
    st.var("y", 1, Norm::Linf(1)).unwrap();
    st.var("z", 1, Norm::Linf(3)).unwrap();
    let names = [("s", 0), ("s", 1), ("x", 0), ("y", 0), ("z", 0)];
    let mut forms = Vec::new();
    for (j, r) in rings.into_iter().enumerate() {
        let v = |i: usize| st.variable_in(r, names[i].0, names[i].1).unwrap();
        let value = |i: usize| poly(r, w[i].clone());
        let mut public = || sparse(&mut rng, r, 4);
        // The form without its constant, and its value at the witness.
        let (form, at): (QuadEq, Poly) = match j {
            0 => {
                let (b, c) = (public(), public());
                let three = Poly::constant(r.clone(), 3);
                let form = v(0)
                    .product_affine(&v(3))
                    .unwrap()
                    .scale(&three)
                    .unwrap()
                    .add(&v(4).scale(&b).unwrap())
                    .unwrap()
                    .add(&v(2).scale(&c).unwrap())
                    .unwrap();
                let at = value(0)
                    .mul(&value(3))
                    .unwrap()
                    .mul(&three)
                    .unwrap()
                    .add(&b.mul(&value(4)).unwrap())
                    .unwrap()
                    .add(&c.mul(&value(2)).unwrap())
                    .unwrap();
                (form, at)
            }
            1 => {
                let e = public();
                let form = v(3)
                    .product_affine(&v(4))
                    .unwrap()
                    .add(&v(1).scale(&e).unwrap())
                    .unwrap();
                let at = value(3)
                    .mul(&value(4))
                    .unwrap()
                    .add(&e.mul(&value(1)).unwrap())
                    .unwrap();
                (form, at)
            }
            2 => {
                let (g, h) = (public(), public());
                let form = v(3)
                    .scale(&g)
                    .unwrap()
                    .add(&v(0).scale(&h).unwrap())
                    .unwrap();
                let at = g
                    .mul(&value(3))
                    .unwrap()
                    .add(&h.mul(&value(0)).unwrap())
                    .unwrap();
                (form, at)
            }
            _ => {
                let (jj, k) = (public(), public());
                let form = v(4)
                    .scale(&jj)
                    .unwrap()
                    .add(&v(2).scale(&k).unwrap())
                    .unwrap();
                let at = jj
                    .mul(&value(4))
                    .unwrap()
                    .add(&k.mul(&value(2)).unwrap())
                    .unwrap();
                (form, at)
            }
        };
        let constant = if j == 3 {
            // Only the constant coefficient must cancel; the others are free.
            let mut l = public();
            l.set_coefficient(0, -at.coefficient_i128(0).unwrap())
                .unwrap();
            l
        } else {
            at.neg()
        };
        forms.push(form.add(&st.constant_in(constant).unwrap()).unwrap());
    }
    for (j, form) in forms.iter().enumerate() {
        if j == 3 {
            st.const_coeff_zero(form.clone()).unwrap();
        } else {
            st.eq_mod_p(form.clone()).unwrap();
        }
    }
    let witness = Witness::from([
        (
            "s".into(),
            vec![poly(ring, w[0].clone()), poly(ring, w[1].clone())],
        ),
        ("x".into(), vec![poly(ring, w[2].clone())]),
        ("y".into(), vec![poly(ring, w[3].clone())]),
        ("z".into(), vec![poly(ring, w[4].clone())]),
    ]);
    (st, forms, witness, w)
}

#[test]
fn linf_exact_norm_and_binary_variables_with_carries_prove_at_both_degrees() {
    const CTX: &[u8] = b"jali-test/linf-mixed";
    let r13 = Ring::new(13, 128).unwrap();
    let (st, forms, witness, w) = mixed(&r13, [&r13; 4], SEED, |_| {});
    let honest: Vec<u128> = forms
        .iter()
        .map(|form| f_bound_linf(form, &mixed_bounds(None)))
        .collect();
    for degree in [64, 128] {
        let k = 128 / degree;
        let req = st.requirements(degree, U256::from_u64(Q)).unwrap();
        // Bounded: s, x, y, z. Messages: the carries of constraints 0 and 1 and of the clause.
        // Range rows: those, the quotients of constraint 2, and the polynomials of y and z.
        assert_eq!(
            (req.m1, req.l, req.n_bin, req.n_prime),
            (5 * k, 2 * k + 1, k, 5 * k + 1),
            "degree {degree}"
        );
        assert_eq!(
            (req.l2_rows.clone(), req.l2_bounds_squared.clone()),
            (vec![2 * k], vec![64])
        );
        // ||s||^2 <= 64, 128 binary coefficients, and 128 coefficients of y and z at 1 and 3.
        assert_eq!(req.alpha_squared, 64 + 128 + 128 + 128 * 9);
        // The honest integer bounds, with beta, and the quotient bound, at least both betas.
        assert_eq!(
            narrow(&req.max_integer_coefficient),
            *honest.iter().max().unwrap()
        );
        let quotient = honest.iter().map(|f| f.div_ceil(13)).max().unwrap();
        assert_eq!(req.linf_bound, quotient.max(3));
        assert!(req.lifted_moduli.is_empty());
        // Every constraint reads y or z, all at the statement modulus: no range limit.
        let linf = req.linf.as_ref().unwrap();
        assert_eq!(linf.extraction_limit, None);
        assert_eq!(linf.lifting.len(), 4);
        for (entry, form) in linf.lifting.iter().zip(&forms) {
            assert_eq!(narrow(&entry.modulus), 13);
            // The exported polynomial in E is the bound at E, for any E.
            for e in [1, 3, 1 << 20, 123_456_789] {
                assert_eq!(linf_f(entry, e), f_bound_linf(form, &mixed_bounds(Some(e))));
            }
        }
        // Only constraint 1 has an E^2 term, from y z: d * ||1||_1 = 128.
        let quadratic: Vec<u128> = linf.lifting.iter().map(|e| narrow(&e.quadratic)).collect();
        assert_eq!(quadratic, vec![0, 128, 0, 0]);
        let params = fitted(&req, 13, degree, "linf-mixed-test-only");
        let checked = params.check().unwrap();
        let e = checked.approx_extraction_bound;
        assert_eq!(e, extraction_bound(params.log_sigma[3]));
        let compiled = st.compile(params.clone()).unwrap();
        // The lifting boundary, with F_j(E): a prime at or below the threshold is refused.
        let threshold = lifting_threshold(&req, 13, params.log_sigma[3]);
        assert!(narrow(&params.prime_factors[0]) > threshold);
        let (low, below) = fit_downward(&req, "linf-below-test-only", degree, threshold as u64);
        assert!(u128::from(low) <= threshold);
        assert_eq!(below.log_sigma[3], params.log_sigma[3]);
        // That prime is far above the threshold the declared bounds would give.
        let declared = Requirements {
            linf: None,
            ..req.clone()
        };
        assert!(u128::from(low) > lifting_threshold(&declared, 13, params.log_sigma[3]));
        assert_eq!(
            st.compile(below).err(),
            Some(Error::Parameter("modulus lifting bound"))
        );
        // What extraction guarantees, per variable.
        assert_eq!(
            compiled.extraction_bound("y"),
            Ok(Extraction::Linf(2 * checked.z4_bound))
        );
        assert_eq!(compiled.extraction_bound("z"), Ok(Extraction::Linf(e)));
        assert_eq!(
            compiled.extraction_bound("s"),
            Ok(Extraction::L2Squared(64))
        );
        assert_eq!(compiled.extraction_bound("x"), Ok(Extraction::Binary));
        assert_eq!(compiled.extraction_bound("w").err(), Some(Error::Index));
        // Carries and quotients are the integer quotients by 13, computed here; the range rows
        // are the carries, the quotients, then y and z.
        let (s1, m) = compiled.map_witness(&witness).unwrap();
        let q = narrow(&params.prime_factors[0]) as i128;
        let proof_ring = Ring::new(q, degree).unwrap();
        let lifted = Ring::new(q, 128).unwrap();
        let split = |c: Vec<i128>| iso::split(&poly(&lifted, c), proof_ring.clone()).unwrap();
        let c: Vec<Vec<i128>> = forms[..3]
            .iter()
            .map(|form| quotients(&integer_value(form, &w), 13))
            .collect();
        let clause = quotients(&integer_value(&forms[3], &w)[..1], 13)[0];
        for c in &c {
            assert!(c.iter().any(|x| *x != 0));
        }
        let carries = [
            split(c[0].clone()),
            split(c[1].clone()),
            vec![Poly::constant(proof_ring.clone(), clause)],
        ]
        .concat();
        assert_eq!(m.entries(), carries.as_slice());
        let arp = compiled
            .statement()
            .arp
            .as_ref()
            .unwrap()
            .evaluate(&s1, &m)
            .unwrap();
        let rows = [
            split(c[0].clone()),
            split(c[1].clone()),
            split(c[2].clone()),
            vec![Poly::constant(proof_ring.clone(), clause)],
            split(w[3].clone()),
            split(w[4].clone()),
        ]
        .concat();
        assert_eq!(arp.entries(), rows.as_slice());
        let proof = compiled
            .prove_with_seed([1; 32], &witness, CTX, [2; 32])
            .unwrap();
        compiled.verify([1; 32], &proof, CTX).unwrap();
        assert!(compiled.verify([1; 32], &proof, b"other").is_err());
        let bytes = compiled
            .prove_bytes_with_seed([1; 32], &witness, CTX, [3; 32])
            .unwrap();
        compiled.verify_bytes([1; 32], &bytes, CTX).unwrap();
        // Parameters whose common bound is below a beta are refused.
        let mut small = params.clone();
        small.linf_bound = 2;
        small.check().unwrap();
        assert_eq!(
            st.compile(small).err(),
            Some(Error::Parameter("approximate range bound"))
        );
    }
}

#[test]
fn the_witness_map_refuses_beta_plus_one_while_a_proof_only_bounds_e() {
    const CTX: &[u8] = b"jali-test/linf-relaxed";
    // The statements of `mixed`, each built for a witness with one coefficient changed, so that
    // every constraint holds: only the norm of y or z can refuse it. Their constants differ, and
    // with them F, so each gets its own parameters.
    let r13 = Ring::new(13, 128).unwrap();
    for (index, value, admitted) in [
        (3, 1, true),
        (3, -1, true),
        (3, 2, false),
        (3, -2, false),
        (4, 3, true),
        (4, -3, true),
        (4, 4, false),
        (4, -4, false),
    ] {
        let (st, _, witness, _) = mixed(&r13, [&r13; 4], SEED, |w| w[index][5] = value);
        let req = st.requirements(64, U256::from_u64(Q)).unwrap();
        let params = fitted(&req, 13, 64, "linf-mixed-test-only");
        let result = st.compile(params).unwrap().map_witness(&witness);
        assert_eq!(result.is_ok(), admitted, "variable {index}, value {value}");
        if !admitted {
            assert_eq!(result.err(), Some(Error::Witness));
        }
    }
    // Over Z_13 with s (four polynomials) and u (ell_inf <= 1), the clause ct(u - 5) = 0 has no
    // witness within beta = 1, as 5 is not -1, 0 or 1 modulo 13. The integer 5, far below E,
    // satisfies it with carry 0. With a common bound of 8, as the carries of another
    // constraint could need, the range proof's own witness check admits 5: a prover that skips
    // the witness map proves it, and the compiled statement's verifier accepts. A proof shows
    // |u_0| <= E, not |u_0| <= 1.
    let mut st = Statement::new(r13.clone());
    st.var("s", 4, Norm::L2Squared(64)).unwrap();
    st.var("u", 1, Norm::Linf(1)).unwrap();
    let clause = st
        .variable("u", 0)
        .unwrap()
        .add(&st.constant(Poly::constant(r13.clone(), -5)).unwrap())
        .unwrap();
    st.const_coeff_zero(clause).unwrap();
    for v in -1..=1i128 {
        assert_ne!((v - 5).rem_euclid(13), 0);
    }
    let mut req = st.requirements(64, U256::from_u64(Q)).unwrap();
    assert_eq!(req.linf_bound, 1);
    req.linf_bound = 8;
    let params = fitted(&req, 13, 64, "linf-relaxed-test-only");
    let compiled = st.compile(params.clone()).unwrap();
    let e = params.check().unwrap().approx_extraction_bound;
    assert_eq!(compiled.extraction_bound("u"), Ok(Extraction::Linf(e)));
    assert!(e >= 5);
    let mut s = vec![Poly::zero(r13.clone()); 4];
    s[0] = Poly::constant(r13.clone(), 1);
    let named = Witness::from([
        ("s".into(), s),
        ("u".into(), vec![Poly::constant(r13.clone(), 5)]),
    ]);
    assert_eq!(compiled.map_witness(&named).err(), Some(Error::Witness));
    // The proof-ring witness the map would give: s and u split into their two components, and
    // the clause's carry (5 - 5) / 13 = 0.
    let q = narrow(&params.prime_factors[0]) as i128;
    let proof_ring = Ring::new(q, 64).unwrap();
    let lifted = Ring::new(q, 128).unwrap();
    let mut bounded = vec![Poly::zero(lifted.clone()); 5];
    bounded[0] = Poly::constant(lifted.clone(), 1);
    bounded[4] = Poly::constant(lifted.clone(), 5);
    let s1 = PolyVec::new(
        proof_ring.clone(),
        bounded
            .iter()
            .flat_map(|p| iso::split(p, proof_ring.clone()).unwrap())
            .collect(),
    )
    .unwrap();
    let m = PolyVec::new(proof_ring.clone(), vec![Poly::zero(proof_ring.clone())]).unwrap();
    let params = Arc::new(params);
    let mut prover = Prover::new([4; 32], params.clone()).unwrap();
    prover.set_statement(compiled.statement().clone()).unwrap();
    prover.set_witness(&s1, &m).unwrap();
    let proof = prover.prove_with_seed(CTX, [5; 32]).unwrap();
    compiled.verify([4; 32], &proof, CTX).unwrap();
    let mut verifier = Verifier::new([4; 32], params).unwrap();
    verifier
        .set_statement(compiled.statement().clone())
        .unwrap();
    verifier.verify(&proof, CTX).unwrap();
}

/// Over a statement ring modulo `p` (degree 128): `s` (four polynomials, $`\|s\|^2\le64`$) and
/// `y` ($`\ell_\infty\le1`$), with the clause $`\mathrm{ct}(a\,y)=0`$ modulo `p_j` for the
/// all-ones $`a`$, so that $`F=128\beta`$ on a witness and $`128E`$ at extraction. The witness
/// $`y=0`$ satisfies it for every modulus.
fn ones(p: U256, p_j: U256) -> Statement {
    let mut st = Statement::new(Ring::with_modulus(p, 128).unwrap());
    st.var("s", 4, Norm::L2Squared(64)).unwrap();
    st.var("y", 1, Norm::Linf(1)).unwrap();
    let ring = Ring::with_modulus(p_j, 128).unwrap();
    let clause = st
        .variable_in(&ring, "y", 0)
        .unwrap()
        .scale(&poly(&ring, vec![1; 128]))
        .unwrap();
    st.const_coeff_zero(clause).unwrap();
    st
}

/// $`2^{61}-1`$, a prime statement modulus that holds every extracted value below.
fn mersenne61() -> U256 {
    U256::from_u64((1 << 61) - 1)
}

#[test]
fn the_lifting_check_uses_the_extraction_bound() {
    // For the clause of `ones`, F = 128 at the declared bound and 128 E at extraction. With the
    // parameters fixed, the largest clause modulus the compiler admits is set by 128 E: a
    // modulus between that and the one that 128 would allow is refused, although the honest
    // bound would admit it.
    let probe = ones(mersenne61(), U256::from_u64(1 << 20));
    let req = probe.requirements(64, U256::from_u64(Q)).unwrap();
    // One clause row and the two of y; the quotient bound is 1 for every modulus above 128.
    assert_eq!((req.n_prime, req.linf_bound), (3, 1));
    let entry = &req.linf.as_ref().unwrap().lifting[0];
    assert_eq!(
        [&entry.exact, &entry.linear, &entry.quadratic].map(narrow),
        [0, 128, 0]
    );
    let params = fitted(&req, 0, 64, "linf-lifting-test-only");
    let checked = params.check().unwrap();
    let (q, e) = (narrow(&checked.q), checked.approx_extraction_bound);
    let psi =
        (28.0 * 1.55 * 2f64.powi(params.log_sigma[3] as i32) / params.linf_bound as f64).ceil();
    let slack = u128::from(params.linf_bound) * psi as u128;
    // The largest p_j with 2 (F + p_j slack) < q, for F at the declared bound and at E.
    let largest = |f: u128| (q - 1 - 2 * f) / (2 * slack);
    let (honest, extracted) = (largest(128), largest(128 * e));
    assert!(extracted + 1 < honest && extracted > 128);
    let compile = |p_j: u128| {
        let st = ones(mersenne61(), U256::from_u128(p_j));
        assert_eq!(
            st.requirements(64, checked.q).unwrap().linf_bound,
            1,
            "{p_j}"
        );
        st.compile(params.clone())
    };
    compile(extracted).unwrap();
    for p_j in [extracted + 1, (extracted + honest) / 2, honest] {
        assert_eq!(
            compile(p_j).err(),
            Some(Error::Parameter("modulus lifting bound")),
            "{p_j}"
        );
    }
    // At the boundary the lifted relation cannot wrap around at extraction.
    assert!(128 * e + extracted * e < q);
}

#[test]
fn the_range_condition_bounds_the_extraction_bound() {
    // `ones` with a clause modulo 13: over a statement modulus p that 13 does not divide, an
    // extracted y is bounded by E only, which p must hold: E <= (p - 1) / 2.
    let thirteen = U256::from_u8(13);
    let req = ones(mersenne61(), thirteen)
        .requirements(64, U256::from_u64(Q))
        .unwrap();
    let params = fitted(&req, 0, 64, "linf-range-test-only");
    let e = params.check().unwrap().approx_extraction_bound;
    let limit = |p: u128| {
        ones(U256::from_u128(p), thirteen)
            .requirements(64, U256::from_u64(Q))
            .unwrap()
            .linf
            .unwrap()
            .extraction_limit
    };
    for (p, admitted) in [
        (2 * e + 1, true),
        (2 * e + 3, true),
        (2 * e, false),
        (e, false),
    ] {
        assert_ne!(p % 13, 0);
        assert_eq!(limit(p), Some(U256::from_u128((p - 1) / 2)));
        let result = ones(U256::from_u128(p), thirteen).compile(params.clone());
        if admitted {
            result.unwrap();
        } else {
            assert_eq!(result.err(), OUT_OF_RANGE, "p {p}");
        }
    }
    // A clause whose modulus divides p reads no value that the reduction modulo p changes: no
    // limit, even for p far below 2E.
    let p = 13 * 11;
    assert!(p < 2 * e);
    assert_eq!(limit(p), None);
    ones(U256::from_u128(p), thirteen)
        .compile(params.clone())
        .unwrap();
    // A statement without ell_inf variables has no `linf`.
    let mut plain = Statement::new(Ring::new(1009, 128).unwrap());
    plain.var("s", 4, Norm::L2Squared(64)).unwrap();
    assert_eq!(
        plain.requirements(64, U256::from_u64(Q)).unwrap().linf,
        None
    );
}

/// Over `ring` (degree 128): `s` (four polynomials, $`\|s\|^2\le64`$) and `y`
/// ($`\ell_\infty\le1`$), with the clause $`\mathrm{ct}(y_0)=7`$ over $`\mathbb Z_q`$, which is
/// native when `q` is the proof modulus.
fn native_clause(ring: Arc<Ring>, q: i128) -> Statement {
    let mut st = Statement::new(ring);
    st.var("s", 4, Norm::L2Squared(64)).unwrap();
    st.var("y", 1, Norm::Linf(1)).unwrap();
    let rq = Ring::new(q, 128).unwrap();
    let form = st
        .variable_in(&rq, "y", 0)
        .unwrap()
        .add(&st.constant_in(Poly::constant(rq, -7)).unwrap())
        .unwrap();
    st.const_coeff_zero(form).unwrap();
    st
}

/// Over `ring`: `s` and `y` as in `native_clause` and a binary `b`, with $`b=1`$ over the
/// statement ring and the clauses $`\mathrm{ct}(y\,b)=7`$ modulo 12 and 13, so that `y` meets
/// the modulus 12 only in a product.
fn product_clauses(ring: Arc<Ring>) -> Statement {
    let mut st = Statement::new(ring.clone());
    st.var("s", 4, Norm::L2Squared(64)).unwrap();
    st.var("y", 1, Norm::Linf(1)).unwrap();
    st.var("b", 1, Norm::Binary).unwrap();
    let one = st
        .variable("b", 0)
        .unwrap()
        .add(&st.constant(Poly::constant(ring, -1)).unwrap())
        .unwrap();
    st.eq_mod_p(one).unwrap();
    for m in [12, 13] {
        let r = Ring::new(m, 128).unwrap();
        let form = st
            .variable_in(&r, "y", 0)
            .unwrap()
            .product_affine(&st.variable_in(&r, "b", 0).unwrap())
            .unwrap()
            .add(&st.constant_in(Poly::constant(r, -7)).unwrap())
            .unwrap();
        st.const_coeff_zero(form).unwrap();
    }
    st
}

#[test]
fn the_range_limit_covers_native_constraints_and_products() {
    // No other test fails when the limit is dropped for native constraints, for variables in
    // products, or when nothing is lifted. Native: over Z_13 an
    // extracted y_0 = 7, below E, satisfies ct(y_0) = 7 modulo q, which no value in [-6, 6]
    // does. The limit is 6, and E exceeds it; over Z_(2^61-1) there is room.
    let r13 = Ring::new(13, 128).unwrap();
    // 2^60 + 33: the clause over it is native when the requirements take it as proof modulus.
    let probe = 1152921504606847009;
    let req = native_clause(r13.clone(), probe)
        .requirements(64, U256::from_u128(probe as u128))
        .unwrap();
    assert!(req.lifted_moduli.is_empty() && req.linf.as_ref().unwrap().lifting.is_empty());
    assert_eq!(
        req.linf.as_ref().unwrap().extraction_limit,
        Some(U256::from_u8(6))
    );
    let (q, params) = fit_upward(&req, "range-limit-native-test-only", 64, Q);
    assert!(params.check().unwrap().approx_extraction_bound > 6);
    let st = native_clause(r13, q as i128);
    assert_eq!(st.compile(params.clone()).err(), OUT_OF_RANGE);
    native_clause(Ring::with_modulus(mersenne61(), 128).unwrap(), q as i128)
        .compile(params)
        .unwrap();
    // Products: the clause modulo 12 reads y only in y b and still sets the limit over Z_13;
    // over Z_156, which 12 and 13 divide, there is none.
    let st = product_clauses(Ring::new(13, 128).unwrap());
    let req = st.requirements(64, U256::from_u64(Q)).unwrap();
    assert_eq!(
        req.linf.as_ref().unwrap().extraction_limit,
        Some(U256::from_u8(6))
    );
    let params = fitted(&req, 13, 64, "range-limit-product-test-only");
    assert_eq!(st.compile(params.clone()).err(), OUT_OF_RANGE);
    let over_156 = product_clauses(Ring::new(156, 128).unwrap());
    let req = over_156.requirements(64, U256::from_u64(Q)).unwrap();
    assert_eq!(req.linf.as_ref().unwrap().extraction_limit, None);
    let params = fitted(&req, 156, 64, "range-limit-product-156-test-only");
    assert!(params.check().unwrap().approx_extraction_bound > 77);
    over_156.compile(params).unwrap();
}

#[test]
fn the_extracted_bound_must_be_below_half_the_proof_modulus() {
    // Only y's two range rows, no constraint: nothing is lifted. Widening sigma_4 until
    // 2E + 1 reaches q makes E say nothing modulo q, and the compiler refuses.
    let mut st = Statement::new(Ring::new(13, 128).unwrap());
    st.var("s", 4, Norm::L2Squared(64)).unwrap();
    st.var("y", 1, Norm::Linf(1)).unwrap();
    let req = st.requirements(64, U256::from_u64(Q)).unwrap();
    assert!(req.linf.as_ref().unwrap().lifting.is_empty());
    let mut params = fit_upward(&req, "linf-extraction-test-only", 64, Q).1;
    let q = narrow(&params.check().unwrap().q);
    let t = (0..=100u32)
        .rev()
        .find(|t| 2 * extraction_bound(*t) + 1 < q)
        .unwrap();
    params.log_sigma[3] = t;
    params.check().unwrap();
    st.compile(params.clone()).unwrap();
    params.log_sigma[3] = t + 1;
    params.check().unwrap();
    assert_eq!(
        st.compile(params).err(),
        Some(Error::Parameter("approximate extraction bound"))
    );
}

#[test]
fn constraints_with_their_own_moduli_bound_linf_variables_by_e() {
    const CTX: &[u8] = b"jali-test/linf-moduli";
    // `mixed` with constraints 0 and 2 modulo 13 and 1 and 3 modulo 12. Over Z_156, which both
    // moduli divide, the range condition does not apply; over Z_1009 it limits E to 504.
    let (r12, r13) = (Ring::new(12, 128).unwrap(), Ring::new(13, 128).unwrap());
    let rings = [&r13, &r12, &r13, &r12];
    let r156 = Ring::new(156, 128).unwrap();
    let (st, forms, witness, w) = mixed(&r156, rings, SEED + 1, |_| {});
    let req = st.requirements(64, U256::from_u64(Q)).unwrap();
    let linf = req.linf.as_ref().unwrap();
    assert_eq!(linf.extraction_limit, None);
    let moduli: Vec<u128> = linf.lifting.iter().map(|e| narrow(&e.modulus)).collect();
    assert_eq!(moduli, vec![13, 12, 13, 12]);
    let honest: Vec<u128> = forms
        .iter()
        .map(|form| f_bound_linf(form, &mixed_bounds(None)))
        .collect();
    let lifted: Vec<(u128, u128)> = req
        .lifted_moduli
        .iter()
        .map(|m| (narrow(&m.modulus), narrow(&m.max_integer_coefficient)))
        .collect();
    assert_eq!(
        lifted,
        vec![
            (12, honest[1].max(honest[3])),
            (13, honest[0].max(honest[2]))
        ]
    );
    let params = fitted(&req, 156, 64, "linf-moduli-test-only");
    let compiled = st.compile(params.clone()).unwrap();
    let (s1, m) = compiled.map_witness(&witness).unwrap();
    // Each carry and quotient divides the integer value by the constraint's own modulus.
    let q = narrow(&params.prime_factors[0]) as i128;
    let proof_ring = Ring::new(q, 64).unwrap();
    let lifted = Ring::new(q, 128).unwrap();
    let split = |c: Vec<i128>| iso::split(&poly(&lifted, c), proof_ring.clone()).unwrap();
    let c0 = quotients(&integer_value(&forms[0], &w), 13);
    let c1 = quotients(&integer_value(&forms[1], &w), 12);
    let c2 = quotients(&integer_value(&forms[2], &w), 13);
    let c3 = quotients(&integer_value(&forms[3], &w)[..1], 12)[0];
    let clause = vec![Poly::constant(proof_ring.clone(), c3)];
    let carries = [split(c0), split(c1), clause.clone()].concat();
    assert_eq!(m.entries(), carries.as_slice());
    let arp = compiled
        .statement()
        .arp
        .as_ref()
        .unwrap()
        .evaluate(&s1, &m)
        .unwrap();
    let rows = [
        carries[..4].to_vec(),
        split(c2),
        clause,
        split(w[3].clone()),
        split(w[4].clone()),
    ]
    .concat();
    assert_eq!(arp.entries(), rows.as_slice());
    let proof = compiled
        .prove_with_seed([6; 32], &witness, CTX, [7; 32])
        .unwrap();
    compiled.verify([6; 32], &proof, CTX).unwrap();
    assert!(compiled.verify([6; 32], &proof, b"other").is_err());
    // Over Z_1009 the statement has the same requirements apart from the limit, and E, far
    // above 504, fails it.
    let (over, _, _, _) = mixed(&Ring::new(1009, 128).unwrap(), rings, SEED + 1, |_| {});
    let other = over.requirements(64, U256::from_u64(Q)).unwrap();
    assert_eq!(
        other.linf.as_ref().unwrap().extraction_limit,
        Some(U256::from_u16(504))
    );
    let mut same = other.clone();
    same.linf.as_mut().unwrap().extraction_limit = None;
    assert_eq!(same, req);
    assert!(params.check().unwrap().approx_extraction_bound > 504);
    assert_eq!(over.compile(params).err(), OUT_OF_RANGE);
}

#[test]
fn declarations_and_the_linear_front_end() {
    const CTX: &[u8] = b"jali-test/linf-lin";
    let ring = Ring::new(13, 128).unwrap();
    let mut st = Statement::new(ring.clone());
    assert_eq!(
        st.var("y", 1, Norm::Linf(0)).err(),
        Some(Error::Parameter("variable declaration"))
    );
    st.var("y", 1, Norm::Linf(1)).unwrap();
    // With nothing lifted, the requirements hold only the variables: `linf_bound` is the
    // largest beta, `alpha_squared` sums 128 beta^2 per polynomial.
    let mut free = Statement::new(ring.clone());
    free.var("y", 1, Norm::Linf(2)).unwrap();
    free.var("z", 2, Norm::Linf(7)).unwrap();
    let req = free.requirements(64, U256::from_u64(Q)).unwrap();
    assert_eq!(
        (req.m1, req.n_prime, req.linf_bound, req.alpha_squared),
        (6, 6, 7, 128 * 4 + 2 * 128 * 49)
    );
    assert_eq!(req.linf.unwrap().lifting, Vec::new());
    // alpha_squared = 128 beta^2 needs more than 128 bits.
    let mut wide = Statement::new(ring.clone());
    wide.var("y", 1, Norm::Linf(u64::MAX)).unwrap();
    assert_eq!(
        wide.requirements(64, U256::from_u64(Q)).err(),
        Some(Error::Overflow)
    );
    // lin::compile: 3 y_0 + s_0 - t = 0 modulo 13, with y bounded by 2 in ell_inf.
    let mut entries = vec![Poly::zero(ring.clone()); 5];
    entries[0] = Poly::constant(ring.clone(), 3);
    entries[1] = Poly::constant(ring.clone(), 1);
    let a = PolyMat::new(ring.clone(), 1, 5, entries).unwrap();
    let t = PolyVec::new(ring.clone(), vec![Poly::constant(ring.clone(), -5)]).unwrap();
    let blocks = [
        lin::Block {
            name: "y".into(),
            length: 1,
            norm: Norm::Linf(2),
        },
        lin::Block {
            name: "s".into(),
            length: 4,
            norm: Norm::L2Squared(64),
        },
    ];
    // The same statement through the named builder, for its requirements.
    let mut probe = Statement::new(ring.clone());
    probe.var("y", 1, Norm::Linf(2)).unwrap();
    probe.var("s", 4, Norm::L2Squared(64)).unwrap();
    let three = Poly::constant(ring.clone(), 3);
    let row = probe
        .constant(t.entries()[0].clone())
        .unwrap()
        .add(&probe.variable("y", 0).unwrap().scale(&three).unwrap())
        .unwrap()
        .add(&probe.variable("s", 0).unwrap())
        .unwrap();
    probe.eq_mod_p(row).unwrap();
    let req = probe.requirements(64, U256::from_u64(Q)).unwrap();
    assert_eq!(req.n_prime, 4);
    let params = fitted(&req, 13, 64, "linf-lin-test-only");
    let compiled = lin::compile(&a, &t, &blocks, params).unwrap();
    // y_0 = 2, s_0 = -1: 3 * 2 - 1 - 5 = 0.
    let mut s = vec![Poly::zero(ring.clone()); 4];
    s[0] = Poly::constant(ring.clone(), -1);
    let witness = Witness::from([
        ("y".into(), vec![Poly::constant(ring.clone(), 2)]),
        ("s".into(), s),
    ]);
    let proof = compiled
        .prove_with_seed([8; 32], &witness, CTX, [9; 32])
        .unwrap();
    compiled.verify([8; 32], &proof, CTX).unwrap();
    // y_0 = 3, s_0 = -4 satisfies the row too, but 3 is above beta = 2.
    let mut bad = witness.clone();
    bad.get_mut("y").unwrap()[0] = Poly::constant(ring.clone(), 3);
    bad.get_mut("s").unwrap()[0] = Poly::constant(ring, -4);
    assert_eq!(compiled.map_witness(&bad).err(), Some(Error::Witness));
}

/// The serde forms of `Requirements::linf`: JSON writes it only when present, with the number
/// rule; postcard always writes it.
#[cfg(feature = "serde")]
mod serde_forms {
    use super::*;
    use jali::statement::LinfLifting;
    use serde_json::json;

    #[test]
    fn requirements_with_linf_round_trip() {
        let r13 = Ring::new(13, 128).unwrap();
        let (st, _, _, _) = mixed(&r13, [&r13; 4], SEED, |_| {});
        let req = st.requirements(64, U256::from_u64(Q)).unwrap();
        let value = serde_json::to_value(&req).unwrap();
        let linf = req.linf.as_ref().unwrap();
        let entries: Vec<_> = linf
            .lifting
            .iter()
            .map(|e| {
                json!({
                    "modulus": 13,
                    "exact": narrow(&e.exact) as u64,
                    "linear": narrow(&e.linear) as u64,
                    "quadratic": narrow(&e.quadratic) as u64,
                })
            })
            .collect();
        assert_eq!(
            value["linf"],
            json!({"lifting": entries, "extraction_limit": null})
        );
        assert!(value.get("lifted_moduli").is_none());
        let text = serde_json::to_string(&req).unwrap();
        assert!(text.ends_with(r#""extraction_limit":null}}"#));
        let read: Requirements = serde_json::from_str(&text).unwrap();
        assert_eq!(read, req);
        assert_eq!(serde_json::to_string(&read).unwrap(), text);
        // Values of 2^64 and more are decimal strings; a limit is a number or a string.
        let mut wide = req.clone();
        wide.linf = Some(jali::statement::LinfRequirements {
            lifting: vec![LinfLifting {
                modulus: U256::from_u8(12),
                exact: U256::from_u64(u64::MAX),
                linear: U256::from_u128(1 << 64),
                quadratic: common::params::power_minus(256, 435),
            }],
            extraction_limit: Some(U256::from_u128(1 << 70)),
        });
        let value = serde_json::to_value(&wide).unwrap();
        assert_eq!(
            value["linf"],
            json!({
                "lifting": [{
                    "modulus": 12,
                    "exact": u64::MAX,
                    "linear": "18446744073709551616",
                    "quadratic": concat!(
                        "11579208923731619542357098500868790785326998466564",
                        "0564039457584007913129639501"
                    ),
                }],
                "extraction_limit": "1180591620717411303424",
            })
        );
        assert_eq!(serde_json::from_value::<Requirements>(value).unwrap(), wide);
        let mut small = wide.clone();
        small.linf.as_mut().unwrap().extraction_limit = Some(U256::from_u8(6));
        let text = serde_json::to_string(&small).unwrap();
        assert!(text.ends_with(r#""extraction_limit":6}}"#));
        assert_eq!(serde_json::from_str::<Requirements>(&text).unwrap(), small);
        // Postcard: the tag of the present linf, the count, each entry as four 32-byte values,
        // then the limit's tag and its 32 bytes.
        let bytes = postcard::to_allocvec(&wide).unwrap();
        assert_eq!(postcard::from_bytes::<Requirements>(&bytes).unwrap(), wide);
        let tail = &bytes[bytes.len() - (1 + 1 + 128 + 1 + 32)..];
        assert_eq!(tail[..2], [1, 1]);
        let entry = &wide.linf.as_ref().unwrap().lifting[0];
        for (i, v) in [
            &entry.modulus,
            &entry.exact,
            &entry.linear,
            &entry.quadratic,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(tail[2 + 32 * i..34 + 32 * i], v.to_le_bytes()[..]);
        }
        assert_eq!(tail[130], 1);
        assert_eq!(tail[131..], U256::from_u128(1 << 70).to_le_bytes()[..]);
        assert!(postcard::from_bytes::<Requirements>(&bytes[..bytes.len() - 1]).is_err());
    }
}
