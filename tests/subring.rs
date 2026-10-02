//! Subring variables ([`Statement::var_subring`]): a block of elements $`v(X^K)`$ of the subring
//! of degree $`d_v`$, $`K=d'/d_v`$, commits only the $`d_v/d`$ components per element that can be
//! nonzero. The tests check the declarations and the refusal of a proof degree above $`d_v`$
//! (by `requirements`, `blocks` and `compile`), the rows, shares and integer bounds that
//! `requirements` and `blocks` export (against the oracle of `common::ring`, with $`d_v`$
//! coefficients for binary and $`\ell_\infty`$ blocks), `iso::embed` and `coefficient_in`, and a
//! statement with subring variables in a lifted quadratic `eq_mod_p`, in constant-coefficient
//! clauses and in the approximate range proof, which proves at both proof degrees; the witness
//! map refuses a coefficient outside the subring. Parameters come from the test port of the
//! parameter tool (`common::params`).
use jali::{
    Error,
    math::{Poly, PolyVec, Ring, U256, iso},
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
const CTX: &[u8] = b"jali-test/subring";
type Witness = BTreeMap<String, Vec<Poly>>;

/// The port's parameters at the first prime above the lifting threshold, $`E`$ included.
fn fitted(req: &Requirements, degree: usize, id: &str) -> TboxParams {
    let threshold = lifting_threshold(req, 156, range_width(req, degree));
    fit_upward(req, id, degree, (threshold as u64 + 1).max(Q)).1
}

#[test]
fn declarations_and_proof_degrees() {
    let ring = Ring::new(13, 256).unwrap();
    let refused = Err(Error::Parameter("variable declaration"));
    for (degree, norm) in [
        (32, Norm::Binary),
        (96, Norm::Binary),
        (512, Norm::Binary),
        (0, Norm::Binary),
        (128, Norm::Linf(0)),
        (128, Norm::LinfExact(0)),
        (128, Norm::LinfExact(1 << 63)),
    ] {
        let mut st = Statement::new(ring.clone());
        assert_eq!(
            st.var_subring("v", 1, degree, norm, Placement::Ajtai),
            refused,
            "{degree} {norm:?}"
        );
    }
    let mut st = Statement::new(ring.clone());
    assert_eq!(
        st.var_subring("u", 1, 128, Norm::Unbounded, Placement::Ajtai),
        refused
    );
    assert_eq!(
        st.var_subring("v", 0, 128, Norm::Binary, Placement::Ajtai),
        refused
    );
    assert_eq!(
        st.var_subring("v", 2, 64, Norm::Binary, Placement::Ajtai),
        Ok(0)
    );
    assert_eq!(
        st.var_subring("w", 1, 256, Norm::L2Squared(9), Placement::Bdlop),
        Ok(2)
    );
    assert_eq!(
        st.var_subring("v", 1, 128, Norm::Binary, Placement::Ajtai),
        refused
    );
    // Proof degree 128 does not divide the subring degree 64.
    let target = Some(Error::Parameter("target ring"));
    assert_eq!(st.requirements(128, U256::from_u64(Q)).err(), target);
    assert_eq!(st.blocks(128).err(), target);
    // Nor does compile take it, from a set of degree 128 that passes its check.
    assert_eq!(st.compile(jali::params::kyber1024_d128()).err(), target);
    let blocks = st.blocks(64).unwrap();
    assert_eq!(blocks.iter().map(|b| b.rows).collect::<Vec<_>>(), [2, 4]);
    // A full-degree subring block is `var_placed`.
    let mut a = Statement::new(ring.clone());
    a.var_subring("w", 3, 256, Norm::Linf(4), Placement::Bdlop)
        .unwrap();
    let mut b = Statement::new(ring);
    b.var_placed("w", 3, Norm::Linf(4), Placement::Bdlop)
        .unwrap();
    for d in [64, 128] {
        assert_eq!(
            a.requirements(d, U256::from_u64(Q)),
            b.requirements(d, U256::from_u64(Q))
        );
        assert_eq!(a.blocks(d), b.blocks(d));
    }
}

#[test]
fn embed_and_coefficient_in() {
    let big = Ring::new(13, 1024).unwrap();
    let mut rng = SplitMix64(0x2567_2200);
    for degree in [64, 128, 256, 1024] {
        let small = Ring::new(13, degree).unwrap();
        let k = 1024 / degree;
        let (a, b) = (
            poly(&small, rng.uniform(degree, 13)),
            poly(&small, rng.uniform(degree, 13)),
        );
        let e = iso::embed(&a, big.clone()).unwrap();
        for (i, x) in e.coefficients().iter().enumerate() {
            let want = if i % k == 0 {
                a.coefficients()[i / k]
            } else {
                U256::ZERO
            };
            assert_eq!(*x, want, "degree {degree}, coefficient {i}");
        }
        // A ring homomorphism.
        let embed = |p: &Poly| iso::embed(p, big.clone()).unwrap();
        assert_eq!(
            embed(&a.mul(&b).unwrap()),
            embed(&a).mul(&embed(&b)).unwrap()
        );
        assert_eq!(
            embed(&a.add(&b).unwrap()),
            embed(&a).add(&embed(&b)).unwrap()
        );
        // Its components at proof degree 64: those at c = 0 mod K are v's, the others 0.
        let proof = Ring::new(13, 64).unwrap();
        let parts = iso::split(&e, proof.clone()).unwrap();
        let own = iso::split(&a, proof).unwrap();
        for (c, part) in parts.iter().enumerate() {
            if c % k == 0 {
                assert_eq!(*part, own[c / k]);
            } else {
                assert!(part.is_zero());
            }
        }
        // coefficient_in reads coefficient i of v through the constant coefficient.
        let mut st = Statement::new(big.clone());
        st.var_subring("v", 2, degree, Norm::Binary, Placement::Ajtai)
            .unwrap();
        let w = PolyVec::new(big.clone(), vec![Poly::zero(big.clone()), e.clone()]).unwrap();
        for i in [0, 1, degree / 2, degree - 1] {
            let form = st.coefficient_in(&big, "v", 1, i).unwrap();
            assert_eq!(
                form.evaluate(&w).unwrap().coefficients()[0],
                a.coefficients()[i],
                "degree {degree}, index {i}"
            );
        }
        assert_eq!(
            st.coefficient_in(&big, "v", 1, degree).err(),
            Some(Error::Index)
        );
        assert_eq!(st.coefficient_in(&big, "v", 2, 0).err(), Some(Error::Index));
        assert_eq!(
            st.coefficient_in(&Ring::new(13, 512).unwrap(), "v", 0, 0)
                .err(),
            Some(Error::RingMismatch)
        );
    }
    let a = Poly::constant(Ring::new(13, 128).unwrap(), 1);
    assert_eq!(
        iso::embed(&a, Ring::new(12, 256).unwrap()).err(),
        Some(Error::RingMismatch)
    );
    assert_eq!(
        iso::embed(&a, Ring::new(13, 64).unwrap()).err(),
        Some(Error::Dimension)
    );
}

/// A statement over $`\mathbb Z_{156}[X]/(X^{256}+1)`$ with its forms, the named witness and
/// the integer witness per variable index ($`s_0..s_3,x,y`$):
///
/// - `s`: four polynomials, $`\|s\|^2\le64`$; `x`: binary, subring of degree 128; `y`:
///   $`\|y\|_\infty\le2`$, subring of degree 128;
/// - A: $`a\,s_0x+b\,y\,s_2+e\,y^2+c=0`$ modulo 13 (committed carries);
/// - B: $`\mathrm{ct}(X^{-2j}y+\sigma(a_j)s_1)=t_j`$ modulo 12 for $`j<3`$, reading
///   coefficient $`j`$ of $`y`$ through `coefficient_in`;
/// - C: $`\mathrm{ct}(X^{-10}x)=x_5`$ modulo 13 (coefficient 5 of $`x`$);
///
/// each made true on the witness. `edit` changes the integer witness first.
struct Built {
    st: Statement,
    forms: Vec<QuadEq>,
    witness: Witness,
}
fn build(seed: u64, edit: impl Fn(&mut [Vec<i128>])) -> Built {
    let ring = Ring::new(156, 256).unwrap();
    let (r12, r13) = (Ring::new(12, 256).unwrap(), Ring::new(13, 256).unwrap());
    let mut rng = SplitMix64(seed);
    let mut st = Statement::new(ring.clone());
    st.var("s", 4, Norm::L2Squared(64)).unwrap();
    st.var_subring("x", 1, 128, Norm::Binary, Placement::Ajtai)
        .unwrap();
    st.var_subring("y", 1, 128, Norm::Linf(2), Placement::Ajtai)
        .unwrap();
    let mut w = vec![vec![0i128; 256]; 6];
    for _ in 0..40 {
        let (i, j) = (rng.below(4) as usize, rng.below(256) as usize);
        w[i][j] = rng.below(3) as i128 - 1;
    }
    for j in 0..128 {
        w[4][2 * j] = rng.below(2) as i128;
        w[5][2 * j] = rng.below(5) as i128 - 2;
    }
    (w[5][0], w[5][2]) = (2, -2);
    edit(&mut w);
    let sparse = |rng: &mut SplitMix64, r: &Arc<Ring>| {
        let mut c = vec![0i128; 256];
        for _ in 0..3 {
            c[rng.below(256) as usize] = rng.below(7) as i128 - 3;
        }
        poly(r, c)
    };
    let names = [("s", 0), ("s", 1), ("s", 2), ("s", 3), ("x", 0), ("y", 0)];
    let v = |r: &Arc<Ring>, i: usize| st.variable_in(r, names[i].0, names[i].1).unwrap();
    // Minus the value at the witness in the form's own ring: the constraint then holds.
    let settle = |form: QuadEq| {
        let r = form.r0.ring().clone();
        let value = poly(&r, integer_value(&form, &w));
        form.add(&st.constant_in(value.neg()).unwrap()).unwrap()
    };
    let (a, b, e) = (
        sparse(&mut rng, &r13),
        sparse(&mut rng, &r13),
        Poly::constant(r13.clone(), 1),
    );
    let mut forms = vec![settle(
        v(&r13, 0)
            .product_affine(&v(&r13, 4))
            .unwrap()
            .scale(&a)
            .unwrap()
            .add(
                &v(&r13, 5)
                    .product_affine(&v(&r13, 2))
                    .unwrap()
                    .scale(&b)
                    .unwrap(),
            )
            .unwrap()
            .add(
                &v(&r13, 5)
                    .product_affine(&v(&r13, 5))
                    .unwrap()
                    .scale(&e)
                    .unwrap(),
            )
            .unwrap(),
    )];
    for j in 0..3 {
        let aj = poly(&r12, rng.uniform(256, 12));
        forms.push(settle(
            st.coefficient_in(&r12, "y", 0, j)
                .unwrap()
                .add(&v(&r12, 1).scale(&aj.auto()).unwrap())
                .unwrap(),
        ));
    }
    forms.push(settle(st.coefficient_in(&r13, "x", 0, 5).unwrap()));
    st.eq_mod_p(forms[0].clone()).unwrap();
    for clause in &forms[1..] {
        st.const_coeff_zero(clause.clone()).unwrap();
    }
    let witness = Witness::from([
        (
            "s".into(),
            w[..4].iter().map(|c| poly(&ring, c.clone())).collect(),
        ),
        ("x".into(), vec![poly(&ring, w[4].clone())]),
        ("y".into(), vec![poly(&ring, w[5].clone())]),
    ]);
    Built { st, forms, witness }
}

/// The oracle's bounds per variable index: $`\|s\|^2\le64`$, binary `x` with 128 possibly
/// nonzero coefficients, `y` at `b` over 128 coefficients.
fn bounds(b: u128) -> [VarBound; 6] {
    [
        Squared(64),
        Squared(64),
        Squared(64),
        Squared(64),
        Squared(128),
        LinfIn(b, 128),
    ]
}

#[test]
fn requirements_and_blocks_count_the_subring_components() {
    let b = build(0x2567_2201, |_| {});
    let f: Vec<u128> = b
        .forms
        .iter()
        .map(|f| f_bound_linf(f, &bounds(2)))
        .collect();
    for (degree, m1, l, n_bin, n_prime) in [(64, 20, 5, 2, 7), (128, 10, 3, 1, 4)] {
        let req = b.st.requirements(degree, U256::from_u64(Q)).unwrap();
        // s: 256/d rows each; x and y: 128/d each. l: A's 256/d carries and one packed
        // polynomial; n': those and y's rows.
        assert_eq!(
            (req.m1, req.l, req.n_bin, req.n_prime),
            (m1, l, n_bin, n_prime),
            "degree {degree}"
        );
        assert_eq!(
            (req.l2_rows.clone(), req.l2_bounds_squared.clone()),
            (vec![1024 / degree], vec![64])
        );
        // Shares: 64, 128 binary coefficients, 128 coefficients at 2.
        assert_eq!(req.alpha_squared, 64 + 128 + 128 * 4);
        let blocks = b.st.blocks(degree).unwrap();
        assert_eq!(
            blocks.iter().map(|b| (b.rows, b.norm)).collect::<Vec<_>>(),
            [
                (1024 / degree, Norm::L2Squared(64)),
                (128 / degree, Norm::Binary),
                (128 / degree, Norm::Linf(2))
            ]
        );
        // F per modulus: A at 13 (the largest there, above C), the clauses at 12.
        let f13 = f[0].max(f[4]);
        let f12 = f[1..4].iter().copied().max().unwrap();
        assert_eq!(
            req.lifted_moduli
                .iter()
                .map(|m| (narrow(&m.modulus), narrow(&m.max_integer_coefficient)))
                .collect::<Vec<_>>(),
            [(12, f12), (13, f13)]
        );
        let quotient = f13.div_ceil(13).max(f12.div_ceil(12));
        assert_eq!(req.linf_bound, quotient.max(2));
        // The per-slot bound: A's 256 slots, one per clause, and y's 128 coefficients at 2.
        let per_slot = 256 * f[0].div_ceil(13).pow(2)
            + f[1..4].iter().map(|f| f.div_ceil(12).pow(2)).sum::<u128>()
            + f[4].div_ceil(13).pow(2)
            + 128 * 4;
        assert_eq!(req.approx_alpha_squared, Some(per_slot));
        // The constraints on y, at E: A and the three clauses.
        let linf = req.linf.as_ref().unwrap();
        assert_eq!(linf.lifting.len(), 4);
        for (entry, form) in linf.lifting.iter().zip(&b.forms) {
            for e in [2, 1 << 20] {
                assert_eq!(linf_f(entry, e), f_bound_linf(form, &bounds(e)));
            }
        }
        // y y weighs ceil(sqrt(128 * 128)) = 128 per unit of ||e||_1, not the full degree 256.
        let e_l1: u128 = b.forms[0]
            .r2
            .entries()
            .filter(|((i, j), _)| (*i, *j) == (5, 5))
            .map(|(_, p)| {
                p.coefficients_i128()
                    .unwrap()
                    .iter()
                    .map(|x| x.unsigned_abs())
                    .sum::<u128>()
            })
            .sum();
        assert_eq!(narrow(&linf.lifting[0].quadratic), 128 * e_l1);
        assert!(e_l1 > 0);
    }
}

#[test]
fn subring_variables_prove_at_both_degrees() {
    let b = build(0x2567_2202, |_| {});
    for degree in [64, 128] {
        let req = b.st.requirements(degree, U256::from_u64(Q)).unwrap();
        let params = fitted(&req, degree, "subring-test-only");
        let compiled = b.st.compile(params.clone()).unwrap();
        assert_eq!(
            compiled.extraction_bound("y"),
            Ok(Extraction::Linf(extraction_bound(params.log_sigma[3])))
        );
        let (s1, m) = compiled.map_witness(&b.witness).unwrap();
        // The Ajtai part: s's components, then the committed components of x and y, the
        // components of each as a polynomial of degree 128.
        let q = narrow(&params.prime_factors[0]) as i128;
        let proof = Ring::new(q, degree).unwrap();
        let k = 256 / degree;
        let lift = |p: &Poly, d: usize| {
            Poly::new(
                Ring::new(q, d).unwrap(),
                p.coefficients_i128().unwrap().to_vec(),
            )
            .unwrap()
        };
        for (at, name) in [(4 * k, "x"), (4 * k + k / 2, "y")] {
            let full = lift(&b.witness[name][0], 256);
            // v(Z) with coefficient j of v at 2j of the statement polynomial.
            let values: Vec<i128> = full
                .coefficients_i128()
                .unwrap()
                .iter()
                .step_by(2)
                .copied()
                .collect();
            let own = iso::split(
                &Poly::new(Ring::new(q, 128).unwrap(), values).unwrap(),
                proof.clone(),
            )
            .unwrap();
            assert_eq!(&s1.entries()[at..at + k / 2], &own[..], "{name}");
        }
        assert_eq!(s1.len(), 5 * k);
        // The range rows: A's carries, the packed carries, then y's committed components.
        let arp = compiled.statement().arp.as_ref().unwrap();
        let rows = arp.evaluate(&s1, &m).unwrap();
        assert_eq!(&rows.entries()[k + 1..], &s1.entries()[4 * k + k / 2..]);
        let bytes = compiled
            .prove_bytes_with_seed([1; 32], &b.witness, CTX, [2; 32])
            .unwrap();
        compiled.verify_bytes([1; 32], &bytes, CTX).unwrap();
        assert!(compiled.verify_bytes([1; 32], &bytes, b"other").is_err());
        // The honest y holds 2 and -2. Outside the subring the witness map refuses any nonzero
        // coefficient, and inside it one above the norm, before it evaluates a constraint.
        for (name, index, value) in [("x", 1, 1), ("y", 255, -1), ("y", 100, 3), ("x", 100, 2)] {
            let mut edited = b.witness.clone();
            edited.get_mut(name).unwrap()[0]
                .set_coefficient(index, value)
                .unwrap();
            assert_eq!(
                compiled.map_witness(&edited).err(),
                Some(Error::Witness),
                "{name}[{index}] = {value}"
            );
        }
    }
}
