//! A statement with two moduli on one short key, end to end, at toy sizes: a quadratic relation
//! modulo $`q`$ and LWR rounding clauses modulo $`q_{tag}=\gamma p`$, which share the key
//! $`usk`$. Over $`\mathbb Z_q[X]/(X^{1024}+1)`$ it combines
//!
//! - the quadratic relation (called possession below)
//!   $`(A_0s_0+A_1s_1-b_u\,usk-B_rx_0-b_\varphi\varphi)(v-x_1)=1`$ modulo $`q`$;
//! - the tag clauses $`\langle a_j,usk\rangle+e_j=\gamma T_j`$ modulo $`q_{tag}`$ for $`j<4`$,
//!   as $`\mathrm{ct}(\sigma(a_j)usk+X^{-j}E)=\gamma T_j`$ with $`e_j`$ at coefficient $`j`$ of
//!   $`E`$, on the same $`usk`$, each clause over $`\mathbb Z_{q_{tag}}[X]/(X^{1024}+1)`$ and so
//!   lifted with its own modulus, the four carries packed into one proof-ring polynomial;
//! - an exact Euclidean bound on $`s`$, and $`\ell_\infty`$ bounds of 1 on $`x_1,usk,x_0,E`$.
//!
//! The fast test also builds the statement with a single statement modulus, for comparison:
//! over $`P=q\,q_{tag}`$, each clause scaled by the other modulus (`modulo`), which costs about
//! $`\log_2q_{tag}`$ bits of proof modulus. The $`\ell_\infty`$ bounds of 1 are exact: each is
//! $`x=x^+-x^-`$ with two binary blocks (exact for bound 1 only). A third form declares them
//! `Norm::Linf(1)` instead, which the approximate range proof bounds only by its extraction
//! bound, $`2\cdot`$`z4_bound` $`\approx2^{37.6}`$ here (one range block). Its statement
//! ring is $`\mathbb Z_P`$: over $`\mathbb Z_q`$ the range condition would need that bound to
//! be at most 6 for the variables of the tag clauses. Its proof modulus is about $`2^{99}`$, as
//! the bound enters the lifting inequality squared. A fourth form, over $`\mathbb Z_q`$ as the
//! first, bounds $`x_1,usk,x_0,E`$ by 1 exactly with `Norm::LinfExact(1)` (two bits per
//! coefficient, substituted into the constraints), and holds $`E`$ in the subring of degree 128,
//! $`e_j`$ at its coefficient $`j`$, which clause $`j`$ reads with `coefficient_in`; its proof
//! modulus is about $`2^{41.1}`$. The moduli are toys ($`q=13`$, $`\gamma=3`$, $`p=4`$), and
//! parameters come from the test port of the parameter tool with synthetic MLWE metadata. The
//! proofs take minutes and are ignored in ordinary runs; `CONTRIBUTING.md` gives the command.
use jali::{
    Error,
    abdlop::Abdlop,
    math::{Poly, Ring, U256, iso},
    params::TboxParams,
    quad::QuadEq,
    statement::{Extraction, LiftedModulus, Norm, Placement, Requirements, Statement},
};
use std::{collections::BTreeMap, sync::Arc};

mod common;
use common::{
    SplitMix64, narrow,
    params::{
        even_divisors, extraction_bound, fit_downward, fit_upward, fit_wide, lifting_threshold,
        linf_f, per_slot_rule, range_width,
    },
    ring::{
        VarBound::{self, Linf as L, LinfIn, Squared},
        centred, f_bound, f_bound_linf, integer_value, inverse, poly, split_ternary,
    },
};

const D: usize = 1024;
/// Modulus of the quadratic relation, a prime $`q\equiv5\pmod8`$. The order of 13 modulo 2048
/// is 512, so $`X^{1024}+1`$ has two irreducible factors of degree 512 modulo 13.
const QC: i128 = 13;
/// Tag rounding: $`q_{tag}=\gamma p`$ with $`\gamma`$ odd.
const GAMMA: i128 = 3;
const PT: i128 = 4;
const QT: i128 = GAMMA * PT;
/// The one statement modulus of the CRT form.
const P: i128 = QC * QT;
const M_TAG: usize = 4;
const S_BOUND: u64 = 2048;
const SEED: u64 = 2026;
const CTX: &[u8] = b"jali-test/tag-preimage";

/// How the statement holds its two moduli and its $`\ell_\infty`$ bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Moduli {
    /// Possession over $`\mathbb Z_q`$, the statement ring, and each tag clause over
    /// $`\mathbb Z_{q_{tag}}`$: every constraint lifted with its own modulus.
    Own,
    /// Everything over $`\mathbb Z_P`$, each clause scaled by the other modulus.
    Crt,
    /// As `Own`, with $`x_1,usk,x_0,E`$ declared `Norm::Linf(1)` over the statement ring
    /// $`\mathbb Z_P`$, which both moduli divide.
    Linf,
    /// As `Own`, with $`x_1,usk,x_0`$ declared `Norm::LinfExact(1)` and $`E`$ an element of the
    /// subring of degree 128, `Norm::LinfExact(1)`, read at coefficient $`j`$ by clause $`j`$.
    Exact,
}
use Moduli::{Crt, Exact, Linf, Own};
impl Moduli {
    /// The statement modulus, which witness values are read modulo.
    fn statement(self) -> i128 {
        match self {
            Own | Exact => QC,
            Crt | Linf => P,
        }
    }
    /// The modulus the compiled possession constraint and tag clauses are lifted from.
    fn possession(self) -> i128 {
        match self {
            Own | Linf | Exact => QC,
            Crt => P,
        }
    }
    fn tag(self) -> i128 {
        match self {
            Own | Linf | Exact => QT,
            Crt => P,
        }
    }
}

struct Instance {
    a0: Vec<i128>,
    a1: Vec<i128>,
    b_u: Vec<i128>,
    b_r: Vec<i128>,
    b_phi: Vec<i128>,
    v: Vec<i128>,
    phi: Vec<i128>,
    tag_a: Vec<Vec<i128>>,
    tag_t: Vec<i128>,
    witness: BTreeMap<String, Vec<Vec<i128>>>,
    /// $`x_1,usk,x_0,E`$ as ternary vectors, for the `Linf` form.
    ternary: BTreeMap<String, Vec<i128>>,
}

/// An honest instance without a trapdoor: $`A_1=(t-A_0s_0)s_1^{-1}`$ for
/// $`t=(v-x_1)^{-1}+b_u\,usk+b_\varphi\varphi+B_rx_0`$. With `s_norm`, coefficients of
/// $`s_0`$ are raised until $`\|s\|^2`$ equals it (so $`A_1`$ matches a too-long $`s`$).
fn instance(seed: u64, s_norm: Option<u64>) -> Instance {
    let rq = Ring::new(QC, D).unwrap();
    let mut rng = SplitMix64(seed);
    let pq = |c: &[i128]| poly(&rq, c.to_vec());
    let x1 = rng.ternary(D);
    let usk = rng.ternary(D);
    let x0 = rng.ternary(D);
    let phi = rng.binary(D);
    let (b_u, b_r, b_phi, a0) = (
        rng.uniform(D, QC),
        rng.uniform(D, QC),
        rng.uniform(D, QC),
        rng.uniform(D, QC),
    );
    let (v, y) = loop {
        let v = rng.uniform(D, QC);
        let g: Vec<i128> = v.iter().zip(&x1).map(|(a, b)| a - b).collect();
        if let Some(y) = inverse(&g, QC) {
            break (v, y);
        }
    };
    let t = pq(&y)
        .add(&pq(&b_u).mul(&pq(&usk)).unwrap())
        .unwrap()
        .add(&pq(&b_phi).mul(&pq(&phi)).unwrap())
        .unwrap()
        .add(&pq(&b_r).mul(&pq(&x0)).unwrap())
        .unwrap();
    let mut s0 = rng.ternary(D);
    let (s1, s1_inverse) = loop {
        let s1 = rng.ternary(D);
        if let Some(inverse) = inverse(&s1, QC) {
            break (s1, inverse);
        }
    };
    if let Some(target) = s_norm {
        let norm = |s0: &[i128]| s0.iter().chain(&s1).map(|x| (x * x) as u64).sum::<u64>();
        let mut i = 0;
        while norm(&s0) < target {
            // A zero becomes 1 (+1); a +-1 becomes +-2 (+3) while that does not overshoot.
            let add = if s0[i] == 0 { 1 } else { 3 };
            if s0[i].abs() <= 1 && norm(&s0) + add <= target {
                s0[i] += if s0[i] < 0 { -1 } else { 1 };
            }
            i = (i + 1) % D;
        }
        assert_eq!(norm(&s0), target);
    }
    let a1 = t
        .sub(&pq(&a0).mul(&pq(&s0)).unwrap())
        .unwrap()
        .mul(&pq(&s1_inverse))
        .unwrap();
    let r = pq(&a0)
        .mul(&pq(&s0))
        .unwrap()
        .add(&a1.mul(&pq(&s1)).unwrap())
        .unwrap()
        .sub(&pq(&b_u).mul(&pq(&usk)).unwrap())
        .unwrap()
        .sub(&pq(&b_r).mul(&pq(&x0)).unwrap())
        .unwrap()
        .sub(&pq(&b_phi).mul(&pq(&phi)).unwrap())
        .unwrap();
    let g = pq(&v).sub(&pq(&x1)).unwrap();
    assert_eq!(r.mul(&g).unwrap(), Poly::constant(rq, 1));
    let mut tag_a = Vec::new();
    let mut tag_t = Vec::new();
    let mut e = vec![0i128; D];
    // e_j sits at coefficient j of E.
    for ej_slot in e.iter_mut().take(M_TAG) {
        let a: Vec<i128> = rng
            .uniform(D, QT)
            .into_iter()
            .map(|x| centred(x, QT))
            .collect();
        let w: i128 = a.iter().zip(&usk).map(|(x, y)| x * y).sum();
        // T_j = round((w mod q_tag) / gamma) mod p, and e_j = gamma T_j - w modulo q_tag.
        let t = ((2 * w.rem_euclid(QT) + GAMMA) / (2 * GAMMA)).rem_euclid(PT);
        let ej = centred(GAMMA * t - w, QT);
        assert!(ej.abs() <= GAMMA / 2, "honest rounding error");
        *ej_slot = ej;
        tag_a.push(a);
        tag_t.push(t);
    }
    let mut witness = BTreeMap::new();
    witness.insert("s".to_string(), vec![s0, s1]);
    let mut ternary = BTreeMap::new();
    for (name, t) in [("x1", &x1), ("usk", &usk), ("x0", &x0), ("e", &e)] {
        let (plus, minus) = split_ternary(t);
        witness.insert(format!("{name}+"), vec![plus]);
        witness.insert(format!("{name}-"), vec![minus]);
        ternary.insert(name.to_string(), t.clone());
    }
    Instance {
        a0,
        a1: common::values(&a1),
        b_u,
        b_r,
        b_phi,
        v,
        phi,
        tag_a,
        tag_t,
        witness,
        ternary,
    }
}

/// Require `form` to vanish modulo `modulus`, a divisor of the statement modulus $`P`$, by
/// scaling it with $`P/m`$: $`(P/m)f\equiv0\pmod P`$ if and only if $`f\equiv0\pmod m`$. This
/// CRT scaling was the only way to combine two moduli before per-constraint moduli; it costs
/// $`\log_2(P/m)`$ bits of proof modulus.
fn modulo(form: QuadEq, modulus: i128) -> QuadEq {
    let ring = form.r0.ring().clone();
    let big = common::modulus(&ring);
    assert_eq!(big % modulus, 0);
    form.scale(&Poly::constant(ring.clone(), big / modulus))
        .unwrap()
}

/// $`x^+-x^-`$ for the variable pair of `name`, over `ring`, or the variable `name` itself in
/// the `Linf` and `Exact` forms.
fn ternary(st: &Statement, ring: &Arc<Ring>, name: &str, moduli: Moduli) -> QuadEq {
    if matches!(moduli, Linf | Exact) {
        return st.variable_in(ring, name, 0).unwrap();
    }
    let minus = st
        .variable_in(ring, &format!("{name}-"), 0)
        .unwrap()
        .scale(&Poly::constant(ring.clone(), -1))
        .unwrap();
    st.variable_in(ring, &format!("{name}+"), 0)
        .unwrap()
        .add(&minus)
        .unwrap()
}

/// The statement and its constraint forms (possession first, then the tag clauses).
fn statement(inst: &Instance, moduli: Moduli) -> (Statement, Vec<QuadEq>) {
    statement_over(inst, moduli, &Ring::new(moduli.statement(), D).unwrap())
}

/// `statement` over the statement ring `ring`.
fn statement_over(inst: &Instance, moduli: Moduli, ring: &Arc<Ring>) -> (Statement, Vec<QuadEq>) {
    // The ring of the possession constraint and that of the tag clauses: their own, or the
    // statement ring for the CRT form.
    let (pos_ring, tag_ring) = match moduli {
        Own | Linf | Exact => (Ring::new(QC, D).unwrap(), Ring::new(QT, D).unwrap()),
        Crt => (ring.clone(), ring.clone()),
    };
    // Integer lifts of the public values modulo q, centred.
    let lift = |c: &[i128]| poly(&pos_ring, c.iter().map(|x| centred(*x, QC)).collect());
    let mut st = Statement::new(ring.clone());
    st.var("s", 2, Norm::L2Squared(S_BOUND)).unwrap();
    for name in ["x1", "usk", "x0", "e"] {
        match moduli {
            Linf => {
                st.var(name, 1, Norm::Linf(1)).unwrap();
            }
            Exact if name == "e" => {
                st.var_subring(name, 1, E_DEGREE, Norm::LinfExact(1), Placement::Ajtai)
                    .unwrap();
            }
            Exact => {
                st.var(name, 1, Norm::LinfExact(1)).unwrap();
            }
            Own | Crt => {
                st.var(format!("{name}+"), 1, Norm::Binary).unwrap();
                st.var(format!("{name}-"), 1, Norm::Binary).unwrap();
            }
        }
    }
    let neg = Poly::constant(pos_ring.clone(), -1);
    let rq = Ring::new(QC, D).unwrap();
    let b_phi_phi = poly(&rq, inst.b_phi.clone())
        .mul(&poly(&rq, inst.phi.clone()))
        .unwrap();
    let r = st
        .variable_in(&pos_ring, "s", 0)
        .unwrap()
        .scale(&lift(&inst.a0))
        .unwrap()
        .add(
            &st.variable_in(&pos_ring, "s", 1)
                .unwrap()
                .scale(&lift(&inst.a1))
                .unwrap(),
        )
        .unwrap()
        .add(
            &ternary(&st, &pos_ring, "usk", moduli)
                .scale(&lift(&inst.b_u).neg())
                .unwrap(),
        )
        .unwrap()
        .add(
            &ternary(&st, &pos_ring, "x0", moduli)
                .scale(&lift(&inst.b_r).neg())
                .unwrap(),
        )
        .unwrap()
        .add(
            &st.constant_in(lift(&common::values(&b_phi_phi)).neg())
                .unwrap(),
        )
        .unwrap();
    let g = st
        .constant_in(lift(&inst.v))
        .unwrap()
        .add(&ternary(&st, &pos_ring, "x1", moduli).scale(&neg).unwrap())
        .unwrap();
    let possession = r
        .product_affine(&g)
        .unwrap()
        .add(
            &st.constant_in(Poly::constant(pos_ring.clone(), -1))
                .unwrap(),
        )
        .unwrap();
    let mut forms = vec![match moduli {
        Own | Linf | Exact => possession,
        Crt => modulo(possession, QC),
    }];
    for (j, (a, t)) in inst.tag_a.iter().zip(&inst.tag_t).enumerate() {
        // e_j: coefficient j of E, or of the subring element e at coefficient 8 j of E.
        let e_j = if moduli == Exact {
            st.coefficient_in(&tag_ring, "e", 0, j).unwrap()
        } else {
            let x_minus_j = Poly::constant(tag_ring.clone(), 1).rotate(-(j as i64));
            ternary(&st, &tag_ring, "e", moduli)
                .scale(&x_minus_j)
                .unwrap()
        };
        let clause = ternary(&st, &tag_ring, "usk", moduli)
            .scale(&poly(&tag_ring, a.clone()).auto())
            .unwrap()
            .add(&e_j)
            .unwrap()
            .add(
                &st.constant_in(Poly::constant(tag_ring.clone(), -GAMMA * t))
                    .unwrap(),
            )
            .unwrap();
        forms.push(match moduli {
            Own | Linf | Exact => clause,
            Crt => modulo(clause, QT),
        });
    }
    st.eq_mod_p(forms[0].clone()).unwrap();
    for clause in &forms[1..] {
        st.const_coeff_zero(clause.clone()).unwrap();
    }
    (st, forms)
}

fn witness_map(inst: &Instance, moduli: Moduli) -> BTreeMap<String, Vec<Poly>> {
    let ring = Ring::new(moduli.statement(), D).unwrap();
    let named: Vec<(String, Vec<Vec<i128>>)> = match moduli {
        Linf | Exact => std::iter::once(("s".to_string(), inst.witness["s"].clone()))
            .chain(inst.ternary.iter().map(|(k, v)| {
                if moduli == Exact && k == "e" {
                    (k.clone(), vec![subring_e(v)])
                } else {
                    (k.clone(), vec![v.clone()])
                }
            }))
            .collect(),
        Own | Crt => inst.witness.clone().into_iter().collect(),
    };
    named
        .into_iter()
        .map(|(k, v)| (k, v.iter().map(|c| poly(&ring, c.clone())).collect()))
        .collect()
}

/// The degree of the subring that holds $`E`$ in the `Exact` form: both proof degrees divide it,
/// and it holds the four tag errors.
const E_DEGREE: usize = 128;

/// $`E`$ of the `Exact` form: the element $`e(X^8)`$ of the subring of degree 128 whose
/// coefficient $`j`$ is $`e_j`$, from $`E`$ with $`e_j`$ at coefficient $`j`$ (the other forms').
fn subring_e(e: &[i128]) -> Vec<i128> {
    let k = D / E_DEGREE;
    assert!(e[E_DEGREE..].iter().all(|x| *x == 0));
    let mut out = vec![0; D];
    for (j, x) in e[..E_DEGREE].iter().enumerate() {
        out[k * j] = *x;
    }
    out
}

/// A prime $`q\equiv5\pmod8`$ and the factorization of $`q-1`$, both from Sage
/// (`is_prime(proof=True)`).
type WidePrime = (u128, [(&'static str, u32); 5]);
/// The first prime $`\equiv5\pmod8`$ above the lifting threshold of the `Linf` form,
/// 609,886,828,027,574,923,051,022,690,574 at both degrees, and the last one below it.
const LINF_ABOVE: WidePrime = (
    609_886_828_027_574_923_051_022_690_989,
    [
        ("2", 2),
        ("3", 2),
        ("612923", 1),
        ("4047473", 1),
        ("6828996248247577", 1),
    ],
);
const LINF_BELOW: WidePrime = (
    609_886_828_027_574_923_051_022_690_053,
    [
        ("2", 2),
        ("3", 2),
        ("37", 1),
        ("127", 1),
        ("3605299165470046363594043", 1),
    ],
);
/// Parameters at `prime`, which is above $`2^{64}`$, with the divisors of $`q-1`$ from its
/// factorization.
fn wide(req: &Requirements, id: &str, degree: usize, prime: &WidePrime) -> TboxParams {
    let q = U256::from_u128(prime.0);
    fit_wide(req, id, q, &even_divisors(&q, &prime.1), degree).unwrap()
}

/// The first prime above the compiler's lifting threshold that the parameter port fits; for the
/// `Linf` form, `LINF_ABOVE`.
fn params(st: &Statement, degree: usize, moduli: Moduli) -> (Requirements, TboxParams) {
    let req = st
        .requirements(degree, U256::from_u64(1099511627917))
        .unwrap();
    let threshold = lifting_threshold(&req, moduli.statement() as u128, range_width(&req, degree));
    let id = match moduli {
        Own => format!("tag-preimage-own-moduli-d{degree}-test-only"),
        Crt => format!("tag-preimage-crt-d{degree}-test-only"),
        Linf => format!("tag-preimage-linf-d{degree}-test-only"),
        Exact => format!("tag-preimage-linf-exact-d{degree}-test-only"),
    };
    if moduli == Linf {
        assert!(LINF_BELOW.0 <= threshold && threshold < LINF_ABOVE.0);
        return (req.clone(), wide(&req, &id, degree, &LINF_ABOVE));
    }
    let (_, params) = fit_upward(&req, &id, degree, threshold as u64 + 1);
    (req, params)
}

/// The squared bound of each variable's block: 2048 for `s_0,s_1`, and $`1\cdot1024`$ for
/// each binary block.
const BETA: [u128; 10] = [2048, 2048, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024];

#[test]
fn tag_preimage_statement_compiles_and_maps_witnesses() {
    let inst = instance(SEED, None);
    let s_norm: u64 = inst.witness["s"]
        .iter()
        .flatten()
        .map(|x| (x * x) as u64)
        .sum();
    assert!(s_norm <= S_BOUND);
    // Transcript-independent pinned values for this instance generator and seed; they change
    // only with the generator, the statement compiler or the parameter derivation. Both forms
    // have the same shape. The CRT form's bounds are those of the checked-in fixtures
    // `tests/fixtures/params/tag-preimage-crt-d{128,64}`, which count one carry polynomial per
    // tag clause (`l` and `n_prime` 12 and 20); the compiler packs the four tag carries into one
    // polynomial, three fewer rows at each degree.
    for (moduli, linf_bound, max_f) in [(Own, 3_574_883, 46_473_472), (Crt, 3_574_883, 557_681_610)]
    {
        let (st, forms) = statement(&inst, moduli);
        let witness = witness_map(&inst, moduli);
        // The integer bounds from their formula: the largest per modulus, and for the own
        // moduli the list of both.
        let f: Vec<u128> = forms.iter().map(|form| f_bound(form, &BETA)).collect();
        let f_tag = *f[1..].iter().max().unwrap();
        assert_eq!(max_f, f[0].max(f_tag), "{moduli:?}");
        assert_eq!(
            linf_bound,
            f[0].div_ceil(moduli.possession() as u128)
                .max(f_tag.div_ceil(moduli.tag() as u128)),
            "{moduli:?}"
        );
        let lifted_moduli = match moduli {
            Own | Linf | Exact => [(QT, f_tag), (QC, f[0])]
                .map(|(p, f)| LiftedModulus {
                    modulus: U256::from_u128(p as u128),
                    max_integer_coefficient: U256::from_u128(f),
                })
                .to_vec(),
            Crt => Vec::new(),
        };
        for (degree, m1, l, n_bin, rows, n_prime) in
            [(128, 80, 9, 64, 16, 9), (64, 160, 17, 128, 32, 17)]
        {
            let (req, params) = params(&st, degree, moduli);
            assert_eq!(
                (req.m1, req.l, req.alpha_squared, req.n_bin, req.n_prime),
                (m1, l, 10240, n_bin, n_prime),
                "{moduli:?}, degree {degree}"
            );
            // The per-slot bound: 1024 slots at the possession quotient bound and one per tag
            // clause at its own, about 2^53.54 against n'd linf_bound^2. The per-slot rule keeps
            // log_sigma[3] at 33 (its guard would allow 31; computed).
            let per_slot = 1024 * f[0].div_ceil(moduli.possession() as u128).pow(2)
                + f[1..]
                    .iter()
                    .map(|f| f.div_ceil(moduli.tag() as u128).pow(2))
                    .sum::<u128>();
            assert_eq!(req.approx_alpha_squared, Some(per_slot));
            assert_eq!(per_slot, 13_086_503_388_246_863);
            assert_eq!(per_slot_rule(&req, degree, per_slot), (33, 33, 31));
            assert_eq!(params.log_sigma[3], 33);
            assert_eq!(
                (req.l2_rows, req.l2_bounds_squared),
                (vec![rows], vec![S_BOUND])
            );
            assert_eq!(
                (req.linf_bound, narrow(&req.max_integer_coefficient)),
                (linf_bound, max_f)
            );
            assert_eq!(req.lifted_moduli, lifted_moduli);
            let compiled = st.compile(params.clone()).unwrap();
            // The lifting boundary: a prime at or below the threshold is refused, the one
            // chosen above it accepted.
            let req = st
                .requirements(degree, U256::from_u64(1099511627917))
                .unwrap();
            let threshold =
                lifting_threshold(&req, moduli.statement() as u128, params.log_sigma[3]);
            assert!(narrow(&params.prime_factors[0]) > threshold);
            let (low, below) = fit_downward(
                &req,
                "tag-preimage-below-test-only",
                degree,
                threshold as u64,
            );
            assert!(u128::from(low) <= threshold);
            assert_eq!(below.log_sigma[3], params.log_sigma[3]);
            assert_eq!(
                st.compile(below).err(),
                Some(Error::Parameter("modulus lifting bound")),
                "{moduli:?}, degree {degree}"
            );
            // The committed carries c satisfy f = p c over the integers, p the modulus of each
            // constraint: all k components for possession, and for tag clause j coefficient j
            // of the one packed polynomial, whose other coefficients are 0.
            let (_, m) = compiled.map_witness(&witness).unwrap();
            let k = D / degree;
            let q = narrow(&params.prime_factors[0]) as i128;
            let lifted = Ring::new(q, D).unwrap();
            let order = ["s", "x1+", "x1-", "usk+", "usk-", "x0+", "x0-", "e+", "e-"];
            let w: Vec<Vec<i128>> = order
                .iter()
                .flat_map(|n| inst.witness[*n].clone())
                .collect();
            let f = integer_value(&forms[0], &w);
            let carry = iso::join(&m.entries()[..k], lifted.clone()).unwrap();
            assert!(
                f.iter()
                    .zip(common::values(&carry))
                    .all(|(f, c)| *f == moduli.possession() * c)
            );
            let packed = common::values(&m.entries()[k]);
            for (j, form) in forms[1..].iter().enumerate() {
                let f = integer_value(form, &w)[0];
                assert_eq!(f, moduli.tag() * packed[j], "tag clause {j}");
            }
            assert!(packed[M_TAG..].iter().all(|x| *x == 0));
            assert_eq!(m.len(), k + 1);
            if degree != 128 {
                continue;
            }
            // Negatives, each a single change: a usk coefficient, a coefficient 2 in e+, a too
            // long s with a matching A_1, and a statement with T_3 + 1.
            let mut bad = witness.clone();
            let (u, um) = (
                bad["usk+"][0].coefficient_i128(100).unwrap(),
                bad["usk-"][0].coefficient_i128(100).unwrap(),
            );
            if u == 0 && um == 0 {
                bad.get_mut("usk+").unwrap()[0]
                    .set_coefficient(100, 1)
                    .unwrap();
            } else {
                bad.get_mut("usk+").unwrap()[0]
                    .set_coefficient(100, 0)
                    .unwrap();
                bad.get_mut("usk-").unwrap()[0]
                    .set_coefficient(100, 0)
                    .unwrap();
            }
            assert_eq!(
                compiled.map_witness(&bad).err(),
                Some(Error::Witness),
                "usk"
            );
            let mut bad = witness.clone();
            // Coefficient 100 of E is in no clause, so only the binary check refuses the 2.
            bad.get_mut("e+").unwrap()[0]
                .set_coefficient(100, 2)
                .unwrap();
            assert_eq!(compiled.map_witness(&bad).err(), Some(Error::Witness), "e");
            let long = instance(SEED, Some(S_BOUND + 1));
            let (long_st, _) = statement(&long, moduli);
            let long_compiled = long_st.compile(params.clone()).unwrap();
            assert_eq!(
                long_compiled.map_witness(&witness_map(&long, moduli)).err(),
                Some(Error::Witness),
                "s"
            );
            // At exactly the bound, the same construction is accepted.
            let exact = instance(SEED, Some(S_BOUND));
            let (exact_st, _) = statement(&exact, moduli);
            exact_st
                .compile(params.clone())
                .unwrap()
                .map_witness(&witness_map(&exact, moduli))
                .unwrap();
            let mut other = instance(SEED, None);
            other.tag_t[M_TAG - 1] = (other.tag_t[M_TAG - 1] + 1) % PT;
            let other = statement(&other, moduli).0.compile(params).unwrap();
            assert_eq!(
                other.map_witness(&witness).err(),
                Some(Error::Witness),
                "tag"
            );
        }
    }
}

/// The bounds of the `Linf` form per variable index: $`\|s\|^2\le2048`$ for $`s_0,s_1`$,
/// then $`x_1,usk,x_0,E`$ bounded by `b` in $`\ell_\infty`$.
fn linf_bounds(b: u128) -> [VarBound; 6] {
    [Squared(2048), Squared(2048), L(b), L(b), L(b), L(b)]
}

#[test]
fn tag_preimage_linf_statement_compiles_and_maps_witnesses() {
    let inst = instance(SEED, None);
    let (st, forms) = statement(&inst, Linf);
    let witness = witness_map(&inst, Linf);
    // Transcript-independent pinned values for this instance generator and seed. The honest
    // bounds take |x| <= 1 per coefficient, which the l1 form of the bound uses better than the
    // binary pairs' l2 form: 16,521,079 for possession against 46,473,472.
    let honest: Vec<u128> = forms
        .iter()
        .map(|form| f_bound_linf(form, &linf_bounds(1)))
        .collect();
    let f_tag = *honest[1..].iter().max().unwrap();
    assert_eq!((honest[0], f_tag), (16_521_079, 3_093));
    // Over Z_13, the range condition would need an extraction bound of at most 6 for usk and E,
    // which the tag clauses read modulo 12, not a divisor of 13.
    let (over_13, _) = statement_over(&inst, Linf, &Ring::new(QC, D).unwrap());
    for (degree, m1, l, rows, n_prime) in [(128, 48, 9, 16, 41), (64, 96, 17, 32, 81)] {
        let (req, params) = params(&st, degree, Linf);
        // Bounded: s, and x1, usk, x0, E, each 1024 coefficients at 1; range rows: the
        // possession carries, one packed polynomial for the four tag clauses' carries, and the
        // four variables.
        assert_eq!(
            (req.m1, req.l, req.alpha_squared, req.n_bin, req.n_prime),
            (m1, l, 2048 + 4 * 1024, 0, n_prime),
            "degree {degree}"
        );
        assert_eq!(
            (req.l2_rows.clone(), req.l2_bounds_squared.clone()),
            (vec![rows], vec![S_BOUND])
        );
        assert_eq!(
            (req.linf_bound, narrow(&req.max_integer_coefficient)),
            (1_270_853, 16_521_079)
        );
        assert_eq!(
            req.linf_bound,
            honest[0].div_ceil(13).max(f_tag.div_ceil(12))
        );
        let lifted_moduli = [(QT, f_tag), (QC, honest[0])].map(|(p, f)| LiftedModulus {
            modulus: U256::from_u128(p as u128),
            max_integer_coefficient: U256::from_u128(f),
        });
        assert_eq!(req.lifted_moduli, lifted_moduli.to_vec());
        // The extraction bound e, 2^37.6, and the bounds at it: the possession constraint's
        // x1 usk and x1 x0 terms grow with e^2.
        let t4 = params.log_sigma[3];
        let e = extraction_bound(t4);
        assert_eq!((t4, e), (32, 213_030_377_880));
        // The per-slot bound: the possession and tag slots, and 1024 coefficients at 1 for each
        // of the four variables, about 2^50.55. The per-slot rule narrows log_sigma[3] to 31
        // and so halves E up to rounding (computed; the proofs use `params`, at 32).
        let per_slot = 1024 * honest[0].div_ceil(13).pow(2)
            + honest[1..]
                .iter()
                .map(|f| f.div_ceil(12).pow(2))
                .sum::<u128>()
            + 4 * 1024;
        assert_eq!(req.approx_alpha_squared, Some(per_slot));
        assert_eq!(per_slot, 1_653_828_964_216_337);
        assert_eq!(per_slot_rule(&req, degree, per_slot), (31, 31, 30));
        assert_eq!(extraction_bound(31), 106_515_188_940);
        let checked = params.check().unwrap();
        assert_eq!(checked.approx_extraction_bound, e);
        let linf = req.linf.as_ref().unwrap();
        assert_eq!(linf.extraction_limit, None);
        assert_eq!(linf.lifting.len(), 1 + M_TAG);
        for (entry, form) in linf.lifting.iter().zip(&forms) {
            assert_eq!(linf_f(entry, e), f_bound_linf(form, &linf_bounds(e)));
            assert_eq!(linf_f(entry, 1), f_bound_linf(form, &linf_bounds(1)));
        }
        assert_eq!(narrow(&linf.lifting[0].quadratic), 6_719_488);
        // The threshold with F_j(e): the prime above is admitted, the one below refused.
        let threshold = lifting_threshold(&req, P as u128, t4);
        assert_eq!(threshold, 609_886_828_027_574_923_051_022_690_574);
        let compiled = st.compile(params.clone()).unwrap();
        let below = wide(
            &req,
            "tag-preimage-linf-below-test-only",
            degree,
            &LINF_BELOW,
        );
        assert_eq!(below.log_sigma[3], t4);
        // With the declared bounds of 1 the threshold would be below 2^43.
        let declared = Requirements {
            linf: None,
            ..req.clone()
        };
        assert!(lifting_threshold(&declared, P as u128, t4) < 1 << 43);
        assert_eq!(
            st.compile(below).err(),
            Some(Error::Parameter("modulus lifting bound")),
            "degree {degree}"
        );
        // What a proof bounds: e for each of these variables, where the binary pairs of the
        // other forms bound usk exactly by 1.
        for name in ["x1", "usk", "x0", "e"] {
            assert_eq!(compiled.extraction_bound(name), Ok(Extraction::Linf(e)));
        }
        assert_eq!(
            statement(&inst, Own)
                .0
                .compile(params_for_own(degree))
                .unwrap()
                .extraction_bound("usk+"),
            Ok(Extraction::Binary)
        );
        // The committed carries: f = 13 c for possession and f = 12 c for each tag clause, at
        // its coefficient of the packed polynomial.
        let (_, m) = compiled.map_witness(&witness).unwrap();
        let k = D / degree;
        let lifted = Ring::new(narrow(&checked.q) as i128, D).unwrap();
        let w: Vec<Vec<i128>> = std::iter::once(inst.witness["s"].clone())
            .flatten()
            .chain(["x1", "usk", "x0", "e"].map(|n| inst.ternary[n].clone()))
            .collect();
        let f = integer_value(&forms[0], &w);
        let carry = iso::join(&m.entries()[..k], lifted).unwrap();
        assert!(
            f.iter()
                .zip(common::values(&carry))
                .all(|(f, c)| *f == QC * c)
        );
        for (j, form) in forms[1..].iter().enumerate() {
            let f = integer_value(form, &w)[0];
            assert_eq!(f, QT * m.entries()[k].coefficient_i128(j).unwrap());
        }
        assert_eq!(m.len(), k + 1);
        // Over Z_13: the same requirements but for the limit, which e fails.
        let req_13 = over_13
            .requirements(degree, U256::from_u64(1099511627917))
            .unwrap();
        assert_eq!(
            req_13.linf.as_ref().unwrap().extraction_limit,
            Some(U256::from_u8(6))
        );
        let mut same = req_13.clone();
        same.linf.as_mut().unwrap().extraction_limit = None;
        assert_eq!(same, req);
        assert_eq!(
            over_13.compile(params.clone()).err(),
            Some(Error::Parameter("variable range above statement modulus"))
        );
        // Coefficient 100 of E is in no clause: 1 is admitted, 2 refused.
        for (value, admitted) in [(1, true), (-1, true), (2, false), (-2, false)] {
            let mut edited = witness.clone();
            edited.get_mut("e").unwrap()[0]
                .set_coefficient(100, value)
                .unwrap();
            assert_eq!(compiled.map_witness(&edited).is_ok(), admitted, "{value}");
        }
    }
}

/// The bounds of the `Exact` form per variable index: $`\|s\|^2\le2048`$ for $`s_0,s_1`$,
/// $`x_1,usk,x_0`$ bounded by 1, and $`E`$ by 1 on its 128 subring coefficients.
fn exact_bounds() -> [VarBound; 6] {
    [
        Squared(2048),
        Squared(2048),
        L(1),
        L(1),
        L(1),
        LinfIn(1, E_DEGREE as u128),
    ]
}

#[test]
fn tag_preimage_linf_exact_statement_compiles_and_maps_witnesses() {
    let inst = instance(SEED, None);
    let (st, forms) = statement(&inst, Exact);
    let witness = witness_map(&inst, Exact);
    // Transcript-independent pinned values for this instance generator and seed. The exact
    // bounds of 1 give the l1 form of the integer bounds, as the Linf form's honest ones:
    // 16,521,079 for possession against 46,473,472 for the binary pairs.
    let f: Vec<u128> = forms
        .iter()
        .map(|form| f_bound_linf(form, &exact_bounds()))
        .collect();
    assert_eq!(f, [16_521_079, 3_027, 3_093, 3_040, 3_064]);
    let f_tag = *f[1..].iter().max().unwrap();
    for (degree, m1, l, n_bin, rows, n_prime) in
        [(128, 66, 9, 50, 16, 9), (64, 132, 17, 100, 32, 17)]
    {
        let (req, params) = params(&st, degree, Exact);
        // Bounded: s, and two bits per coefficient of x1, usk, x0 (1024 each) and of E's 128
        // subring coefficients; binary rows: those bits; range rows: the possession carries and
        // one packed polynomial for the tag carries.
        assert_eq!(
            (req.m1, req.l, req.alpha_squared, req.n_bin, req.n_prime),
            (m1, l, 2048 + 2 * (3 * 1024 + 128), n_bin, n_prime),
            "degree {degree}"
        );
        assert_eq!(
            (req.l2_rows.clone(), req.l2_bounds_squared.clone()),
            (vec![rows], vec![S_BOUND])
        );
        assert_eq!(
            (req.linf_bound, narrow(&req.max_integer_coefficient)),
            (1_270_853, 16_521_079)
        );
        assert_eq!(req.linf_bound, f[0].div_ceil(13).max(f_tag.div_ceil(12)));
        let lifted_moduli = [(QT, f_tag), (QC, f[0])].map(|(p, f)| LiftedModulus {
            modulus: U256::from_u128(p as u128),
            max_integer_coefficient: U256::from_u128(f),
        });
        assert_eq!(req.lifted_moduli, lifted_moduli.to_vec());
        // Exact bounds need no condition on E.
        assert_eq!(req.linf, None);
        // The per-slot bound: the possession and tag slots; the default rule and the per-slot
        // rule both give log_sigma[3] = 31 (computed), one below the binary pairs' 33.
        let per_slot = 1024 * f[0].div_ceil(13).pow(2)
            + f[1..].iter().map(|f| f.div_ceil(12).pow(2)).sum::<u128>();
        assert_eq!(req.approx_alpha_squared, Some(per_slot));
        assert_eq!(per_slot_rule(&req, degree, per_slot).0, 31);
        assert_eq!(params.log_sigma[3], 31);
        // The first prime = 5 mod 8 above the lifting threshold that the port fits, at both
        // degrees.
        assert_eq!(narrow(&params.prime_factors[0]), 2_423_280_293_101);
        let compiled = st.compile(params.clone()).unwrap();
        let threshold = lifting_threshold(&req, QC as u128, params.log_sigma[3]);
        assert!(narrow(&params.prime_factors[0]) > threshold);
        let (low, below) = fit_downward(
            &req,
            "tag-preimage-below-test-only",
            degree,
            threshold as u64,
        );
        assert!(u128::from(low) <= threshold);
        assert_eq!(below.log_sigma[3], params.log_sigma[3]);
        assert_eq!(
            st.compile(below).err(),
            Some(Error::Parameter("modulus lifting bound")),
            "degree {degree}"
        );
        // What a proof bounds: 1, exactly, for each of these variables.
        for name in ["x1", "usk", "x0", "e"] {
            assert_eq!(
                compiled.extraction_bound(name),
                Ok(Extraction::LinfExact(1))
            );
        }
        // The witness map: the carries as in the other forms (the messages), and for x1, usk,
        // x0 and E two bits per coefficient, x = v + 1 = x_0 + x_1 (the weights (1, 1)).
        let (s1, m) = compiled.map_witness(&witness).unwrap();
        let k = D / degree;
        let q = narrow(&params.prime_factors[0]) as i128;
        let w: Vec<Vec<i128>> = std::iter::once(inst.witness["s"].clone())
            .flatten()
            .chain(["x1", "usk", "x0"].map(|n| inst.ternary[n].clone()))
            .chain([subring_e(&inst.ternary["e"])])
            .collect();
        let possession = integer_value(&forms[0], &w);
        let carry = iso::join(&m.entries()[..k], Ring::new(q, D).unwrap()).unwrap();
        assert!(
            possession
                .iter()
                .zip(common::values(&carry))
                .all(|(f, c)| *f == QC * c)
        );
        for (j, form) in forms[1..].iter().enumerate() {
            let f = integer_value(form, &w)[0];
            assert_eq!(
                f,
                QT * m.entries()[k].coefficient_i128(j).unwrap(),
                "tag {j}"
            );
        }
        assert_eq!(m.len(), k + 1);
        // After s (2k polynomials): x1, usk, x0 with k components of two bits each, then E's
        // committed components (d_v / d of them, at c = 0 mod 8), two bits each.
        let bits = |at: usize| {
            (
                common::values(&s1.entries()[at]),
                common::values(&s1.entries()[at + 1]),
            )
        };
        for (v, values) in w[2..].iter().enumerate() {
            let stride = if v == 3 { D / E_DEGREE } else { 1 };
            for (r, c) in (0..k).step_by(stride).enumerate() {
                let (x0, x1) = bits(2 * k + 2 * k * v + 2 * r);
                for t in 0..degree {
                    assert!(x0[t] == 0 || x0[t] == 1);
                    assert!(x1[t] == 0 || x1[t] == 1);
                    assert_eq!(x0[t] + x1[t] - 1, values[k * t + c], "variable {v}");
                }
            }
        }
        assert_eq!(s1.len(), 2 * k + 3 * 2 * k + 2 * (E_DEGREE / degree));
        // Every compiled equation, with the bits substituted, holds on the mapped witness.
        let full = jali::quad::interleave(&s1, &m).unwrap();
        let lowered = compiled.statement();
        for equation in &lowered.quadratic {
            assert!(equation.evaluate(&full).unwrap().is_zero());
        }
        for equation in &lowered.evaluation {
            assert_eq!(equation.evaluate(&full).unwrap().coefficient_i128(0), Ok(0));
        }
        if degree != 128 {
            continue;
        }
        // Negatives, each a single change: a usk coefficient 2, a nonzero coefficient of E
        // outside the subring, and a statement with T_3 + 1. Coefficient 800 = 8 * 100 of E is
        // in the subring and in no clause: 1 is admitted, 2 refused.
        for (name, index, value, admitted) in [
            ("usk", 100, 2, false),
            ("e", 1, 1, false),
            ("e", 800, 1, true),
            ("e", 800, -1, true),
            ("e", 800, 2, false),
        ] {
            let mut edited = witness.clone();
            edited.get_mut(name).unwrap()[0]
                .set_coefficient(index, value)
                .unwrap();
            assert_eq!(
                compiled.map_witness(&edited).is_ok(),
                admitted,
                "{name}[{index}] = {value}"
            );
        }
        let mut other = instance(SEED, None);
        other.tag_t[M_TAG - 1] = (other.tag_t[M_TAG - 1] + 1) % PT;
        let other = statement(&other, Exact).0.compile(params).unwrap();
        assert_eq!(other.map_witness(&witness).err(), Some(Error::Witness));
    }
}

/// The fitted parameters of the `Own` form at `degree`.
fn params_for_own(degree: usize) -> TboxParams {
    params(&statement(&instance(SEED, None), Own).0, degree, Own).1
}

/// Prove at `degree`, verify the encoding and round-trip it, and reject another context,
/// public seed, tag or $`v`$. The size must be within [0.9, 1.05] of the parameter estimate:
/// the encodings are variable length.
fn prove_and_verify(degree: usize, moduli: Moduli) {
    let inst = instance(SEED, None);
    let (st, _) = statement(&inst, moduli);
    let (_, params) = params(&st, degree, moduli);
    let estimate = params.check().unwrap().estimated_proof_bytes;
    let compiled = st.compile(params.clone()).unwrap();
    let witness = witness_map(&inst, moduli);
    let bytes = compiled
        .prove_bytes_with_seed([1; 32], &witness, CTX, [2; 32])
        .unwrap();
    compiled.verify_bytes([1; 32], &bytes, CTX).unwrap();
    let scheme = Abdlop::new([1; 32], params.clone()).unwrap();
    let decoded = jali::codec::proof::decode(&scheme, &bytes).unwrap();
    assert_eq!(
        jali::codec::proof::encode(&scheme, &decoded).unwrap(),
        bytes
    );
    let ratio = bytes.len() as f64 / estimate as f64;
    assert!(
        (0.9..=1.05).contains(&ratio),
        "{} bytes, estimate {estimate}",
        bytes.len()
    );
    assert!(compiled.verify_bytes([1; 32], &bytes, b"other").is_err());
    assert!(compiled.verify_bytes([3; 32], &bytes, CTX).is_err());
    let mut tag = instance(SEED, None);
    tag.tag_t[M_TAG - 1] = (tag.tag_t[M_TAG - 1] + 1) % PT;
    let tag = statement(&tag, moduli).0.compile(params.clone()).unwrap();
    assert_eq!(tag.map_witness(&witness).err(), Some(Error::Witness));
    assert!(tag.verify_bytes([1; 32], &bytes, CTX).is_err());
    // Another v changes the integer bounds, and so may need more carry range or modulus than
    // the fitted set has: take the first single-coefficient change that compiles under it.
    let v = (0..64)
        .find_map(|i| {
            let mut other = instance(SEED, None);
            other.v[i] = (other.v[i] + 1) % QC;
            statement(&other, moduli).0.compile(params.clone()).ok()
        })
        .expect("a changed v that the parameters admit");
    assert!(v.verify_bytes([1; 32], &bytes, CTX).is_err());
}

#[test]
#[ignore = "tag-preimage proof at degree 128: minutes"]
fn tag_preimage_proof_degree_128() {
    prove_and_verify(128, Own);
}

#[test]
#[ignore = "tag-preimage proof at degree 64: minutes"]
fn tag_preimage_proof_degree_64() {
    prove_and_verify(64, Own);
}

/// The CRT form at degree 128, for comparison with the form with its own moduli.
#[test]
#[ignore = "tag-preimage proof, CRT form, at degree 128: minutes"]
fn tag_preimage_crt_proof_degree_128() {
    prove_and_verify(128, Crt);
}

#[test]
#[ignore = "tag-preimage proof, ell_inf form, at degree 128: minutes"]
fn tag_preimage_linf_proof_degree_128() {
    prove_and_verify(128, Linf);
}

#[test]
#[ignore = "tag-preimage proof, ell_inf form, at degree 64: minutes"]
fn tag_preimage_linf_proof_degree_64() {
    prove_and_verify(64, Linf);
}

#[test]
#[ignore = "tag-preimage proof, exact ell_inf form, at degree 128: minutes"]
fn tag_preimage_linf_exact_proof_degree_128() {
    prove_and_verify(128, Exact);
}

#[test]
#[ignore = "tag-preimage proof, exact ell_inf form, at degree 64: minutes"]
fn tag_preimage_linf_exact_proof_degree_64() {
    prove_and_verify(64, Exact);
}
