//! With the `parallel` feature, proofs do not depend on the number of threads: each layer
//! proves in pools of 1, 2, 3, 4 and 8 threads, with the bytes of the pool of one thread (whose
//! single thread runs the attempts and rows one at a time, as without the feature), and
//! verifies in each pool. The possession proof also keeps its pinned fingerprint.
#![cfg(feature = "parallel")]
use jali::{
    abdlop::Abdlop,
    lnp::{AffineBlock, L2Block, Statement as Toolbox},
    math::{Poly, PolyMat, PolyVec, Ring, SparsePolyMat, SparsePolyVec},
    params::toy_d64,
    quad::{self, QuadEq},
    quad_eval, quad_many,
    statement::{Norm, Statement},
    tbox,
};
use std::{collections::BTreeMap, fmt::Debug, sync::Arc};

/// The value of `f` in pools of 1, 2, 3, 4 and 8 threads, which must all be equal.
fn in_pools<T: PartialEq + Debug + Send>(what: &str, f: impl Fn() -> T + Send + Sync) -> T {
    let mut first: Option<T> = None;
    for threads in [1, 2, 3, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let value = pool.install(&f);
        match &first {
            None => first = Some(value),
            Some(first) => assert_eq!(&value, first, "{what}: {threads} threads"),
        }
    }
    first.unwrap()
}

fn shake(bytes: &[u8]) -> String {
    use shake::{ExtendableOutput, Shake128, Update, XofReader};
    let mut hash = Shake128::default();
    hash.update(bytes);
    let mut digest = [0u8; 32];
    hash.finalize_xof().read(&mut digest);
    hex::encode(digest)
}

#[test]
fn opening_quadratic_many_and_evaluation_proofs_do_not_depend_on_the_thread_count() {
    let scheme = Abdlop::new([1; 32], toy_d64()).unwrap();
    let ring = scheme.ring().clone();
    let s1 = PolyVec::new(
        ring.clone(),
        (0..scheme.bounded_len())
            .map(|i| Poly::constant(ring.clone(), (i % 3) as i128))
            .collect(),
    )
    .unwrap();
    let m = PolyVec::new(
        ring.clone(),
        vec![
            Poly::constant(ring.clone(), 7),
            Poly::constant(ring.clone(), 11),
        ],
    )
    .unwrap();
    let witness = quad::interleave(&s1, &m).unwrap();
    let dim = witness.len();
    let one = Poly::constant(ring.clone(), 1);
    // The equation of tests/quadratic.rs.
    let mut equation = QuadEq {
        r2: SparsePolyMat::new(
            ring.clone(),
            dim,
            vec![
                (0, 2, one.clone()),
                (2, 3, one.rotate(1)),
                (0, 25, one.neg()),
            ],
        )
        .unwrap(),
        r1: SparsePolyVec::new(ring.clone(), dim, vec![(0, one.clone()), (24, one.clone())])
            .unwrap(),
        r0: Poly::zero(ring.clone()),
    };
    equation.r0 = equation.evaluate(&witness).unwrap().neg();
    let equations = vec![equation.clone(), equation.scale(&one.rotate(3)).unwrap()];
    let mut eval = equation.clone();
    eval.r0 = eval.r0.add(&one.rotate(1)).unwrap();
    let (commitment, opening) = scheme.commit_with_seed(s1, m, [2; 32]).unwrap();
    for seed in [3u8, 4, 5] {
        let proof = in_pools("opening", || {
            scheme
                .prove_with_seed(&commitment, &opening, b"t", [seed; 32])
                .unwrap()
        });
        in_pools("opening verify", || {
            scheme.verify(&commitment, &proof, b"t")
        })
        .unwrap();
        let proof = in_pools("quadratic", || {
            quad::prove_with_seed(&scheme, &commitment, &opening, &equation, b"t", [seed; 32])
                .unwrap()
        });
        in_pools("quadratic verify", || {
            quad::verify(&scheme, &commitment, &equation, &proof, b"t")
        })
        .unwrap();
        let proof = in_pools("many", || {
            quad_many::prove_with_seed(&scheme, &commitment, &opening, &equations, b"t", [seed; 32])
                .unwrap()
        });
        in_pools("many verify", || {
            quad_many::verify(&scheme, &commitment, &equations, &proof, b"t")
        })
        .unwrap();
        let evals = [eval.clone()];
        let proof = in_pools("evaluation", || {
            quad_eval::prove_with_seed(
                &scheme,
                &commitment,
                &opening,
                &equations,
                &evals,
                b"t",
                [seed; 32],
            )
            .unwrap()
        });
        in_pools("evaluation verify", || {
            quad_eval::verify(&scheme, &commitment, &equations, &evals, &proof, b"t")
        })
        .unwrap();
    }
}

fn selector(ring: &Arc<Ring>, columns: usize, indices: &[usize]) -> PolyMat {
    PolyMat::new(
        ring.clone(),
        indices.len(),
        columns,
        indices
            .iter()
            .flat_map(|i| {
                (0..columns).map(move |j| Poly::constant(ring.clone(), i128::from(*i == j)))
            })
            .collect(),
    )
    .unwrap()
}

#[test]
fn toolbox_proofs_do_not_depend_on_the_thread_count() {
    // The statement and witness of tests/toolbox.rs: binary, two exact and an approximate block.
    let scheme = Abdlop::new([1; 32], toy_d64()).unwrap();
    let ring = scheme.ring().clone();
    let select_s = |indices: &[usize]| AffineBlock {
        rows: indices.len(),
        s: Some(selector(&ring, 10, indices)),
        m: None,
        offset: None,
    };
    let statement = Toolbox {
        quadratic: vec![],
        evaluation: vec![],
        binary: Some(select_s(&[0, 1])),
        l2: vec![
            L2Block {
                map: select_s(&[2, 3]),
                bound_squared: 128,
            },
            L2Block {
                map: select_s(&[4]),
                bound_squared: 64,
            },
        ],
        arp: Some(AffineBlock {
            rows: 2,
            s: None,
            m: Some(selector(&ring, 2, &[0, 1])),
            offset: None,
        }),
    };
    let s1 = PolyVec::new(
        ring.clone(),
        (0..10)
            .map(|i| {
                Poly::new(
                    ring.clone(),
                    (0..64).map(|j| ((i + j) % 2) as i128).collect(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let m = PolyVec::new(
        ring.clone(),
        vec![
            Poly::constant(ring.clone(), 2),
            Poly::constant(ring.clone(), -3),
        ],
    )
    .unwrap();
    for seed in [2u8, 3] {
        let bytes = in_pools("toolbox", || {
            let proof =
                tbox::prove_with_seed(&scheme, &statement, &s1, &m, b"t", [seed; 32]).unwrap();
            jali::codec::proof::encode(&scheme, &proof).unwrap()
        });
        let proof = jali::codec::proof::decode(&scheme, &bytes).unwrap();
        in_pools("toolbox verify", || {
            tbox::verify(&scheme, &statement, &proof, b"t")
        })
        .unwrap();
    }
}

#[test]
fn the_possession_proof_keeps_its_fingerprint_in_every_pool() {
    // The parameters, statement and witness of tests/statement.rs.
    let mut p = toy_d64();
    p.id = "quadratic-possession-test-only".into();
    p.m1 = 20;
    p.m2 = 55;
    p.n_msis = 15;
    p.l = 2;
    p.alpha_squared = 192;
    p.n_bin = 0;
    p.l2_rows = vec![16, 4];
    p.n_prime = 2;
    p.linf_bound = 13;
    p.log_sigma = [14, 12, 10, 13];
    p.d_bits = 6;
    let source = Ring::new(13, 128).unwrap();
    let mut statement = Statement::new(source.clone());
    statement.var("s", 8, Norm::L2Squared(128)).unwrap();
    statement.var("x", 2, Norm::L2Squared(64)).unwrap();
    let constant = |c| {
        statement
            .constant(Poly::constant(source.clone(), c))
            .unwrap()
    };
    let (one, five, minus_one) = (constant(1), constant(5), Poly::constant(source.clone(), -1));
    let s = statement.variable("s", 0).unwrap();
    let x = statement.variable("x", 0).unwrap();
    let left = s.add(&one.scale(&minus_one).unwrap()).unwrap();
    let right = five.add(&x.scale(&minus_one).unwrap()).unwrap();
    let equation = left
        .product_affine(&right)
        .unwrap()
        .add(&one.scale(&minus_one).unwrap())
        .unwrap();
    statement.eq_mod_p(equation).unwrap();
    let compiled = statement.compile(p).unwrap();
    let mut s = vec![Poly::zero(source.clone()); 8];
    s[0] = Poly::constant(source.clone(), 5);
    let mut x = vec![Poly::zero(source.clone()); 2];
    x[0] = Poly::constant(source.clone(), -5);
    let witness = BTreeMap::from([("s".to_string(), s), ("x".to_string(), x)]);
    let bytes = in_pools("possession", || {
        compiled
            .prove_bytes_with_seed([33; 32], &witness, b"example/possession", [44; 32])
            .unwrap()
    });
    assert_eq!(bytes.len(), 18395);
    assert_eq!(
        shake(&bytes),
        "c7811a80e70d3f1653090b9f14f80a2436fb3fac1244b30821a41b443c7245a0"
    );
    in_pools("possession verify", || {
        compiled.verify_bytes([33; 32], &bytes, b"example/possession")
    })
    .unwrap();
}
