use criterion::{Criterion, criterion_group, criterion_main};
use jali::{
    abdlop::Abdlop,
    lnp::{AffineBlock, L2Block, Statement},
    math::{Poly, PolyMat, PolyVec, SparsePolyMat},
    params::toy_d64,
    quad::{self, QuadEq},
    quad_eval, quad_many, tbox,
};
use std::{hint::black_box, time::Duration};

fn protocols(c: &mut Criterion) {
    let p = toy_d64();
    let scheme = Abdlop::new([1; 32], p.clone()).unwrap();
    let ring = scheme.ring().clone();
    let one = Poly::constant(ring.clone(), 1);
    let s = PolyVec::new(ring.clone(), vec![one.clone(); scheme.bounded_len()]).unwrap();
    let m = PolyVec::new(ring.clone(), vec![Poly::constant(ring.clone(), 2); 2]).unwrap();
    let (commitment, opening) = scheme
        .commit_with_seed(s.clone(), m.clone(), [2; 32])
        .unwrap();
    let proof = scheme
        .prove_with_seed(&commitment, &opening, b"benchmark", [3; 32])
        .unwrap();
    eprintln!(
        "toy_d64: ppseed=[1;32], commit_seed=[2;32], proof_seed=[3;32], opening proof={} bytes",
        scheme.encode_proof(&proof).unwrap().len()
    );
    let mut group = c.benchmark_group("protocols");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    group.bench_function("abdlop/commit", |b| {
        b.iter(|| {
            scheme
                .commit_with_seed(s.clone(), m.clone(), black_box([2; 32]))
                .unwrap()
        })
    });
    group.bench_function("abdlop/prove", |b| {
        b.iter(|| {
            scheme
                .prove_with_seed(&commitment, &opening, b"benchmark", black_box([3; 32]))
                .unwrap()
        })
    });
    group.bench_function("abdlop/verify", |b| {
        b.iter(|| {
            scheme
                .verify(&commitment, black_box(&proof), b"benchmark")
                .unwrap()
        })
    });
    let dim = 2 * (scheme.bounded_len() + scheme.message_len());
    let mut eq = QuadEq::zero(ring.clone(), dim).unwrap();
    eq.r2 = SparsePolyMat::new(ring.clone(), dim, vec![(0, 2, one.clone())]).unwrap();
    eq.r0 = one.neg();
    let proof =
        quad::prove_with_seed(&scheme, &commitment, &opening, &eq, b"benchmark", [3; 32]).unwrap();
    group.bench_function("quad/prove", |b| {
        b.iter(|| {
            quad::prove_with_seed(
                &scheme,
                &commitment,
                &opening,
                &eq,
                b"benchmark",
                black_box([3; 32]),
            )
            .unwrap()
        })
    });
    group.bench_function("quad/verify", |b| {
        b.iter(|| quad::verify(&scheme, &commitment, &eq, black_box(&proof), b"benchmark").unwrap())
    });
    let eqs: Vec<_> = (0..16).map(|i| eq.scale(&one.rotate(i)).unwrap()).collect();
    let proof =
        quad_many::prove_with_seed(&scheme, &commitment, &opening, &eqs, b"benchmark", [3; 32])
            .unwrap();
    group.bench_function("quad-many-16/prove", |b| {
        b.iter(|| {
            quad_many::prove_with_seed(
                &scheme,
                &commitment,
                &opening,
                &eqs,
                b"benchmark",
                black_box([3; 32]),
            )
            .unwrap()
        })
    });
    group.bench_function("quad-many-16/verify", |b| {
        b.iter(|| {
            quad_many::verify(&scheme, &commitment, &eqs, black_box(&proof), b"benchmark").unwrap()
        })
    });
    let mut eval = eq.clone();
    eval.r0 = eval.r0.add(&one.rotate(1)).unwrap();
    let evals = vec![eval; 16];
    let proof = quad_eval::prove_with_seed(
        &scheme,
        &commitment,
        &opening,
        &eqs,
        &evals,
        b"benchmark",
        [3; 32],
    )
    .unwrap();
    group.bench_function("quad-eval-16/prove", |b| {
        b.iter(|| {
            quad_eval::prove_with_seed(
                &scheme,
                &commitment,
                &opening,
                &eqs,
                &evals,
                b"benchmark",
                black_box([3; 32]),
            )
            .unwrap()
        })
    });
    group.bench_function("quad-eval-16/verify", |b| {
        b.iter(|| {
            quad_eval::verify(
                &scheme,
                &commitment,
                &eqs,
                &evals,
                black_box(&proof),
                b"benchmark",
            )
            .unwrap()
        })
    });
    let select = |cols, indices: &[usize]| {
        let mut entries = vec![Poly::zero(ring.clone()); cols * indices.len()];
        for (row, col) in indices.iter().enumerate() {
            entries[row * cols + col] = one.clone();
        }
        PolyMat::new(ring.clone(), indices.len(), cols, entries).unwrap()
    };
    let sblock = |indices: &[usize]| AffineBlock {
        rows: indices.len(),
        s: Some(select(p.m1, indices)),
        m: None,
        offset: None,
    };
    let statement = Statement {
        quadratic: vec![],
        evaluation: vec![],
        binary: Some(sblock(&[0, 1])),
        l2: vec![
            L2Block {
                map: sblock(&[2, 3]),
                bound_squared: 128,
            },
            L2Block {
                map: sblock(&[4]),
                bound_squared: 64,
            },
        ],
        arp: Some(AffineBlock {
            rows: 2,
            s: None,
            m: Some(select(2, &[0, 1])),
            offset: None,
        }),
    };
    let bounded = PolyVec::new(ring, s.entries()[..p.m1].to_vec()).unwrap();
    let proof =
        tbox::prove_with_seed(&scheme, &statement, &bounded, &m, b"benchmark", [3; 32]).unwrap();
    eprintln!(
        "toolbox: actual={} bytes, estimate={} bytes",
        jali::codec::proof::encode(&scheme, &proof).unwrap().len(),
        p.check().unwrap().estimated_proof_bytes
    );
    group.bench_function("toolbox/prove", |b| {
        b.iter(|| {
            tbox::prove_with_seed(
                &scheme,
                &statement,
                &bounded,
                &m,
                b"benchmark",
                black_box([3; 32]),
            )
            .unwrap()
        })
    });
    group.bench_function("toolbox/verify", |b| {
        b.iter(|| tbox::verify(&scheme, &statement, black_box(&proof), b"benchmark").unwrap())
    });
    group.finish();
}
criterion_group!(benches, protocols);
criterion_main!(benches);
