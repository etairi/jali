use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use crypto_bigint::U256;
use jali::{
    math::{Poly, PolyMat, PolyVec, Ring, SparsePolyMat, ntt::NttPlan},
    params::{moduli::NTT_PRIMES, toy_d64},
    rand::{
        AesPrg, autostable, challenge, domain, eta_norm_power, gaussian,
        reject::{self, Policy, Variance},
        within_eta,
    },
};
use std::hint::black_box;

fn arithmetic(c: &mut Criterion) {
    for d in [64, 128, 256] {
        let ring = Ring::new(1099511627917, d).unwrap();
        let a = Poly::new(ring.clone(), (0..d).map(|i| i as i128 - 31).collect()).unwrap();
        let b = a.rotate(17);
        eprintln!(
            "degree={d} q=1099511627917 RNS primes={} coefficient-buffer={} bytes",
            ring.rns_primes(),
            d * size_of::<U256>()
        );
        c.bench_function(&format!("negacyclic-product/{d}"), |bencher| {
            bencher.iter(|| black_box(&a).mul(black_box(&b)).unwrap())
        });
        // The first four primes, the ones rings below 2^115 use; the other five run the same
        // code at the same width.
        for (i, (p, root)) in NTT_PRIMES.into_iter().take(4).enumerate() {
            let mut psi = root;
            for _ in 0..(1024 / d).trailing_zeros() {
                psi = (u128::from(psi) * u128::from(psi) % u128::from(p)) as u64;
            }
            let plan = NttPlan::new(p, d, psi).unwrap();
            let values: Vec<u64> = (0..d).map(|j| (j * j) as u64).collect();
            c.bench_function(&format!("ntt-forward/{d}/prime-{i}"), |b| {
                b.iter_batched(
                    || values.clone(),
                    |mut x| plan.forward(black_box(&mut x)).unwrap(),
                    BatchSize::SmallInput,
                )
            });
            c.bench_function(&format!("ntt-inverse/{d}/prime-{i}"), |b| {
                b.iter_batched(
                    || values.clone(),
                    |mut x| plan.inverse(black_box(&mut x)).unwrap(),
                    BatchSize::SmallInput,
                )
            });
        }
    }
    // Wide moduli: 2^128 + 165 (5 RNS primes) and 2^240 + 325 (8), pseudorandom operands.
    for (label, q) in [
        (
            "q129",
            U256::ONE.shl_vartime(128).wrapping_add(&U256::from_u8(165)),
        ),
        (
            "q241",
            U256::ONE
                .shl_vartime(240)
                .wrapping_add(&U256::from_u16(325)),
        ),
    ] {
        for d in [64, 128] {
            let ring = Ring::with_modulus(q, d).unwrap();
            let mut state = 0x9e37_79b9_7f4a_7c15u64;
            let mut word = || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                state
            };
            let mut operand = || {
                Poly::from_u256(
                    ring.clone(),
                    (0..d)
                        .map(|_| {
                            let words = [word(), word(), word(), word()];
                            U256::from_le_slice(&words.map(u64::to_le_bytes).concat())
                        })
                        .collect(),
                )
                .unwrap()
            };
            let (a, b) = (operand(), operand());
            eprintln!(
                "degree={d} {label} RNS primes={} coefficient-buffer={} bytes",
                ring.rns_primes(),
                d * size_of::<U256>()
            );
            c.bench_function(&format!("negacyclic-product-{label}/{d}"), |bencher| {
                bencher.iter(|| black_box(&a).mul(black_box(&b)).unwrap())
            });
            c.bench_function(&format!("addition-{label}/{d}"), |bencher| {
                bencher.iter(|| black_box(&a).add(black_box(&b)).unwrap())
            });
        }
    }
    let p = toy_d64();
    let ring = Ring::with_modulus(p.check().unwrap().q, p.degree).unwrap();
    let one = Poly::constant(ring.clone(), 1);
    let columns = p.m2 - p.n_msis;
    let vector = PolyVec::new(ring.clone(), vec![one.clone(); columns]).unwrap();
    let matrix = PolyMat::new(
        ring.clone(),
        p.n_msis,
        columns,
        vec![one.clone(); columns * p.n_msis],
    )
    .unwrap();
    c.bench_function("dense-matvec/16x40", |b| {
        b.iter(|| matrix.mul(black_box(&vector)).unwrap())
    });
    let sparse = SparsePolyMat::new(
        ring.clone(),
        columns,
        (0..columns)
            .map(|i| (i as u16, i as u16, one.clone()))
            .collect(),
    )
    .unwrap();
    c.bench_function("sparse-bilinear/40-diagonal", |b| {
        b.iter(|| sparse.bilinear(black_box(&vector), &vector).unwrap())
    });
    let mut stream = AesPrg::new(&[7; 32], 0);
    for t in p.log_sigma {
        c.bench_function(&format!("gaussian/64/t-{t}"), |b| {
            b.iter(|| gaussian(black_box(&mut stream), t, 64).unwrap())
        });
    }
    // The challenge set's exact operator-norm test on one challenge (decided after the first
    // product at degree 64 and the second at degree 128 for these), the full norm power that a
    // draw near or above eta needs, and a whole derivation (draws until one is within eta,
    // about 1.01 draws), at both degrees.
    for (d, omega, eta) in [(64, 8, 140), (128, 2, 59)] {
        let ring = Ring::new(1099511627917, d).unwrap();
        let x = autostable(&mut AesPrg::new(&[1; 32], 0), ring.clone(), omega).unwrap();
        c.bench_function(&format!("challenge/within-eta/d{d}"), |b| {
            b.iter(|| within_eta(black_box(&x), eta).unwrap())
        });
        c.bench_function(&format!("challenge/norm-power/d{d}"), |b| {
            b.iter(|| eta_norm_power(black_box(&x)).unwrap())
        });
        let mut index = 0;
        c.bench_function(&format!("challenge/derive/d{d}"), |b| {
            b.iter(|| {
                index += 1;
                let mut stream = AesPrg::new(&[2; 32], domain(0, index));
                challenge(&mut stream, ring.clone(), omega, eta).unwrap()
            })
        });
    }
    let (dot, norm) = reject::moments(&[10, -3, 17], &[2, 1, 4]).unwrap();
    c.bench_function("rejection/bimodal", |b| {
        b.iter(|| {
            reject::accept(
                Policy::Bimodal,
                dot,
                norm,
                Variance::gaussian(0).unwrap(),
                U256::from(2u8).shl_vartime(128),
                // (2^256 - 1)/3.
                black_box(U256::from_be_hex(&"5".repeat(64))),
            )
            .unwrap()
        })
    });
}
criterion_group!(benches, arithmetic);
criterion_main!(benches);
