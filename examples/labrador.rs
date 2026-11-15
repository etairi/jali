//! A LaBRADOR proof of knowledge of a short witness for random dot-product constraints.
//!
//! `cargo run --release --example labrador -- [r] [n] [bound]` proves knowledge of `r` vectors
//! of `n` polynomials of degree 64 with coefficients in `[-bound, bound]` (default 8, 512, 1)
//! that satisfy three random linear constraints over the ring, one quadratic constraint, and a
//! constant-term constraint that states an integer relation: the sum of the coefficients of
//! the first vector. It prints the witness size, the proof size and the times.
//!
//! The proof is not zero knowledge: it reveals information about the witness.
use jali::labrador::{Constraint, Policy, Rq, Statement, Witness, prove, verify};
use std::time::Instant;

/// SplitMix64, for example data only.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, m: u64) -> u64 {
        ((u128::from(self.next()) * u128::from(m)) >> 64) as u64
    }
}

/// The product of two polynomials modulo X^64 + 1 and q, by schoolbook multiplication.
fn mul(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
    let d = a.len();
    let mut acc = vec![0i128; d];
    for i in 0..d {
        for j in 0..d {
            let p = i128::from(a[i]) * i128::from(b[j]);
            if i + j < d {
                acc[i + j] += p;
            } else {
                acc[i + j - d] -= p;
            }
        }
    }
    acc.iter()
        .map(|x| x.rem_euclid(i128::from(q)) as u64)
        .collect()
}

/// The sum of the products of the polynomials of two vectors.
fn inner(a: &[u64], b: &[u64], q: u64, d: usize) -> Vec<u64> {
    let mut out = vec![0u64; d];
    for (x, y) in a.chunks_exact(d).zip(b.chunks_exact(d)) {
        for (o, v) in out.iter_mut().zip(mul(x, y, q)) {
            *o = ((u128::from(*o) + u128::from(v)) % u128::from(q)) as u64;
        }
    }
    out
}

/// sigma_{-1}: X -> X^{-1}, coefficients (a_0, -a_{d-1}, ..., -a_1).
fn sigma(a: &[u64], q: u64) -> Vec<u64> {
    let d = a.len();
    (0..d)
        .map(|i| if i == 0 { a[0] } else { (q - a[d - i]) % q })
        .collect()
}

fn main() -> Result<(), jali::Error> {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("integer"))
        .collect();
    let (r, n, bound) = (
        args.first().copied().unwrap_or(8),
        args.get(1).copied().unwrap_or(512),
        args.get(2).copied().unwrap_or(1) as u64,
    );
    let (q, d) = (4294967197u64, 64usize); // 2^32 - 99, prime and 5 mod 8
    let ring = Rq::new(q, d)?;
    let mut rng = Rng(1);

    let coefficients: Vec<Vec<i64>> = (0..r)
        .map(|_| {
            (0..n * d)
                .map(|_| rng.below(2 * bound + 1) as i64 - bound as i64)
                .collect()
        })
        .collect();
    let canonical: Vec<Vec<u64>> = coefficients
        .iter()
        .map(|v| v.iter().map(|x| x.rem_euclid(q as i64) as u64).collect())
        .collect();
    let norm: u128 = coefficients.iter().flatten().map(|x| (x * x) as u128).sum();
    let witness = Witness::new(&ring, coefficients.clone());
    let mut statement = Statement::new(ring.clone(), r, n, norm)?;

    // Three random linear constraints over the ring on the first vectors.
    for k in 0..3 {
        let i = k % r;
        let phi: Vec<u64> = (0..n * d).map(|_| rng.below(q)).collect();
        let b = inner(&phi, &canonical[i], q, d);
        statement.push(Constraint::new(b).linear(i, 0, phi))?;
    }
    // One quadratic constraint: a <s_0, s_last> = b.
    let a: Vec<u64> = (0..d).map(|_| rng.below(q)).collect();
    let b = mul(&a, &inner(&canonical[0], &canonical[r - 1], q, d), q);
    statement.push(Constraint::new(b).quadratic(0, r - 1, a))?;
    // An integer relation through a constant coefficient: ct(<sigma(1), s_0>) is the sum of the
    // coefficients of s_0. (A quadratic integer relation, such as the dot product of two
    // vectors, needs sigma of one of them as another witness vector, tied to it by linear
    // constraints.)
    let ones = vec![1u64; n * d];
    let phi: Vec<u64> = ones.chunks_exact(d).flat_map(|p| sigma(p, q)).collect();
    let sum: i64 = coefficients[0].iter().sum();
    let mut b = vec![0u64; d];
    b[0] = sum.rem_euclid(q as i64) as u64;
    statement.push_constant_term(Constraint::new(b).linear(0, 0, phi))?;
    statement.check(&witness)?;

    let policy = Policy::default();
    let start = Instant::now();
    let proof = prove(&statement, &witness, &[0u8; 32], b"example", &policy)?;
    let proving = start.elapsed();
    let start = Instant::now();
    verify(&statement, &proof, &[0u8; 32], b"example", &policy)?;
    let verifying = start.elapsed();
    let bits = 64 - (2 * bound).leading_zeros();
    println!(
        "witness: {r} x {n} polynomials, {} coefficients in [-{bound}, {bound}] ({} bytes at {bits} bits each)",
        r * n * d,
        (r * n * d * bits as usize).div_ceil(8)
    );
    println!("proof: {} bytes", proof.len());
    println!(
        "prove: {:.0} ms, verify: {:.0} ms",
        proving.as_secs_f64() * 1e3,
        verifying.as_secs_f64() * 1e3
    );
    Ok(())
}
