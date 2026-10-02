# Jali

Jali is a Rust library for lattice-based zero-knowledge proofs. It implements the proof system
of Lyubashevsky, Nguyen and Plançon (LNP22): ABDLOP commitments with compressed opening proofs,
proofs of quadratic relations with automorphisms, evaluation proofs, and the combined protocol
with binary, exact Euclidean-norm and approximate range proofs. On top of it, a statement
compiler turns relations over other polynomial rings and moduli into those proofs: named
variables with norm bounds, degree lowering, modulus lifting with carries, constraints modulo
several moduli, $`\ell_\infty`$ bounds and subring variables.

## Status

Jali is research code, version 0.1, not yet published on crates.io.

- **Not audited.** No independent review of the code or of its parameters has taken place.
- **No constant-time guarantee.** Arithmetic, sampling and rejection run in variable time, so
  timing may depend on secret values.
- **Unstable formats.** The API, the proof encoding and the transcripts may change between
  versions. Proofs are not interoperable with other implementations.
- **Randomness.** `commit`, `prove` and `prove_bytes` draw a fresh seed per call from a
  `rand_core` 0.10 `CryptoRng` (for example from `rand` 0.10); prefer them. The `_with_seed`
  variants take a 32-byte seed that must be secret and uniformly random. A seed may be reused:
  each call derives its keys from the seed and all of its inputs (assuming that SHAKE128 and
  AES-256 behave as pseudorandom functions). But identical calls return identical outputs: two
  commitments to the same values, or two proofs of the same statement, context and witness, are
  equal and therefore linkable. Use fresh seeds wherever multi-theorem zero-knowledge is needed
  and wherever faults can be injected (a fault that changes a challenge but not the key can
  reveal the witness).
- **Context binding.** Every proof takes an application `context: &[u8]` and verifies only under
  the same bytes.
- **Parameters.** The shipped parameter sets meet a root-Hermite-factor criterion
  ($`\delta\le1.0044`$, about 101 bits under core-SVP), not 128 bits. Hardness figures are
  heuristic estimates.

The [security notes](https://github.com/etairi/jali/blob/main/docs/security.md) give the
details, including what the Fiat–Shamir proofs do and do not guarantee.

## Installation

Until Jali is on crates.io, use it as a git or path dependency. It needs Rust 1.95 or later
(edition 2024).

```toml
[dependencies]
jali = { git = "https://github.com/etairi/jali", features = ["serde"] }
# For reproducible builds, pin a commit: rev = "<commit>".
# Or a local checkout: jali = { path = "../jali" }
```

The crate forbids `unsafe` code and has no C dependencies.

## Quick start

Commit to two vectors of polynomials and prove knowledge of an opening. The fixed seeds keep the
example reproducible; applications use `commit` and `prove` with a `rand_core` 0.10
`CryptoRng`.

```rust
use jali::{Error, abdlop::Abdlop, math::PolyVec, params::toy_d64};

fn main() -> Result<(), Error> {
    // The 32-byte public seed from which the public matrices are expanded.
    let scheme = Abdlop::new([1; 32], toy_d64())?;
    let ring = scheme.ring().clone();
    let bounded = PolyVec::zero(ring.clone(), scheme.bounded_len());
    let messages = PolyVec::zero(ring, scheme.message_len());
    let (commitment, opening) = scheme.commit_with_seed(bounded, messages, [2; 32])?;
    let proof = scheme.prove_with_seed(&commitment, &opening, b"example/opening", [3; 32])?;
    let bytes = scheme.encode_proof(&proof)?;
    scheme.verify(&commitment, &scheme.decode_proof(&bytes)?, b"example/opening")?;
    Ok(())
}
```

## Proving a statement

A `statement::Statement` declares named variables with norm bounds over a statement ring and
adds constraints built from affine and quadratic forms. `compile` checks the statement against a
parameter set and returns a `Compiled` statement that proves and verifies. This example proves
knowledge of a short $`w`$ with $`Aw+t=0`$ over $`\mathbb Z_{3329}[X]/(X^{256}+1)`$, with $`A`$
of size $`4\times8`$ and $`\|w\|^2\le2950`$, the shape of a Kyber-1024 key pair, for which the
crate ships parameters.

```rust
use jali::{
    Error,
    math::{Poly, Ring},
    params::kyber1024_d64,
    statement::{Norm, Statement},
};
use std::collections::BTreeMap;

fn main() -> Result<(), Error> {
    let ring = Ring::new(3329, 256)?;
    // A deterministic toy instance: ternary w, pseudorandom A, t = -Aw.
    let mut x = 1u64;
    let mut sample = |m: u64| {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((x >> 33) % m) as i128
    };
    let w = (0..8)
        .map(|_| Poly::new(ring.clone(), (0..256).map(|_| sample(3) - 1).collect()))
        .collect::<Result<Vec<_>, _>>()?;

    let mut statement = Statement::new(ring.clone());
    statement.var("w", 8, Norm::L2Squared(2950))?;
    for _ in 0..4 {
        let row = (0..8)
            .map(|_| Poly::new(ring.clone(), (0..256).map(|_| sample(3329)).collect()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut t = Poly::zero(ring.clone());
        for (a, wj) in row.iter().zip(&w) {
            t = t.sub(&a.mul(wj)?)?;
        }
        // The constraint t + sum_j A_ij w_j = 0 modulo 3329.
        let mut form = statement.constant(t)?;
        for (j, a) in row.iter().enumerate() {
            form = form.add(&statement.variable("w", j)?.scale(a)?)?;
        }
        statement.eq_mod_p(form)?;
    }

    let compiled = statement.compile(kyber1024_d64())?;
    let witness = BTreeMap::from([("w".to_string(), w)]);
    let public_seed = [7; 32];
    let proof = compiled.prove_bytes_with_seed(public_seed, &witness, b"example/kyber", [9; 32])?;
    compiled.verify_bytes(public_seed, &proof, b"example/kyber")?;
    Ok(())
}
```

Constraints can also be quadratic (`product_affine`), hold only for the constant coefficient
(`const_coeff_zero`), or use a modulus of their own (`variable_in`, `constant_in`). The
statement degree may exceed the proof degree (64 or 128); the compiler lowers it. For a
statement of a new shape, `Statement::requirements` exports what a parameter search needs; see
[parameter sets](#parameter-sets).
[`examples/possession.rs`](https://github.com/etairi/jali/blob/main/examples/possession.rs)
proves a quadratic relation modulo 13, lowered from degree 128 to 64, with a committed carry and
its own parameter set:

```sh
cargo run --release --features serde --example possession
```

## Features

| Feature | Purpose |
| --- | --- |
| `serde` | Parameter sets from JSON (`TboxParams::from_json`) and export of statement requirements. Proofs always use the binary codec. |
| `parallel` | Rayon parallelism inside proving and verifying. Outputs do not depend on the number of threads. |
| `test-utils` | Capture and replay of sampler randomness, for tests only. |

There are no default features.

## Supported parameters and limits

- **Statement rings** $`\mathbb Z_p[X]/(X^{d'}+1)`$: any modulus $`2\le p<2^{256}`$ and
  power-of-two degrees $`64\le d'\le1024`$. Each constraint may have its own modulus.
- **Proof rings**: degree 64 or 128, with a modulus $`q<2^{256}`$ of one or two prime factors
  $`\equiv5\pmod8`$. Binary, exact-norm and range proofs need a single prime. Factors below
  $`2^{64}`$ are proven prime (deterministic Miller–Rabin); larger factors pass the Baillie–PSW
  probable-prime test, which is not a proof of primality.
- **Norms**: exact squared Euclidean (`Norm::L2Squared`), binary, exact $`\ell_\infty`$ through
  bits (`Norm::LinfExact`), approximate $`\ell_\infty`$ (`Norm::Linf`, which a proof bounds only
  by the extraction bound of the approximate range proof), or unbounded. Bounds fit `u64`.
- **Capacities**: Gaussian width exponents up to 100, derived response bounds below $`2^{128}`$,
  encoded proofs up to 16 MiB. Violations return errors.
- **Arithmetic** is exact, with 62-bit RNS/NTT products over up to nine primes: two at
  $`q\approx2^{40}`$, eight or nine at $`q\approx2^{240}`$.

## Parameter sets

| Function | Relation | Proof degree | $`\log_2q`$ | Proof size |
| --- | --- | ---: | ---: | ---: |
| `params::kyber1024_d64` | $`Aw+t=0`$ over $`\mathbb Z_{3329}[X]/(X^{256}+1)`$, $`A\in R^{4\times8}`$, $`\|w\|^2\le2950`$ | 64 | 41.14 | about 20.3 kB |
| `params::kyber1024_d128` | the same | 128 | 41.14 | about 21.8 kB |
| `params::demo_d64` | the same shape with $`p=2^{32}-4607`$ and $`\|w\|^2\le2048`$, the size of the Module-LWE example of LNP22 §6.2 | 64 | 60.44 | about 24.0 kB |
| `params::toy_d64` | a small statement with every kind of block, for tests and examples | 64 | 40.00 | about 17.3 kB |

The first three were derived by the parameter tool for the worst case over public data, so they
compile every instance of their shape. Their hardness criterion is $`\delta\le1.0044`$ for MLWE
and MSIS, about 101 bits under core-SVP; the [parameter
notes](https://github.com/etairi/jali/blob/main/docs/parameters.md) give the estimates and
lattice-estimator cross-checks. Proof sizes vary by up to a few tens of bytes with the seed.

For a new statement, export `Statement::requirements` (feature `serde`), derive a set with the
Python [parameter tool](https://github.com/etairi/jali/blob/main/tools/README.md), load it with
`TboxParams::from_json`, which checks it, and compile. The tool is not part of the crate
package.

## Performance

Indicative timings from one benchmark session on an Apple M4 (4 performance and 6 efficiency
cores, 16 GiB), macOS 27.0.1, rustc 1.99.0, default release profile. Each figure is the median
wall-clock time of one call over seeds 1 to 3 and five rounds (15 proofs and 45 verifications
per row); `parallel` ran on two threads (`RAYON_NUM_THREADS=2`). The machine carried other load
(median one-minute load average 5.15), and the number of rejection-sampling restarts varies with
the seed and the statement, so single proofs can take much longer or shorter than the median.
`cargo bench --bench protocols` times the opening proof and the other protocols on `toy_d64` at
one fixed seed, and `cargo run --release --features serde --example possession` times one proof
of the example.

| Statement | Prove, 1 thread | Prove, `parallel` (2 threads) | Verify, 1 thread |
| --- | ---: | ---: | ---: |
| Opening proof (`toy_d64`) | 31.1 ms | 17.8 ms | 0.52 ms |
| Kyber-1024 shape (`kyber1024_d64`) | 284 ms | 214 ms | 126 ms |
| Kyber-1024 shape (`kyber1024_d128`) | 277 ms | 190 ms | 118 ms |
| 32-bit shape (`demo_d64`) | 420 ms | 291 ms | 155 ms |
| `examples/possession.rs` | 292 ms | 190 ms | 57.4 ms |

Verification of the compiled statements includes decoding the proof and expanding the public
matrices; that of the opening proof includes decoding only.

## Documentation

- [Protocol and statement compiler](https://github.com/etairi/jali/blob/main/docs/protocol.md)
- [Encodings, transcripts and key
  derivation](https://github.com/etairi/jali/blob/main/docs/encodings.md)
- [Parameters](https://github.com/etairi/jali/blob/main/docs/parameters.md)
- [Security notes](https://github.com/etairi/jali/blob/main/docs/security.md)
- [Parameter tool](https://github.com/etairi/jali/blob/main/tools/README.md)
- [Contributing: tests, oracles and
  benchmarks](https://github.com/etairi/jali/blob/main/CONTRIBUTING.md)

The API documentation writes formulas as KaTeX. `cargo doc` leaves them readable as LaTeX;
`cargo docs` builds the documentation with the KaTeX header in `docs/katex-header.html`, as
docs.rs does.

## References

- V. Lyubashevsky, N. K. Nguyen and M. Plançon. Lattice-Based Zero-Knowledge Proofs and
  Applications: Shorter, Simpler, and More General. CRYPTO 2022. Full version: Cryptology ePrint
  Archive, Paper 2022/284, revision of 14 August 2022, to which the section, figure and equation
  numbers in this crate refer.
- L. Ducas, A. Durmus, T. Lepoint and V. Lyubashevsky. Lattice Signatures and Bimodal Gaussians.
  CRYPTO 2013. (Bimodal rejection sampling.)
- L. Ducas, T. Lepoint, V. Lyubashevsky, P. Schwabe, G. Seiler and D. Stehlé. CRYSTALS –
  Dilithium: Digital Signatures from Module Lattices. Cryptology ePrint Archive, Report
  2017/633. (Compression with hints and the hint code, as LNP22 §6.1 uses them.)

## License

MIT; see [LICENSE](https://github.com/etairi/jali/blob/main/LICENSE).
