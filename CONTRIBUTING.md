# Contributing

Jali is research code, and changes to its protocol, transcripts or parameters need the same care
as the code that is there: say what a change does to soundness, zero-knowledge and the pinned
values, and test the edge cases, not only the comfortable middle.

## Checks

The CI workflow (`.github/workflows/ci.yml`) runs these; run them locally before a pull request.

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo clippy --locked --all-targets --no-default-features -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --locked --all-features --no-deps
cargo test --locked --all-features
cargo test --locked --no-default-features
cargo test --locked --release --all-features
# crypto-bigint's 32-bit limbs on a 64-bit host:
RUSTFLAGS='--cfg cpubits="32"' cargo test --locked --release --all-features
# The library for 32-bit and wasm targets (a check only):
cargo check --locked --target i686-unknown-linux-gnu --all-features
cargo check --locked --target wasm32-unknown-unknown --no-default-features
cargo deny check
```

`cargo docs` builds the API documentation with the KaTeX header, as docs.rs does. Set
`RAYON_NUM_THREADS` to limit the threads of the `parallel` feature; outputs do not depend on it.

## Slow tests

Ten tests are ignored in ordinary runs; `.github/workflows/slow.yml` runs them nightly. They are
the proofs of `tests/tag_preimage.rs` in its four forms, the dense maps of every block
combination (`tests/block_combinations.rs`), the extended NTT oracle
(`ntt_ten_thousand_pairs_per_prime`) and the Gaussian sampler against its reference on 100,000
coefficients per width. Each takes up to minutes in a release build.

```sh
cargo test --locked --release --all-features -- --ignored
cargo test --locked --release --all-features --test tag_preimage -- \
  --ignored tag_preimage_proof_degree_128
```

## Oracles and known-answer vectors

Several modules keep a slow reference implementation next to the fast one and test that both
give the same outputs and read the same bytes (the `oracle_tests` and `*_tests` modules under
`src/`). `tests/arithmetic.rs` compares the RNS arithmetic with num-bigint schoolbook products.
`kat/primitives.json` holds known-answer vectors written by an independent Python implementation
of the streams, samplers, rejection decisions and codecs (`tools/kat/generate.py`); a larger run
of the rejection oracle is

```sh
python3 tools/kat/generate.py --count 100000 --output primitives-100k.json
JALI_KAT_PATH=primitives-100k.json cargo test --locked --release \
  fixed_point_rejection_matches_300_bit_mpmath_including_boundaries
```

## Pinned values

`src/golden_tests.rs` pins the parameter transcripts, scheme fingerprints, opening proofs and
ring arithmetic; `tests/statement.rs`, `tests/threads.rs` and `tests/param_sets.rs` pin seeded
proofs; `src/rand/gauss/table_tests.rs` pins the base Gaussian table, which only
`tools/kat/half_gaussian_cdf.py` writes. A change to any of them changes what verifiers accept
or what a seed produces, so update a pin only with the change that explains it. Every parameter
field enters the transcript, the `estimator` string included, so parameter metadata changes the
pins too.

## Benchmarks

```sh
cargo bench --bench arithmetic
cargo bench --bench protocols
cargo run --release --features serde --example possession
```

The protocol benchmarks use fixed seeds and so measure one restart path; they do not estimate
mean rejection counts. For comparisons on a loaded machine, retired instructions are steadier
than wall time.

## Parameter tool

The Python tools under `tools/` have their own tests and regeneration checks; see
[tools/README.md](tools/README.md). A change to the crate's parameter checks needs
`python3 tools/params/rust_reference.py` to record Rust's decisions again, and a change to the
tool needs `python3 tools/params/lnp_params.py regenerate`.
