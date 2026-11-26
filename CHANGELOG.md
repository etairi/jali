# Changelog

All notable changes to this crate are listed here. The crate follows [semantic
versioning](https://semver.org); before 1.0, a minor version may break the API.

## [0.3.4] - 2026-11-20

Proofs and transcripts are those of 0.3.3, in both modes and with every feature (checked on
aarch64 and x86-64).

### Changed

- **`simd` is a default feature.** Its kernels are chosen at run time and fall back to the scalar
  code on processors without the instructions, and the environment variable `JALI_SCALAR` keeps
  the scalar code everywhere. With `default-features = false` the crate compiles no `unsafe`
  code.
- **Faster, with byte-identical proofs.**
  - LNP22: the range proofs' projections sum the signed entries straight from the packed matrix
    bytes, in `i64` where their bound allows, the public matrices take every entry's stream
    from one AES-256 key schedule and keep sampled words without a branch, the verifier weights
    its projection rows below $`2^{55}`$ from a table of the 81 sign patterns of four entries,
    substitutions transform each constant of an affine map once, and the uniform sampler and
    the bit writer read and write whole words.
  - The statement compiler's exact norms and the transcript's residues modulo two-word moduli
    are computed in machine words instead of wide integers.
  - SHAKE128 is the crate's own (FIPS 202), over the `keccak` crate's permutation, so that the
    permutation can suit the processor. `keccak` is now a dependency and `shake` only a
    dev-dependency.
  - LaBRADOR: the projection's column sums take a table of the 81 sign patterns, in split
    64-bit lanes from $`2^{58}`$, the witness projection runs in 8-bit lanes, the amortized
    opening and $`\varphi_z`$ combine the challenges in the transform domain, products are
    reconstructed from the transform blocks without copies, the public matrices decode their
    residues in runs and keep sampled words without a branch, the prover reuses the transforms
    of its statement check and of $`\varphi`$, the planner computes the garbage's
    decompositions once per level, and challenges are rejected early by an exact lower bound
    on their moment, with the same decisions.
  - With `simd` on x86-64: a Keccak permutation compiled for BMI1 and BMI2, AVX-512 kernels for
    the centred residues, lazy sums and CRT of the RNS products and for LaBRADOR's ring
    arithmetic, projection and column sums (AVX-512BW, VL and VBMI), and with AVX-512 IFMA the
    transforms, products and CRT of every LaBRADOR ring (the fewest of four primes below
    $`2^{51}`$ whose product meets the ring's bound) and of the LNP22 rings that need no more
    of these primes than of the 62-bit ones. Each kernel returns the scalar code's outputs,
    which tests compare.

  On one thread of an Apple M4, an end-to-end lattice test with $`q\approx2^{69.4}`$ needs 19.4%
  fewer instructions than with 0.3.3, 15.6% in the straight-line extractable mode, LNP22 provers
  1.07 to 1.26 times fewer and their verifiers 1.09 to 1.53 times, and LaBRADOR proves with 1.28
  to 1.30 times fewer and verifies with 1.25 to 1.26 times fewer at $`q\approx2^{32}`$, 1.19 to
  1.20 and 1.08 to 1.10 times fewer at $`q\approx2^{61}`$. On one thread of an Intel Xeon
  Platinum 8581C the default build, now with `simd`, takes 38.4% fewer cycles than 0.3.3's for
  the lattice test (36.9% extractable), proves LNP22 statements 1.64 to 2.16 times and verifies
  them 1.40 to 2.03 times faster, and LaBRADOR proves 6.05 to 6.43 times and verifies 4.74 to
  4.93 times faster at $`q\approx2^{32}`$, 4.75 to 4.98 and 3.55 to 3.60 times at
  $`q\approx2^{61}`$. Without `simd` the lattice test takes 17.0% fewer cycles there (13.0%
  extractable), LNP22 proves 1.07 to 1.26 times and verifies 1.06 to 1.44 times faster, and
  LaBRADOR 1.24 to 1.27 and 1.15 to 1.17 times at $`q\approx2^{32}`$, 1.25 and 1.08 to 1.11
  times at $`q\approx2^{61}`$. The lattice test's peak memory is unchanged.

## [0.3.3] - 2026-11-19

Proofs and transcripts are those of 0.3.2, in both modes and with every feature (checked on
aarch64 and x86-64).

### Added

- **The `simd` feature**: AVX-512 versions of the NTT on x86-64, chosen at run time when the
  processor has AVX-512F and AVX-512DQ and the environment variable `JALI_SCALAR` is unset.
  They return the scalar code's outputs, which a test compares for every NTT prime at every
  degree from 16 to 1024. Their module is the only code that may use `unsafe`, and without the
  feature, or off x86-64, the crate forbids `unsafe` code as before.

### Changed

- **Faster, with byte-identical proofs.**
  - Polynomials keep their coefficients in the words of the ring's arithmetic, one word below
    $`2^{63}`$, two words up to $`2^{126}`$ and 256-bit values above, instead of 256-bit values
    for every modulus. `Poly::coefficients` still returns `&[U256]`, now a copy made on its
    first call and dropped when the coefficients change.
  - LaBRADOR sums the projection columns of moduli from $`2^{58}`$ in fixed-size accumulators
    reduced by Shoup multiplications instead of 128-bit divisions, and its lifts, scaled vector
    operations and proof codes take fewer divisions.
  - The LNP22 norm and range proofs weight their projection rows by table lookups from 128
    columns on, the transcript encodes runs of uniform coefficients with one width computation,
    and the Gaussian sampler's Bernoulli step encloses the exponential with five Taylor terms
    instead of thirteen.
  - On aarch64 the NTT runs two stages per pass. On x86 the Gaussian masks read their keystream
    in batches of 1 KiB, the batch of VAES-512.

  On one thread of an Apple M4, an end-to-end lattice test with $`q\approx2^{69.4}`$ needs 17.4%
  fewer instructions, 15.6% in the straight-line extractable mode, LNP22 provers 1.11 to 1.27
  times fewer and their verifiers 1.28 to 1.58 times, and LaBRADOR 1.09 to 1.13 times fewer at
  $`q\approx2^{32}`$ and 1.40 to 1.69 times at $`q\approx2^{61}`$. On one thread of an Intel Xeon
  Platinum 8581C the lattice test takes 18% fewer cycles in both modes (27% with `simd`), and
  its peak memory falls from 237 to 137 MiB (301 to 175 MiB in the extractable mode). There
  LNP22 provers run 1.20 to 1.30 times faster (1.39 to 1.52 with `simd`), their verifiers 1.26 to
  1.50 times (1.43 to 1.71), and LaBRADOR 1.10 to 1.27 times at $`q\approx2^{61}`$ (1.27 to 1.47).
  At $`q\approx2^{32}`$ LaBRADOR proves within 1% of 0.3.2 there and verifies up to 1.8% slower,
  and with `simd` it runs 1.11 to 1.16 times faster.
- Documentation: the notes of 0.3.0 and some test comments describe the statements they
  measured instead of naming the system the statements came from.

## [0.3.2] - 2026-11-17

### Changed

- Documentation only: the notes of 0.3.1 and a comment in the ring's tests are reworded. The
  code is that of 0.3.1.

## [0.3.1] - 2026-11-17

### Changed

- **LaBRADOR moduli up to $`2^{62}`$** (from $`2^{48}`$). The norms of witnesses and last
  witnesses are summed with saturation (squares of centred 61-bit residues reach $`2^{122}`$, so
  sums of more than 64 of them could wrap in release builds), the projection condition
  computes $`30q^2`$ without overflow, and the projection table holds centred residues (at most
  $`q/2`$ in magnitude, where its sums of 4 weighted entries reached $`4q`$), so that the
  column sums of 64 table entries stay in `i64` below $`2^{58}`$, and from $`2^{58}`$ they are
  taken in `i128`. The projection constraints only take residues of these sums, so they do not
  change. The ring's tests run at three more moduli (53.3, 56.4 and 61.1 bits) with an oracle
  that reduces each product.
- **A bounded base search in the level planner.** It tried every odd base up to four times
  its heuristic one, which takes minutes for witnesses with large coefficients. It still does
  when there are at most 160 of them (so those levels choose what they chose), and otherwise
  tries 24 bases spread geometrically and 12 more between the best one's neighbours.
- **Parallel aggregation and statement checks** with `parallel`: the aggregation sums the
  linear terms vector by vector, and the check evaluates the constraints apart. Results do not
  depend on the threads.

### Added

- `labrador::predicted_proof_bytes`: the proof size the planner predicts for a statement's
  shape and bound.

## [0.3.0] - 2026-11-16

The proofs of the LNP22 modules and their transcripts are those of 0.2.0, in both modes.

### Added

- **LaBRADOR** (`labrador`): succinct proofs of knowledge of short witnesses that satisfy
  dot-product constraints over $`\mathbb Z_q[X]/(X^{64}+1)`$, after Beullens and Seiler
  (ePrint 2022/1341). **Not zero knowledge.** The main protocol of the paper, recursed, with the
  last witness sent in the clear, in a radix code and a Gaussian code close to its entropy. The
  prover chooses each level's parameters from the witness, among them the base of the
  amortized opening, and sends them, and the verifier checks the conditions of the paper's
  Theorem 5.1 for them, with a Module-SIS policy of block size at least 439 by default.
  `docs/labrador.md` describes the protocol, its encoding and its departures from the paper,
  and `examples/labrador.rs` runs it: proofs of 57,856 bytes for 262,144 ternary coefficients
  and 71,621 bytes for 4,194,304. The proofs of the other modules and their transcripts are
  unchanged. The new transcript tag is `JALI-V02-LABRADOR-C1-SHAKE128`.

### Changed

- **Faster proving and verification, with byte-identical proofs.** Every proof and transcript
  is unchanged. Measured on an Apple M4, one thread, against the first 0.3.0 commit:
  - Moduli from $`2^{63}`$ to $`2^{126}`$ use native 128-bit arithmetic: Barrett products,
    Shoup multipliers, residues and the CRT in words, instead of `crypto-bigint`'s generic
    division. An end-to-end lattice test whose statements have $`q\approx2^{69.4}`$ needs 6.2
    times fewer cycles, its two largest proofs about 7 times less time.
  - The prover's witness checks compute only the constant coefficients they test, from witness
    products shared by all equations. The exact integer evaluation behind lifted constraints
    uses 64-bit coefficients when a bound shows that nothing overflows. NTT plans are built once
    per process. The transforms of RNS products are merged negacyclic ones with lazy
    reduction. Scalar products are fused, and the quadratic proof computes the witness-only
    half of its products once instead of once per attempt. Kyber1024 proofs need 1.8 to 2.1
    times fewer cycles, their verification 1.1 to 1.2 times.
  - LaBRADOR multiplies by sparse challenges without 128-bit divisions and uses the faster
    transforms: 1.3 times fewer cycles to prove, 1.35 times to verify. Then it sums its
    projections in 32-bit lanes and reads the aggregation sums of all challenges with one table
    lookup per projection byte: another 1.6 times fewer instructions to prove and 1.4 to 1.5
    times to verify.
- **Less memory, with byte-identical proofs.** The toolbox produces the evaluation equations of
  its projections in chunks of 16 rows, which the evaluation proof combines as they arrive,
  instead of holding the 256 rows of each family at once. The statement's equations move into
  the evaluation proof instead of being copied, public matrices share their entries between
  clones, and the selectors and approximate blocks of a compiled statement hold only their
  nonzero entries. In that end-to-end lattice test the peak live heap of each step falls 2.1 to
  4.2 times (the largest proof's from 601 to 225 MiB), and the memory that Kyber1024 proofs and
  their verification allocate 6.3 to 8.5 times.
- **Faster again, with byte-identical proofs.** Measured on an Apple M4, one thread, against the
  0.3.0 work before these changes (commit fdb6435):
  - The evaluation proof combines the 256 projection rows of a family with the weights of each
    challenge row before projecting them, so it projects once per challenge row instead of
    once per projection row.
  - The fold of the many-equation proof sums each key's products in the transform domain and
    reconstructs each key once, products of transforms are added as 128-bit integers and
    reduced once per 15 products, and the CRT reconstructs a whole polynomial at a time. The
    quadratic proof transforms its equation's coefficients once per proof instead of once per
    attempt, and products of affine forms insert their keys by merging sorted lists.
  - The AES-256-CTR keystream is buffered, squeezed values below $`2^{64}`$ are reduced by
    Barrett's method, and LinHC's two key products share one transform of their vector.

  The end-to-end lattice test needs 1.88 times fewer cycles, 1.79 times in the extractable
  mode, Kyber1024 proofs 2.7 to 3.0 times fewer and their verification 4.9 to 5.2 times fewer.
  The provers of that test hold 1.5% to 2.8% more memory, the transformed quadratic
  coefficients.
- **LaBRADOR, faster and parallel, with byte-identical proofs.** The prover keeps the
  polynomials of the aggregated projection constraints between their two uses when they take
  at most 64 MiB, instead of streaming the projection a third time. The projection is read in
  blocks of a seekable stream, which the `parallel` feature processes at once. On one thread
  the prover is 1.21 times faster for 2^18 and 2^20 witness coefficients and 1.04 times for
  2^22. With `parallel` on the same machine (4 performance and 6 efficiency cores, shared with
  other jobs) the prover is 1.8 to 1.9 times faster and the verifier 1.9 to 2.4 times, in wall
  time. The prover holds 1.2 MiB more for 2^18 coefficients and 7 MiB more for 2^20.

## [0.2.0] - 2026-11-15

The Fiat–Shamir transcripts and the proof encoding follow the CFRG drafts
draft-irtf-cfrg-fiat-shamir-04 and, for the final round, the compact flavor of
draft-irtf-cfrg-sigma-protocols-04. Every proof changes, in both modes: proofs of 0.1 do not
verify under 0.2, nor the reverse. The parameter sets, the extraction sets, the commitments of
`commit_with_seed` and the scheme fingerprints are unchanged.

### Changed

- **Transcript.** One SHAKE128 duplex sponge per proof (FS draft §4), started from
  `DeriveSessionID` of a tag that names the argument, the codec and the hash suite
  (`JALI-V02-LNP22-TBOX-C1-SHAKE128` and five others) followed by the caller's context. The
  instance (parameters, public seed, statement, and the extraction set in the extractable mode)
  is absorbed once, first, and then each prover message as its serialization, and each verifier
  message is squeezed with a length that its type fixes. The inner layers continue the toolbox's
  transcript instead of starting their own, and data that the instance and the messages
  determine, such as the folded and projected equations, is no longer absorbed.
- **Decodings.** $`\Gamma`$, $`\mu`$ and the LinHC key by `DecodeUint` ($`N_s+16`$ squeezed
  bytes per value), the projection matrices by the centred binomial decoding of FIPS 203 from 2
  bits per entry, and the challenge by `rand::challenge_from_bytes` from a fixed 4032 bytes, with
  a fallback to the constant 1 of probability at most $`2^{-208}`$. The projection matrices, the
  folding weights and the LinHC key are no longer expanded from seeds with AES-256-CTR or
  SHAKE128.
- **Codec C1.** Every field of a proof has a fixed width, in FIPS 204's bit packing: Gaussian
  responses in $`t+4`$ bits, hints in `CheckedParams::hint_width` bits (2 for every shipped
  set), the challenge as its first $`d/2`$ coefficients. A proof is the concatenation of its
  messages, each a whole number of bytes, then the challenge and the response, with no end bit.
  Every proof of a parameter set has the same length, about 3.3% more than 0.1's measured plain
  proofs and 1.9% more for extractable ones (`kyber1024_d64`: 21,004 bytes, was 20,318, and
  120,964 extractable, was 118,669). The prover restarts an attempt whose responses or hints do
  not fit their widths, about 0.1% of the opening proof's attempts and 0.01% of the range
  attempts (computed from the Gaussian tails, not measured).
- **Cost.** Proving takes 0.73 to 0.81 and verifying 0.69 to 0.79 times the retired
  instructions of 0.1.1 for the shipped sets' Kyber-shape instance, plain and extractable
  (means over seeds 1 to 100, one thread, Apple M4). The derived equations are no longer encoded
  and hashed at every layer: in 0.1.1, SHAKE128, the equation encoder and the AES expansion of
  the projection rows took about a quarter of the sampled time, in 0.2 about 1.5%.
- `CheckedParams::estimated_proof_bytes` is the exact proof length, and the parameter tool's
  mirror computes it exactly.
- Key labels `abdlop/opening-proof/v3`, `tbox/proof/v3`, `quad-eval/garbage/v2`,
  `tbox/proof-sl/v2` and `abdlop/opening-proof-sl/v2`. The keys bind the transcript through 32
  bytes squeezed from a copy of the sponge.
- `BitReader` reads fields of up to 128 bits at once.

### Added

- `rand::challenge_from_bytes` and `rand::CHALLENGE_BYTES`, `ExtractionParams::proof_bytes`, and
  `CheckedParams::hint_width`. `BitWriter::signed`, `BitWriter::into_bytes`, `BitReader::signed`
  and `BitReader::finish_bytes` for fixed-width messages.
- Tests: the draft's SHAKE128 vectors and its sumcheck example (`src/fiat_shamir.rs`), the IO
  pattern and the NARG layout of every verifier as recorded call by call
  (`tbox::io_pattern_tests`), and a Python oracle of the challenge decoding
  (`tools/kat/generate.py`). The draft's vectors are vendored as JSON in `kat/ietf/`.

### Removed

- The public module `transcript` and its chained 32-byte `Transcript`. The transcript is
  internal, and its format is specified in `docs/encodings.md`.
- The variable-length Gaussian and hint codes are no longer used by any proof. They remain on
  `BitWriter` and `BitReader`.

## [0.1.1] - 2026-11-14

A straight-line extractable proof mode, with Katsumata's transform (CRYPTO 2021, full version
ePrint 2021/927). Plain proofs, their transcripts and encodings, the parameter sets and the
existing API are unchanged.

### Added

- **Extractable mode.** `tbox::prove_extractable_with_seed` and `tbox::verify_extractable`, the
  `prove_extractable` and `verify_extractable` methods of `lnp::Prover`, `lnp::Verifier` and
  `statement::Compiled` with their seeded and byte forms, the proof type `tbox::ExtractableProof`
  and its strict codec (`codec::proof::encode_extractable`, `decode_extractable`).
  - An extractable linear homomorphic commitment, the MLWE construction of the paper's Sec. 3.4,
    commits to the final opening's witness beside the opening proof. Its key is a random-oracle
    output on the parameters, the extraction set, the context and the statement, so an extractor
    that holds a trapdoor recovers the witness from one proof, without rewinding.
  - The mode has its own transcript tag (`LNP22-toolbox-sl-v1`) and key labels
    (`tbox/proof-sl/v1`, `abdlop/opening-proof-sl/v1`), so it reads no stream of the plain mode.
  - The paper's constants are corrected in two places: the shift bound uses the challenges'
    operator norm, and each block of the opening has its own norm bound ([security
    notes](docs/security.md#straight-line-extraction)).
- **Extraction sets.** `params::ExtractionParams` and `params::OracleModel`, with
  `ExtractionParams::check`, `modulus_for` and, with `serde`, `from_json`, which check a set
  against its parameter set and derive the commitment's moduli and widths. The extraction sets of
  the shipped sets: `kyber1024_d64_extraction`, `kyber1024_d128_extraction`,
  `demo_d64_extraction` and `toy_d64_extraction` (`src/params/sets/*.extraction.json`).
- **Parameter tool.** `lnp_params.py extraction` derives an extraction set: the commitment's
  parameters exactly as the crate derives them, and the least rank that a policy accepts at the
  modulus that rank gives. `extraction --shipped --check` checks the shipped ones, in CI too.
- `examples/extractable.rs`, extractable benchmarks in `benches/protocols.rs`, and tests:
  decryption of the commitment at an aligned worst-case opening and at its failure boundary,
  straight-line extraction of the committed witness from one proof with a trapdoored key, the
  codec's strictness, and the sizes of the shipped sets' extractable proofs.

### Changed

- The parameter tool hashes `jali_params/extraction.py` with its other files, so its reports
  record a new tool hash. The sets they describe are unchanged.

### Notes

- For the shipped sets an extractable proof is about 5.7 to 5.9 times as long as a plain one,
  proving takes 1.17 to 1.49 times the retired instructions and verifying 1.14 to 1.17 times
  (seeds 1 to 20, one thread, Apple M4) [measured].
- The extraction argument for the whole toolbox is an outline, in the classical random-oracle
  model, and no quantum random-oracle security is claimed. The [security
  notes](docs/security.md#straight-line-extraction) give each step with its status.

## [0.1.0] - 2026-11-02

First release: the LNP22 proof system (Lyubashevsky, Nguyen and Plançon, CRYPTO 2022, ePrint
2022/284), with ABDLOP commitments, quadratic and evaluation proofs, binary, exact-norm and
approximate range proofs, a statement compiler for relations over other rings and moduli,
parameter checks and four parameter sets.

### Security

- **Gaussian sampler.** Masks of width $`1.55\cdot2^t`$ are drawn, for $`t\ge1`$, as
  $`2^{t-1}k-u`$ from a 256-bit table of the half-Gaussian of width 3.1 and the Bernoulli test
  (exact rational exponent, 192-bit cutoff) at the same width exponent, and for $`t=0`$ as
  $`\pm K`$ from a 256-bit table of width 1.55, with no offset. Per coefficient, the relative
  error of the normalizers falls from $`2^{-67.4}`$ to $`2^{-272.6}`$ (0 for $`t\le1`$) and the
  statistical distance of a table draw from $`2^{-126.2}`$ to $`2^{-252.4}`$; $`2^{14}`$
  coefficients are within $`2^{-171.6}`$ of ideal Gaussians, against $`2^{-60.9}`$ before, and
  $`2^{64}`$ times as many within $`2^{-139.6}`$ (computed; [security
  notes](docs/security.md#gaussian-sampling)). These figures concern the masks only; whole
  proofs are covered by the next item. Widths, variances, parameter sets and encodings are
  unchanged.
- **256-bit rejection coins.** Every rejection test compares a uniform 256-bit coin
  (`reject::coin`) with its acceptance probability, which is then resolved to $`2^{-256}`$ instead
  of $`2^{-128}`$. This covers Rej_1 and Rej_2 of the opening proof, also inside the quadratic,
  evaluation and toolbox proofs, and the bimodal tests of the range proofs.
  - The 192-bit exponentials are now the limit of the tests: coins and cutoffs together come
    to at most $`2^{-172.6}`$ per proof for the shipped sets. With 128-bit coins, the coins
    alone allowed up to about $`2^{-124}`$.
  - Per proof, a whole proof is within $`2^{-151.2}`$ of an ideal prover's output for
    `kyber1024_d64`, $`2^{-165.8}`$ for `toy_d64`, $`2^{-170.3}`$ for `kyber1024_d128` and
    $`2^{-170.0}`$ for `demo_d64` (computed; [security
    notes](docs/security.md#statistical-distance-of-whole-proofs)). The ideal responses follow
    the laws of LNP22 Lemma 2.14 conditioned on the verifier's norm bounds, on which Jali's
    prover restarts; against the unconditioned laws, every shipped set is within
    $`2^{-129.3}`$ per proof. The tail of Rej_1 or the masks dominate.
  - $`Q`$ proofs of one witness, with fresh seeds, are within $`Q`$ times that: $`2^{-87.2}`$
    for $`2^{64}`$ proofs with `kyber1024_d64`. This is an upper bound; whether $`2^{64}`$
    proofs are within $`2^{-128}`$ is not established.
  - The tail of Rej_1 is at most $`2^{-129}`$ for every parameter set that `check` accepts (up
    to the `f64` evaluation of $`M_1`$), and smaller for the shipped sets, whose $`M_1`$ is
    rounded up.
- **Challenges within $`\eta`$.** Every challenge meets the operator-norm bound $`\eta`$ of
  LNP22 §2.7, which the parameter analysis assumes: `rand::challenge` draws again from the same
  stream until $`\|(\sigma_{-1}(c)c)^{32}\|_1\le\eta^{64}`$ (an exact integer test, at most 64
  draws), and the verifier accepts only the challenge that it derives the same way. About 0.74%
  of the draws at degree 64 and 1.16% at degree 128 are drawn again (measured;
  [parameters](docs/parameters.md#challenges)). The test accepts as soon as a lower power of
  $`\sigma_{-1}(c)c`$ shows the bound, with the same decisions; a derivation takes about 6 µs
  at degree 64 and 20 µs at degree 128 (measured).
- **Versioned proof keys.** The key labels of the opening proof and the toolbox are
  `abdlop/opening-proof/v2` and `tbox/proof/v2`, so that a seed used with both samplers does
  not give correlated masks, nor coins read at other offsets
  ([security notes](docs/security.md#randomness-and-seeds)).

### Added

- `statement::Norm::L2SquaredRange { min, max }`: a two-sided exact Euclidean bound
  $`\mathrm{min}\le\|s\|^2\le\mathrm{max}`$, $`1\le\mathrm{min}\le\mathrm{max}`$, for example
  $`s\neq0`$ with `min = 1`. The upper bound is the exact-norm proof of `Norm::L2Squared(max)`;
  the compiler adds one binary proof-ring polynomial in the Ajtai part and one evaluation
  equation for the lower bound ([protocol notes](docs/protocol.md#two-sided-euclidean-bounds)),
  and `Compiled::extraction_bound` reports `Extraction::L2SquaredRange`. In JSON,
  `{"l2_squared_range": {"min": a, "max": b}}`, which the parameter tool reads. Statements
  without such a block compile, prove and encode exactly as before, and every parameter set,
  fixture and pinned proof is unchanged. `Norm` and `Extraction` gain a variant each, which an
  exhaustive `match` outside the crate must cover.

### Changed

- The toolbox's transcript tag is `LNP22-toolbox-v3`. Masks and challenges change, and with
  them every proof, including the commitment inside a toolbox proof, whose randomness comes from
  the toolbox key and which commits to the range masks. Commitments made with the
  `Abdlop::commit` methods, parameter transcripts, scheme fingerprints and opening-prefix
  digests do not change. The known-answer vectors (now with challenge vectors), the table digest
  and the pinned proofs are regenerated.
- New public items: `rand::challenge`, `rand::within_eta`, `rand::eta_norm_power` and
  `rand::MAX_CHALLENGE_DRAWS`. `rand::challenge` refuses an $`\omega`$ with
  $`(d-1)\omega>2^{15}`$, the capacity of the exact test, before it reads the stream.
- `reject::accept` takes its coin as a `U256`, uniform in $`[0,2^{256})`$, instead of a `u128`,
  and compares exact products in 1024 bits. New public items: `reject::COIN_BYTES` (32) and
  `reject::coin`, which reads a coin little-endian.
- Coin layout. Attempt $`a`$ of the opening proof reads bytes $`64a`$ to $`64a+63`$ of its coin
  stream: Rej_1's coin, then Rej_2's. Each range test of the toolbox reads one 32-byte coin.
  Coins decide differently, so the accepted attempts change, and with them the proofs.
- Regenerated with the coins:
  - the rejection vectors of `kat/primitives.json`, on the same inputs. A random coin keeps the
    earlier 128-bit coin as its high half; a boundary coin lies $`2^{112}`$ below, or
    $`2^{100}`$ or $`2^{112}`$ above, the exact threshold, so a 128-bit test would fail the
    vectors above. The other vectors do not change;
  - the pinned proofs, their lengths and the measured sizes of `docs/parameters.md`;
  - the prover seeds that the tests found by search, and the Rust reference data of the
    parameter tool.
- Equation encodings in bounded buffers. The transcripts no longer build the encoding of an
  equation (`QuadEq::to_bytes`) in one buffer before hashing it. They write it into chunks of
  at most 32 KiB, which never grow (the last one is shrunk to its length), and SHAKE128 absorbs
  it piece by piece. The zero codes of a zero polynomial are held as a count once they span at
  least 64 whole bytes. The hashed bytes are those of `QuadEq::to_bytes`, so transcripts,
  challenges and proofs do not change, and the pinned proofs pass unchanged. Before, each
  encoding grew by doubling in one `Vec<u8>`. In one proof of an application's degree-64
  statement with 18 quadratic and 297 evaluation equations (measured outside this
  repository), the prover encoded 2181 equations, 666 MiB in all, 61 % of it zero runs; 72 of
  its buffers grew beyond 2 MiB, up to 36 MiB, and the macOS allocator was observed to keep
  freed blocks of that size charged to the process. The encodings that the toolbox keeps across
  its range attempts took about 39 MiB there instead of 164 MiB. The lists that hold a family's
  encodings take 40 bytes per equation. `QuadEq::to_bytes` returns the same bytes as before and
  no longer collects the entries first.
- Selector matrices at their exact size. `Statement::compile` builds the 0/1 matrices that
  select the binary and Euclidean blocks from the witness in one buffer of exactly their number
  of entries. Collected from their flattened rows, the buffer grew by doubling: for a binary
  block of 140 polynomials over 140 bounded ones, to room for 35,840 entries for its 19,600, or
  1,146,880 bytes for 627,200 on 64-bit targets. The matrices, and so the proofs, do not change.
- `crypto-bigint` 0.7.10 or later. Every release from 0.7.0 to 0.7.8 is yanked, and 0.7.6
  fixed carry and timing bugs. Transcripts, proofs and known-answer vectors do not change.
