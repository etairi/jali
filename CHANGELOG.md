# Changelog

All notable changes to this crate are listed here. The crate follows [semantic
versioning](https://semver.org); before 1.0, a minor version may break the API.

## [Unreleased]

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
