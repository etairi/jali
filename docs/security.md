# Security notes

Jali is a research implementation of the LNP22 proof system (Lyubashevsky, Nguyen and Plançon,
CRYPTO 2022, ePrint 2022/284). This page collects what it does and does not guarantee. To report
a vulnerability, see [SECURITY.md](../SECURITY.md).

## Implementation

- **Not audited.** Neither the code nor the parameters have had an independent review.
- **Variable time.** Arithmetic, sampling, rejection and decoding run in variable time, so
  timing and memory access may depend on secret values. There is no constant-time guarantee.
- **Zeroization is best effort.** Owned polynomial, transform, witness and random-state buffers
  are wiped on drop, and so are seeds, once received, and the keys derived from them; the
  caller's own copy of a seed is the caller's to wipe. Integer copies and compiler-generated
  temporaries may remain on the stack.
- **No `unsafe` code** (`#![forbid(unsafe_code)]`) and no C dependencies in the library.
- **Formats are unstable.** The API, the proof encoding and the transcripts may change between
  versions, and proofs do not interoperate with other implementations.

## Randomness and seeds

`commit`, `prove` and `prove_bytes` draw a fresh 32-byte seed per call from a
`rand_core::CryptoRng`. The `_with_seed` variants take the seed from the caller, and it must be
secret and uniformly random. Each call derives its keys from the seed and all of its inputs
([seeds](protocol.md#seeds)), so a reused seed does not reuse masks across different inputs,
assuming that SHAKE128, keyed by the seed, and AES-256 behave as pseudorandom functions; this
argument has not been written out as a proof. Identical calls give identical outputs, which
makes them linkable: with a reused seed, zero-knowledge holds only relative to the equality
pattern of the inputs. A deterministic prover is also exposed to fault attacks, since a fault
that changes a challenge but not the key can give two accepted responses with the same masks,
and from them the witness. Use fresh seeds wherever multi-theorem zero-knowledge is needed or
faults can be injected.

## Fiat–Shamir and extraction

The proofs are non-interactive through the Fiat–Shamir transform: every challenge is a SHAKE128
hash of the transcript so far, application context included, and Jali keeps the paper's order of
messages and challenges. Knowledge soundness rests on extraction by rewinding.

- LNP22 analyses the Fiat–Shamir version of its protocol in Appendix B.3. Theorem B.7 (p. 77)
  proves knowledge soundness in the random-oracle model, for a prover making $`Q`$ oracle
  queries, for one relation: knowledge of $`s_1`$ with $`Ps_1=u`$ and $`\|s_1\|=B`$. Its
  extractor rewinds the prover at each of the four challenges, in the framework of Attema, Fehr
  and Klooß for multi-round protocols (ePrint 2021/1377), runs in expected time polynomial in
  the statement size and $`Q`$, and outputs a witness or an MSIS solution.
- For the full toolbox with binary, approximate-range, quadratic and evaluation equations,
  compressed openings and bimodal signs, and for the statement compiler on top of it, no
  Fiat–Shamir proof is written down; knowledge soundness is the argument above carried over,
  which has not been checked step by step. The compiler's own soundness arguments are in
  [protocol](protocol.md#soundness-arguments-of-the-compiler).
- LNP22 obtains non-interactive zero-knowledge from the (non-abort) honest-verifier
  zero-knowledge of the interactive protocol (Appendix B). Jali relies on the same argument,
  which has not been written out for its full toolbox either.
- Jali claims no straight-line (online) extraction, implements no transform that would provide
  it, and claims no extraction in the presence of simulated proofs, nor consistent extraction of
  a witness shared by separately proved relations.
- Security against quantum adversaries in the quantum random-oracle model has not been analysed.

The extraction tests in `src/abdlop.rs` and `src/quad/extraction_tests.rs` fork proofs at
injected challenges and check the algebra these arguments rely on; they do not establish the
theorems.

## Two-prime moduli

The range proofs conclude that a committed sign is $`\pm1`$ from $`b^2=1`$ in $`\mathbb Z_q`$,
which needs $`\mathbb Z_q`$ to be a field (LNP22, proof of Prop. 5.1). For $`q=q_1q_2`$ the CRT
value $`(1\bmod q_1,-1\bmod q_2)`$ also squares to one, and with it a long committed vector
projects like a short one. `TboxParams::check` therefore refuses binary, exact-norm and range
blocks over a two-prime modulus; `src/tbox/two_prime_forgery_tests.rs` shows that, with only
that refusal switched off, the verifier accepts forged proofs of each kind. Two-prime moduli
remain available for the opening, quadratic and evaluation proofs.

## Deviations from LNP22

- **No $`\eta`$ filter on challenges.** LNP22 §2.7 restricts the challenge space to challenges
  whose operator-norm bound is at most $`\eta`$; Jali samples from the whole
  $`\sigma_{-1}`$-stable set and rejects none, so about 1% of challenges exceed the $`\eta`$
  that the rejection widths assume ([parameters](parameters.md#challenges)). No claim is made
  that every challenge meets it.
- **Heuristic projection bound.** The approximate range proof's rejection constant uses
  $`\|Re\|^2\le337\|e\|^2`$, which LNP22 Lemma 2.8 derives under the heuristic substitution of
  the binomial projection by a normal one.
- **Gaussian sampling.** Masks are drawn at widths $`1.55\cdot2^t`$ from a 128-bit table of the
  half-Gaussian of width 1.55, within statistical distance $`2^{-126.2}`$ of it, combined with
  an exact rational Bernoulli test; the normalizers differ from their mean by a relative
  $`2^{-67.4}`$ at most (both figures computed). Rejection sampling uses the exact rational
  variance and 192-bit fixed-point exponentials, and rounds rejection constants up to integers.
- **Approximate $`\ell_\infty`$ bounds are relaxed.** A `Norm::Linf(β)` variable is proven only
  within the extraction bound $`E=2\cdot`$`z4_bound`, not within $`\beta`$; `Norm::LinfExact`
  proves $`\beta`$ exactly ([protocol](protocol.md#bounds-in-ell_infty)).
- **Verifier bound of the approximate range proof.** The verifier accepts responses up to
  $`16\sigma_4`$ in $`\ell_\infty`$, so the extraction bound is $`2\lfloor16\sigma_4\rfloor`$;
  the compiler's lifting check, $`q>2(F_j+28\sigma_4p_j)`$, covers it through its factor 2.

## Parameters

- The shipped sets meet root Hermite factors of at most 1.0044, about 101 bits under core-SVP,
  not 128 ([parameters](parameters.md#hardness)). All hardness figures are cost-model
  heuristics; hiding relies on Extended-MLWE being as hard as MLWE.
- Prime factors above $`2^{64}`$ pass the Baillie–PSW probable-prime test, which is not a proof
  of primality; the crate checks no primality certificate.
- The MSIS estimate, the rejection constants and the size estimate are computed in `f64`, not in
  certified interval arithmetic. Integer norms, sampler decisions and lifting bounds are exact.
- `TboxParams::check` validates equations and supplied hardness metadata, not the truth of a
  caller's lattice estimate.
