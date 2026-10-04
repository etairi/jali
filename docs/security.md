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

The keys carry a version for the same reason. The keys of the opening proof and of the toolbox
use the labels `abdlop/opening-proof/v2` and `tbox/proof/v2` since the 256-bit Gaussian sampler,
the 256-bit rejection coins and the challenges within $`\eta`$: the earlier version read the
same streams with another sampler, and masks that two samplers read from one stream are
correlated (the first mask coefficients are congruent modulo $`2^{t-1}`$), so that a proof from
each version, for the same seed and inputs, can reveal linear relations of the witness. Any
change to how proof randomness is read, such as the sampler, the coins or the challenge
derivation, must change these labels.

## Challenges

Every challenge of the opening proof, also inside the quadratic, evaluation and toolbox proofs,
lies in the challenge set of LNP22 §2.7: $`\sigma_{-1}`$-stable, with coefficients in
$`[-\omega,\omega]`$, coefficient $`d/2`$ zero, and the operator-norm bound
$`\|\sigma_{-1}(c^{32})c^{32}\|_1^{1/64}\le\eta`$ that the rejection widths, the verifier bound
and the MSIS bound of `TboxParams::check` assume. `rand::challenge` draws from the transcript's
challenge stream until a draw meets the bound, which it tests as
$`\|(\sigma_{-1}(c)c)^{32}\|_1\le\eta^{64}`$ in exact integer arithmetic (`rand::within_eta`; no
floating point, so every platform decides alike), and fails after 64 rejected draws. Prover and
verifier run it on the same stream. The verifier refuses a challenge of the wrong form (from
another ring, not $`\sigma_{-1}`$-stable, or with a coefficient above $`\omega`$ or a nonzero
coefficient $`d/2`$) before it recomputes the challenge, and then any challenge other than the
recomputed one, which is always within $`\eta`$; testing $`\eta`$ on the proof's challenge as
well would only repeat the test. About 0.74% of the draws at degree 64 and 1.16% at degree 128
exceed $`\eta`$ (measured; [parameters](parameters.md#challenges)). The challenge is uniform on
the set within the bound, which is smaller by those fractions, so knowledge errors of the form
$`k/|\mathcal C|`$ grow by a factor of about 1.007 at degree 64 and 1.012 at degree 128 (at most
1.008 and 1.013 at 95% confidence).

## Gaussian sampling

Masks of width $`\sigma=1.55\cdot2^t`$ are drawn (`rand::gaussian`), for $`t\ge1`$, as
$`2^{t-1}k-u`$ with $`u`$ uniform in $`[0,2^{t-1})`$ and $`k`$ from a random sign, a 256-bit
table of the half-Gaussian of width 3.1 and a Bernoulli test with an exact rational exponent and
a 192-bit cutoff, and for $`t=0`$ as $`\pm K`$ from a 256-bit table of the half-Gaussian of
width 1.55. Per coefficient, the distance from $`D_{\mathbb Z,\sigma}`$ has three sources
(computed with exact rationals and 2000-bit mpmath):

- the normalizers $`\rho_{3.1}(\mathbb Z-c)`$ differ from their mean by a relative
  $`2^{-272.6}`$ at most, and not at all for $`t\le1`$, where every offset is 0;
- a table draw is within statistical distance $`2^{-252.4}`$ (width 3.1) or $`2^{-253.6}`$
  (width 1.55) of its half-Gaussian, so a coefficient within $`2^{-251.2}`$ after rejection;
- the Bernoulli cutoff $`C`$ is within 12286 of $`2^{192}e^{-x}`$ (`rand::bernoulli`), a
  statistical distance of $`2^{-178.2}`$ at most per coefficient, but a relative error of the
  acceptance probabilities, of $`\chi^2`$ divergence $`2^{-356.2}`$ at most.

The relative errors add up as KL divergence, at most $`2^{-356.2}`$ per coefficient, by the
chain rule, and the tables' statistical distances by a union bound. Both arguments hold for
coefficients drawn one after another, as many as earlier outcomes call for (rejection sampling
makes the count random). With $`M`$ the expected number of coefficients drawn, rejected attempts
included, the masks are therefore within $`M\cdot2^{-251.2}+(M\cdot2^{-357.2})^{1/2}`$ of ideal
Gaussians (Pinsker): $`2^{-171.6}`$ for $`M=2^{14}`$, and $`2^{-139.6}`$ for $`2^{64}`$ times as
many, as in $`2^{64}`$ proofs with one witness. An exhaustive computation of the output law for
$`t\le3`$, over every offset and candidate with the actual tables and cutoffs, gives distances
of $`2^{-253.3}`$ ($`t=0`$) and $`2^{-191.4}`$ to $`2^{-191.8}`$ ($`t=1`$ to 3) from
$`D_{\mathbb Z,\sigma}`$. The previous sampler, with a 128-bit table of width 1.55 at every
$`t`$, had a relative normalizer error of $`2^{-67.4}`$, or $`2^{-60.9}`$ for $`M=2^{14}`$.

These figures concern the masks; the next section adds the rejection steps.

## Statistical distance of whole proofs

The bound below compares a proof, every challenge and response that the verifier reads, with
the output of an *ideal prover* for the same statement and witness. Its accepted responses
follow exactly the laws of LNP22 Lemma 2.14, $`z_1`$ Gaussian, $`z_2`$ Gaussian conditioned on
$`\langle z_2,cs_2\rangle\ge0`$ and the range responses Gaussian, all conditioned on the
verifier's norm bounds: like Jali's prover, it draws again when a response exceeds them
([deviations](#deviations-from-lnp22)). Exact acceptance probabilities alone do not give these
laws; items 2 and 6 below bound the difference. A simulator for them is LNP22's with one change:
it draws its responses again until they meet the bounds. As in LNP22, the challenge hash is a
random oracle, so that a challenge is uniform and independent of the masks, and the prover's
AES streams are taken as uniform bytes (a pseudorandomness assumption:
[randomness and seeds](#randomness-and-seeds)). The further steps to a simulator are LNP22's and
are not counted here: programming the oracle, the commitments under MLWE, and the sign of
$`\langle z_2,cs_2\rangle`$ under Extended-MLWE.

For one proof from a fresh seed, the bounds are as follows. They are upper bounds, computed in
interval arithmetic and rounded up, for a toolbox proof, that is its range round and its
evaluation proof. An opening, quadratic or evaluation proof alone is within the same bounds.

| Set | $`M_1`$ (formula) | Rej_1 tail | Coins and cutoffs | Masks | Proof | $`2^{64}`$ proofs |
| --- | --- | --- | --- | --- | --- | --- |
| `toy_d64` | 5 (4.14) | $`2^{-165.9}`$ | $`2^{-172.6}`$ | $`2^{-170.1}`$ | $`2^{-165.8}`$ | $`2^{-101.8}`$ |
| `kyber1024_d64` | 3 (2.76) | $`2^{-151.2}`$ | $`2^{-173.1}`$ | $`2^{-170.2}`$ | $`2^{-151.2}`$ | $`2^{-87.2}`$ |
| `kyber1024_d128` | 3 (2.37) | $`2^{-208.8}`$ | $`2^{-173.5}`$ | $`2^{-170.5}`$ | $`2^{-170.3}`$ | $`2^{-106.3}`$ |
| `demo_d64` | 3 (2.34) | $`2^{-216.2}`$ | $`2^{-173.1}`$ | $`2^{-170.2}`$ | $`2^{-170.0}`$ | $`2^{-106.0}`$ |
| `examples/possession.rs` | 4 (3.76) | $`2^{-141.5}`$ | $`2^{-172.8}`$ | $`2^{-170.1}`$ | $`2^{-141.5}`$ | $`2^{-77.5}`$ |

Proofs made with fresh seeds are independent, so $`Q`$ proofs of one witness are within
$`Q`$ times the bound for one proof (a hybrid over the proofs). The last column is this figure
for $`Q=2^{64}`$. Every shipped set therefore meets $`2^{-128}`$ per proof; for none does this
bound reach $`2^{-128}`$ over $`2^{64}`$ proofs (whether the actual distance does is not
established). The cutoffs and masks alone, added this way, come to about $`2^{-106}`$ over
$`2^{64}`$ proofs. Reaching $`2^{-128}`$ along these lines would take an $`M_1`$ with
$`\epsilon_1\le2^{-193}`$, the KL accounting of the previous section for the masks (at most
$`2^{-138.1}`$ for $`2^{64}`$ proofs of a shipped set), and a like accounting for the rejection
cutoffs, which is not derived.

Against LNP22's unconditioned laws, that is with LNP22's simulator used verbatim, add the
probability that an ideal response fails a norm bound: at most $`2^{-129.3}`$ per proof, nearly
all of it from the exact range block's bound $`\|z\|\le1.64\sqrt{256}s_3`$ (LNP22's
$`(te^{(1-t^2)/2})^{256}`$ at $`t=1.64`$). Every shipped set then stays within $`2^{-129.3}`$
per proof. The derivation of the table, in order of the columns:

1. **Loops.** Take one attempt's accepted outputs, with sub-distribution $`P`$ for the
   implementation and $`P^*`$ for the ideal prover, which accepts with probability $`a`$. The
   loops' outputs are then within $`\|P-P^*\|_1/a`$ of each other [proved:
   $`\|P/a_P-P^*/a\|_1\le|a-a_P|/a+\|P-P^*\|_1/a`$]. Each cause below is replaced by its ideal in
   turn (a hybrid), so the bounds add.
   - For the opening proof, $`a\ge(1-2^{-169})/(2M_1M_2)`$. The factor $`1/2`$ is for the
     negative inner products that Rej_2 rejects, and $`2^{-169}`$ bounds the norm checks that
     both provers apply. It comes from $`\chi^2`$ tail bounds for subgaussian coordinates.
   - For the range round, $`a\ge(1-2^{-128})/(M_3M_4)`$.
2. **Rej_1 tail.** Rej_1 accepts with probability
   $`\min(1,D_{s_1}(z)/(M_1D_{v,s_1}(z)))`$. The cap at 1 is reached only when
   $`-\langle z,v\rangle>s_1^2\ln M_1-\|v\|^2/2`$.
   - Bound [proved]: for $`z\leftarrow D_{\mathbb Z^n,s_1}`$ this has probability at most
     $`\epsilon_1=e^{-r^2/2}`$, where $`r=\gamma_1\ln M_1-1/(2\gamma_1)`$ and
     $`\gamma_1=s_1/(\eta\sqrt\alpha)`$.
   - Subgaussian tail: $`D_{\mathbb Z,s}`$ is $`s`$-subgaussian, since
     $`\mathbb E e^{\lambda z}=e^{\lambda^2s^2/2}\rho_s(\mathbb Z-\lambda s^2)/\rho_s(\mathbb Z)`$
     and $`\rho_s(\mathbb Z+c)\le\rho_s(\mathbb Z)`$ by Poisson summation.
   - Norm of $`v`$: $`\|v\|=\|cs_1\|\le\eta\|s_1\|\le\eta\sqrt\alpha`$, because every challenge
     has operator norm at most $`\eta`$ ([challenges](#challenges)) and `commit_with_randomness`
     refuses $`\|s_1\|^2>\alpha`$, with $`\alpha`$ = `alpha_squared` plus $`d`$ per exact-norm
     block.
   - The accepted $`z_1`$ is within $`\epsilon_1`$ of $`D_{s_1}`$ (LNP22 Lemma 2.14(1) states
     $`2^{-128}`$ for its constant).
   - This is not $`\epsilon_1/M_1`$: that figure is the distance of one attempt counted with
     its rejection.
   - At the formula value of $`M_1`$, $`r=\sqrt{258\ln2}`$ and $`\epsilon_1=2^{-129}`$, for every
     set that `check` accepts, up to the `f64` evaluation of $`M_1`$ ([parameters](#parameters)),
     whose rounding can raise $`\epsilon_1`$ by a factor of $`1+10^{-12}`$ at most (an error
     estimate, not a proof). The shipped sets round $`M_1`$ up, which leaves the smaller tails
     of the table.
3. **Coins.** `reject::accept` accepts exactly when $`u\le2^{256}\tilde p`$, for a uniform 256-bit
   coin $`u`$ (`reject::coin`) and $`\tilde p`$, the acceptance probability with the computed
   exponentials. Its acceptance probability is therefore within $`2^{-256}`$ of
   $`\min(1,\tilde p)`$. With the 128-bit coins of the earlier version, this term alone was up
   to $`2^{-128}`$ per test, about $`2^{-124}`$ per proof.
4. **Cutoffs.**
   - Error of `reject::exp_negative`: it is within $`E`$ units of $`2^{-192}`$ of
     $`2^{192}e^{-x}`$, with $`E=12286`$ for $`x<16`$ (the argument in `rand::bernoulli`, also
     checked in Lean, outside this repository, against a model of the code) and $`E\le1.002`$
     for $`x\ge16`$ (the hand proof below, not machine-checked). The column needs both, since
     the tests can pass inputs of 16 or more.
   - Error of $`\tilde p`$, for Rej_1 and Rej_2: within $`\max(E/M,ME)\cdot2^{-192}`$ of $`p`$.
     The term $`ME`$ is for positive exponents, which lie below $`\ln M<16`$.
   - For the bimodal test: within $`\max(4E/M,(M+1)E)\cdot2^{-192}`$.
   - Each error is divided by its test's ideal acceptance rate: $`1/M_1`$, at least $`1/(2M_2)`$,
     and $`1/M`$. With the coins, this gives the column.
   - Sharper bound [proved by hand, evaluated in interval arithmetic]: $`E\le197.4`$ for every
     input below 256, and $`E\le1.002`$ from 16 on. With it the column falls by about 6 bits.
     The proof:
     - The series' truncations add at most 30.4 units in all, input rounding included: the
       $`k`$-th term's error is at most 1 plus $`1/(8k)`$ times the previous term's error, and
       at most 29 terms carry an error.
     - A squaring maps an error $`e`$ to at most $`e(2e^{-x_i}+e2^{-192})+1`$.
     - The worst case is four squarings, from $`x_0=1/16`$ (input 1).
     - The largest error measured on 20,000 inputs is 31.3.
5. **Masks.** Use the bound of the previous section, with its count $`M`$ the expected number
   of coefficients that a proof draws:
   - opening: $`(\text{bounded\_len}+m_2)d/a`$, from 70,656 (`kyber1024_d128`) to 130,560
     (`toy_d64`);
   - range round: $`512/a`$, that is 2,048.
6. **Bimodal tests.** A bimodal test reproduces $`D_s`$ exactly when
   $`\|Re\|^2\le2s^2\ln M`$, because its acceptance probability then never exceeds 1.
   - Each $`\langle r,e\rangle`$ is $`(\|e\|^2/2)`$-subgaussian, since
     $`\mathbb E e^{\lambda\langle r,e\rangle}=\prod_i\cosh^2(\lambda e_i/2)`$ for binomial rows.
   - Bound [proved]: $`\Pr[\|Re\|^2>\tau]\le e^{-128(x-1-\ln x)}`$ with $`x=\tau/(128\|e\|^2)`$.
   - At LNP22's threshold $`337\|e\|^2`$, this is $`2^{-122.75}`$ at most. LNP22 Lemma 2.8 states
     $`2^{-128}`$, under a normal substitution.
   - The shipped sets use $`M_3=M_4=2`$ for formula values of 1.01 to 1.04, so $`x\ge51`$ and
     the probability per attempt is below $`2^{-8600}`$.
   - Per proof [proved]: let $`B_e`$ and $`B_d`$ be the events $`\|Re\|^2>2s^2\ln M`$ of the two
     range blocks. Under them a test's accepted sub-distribution still lies below the ideal
     one, whose mass is $`1/M`$, so the expected number of attempts cancels and the two tests
     add $`(\Pr[B_e]+\Pr[B_d])/(1-2^{-128})`$. For another set, whose rounding leaves no slack,
     this is $`2^{-121.75}`$ at most, and it then dominates.
7. **Negligible.** Each of the following is below $`2^{-196}`$ per proof:
   - a challenge derivation failing after 64 draws, taking the rate of draws above $`\eta`$ as
     at most 0.02 (a statistical bound from the measured rates, which are below 1.2%, not a
     proof; any rate up to 0.2 keeps this term below $`2^{-143}`$);
   - reaching a 4096-attempt limit;
   - a mask above $`q/2`$, which cannot occur for the shipped sets.

The table's values are computed from the shipped sets and the integer constants that Rust
derives; they are not measured, except for the rate of challenge draws above $`\eta`$ in item 7.
The integer constants meet the rejection lemmas' hypotheses
($`M_1`$ and $`M_2`$ at least their formula values) in interval arithmetic, not only in the
`f64` of `Abdlop::rejection_constants`.

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

- **Heuristic projection bound.** The range proofs' rejection constants use
  $`\|Re\|^2\le337\|e\|^2`$, which LNP22 Lemma 2.8 derives under the heuristic substitution of
  the binomial projection by a normal one. For zero-knowledge a subgaussian bound replaces the
  heuristic: it fails with probability at most $`2^{-122.75}`$, and for the shipped sets, whose
  bimodal constants are rounded up to 2, below $`2^{-8600}`$
  ([whole proofs](#statistical-distance-of-whole-proofs)).
- **Rej_1 constant.** $`M_1=\exp(\sqrt{258\ln2}/\gamma_1+1/(2\gamma_1^2))`$, with
  $`\sqrt{258\ln2}\approx13.37`$ in place of the 14 of LNP22 Lemma 2.14(1), so that the tail is
  $`2^{-129}`$ at the formula value. The shipped values of $`M_1`$ also meet LNP22's own
  formula (computed in interval arithmetic).
- **Norm restarts.** The prover also draws again when $`z_1`$ or a range response exceeds the
  verifier's bound ($`\|z^{(e)}\|^2\le`$`z3_bound_squared`, that is
  $`\|z^{(e)}\|\le1.64\sqrt{256}s_3`$, and $`\|z^{(d)}\|_\infty\le`$`z4_bound`). LNP22's provers
  (Fig. 18 for $`z_1`$, Fig. 9 for the range responses) send such a response and count the
  verifier's refusal against completeness; the joint bound of Fig. 18 is LNP22's own restart.
  Accepted responses therefore follow LNP22's laws conditioned on these public bounds, which the
  [whole-proof bound](#statistical-distance-of-whole-proofs) compares with; against the
  unconditioned laws it grows by at most $`2^{-129.3}`$ per proof.
- **Gaussian sampling.** Masks are drawn from close approximations of the discrete Gaussians,
  not from the exact distributions ([Gaussian sampling](#gaussian-sampling)). Rejection sampling
  uses the exact rational variance, 192-bit fixed-point exponentials and 256-bit coins, and
  rounds rejection constants up to integers
  ([whole proofs](#statistical-distance-of-whole-proofs)).
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
