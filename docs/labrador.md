# LaBRADOR

The module `jali::labrador` implements the succinct proof system LaBRADOR of Beullens and Seiler
(BS23: ePrint 2022/1341, CRYPTO 2023). This note describes the protocol as implemented, its
parameters, its transcript and encoding, its security assumptions, and where it departs from the
paper. Section, figure and theorem numbers refer to the ePrint version of 7 October 2022.

**The proofs are not zero knowledge.** They reveal information about the witness, as in the
paper, which leaves zero knowledge aside (BS23 §1). Use them where the witness may leak, such as
aggregating signatures, or under a zero-knowledge layer.

## Relation

A `Statement` holds the principal relation of BS23 §5.1 over
$`\mathcal R_q=\mathbb Z_q[X]/(X^{64}+1)`$, for a prime $`q\equiv5\pmod8`$ with
$`2^{16}<q<2^{48}`$, so that $`X^{64}+1`$ splits into two irreducible factors modulo $`q`$. A
witness is $`r`$ vectors $`s_1,\dots,s_r\in\mathcal R_q^n`$ with
$`\sum_i\|s_i\|^2\le\beta^2`$ (centred coefficients). A constraint is

```math
f(s)=\sum_{i,j}a_{ij}\langle s_i,s_j\rangle+\sum_i\langle\varphi_i,s_i\rangle-b,
\qquad \langle a,b\rangle=\sum_k a_kb_k\in\mathcal R_q .
```

`Statement::push` adds a constraint that must vanish in $`\mathcal R_q`$, and
`Statement::push_constant_term` one whose constant coefficient must vanish. Since
$`\mathrm{ct}(\langle\sigma_{-1}(a),b\rangle)`$ is the integer dot product of the coefficient
vectors, for $`\sigma_{-1}(X)=X^{-1}`$, constant-term constraints state integer relations (BS23
§2). The rings of degree 128 and 256 are supported by the arithmetic but have no challenge set
yet, so proofs need degree 64.

## One level

`prove` runs levels of the main protocol (BS23 Fig. 2), each a proof of knowledge reduction to
a new instance of the same relation (§5.3), and sends the last witness in the clear. For a
level with input $`(s_i)`$ and bound $`\beta`$:

1. **Commit.** $`t_i=As_i\in\mathcal R_q^\kappa`$ and $`g_{ij}=\langle s_i,s_j\rangle`$ for
   $`i\le j`$, decomposed into centred digits in odd bases $`b_1`$ and $`b_2`$
   ($`\hat t,\hat g`$), committed as $`u_1=(B\mid C)(\hat t\,\|\,\hat g)\in\mathcal R_q^{\kappa_1}`$.
2. **Project.** The verifier's matrix $`\Pi\in\{-1,0,1\}^{256\times rnd}`$ (entries 0 with
   probability 1/2, $`\pm1`$ with 1/4 each). The prover sends $`p=\Pi\cdot\mathrm{coef}(s)`$ and
   the verifier checks $`\|p\|^2\le337\beta^2`$.
3. **Aggregate with integer challenges.** $`\lambda=\lceil128/\lfloor\log_2q\rfloor\rceil`$
   combinations of the constant-term constraints and of the 256 projection constraints, with
   uniform $`\psi^{(k)},\omega^{(k)}`$. The prover sends the lifted polynomials $`b''^{(k)}`$
   without their constant coefficients, which the verifier sets to
   $`\langle\omega^{(k)},p\rangle+\sum_l\psi^{(k)}_lb'^{(l)}_0`$.
4. **Aggregate with polynomial challenges.** All functions into one
   $`F=\sum_{i\le j}a_{ij}\langle s_i,s_j\rangle+\sum_i\langle\varphi_i,s_i\rangle-b`$ with
   uniform $`\alpha,\beta`$. The prover sends $`u_2=D\hat h`$ for the digits of
   $`h_{ii}=\langle\varphi_i,s_i\rangle`$ and $`h_{ij}=\langle\varphi_i,s_j\rangle+\langle
   \varphi_j,s_i\rangle`$ ($`i<j`$).
5. **Amortize.** Challenges $`c_1,\dots,c_r`$, and the next witness: $`z=\sum_ic_is_i`$ in
   digits $`z=z_0+bz_1`$, with $`\hat t,\hat g,\hat h`$.

The next statement holds the verification equations of BS23 Fig. 3 (lines 15 to 20) on the next
witness, with the consolidated norm bound $`\beta'`$ of §5.3:
$`Az=\sum_ic_it_i`$, $`(B\mid C)(\hat t\|\hat g)=u_1`$, $`D\hat h=u_2`$,
$`\langle z,z\rangle=\sum_{i\le j}w_{ij}c_ic_jg_{ij}`$ ($`w_{ii}=1`$, $`w_{ij}=2`$),
$`\langle\varphi_z,z\rangle=\sum_{i\le j}c_ic_jh_{ij}`$ with $`\varphi_z=\sum_ic_i\varphi_i`$, and
$`\sum_{i\le j}a_{ij}g_{ij}+\sum_ih_{ii}=b`$. It is kept in structured form: a level computes
only its aggregation with fresh challenges, expanding the public matrices row by row.

**Layout.** $`z_0`$ and $`z_1`$ are split into $`\nu`$ parts of exactly $`n'=n/\nu`$
polynomials ($`\nu`$ divides $`n`$), and $`v=\hat t\,\|\,\hat g\,\|\,\hat h`$ ($`m`$ polynomials)
into $`\mu=\lceil m/n'\rceil`$ parts of $`n'`$, the last one zero-padded. The next multiplicity
is $`2\nu+\mu`$. Padding inside the parts of $`z`$ would enter $`\langle z,z\rangle`$ without any
constraint on it, so it is not allowed. The padding of $`v`$ is harmless, since the next
statement reads $`v`$ only linearly, at the positions that carry values.

**Last witness.** The last level's next witness is sent in the clear, without the padding of
$`v`$. The verifier checks its norm, then all equations of the last statement at once, aggregated
with challenges squeezed after the witness is absorbed. A wrong witness passes with probability
at most $`q^{-d/2}`$ per attempt.

## Arithmetic and sampling

- **Ring.** Canonical `u64` coefficients. Products go through the negacyclic NTTs of
  `math::ntt` over two or three of the 62-bit primes of `params::moduli::NTT_PRIMES`, in
  Montgomery form, and back through the exact CRT of the crate, with enough primes that
  $`2^{32}`$ accumulated products of centred operands reconstruct exactly. This arithmetic is
  separate from the 256-bit arithmetic of the LNP22 proofs, which it does not change.
- **Challenges.** The set of BS23 §2: 31 coefficients $`\pm1`$, 10 coefficients $`\pm2`$, at
  positions of a Fisher–Yates shuffle, with operator norm at most $`T=15`$. The operator norm is
  tested with the exact integer bound of LNP22 §2.7 that `rand::within_eta` implements, a
  sufficient condition with no floating point. Each challenge has its own AES-256-CTR stream
  from a squeezed seed. On average 5.77 draws give one challenge (measured in the unit tests on
  300 challenges).
- **Projection matrices** are expanded by AES-256-CTR from a 32-byte squeezed seed, 2 bits per
  entry with the centred binomial decoding of FIPS 203 for $`\eta=1`$.
- **Public matrices** $`A`$, $`(B\mid C)`$ and $`D`$ of each level are expanded by AES-256-CTR from
  the caller's 32-byte public seed, uniform modulo $`q`$ by rejection.

## Parameters

The prover chooses each level's parameters when it starts the level, from the norm of that
level's witness and its exact garbage $`g_{ij}`$, with the heuristics of BS23 §5.4 and margins.
It predicts the norms of $`z`$ and $`h`$, which depend on later challenges. A level whose next
witness exceeds the bound it announced is restarted with the next attempt counter (64 at most).
The bases are odd, so that centred digits are symmetric and the garbage of the next level has no
systematic term. The prover also chooses the splits and when to stop, with a model of the proof
size and a lookahead of one level.

The parameters are sent in the proof, 55 bytes per level, and the verifier checks, for each
level with input bound $`\beta`$, next bound $`\beta'`$, slack $`s=\sqrt{337/30}`$ when another
level follows and $`s=1`$ when the verifier checks the next witness directly:

- $`\sqrt{337/30}\,\beta\le q/125`$, in integers,
- Module-SIS of rank $`\kappa`$ hard for the bound
  $`\max(8T(b+1)s\beta',\ 2(b+1)s\beta'+4T\sqrt{337/30}\,\beta)`$,
- Module-SIS of ranks $`\kappa_1`$ and $`\kappa_2`$ hard for the bound $`2s\beta'`$,
- the dimensions of the layout, with at most 1024 vectors and $`2^{24}`$ coefficients per
  level.

These are the conditions of BS23 Theorem 5.1 with Remark 5.2, adapted to the projection bound
$`\sqrt{337}\beta`$. **Hardness.** Module-SIS of rank $`k`$ with Euclidean bound $`B<q`$ is
estimated by the closed form $`\delta=2^{(\log_2B)^2/(4kd\log_2q)}`$ and the smallest BKZ block
size with $`\delta_0\le\delta`$, as for the LNP22 sets (`parameters.md`). `Policy` requires
$`\delta<1.0044`$ and a block size of at least 439 by default, about 128 bits under the core-SVP
model $`2^{0.292b}`$. These are cost-model heuristics, not proofs.

## Transcript and encoding

The transcript is the duplex sponge of draft-irtf-cfrg-fiat-shamir-04 over SHAKE128
(`encodings.md`), with the tag `JALI-V02-LABRADOR-C1-SHAKE128` followed by the context. It
absorbs the public seed, the policy's minimum block size (`LE32`) and the statement (`LE64`
modulus, `LE32` degree, multiplicity and rank, `LE128` bound, then each family as a count and
its constraints). Per level it absorbs the parameters, $`u_1`$, $`p`$, the tails of
$`b''^{(k)}`$ and $`u_2`$, and squeezes the projection seed, $`\psi^{(k)}`$ and $`\omega^{(k)}`$
(`DecodeUint`), $`\alpha`$ and $`\beta`$ (uniform polynomials by `DecodeUint`) and the challenge
seed. At the end it absorbs the last witness and squeezes the final aggregation challenges.

A proof is the concatenation of, per level: the parameters (55 bytes), $`u_1`$ and $`u_2`$ and
the tails of $`b''`$ as residues of $`\lceil\log_2q\rceil`$ bits, $`p`$ in the signed code of
codec C1 with the least width for $`\lfloor\sqrt{337\beta^2}\rfloor`$, each message zero-padded
to a byte. Then the last witness, one bit string zero-padded to a byte, vector by vector, each
with its polynomials that carry values. A part of $`z_0`$ or of $`v`$ (centred digits, close to
uniform) is one byte for the least signed width $`w`$ of its largest magnitude $`h`$, then
$`h`$ in $`w-1`$ bits, then its values in the radix code of base $`R=2h+1`$: chunks of the most
digits $`k`$ with $`R^k<2^{128}`$, each the number $`\sum_j(x_j+h)R^j`$ in the canonical code of
$`[0,R^k)`$, nothing when $`h=0`$. A part of $`z_1`$ (close to a discrete Gaussian) is one byte for
the parameter of the Gaussian code of codec C1 that writes it in the fewest bits, the least on
ties, then its values in that code. Every value, width, maximum, parameter and padding is
checked, so that each witness has one encoding, and no byte may follow.

## Security notes

- **Knowledge soundness** of one level is BS23 Theorem 5.1 under the conditions above, with
  slack $`\sqrt{337/30}\approx3.35`$ instead of $`\sqrt{128/30}`$. The paper sums its error
  terms to $`2^{-125}`$. Here the challenge set after the operator-norm filter has about
  $`2^{128.09}/5.77\approx2^{125.6}`$ elements (the count before the filter is exact, the
  acceptance rate is measured), so the term $`2/|\mathcal C|`$ is about $`2^{-124.6}`$, and a
  level's error is about $`2^{-124.5}`$. Levels compose by BS23 Lemma 3.7, which adds their
  errors.
- **Parameters chosen by the prover.** The paper fixes the parameters. Here the prover sends
  them before each level's messages, and the verifier checks the theorem's conditions for the
  values sent, so every accepted level is one the theorem covers.
- **Fiat–Shamir.** The non-interactive proof rests on the random-oracle heuristic, with the
  usual loss in the number of oracle queries. The Toolkit paper (ePrint 2026/1289, App. B.4)
  discusses the compilation of LaBRADOR through predicate special soundness.
- **Heuristics.** The projection lemma (BS23 Lemma 4.1) substitutes a normal distribution for
  the projection entries. The hardness estimate is a cost model. The completeness analysis is
  heuristic, which only affects how often the prover restarts.
- **Variable time.** As in the rest of the crate.

## Departures from the paper

- The projection check is $`\|p\|\le\sqrt{337}\beta`$, which an honest prover passes except
  with probability about $`2^{-128}`$ (BS23 Lemma 4.1), instead of $`\sqrt{128}\beta`$ with
  retries. The slack grows from 2.07 to 3.35, and no retry needs to enter the transcript.
  Orthus (ePrint 2026/398, App. C) discusses both checks.
- $`h_{ij}`$ for $`i<j`$ is twice the paper's, so that no halving is needed.
- The constant coefficients of $`b''^{(k)}`$ are not sent, since the verifier knows them.
- $`\lambda`$ uses $`\lfloor\log_2q\rfloor`$, so $`q^{-\lambda}\le2^{-128}`$ holds exactly (5
  combinations for $`q=2^{32}-99`$, where the paper's formula gives 4).
- The split of $`z`$ uses divisors of $`n`$, so that no padding enters $`\langle z,z\rangle`$.
  Ranks with many divisors (powers of two) compress best.
- The parameters are chosen by the prover, sent, and checked by the verifier.
- The last level is a plain level whose witness is sent in the clear, not the variant of §5.6.

## Not implemented

The last-level variant of BS23 §5.6, the R1CS reductions of §6, the compressed variant of
Orthus §5.1 (LaBRADOS), the zero-knowledge composition of the Toolkit paper, sublinear
verification (Orthus), challenge sets for degrees 128 and 256, parallelism and SIMD.

## Example and measurements

`cargo run --release --example labrador -- r n bound` proves random constraints on $`r`$
vectors of $`n`$ polynomials with coefficients in $`[-\mathrm{bound},\mathrm{bound}]`$. On an
Apple M4, one thread, load average 5.3 to 5.9 (another process ran), on 15 November 2026:

| Witness (ternary coefficients) | Proof | Prove | Verify |
|---:|---:|---:|---:|
| 4,096 | 9,063 B | 6 ms | 4 ms |
| 262,144 | 68,778 B | 414 ms | 183 ms |
| 1,048,576 | 80,943 B | 1.67 s | 0.70 s |
| 4,194,304 | 85,490 B | 5.29 s | 2.10 s |

At 4,194,304 coefficients the proof has 5 levels of about 4.4 KB and a last witness of about
64 KB, which the variant of §5.6 would shrink.

## References

- W. Beullens and G. Seiler. LaBRADOR: Compact Proofs for R1CS from Module-SIS. CRYPTO 2023.
  Cryptology ePrint Archive, Paper 2022/1341.
- M. Bolboceanu, J. Bootle, V. Lyubashevsky, A. Merino-Gallardo and G. Seiler. Orthus: Practical
  Sublinear Batch-Verification of Lattice Relations from Standard Assumptions. CRYPTO 2026.
  Cryptology ePrint Archive, Paper 2026/398.
- B. Biasioli, M. Bolboceanu, V. Lyubashevsky, A. Merino-Gallardo, M. Osadnik, G. Seiler and
  P. Steuer. A Toolkit for Succinct Lattice-Based Zero Knowledge Proofs. Cryptology ePrint
  Archive, Paper 2026/1289.
- V. Lyubashevsky and G. Seiler. Short, Invertible Elements in Partially Splitting Cyclotomic
  Rings and Applications to Lattice-Based Zero-Knowledge Proofs. EUROCRYPT 2018. (Invertibility
  of challenge differences.)
- C. Gentry, S. Halevi and V. Lyubashevsky. Practical Non-interactive Publicly Verifiable Secret
  Sharing with Thousands of Parties. EUROCRYPT 2022. (The modular Johnson–Lindenstrauss lemma.)
