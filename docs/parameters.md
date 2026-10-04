# Parameters

A parameter set (`TboxParams`) fixes the proof ring, the commitment dimensions, the Gaussian
widths, the compression divisor and the statement shape it was derived for, together with
externally supplied hardness metadata. This page describes what the crate checks, the shipped
sets and the hardness estimates behind them. Deriving a set for a new statement is the job of
the [parameter tool](../tools/README.md).

## What `TboxParams::check` covers

`check` derives the extension dimensions, the challenge repetitions, the response bounds, the
proof-size estimate, the MSIS closed-form estimate and the lifting inequalities of the paper. It
refuses invalid prime factors, inconsistent dimensions, invalid compression divisors, capacities
the implementation does not support (prime factors or products from $`2^{256}`$ on, Gaussian
exponents above 100, response bounds above 128 bits), an MSIS root Hermite factor of 1.0044 or
more or an MSIS bound not below $`q`$, and a reported MLWE root Hermite factor outside
$`[1,1.0044]`$. It refuses binary, exact-norm and range blocks over a two-prime modulus, and it
requires both range rejection constants to stay below $`2^{16}`$. Those constants are computed
from Euclidean bounds on the range vectors: for the approximate block that bound is
$`\sqrt{n'd}\cdot`$`linf_bound` (LNP22 Fig. 10), not `linf_bound`. JSON cannot override derived
quantities; `from_json` refuses unknown fields and runs these checks. `Statement::compile` adds
the checks that depend on the statement, among them the lifting bound of every lifted constraint
([protocol](protocol.md#compilation)).

The MLWE rank, the reported root Hermite factor and the estimator string are supplied evidence:
`check` verifies that they are present and within the accepted interval, not that the estimate
is true. Every parameter field, the estimator string included, enters the transcript, so a set
with other metadata produces other proofs.

## Shipped sets

| Set | $`\log_2q`$ | MLWE rank, block size | MSIS rank, block size | $`\gamma`$, $`D`$ | Estimate | Measured |
| --- | ---: | --- | --- | --- | ---: | ---: |
| `kyber1024_d64` | 41.14 | 26, 348 | 17, 360 | 262142, 9 | 21,058 B | 20,318 B |
| `kyber1024_d128` | 41.14 | 13, 348 | 8, 359 | 524286, 11 | 22,608 B | 21,818 B |
| `demo_d64` | 60.44 | 38, 349 | 11, 347 | 65534, 7 | 24,814 B | 23,991 B |

The three sets were derived by the parameter tool from the requests in `tools/params/sets/` for
the relation $`Aw+t=0`$ over $`\mathbb Z_p[X]/(X^{256}+1)`$ with $`A`$ of size $`4\times8`$: the
Kyber-1024 shape ($`p=3329`$, $`\|w\|^2\le2950=\lceil(1.2\sqrt{2048})^2\rceil`$) at proof
degrees 64 and 128, and the same shape with $`p=2^{32}-4607`$ and $`\|w\|^2\le2048`$, the size
of the Module-LWE example of LNP22 §6.2, at degree 64. The requirements are the worst case over
public data (every coefficient of $`A`$ and $`t`$ at the largest centred absolute value), so the
sets compile every instance of their shape; in all three the lifting check of `compile` sizes
$`q`$. The demo request caps $`q`$ below $`2^{64}`$. "Estimate" is the set's
`estimated_proof_bytes`; "measured" is the length of one proof of a seeded random instance with
a ternary witness, which `tests/param_sets.rs` pins. `src/params/sets/` holds byte-identical
copies of the tool's output, and `src/golden_tests.rs` pins each set's transcript digest and
scheme fingerprint.

`toy_d64` is a small set for tests and examples: a statement shape at proof degree 64 with every
kind of block (two binary rows, exact-norm blocks of two and one rows with squared bounds 128
and 64, an approximate range block of two rows with $`\ell_\infty`$ bound 4) and two BDLOP
messages, over $`q=2^{40}+141`$. The tool derives it from `tools/params/toy-d64.request.json`
and the MLWE report `tools/params/toy-d64.report.json` (rank 26, block size 361), which
`lnp_params.py rank --mlwe-report` writes. The parameter set of `examples/possession.rs`
(`examples/fixtures/possession.json`) is derived from `tools/params/possession.request.json` and
the same report, as its proof ring is the same.

## Hardness

**Criterion.** The shipped sets meet root Hermite factors $`\delta_0\le1.0044`$ for MLWE (block
size at least 346) and $`\delta<1.0044`$ for MSIS, the tool's `delta` policy. LNP22 aims for
$`\delta<1.0045`$ (§6.1); 1.0044 is a slightly stricter choice. Under the core-SVP cost model,
$`2^{0.292b}`$ classical and $`2^{0.265b}`$ quantum for block size $`b`$, block size 346 is
about 101 classical bits, not 128.

**Models.** MLWE of rank $`k`$ over degree $`d`$ is estimated as LWE of dimension $`kd`$ with
secret and error uniform on $`\{-1,0,1\}`$ and any number of samples, by the primal uSVP
estimate of Alkim, Ducas, Pöppelmann and Schwabe (USENIX Security 2016) with Kannan's embedding
and the root Hermite factor of Chen's thesis (2013). The success condition is Eq. (2) of
Albrecht, Göpfert, Virdia and Wunderer (ASIACRYPT 2017; §3.2 of ePrint 2017/815), with the
exponent $`2b-D`$ for block size $`b`$ and lattice dimension $`D`$. Eq. (1) of the 2016 paper
(§6.3 of ePrint 2015/1092) has $`2b-D-1`$, which footnote 7 of the 2017 paper takes for an
error; that exponent never gives a smaller block size. MSIS uses the closed form
$`\delta=2^{(\log_2B)^2/(4nd\log_2q)}`$ of the methodology of Micciancio and Regev (2009) and
Gama and Nguyen (EUROCRYPT 2008), as LNP22 §6.1 applies it; its block size is the smallest $`b`$
with $`\delta_0(b)\le\delta`$. Hiding needs Extended-MLWE, which is taken to be as hard as MLWE,
following LNP22 §6.1; binding needs MSIS. These are cost-model heuristics, not proofs, and the
uSVP estimate bounds no other attack. The [parameter tool](../tools/README.md) documents its
implementation and tests.

**Cross-checks.** lattice-estimator (commit `53da5982597709ba0fdf94ea37a84d822310fd84`, under
SageMath 10.9) was run on each set's MLWE instance, with the proof's actual number of samples,
and on its MSIS instance (`tools/params/sets/*.xcheck.json` and `*.xcheck-full.json`):

| Set | MLWE, rough | MSIS, rough | MLWE, default model | MSIS, default model |
| --- | ---: | ---: | ---: | ---: |
| `kyber1024_d64` | $`2^{100.6}`$ | $`2^{105.1}`$ | $`2^{128.6}`$ | $`2^{133.0}`$ |
| `kyber1024_d128` | $`2^{100.6}`$ | $`2^{104.5}`$ | $`2^{128.6}`$ | $`2^{132.4}`$ |
| `demo_d64` | $`2^{101.1}`$ | $`2^{101.0}`$ | $`2^{130.1}`$ | $`2^{129.0}`$ |

Each MLWE figure is the cheapest attack the estimator found, in every case the dual hybrid. The
rough mode uses the core-SVP cost model; in it the dual hybrid is slightly cheaper than the uSVP
estimate of the `delta` policy ($`2^{101.6}`$ for the Kyber sets). For a 128-bit target, derive
a set with the tool's `beta` policy and check it with `derive --estimator-target 128`; a
block-size floor alone does not guarantee the target, because the estimator also considers other
attacks.

## Challenges

The challenge polynomials are $`\sigma_{-1}`$-stable, with coefficients in $`[-\omega,\omega]`$
and coefficient $`d/2`$ zero, and they meet the operator-norm bound of LNP22 §2.7,
$`(\|\sigma_{-1}(c^k)c^k\|_1)^{1/(2k)}\le\eta`$ with $`k=32`$. LNP22 chooses $`\eta`$ so that at
least 99% of the challenges meet it and lists $`(\omega,\eta)=(2,59)`$ for degree 128 (Fig. 3).
Jali uses $`(2,59)`$ at degree 128 and, at degree 64, $`(8,140)`$, the pair of LaZer's parameter
script (`scripts/lnp-tbox-codegen.sage`). `rand::challenge` draws a $`\sigma_{-1}`$-stable
polynomial with uniform coefficients (`rand::autostable`) and keeps it if
$`\|(\sigma_{-1}(c)c)^{32}\|_1\le\eta^{64}`$, a test in exact integer arithmetic
(`rand::within_eta`); otherwise it draws again from the same stream, at most 64 times ([security
notes](security.md#challenges)). The bound is what the parameter check assumes: the verifier
bound $`B`$, the MSIS bound and the rejection constants use $`\eta`$.

Measured with exact arithmetic (SageMath 10.9) on $`2\cdot10^5`$ uniform draws per degree and
seed, 0.757% (seed 7) and 0.726% (seed 11) of the draws exceed $`\eta=140`$ at degree 64, and
1.127% and 1.195% exceed $`\eta=59`$ at degree 128: pooled, 0.741% and 1.161%, with 95%
intervals of $`\pm0.027\%`$ and $`\pm0.033\%`$. LNP22's $`(2,59)`$ is therefore met by about
98.84% of the draws (95% interval 98.81% to 98.87%), slightly fewer than the 99% LNP22 aims at,
and LaZer's $`(8,140)`$ by about 99.26%; as the test is exact, this changes only the size of the
set. Before the test the sets have $`17^{32}\approx2^{130.80}`$ and $`5^{64}\approx2^{148.60}`$
elements, after it about $`2^{130.79}`$ and $`2^{148.59}`$. An earlier measurement on 20,000
draws (Python with seed 1, SageMath 10.9 with seed 4) found 99.30% within the bound at degree
64, and 98.82% and 98.87% at degree 128.

The test squares $`u=\sigma_{-1}(c)c`$ five times and accepts as soon as a norm shows the bound:
$`\|u^{2^{j-1}}\|_1\le\eta^{2^j}`$ implies it, as $`\|xy\|_1\le\|x\|_1\|y\|_1`$, and for $`j=6`$
it is the bound. Measured on $`2\cdot10^5`$ uniform draws per degree, 98.9% of the draws at
degree 64 and 98.2% at degree 128 are decided within the first three products, in 128-bit
integers; the others take the products in 256 to 1024 bits. A derivation takes 6.4 µs at degree
64 and 20 µs at degree 128, a full test 67 µs and 259 µs
(`cargo bench --bench arithmetic -- challenge/` on an Apple M4). Prover and verifier run one
test per draw: a prover in each attempt of each opening proof, a verifier once per opening
proof, and each one more per rejected draw.

## Size estimate

`estimated_proof_bytes` is what the parameter tool's searches minimize. It counts the full-size
polynomials exactly, as the encoder writes them, each Gaussian coefficient as
$`\lceil\log_2\sigma+2.5\rceil`$ bits, and each of the $`n\cdot d`$ hint coefficients as
$`\max(2.25,\,1.6\sigma_2/\gamma+0.7)`$ bits. LNP22 §6.1 counts 2.25 bits per hint, the average
computed for the prefix-free code by Ducas, Lepoint, Lyubashevsky, Schwabe, Seiler and Stehlé
(ePrint 2017/633) for hints in $`\{-1,0,1\}`$. A hint is about
$`\mathrm{round}(z_{2,2}/\gamma)`$, so its expected length grows with $`s=\sigma_2/\gamma`$.
Modelling $`z_{2,2}/\gamma`$ as a normal variable of deviation $`s`$ plus a uniform rounding
error, the expected code length is $`H(s)=2-3Q(1/s)-2Q(2/s)+3s\varphi(1/s)+s\varphi(2/s)`$, with
$`\varphi`$ the standard normal density and $`Q`$ its tail; $`H`$ is nondecreasing and convex
with slope at most $`2\sqrt{2/\pi}<1.6`$, and $`H(s)=2.25`$ at $`s\approx0.989`$ (computed), so
the allowance bounds $`H`$ for every $`s`$ and equals 2.25 up to $`s=0.96875`$. The shipped
sets, the toy set and the test fixtures have $`s<0.1`$. The allowance bounds the expected
length, not each proof: at $`s=12.45`$, where $`H=19.48`$ and the allowance is 20.62, ten
measured proofs had 18.5 to 19.7 bits per hint. The estimate is not a proven bound on the length
of a proof.
