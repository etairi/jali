# Tools

Python tools that derive parameter sets, cross-check them, and regenerate the test data of the
crate. They are part of the repository, not of the crate package. They need Python 3.12 or later
with the pinned packages of `requirements.txt`:

```sh
python3 -m pip install -r tools/requirements.txt
```

SageMath is needed only for the optional lattice-estimator cross-checks. All commands below run
from the repository root.

| Path | Purpose |
| --- | --- |
| `params/lnp_params.py` | The parameter tool: derivation, checks, rank and prime searches |
| `params/jali_params/` | Its modules: hardness estimates, modulus and placement search, request decoding, the emulation of the Rust checks |
| `params/sets/` | Requests, parameter sets, reports and cross-checks of the shipped sets |
| `params/rust_reference.py` | Records what the crate's checks decide, for the tool's tests |
| `params/moduli.py` | The 62-bit NTT primes of `src/params/moduli.rs` |
| `params/tests/` | The tool's tests (standard-library `unittest`) and their recorded data |
| `kat/generate.py` | Independent known-answer vectors of the primitives, `kat/primitives.json`, the challenges within $`\eta`$ included |
| `kat/half_gaussian_cdf.py` | The 256-bit tables of the base Gaussians (widths 3.1 and 1.55) of `src/rand/cdf.rs`, from their definition; `--digest` prints the digest that `cargo test` pins |

## Deriving a parameter set

A statement of a new shape needs its own parameter set:

1. Build the statement and export `Statement::requirements(d, q)` and `Statement::blocks(d)`
   with the `serde` feature, at the proof degree `d` and a probe modulus `q` that no constraint
   uses.
2. Write a request (format below) with the requirements, the statement modulus and a policy.
3. Run `derive`, which writes the parameter set and a report.
4. Load the set with `TboxParams::from_json`, which checks it, and compile the statement.

```sh
p=tools/params
python3 $p/lnp_params.py derive request.json --params params.json --report report.json
python3 $p/lnp_params.py check params.json --request request.json
```

`derive` searches the modulus and, for the blocks the request calls movable, the placement, and
keeps the set with the smallest proof-size estimate. The report is deterministic: request and
tool hashes, the policy, the MLWE and MSIS block sizes and core-SVP exponents, the modulus
search with every rejected prime and its reason, every checked inequality with its margin, the
rejection constants, the result under the other policy, and the placements evaluated. `check`
emulates the Rust checks on a set and prints the derived values; with `--request` it also checks
the set against the request's statement, in the placement the set was derived for.

### Requests

```json
{
  "schema": "jali-params-request/2",
  "id": "jali-kyber1024-d64",
  "mode": "native",
  "degree": 64,
  "modulus": {"search": {"factors": 1, "window_bits": 4}},
  "hardness": {"policy": "delta"},
  "statement": {"source": "lin-worst-case", "statement_degree": 256,
                "statement_modulus": 3329, "rows": 4,
                "blocks": [{"name": "w", "length": 8, "norm": {"l2_squared": 2950}}]}
}
```

- `modulus` is a search (`window_bits`, default 4; `min_bits`; `max_bits`, which keeps
  $`q<2^{\text{max\_bits}}`$) or fixed `prime_factors` with an optional even `gamma`. Without
  `gamma`, a fixed modulus takes the largest even divisor of $`q-1`$ in
  $`(4\gamma_0/5,\gamma_0]`$; the search may choose a divisor outside that window, so
  re-deriving a searched set at its own prime needs its `gamma`.
- `statement` is either `lin-worst-case`, the shape that `lin::compile` builds from $`Aw+t=0`$,
  for which the tool computes the requirements in the worst case over public data, or
  `requirements`, the JSON that `Statement::requirements` writes, with `statement_modulus` and,
  as `blocks`, the JSON of `Statement::blocks` (required with $`\ell_\infty`$ variables). With a
  range block the copied fields must describe one statement: `linf_bound` must be the value
  `Statement::requirements` computes from `max_integer_coefficient` or `lifted_moduli`, and the
  blocks must add up to the dimensions and bounds.
- `movable` lists at most 12 bounded blocks whose part the placement search chooses; the report
  names the blocks to declare with `var_placed` or to pass to `lin::compile_placed`.
- A constraint over the export's own probe modulus is native there and leaves no trace in the
  requirements. For such a statement give `export_modulus`, which pins $`q`$ to that modulus.

Requests are decoded strictly: duplicate keys, NaN, unknown or missing keys, wrong types and
statements that Rust's `Ring` or `Statement::var` refuses are refused with a message.
`jali_params/request.py` documents every field. Integers of $`2^{64}`$ or more are written as
decimal strings in every file the tools write; readers accept both forms.

### Policies

- `delta` (default): the MLWE rank is the smallest whose uSVP estimate has $`\delta_0\le1.0044`$
  (block size at least 346), and the MSIS rank the smallest with a closed-form
  $`\delta<1.0044`$. About 101 bits under core-SVP.
- `beta` with `beta_min`: both block sizes at least `beta_min`, and the Rust bound
  $`\delta\le1.0044`$ kept. As $`0.292\cdot439\approx128.2`$, `beta_min` 439 puts these two
  estimates at 128 classical bits under core-SVP; it bounds no other attack, and not the proof's
  actual number of MLWE samples. Check such a set with `--estimator-target`.
- `external`: a rank, delta and provenance from an external estimate, as the positional form
  takes them.

### Hardness estimates

`jali_params/hardness.py` estimates MLWE by the primal uSVP estimate of Alkim, Ducas, Pöppelmann
and Schwabe (USENIX Security 2016), examined by Albrecht, Göpfert, Virdia and Wunderer
(ASIACRYPT 2017), with Kannan's embedding, secret and error uniform on $`\{-\nu,\dots,\nu\}`$
and any number of samples, and the root Hermite factor of Chen's thesis (2013), used from block
size 50 on. Its success condition is Eq. (2) of Albrecht et al. (§3.2 of ePrint 2017/815), with
the exponent $`2b-D`$ for block size $`b`$ and lattice dimension $`D`$. Eq. (1) of the 2016 paper
(§6.3 of ePrint 2015/1092) has $`2b-D-1`$, which footnote 7 of Albrecht et al. takes for an
error; that exponent never gives a smaller block size. MSIS uses the closed form of
Micciancio and Regev (2009) and Gama and Nguyen (EUROCRYPT 2008) that LNP22 §6.1 applies.
Core-SVP costs are $`2^{0.292b}`$ classical (Becker, Ducas, Gama and Laarhoven, SODA 2016) and
$`2^{0.265b}`$ quantum (Laarhoven's thesis, 2015). The module docstring gives the formulas.
`tests/data/mlwe_grid.jsonl` records the rank searches of the estimate on a grid of degrees and
moduli (`tests/mlwe_grid.py` regenerates it).

### Modulus search

The search starts at the largest lower bound on $`q`$ that `TboxParams::check` and the lifting
check of `Statement::compile` impose independently of $`q`$, moves up to the point where the
MSIS bound is below $`q`$, and fixes the compression divisor first. As $`q\equiv5\pmod8`$ gives
$`v_2(q-1)=2`$, an even $`\gamma\mid q-1`$ has $`v_2(\gamma)\in\{1,2\}`$, and the primes
$`q=1+\gamma t`$ with $`t`$ odd, respectively $`t\equiv2\pmod4`$, are exactly the primes
$`\equiv5\pmod8`$ with $`\gamma\mid q-1`$, so no factoring is needed. For a few $`\gamma`$ near
the largest power of two that keeps MSIS hard, the tool takes the smallest prime of the class at
which $`\gamma`$ keeps MSIS hard, derives the set there and requires the emulated Rust checks to
pass with a relative margin of $`10^{-9}`$ on every `f64` comparison. Over a few bit lengths it
keeps the smallest proof-size estimate. Primality is `sympy.isprime`: deterministic below
$`2^{64}`$, a strong BPSW probable-prime test above. $`\lambda`$ follows Rust's `f64` rule.

Known limitation: each attempt starts at the lower bound or at a power of two, so when the
attempt from the lower bound stops at the MSIS fixed point with a small $`\gamma`$, a better
prime of a larger $`\gamma`$ class just below the next power of two can be missed. The chosen
prime is minimal within its class only above the start it was found from.

### Per-slot range width

The approximate range proof's width $`\sigma_4=1.55\cdot2^{t_4}`$ is sized by default from the
Euclidean bound $`\sqrt{n'd}\,\beta_\infty`$ of the range vector: $`t_4`$ is the exponent whose
width $`1.55\cdot2^{t_4}`$ is nearest $`5\sqrt{337}\sqrt{n'd}\,\beta_\infty`$ (`rounded`). When
the requirements carry `approx_alpha_squared` $`=\alpha_{slot}^2`$, a slot-by-slot bound below
$`n'd\beta_\infty^2`$, `derive` takes

```math
t_4=\max\bigl(\mathrm{rounded}(5\sqrt{337}\,\alpha_{slot}),\ t_{guard}\bigr),
```

where $`t_{guard}`$ is the smallest $`t`$ at which the rejection constant for
$`n'd\beta_\infty^2`$ is 2. Rejection sampling stays exact, because the prover's constant keeps
$`n'd\beta_\infty^2`$, which its witness check enforces; soundness depends on $`\sigma_4`$ only
through the extraction bound $`E=2\cdot`$`z4_bound`, which shrinks with it, and so does the
lifting bound on $`q`$. The report's `range_width` block names both bounds and the widths.

### Emulation of the Rust checks

`jali_params/rustcheck.py` emulates `TboxParams::from_json` (strict decoding), every step of
`TboxParams::check` with its error message, the prover's rejection constants and the checks of
`Statement::compile` in their order. `rust_reference.py` builds a small verifier against a copy
of the crate, with its own target directory, and records what Rust decides: `from_json`, `check`
and `lin::compile` on 264 cases (every refusal, JSON edge cases, moduli up to 255 bits and
seeded random mutations), and for each shipped set and each statement kind of the verifier the
checked values, the requirements, the compilation and one seeded proof (`tests/data/`). The
tests require the emulation to decide every case as Rust did, with equal integers; `f64` outputs
from the platform's libm are compared exactly on the recording platform and within 4 ULP
elsewhere. The copy and its build go to a temporary directory (`--work DIR` chooses where),
which is removed at the end unless `--keep` is given.

```sh
python3 tools/params/rust_reference.py --check   # does Rust still decide as recorded?
python3 tools/params/rust_reference.py           # record again after a change to the checks
```

## Regenerating checked-in files

```sh
p=tools/params
# The shipped sets and their copies in src/params/sets/, byte for byte.
python3 $p/lnp_params.py regenerate --check
# The toy set: its MLWE report, then the set (compare with src/params/sets/toy-d64.json).
python3 $p/lnp_params.py rank --degree 64 --q 1099511627917 --mlwe-report
python3 $p/lnp_params.py $p/toy-d64.request.json $p/toy-d64.report.json --output toy.json
# The known-answer vectors (compare with kat/primitives.json) and the base Gaussian tables.
python3 tools/kat/generate.py --output primitives.json
python3 tools/kat/half_gaussian_cdf.py --check
# The NTT primes of src/params/moduli.rs, as JSON.
python3 $p/moduli.py --count 9
```

`regenerate --check` compares each set under `params/sets/` with the tool's output and with the
crate's copy under `src/params/sets/`, byte for byte, and each report except its `environment`
block. Reports record the hash of the tool's files, so a change to the tool needs a
`regenerate`. The tool's version (`VERSION` in `jali_params/__init__.py`) starts each set's
`estimator` string, which enters the transcript, and changes whenever a derivation's output
does. The positional form of `lnp_params.py` (request, MLWE report, `--output`) derives a set at
the request's modulus from an external MLWE report; the toy set, the possession example's set
and the test fixtures under `tests/fixtures/params/` are its outputs. The CI workflow
(`.github/workflows/ci.yml`, job `tools`) runs all of these checks.

## Cross-checks with lattice-estimator

`xcheck NAME` runs lattice-estimator (LGPLv3+, so not vendored) on a set's MLWE instance, with
its actual number of samples, and on its MSIS instance; `--full` uses the default cost model.
Fetch it at the pinned commit, point `JALI_LATTICE_ESTIMATOR_DIR` at the clone and run under
Sage:

```sh
git clone https://github.com/malb/lattice-estimator
git -C lattice-estimator checkout 53da5982597709ba0fdf94ea37a84d822310fd84
JALI_LATTICE_ESTIMATOR_DIR=$PWD/lattice-estimator \
  sage -python tools/params/lnp_params.py xcheck demo
```

Results are kept as `params/sets/<name>.xcheck.json` and `.xcheck-full.json`; a test checks by
hash that they describe the current parameter file. `xcheck NAME --target BITS` exits 1 unless
the cheapest attack in the rough mode reaches `BITS` on both instances, and
`derive REQUEST --estimator-target BITS` runs that check on a new set and refuses it below the
target. The target checks refuse an estimator checkout that is not at the pinned commit or has
local changes, unless `--allow-unpinned-estimator` is given; the verdict records the checkout
either way.

## Tests

```sh
python3 -m unittest discover -s tools/params/tests -v
JALI_LATTICE_ESTIMATOR_DIR=... sage -python -m unittest discover -s tools/params/tests -v
```

The tests compare the hardness module with its recorded grid, check the prime construction, fire
each refusal of the emulated checks, compare the emulation with the recorded Rust decisions,
test the modulus search where each bound binds, refuse malformed requests and regenerate every
checked-in set and fixture byte for byte. The tests that need SageMath and the lattice-estimator
skip, printing why, when either is missing.
