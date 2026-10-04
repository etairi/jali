# Protocol and statement compiler

This page describes what the crate proves and how its two statement interfaces map onto the
proof system of LNP22 (Lyubashevsky, Nguyen and Plançon, CRYPTO 2022, ePrint 2022/284). Figure,
section, equation and page numbers here and in the API documentation refer to the ePrint full
version of 14 August 2022; they differ between revisions.

## Layers

`abdlop::Abdlop` validates parameters and expands the public matrices from a 32-byte seed. An
opening owns the bounded witness, the message witness and the commitment randomness; its fields
are private. The compressed opening proof follows LNP22 Fig. 18, including the joint bound on
both parts of the randomness response.

Above it are `quad` (one quadratic relation, Fig. 6), `quad_many` (many relations, Fig. 7),
`quad_eval` (evaluations with vanishing constant coefficient, Fig. 8) and `tbox` (the combined
protocol of §5.2, Fig. 10, with binary, exact-norm and approximate range blocks). Each layer
takes an explicit commitment and opening and an application context. `quad_eval` checks both
coefficients 0 and $`d/2`$ of every response. The toolbox commits the range masks, absorbs the
range responses before $`\Gamma`$ and absorbs $`h`$ before $`\mu`$; $`\Gamma`$ has an
independent scalar for each equation and each row.

## Moduli and coefficients

Rings accept any modulus $`2\le q<2^{256}`$ and store coefficients as canonical 256-bit values
in $`[0,q)`$. `Poly::new`, `constant` and `set_coefficient` take `i128` values and reduce them;
`Poly::from_u256`, `constant_u256` and `scale_u256` take 256-bit ones. `coefficients` returns
the canonical values; `coefficients_i128` and `coefficient_i128` return centred values in
$`(-q/2,q/2]`$ and fail with `Error::Overflow` when one does not fit `i128`, which can happen
only for $`q\ge2^{128}`$. Norms are of centred values: `norm_squared` is exact below $`2^{256}`$
and saturates at `U256::MAX` above. Every bound compared with a norm is smaller, so the decision
stays exact, and a response whose norm saturates is an invalid proof, not an arithmetic error.

Proof parameters have one or two prime factors $`\equiv5\pmod8`$ with a product below
$`2^{256}`$. `TboxParams::check` tests factors below $`2^{64}`$ with a deterministic
Miller–Rabin test and larger ones with Baillie–PSW (`crypto-primes`), a probable-prime test with
no known counterexample, not a proof of primality. A factor above $`2^{64}`$ can be proven prime
elsewhere, for instance with Sage's `is_prime(proof=True)`; the crate checks no certificate.
Binary, exact-norm and range blocks need a single prime factor (see the [security
notes](security.md#two-prime-moduli)). Gaussian exponents go up to 100 (`rand::MAX_LOG_SIGMA`);
the check computes every derived response bound exactly and refuses a set whose bound needs more
than 128 bits.

The statement compiler computes its integer bounds exactly in 1024 bits and requires them below
$`2^{256}`$. It evaluates a lifted relation on centred representatives in `i128` and, when a
value does not fit, in 512 bits, which hold every product of two centred 256-bit values. A carry
must fit `i128`; the compiler bounds it by `linf_bound`, a `u64`.

## Two statement interfaces

`lnp::Statement` describes a relation directly in the proof ring. `Prover` and `Verifier` own
that statement; setters validate dimensions and clone inputs. Proving also checks the witness
against every original constraint, and a false witness returns `Error::Witness`. Verification
reconstructs the public constraints and the transcript independently.

The original bounded witness has `params.m1` polynomials. User equation indices interleave
`s, sigma(s), m, sigma(m)`: witness polynomial `i` has index `2i` and its image `2i + 1`. Slack
polynomials are inserted internally, and callers must not include them in user-equation indices.
The lower-level ABDLOP interface instead expects its full `bounded_len()`, slack slots included,
because it proves only an opening.

An `AffineBlock` describes `E_s s + E_m m + offset`; absent matrices are zero maps with explicit
dimensions. Binary constraints cover every coefficient, and an `L2Block` bounds the squared
Euclidean norm of the whole block. The approximate range block checks the honest witness against
`linf_bound` per coefficient. Its rejection constant comes from the Euclidean bound
$`\sqrt{n'd}\cdot`$`linf_bound` of the whole vector, and so, by default, does the parameter
tool's Gaussian width; extraction guarantees only `approx_extraction_bound`
$`=2\cdot`$`z4_bound` $`\le32\sigma_4`$ per coefficient.

`statement::Statement` takes a separate statement ring and named witness blocks. Declare all
variables first with `var(name, length, Norm)`: an exact squared Euclidean bound, binary, an
approximate $`\ell_\infty`$ bound, an exact one, or unbounded.
`var_placed(name, length, Norm, Placement)` also chooses the part of the commitment that holds
the block ([placement](#placement)), and `var_subring(name, length, degree, Norm, Placement)`
declares elements of a subring of smaller degree, of which only the nonzero components are
committed. Build affine expressions with `variable`, `constant`, `add` and `scale`;
`product_affine` forms a quadratic expression and refuses products of higher degree;
`coefficient_in` reads one coefficient of a subring element. Add constraints with `eq_mod_p` or
`const_coeff_zero`.

### Several moduli

Each constraint holds modulo the modulus of the ring its form is built over. `variable` and
`constant` build over the statement ring; `variable_in(ring, name, component)` and
`constant_in(value)` build over any ring of the statement's degree, with the same variables. One
statement can so combine constraints modulo different moduli on shared witness blocks, for
instance a relation modulo a prime $`q_c`$ and a rounding clause modulo a composite
$`q_{tag}=\gamma p`$:

```rust,ignore
let tag_ring = Ring::new(q_tag, degree)?;
let clause = st
    .variable_in(&tag_ring, "key", 0)?
    .scale(&a)? // a is over tag_ring
    .add(&st.constant_in(Poly::constant(tag_ring.clone(), -gamma * t))?)?;
st.const_coeff_zero(clause)?;
```

A form over a ring of another degree is refused with `Error::RingMismatch`. The compiler lifts
every constraint whose modulus $`p_j`$ is not the proof modulus $`q`$ with that modulus, which
may be composite, and compiles a constraint at $`q`$ natively, without carry. The lifting
condition below is checked per constraint, and a modulus above $`q`$ fails it.
`tests/multi_modulus.rs` and `tests/tag_preimage.rs` prove such statements.

### Witness values and the range condition

Witness blocks are polynomials over the statement ring, and each coefficient stands for the
integer of its centred representative modulo the statement modulus $`p`$, in $`(-p/2,p/2]`$. A
witness satisfies the statement if these integers meet every declared norm and every constraint
$`f_j`$ vanishes on them modulo its own modulus $`p_j`$.

A proof establishes an integer relation: knowledge of integers $`\bar s`$ within the declared
norms, zero outside the subring of a block declared with `var_subring`, with values modulo $`q`$
for unbounded blocks, such that $`f_j(\bar s)\equiv0\pmod{p_j}`$ for every constraint. For a
`Norm::Linf` block the norm is relaxed: a proof bounds its values by the extraction bound $`E`$,
not by $`\beta`$; `Norm::LinfExact` keeps $`\beta`$. Reducing $`\bar s`$ modulo $`p`$ keeps the
norms, since it never increases an absolute value, and keeps every constraint whose modulus
divides $`p`$. A constraint of another modulus stays true only if the values it reads lie in
$`(-p/2,p/2]`$. This range condition is a soundness condition, which `requirements` and
`compile` enforce: for every constraint whose modulus does not divide $`p`$, every variable it
uses with a nonzero coefficient must be binary, have $`\lfloor\sqrt B\rfloor\le(p-1)/2`$ for a
squared bound $`B`$, be bounded exactly by $`\beta\le(p-1)/2`$, be an $`\ell_\infty`$ block with
$`E\le(p-1)/2`$, or be unbounded with $`p\ge q`$. A statement that violates it is refused with
“variable range above statement modulus”. Without the check a proof could verify for a statement
with no witness: over $`\mathbb Z_{13}`$ with $`\|x\|^2\le64`$, no value in $`[-6,6]`$ satisfies
$`\mathrm{ct}(x_0)=7`$ both modulo 12 and modulo 13, but the integer 7 does.

### Compilation

The compiler performs the following steps.

1. Split degree $`d'`$ into $`d'/d`$ proof-ring components, using the full quadratic
   multiplication identity, including squared variables and negacyclic wraparound.
2. Assign bounded variables to the Ajtai part of the commitment, or to BDLOP messages where
   `var_placed` puts them, and unbounded variables to BDLOP messages. A subring variable commits
   only the components that can be nonzero, and an exactly bounded one the bits of each
   component; the lowered equations then get, for each lowered variable, the committed
   polynomial, zero, or the bits' affine form.
3. For each constraint of modulus $`p_j\ne q`$, lift a linear equation with the implicit
   quotient $`p_j^{-1}f_j(s)\bmod q`$; for a quadratic equation or a constant-coefficient
   clause, add committed carries $`c_j`$ with $`f_j(s)-p_jc_j\equiv0\pmod q`$ and range rows for
   them. The carries of the $`m`$ lifted constant-coefficient clauses are packed, $`d`$ to a
   proof-ring polynomial: clause $`i`$, in declaration order, holds coefficient $`t=i\bmod d`$
   of $`C_r`$, $`r=\lfloor i/d\rfloor`$, and its evaluation equation adds $`-p_jX^{-t}C_r`$.
   Then add a range row for each proof-ring polynomial of an $`\ell_\infty`$ variable.
4. Check conservative integer bounds and require a large enough proof modulus for each lifted
   constraint $`j`$: $`q>2(F_j+28\sigma_4p_j)`$, where $`F_j`$ bounds the integer value of the
   constraint on every witness within the norms that extraction gives (the declared ones for
   exact-norm and binary blocks, $`E`$ for $`\ell_\infty`$ blocks). Unbounded variables cannot
   occur in a lifted expression.
5. Refuse a statement that violates the range condition; `requirements` checks it, and `compile`
   calls `requirements` first.

Source degrees must be divisible by the proof degree. The statement modulus and every constraint
modulus other than the proof modulus must be invertible modulo it, which a prime proof modulus
above them ensures.

`requirements(d, q)` exports the dimensions and integer bounds needed for parameter selection,
and `blocks(d)` each block's rows, norm and placement. `Requirements::lifted_moduli` lists every
lifted modulus with its largest $`F_j`$ when a lifted constraint has a modulus other than the
statement modulus; `Requirements::linf`, present only for statements with $`\ell_\infty`$
variables, holds the bounds that depend on $`E`$; `Requirements::approx_alpha_squared` bounds
the squared norm of every honest range vector slot by slot, and the parameter tool may size
$`\sigma_4`$ from it (see the [parameter tool](../tools/README.md#per-slot-range-width)). The
prover's rejection constant keeps $`n'd\cdot`$`linf_bound`$`^2`$, which its witness check
enforces, so rejection sampling stays exact at every width `TboxParams::check` accepts.
`compile(params)` checks the selected dimensions and the lifting inequalities again,
`map_witness` checks each constraint modulo its own modulus on the witness integers and computes
the carries, `Compiled::prove` and `verify` handle typed proofs and `prove_bytes` and
`verify_bytes` wire proofs. Deterministic variants end in `_with_seed`.

`lin::compile` is a convenience wrapper for `A w + t = 0` and a partition into named norm
blocks; `lin::compile_placed` puts the blocks it names in the BDLOP part. The transcript of a
compiled statement binds the whole compiled statement, the parameters, the public seed and the
application context. `examples/possession.rs` is a complete cross-ring quadratic proof;
`tests/statement.rs`, `tests/compiler_paths.rs`, `tests/linf.rs`, `tests/linf_exact.rs` and
`tests/subring.rs` cover the other paths.

## Bounds in $`\ell_\infty`$

**Approximate bounds.** `Norm::Linf(β)`, $`\beta\ge1`$, puts each proof-ring polynomial of the
variable in the one approximate range proof, after the carries and quotients, under a common
bound `linf_bound` that is at least every $`\beta`$ and every honest carry or quotient bound.
The witness map refuses a coefficient above $`\beta`$, but the range proof checks only
`linf_bound`, and a proof bounds an extracted coefficient only by $`E=2\cdot`$`z4_bound`, the
same for every such variable. `Compiled::extraction_bound(name)` reports it. The compiler uses
$`E`$ wherever soundness bounds an integer: in $`F_j`$ of the lifting check and in the range
condition, and it requires $`2E+1<q`$. Because $`E`$ depends on `log_sigma[3]`, which the
parameter tool chooses after `requirements`, `Requirements::linf` exports, for each lifted
constraint that reads an $`\ell_\infty`$ variable, its modulus and the polynomial
$`F_j(E)=`$`exact`$`+`$`linear`$`\cdot E+`$`quadratic`$`\cdot E^2`$, and the range limit on
$`E`$ where it applies; `compile` checks the same inequalities with the actual $`E`$.

**Exact bounds.** `Norm::LinfExact(β)`, $`1\le\beta<2^{63}`$, bounds every coefficient by
$`\beta`$ exactly. The compiler writes each coefficient as $`v=\sum_{b<n}c_bx_b-\beta`$ with
$`n=\mathrm{bitlen}(2\beta)`$ bits $`x_b`$ and the weights
$`c_b=\lfloor(2\beta+2^b)/2^{b+1}\rfloor`$: $`(1,1)`$ for $`\beta=1`$, $`(5,3,1,1)`$ for
$`\beta=5`$. Each committed proof-ring polynomial of the variable becomes $`n`$ bit polynomials,
in the variable's place in its part and in the binary block; $`v`$ itself is not committed. In
every lowered equation the variable's component is replaced by $`\sum_bc_bB_b-\beta J`$, with
$`J`$ the all-ones polynomial, and the products are expanded. The witness map refuses a
coefficient above $`\beta`$ and writes the greedy bits of $`v+\beta`$, taken from the exact
integer. `Compiled::extraction_bound` reports `Extraction::LinfExact(β)`.

**Which to choose.** `Norm::Linf` adds one range row per committed polynomial and no bits, but
it proves only the relaxed bound $`E`$, which is typically many orders of magnitude above
$`\beta`$ (about $`2^{37.6}`$ for $`\beta=1`$ in `tests/tag_preimage.rs`). Every lifted
constraint that reads the variable is sized with $`E`$, so a constraint that multiplies two such
variables needs a much larger proof modulus: in `tests/tag_preimage.rs` the form with
`Norm::Linf(1)` needs $`q\approx2^{98.9}`$, the one with `Norm::LinfExact(1)`
$`q\approx2^{41.1}`$. Use `Norm::Linf` where the relaxed bound suffices and the variable does
not enter products of lifted constraints; use `Norm::LinfExact` where the exact bound matters,
at a cost of $`\mathrm{bitlen}(2\beta)`$ committed bit polynomials per committed polynomial.
Carries and quotients always stay in the approximate block.

## Subring variables

`var_subring(name, count, degree, norm, placement)` declares `count` elements of the subring
$`\{v(X^K)\}`$ of degree $`d_v`$ = `degree`, $`K=d'/d_v`$, a power of two with
$`64\le d_v\le d'`$; `var_placed` is the case $`d_v=d'`$. A witness element is a statement-ring
polynomial that is 0 at every index not divisible by $`K`$ (`iso::embed` builds one), and
`coefficient_in(ring, name, element, i)` is the variable times $`X^{-Ki}`$, whose constant
coefficient is coefficient $`i`$ of $`v`$. Lowered to proof degree $`d`$ with $`d\mid d_v`$,
only the $`d_v/d`$ components $`c\equiv0\pmod K`$ of an element can be nonzero; only those are
committed. A proof degree above $`d_v`$ is refused (“target ring”), as one of its components
would mix the subring's coefficients with positions that nothing constrains. The witness map
refuses a nonzero coefficient outside the subring. A binary block's number of coefficients and
an $`\ell_\infty`$ block's Euclidean norm use $`\mathrm{count}\cdot d_v`$, in `alpha_squared`,
in the per-slot range bound and in the integer bounds.

## Placement

The ABDLOP commitment has two parts (LNP22 §3.1). The Ajtai part $`s_1`$ holds short vectors:
`alpha_squared` bounds its Euclidean norm, and the proof sends its masked opening with a
Gaussian whose width $`\sigma_1`$ grows with that bound. The BDLOP part $`m`$ holds messages,
which the commitment carries at full size modulo $`q`$. `var` puts every bounded block in the
Ajtai part; `var_placed(name, length, norm, Placement::Bdlop)` puts it in the BDLOP part, and an
unbounded block in the Ajtai part is refused. A placed block keeps its own proof, since every
norm statement is an affine map of both parts (LNP22 §5.2 states them over
$`(s_1,\sigma(s_1),m,\sigma(m))`$), and the witness map checks its norm as usual. In the
requirements it leaves `m1` and `alpha_squared` and joins `l`. The compiled order is the Ajtai
part, then the placed bounded blocks, the unbounded blocks and the carries.

Moving a block with a large bound shrinks $`\sigma_1`$ and the MSIS bound, at the cost of its
rows at full size: in `tests/placement.rs` a statement with $`\|w\|^2\le2^{24}`$ on one
polynomial and $`\|s\|^2\le1000`$ on ten proves with a smaller estimate and a shorter proof with
`w` in the BDLOP part. `Statement::blocks(d)` exports what a tool needs to move blocks without
the statement, in JSON as
`{"name": "w", "rows": 2, "norm": {"l2_squared": 16777216}, "placement": "bdlop"}`; `rows`
counts the committed polynomials (the bits, for an exact block). The parameter tool's `derive`
searches the placements of the blocks a request calls movable and keeps the smallest estimate.
Parameters for one placement do not compile another: `compile` refuses them with “compiled
witness dimensions” or “bounded witness norm budget”.

## Soundness arguments of the compiler

LNP22 proves the protocol; the compiler adds the steps below, whose arguments are Jali's own.
They assume what the toolbox's extraction gives: an extracted witness that satisfies every
compiled equation modulo $`q`$, exact-norm and binary blocks within their declared bounds, and
every coefficient of the approximate range vector within $`E=2\cdot`$`z4_bound` (LNP22 Lemma
2.7).

**Lifting, per constraint.** LNP22 proves a relation modulo one modulus inside a proof modulo
another by proving it over the integers, with the quotient shown small by the approximate range
proof (§1.3, Eq. (12)). For a constraint $`f_j`$ of modulus $`p_j\ne q`$, the proof shows
$`f_j(\bar s)-p_j\bar c_j\equiv0\pmod q`$ with $`|f_j(\bar s)|\le F_j`$ and $`|\bar c_j|\le E`$.
If $`q>F_j+p_jE`$, the integer $`f_j(\bar s)-p_j\bar c_j`$ is a multiple of $`q`$ below $`q`$ in
absolute value, hence zero, and $`f_j(\bar s)\equiv0\pmod{p_j}`$. The condition involves only
the constraint's own $`F_j`$ and $`p_j`$, so each lifted constraint is checked separately, and
$`p_j`$ need not be prime. The compiler checks $`q>2(F_j+28\sigma_4p_j)`$, which implies it
because $`E=2\cdot`$`z4_bound`$`\le32\sigma_4`$.

**Carry packing.** In $`\mathbb Z_q[X]/(X^d+1)`$, $`\mathrm{ct}(X^{-t}C)=C_t`$ for $`0\le t<d`$:
for $`0\le u<d`$ the exponent $`u-t`$ is 0 only at $`u=t`$, and otherwise $`X^{u-t}=-X^{d+u-t}`$
has an exponent in $`(0,d)`$. So the evaluation equation of clause $`i`$ gives
$`\mathrm{ct}(f_i(\bar s))-p_i\bar C_{r,t}\equiv0\pmod q`$; the range proof bounds every
coefficient of $`\bar C_r`$ by $`E`$, and the argument above applies to each clause with its own
$`F_i`$ and $`p_i`$.

**Subring lowering.** With $`d\mid d_v`$, so $`K\mid k`$ for $`k=d'/d`$, component $`c`$ of
$`V=v(X^K)`$ holds $`V_{kj+c}`$ at position $`j`$, which is nonzero only if $`K\mid kj+c`$, that
is $`K\mid c`$; for $`c=rK`$ it is component $`r`$ of $`v`$ split as a polynomial of degree
$`d_v`$. An extracted element is the join of its committed components and zeros, which lies in
the subring, and the compiled equations are the exact lowering of the constraints at that
element.

**Exact-bound weights.** Let $`B=2\beta`$. By Hermite's identity
$`\lfloor y+\tfrac12\rfloor=\lfloor2y\rfloor-\lfloor y\rfloor`$ at $`y=B/2^{b+1}`$,
$`c_b=\lfloor B/2^b\rfloor-\lfloor B/2^{b+1}\rfloor`$, so the weights telescope to $`B`$ and
those after $`b`$ sum to $`a=\lfloor B/2^{b+1}\rfloor`$. As
$`\lfloor B/2^b\rfloor\in\{2a,2a+1\}`$ and it is at least 1 for $`b<n`$, $`1\le c_b\le a+1`$.
Every subset sum therefore lies in $`[0,B]`$, and greedy digits reach every $`x\in[0,B]`$: in
index order the remainder stays at most the sum of the weights not yet visited, which is 0 after
the last. (Plain powers of two are wrong: for $`B=10`$ they reach 15.) Extraction gives binary
bits (LNP22 Lemma 5.2 with the binary lifting bound), so each extracted coefficient
$`\sum_bc_b\bar x_b-\beta`$ is an integer in $`[-\beta,\beta]`$; a library test checks the
weights exhaustively for $`B\le2^{12}`$.

## Application context

Every proof, at every layer, takes an application `context: &[u8]`, and verification must be
given the same bytes. The empty context is allowed. Use it for what the proof must be bound to
besides the statement, such as a session identifier, a verifier's nonce or a protocol tag.

Every transcript starts in `transcript::Transcript::new`, which absorbs, each length-framed, the
protocol label, the parameters, the public seed and a field under the label `statement`. The
lower layers (`Abdlop`, `quad`, `quad_many`, `quad_eval`) pass the context as that field, so it
is absorbed before the commitment and the equations. The toolbox (`tbox`, `Prover` and
`Verifier`, `Compiled`) passes its version tag `LNP22-toolbox-v3` there instead and absorbs the
context under the label `application-context` right after the commitment and before the
equations, so the range-projection challenge and every later challenge depend on it. Its inner
evaluation proof takes the transcript digest after the range responses as its context.

## Seeds

The deterministic variants take a 32-byte seed, which must be secret and uniformly random. A
seed may be reused, for commitments and proofs alike. Each call derives its secret keys from the
seed and its own inputs, as deterministic signature nonces do (EdDSA, RFC 6979), and reads every
random stream under those keys, one domain per purpose:

| Randomness | Key inputs besides the seed |
| --- | --- |
| Commitment of `Abdlop::commit_with_seed` | parameters, public seed, bounded witness, messages |
| Masks and coins of the opening proof, also inside `quad`, `quad_many` and `quad_eval` | caller (`Abdlop` or `quad`), transcript digest before the first prover message (parameters, public seed, context, commitment, equations), opening |
| Garbage polynomials of `quad_eval` | transcript digest before the garbage is committed, opening |
| Toolbox (`tbox`, `Prover`, `Compiled`): its commitment, range masks, signs, coins and the seed of its evaluation proof | parameters, public seed, statement, context, witness |

Identical calls therefore return identical outputs, and calls that differ in any input read
independent randomness. The independence is computational: it assumes that SHAKE128, keyed by
the secret seed, and AES-256 behave as pseudorandom functions. The encoding is in
[encodings](encodings.md#secret-key-derivation).

What remains is determinism. The same seed and identical inputs give the identical output, so
two commitments to the same values from one seed are equal and hence linkable, and so are two
proofs of the same statement, context and witness. With a reused seed, zero-knowledge (and
hiding, for commitments) holds only relative to the equality pattern of the inputs: a simulator
must be told which calls repeat an earlier one. Uses that need multi-theorem zero-knowledge must
use fresh seeds, for instance `commit` and `prove` with a `rand_core::CryptoRng`, which draw one
per call. Where faults can be injected, use fresh seeds as well: a fault that changes a
challenge but not the key can yield two accepted responses with the same masks, and
$`z_1-z_1'=(c-c')s_1`$ gives the witness once $`c-c'`$ is invertible.
