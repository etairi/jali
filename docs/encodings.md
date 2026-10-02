# Encodings and transcripts

## Proof encoding

Toolbox proofs have no wire-controlled vector lengths. Validated parameters determine the
layout, including which range blocks exist. In order:

| Field | Encoding |
| --- | --- |
| $`t_B`$, including masks and signs, garbage commitments, final $`t`$ | Uniform coefficients modulo $`q`$ |
| $`h`$ | Uniform coefficients modulo $`q`$ |
| $`t_{A,1}`$ | Fixed-width rounded commitment coefficients |
| $`c`$ | Coefficients in $`[-\omega,\omega]`$ |
| hint | Prefix-free hint code (LNP22 Table 3) |
| $`z_1,z_{2,1},z_3,z_4`$ | Signed unary quotient and fixed-width Gaussian remainder |
| end | One bit followed by zero bits to the byte boundary |

Bits are least-significant first. A uniform coefficient takes $`\mathrm{bits}(q-1)`$ bits, up to
256, and a coefficient of $`t_{A,1}`$ takes $`\mathrm{bits}(q-1)-D`$ bits. Response and hint
codes hold `i128` values; a response or hint that does not fit cannot be encoded, and no valid
one comes near that bound. Uniform coefficients outside the modulus, oversized unary codes,
noncanonical hints, truncation, nonzero padding and trailing bytes are rejected. Decoding
accepts at most 16 MiB and never takes dimensions from untrusted bytes.

Decoding does not establish validity: use `Verifier::verify_bytes` or `Compiled::verify_bytes`.
The parameter identifier, the public statement and the application context travel outside the
proof: the verifier must select the expected parameters and statement before decoding, and
supply the context itself when verifying. A proof carries no copy of its context.

## Transcripts

Every transcript starts from SHAKE128 of the tag `LNP22-RUST-v1` and absorbs, each
length-framed, a protocol label, the parameters, the public seed and one more field
(`transcript::Transcript::new`). The lower layers put the application context in that field. The
toolbox puts its own version tag `LNP22-toolbox-v2` there and absorbs the context after the
commitment (see [application context](protocol.md#application-context)). Messages are chained
through 32-byte SHAKE128 digests, and each challenge has its own domain. The transcript binds
the full parameter fingerprint, the public seed, the application context and every public
equation. The wire and transcript formats are not a promise of future compatibility.

## Parameters in the transcript

Transcripts absorb the parameter bytes of `Abdlop::parameter_bytes`: the parameter set's own
encoding (`TboxParams::transcript_bytes`), then the modulus, the degree and the derived bounds.
Two fields take a wider form only where needed:

- **Prime factors.** When every factor is below $`2^{64}`$, the narrow form:
  $`\mathrm{LE64}(n)`$ and each factor as $`\mathrm{LE64}`$. Otherwise the wide form:
  $`\mathrm{LE64}(n+2^{63})`$, then for each factor $`p`$ the byte count $`\mathrm{LE64}(k)`$
  with $`k=\lceil\mathrm{bits}(p)/8\rceil`$, and the $`k`$ low-order bytes of $`p`$. Narrow
  counts are one or two, so the top bit of the first word tells the forms apart, and the counts
  make the wide form self-delimiting.
- **The modulus** $`q`$: 16 little-endian bytes when $`q<2^{128}`$, 32 otherwise. The prime
  factors, which come first, determine $`q`$ and hence the width.

The derived bounds are 16-byte integers: `TboxParams::check` refuses a set whose response bounds
do not fit 128 bits.

## Serde forms

The `serde` feature serializes parameters and compilation requirements. It does not replace the
proof codec, which is the only encoding of proofs. Integer fields that can exceed $`2^{64}`$
(prime factors and bounds) are written in JSON as numbers below $`2^{64}`$ and as decimal
strings otherwise. Readers accept both forms, but refuse a JSON number of $`2^{64}`$ or more,
which a JSON parser may have rounded to a float, and a string with a sign, leading zeros or
other characters. Parameter sets and `BlockRequirement` refuse unknown fields.

Reading a number or a string needs `deserialize_any`, which formats that do not describe
themselves, such as postcard or bincode, lack. Formats that are not human-readable therefore
take fixed forms: a prime factor or another 256-bit value is 32 little-endian bytes, and the
other integer fields are the format's own integers (tested with postcard). Such formats read
fields by position and so always write every field, an absent optional field as the format's
empty option.

In `Requirements`, JSON leaves out the optional fields when they are absent or empty:
`approx_alpha_squared` (a number or a decimal string; `null` reads as absent), `lifted_moduli`
(a list of objects with `modulus` and `max_integer_coefficient`) and `linf` (an object with
`lifting`, a list of objects with `modulus`, `exact`, `linear` and `quadratic`, and
`extraction_limit`, `null` or a value). `Statement::blocks` exports a list of
`BlockRequirement`: objects with `name`, `rows`, `norm` and `placement`, where a `Norm` is
`{"l2_squared": B}`, `"binary"`, `{"linf": β}`, `"unbounded"` or `{"linf_exact": β}` and a
`Placement` is `"ajtai"` or `"bdlop"`. Formats that read by position write a `Norm` as its
variant index, 0 to 4 in the order just listed, and its bound.

## Uniform sampling modulo large moduli

`rand::uniform_u256` samples modulo any $`m<2^{256}`$ by the rule of `rand::uniform`: each pass
reads $`\lceil kw/8\rceil`$ bytes for the $`k`$ values still missing, where
$`w=\mathrm{bits}(m-1)`$, splits them least significant bit first into $`w`$-bit words and keeps
the words below $`m`$, for at most 1024 passes. Below $`2^{128}`$ it is `rand::uniform` itself,
with the same values and bytes, which the known-answer vectors pin. The kept words are uniform
and independent, and each is kept with probability above 1/2. Public matrices, the $`\Gamma`$
weights of the evaluation proof, its garbage polynomials and the weights of `quad_many` are
drawn this way modulo $`q`$.

## Secret key derivation

Every secret stream is AES-256-CTR under a key derived from the caller's seed (see
[seeds](protocol.md#seeds)). With $`f(x)=\mathrm{LE64}(|x|)\,\|\,x`$, the key is the first 32
bytes of

```math
\mathrm{SHAKE128}\big(f(\texttt{jali/key/v1})\,\|\,f(label)\,\|\,f(seed)\,\|\,f(part_1)\,\|\cdots\|\,f(part_n)\big).
```

Every field carries its length, so distinct sequences of fields are distinct inputs. The labels
and parts are:

| Label | Parts, in order |
| --- | --- |
| `abdlop/commitment` | scheme fingerprint; $`s_1`$; $`m`$ |
| `abdlop/opening-proof` | caller, `opening` (`Abdlop`) or `quadratic` (`quad`, also inside `quad_many` and `quad_eval`); transcript digest before the first prover message; $`s_1`$; $`m`$; $`s_2`$ |
| `quad-eval/garbage` | transcript digest after both equation lists, before the garbage commitments; $`s_1`$; $`m`$; $`s_2`$ |
| `tbox/proof` | scheme fingerprint; statement digest; context; $`s_1`$; $`m`$ |

The scheme fingerprint is the transcript digest of the parameter bytes and the public seed under
the protocol label `abdlop-scheme`. The statement digest is the transcript digest of the
extended scheme's parameters and public seed under `LNP22-toolbox-statement`, followed by the
four equation families and the range parameters exactly as the toolbox transcript absorbs them.
Secret vectors are encoded as their centred coefficients in order, each a little-endian
two's-complement integer of 16 bytes when $`q<2^{128}`$ and of 32 bytes otherwise. Every label
binds $`q`$ before the secrets, through the scheme fingerprint or a transcript digest, both of
which absorb the parameters. For the toolbox, $`s_1`$ is the caller's bounded witness without
slack. The toolbox draws the seed of its evaluation proof from its own stream, and that proof
derives its keys from it as above. The caller part of the opening-proof key keeps the keys of
the opening proof and the quadratic proof apart even if their transcript digests were equal
(they differ, because the quadratic proof absorbs its equation).
