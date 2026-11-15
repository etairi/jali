# Encodings and transcripts

Since version 0.2 every proof follows draft-irtf-cfrg-fiat-shamir-04 (Orrù and Yun, "the FS
draft" below): one duplex sponge per proof, started from a session identifier, absorbing the
instance once and then every prover message as its serialization, and squeezing every verifier
message with a fixed length. The proofs are NARG strings in the draft's sense: the
concatenation of the serialized prover messages, with the final round in the compact flavor of
draft-irtf-cfrg-sigma-protocols-04 (the challenge in place of the part of the final message
that the verifier recomputes). The drafts are Internet-Drafts and can still change, so the
tags name the revision they follow through the transcript version `V02`. Proofs of version 0.1
do not verify under 0.2.

## Transcript

The sponge is the XOF duplex sponge of FS draft §4 with SHAKE128 (§9.1, rate 168 bytes):
`Init(session_id)` absorbs $`session\_id\,\|\,0^{136}`$, absorbing appends with no separator,
and consecutive squeezes continue one output stream, which the next absorb restarts. The draft's
SHAKE128 test vectors pass against it (`src/fiat_shamir.rs`). A proof's session identifier is
`DeriveSessionID(tag)` (FS draft §5.1) for

```math
tag=\mathrm{ASCII\ tag}\,\|\,\mathrm{LE32}(|ctx|)\,\|\,ctx,
```

with the caller's application context $`ctx`$ of fewer than $`2^{32}`$ bytes. The ASCII tags name
the crate, the transcript version, the argument, the codec version `C1` (the serializations and
decodings below) and the hash suite:

| Proof | ASCII tag |
| --- | --- |
| toolbox (`tbox::prove_with_seed`, `Prover`, `Compiled`) | `JALI-V02-LNP22-TBOX-C1-SHAKE128` |
| toolbox, straight-line extractable mode | `JALI-V02-LNP22-TBOX-SL-C1-SHAKE128` |
| opening proof on its own (`Abdlop::prove_with_seed`) | `JALI-V02-ABDLOP-OPENING-C1-SHAKE128` |
| one quadratic equation on its own (`quad::prove_with_seed`) | `JALI-V02-LNP22-QUAD-C1-SHAKE128` |
| several quadratic equations on their own (`quad_many`) | `JALI-V02-LNP22-QUAD-MANY-C1-SHAKE128` |
| evaluation proof on its own (`quad_eval`) | `JALI-V02-LNP22-QUAD-EVAL-C1-SHAKE128` |

No ASCII tag is a prefix of another, and the context is length-prefixed, so distinct (argument,
context) pairs give distinct tags, as FS draft §5.1 requires of composite tags. Nothing but the
instance and the messages below enters the sponge: data that they determine, such as folded,
combined or projected equations and the extended commitments of the inner layers, is not
absorbed again.

## Instances

The instance is absorbed once, right after `Init`, and every instance starts with the scheme:
`SerializeVarLenString` of the parameter bytes (`Abdlop::parameter_bytes`, see below), then the
32-byte public seed, which determines the public matrices $`A_1,A_2,B`$. Lists of equations
follow as their number ($`\mathrm{LE32}`$), then each equation's encoding
(`QuadEq::to_bytes`) as `SerializeVarLenString`.

| Proof | Instance after the scheme |
| --- | --- |
| toolbox | the four equation families of the statement: quadratic, evaluation, exact maps, approximate maps |
| toolbox, extractable | the same, then `SerializeVarLenString` of the extraction set (`ExtractionParams`) |
| opening proof | the commitment (its serialization below) |
| one quadratic equation | the commitment, then the equation's encoding as `SerializeVarLenString` |
| several quadratic equations | the commitment, then the equation list |
| evaluation proof | the commitment, then the list of full equations and the list of evaluation equations |

Every field is length-prefixed or has a length that the parameters fix, so the encoding is
prefix-free and has a left inverse, as FS draft §3.1 requires.

## IO patterns

`A` absorbs a prover message (its serialization below) and `S` squeezes a verifier message.
The toolbox runs one transcript through all its layers:

| Step | Plain mode | Extractable mode |
| --- | --- | --- |
| 1 | | `S` the LinHC key |
| 2 | `A` the commitment $`(t_{A,1},t_B)`$ | `A` the commitment, then the LinHC commitment $`(t'_1,t'_2)`$ |
| 3 | `S` the projection matrices, exact block then approximate block | the same |
| 4 | `A` the range responses $`z^{(e)},z^{(d)}`$, then the garbage commitments | the same |
| 5 | `S` $`\Gamma`$ | the same |
| 6 | `A` $`h`$ | the same |
| 7 | `S` $`\mu`$ | the same |
| 8 | `A` $`(t,v,w_1)`$ | `A` $`(t,v)`$, the LinHC mask commitment $`(w'_1,w'_2)`$, $`w_1`$ |
| 9 | `S` the challenge $`c`$ | the same |

The proofs of the lower layers on their own run the end of this pattern after their instance:
the evaluation proof from step 4 on with only the garbage commitments, the many-equation proof
from step 7, the quadratic proof from step 8, and the opening proof absorbs $`w_1`$ alone and
squeezes $`c`$. The tests `tbox::io_pattern_tests` record the verifiers' calls on the sponge and
check this pattern, the length of every squeeze, and that the proof is exactly the absorbed
messages that it carries followed by the final round.

The prover forks the transcript for each rejection-sampling attempt: an attempt of the range
proofs starts from the state after the instance (and the LinHC key), an attempt of the opening
proof from the state after $`\mu`$. Only the accepted attempt's messages are those of the proof.

## Codec C1

### Serializations

Every prover message is a sequence of fixed-width fields, written least significant bit first
and coefficient after coefficient, as FIPS 204 §7.1 packs lattice coordinates (the FS draft,
Appendix A.2.4, lets an application pin such a serialization in place of its default):

| Field | Code | Width |
| --- | --- | --- |
| $`t_B`$, garbage commitments, $`h`$, $`t`$, $`v`$ | canonical representative modulo $`q`$ (`SimpleBitPack`) | $`\mathrm{bitlen}(q-1)`$ |
| $`t_{A,1}`$ | high part, at most that of $`q-1`$ | $`\mathrm{bitlen}(q-1)-D`$ |
| $`w_1`$ | high part in $`[0,m)`$, $`m=(q-1)/\gamma`$ | $`\mathrm{bitlen}(m-1)`$ |
| LinHC $`t'_1,t'_2,w'_1,w'_2`$ | canonical representative modulo $`q'`$ | $`\mathrm{bitlen}(q'-1)`$ |
| $`z_1,z_{21},z^{(e)},z^{(d)}`$ | $`b-x`$ for $`b=2^{w-1}-1`$ (`BitPack(x, b, b)`) | $`w=t+4`$ for the width $`1.55\cdot2^t`$ |
| LinHC opening $`z'_1,z'_2,z'_3`$ | the same | $`t'+4`$ |
| hints | the same | `CheckedParams::hint_width` |
| challenge $`c`$ | coefficients $`0`$ to $`d/2-1`$, each as $`\omega-c_i`$ (`BitPack(c, ω, ω)`) | $`\mathrm{bitlen}(2\omega)`$ |

A response code of width $`t+4`$ holds every coefficient of absolute value at most
$`2^{t+3}-1`$, about $`5.16\sigma`$. The prover restarts every attempt with a response
coefficient beyond its width or a hint beyond the hint width, tests of the accepted response and
the public data only (see the [security notes](security.md#fixed-width-codes)). The other
coefficients of a challenge follow from the first half: coefficient $`d/2`$ is zero and
coefficient $`d-i`$ is minus coefficient $`i`$. With degrees 64 and 128 every field fills whole
bytes.

The hint width is $`1+\mathrm{bitlen}(H)`$ for $`H=\max(1,\lceil9s\rceil)`$ and
$`s=\sqrt{\sigma_2^2+(d-1)\omega^2(1+4^{D-1}/3)}/\gamma`$. In a model of the hints (a heuristic),
$`z_{22}/\gamma`$ is a rounding error uniform on $`(-1/2,1/2]`$ plus a normal of deviation at
most $`s`$, so a hint above $`H`$ needs that normal beyond $`9s`$, which happens with probability
below $`2^{-61}`$ per coefficient. The model sets the restart rate only. Every shipped set has
2-bit hints ($`s`$ from 0.026 to 0.104).

Deserialization is strict, as FS draft §3.3 requires: every length comes from the parameters,
every code must be in its field's range (below the modulus, a high part at most that of
$`q-1`$, a signed code at most $`2b`$, a challenge code at most $`2\omega`$, a hint in the
canonical interval $`(-m/2,m/2]`$), and no byte may follow the last field. Decoding accepts at
most 16 MiB and never takes dimensions from untrusted bytes.

### Decodings

Each verifier message squeezes a number of bytes that its type and the instance fix, and its
decoding is infallible (FS draft §3.2):

| Verifier message | Squeezed bytes | Decoding |
| --- | --- | --- |
| projection matrices | 2 bits per entry, $`256\times n_{ex}d`$ then $`256\times n'd`$ entries, row-major | $`b_{2i}-b_{2i+1}`$, `SamplePolyCBD` of FIPS 203 with $`\eta=1`$: $`P(0)=1/2`$, $`P(\pm1)=1/4`$ |
| $`\Gamma`$ | $`N_s+16`$ per scalar, $`\lambda\times\vert\mathrm{evaluation\ equations}\vert`$ scalars, row-major | `DecodeUint` modulo $`q`$ |
| $`\mu`$ | $`N_s+16`$ per coefficient, one ring element per final equation | `DecodeUint` modulo $`q`$ |
| LinHC key $`(A,B)`$ | $`N_s+16`$ per coefficient, $`2m'n'`$ ring elements, row-major | `DecodeUint` modulo $`q'`$ |
| challenge $`c`$ | 4032 (`rand::CHALLENGE_BYTES`, 24 blocks of the rate) | `rand::challenge_from_bytes` |

`DecodeUint` reads its bytes as a little-endian integer and reduces it modulo the modulus, with
bias at most $`2^{-128}`$ per value (FS draft, Appendix A.1.2). The binomial decoding is exact.

The challenge decoding reads the bytes as $`k`$-bit words, $`k=\mathrm{bitlen}(2\omega)`$, least
significant bit first, keeps the words below $`2\omega+1`$ as coefficients $`w-\omega`$, takes
each $`d/2`$ kept coefficients as a draw of the $`\sigma_{-1}`$-stable form above, and returns
the first draw with $`\|(\sigma_{-1}(c)c)^{32}\|_1\le\eta^{64}`$ (LNP22 §2.7). If no draw in the
4032 bytes passes, it returns the constant 1, which is in the challenge set. This is an
alternative decoding in the sense of FS draft §3.2: on uniform bytes its output is uniform on
the challenge set except for the weight of that fallback, so its bias is the probability that
no draw passes. For the two challenge sets of the parameter check that probability is at most
$`2^{-210.9}`$ ($`d=64,\omega=8`$, 6451 words) and $`2^{-208.0}`$ ($`d=128,\omega=2`$, 10752
words) if at most a quarter of the draws fail the $`\eta`$ test, and at most $`2^{-582}`$ if a
fiftieth do, computed exactly in Sage (rational arithmetic over the binomial law of the kept
words). About 1% of draws fail it (measured, [security notes](security.md#challenges)). The
bias is far below the soundness error of the interactive proofs, and below $`2^{-144}`$ over
$`2^{64}`$ oracle queries. The draws come from the start of the squeezed stream, so a longer
budget would change no challenge, only the weight of the fallback. An independent Python
oracle (`tools/kat/generate.py`) pins the decoding.

## Proof encoding

A toolbox proof is the concatenation of its messages' serializations in protocol order, the
parts of the final message that the verifier recomputes ($`v`$, $`w_1`$ and the LinHC mask
commitment) left out, with the challenge and the response at the end:

| Message | Fields |
| --- | --- |
| 1 | $`t_{A,1}`$, $`t_B`$ ($`\ell+e+a+1`$ polynomials for $`e`$ exact and $`a`$ approximate slots), in the extractable mode $`t'_1,t'_2`$ |
| 2 | $`z^{(e)}`$, $`z^{(d)}`$, the garbage commitments ($`\lambda/2`$ polynomials) |
| 3 | $`h`$ ($`\lambda/2`$ polynomials) |
| 4 | $`t`$ |
| final round | $`c`$, $`z_1`$, $`z_{21}`$, the hints, in the extractable mode $`z'_1,z'_2,z'_3`$ |

An opening proof on its own is its final round only: $`c`$, $`z_1`$, $`z_{21}`$ and the hints.
Every proof of a parameter set has the same length, `CheckedParams::estimated_proof_bytes` for
the toolbox and `ExtractionParams::proof_bytes` in the extractable mode.

Decoding does not establish validity: use `Verifier::verify_bytes` or `Compiled::verify_bytes`.
The parameter identifier, the public statement and the application context travel outside the
proof: the verifier must select the expected parameters and statement before decoding, and
supply the context itself when verifying. A proof carries no copy of its context.

## Where the drafts are not followed literally

- The drafts define no codecs for ring elements, Gaussian responses, hints or lattice
  challenges. Codec C1 pins FIPS 204's bit packing for them, which the tag names.
- The challenge is rejection-sampled, so its decoding is the alternative decoding above, with a
  fixed length and a fallback, instead of `DecodeUint`.
- The prover's keys bind the transcript through 32 bytes squeezed from a copy of the sponge
  after the label `JALI-V02-PROVER-KEY-BINDING`, which leaves the transcript unchanged
  (`Transcript::binding`). The drafts do not specify how a prover derives its randomness.

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
`approx_alpha_squared` (a number or a decimal string, and `null` reads as absent), `lifted_moduli`
(a list of objects with `modulus` and `max_integer_coefficient`) and `linf` (an object with
`lifting`, a list of objects with `modulus`, `exact`, `linear` and `quadratic`, and
`extraction_limit`, `null` or a value). `Statement::blocks` exports a list of
`BlockRequirement`: objects with `name`, `rows`, `norm` and `placement`, where a `Norm` is
`{"l2_squared": B}`, `"binary"`, `{"linf": β}`, `"unbounded"`, `{"linf_exact": β}` or
`{"l2_squared_range": {"min": a, "max": b}}` and a `Placement` is `"ajtai"` or `"bdlop"`.
Formats that read by position write a `Norm` as its variant index, 0 to 5 in the order just
listed, and its bounds (`min`, then `max`, for the last).

## Uniform sampling modulo large moduli

`rand::uniform_u256` samples modulo any $`m<2^{256}`$ by the rule of `rand::uniform`: each pass
reads $`\lceil kw/8\rceil`$ bytes for the $`k`$ values still missing, where
$`w=\mathrm{bits}(m-1)`$, splits them least significant bit first into $`w`$-bit words and keeps
the words below $`m`$, for at most 1024 passes. Below $`2^{128}`$ it is `rand::uniform` itself,
with the same values and bytes, which the known-answer vectors pin. The kept words are uniform
and independent, and each is kept with probability above 1/2. The public matrices (from the
public seed, by AES-256-CTR) and the garbage polynomials of the evaluation proof (from a secret
key) are drawn this way modulo $`q`$. The verifier messages $`\Gamma`$ and $`\mu`$ come from the
transcript by `DecodeUint` instead (above).

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
| `abdlop/commitment` | scheme fingerprint, $`s_1`$, $`m`$ |
| `abdlop/opening-proof/v3` | caller (`opening` for `Abdlop`, `quadratic` for `quad`, also inside `quad_many` and `quad_eval`), binding of the transcript before the first prover message, $`s_1`$, $`m`$, $`s_2`$ |
| `quad-eval/garbage/v2` | binding of the transcript before the garbage commitments, $`s_1`$, $`m`$, $`s_2`$ |
| `tbox/proof/v3` | binding of the transcript after the instance, $`s_1`$, $`m`$ |

The extractable mode replaces the toolbox and opening labels with `tbox/proof-sl/v2` and
`abdlop/opening-proof-sl/v2`, as it reads more streams. A binding is `Transcript::binding`: 32
bytes squeezed from a copy of the sponge after the label `JALI-V02-PROVER-KEY-BINDING`, a
function of the tag (the argument and the context), the instance (the parameters, the public seed
and the statement) and the messages so far. The scheme fingerprint, unchanged since version 0.1,
is the 32-byte state of the chain of 0.1's transcript over the protocol label `abdlop-scheme`,
the parameter bytes, the public seed and an empty statement (`Abdlop::fingerprint`). Secret
vectors are encoded as their centred coefficients in order, each a little-endian
two's-complement integer of 16 bytes when $`q<2^{128}`$ and of 32 bytes otherwise. Every label
binds $`q`$ before the secrets, through the scheme fingerprint or a binding, both of which
absorb the parameters. For the toolbox, $`s_1`$ is the caller's bounded witness without slack.
The toolbox draws the seed of its evaluation proof from its own stream, and that proof derives
its keys from it as above. The caller part of the opening-proof key keeps the keys of the
opening proof and the quadratic proof apart even if their bindings were equal. The suffixes
`/v3`, and `/v2` of the garbage and extractable labels, mark the transcript of version 0.2. The
labels change whenever the way these keys' streams are read changes
([security notes](security.md#randomness-and-seeds)).
