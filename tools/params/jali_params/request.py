"""Version-2 requests: what a parameter set is for, and how to find its modulus and ranks.

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

`modulus` is either `{"search": {...}}` (optional `window_bits`, default 4; `min_bits`, which
keeps $`q\\ge2^{\\text{min\\_bits}-1}`$; and `max_bits`, which keeps $`q<2^{\\text{max\\_bits}}`$,
so that 0 is a cap no modulus meets, not the absence of one) or `{"prime_factors": [...]}` with
an optional `gamma`, even and at least 2. Without `gamma` a fixed modulus takes the
positional form's divisor: the largest even divisor of $`q-1`$ in the window
$`(4\\gamma_0/5,\\gamma_0]`$. The search may choose a divisor outside that window, so to
re-derive a searched set at its own prime, give its `gamma`. `hardness` is
`{"policy": "delta"}`, `{"policy": "beta", "beta_min": 439}` or `{"policy": "external",
"mlwe_rank": k, "mlwe_delta": x, "provenance": "..."}` (`hardness.Policy`).

`statement` is either a `lin::compile` shape or requirements copied from Rust.

- `lin-worst-case`: requirements for the worst case over public data, whose coefficients all
  have the largest centred absolute value (optionally `public_linf`). Each block has `name`,
  `length` and a `norm` with exactly one of `l2_squared` (a bound), `binary: true` or
  `unbounded: true`, and optionally `placement`, `"ajtai"` (the default for a bounded block)
  or `"bdlop"` (the only one for an unbounded block), as `lin::compile_placed` places it.
- `requirements`: the fields of `Statement::requirements` and the `statement_modulus`: the
  nine fields, where `max_integer_coefficient` and `statement_modulus` may be left out only
  when `n_prime` is 0, and `approx_alpha_squared`, `lifted_moduli` and `linf` as Rust writes
  them when present. `approx_alpha_squared`, a per-slot bound on the squared Euclidean norm of
  the approximate-range vector, is refused without a range block and when it is not below
  $`n'd\\beta_\\infty^2`$ (Rust exports it only below); `derive` then sizes $`\\sigma_4`$ from it
  (`derive.Shape`).
  `blocks`, the JSON of `Statement::blocks` at the same proof degree, is optional without an
  $`\\ell_\\infty`$ variable and required with one (`linf`), since only the blocks tell which
  range rows belong to such a variable and what its bound $`\\beta`$ is. A block's `rows` are
  the proof-ring polynomials it commits: for a subring block (`Statement::var_subring`) only
  its nonzero components, and for an exactly bounded one (`{"linf_exact": b}`) their bits,
  which count as binary rows.

Either source may list in `movable` the names of at most 12 bounded blocks whose part the
placement search chooses; the requirements source needs `blocks` for that.

`Statement::requirements` exports at one proof modulus, and a constraint over that modulus is
native there: it adds no range rows, carries or lifted modulus, so the requirements do not show
it, and at another $`q`$ it would be lifted and compile to another shape. For such a statement,
give the requirements source `export_modulus`, the proof modulus of the export. It pins $`q`$:
the request's `modulus` must then be `prime_factors` with that product (a search is refused),
and `check --request` refuses a set at another $`q`$.

The copied fields must describe one statement. With a range block, `linf_bound` must be the
largest of 1, $`\\lceil F_j/p_j\\rceil`$ over the lifted moduli (`lifted_moduli`, or the
statement modulus with `max_integer_coefficient` when it is empty) and the $`\\beta`$ of every
$`\\ell_\\infty`$ block, as `Statement::requirements` computes it, so a mismatched $`F`$, such
as 0 next to a larger `linf_bound`, is refused. `lifted_moduli` must be strictly ascending, and
its largest $`F_j`$ is `max_integer_coefficient`. The `blocks` must add up to `m1`,
`alpha_squared`, `n_bin` and the exact-norm lists, and leave room in `l` and `n_prime` for
carries and lifted rows. As Rust's `Ring` does, the statement ring needs a
modulus in $`[2,2^{256})`$ and a degree that is a power of two from 64 to 1024. Blocks follow
`Statement::var`: unique names of at most 256 bytes, positive lengths, at most 32768
polynomials in all.

Requests are decoded strictly (no duplicate keys, no NaN) and every object is checked for
unknown and missing keys, so a typo is refused instead of ignored. Integers follow the number
rule of `jsonio`.
"""
import math

from . import hardness, jsonio, partition, rustcheck

SCHEMA = "jali-params-request/2"
TOP_KEYS = ("schema", "id", "mode", "degree", "modulus", "hardness", "statement", "note")
TOP_REQUIRED = ("schema", "id", "degree", "modulus", "hardness", "statement")
LIN_KEYS = ("source", "statement_degree", "statement_modulus", "rows", "blocks", "public_linf",
            "movable")
LIN_REQUIRED = ("statement_degree", "statement_modulus", "rows", "blocks")
BLOCK_KEYS = ("name", "length", "norm", "placement")
BLOCK_REQUIRED = ("name", "length", "norm")
NORM_KINDS = ("l2_squared", "binary", "unbounded")
PLACEMENTS = ("ajtai", "bdlop")
REQUIREMENTS_STATEMENT_KEYS = ("source", "requirements", "statement_modulus", "blocks",
                               "movable", "export_modulus")
# The nine fields of `statement::Requirements` that every export has; the last one only
# matters with a range block.
REQUIREMENTS_KEYS = ("m1", "l", "alpha_squared", "n_bin", "l2_rows", "l2_bounds_squared",
                     "n_prime", "linf_bound", "max_integer_coefficient")
# The fields Rust writes only when they are not empty.
OPTIONAL_REQUIREMENTS_KEYS = ("approx_alpha_squared", "lifted_moduli", "linf")
EXPORTED_BLOCK_KEYS = ("name", "rows", "norm", "placement")
# Rust's `Ring::with_modulus`: degrees are powers of two from 64 to 1024, moduli in [2, 2^256).
RING_DEGREES = (64, 128, 256, 512, 1024)


class RequestError(ValueError):
    pass


def _int(v, what):
    try:
        return jsonio.decode_int(v)
    except ValueError as e:
        raise RequestError(f"{what}: {e}") from None


def _known(obj, keys, what):
    if not isinstance(obj, dict):
        raise RequestError(f"{what} must be an object")
    if set(obj) - set(keys):
        raise RequestError(f"unknown {what} fields {sorted(set(obj) - set(keys))}")


def _need(obj, keys, what):
    missing = [k for k in keys if k not in obj]
    if missing:
        raise RequestError(f"{what} needs {', '.join(missing)}")


def _list(v, what):
    if not isinstance(v, list):
        raise RequestError(f"{what} must be a list")
    return v


def _modulus(v, what):
    m = _int(v, what)
    if not 2 <= m < 1 << 256:
        raise RequestError(f"{what} {m} is not in [2, 2^256), as Rust's Ring requires")
    return m


def _statement_modulus(s):
    return _modulus(s["statement_modulus"], "statement_modulus")


def _name(v, what):
    if not isinstance(v, str) or not v:
        raise RequestError(f"{what} must be a non-empty string")
    return v


def parse(text):
    try:
        r = jsonio.loads_strict(text)
    except ValueError as e:  # includes json.JSONDecodeError
        raise RequestError(f"not a valid JSON request: {e}") from None
    if not isinstance(r, dict):
        raise RequestError("a request is a JSON object")
    if r.get("schema") != SCHEMA:
        raise RequestError(f"schema must be {SCHEMA}")
    if set(r) - set(TOP_KEYS):
        raise RequestError(f"unknown request fields {sorted(set(r) - set(TOP_KEYS))}")
    _need(r, TOP_REQUIRED, "a request")
    if r.get("mode", "native") != "native":
        raise RequestError("mode must be native")
    if not isinstance(r["id"], str) or not r["id"]:
        raise RequestError("id must be a non-empty string")
    d = r["degree"]
    if isinstance(d, bool) or not isinstance(d, int) or d not in (64, 128):
        raise RequestError("degree must be 64 or 128")
    if "note" in r and not isinstance(r["note"], str):
        raise RequestError("note must be a string")
    return r


def policy_of(r):
    h = r["hardness"]
    name = h.get("policy") if isinstance(h, dict) else None
    _known(h, {"external": ("policy", "mlwe_rank", "mlwe_delta", "provenance"),
               "beta": ("policy", "beta_min")}.get(name, ("policy",)), "hardness")
    if name == "external":
        _need(h, ("mlwe_rank", "mlwe_delta", "provenance"), "external hardness")
        rank = _int(h["mlwe_rank"], "mlwe_rank")
        delta, provenance = h["mlwe_delta"], h["provenance"]
        if rank == 0:
            raise RequestError("mlwe_rank must be positive")
        if isinstance(delta, bool) or not isinstance(delta, (int, float)):
            raise RequestError("mlwe_delta must be a number")
        if not isinstance(provenance, str) or not provenance:
            raise RequestError("provenance must be a non-empty string")
        return None, {"rank": rank, "delta": delta, "provenance": provenance}
    try:
        return hardness.Policy(name, h.get("beta_min")), None
    except ValueError as e:
        raise RequestError(str(e)) from None


def _placement(v, norm, what):
    if v not in PLACEMENTS:
        raise RequestError(f"{what} must be one of {', '.join(PLACEMENTS)}")
    if norm[0] == "unbounded" and v != "bdlop":
        raise RequestError(f"{what}: an unbounded block goes to the BDLOP part; "
                           "Statement::var_placed refuses it in the Ajtai part")
    return v


def lin_shape(s):
    """The statement degree and the blocks (name, length, norm) of a `lin-worst-case`
    statement, checked and decoded."""
    degree, blocks = lin_blocks(s)
    return degree, [(b["name"], b["length"], b["norm"]) for b in blocks]


def lin_blocks(s):
    """The statement degree and the blocks of a `lin-worst-case` statement as dicts with
    `name`, `length`, `norm` and `placement`."""
    _known(s, LIN_KEYS, "statement")
    _need(s, LIN_REQUIRED, "a lin-worst-case statement")
    blocks = s["blocks"]
    if not isinstance(blocks, list) or not blocks:
        raise RequestError("statement.blocks must be a non-empty list")
    out, names = [], set()
    for i, b in enumerate(blocks):
        what = f"statement.blocks[{i}]"
        _known(b, BLOCK_KEYS, what)
        _need(b, BLOCK_REQUIRED, what)
        _name(b["name"], f"{what}.name")
        norm = b["norm"]
        _known(norm, NORM_KINDS, f"{what}.norm")
        kinds = [k for k in NORM_KINDS if k in norm]
        if len(kinds) != 1:
            raise RequestError(f"{what}.norm must give exactly one of {', '.join(NORM_KINDS)}")
        kind = kinds[0]
        if kind == "l2_squared":
            nm = (kind, _int(norm[kind], f"{what}.norm.l2_squared"))
        elif norm[kind] is True:
            nm = (kind,)
        else:
            raise RequestError(f"{what}.norm.{kind} must be true")
        length = _int(b["length"], f"{what}.length")
        # What Statement::var refuses, which lin::compile calls for each block.
        if len(b["name"].encode()) > 256 or b["name"] in names or length == 0:
            raise RequestError(f"{what}: Statement::var refuses a name of more than 256 bytes, "
                               "a repeated name or a length of 0")
        names.add(b["name"])
        default = "bdlop" if kind == "unbounded" else "ajtai"
        out.append({"name": b["name"], "length": length, "norm": nm,
                    "placement": _placement(b.get("placement", default), nm,
                                            f"{what}.placement")})
    if sum(x["length"] for x in out) > 32768:
        raise RequestError("the blocks have more than 32768 polynomials in all, which "
                           "Statement::var refuses")
    degree = _int(s["statement_degree"], "statement_degree")
    if degree not in RING_DEGREES:
        raise RequestError(f"statement_degree {degree} is not a power of two from 64 to 1024, "
                           "as Rust's Ring requires")
    return degree, out


def _exported_norm(v, what):
    """A norm as `serde_json` writes `statement::Norm`: `{"l2_squared": B}`, `"binary"`,
    `{"linf": b}`, `"unbounded"` or `{"linf_exact": b}`."""
    if v in ("binary", "unbounded"):
        return (v,)
    if isinstance(v, dict) and len(v) == 1 and set(v) <= {"l2_squared", "linf", "linf_exact"}:
        (kind, bound), = v.items()
        bound = _int(bound, f"{what}.{kind}")
        if bound >= 1 << 64:
            raise RequestError(f"{what}.{kind} must fit a u64")
        if kind in ("linf", "linf_exact") and bound == 0:
            raise RequestError(f"{what}: Statement::var refuses {kind} 0")
        if kind == "linf_exact" and bound >= 1 << 63:
            raise RequestError(f"{what}: Statement::var refuses linf_exact from 2^63 on")
        return (kind, bound)
    raise RequestError(f"{what} must be {{\"l2_squared\": B}}, \"binary\", {{\"linf\": b}}, "
                       "\"unbounded\" or {\"linf_exact\": b}, as Rust writes a Norm")


def share(norm, rows, degree):
    """A block's share of `alpha_squared` in the Ajtai part, for `rows` polynomials of the
    proof degree (`statement::BlockRequirement`): an exactly bounded block's rows are binary
    bits."""
    if norm[0] == "l2_squared":
        return norm[1]
    if norm[0] in ("binary", "linf_exact"):
        return rows * degree
    if norm[0] == "linf":
        return rows * degree * norm[1] ** 2
    return 0


def exported_blocks(v, degree):
    """The blocks of `Statement::blocks` as dicts with `name`, `rows`, `norm`, `placement` and
    `share`."""
    out, names = [], set()
    for i, b in enumerate(_list(v, "statement.blocks")):
        what = f"statement.blocks[{i}]"
        _known(b, EXPORTED_BLOCK_KEYS, what)
        _need(b, EXPORTED_BLOCK_KEYS, what)
        name = _name(b["name"], f"{what}.name")
        if len(name.encode()) > 256 or name in names:
            raise RequestError(f"{what}: a repeated name or one of more than 256 bytes")
        names.add(name)
        rows = _int(b["rows"], f"{what}.rows")
        if rows == 0:
            raise RequestError(f"{what}.rows must be positive")
        norm = _exported_norm(b["norm"], f"{what}.norm")
        out.append({"name": name, "rows": rows, "norm": norm,
                    "placement": _placement(b["placement"], norm, f"{what}.placement"),
                    "share": share(norm, rows, degree)})
    if not out:
        raise RequestError("statement.blocks must not be empty")
    return out


def _lifted_moduli(v):
    pairs = []
    for i, e in enumerate(_list(v, "lifted_moduli")):
        what = f"lifted_moduli[{i}]"
        _known(e, ("modulus", "max_integer_coefficient"), what)
        _need(e, ("modulus", "max_integer_coefficient"), what)
        pairs.append([_modulus(e["modulus"], f"{what}.modulus"),
                      _int(e["max_integer_coefficient"], f"{what}.max_integer_coefficient")])
    if any(a[0] >= b[0] for a, b in zip(pairs, pairs[1:])):
        raise RequestError("lifted_moduli must be strictly ascending, as Rust writes them")
    return pairs


def _linf(v):
    _known(v, ("lifting", "extraction_limit"), "linf")
    _need(v, ("lifting", "extraction_limit"), "linf")
    entries = []
    for i, e in enumerate(_list(v["lifting"], "linf.lifting")):
        what = f"linf.lifting[{i}]"
        keys = ("modulus", "exact", "linear", "quadratic")
        _known(e, keys, what)
        _need(e, keys, what)
        entries.append([_modulus(e["modulus"], f"{what}.modulus")]
                       + [_int(e[k], f"{what}.{k}") for k in keys[1:]])
    limit = v["extraction_limit"]
    return entries, None if limit is None else _int(limit, "linf.extraction_limit")


def _approx(v, req, d):
    """`approx_alpha_squared` as `Statement::requirements` writes it: present only with a
    range block and below $`n'd\\beta_\\infty^2`$, and below $`2^{128}`$ (a `u128`)."""
    a = _int(v, "approx_alpha_squared")
    default = req["n_prime"] * d * req["linf_bound"] ** 2
    if not req["n_prime"]:
        raise RequestError("approx_alpha_squared without a range block (n_prime 0), which "
                           "Statement::requirements never exports")
    if a >= 1 << 128 or a >= default:
        raise RequestError(f"approx_alpha_squared {a} is not below n' d linf_bound^2 = {default} "
                           "and 2^128; Statement::requirements exports it only below both")
    return a


def _check_blocks(req, blocks, has_linf):
    """The blocks describe the requirements: they add up to the aggregate fields."""
    ajtai = [b for b in blocks if b["placement"] == "ajtai"]
    bdlop = [b for b in blocks if b["placement"] == "bdlop"]
    l2 = [b for b in blocks if b["norm"][0] == "l2_squared"]
    linf_rows = sum(b["rows"] for b in blocks if b["norm"][0] == "linf")
    got = {"m1": sum(b["rows"] for b in ajtai),
           "alpha_squared": sum(b["share"] for b in ajtai),
           "n_bin": sum(b["rows"] for b in blocks
                        if b["norm"][0] in ("binary", "linf_exact")),
           "l2_rows": [b["rows"] for b in l2],
           "l2_bounds_squared": [b["norm"][1] for b in l2]}
    for key, value in got.items():
        if req[key] != value:
            raise RequestError(f"the blocks give {key} {value}, the requirements {req[key]}: "
                               "export both from one statement and one proof degree")
    if req["l"] < sum(b["rows"] for b in bdlop) or req["n_prime"] < linf_rows:
        raise RequestError("the blocks leave no room for them in l or n_prime: export both "
                           "from one statement and one proof degree")
    if has_linf != (linf_rows > 0):
        raise RequestError("linf is present exactly when a block has a linf norm")
    return linf_rows


def requirements_of(r):
    """(requirements with max_integer_coefficient, lifting dict or None), the requirements at
    the declared placement. The lifting dict has the statement modulus, `max_integer_
    coefficient`, `pairs` (each lifted modulus with its largest honest bound $`F_j`$), `linf`
    (each lifted constraint with an $`\\ell_\\infty`$ variable: modulus and the coefficients of
    $`F_j(E)`$), `extraction_limit`, `has_linf`, `betas` (the $`\\beta`$ of every
    $`\\ell_\\infty`$ block) and `lifted` (whether any constraint is lifted)."""
    s = r["statement"]
    d = r["degree"]
    source = s.get("source") if isinstance(s, dict) else None
    if source == "lin-worst-case":
        statement_degree, blocks = lin_blocks(s)
        p = _statement_modulus(s)
        public = _int(s["public_linf"], "public_linf") if "public_linf" in s else None
        # Any proof modulus other than p: the requirements do not depend on its value.
        req = rustcheck.lin_requirements(
            d, p + 1, statement_degree, p, _int(s["rows"], "rows"),
            [(b["name"], b["length"], b["norm"]) for b in blocks], public)
        k = statement_degree // d
        for b in blocks:
            if b["placement"] == "bdlop" and b["norm"][0] != "unbounded":
                rows = b["length"] * k
                req["m1"] -= rows
                req["l"] += rows
                req["alpha_squared"] -= share(b["norm"], rows, d)
        f = req["max_integer_coefficient"]
        lifting = {"statement_modulus": p, "max_integer_coefficient": f, "pairs": [[p, f]],
                   "linf": [], "extraction_limit": None, "has_linf": False, "betas": [],
                   "lifted": True} if req["n_prime"] else None
        return req, lifting
    if source == "requirements":
        _known(s, REQUIREMENTS_STATEMENT_KEYS, "statement")
        _need(s, ("requirements",), "a requirements statement")
        q = s["requirements"]
        _known(q, REQUIREMENTS_KEYS + OPTIONAL_REQUIREMENTS_KEYS, "statement.requirements")
        _need(q, REQUIREMENTS_KEYS[:-1], "statement.requirements")
        req = {k: _int(q[k], k) for k in ("m1", "l", "alpha_squared", "n_bin", "n_prime",
                                          "linf_bound")}
        for key in ("l2_rows", "l2_bounds_squared"):
            req[key] = [_int(x, key) for x in _list(q[key], key)]
        pairs = _lifted_moduli(q["lifted_moduli"]) if "lifted_moduli" in q else []
        has_linf = "linf" in q
        linf, limit = _linf(q["linf"]) if has_linf else ([], None)
        blocks = exported_blocks(s["blocks"], d) if "blocks" in s else None
        if has_linf and blocks is None:
            raise RequestError("requirements with linf need the statement's blocks "
                               "(Statement::blocks) to tell the linf rows and bounds")
        linf_rows = _check_blocks(req, blocks, has_linf) if blocks is not None else 0
        betas = [b["norm"][1] for b in blocks or [] if b["norm"][0] == "linf"]
        # A constraint is lifted if there are range rows beyond the linf variables' own.
        lifted = req["n_prime"] > linf_rows
        if req["n_prime"]:
            # Without F the lower bound on q would be too small, and compile would refuse the
            # set with "modulus lifting bound"; compile also checks p's inverse modulo q.
            if "max_integer_coefficient" not in q:
                raise RequestError("a range block (n_prime > 0) needs max_integer_coefficient "
                                   "from Statement::requirements for the lifting bound")
            if "statement_modulus" not in s:
                raise RequestError("a range block needs statement_modulus for the lifting bound")
        if not lifted and (pairs or linf):
            raise RequestError("lifted_moduli or linf.lifting without lifted rows in n_prime")
        req["max_integer_coefficient"] = _int(q.get("max_integer_coefficient", 0),
                                              "max_integer_coefficient")
        if "approx_alpha_squared" in q:
            req["approx_alpha_squared"] = _approx(q["approx_alpha_squared"], req, d)
        if not req["n_prime"]:
            return req, None
        p = _statement_modulus(s)
        f = req["max_integer_coefficient"]
        if pairs and max(x[1] for x in pairs) != f:
            raise RequestError("the largest max_integer_coefficient of lifted_moduli is not "
                               "the requirements' max_integer_coefficient")
        if lifted and not pairs:
            pairs = [[p, f]]
        # Statement::requirements takes linf_bound = max(1, ceil(F_j / p_j), beta) over the
        # lifted moduli and the linf variables (statement.rs: requirements, quotient_bound).
        # Any other value means F or p is not the statement's, for example F = 0 left at a
        # default. With linf, the blocks give every beta.
        want = max([1] + [-(-fj // pj) for pj, fj in pairs] + betas)
        if req["linf_bound"] != want:
            raise RequestError(f"linf_bound {req['linf_bound']} does not match the lifted "
                               f"bounds: Statement::requirements gives {want} for "
                               f"max_integer_coefficient {f}, statement_modulus {p} and "
                               "lifted_moduli; copy every field from one "
                               "Statement::requirements")
        return req, {"statement_modulus": p, "max_integer_coefficient": f, "pairs": pairs,
                     "linf": linf, "extraction_limit": limit, "has_linf": has_linf,
                     "betas": betas, "lifted": lifted}
    raise RequestError("statement must be an object whose source is lin-worst-case or "
                       "requirements")


def placement_of(r):
    """The blocks a placement search moves: (every block with `name`, `rows`, `norm`,
    `placement` and `share`, the names in `movable`), or None without `movable`."""
    s = r["statement"]
    if "movable" not in s:
        return None
    d = r["degree"]
    if s.get("source") == "lin-worst-case":
        statement_degree, blocks = lin_blocks(s)
        k = statement_degree // d
        blocks = [dict(b, rows=b["length"] * k, share=share(b["norm"], b["length"] * k, d))
                  for b in blocks]
    else:
        if "blocks" not in s:
            raise RequestError("movable needs the statement's blocks (Statement::blocks)")
        blocks = exported_blocks(s["blocks"], d)
    names = [_name(x, "movable entry") for x in _list(s["movable"], "movable")]
    known = {b["name"]: b for b in blocks}
    if len(set(names)) != len(names) or any(n not in known for n in names):
        raise RequestError("movable must name distinct blocks of the statement")
    if any(known[n]["norm"][0] == "unbounded" for n in names):
        raise RequestError("an unbounded block cannot move to the Ajtai part")
    if not names:
        raise RequestError("movable must name at least one block")
    if len(names) > partition.EXHAUSTIVE_LIMIT:
        raise RequestError(partition.too_many(len(names)))
    return blocks, names


def modulus_of(r):
    m = r["modulus"]
    if not isinstance(m, dict):
        raise RequestError("modulus must be an object")
    if "prime_factors" in m:
        _known(m, ("prime_factors", "gamma"), "modulus")
        if not isinstance(m["prime_factors"], list) or not m["prime_factors"]:
            raise RequestError("prime_factors must be a non-empty list")
        out = {"prime_factors": [_int(x, "prime factor") for x in m["prime_factors"]]}
        if "gamma" in m:
            gamma = _int(m["gamma"], "gamma")
            # Rust's compression takes an even divisor of q - 1; 0 would also divide by zero.
            if gamma < 2 or gamma % 2:
                raise RequestError(f"gamma must be even and at least 2, not {gamma}")
            out["gamma"] = gamma
        return out
    if "search" in m:
        _known(m, ("search",), "modulus")
        s = m["search"]
        _known(s, ("factors", "window_bits", "min_bits", "max_bits"), "modulus search")
        if _int(s.get("factors", 1), "factors") != 1:
            raise RequestError("the search constructs one prime; two-prime moduli are not "
                               "searched (range and norm blocks need a prime modulus anyway)")
        # min_bits 0 asks for q >= 1/2, which every q meets; max_bits 0 asks for q < 1, which
        # none does, so it stays a cap and the search refuses it.
        return {"search": {"window_bits": _int(s.get("window_bits", 4), "window_bits"),
                           "min_bits": _int(s["min_bits"], "min_bits") or None
                           if "min_bits" in s else None,
                           "max_bits": _int(s["max_bits"], "max_bits")
                           if "max_bits" in s else None}}
    raise RequestError("modulus must give prime_factors or search")


def export_modulus(r):
    """The requirements source's `export_modulus`, or None."""
    s = r["statement"]
    if not isinstance(s, dict) or "export_modulus" not in s:
        return None
    return _modulus(s["export_modulus"], "export_modulus")


def _pinned(r, mod):
    """A request with `export_modulus` derives only at that modulus."""
    pin = export_modulus(r)
    if pin is None:
        return
    factors = mod.get("prime_factors")
    if factors is None or math.prod(factors) != pin:
        raise RequestError(
            f"export_modulus {pin} pins q: the requirements were exported at that modulus, "
            "where constraints over it are native, so give modulus prime_factors with that "
            f"product, not {'a search' if factors is None else factors}")


def check_export_modulus(r, q):
    """`check --request`: a set at another $`q`$ than the request's `export_modulus` compiles
    another shape; refused as "export modulus", a message Rust does not have."""
    pin = export_modulus(r)
    if pin is not None and q != pin:
        raise rustcheck.RustCheckError(
            "export modulus", f"the set's q = {q} is not the export_modulus {pin} of the "
            "requirements, at which constraints over it are native")


def read(text):
    """Every part of a request: (request, policy, external report, requirements, lifting,
    modulus). A request that still reaches a missing key or a wrong type raises RequestError,
    not a traceback."""
    r = parse(text)
    try:
        policy, external = policy_of(r)
        req, lifting = requirements_of(r)
        mod = modulus_of(r)
        placement_of(r)  # checked here, used by the placement search
        _pinned(r, mod)
    except (KeyError, TypeError, AttributeError) as e:
        raise RequestError(f"malformed request: {e!r}") from None
    return r, policy, external, req, lifting, mod
