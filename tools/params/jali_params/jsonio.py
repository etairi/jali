"""JSON conventions shared with the Rust crate.

Number rule: an integer field that can exceed $`2^{64}`$ (prime factors, bounds) is written as
a JSON number when it is below $`2^{64}`$ and as a decimal string otherwise; readers accept
both. As in the crate (`src/json.rs`), a JSON number of $`2^{64}`$ or more is refused,
since a JSON parser may round it to a float. Output is `json.dumps(..., indent=2)` plus a
newline, as the original tool wrote it, so existing files load and re-serialize byte-identically.
"""
import json

LIMIT = 1 << 64

# Key order of parameter files, as the original tool wrote them (request keys, then derived).
PARAMS_KEYS = [
    "id", "prime_factors", "degree", "m1", "l", "alpha_squared", "n_bin", "l2_rows",
    "l2_bounds_squared", "n_prime", "linf_bound", "m2", "n_msis", "log_sigma", "gamma",
    "d_bits", "mlwe_rank", "mlwe_delta", "estimator",
]
# Integer fields that follow the number rule.
WIDE_FIELDS = {"prime_factors", "alpha_squared", "l2_bounds_squared", "linf_bound", "gamma"}


def encode_int(x):
    """A JSON number below $`2^{64}`$, a decimal string otherwise."""
    if isinstance(x, bool) or not isinstance(x, int) or x < 0:
        raise TypeError(f"not a non-negative integer: {x!r}")
    return x if x < LIMIT else str(x)


def decode_int(v):
    """Accept a JSON integer below $`2^{64}`$ or a decimal string (no sign, no leading zeros,
    no spaces)."""
    if isinstance(v, bool):
        raise ValueError("boolean is not an integer")
    if isinstance(v, int):
        if v < 0:
            raise ValueError("negative integer")
        if v >= LIMIT:
            raise ValueError("a JSON number of 2^64 or more is refused; write a decimal string")
        return v
    if isinstance(v, str) and v.isdigit() and v.isascii() and (v == "0" or v[0] != "0"):
        return int(v)
    raise ValueError(f"not an integer: {v!r}")


def decode_wide(value):
    """Decode a wide field, which may be a list."""
    if isinstance(value, list):
        return [decode_int(v) for v in value]
    return decode_int(value)


def encode_wide(value):
    if isinstance(value, list):
        return [encode_int(v) for v in value]
    return encode_int(value)


def dumps(obj):
    """Canonical text: two-space indent, insertion order, trailing newline."""
    return json.dumps(obj, indent=2) + "\n"


def _unique_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate field {key}")
        out[key] = value
    return out


def _no_constant(name):
    raise ValueError(f"{name} is not JSON")


def loads_strict(text):
    """`json.loads` that refuses duplicate keys and NaN or infinite numbers, as serde does."""
    return json.loads(text, object_pairs_hook=_unique_keys, parse_constant=_no_constant)


def load_params(text):
    """Parse a parameter file into a dict with Python integers in the wide fields."""
    raw = loads_strict(text)
    if not isinstance(raw, dict):
        raise ValueError("a parameter file is a JSON object")
    unknown = set(raw) - set(PARAMS_KEYS)
    if unknown:
        raise ValueError(f"unknown parameter fields: {sorted(unknown)}")
    out = {}
    for key in PARAMS_KEYS:
        if key not in raw:
            raise ValueError(f"missing parameter field {key}")
        out[key] = decode_wide(raw[key]) if key in WIDE_FIELDS else raw[key]
    return out


def dump_params(params):
    """Parameter file text in the fixed key order, with the number rule applied."""
    out = {}
    for key in PARAMS_KEYS:
        value = params[key]
        out[key] = encode_wide(value) if key in WIDE_FIELDS else value
    return dumps(out)
