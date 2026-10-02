#!/usr/bin/env python3
"""The tail table of the base Gaussian, `src/rand/cdf.rs`, computed from its definition.

The sampler's base distribution is the half-Gaussian K on {0, 1, 2, ...} with
Pr[K = i] proportional to rho(i) = exp(-i^2 / (2 s^2)), s = 1.55, so that 2 s^2 = 961/200.
Entry j of the table, j = 0, ..., 20, is 2^128 Pr[K > j] rounded to the nearest integer. The
sampler draws a uniform 128-bit integer v and takes K = #{j : v < entry j}, so that
Pr[K > j] = entry j / 2^128 exactly. The last entry is 0, since 2^128 Pr[K > 20] < 0.02.

Every entry is enclosed in interval arithmetic (mpmath.iv, 640-bit endpoints). The sums are
cut after TERMS terms and the rest is bounded by a geometric series:
(N + m)^2 >= N^2 + 2 N m gives sum_{i >= N} rho(i) <= rho(N) / (1 - exp(-N / s^2)). The script
fails if an enclosure contains a rounding boundary, so every rounding is certain.

Usage:
    python3 tools/kat/half_gaussian_cdf.py            print the Rust file
    python3 tools/kat/half_gaussian_cdf.py --check    compare it with src/rand/cdf.rs
    python3 tools/kat/half_gaussian_cdf.py --json     print the entries as a JSON list
    python3 tools/kat/half_gaussian_cdf.py --digest   print the digest that the crate's tests
                                                      pin (src/rand/gauss/table_tests.rs)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import mpmath as mp

ENTRIES = 21
TERMS = 64
PRECISION = 640
RUST = Path(__file__).resolve().parents[2] / "src" / "rand" / "cdf.rs"


def _rho(i):
    """An enclosure of exp(-i^2 / (2 s^2)) = exp(-200 i^2 / 961)."""
    return mp.iv.exp(-mp.iv.mpf(200 * i * i) / 961)


def table():
    """The entries, as Python integers."""
    iv = mp.iv
    saved = iv.prec, mp.mp.prec
    try:
        iv.prec = mp.mp.prec = PRECISION
        rho = [_rho(i) for i in range(TERMS)]
        # The omitted terms i >= TERMS lie in [0, rest].
        rest = _rho(TERMS) / (1 - iv.exp(-iv.mpf(400 * TERMS) / 961))
        omitted = iv.mpf([0, rest.b])
        total = sum(rho, iv.mpf(0)) + omitted
        scale = iv.mpf(2) ** 128
        out = []
        for j in range(ENTRIES):
            x = (sum(rho[j + 1:], iv.mpf(0)) + omitted) / total * scale
            lo, hi = mp.mpf(x.a), mp.mpf(x.b)
            r = int(mp.nint((lo + hi) / 2))
            if not (r - mp.mpf(1) / 2 < lo and hi < r + mp.mpf(1) / 2):
                raise ArithmeticError(f"entry {j}: the enclosure [{lo}, {hi}] does not fix "
                                      "the rounding")
            out.append(r)
        return out
    finally:
        iv.prec, mp.mp.prec = saved


HEADER = r"""//! The tail table of the base Gaussian of `gauss`, written by
//! `tools/kat/half_gaussian_cdf.py` from its definition; do not edit by hand.

/// Entry $`j`$ is $`2^{128}\Pr[K>j]`$ rounded to the nearest integer, for the half-Gaussian
/// $`K\in\{0,1,2,\dots\}`$ with $`\Pr[K=i]\propto e^{-i^2/(2\cdot1.55^2)}`$.
"""


def rust(values):
    """The text of `src/rand/cdf.rs`."""
    lines = [f"pub(super) const CDF: [u128; {len(values)}] = ["]
    lines += [f"    {v}," for v in values]
    lines.append("];")
    return HEADER + "\n".join(lines) + "\n"


def digest(values):
    """SHAKE128 of the entries as 16-byte little-endian integers, 32 bytes in hex."""
    return hashlib.shake_128(b"".join(v.to_bytes(16, "little") for v in values)).hexdigest(32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help=f"compare with {RUST.name}")
    ap.add_argument("--json", action="store_true", help="print the entries as JSON")
    ap.add_argument("--digest", action="store_true", help="print the pinned digest")
    a = ap.parse_args()
    values = table()
    if a.json:
        print(json.dumps(values, indent=2))
        return
    if a.digest:
        print(digest(values))
        return
    text = rust(values)
    if a.check:
        same = RUST.read_bytes() == text.encode()
        print(f"{RUST.relative_to(RUST.parents[2])}: "
              f"{'identical' if same else 'DIFFERS from the computed table'}")
        sys.exit(0 if same else 1)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
