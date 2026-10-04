#!/usr/bin/env python3
"""The tail tables of the base Gaussians, `src/rand/cdf.rs`, computed from their definition.

The sampler's base distributions are two half-Gaussians K on {0, 1, 2, ...} with
Pr[K = i] proportional to rho(i) = exp(-i^2 / (2 s^2)):

    HALF_3_1   s = 3.1,  2 s^2 = 961/50    masks at width 1.55 * 2^t for t >= 1 (scale 2^(t-1))
    HALF_1_55  s = 1.55, 2 s^2 = 961/200   masks at width 1.55 (t = 0)

Entry j of a table is 2^256 Pr[K > j] rounded to the nearest integer, and a table ends with
its first entry that rounds to 0 (59 and 30 entries). The sampler draws a uniform 256-bit
integer V (32 bytes, little-endian) and takes K = #{j : V < entry j}; the entries decrease, so
Pr[K > j] = entry j / 2^256 exactly.

Every entry is enclosed in interval arithmetic (mpmath.iv, 1024-bit endpoints). The sums are
cut after TERMS terms and the rest is bounded by a geometric series:
(N + m)^2 >= N^2 + 2 N m gives sum_{i >= N} rho(i) <= rho(N) / (1 - exp(-N / s^2)). The script
fails if an enclosure contains a rounding boundary, so every rounding is certain.

Usage:
    python3 tools/kat/half_gaussian_cdf.py            print the Rust file
    python3 tools/kat/half_gaussian_cdf.py --check    compare it with src/rand/cdf.rs
    python3 tools/kat/half_gaussian_cdf.py --json     print the tables as a JSON object
    python3 tools/kat/half_gaussian_cdf.py --digest   print the digest that the crate's tests
                                                      pin (src/rand/gauss/table_tests.rs)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import mpmath as mp

BITS = 256
TERMS = 160
PRECISION = 1024
# Name: the numerator and denominator of 1 / (2 s^2), and the width s as the documentation
# writes it.
TABLES = {
    "HALF_3_1": (50, 961, "3.1"),
    "HALF_1_55": (200, 961, "1.55"),
}
RUST = Path(__file__).resolve().parents[2] / "src" / "rand" / "cdf.rs"


def _rho(i, num, den):
    """An enclosure of exp(-num i^2 / den)."""
    return mp.iv.exp(-mp.iv.mpf(num * i * i) / den)


def table(num, den):
    """The entries of one table, as Python integers, up to and including the first 0."""
    iv = mp.iv
    saved = iv.prec, mp.mp.prec
    try:
        iv.prec = mp.mp.prec = PRECISION
        rho = [_rho(i, num, den) for i in range(TERMS)]
        # The omitted terms i >= TERMS lie in [0, rest]; 1 / s^2 = 2 num / den.
        rest = _rho(TERMS, num, den) / (1 - iv.exp(-iv.mpf(2 * num * TERMS) / den))
        omitted = iv.mpf([0, rest.b])
        total = sum(rho, iv.mpf(0)) + omitted
        scale = iv.mpf(2) ** BITS
        out = []
        for j in range(TERMS - 1):
            x = (sum(rho[j + 1:], iv.mpf(0)) + omitted) / total * scale
            lo, hi = mp.mpf(x.a), mp.mpf(x.b)
            r = int(mp.nint((lo + hi) / 2))
            if not (r - mp.mpf(1) / 2 < lo and hi < r + mp.mpf(1) / 2):
                raise ArithmeticError(f"entry {j}: the enclosure [{lo}, {hi}] does not fix "
                                      "the rounding")
            out.append(r)
            if r == 0:
                return out
        raise ArithmeticError("TERMS too small: no entry rounds to 0")
    finally:
        iv.prec, mp.mp.prec = saved


def tables():
    """Every table by name, in the order of TABLES."""
    return {name: table(num, den) for name, (num, den, _) in TABLES.items()}


HEADER = r"""//! The tail tables of the base Gaussians of `gauss`, written by
//! `tools/kat/half_gaussian_cdf.py` from their definition; do not edit by hand.
use crypto_bigint::U256;
"""


def rust(values):
    """The text of `src/rand/cdf.rs`."""
    parts = [HEADER]
    for name, (num, den, width) in TABLES.items():
        entries = values[name]
        parts.append(
            "\n"
            "/// Entry $`j`$ is $`2^{256}\\Pr[K>j]`$ rounded to the nearest integer, for the "
            "half-Gaussian\n"
            f"/// $`K\\in\\{{0,1,2,\\dots\\}}`$ of width {width}, "
            f"$`\\Pr[K=i]\\propto e^{{-{num}i^2/{den}}}`$.\n"
            f"pub(super) const {name}: [U256; {len(entries)}] = [\n")
        parts.extend(f'    U256::from_be_hex("{v:064x}"),\n' for v in entries)
        parts.append("];\n")
    return "".join(parts)


def digest(values):
    """SHAKE128, 32 bytes in hex, of each table in the order of TABLES: LE64(length), then
    every entry as 32 little-endian bytes."""
    data = b""
    for name in TABLES:
        entries = values[name]
        data += len(entries).to_bytes(8, "little")
        data += b"".join(v.to_bytes(32, "little") for v in entries)
    return hashlib.shake_128(data).hexdigest(32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help=f"compare with {RUST.name}")
    ap.add_argument("--json", action="store_true", help="print the tables as JSON")
    ap.add_argument("--digest", action="store_true", help="print the pinned digest")
    a = ap.parse_args()
    values = tables()
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
              f"{'identical' if same else 'DIFFERS from the computed tables'}")
        sys.exit(0 if same else 1)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
