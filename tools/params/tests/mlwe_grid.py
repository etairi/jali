#!/usr/bin/env python3
"""The regression grid of the MLWE estimate, `tests/data/mlwe_grid.jsonl`.

For degrees 64 and 128 and moduli q = 2^e for the exponents in LOG2Q, it records the rank search
of `hardness.rank` under the delta policy: every rank it evaluates, in order, with the block
size and delta of the uSVP estimate, and the rank it selects. `test_hardness.py` checks that
the module still gives these values.

    python3 tools/params/tests/mlwe_grid.py           write the file
    python3 tools/params/tests/mlwe_grid.py --check   compare with it
"""
import argparse
import json
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL))
from jali_params import hardness  # noqa: E402

DATA = TOOL / "tests" / "data" / "mlwe_grid.jsonl"
LOG2Q = (16, 20, 24, 28, 32, 35, 36, 37, 40, 41, 42, 45, 50, 55, 60, 64, 70, 76, 80, 90, 100,
         110, 128, 150, 180, 214, 256)


def record(d, log2q):
    """One rank search, with every evaluation of the estimate it makes."""
    q = 1 << log2q
    evals = []
    mlwe = hardness.mlwe

    def recorded(k, degree, modulus, nu=1):
        beta, delta = mlwe(k, degree, modulus, nu)
        evals.append({"k": k, "beta": beta, "delta": delta})
        return beta, delta
    hardness.mlwe = recorded
    try:
        k, _, _ = hardness.rank(d, q, hardness.Policy("delta"))
    finally:
        hardness.mlwe = mlwe
    return {"d": d, "log2q": log2q, "q": str(q), "rank": k, "evals": evals}


def text():
    records = [record(d, e) for d in (64, 128) for e in LOG2Q]
    head = {"provenance": {
        "what": "hardness.rank(d, 2^log2q, Policy('delta')) for d in {64, 128}: each rank it "
                "evaluates, in order, with the uSVP block size and delta, and the rank selected",
        "command": "python3 tools/params/tests/mlwe_grid.py",
        "records": len(records)}}
    return "\n".join(json.dumps(x) for x in [head] + records) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    new = text()
    if a.check:
        same = DATA.exists() and DATA.read_text() == new
        print(f"{DATA.name}: {'identical' if same else 'DIFFERS'}")
        sys.exit(0 if same else 1)
    DATA.write_text(new)


if __name__ == "__main__":
    main()
