"""Witness placement: which bounded blocks go to the Ajtai part and which to BDLOP.

`Statement::var_placed` (and `lin::compile_placed`) put a bounded block in either part of the
ABDLOP commitment. A block moved to BDLOP keeps its own norm proof; it leaves $`m_1`$ and
$`\\alpha^2`$ and joins $`\\ell`$, so $`\\sigma_1`$, the MSIS bound and the proof size change.
This module derives a set for every placement of the blocks a request calls movable, at most
`EXHAUSTIVE_LIMIT` of them, at the request's modulus or with its modulus search, and keeps the
smallest Rust proof-size estimate; ties go to fewer blocks in BDLOP. Without movable blocks,
reports still evaluate moving each exact-norm block of the Ajtai part as a what-if. Binary
blocks stay in the Ajtai part unless a request moves them. `candidates` lists the placements
the search can choose, so that `check` can find the one a set was derived for.
"""
import itertools

from . import derive, rustcheck

EXHAUSTIVE_LIMIT = 12  # 4096 placements


def too_many(n):
    return (f"{n} movable blocks: the placement search tries every placement of at most "
            f"{EXHAUSTIVE_LIMIT}")


def placed_requirements(req, blocks, bdlop):
    """Requirements with the blocks whose indices are in `bdlop` moved from the Ajtai part to
    the BDLOP part. `blocks` lists (rows in the proof ring, share of `alpha_squared`) of the
    movable blocks, all in the Ajtai part in `req`."""
    out = dict(req)
    for i in bdlop:
        rows, share = blocks[i]
        out["m1"] -= rows
        out["l"] += rows
        out["alpha_squared"] -= share
    return out


def placements(n_blocks):
    """Every subset of movable blocks, smallest first."""
    for size in range(n_blocks + 1):
        yield from itertools.combinations(range(n_blocks), size)


def candidates(blocks):
    """The placements `evaluate` can choose from: every subset of the blocks."""
    if len(blocks) > EXHAUSTIVE_LIMIT:
        raise derive.DeriveError("placement", too_many(len(blocks)), final=True)
    return placements(len(blocks))


def evaluate(req, blocks, degree, policy, lifting, window_bits=4, capacity=rustcheck.CAPACITY,
             max_bits=None, names=None, fixed=None, external=None, min_bits=None):
    """The estimated proof size of each placement (or the reason it fails). With `fixed`
    (`prime_factors` and an optional `gamma`) each placement is derived at that modulus and
    checked as the search checks a candidate; otherwise with the modulus search of the set
    itself (window, capacity, `min_bits` and `max_bits`). Entries name the BDLOP blocks by
    index, or by `names` when given."""
    label = (lambda i: names[i]) if names is not None else (lambda i: i)

    def one(bdlop):
        r = placed_requirements(req, blocks, bdlop)
        entry = {"bdlop_blocks": [label(i) for i in bdlop], "m1": r["m1"], "l": r["l"],
                 "alpha_squared": r["alpha_squared"]}
        if r["alpha_squared"] == 0 and r["m1"] == 0:
            entry["failed"] = "empty Ajtai part (Rust: alpha_squared = 0 fails `dimensions`)"
            return entry
        try:
            shape = derive.Shape(r, degree)
            if fixed is None:
                params, _, derived, _, _ = derive.search(shape, policy, lifting, window_bits,
                                                         min_bits, capacity, max_bits)
            else:
                params, _ = derive.solve(shape, fixed["prime_factors"], policy, external,
                                         gamma=fixed.get("gamma"))
                params["id"], params["estimator"] = "probe", "probe"
                derived, _ = derive.check(params, shape, lifting, capacity)
            entry["q_bits"] = params["prime_factors"][0].bit_length()
            entry["estimated_proof_bytes"] = derived["estimated_proof_bytes"]
        except (derive.DeriveError, ValueError) as e:  # RustCheckError is a ValueError
            entry["failed"] = str(e)
        return entry
    return {"method": "exhaustive", "evaluated": [one(b) for b in candidates(blocks)]}


def best(result):
    """The smallest estimate; ties go to fewer BDLOP blocks, then to the first evaluated."""
    ok = [e for e in result["evaluated"] if "estimated_proof_bytes" in e]
    return min(ok, key=lambda e: (e["estimated_proof_bytes"], len(e["bdlop_blocks"]))) \
        if ok else None
