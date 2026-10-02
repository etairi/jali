"""Optional cross-checks that need SageMath; never needed to derive or regenerate a set.

`lattice_estimator`: malb's lattice-estimator (LGPLv3+, so it is not vendored) run on the MLWE
and MSIS instances of a parameter set. Fetch it with

    git clone https://github.com/malb/lattice-estimator
    git -C lattice-estimator checkout 53da5982597709ba0fdf94ea37a84d822310fd84

and set `JALI_LATTICE_ESTIMATOR_DIR` to the clone; run under `sage -python`.

`rough_check`: whether the cheapest attack of the estimator's rough mode reaches a target on
both instances. The rough mode uses the core-SVP cost model ($`2^{0.292\\beta}`$, ADPS16), the
model of the `beta` policy, but it also tries attacks the policy does not bound (dual-hybrid)
and uses the proof's actual number of MLWE samples, where the policy's uSVP estimate lets the
attacker take any number. `derive --estimator-target` and `xcheck --target` run it. Since its
verdict gates a set, it refuses a checkout that is not at the pinned commit or has local
changes, unless the caller allows it (`--allow-unpinned-estimator`); the verdict records the
checkout either way. The estimator's own printout goes to stderr, so that stdout carries only
the tool's JSON.

Integer outputs follow the number rule of `jsonio` (BKW's sample count is near $`2^{292}`$).
"""
import contextlib
import math
import os
import subprocess
import sys

from . import jsonio

LATTICE_ESTIMATOR_COMMIT = "53da5982597709ba0fdf94ea37a84d822310fd84"


class Unavailable(RuntimeError):
    pass


class BelowTarget(RuntimeError):
    """The rough estimate of an instance is below the requested target."""


def _sage():
    try:
        import sage.all  # noqa: F401
    except ImportError:
        raise Unavailable("SageMath is not importable; run under `sage -python`") from None


def _checkout(env, commit, allow_unpinned=True):
    """The checkout that `env` names and its state; unless `allow_unpinned`, one that is not at
    `commit` or has local changes (untracked files included) is refused."""
    path = os.environ.get(env)
    if not path or not os.path.isdir(path):
        raise Unavailable(f"{env} is not set to a checkout")
    head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", path, "status", "--porcelain", "--",
                                 "."], capture_output=True, text=True).stdout.strip())
    if not allow_unpinned and (head != commit or dirty):
        state = f"at {head[:12]}" if head else "not a git checkout"
        raise Unavailable(f"{env} is {state}{' with local changes' if dirty else ''}, not the "
                          f"pinned commit {commit[:12]}; pass --allow-unpinned-estimator to run "
                          "the check anyway")
    return path, {"commit": head, "pinned": head == commit, "dirty": dirty}


def available(kind, allow_unpinned=True):
    """None when the cross-check of `kind` ("lattice-estimator") can run, else the reason it
    cannot."""
    if kind != "lattice-estimator":
        raise ValueError(f"unknown cross-check {kind!r}")
    try:
        _sage()
        _checkout("JALI_LATTICE_ESTIMATOR_DIR", LATTICE_ESTIMATOR_COMMIT, allow_unpinned)
    except Unavailable as e:
        return str(e)
    return None


def _cost(c):
    out = {}
    for key in ("rop", "red", "beta", "delta", "d", "m", "eta", "zeta", "t"):
        if key in c:
            v = c[key]
            try:
                if key in ("rop", "red"):
                    out[key] = round(float(math.log2(float(v))), 1)
                elif key in ("beta", "d", "m", "eta", "zeta", "t"):
                    out[key] = jsonio.encode_int(int(v))  # a decimal string from 2^64 on
                else:
                    out[key] = round(float(v), 6)
                if isinstance(out[key], float) and not math.isfinite(out[key]):
                    out[key] = "inf"  # JSON has no infinity
            except (TypeError, ValueError, OverflowError):
                out[key] = str(v)
    return out


def _estimate(instances, full, allow_unpinned=True):
    """The estimator's raw results on the instances: (LWE and SIS parameters, LWE results,
    SIS results, checkout metadata, Sage version). What the estimator prints goes to stderr."""
    _sage()
    path, meta = _checkout("JALI_LATTICE_ESTIMATOR_DIR", LATTICE_ESTIMATOR_COMMIT,
                           allow_unpinned)
    if path not in sys.path:
        sys.path.insert(0, path)
    with contextlib.redirect_stdout(sys.stderr):
        from estimator import LWE, ND, SIS  # noqa: E402
        from sage.all import RR, version  # noqa: E402
        mlwe, msis = instances["mlwe"], instances["msis"]
        lwe = LWE.Parameters(n=mlwe["n"], q=mlwe["q"], Xs=ND.Uniform(-1, 1),
                             Xe=ND.Uniform(-1, 1), m=mlwe["m"])
        sis = SIS.Parameters(n=msis["n"], q=msis["q"],
                             length_bound=RR(2) ** RR(msis["log2_bound"]), m=msis["m"], norm=2)
        run_lwe = LWE.estimate if full else LWE.estimate.rough
        run_sis = SIS.estimate if full else SIS.estimate.rough
        return lwe, sis, run_lwe(lwe), run_sis(sis), meta, version()


def verdict(costs, target):
    """`costs` maps "mlwe" and "msis" to {attack: (log2 rop, block size)}. The cheapest attack
    of each against `target` bits; `reached` only if both are at least the target."""
    out = {"target_bits": target, "mode": "rough (core-SVP, ADPS16)", "cheapest": {}}
    for problem in ("mlwe", "msis"):
        finite = {k: v for k, v in costs[problem].items() if math.isfinite(v[0])}
        if not finite:
            raise Unavailable(f"no finite attack cost for the {problem} instance")
        name = min(finite, key=lambda k: finite[k][0])
        out["cheapest"][problem] = {"attack": name, "rop_log2": round(finite[name][0], 2),
                                    "beta": finite[name][1],
                                    "reached": finite[name][0] >= target}
    out["reached"] = all(c["reached"] for c in out["cheapest"].values())
    return out


def shortfall(v):
    below = [f"{p.upper()} {c['attack']} 2^{c['rop_log2']} (beta {c['beta']})"
             for p, c in v["cheapest"].items() if not c["reached"]]
    return (f"lattice-estimator's rough estimate is below the target 2^{v['target_bits']:g}: "
            + "; ".join(below))


def rough_check(instances, target, allow_unpinned=False):
    """lattice-estimator's rough mode on the instances against `target` bits, compared
    unrounded. Returns `verdict` with the estimator's commit. Refuses an unpinned or modified
    checkout unless `allow_unpinned`."""
    _, _, lr, sr, meta, ver = _estimate(instances, full=False, allow_unpinned=allow_unpinned)

    def costs(res):
        out = {}
        for k, c in res.items():
            try:
                rop = float(math.log2(float(c["rop"])))
            except (TypeError, ValueError, OverflowError, KeyError):
                continue
            out[k] = (rop, int(c["beta"]) if "beta" in c else None)
        return out
    v = verdict({"mlwe": costs(lr), "msis": costs(sr)}, target)
    v["lattice_estimator"], v["sage"] = meta, ver
    return v


def lattice_estimator(instances, full=False):
    """Run the estimator on {"mlwe": {...}, "msis": {...}} instances as the report records
    them. Returns per-attack costs (log2 rop, block size) with the estimator's commit."""
    lwe, sis, lr, sr, meta, ver = _estimate(instances, full)
    out = {"lattice_estimator": meta, "sage": ver, "mode": "full" if full else "rough",
           "mlwe_parameters": repr(lwe), "msis_parameters": repr(sis)}
    out["mlwe"] = {k: _cost(v) for k, v in sorted(lr.items())}
    out["msis"] = {k: _cost(v) for k, v in sorted(sr.items())}
    out["mlwe_min_rop_log2"] = min(v["rop"] for v in out["mlwe"].values()
                                   if isinstance(v.get("rop"), float))
    out["msis_min_rop_log2"] = min(v["rop"] for v in out["msis"].values()
                                   if isinstance(v.get("rop"), float))
    return out

