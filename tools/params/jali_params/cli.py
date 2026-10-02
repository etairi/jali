"""Subcommands of `lnp_params.py` (the original positional form stays as it was).

    derive REQUEST [--params OUT] [--report OUT]    native derivation from a v2 request, with a
                                                     placement search over its movable blocks
           [--estimator-target BITS]                 ... refused below BITS in the rough
           [--allow-unpinned-estimator]              lattice-estimator (Sage)
    regenerate [--check] [NAME ...]                  the sets under tools/params/sets/ and their
                                                     copies in the crate, src/params/sets/
    check PARAMS [--request REQUEST]                 emulate the Rust checks, print derived values
    prime --min Q --gamma0 G                         the gamma-first prime construction
    rank --degree D --q Q [--policy P] [--beta-min B] [--mlwe-report]
                                                     the smallest MLWE rank, or an MLWE report
                                                     for the positional form
    xcheck NAME [--full] [--output OUT]              lattice-estimator on a set (Sage)
           [--target BITS [--allow-unpinned-estimator]]

The target checks refuse an estimator checkout that is not at the pinned commit or has local
changes, unless `--allow-unpinned-estimator` is given.
"""
import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import mpmath as mp
import sympy

from . import ROOT, VERSION, derive, hardness, jsonio, modulus, partition, request, rustcheck, \
    tool_sha256, xcheck

SETS = ROOT / "sets"
# The crate ships every set of SETS under the same name; regenerate keeps both copies identical.
CRATE_SETS = ROOT.parents[1] / "src" / "params" / "sets"
REPORT_SCHEMA = "jali-params-report/1"
MLWE_MODEL = ("primal uSVP, the 2016 estimate (ADPS16) with Kannan's embedding, secret and error "
              "uniform in {-1,0,1}, any number of samples")
MSIS_MODEL = ("closed form delta = 2^((log2 B)^2 / (4 n d log2 q)) (LNP22 Section 6.1, after MR09 "
              "and GN08); beta = smallest block size with delta_0(beta) <= delta")
ASSUMPTIONS = ("Hiding needs Extended-MLWE, taken as MLWE following LNP22 Section 6.1; binding "
               "needs MSIS. Core-SVP 2^(0.292 beta) classical, 2^(0.265 beta) quantum (ADPS16). "
               "All security figures are cost-model heuristics, not proofs.")


def _nstr(x, digits=10):
    # Fixed precision, so that the text does not depend on mpmath's global precision.
    with mp.workprec(300):
        return mp.nstr(mp.mpf(x), digits)


def _log2(x):
    with mp.workprec(300):
        return round(float(mp.log(mp.mpf(x), 2)), 4)


def estimator_string(policy, info, request_sha):
    """Provenance bound into the transcript fingerprint (at most 4096 bytes)."""
    return (f"jali-params {VERSION}; request sha256 {request_sha[:16]}; policy {policy.label()}; "
            f"MLWE uSVP (ADPS16) beta {info['mlwe_beta']} delta {info['mlwe_delta']!r}; "
            f"MSIS closed form beta {info['msis_beta']} delta {_nstr(info['msis_delta'], 10)}")


def mlwe_report(degree, q, policy):
    """The MLWE report of the positional form for the ring of degree `degree` modulo `q`: the
    smallest rank that `policy` accepts, its delta and the provenance, which the derived set
    takes as its `estimator`."""
    k, beta, delta = hardness.rank(degree, q, policy)
    return {"degree": degree, "modulus": jsonio.encode_int(q), "rank": k, "delta": delta,
            "provenance": f"jali-params {VERSION} rank; policy {policy.label()}; "
                          f"MLWE uSVP (ADPS16) beta {beta} delta {delta!r}"}


def _inequalities(params, derived, shape, lifting):
    """The checks with values computed in 300-bit arithmetic, for the report."""
    with mp.workprec(300):
        q = math.prod(params["prime_factors"])
        d = params["degree"]
        rows = []

        def add(name, lhs, rel, rhs):
            lhs, rhs = mp.mpf(lhs), mp.mpf(rhs)
            big, small = (rhs, lhs) if rel in ("<", "<=") else (lhs, rhs)
            rows.append({"name": name, "value": _nstr(lhs), "relation": rel,
                         "limit": _nstr(rhs),
                         "margin_bits": float(mp.nstr(mp.log(big / small, 2), 6))})
        z = len(params["l2_rows"])
        add("completeness (m1+Z)d", (params["m1"] + z) * d, ">=", 640)
        add("completeness m2 d", params["m2"] * d, ">=", 640)
        s1, s2, s3, s4 = (mp.mpf("1.55") * 2 ** t for t in params["log_sigma"])
        omega, eta, _ = rustcheck.CHALLENGE[d]
        n = params["n_msis"]
        b = s2 * mp.sqrt(2 * params["m2"] * d) + eta * mp.mpf(2) ** (params["d_bits"] - 1) \
            * mp.sqrt(n * d) + params["gamma"] * mp.sqrt(n * d) / 2
        add("response bound b^2", b * b, "<", mp.mpf(2) ** 128)
        bound = derive.msis_bound(shape, params["log_sigma"], n, params["m2"], params["d_bits"],
                                  params["gamma"])
        add("MSIS delta", hardness.delta_msis(bound, n, d, q), "<", hardness.DELTA_MAX)
        add("MSIS bound < q", bound, "<", q)
        add("rounding 2^(D-1) omega d < gamma", mp.mpf(2) ** (params["d_bits"] - 1) * omega * d,
            "<", params["gamma"])
        add("MLWE delta (Rust accepts <= 1.0044)", params["mlwe_delta"], "<=",
            hardness.DELTA_MAX)
        if derived["n_ex"]:
            arp = 2 * mp.sqrt(mp.mpf(256) / 26) * mp.mpf("1.64") * s3
            add("ARP modulus bound", q, ">=", 41 * derived["n_ex"] * d * arp)
            add("binary lifting bound", q, ">", arp ** 2 + arp * mp.sqrt(params["n_bin"] * d))
            add("slack lifting bound", q, ">", arp ** 2 + arp * mp.sqrt(d))
            for i, bsq in enumerate(params["l2_bounds_squared"]):
                add(f"exact norm lifting bound {i}", q, ">", 3 * bsq + arp ** 2)
        lifting = rustcheck.full_lifting(lifting)
        if lifting is not None and params["n_prime"]:
            slack = params["linf_bound"] * rustcheck.psi(params)
            e = rustcheck.extraction_bound(params["log_sigma"][3])
            pairs = lifting["pairs"] if lifting["lifted"] else []
            one = len(pairs) == 1 and not lifting["linf"]
            for m, f in pairs:
                add("modulus lifting bound (compile)" if one else
                    f"modulus lifting bound (compile), modulus {m}", q, ">", 2 * (f + m * slack))
            for i, x in enumerate(lifting["linf"]):
                add(f"modulus lifting bound at E (compile), linf constraint {i}, modulus {x[0]}",
                    q, ">", 2 * (rustcheck.linf_f(x, e) + x[0] * slack))
            if lifting["has_linf"]:
                add("approximate extraction bound 2E+1 < q (compile)", 2 * e + 1, "<", q)
                if lifting["extraction_limit"] is not None:
                    add("range condition E <= (p-1)/2 (compile)", e, "<=",
                        lifting["extraction_limit"])
        for key in ("rejection_M1", "rejection_M2", "range_rejection_M3", "range_rejection_M4"):
            if derived[key] is not None:
                add(f"{key} < 2^16", derived[key], "<", 65536)
        return rows


def _hardness_block(params, info, shape):
    d = params["degree"]
    q = math.prod(params["prime_factors"])
    z = len(params["l2_rows"])
    mlwe_beta = info["mlwe_beta"]
    if mlwe_beta is None:
        mlwe_beta, _ = hardness.mlwe(params["mlwe_rank"], d, q)
    with mp.workprec(300):
        bound = derive.msis_bound(shape, params["log_sigma"], params["n_msis"], params["m2"],
                                  params["d_bits"], params["gamma"])
        log2b = float(mp.log(bound, 2))
    lext = params["m2"] - params["n_msis"] - params["l"] - params["mlwe_rank"]
    return {
        "mlwe": {"model": MLWE_MODEL, "rank": params["mlwe_rank"], "n": params["mlwe_rank"] * d,
                 "q": jsonio.encode_int(q), "samples_m_for_cross_checks":
                     (params["n_msis"] + params["l"] + lext) * d,
                 "beta": mlwe_beta, "delta": params["mlwe_delta"],
                 "core_svp_bits": hardness.core_svp_bits(mlwe_beta)},
        "msis": {"model": MSIS_MODEL, "rank": params["n_msis"], "n": params["n_msis"] * d,
                 "m": (params["m1"] + z + params["m2"]) * d, "q": jsonio.encode_int(q),
                 "log2_bound": round(log2b, 6), "delta": _nstr(info["msis_delta"], 12),
                 "beta": info["msis_beta"],
                 "core_svp_bits": hardness.core_svp_bits(info["msis_beta"])},
        "assumptions": ASSUMPTIONS,
    }


def _summary(params, info, derived):
    q = params["prime_factors"][0]
    return {"q": jsonio.encode_int(q), "log2_q": _log2(q), "mlwe_rank": params["mlwe_rank"],
            "mlwe_beta": info["mlwe_beta"], "n_msis": params["n_msis"],
            "msis_beta": info["msis_beta"], "gamma": params["gamma"], "d_bits": params["d_bits"],
            "log_sigma": params["log_sigma"],
            "estimated_proof_bytes": derived["estimated_proof_bytes"]}


def derive_request(text, capacity=rustcheck.CAPACITY, what_if=True, compare=True):
    """Derive the parameter set of a v2 request. Returns (params text, report dict)."""
    r, policy, external, req, lifting, mod = request.read(text)
    req_sha = hashlib.sha256(text.encode()).hexdigest()
    d = r["degree"]
    record = None
    search = mod.get("search")
    max_bits = search["max_bits"] if search else None
    if search is not None and external is not None:
        raise request.RequestError("a modulus search needs a policy, not an external report")
    placement = _placement_search(r, req, lifting, policy, external, mod, capacity)
    if placement is not None:
        req = placement.pop("requirements")
    shape = derive.Shape(req, d)
    if "prime_factors" in mod:
        params, info = derive.solve(shape, mod["prime_factors"], policy, external,
                                    gamma=mod.get("gamma"))
    else:
        params, info, _, _, record = derive.search(shape, policy, lifting, search["window_bits"],
                                                   search["min_bits"], capacity, max_bits)
    info = dict(info, mlwe_delta=params["mlwe_delta"])
    params["id"] = r["id"]
    params["estimator"] = external["provenance"] if external else \
        estimator_string(policy, info, req_sha)
    params_text = jsonio.dump_params(params)
    # The written text as Rust reads it: from_json, the prover's constants and compile.
    _, derived, _ = rustcheck.from_json(params_text, capacity)
    rustcheck.prover_checks(derived)
    if lifting is not None and shape.n_prime:
        rustcheck.check_compile(params, req, lifting["statement_modulus"], lifting=lifting)
    q = math.prod(params["prime_factors"])
    rep = {
        "schema": REPORT_SCHEMA, "id": r["id"],
        "request_sha256": req_sha,
        "params_sha256": hashlib.sha256(params_text.encode()).hexdigest(),
        "tool": {"name": "jali-params", "version": VERSION, "sha256": tool_sha256()},
        "environment": {"python": platform.python_version(), "mpmath": mp.__version__,
                        "sympy": sympy.__version__},
        "deterministic": True,
        "policy": policy.describe() if policy else {"name": "external",
                                                     "provenance": external["provenance"]},
        "statement": {"source": r["statement"].get("source"),
                      "requirements": {k: jsonio.encode_wide(v) if isinstance(v, (int, list))
                                       and not isinstance(v, bool) else v
                                       for k, v in req.items()},
                      "statement_modulus": jsonio.encode_int(lifting["statement_modulus"])
                      if lifting else None,
                      "note": ("requirements of lin::compile for the worst case over public "
                               "data (every coefficient of A and t has the largest centred "
                               "absolute value), which dominate every instance of the shape")
                      if r["statement"].get("source") == "lin-worst-case" else None},
        "summary": _summary(params, info, derived),
        "modulus": {
            "q": jsonio.encode_int(q), "log2_q": _log2(q), "bits": q.bit_length(),
            "q_mod_8": q % 8, "gamma": params["gamma"], "gamma_0": info["gamma0"],
            "gamma_v2": modulus.v2(params["gamma"]), "d_bits": params["d_bits"],
            "primality": modulus.primality_note(q),
            "rns_primes_required": derived["rns_primes_required"],
            "rust_capacity": dict(capacity),
        },
        "hardness": _hardness_block(params, info, shape),
        "derived": {k: jsonio.encode_int(v) if isinstance(v, int) and not isinstance(v, bool)
                    and v >= 0 else v for k, v in derived.items()
                    if k not in ("omega", "eta", "q")},
        "checks": _inequalities(params, derived, shape, lifting),
    }
    _echo_lifting(rep["statement"], lifting)
    rep["repetitions"] = {"M1": derived["rejection_M1"], "M2": derived["rejection_M2"],
                          "M3": derived["range_rejection_M3"], "M4": derived["range_rejection_M4"],
                          "note": "integer ceilings the Rust prover uses; expected attempts are "
                                  "at most their product"}
    if shape.range_width is not None:
        # Only for requirements with approx_alpha_squared: which rule set log_sigma[3].
        rep["range_width"] = _encode_trace(dict(shape.range_width, note=RANGE_WIDTH))
    if "prime_factors" in mod:
        rep["modulus"]["gamma_rule"] = (
            "given in the request" if "gamma" in mod else
            "the positional form's: the largest even divisor of q-1 in (4 gamma_0/5, gamma_0]; "
            "a searched set may use another divisor")
    if record is not None:
        rep["modulus"]["construction"] = (
            "gamma first: q = 1 + gamma*t with t odd (v2(gamma) = 2) or t = 2 mod 4 "
            "(v2(gamma) = 1), the smallest prime at or above the start")
        rep["modulus"]["lower_bounds"] = [
            {"name": k, "q_at_least": jsonio.encode_int(v), "log2": _log2(v)}
            for k, v in record["lower_bounds"].items()]
        rep["modulus"]["largest_lower_bound"] = record["largest_lower_bound"]
        rep["modulus"]["binding_at_q"] = record["binding_lower_bound"]
        rep["modulus"]["window_bits"] = record["window_bits"]
        if "default_width_q_lo" in record:
            q_def = record["default_width_q_lo"]
            rep["modulus"]["default_width_window_from"] = {
                "q_at_least": jsonio.encode_int(q_def), "log2": _log2(q_def),
                "note": "with a per-slot width, the search also starts where the default width's "
                        "lower bound would start it"}
        if record["max_bits"] is not None:
            rep["modulus"]["max_bits"] = record["max_bits"]
        rep["modulus"]["search"] = _encode_trace(record["trace"])
    if compare and policy is not None and "search" in mod:
        other = hardness.Policy("beta", 439) if policy.name == "delta" \
            else hardness.Policy("delta")
        try:
            p2, i2, d2, _, _ = derive.search(shape, other, lifting, search["window_bits"],
                                             search["min_bits"], capacity, max_bits)
            rep["policy_comparison"] = {"policy": other.describe(),
                                        **_summary(p2, dict(i2), d2)}
        except derive.DeriveError as e:
            rep["policy_comparison"] = {"policy": other.describe(), "failed": str(e)}
    if placement is not None:
        rep["placement"] = _encode_trace(placement)
    elif what_if and r["statement"].get("source") == "lin-worst-case" and "search" in mod:
        statement_degree, blocks = request.lin_blocks(r["statement"])
        k = statement_degree // d
        l2 = [b for b in blocks if b["norm"][0] == "l2_squared" and b["placement"] == "ajtai"]
        if len(l2) > partition.EXHAUSTIVE_LIMIT:
            rep["partition"] = {"realizable": WHAT_IF, "skipped": partition.too_many(len(l2))}
        else:
            res = partition.evaluate(req, [(b["length"] * k, b["norm"][1]) for b in l2], d,
                                     policy, lifting, search["window_bits"], capacity, max_bits,
                                     names=[b["name"] for b in l2], min_bits=search["min_bits"])
            rep["partition"] = {"realizable": WHAT_IF, **_encode_trace(res)}
    return params_text, rep


RANGE_WIDTH = ("log_sigma[3] is the larger of the width for the per-slot bound and the guard, "
               "the smallest width whose rejection constant M4 for n' d linf_bound^2 is 2; the "
               "prover's M4 keeps that bound, so rejection sampling stays exact, and a narrower "
               "sigma_4 only lowers the extraction bound E = 2 z4_bound")
WHAT_IF = ("what-if: the exact-norm blocks of the Ajtai part moved to BDLOP; to let derive "
           "choose, name them in the statement's movable list, and realize the result with "
           "Statement::var_placed or lin::compile_placed")
REALIZE = ("declare each block of chosen.bdlop with Statement::var_placed(name, count, norm, "
           "Placement::Bdlop), or pass the names to lin::compile_placed; the others keep "
           "Placement::default_for(norm)")


def _movable(r, req):
    """For a request with movable blocks: (every block, the movable names, the movable blocks,
    the requirements with every movable block in the Ajtai part, where the search starts, and
    the movable blocks' (rows, share)). None without movable blocks."""
    found = request.placement_of(r)
    if found is None:
        return None
    blocks, movable = found
    moved = [b for b in blocks if b["name"] in movable]
    base = dict(req)
    for b in moved:
        if b["placement"] == "bdlop":
            base.update(m1=base["m1"] + b["rows"], l=base["l"] - b["rows"],
                        alpha_squared=base["alpha_squared"] + b["share"])
    return blocks, movable, moved, base, [(b["rows"], b["share"]) for b in moved]


def _chosen(blocks, movable, bdlop_names):
    """The parts of every block when the movable blocks named in `bdlop_names` are in BDLOP."""
    bdlop = [b["name"] for b in blocks if b["name"] in bdlop_names
             or (b["name"] not in movable and b["placement"] == "bdlop")]
    return {"ajtai": [b["name"] for b in blocks if b["name"] not in bdlop], "bdlop": bdlop}


def set_placement(r, req, params):
    """For `check --request` with movable blocks: the requirements of the placement `params`
    was derived for, and that placement. The placements are those the search can choose
    (`partition.candidates`), and the one whose `m1`, `l` and `alpha_squared` equal the set's
    is taken; placements that agree on all three have the same requirements. Without movable
    blocks, the declared requirements and None. Raises the tool's "placement" refusal, a
    message Rust does not have, when no placement matches."""
    found = _movable(r, req)
    if found is None:
        return req, None
    blocks, movable, moved, base, shares = found
    want = (params["m1"], params["l"], params["alpha_squared"])
    seen = []
    for chosen in partition.candidates(shares):
        placed = partition.placed_requirements(base, shares, chosen)
        got = (placed["m1"], placed["l"], placed["alpha_squared"])
        names = [moved[i]["name"] for i in chosen]
        if got == want:
            return placed, _chosen(blocks, movable, names)
        seen.append(f"{names or 'none'} in BDLOP gives {got}")
    raise rustcheck.RustCheckError(
        "placement", f"the set's (m1, l, alpha_squared) = {want} matches no placement of the "
        f"movable blocks {movable}: {'; '.join(seen)}")


def _placement_search(r, req, lifting, policy, external, mod, capacity):
    """For a request with movable blocks: the requirements of the placement with the smallest
    proof estimate, and the search's record. None without movable blocks."""
    found = _movable(r, req)
    if found is None:
        return None
    # The search starts with every movable block in the Ajtai part.
    blocks, movable, moved, base, shares = found
    search = mod.get("search")
    res = partition.evaluate(base, shares, r["degree"], policy, lifting,
                             search["window_bits"] if search else 4, capacity,
                             search["max_bits"] if search else None,
                             names=[b["name"] for b in moved], fixed=None if search else mod,
                             external=external, min_bits=search["min_bits"] if search else None)
    best = partition.best(res)
    if best is None:
        reasons = sorted({e["failed"] for e in res["evaluated"]})
        raise derive.DeriveError("placement", f"no placement of {movable} gives a set: "
                                 f"{'; '.join(reasons)}")
    chosen = [i for i, b in enumerate(moved) if b["name"] in best["bdlop_blocks"]]
    return {"requirements": partition.placed_requirements(base, shares, chosen),
            "movable": movable, "method": res["method"], "evaluated": res["evaluated"],
            "chosen": _chosen(blocks, movable, best["bdlop_blocks"]),
            "realize": REALIZE}


def _echo_lifting(statement, lifting):
    """The per-constraint lifting of copied requirements, in the report's statement block."""
    if lifting is None:
        return
    enc = jsonio.encode_int
    pairs = lifting["pairs"] if lifting["lifted"] else []
    if len(pairs) > 1 or any(m != lifting["statement_modulus"] for m, _ in pairs):
        statement["lifted_moduli"] = [{"modulus": enc(m), "max_integer_coefficient": enc(f)}
                                      for m, f in pairs]
    if lifting["has_linf"]:
        limit = lifting["extraction_limit"]
        statement["linf"] = {
            "lifting": [dict(zip(("modulus", "exact", "linear", "quadratic"), map(enc, x)))
                        for x in lifting["linf"]],
            "extraction_limit": None if limit is None else enc(limit),
            "betas": [enc(b) for b in lifting["betas"]]}


def _encode_trace(trace):
    def enc(o):
        if isinstance(o, dict):
            return {k: enc(v) for k, v in o.items()}
        if isinstance(o, list):
            return [enc(v) for v in o]
        if isinstance(o, int) and not isinstance(o, bool) and o >= 0:
            return jsonio.encode_int(o)
        return o
    return enc(trace)


def _set_names():
    return sorted(p.name[:-len(".request.json")] for p in SETS.glob("*.request.json"))


def instances(rep):
    """The MLWE and MSIS instances of a report, as the cross-checks take them."""
    h = rep["hardness"]
    return {"mlwe": {"n": h["mlwe"]["n"], "q": jsonio.decode_int(h["mlwe"]["q"]),
                     "m": h["mlwe"]["samples_m_for_cross_checks"]},
            "msis": {"n": h["msis"]["n"], "q": jsonio.decode_int(h["msis"]["q"]),
                     "m": h["msis"]["m"], "log2_bound": h["msis"]["log2_bound"]}}


def cmd_derive(a):
    if a.estimator_target is not None:
        # Refuse before deriving: the check was asked for and cannot run.
        why = xcheck.available("lattice-estimator", a.allow_unpinned_estimator)
        if why:
            raise xcheck.Unavailable(f"--estimator-target needs the lattice-estimator: {why}")
    text = Path(a.request).read_text()
    params_text, rep = derive_request(text, rustcheck.CAPACITY, what_if=not a.no_what_if)
    if a.estimator_target is not None:
        verdict = xcheck.rough_check(instances(rep), a.estimator_target,
                                     a.allow_unpinned_estimator)
        rep["estimator_check"] = verdict
        if not verdict["reached"]:
            raise xcheck.BelowTarget(xcheck.shortfall(verdict))
    if a.params:
        Path(a.params).write_text(params_text)
    else:
        sys.stdout.write(params_text)
    if a.report:
        Path(a.report).write_text(jsonio.dumps(rep))


def _without_env(rep):
    return {k: v for k, v in rep.items() if k != "environment"}


def cmd_regenerate(a):
    names = a.names or _set_names()
    bad = []
    for name in names:
        text = (SETS / f"{name}.request.json").read_text()
        params_text, rep = derive_request(text)
        pj, rj = SETS / f"{name}.json", SETS / f"{name}.report.json"
        cj = CRATE_SETS / f"{name}.json"
        if a.check:
            ok_p = pj.exists() and pj.read_text() == params_text
            # Byte for byte, as the crate includes it (read_text would hide line endings).
            ok_c = pj.exists() and cj.exists() and cj.read_bytes() == pj.read_bytes()
            old = _without_env(json.loads(rj.read_text())) if rj.exists() else {}
            new = json.loads(jsonio.dumps(_without_env(rep)))
            ok_r = old == new
            keys = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            print(f"{name}: params {'identical' if ok_p else 'DIFFER'}, crate copy "
                  f"{'identical' if ok_c else 'DIFFERS from the tool copy'}, report "
                  f"{'identical' if ok_r else 'DIFFERS in ' + ', '.join(keys)} "
                  f"(environment block not compared)")
            if not (ok_p and ok_c and ok_r):
                bad.append(name)
        else:
            pj.write_text(params_text)
            cj.write_text(params_text)
            rj.write_text(jsonio.dumps(rep))
            print(f"{name}: written, with the crate's copy")
    if bad:
        sys.exit(1)


def cmd_check(a):
    """The emulated Rust checks of a set and, with `--request`, of `compile` against the
    request's requirements. With movable blocks those are the requirements of the placement the
    set was derived for (`set_placement`), which the output names as `placement_assumed`."""
    cap = rustcheck.CAPACITY
    assumed = {}
    try:
        params, derived, checks = rustcheck.from_json(Path(a.params).read_text(), cap)
        rustcheck.prover_checks(derived)
        if a.request:
            r, _, _, req, lifting, _ = request.read(Path(a.request).read_text())
            request.check_export_modulus(r, math.prod(params["prime_factors"]))
            req, placement = set_placement(r, req, params)
            if placement is not None:
                assumed = {"placement_assumed": placement}
            p = lifting["statement_modulus"] if lifting else math.prod(params["prime_factors"])
            rustcheck.check_compile(params, req, p, checks, lifting)
    except rustcheck.RustCheckError as e:
        print(json.dumps({"accepted": False, "error": str(e), "rust_message": e.rust_message,
                          **assumed}))
        sys.exit(1)
    out = {"accepted": True, "tight": checks.tight(), **assumed,
           **{k: jsonio.encode_int(v) if isinstance(v, int) and v >= 0 else v
              for k, v in derived.items()}}
    print(json.dumps(out, indent=2))


def cmd_prime(a):
    gamma0 = int(a.gamma0)
    gamma = modulus.top_gamma(gamma0)
    if gamma is None:
        raise SystemExit(f"no admissible gamma below {gamma0}")
    q, tries = modulus.construct_prime(int(a.min), gamma)
    print(json.dumps({"q": jsonio.encode_int(q), "gamma": gamma, "candidates_tested": tries,
                      "log2_q": _log2(q), "primality": modulus.primality_note(q)}))


def cmd_rank(a):
    try:
        pol = hardness.Policy(a.policy, a.beta_min)
    except ValueError as e:
        raise request.RequestError(str(e)) from None
    if a.mlwe_report:
        sys.stdout.write(jsonio.dumps(mlwe_report(a.degree, int(a.q), pol)))
        return
    k, beta, delta = hardness.rank(a.degree, int(a.q), pol)
    print(json.dumps({"rank": k, "beta": beta, "delta": delta,
                      "core_svp_bits": hardness.core_svp_bits(beta)}))


def cmd_xcheck(a):
    rep = json.loads((SETS / f"{a.name}.report.json").read_text())
    inst = instances(rep)
    if a.allow_unpinned_estimator and a.target is None:
        raise SystemExit("--allow-unpinned-estimator applies to --target only")
    if a.target is not None:
        if a.full:
            raise SystemExit("--target checks the rough (core-SVP) mode; run --full separately")
        verdict = xcheck.rough_check(inst, a.target, a.allow_unpinned_estimator)
        print(json.dumps(verdict, indent=2))
        if not verdict["reached"]:
            print(f"error: {xcheck.shortfall(verdict)}", file=sys.stderr)
            sys.exit(1)
        return
    out = xcheck.lattice_estimator(inst, full=a.full)
    out = {"id": rep["id"], "params_sha256": rep["params_sha256"], **out}
    text = jsonio.dumps(out)
    if a.output:
        Path(a.output).write_text(text)
    else:
        sys.stdout.write(text)


def main(argv):
    ap = argparse.ArgumentParser(prog="lnp_params.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("derive")
    p.add_argument("request")
    p.add_argument("--params")
    p.add_argument("--report")
    p.add_argument("--no-what-if", action="store_true")
    p.add_argument("--estimator-target", type=float, metavar="BITS",
                   help="refuse the set unless lattice-estimator's rough (core-SVP) estimate "
                        "of its MLWE and MSIS instances reaches BITS (needs Sage)")
    p.add_argument("--allow-unpinned-estimator", action="store_true",
                   help="let --estimator-target run on an estimator checkout that is not at "
                        "the pinned commit or has local changes (the report records which)")
    p.set_defaults(fn=cmd_derive)
    p = sub.add_parser("regenerate")
    p.add_argument("names", nargs="*")
    p.add_argument("--check", action="store_true")
    p.set_defaults(fn=cmd_regenerate)
    p = sub.add_parser("check")
    p.add_argument("params")
    p.add_argument("--request")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("prime")
    p.add_argument("--min", required=True)
    p.add_argument("--gamma0", required=True)
    p.set_defaults(fn=cmd_prime)
    p = sub.add_parser("rank")
    p.add_argument("--degree", type=int, required=True)
    p.add_argument("--q", required=True)
    p.add_argument("--policy", default="delta")
    p.add_argument("--beta-min", type=int)
    p.add_argument("--mlwe-report", action="store_true",
                   help="print the MLWE report that the positional form takes")
    p.set_defaults(fn=cmd_rank)
    p = sub.add_parser("xcheck")
    p.add_argument("name")
    p.add_argument("--full", action="store_true")
    p.add_argument("--output")
    p.add_argument("--target", type=float, metavar="BITS",
                   help="exit 1 unless the rough estimate of both instances reaches BITS")
    p.add_argument("--allow-unpinned-estimator", action="store_true",
                   help="let --target run on an estimator checkout that is not at the pinned "
                        "commit or has local changes (the verdict records which)")
    p.set_defaults(fn=cmd_xcheck)
    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except (derive.DeriveError, request.RequestError, rustcheck.RustCheckError,
            hardness.HardnessError, modulus.ModulusError, xcheck.Unavailable,
            xcheck.BelowTarget) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


SUBCOMMANDS = {"derive", "regenerate", "check", "prime", "rank", "xcheck"}
