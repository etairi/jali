"""Native derivation: Jali's parameter set for a statement shape, a degree and a policy.

`solve` derives a set at a given modulus in the order of the original tool (`lnp_params.py`
`derive`): Gaussian widths with $`\\sigma_4`$ sized from the Euclidean bound
$`\\sqrt{n'd}\\cdot\\beta_\\infty`$, or, when the requirements carry `approx_alpha_squared`,
by the per-slot rule (`Shape._per_slot`), MSIS rank, the largest power-of-two $`\\gamma_0`$ that
keeps MSIS hard (at most $`2^{63}`$, as `TboxParams::gamma` is a `u64`), the compression
divisor and the rounding exponent. The MLWE rank comes from the policy (`hardness.rank`) at
the exact modulus, or from an external report.

`search` sizes the modulus. It starts at the largest of the Rust lower bounds on $`q`$ that do
not depend on $`q`$ (`lower_bounds`: the lifting check over every lifted modulus, at the
extraction bound for constraints with $`\\ell_\\infty`$ variables, and $`2E+1<q`$), moves up to
the MSIS fixed point (the MSIS bound, which
grows slowly with $`q`$ through the MLWE rank, must stay below $`q`$), and then, for a few
divisors $`\\gamma`$ around $`\\gamma_0`$, finds by bisection the smallest start at which
$`\\gamma`$ keeps MSIS hard and constructs the smallest prime of the class of $`\\gamma`$ there
(`modulus.construct_prime`, no factoring). Each candidate is re-derived and must pass the
emulated Rust checks (`rustcheck`), including the prover's constants, with a guard band on the
f64 comparisons; rejected primes are recorded with the reason. This is repeated from the start
of a few larger bit lengths; the smallest Rust proof-size estimate wins, ties to the smaller
modulus. With a per-slot width (`Shape._per_slot`) it also starts from the bit lengths above the
default width's lower bound, where the default rule searches: the narrower width lowers the
lifting bound but not the MSIS fixed point, so the window above the lower bound alone could
miss the default rule's candidates, and the per-slot set could come out larger. `q` stays
below the Rust capacity and, if the request sets it, below
$`2^{\\text{max\\_bits}}`$: a $`\\gamma`$ whose prime reaches that cap is recorded as rejected
and the next $`\\gamma`$ is tried; the search reports a capacity error only when no $`\\gamma`$
gives a prime below it. $`\\lambda`$ follows Rust's f64 rule (`lambda_for`). Within its
$`\\gamma`$ class the chosen prime is then the smallest that passes, provided the MSIS bound
does not decrease as $`q`$ grows; the tests check the previous prime of the class at each kind
of binding bound.
"""
import copy
import math

import mpmath as mp

from . import hardness, modulus, rustcheck

PREC = 300
# The slack factors gamma_i of the widths sigma_i = gamma_i T_i, T_i the bound on the masked
# vector (LNP22 Section 6.1, which takes other values in its examples); Jali's choice.
GAMMAS = (14, 1, 5, 5)


# Rust check failures that a different modulus can cure; every other failure is final.
Q_DEPENDENT = {"compressed commitment bit width", "MSIS estimate", "rounding bound", "compression",
               "ARP modulus bound", "binary lifting bound", "slack lifting bound",
               "exact norm lifting bound", "modulus lifting bound",
               "compressed response bound capacity", "noninvertible statement modulus",
               "noninvertible constraint modulus", "approximate extraction bound", "shape"}


class DeriveError(ValueError):
    """A typed failure: `kind` names the binding condition."""

    def __init__(self, kind, detail="", need_q=None, final=False):
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind
        self.need_q = need_q
        self.final = final  # no other modulus can cure it


def rounded(value):
    """The exponent $`t`$ whose width $`1.55\\cdot2^t`$ is nearest `value` (lnp_params.py:41-45)."""
    if value == 0:
        return 0
    c = mp.mpf("1.55")
    t = max(0, int(mp.floor(mp.log(value / c, 2))))
    return t if abs(c * 2 ** t - value) <= abs(c * 2 ** (t + 1) - value) else t + 1


class Shape:
    """A statement shape in the proof ring: the fields of `statement::Requirements`."""

    def __init__(self, req, degree):
        if degree not in (64, 128):
            raise DeriveError("invalid proof ring", f"degree {degree}")
        self.d = degree
        self.omega, self.eta, _ = rustcheck.CHALLENGE[degree]
        for key in ("m1", "l", "alpha_squared", "n_bin", "n_prime", "linf_bound"):
            v = req[key]
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise DeriveError("invalid requirements", f"{key}={v!r}")
            setattr(self, key, v)
        self.l2_rows = list(req["l2_rows"])
        self.l2_bounds_squared = list(req["l2_bounds_squared"])
        if len(self.l2_rows) != len(self.l2_bounds_squared):
            raise DeriveError("mismatched exact blocks")
        # `TboxParams` stores these bounds as u64 (Requirements has u128 for two of them).
        wide = [("alpha_squared", self.alpha_squared), ("linf_bound", self.linf_bound)] + \
            [("l2_bounds_squared", b) for b in self.l2_bounds_squared]
        for key, v in wide:
            if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < rustcheck.U64:
                raise DeriveError("field capacity", f"{key} = {v!r} does not fit the u64 field "
                                  "of TboxParams")
        self.z = len(self.l2_rows)
        self.n_ex = sum(self.l2_rows) + self.n_bin + self.z
        dims = [self.m1, self.l, self.n_bin, self.n_prime] + self.l2_rows
        if any(x > 65535 for x in dims):
            raise DeriveError("dimensions", "a dimension exceeds 65535")
        if 0 in self.l2_rows or 0 in self.l2_bounds_squared or self.z > 1024:
            raise DeriveError("exact norm blocks", "zero rows or zero bound")
        if (self.n_prime == 0) != (self.linf_bound == 0):
            raise DeriveError("approximate block", "n_prime and linf_bound must be both zero or "
                              "both positive")
        with mp.workprec(PREC):
            self.log_sigma = [
                rounded(GAMMAS[0] * self.eta * mp.sqrt(self.alpha_squared + self.z * degree)),
                0,
                rounded(GAMMAS[2] * mp.sqrt(337) * mp.sqrt(
                    sum(self.l2_bounds_squared) + (self.n_bin + self.z) * degree)),
                rounded(GAMMAS[3] * mp.sqrt(337) * mp.sqrt(self.n_prime * degree)
                        * self.linf_bound),
            ]
        cap = rustcheck.CAPACITY["log_sigma"]
        # With a per-slot bound (Requirements::approx_alpha_squared) sigma_4 may be narrower.
        self.approx_alpha_squared = req.get("approx_alpha_squared")
        self.range_width = None
        if self.approx_alpha_squared is not None:
            self.range_width = self._per_slot(cap)
            self.log_sigma[3] = self.range_width["log_sigma3"]
        if any(t > cap for t in self.log_sigma):
            raise DeriveError("Gaussian width capacity",
                              f"log_sigma {self.log_sigma} exceeds {cap}")

    def _per_slot(self, cap):
        """`log_sigma[3]` by the per-slot rule: the larger of `rounded` of
        $`5\\sqrt{337}\\,\\alpha_{slot}`$, for $`\\alpha_{slot}^2`$ = `approx_alpha_squared`, and
        the guard, the smallest $`t`$ whose rejection constant for the default bound
        $`n'd\\beta_\\infty^2`$ is 2, in f64 as Rust computes it
        (`rustcheck.range_rejection_constant`). The prover's constant keeps the default bound,
        which its witness check enforces, so rejection sampling stays exact at any accepted
        width; the guard keeps it at 2, as the default rule gives. Returns the record for the
        report."""
        a = self.approx_alpha_squared
        default = self.n_prime * self.d * self.linf_bound ** 2
        if isinstance(a, bool) or not isinstance(a, int) or not 0 <= a < min(default, 1 << 128):
            raise DeriveError("invalid requirements", f"approx_alpha_squared={a!r} is not a "
                              f"u128 below n' d linf_bound^2 = {default}")
        with mp.workprec(PREC):
            slot = rounded(GAMMAS[3] * mp.sqrt(337) * mp.sqrt(a))
        guard = None
        for t in range(cap + 1):
            try:
                if rustcheck.range_rejection_constant(t, default, cap) == 2:
                    guard = t
                    break
            except rustcheck.RustCheckError:
                continue
        if guard is None:
            raise DeriveError("Gaussian width capacity", "no width up to the capacity has "
                              "range rejection constant 2")
        t = max(slot, guard)
        return {"rule": "per-slot (Requirements::approx_alpha_squared)",
                "approx_alpha_squared": a, "default_alpha_squared": default,
                "log_sigma3_default": self.log_sigma[3], "log_sigma3_per_slot": slot,
                "log_sigma3_guard": guard, "log_sigma3": t,
                "binding": "per-slot" if slot >= guard else "guard"}

    def requirements(self):
        return {"m1": self.m1, "l": self.l, "alpha_squared": self.alpha_squared,
                "n_bin": self.n_bin, "l2_rows": list(self.l2_rows),
                "l2_bounds_squared": list(self.l2_bounds_squared), "n_prime": self.n_prime,
                "linf_bound": self.linf_bound}


def lambda_for(q1):
    """$`\\lambda`$ by Rust's rule (`rustcheck.rust_lambda`: log2 of the nearest f64), not the
    exact logarithm, so that a set near $`2^{64}`$ gets the `l_ext` that Rust derives."""
    return rustcheck.rust_lambda(q1)


def l_ext_for(shape, lam):
    return (256 // shape.d if shape.n_ex else 0) + (256 // shape.d if shape.n_prime else 0) \
        + 1 + lam // 2 + 1


def msis_bound(shape, logs, n, m2, D, gamma):
    """LNP22's bound on the extracted MSIS solution, as `TboxParams::check` computes it."""
    s1, s2 = (mp.mpf("1.55") * 2 ** t for t in logs[:2])
    d = shape.d
    b = s2 * mp.sqrt(2 * m2 * d) + shape.eta * mp.mpf(2) ** (D - 1) * mp.sqrt(n * d) \
        + gamma * mp.sqrt(n * d) / 2
    b1 = 2 * s1 * mp.sqrt(2 * (shape.m1 + shape.z) * d)
    return 4 * shape.eta * mp.sqrt(b1 * b1 + 4 * b * b)


def solve(shape, factors, policy=None, external=None, gamma=None):
    """Derive at a fixed modulus. `gamma=None` takes the largest even divisor of $`q-1`$ in
    the window $`(4\\gamma_0/5,\\gamma_0]`$ (the positional form's rule); an integer uses that
    divisor, which must keep MSIS hard. Returns (params without id/estimator, info)."""
    with mp.workprec(PREC):
        return _solve(shape, factors, policy, external, gamma)


def _solve(shape, factors, policy, external, gamma_given):
    d = shape.d
    q = math.prod(factors)
    if len(factors) not in (1, 2) or any(p % 8 != 5 or not modulus.is_prime(p) for p in factors):
        raise DeriveError("invalid proof ring")
    if sorted(set(factors)) != list(factors):
        raise DeriveError("prime factors must be strictly ascending")
    if len(factors) > 1 and (shape.n_ex or shape.n_prime):
        raise DeriveError("binary, exact-norm and range blocks require a prime modulus")
    if (shape.n_prime == 0) != (shape.linf_bound == 0):
        raise DeriveError("approximate block", "n_prime and linf_bound must be both zero or both "
                          "positive")
    lam = lambda_for(factors[0])
    lext = l_ext_for(shape, lam)
    if external is not None:
        k, delta = external["rank"], external["delta"]
        beta = None
    else:
        k, beta, delta = hardness.rank(d, q, policy)
    pol = policy or hardness.Policy("delta")
    logs = list(shape.log_sigma)

    def delta_of(b, n):
        return hardness.delta_msis(b, n, d, q)

    def hard(n, m2, D, g):
        b = msis_bound(shape, logs, n, m2, D, g)
        return b < q and pol.msis_ok(delta_of(b, n))
    n = 1
    while True:
        m2 = k + n + shape.l + lext
        logs[1] = rounded(shape.eta * mp.sqrt(m2 * d))
        if hard(n, m2, 0, 0):
            break
        b = msis_bound(shape, logs, n, m2, 0, 0)
        if b >= q and pol.msis_ok(delta_of(b, n)):
            raise DeriveError("MSIS bound >= q", f"bound 2^{float(mp.log(b, 2)):.3f}",
                              need_q=int(mp.floor(b)) + 1)
        n += 1
        if n > 65535:
            raise DeriveError("MSIS search exhausted")
    # `TboxParams::gamma` is a u64, so gamma_0 starts at most at 2^63.
    gamma0 = min(2 ** (q.bit_length() - 1), 1 << 63)
    while not hard(n, m2, 0, gamma0):
        gamma0 //= 2
    if gamma_given is None:
        gamma = modulus.divisor_in_window(q - 1, gamma0, "largest")
        if gamma is None:
            raise DeriveError("no suitable divisor; choose another prime and rerun the estimator")
    else:
        gamma = gamma_given
        if (q - 1) % gamma or gamma % 2:
            raise DeriveError("compression", f"gamma={gamma} is not an even divisor of q-1")
        if not hard(n, m2, 0, gamma):
            raise DeriveError("gamma too large", f"gamma={gamma}, gamma_0={gamma0}")
    D = q.bit_length() - 1
    while not (hard(n, m2, D, gamma) and mp.mpf(2) ** (D - 1) * shape.omega * d < gamma):
        D -= 1
        if D < 0:
            raise DeriveError("no compression exponent")
    b = msis_bound(shape, logs, n, m2, D, gamma)
    dm = delta_of(b, n)
    if external is None:
        mlwe_delta = delta
    else:
        mlwe_delta = external["delta"]
    params = {
        "prime_factors": list(factors), "degree": d, "m1": shape.m1, "l": shape.l,
        "alpha_squared": shape.alpha_squared, "n_bin": shape.n_bin,
        "l2_rows": list(shape.l2_rows), "l2_bounds_squared": list(shape.l2_bounds_squared),
        "n_prime": shape.n_prime, "linf_bound": shape.linf_bound, "m2": m2, "n_msis": n,
        "log_sigma": logs, "gamma": gamma, "d_bits": D, "mlwe_rank": k,
        "mlwe_delta": mlwe_delta,
    }
    info = {"q": q, "lambda": lam, "l_ext": lext, "gamma0": gamma0, "mlwe_beta": beta,
            "msis_bound": b, "msis_delta": dm, "msis_beta": hardness.beta_from_delta(dm),
            "gamma_in_window": modulus.in_window(gamma, gamma0)}
    return params, info


def lower_bounds(shape, lifting=None, min_bits=None):
    """Lower bounds on $`q`$ that do not depend on $`q`$, as {name: smallest admissible q}, from
    the checks of `TboxParams::check` (t = 1.64) and of `compile`: the lifting check, the
    largest over the lifted moduli and, for constraints with $`\\ell_\\infty`$ variables, at
    their extraction bound $`E`$, and with such variables $`2E+1<q`$. The range condition's
    limit on $`E`$ does not depend on $`q`$: a set that fails it fails at every modulus."""
    out = {}
    lifting = rustcheck.full_lifting(lifting)
    if shape.n_ex:
        s3 = 1.55 * 2.0 ** shape.log_sigma[2]
        arp = 2.0 * math.sqrt(256.0 / 26.0) * 1.64 * s3

        def above(x, strict):
            v = math.floor(x)
            return v + 1 if strict or v < x else v
        out["ARP modulus bound"] = above(41.0 * float(shape.n_ex * shape.d) * arp, False)
        out["binary lifting bound"] = above(arp * arp + arp * math.sqrt(shape.n_bin * shape.d),
                                            True)
        out["slack lifting bound"] = above(arp * arp + arp * math.sqrt(shape.d), True)
        for i, bsq in enumerate(shape.l2_bounds_squared):
            out[f"exact norm lifting bound {i}"] = above(3.0 * bsq + arp * arp, True)
    if shape.n_prime and lifting is not None:
        widths = {"log_sigma": shape.log_sigma, "linf_bound": shape.linf_bound}
        e = rustcheck.extraction_bound(shape.log_sigma[3])
        limit = lifting["extraction_limit"]
        if lifting["has_linf"] and limit is not None and e > limit:
            raise DeriveError("variable range above statement modulus",
                              f"the extraction bound {e} of log_sigma[3] = "
                              f"{shape.log_sigma[3]} exceeds the range limit {limit}",
                              final=True)
        if lifting["lifted"]:
            try:
                out["modulus lifting bound"] = rustcheck.lifting_rhs(
                    widths, lifting["pairs"], lifting["linf"]) + 1
            except rustcheck.RustCheckError as err:
                raise DeriveError(err.rust_message, "at every modulus", final=True) from None
        if lifting["has_linf"]:
            out["approximate extraction bound"] = 2 * e + 2
    if min_bits:
        out["min_bits"] = 1 << (min_bits - 1)
    return out


def check(params, shape, lifting, capacity):
    """The emulated Rust checks of a derived set: `TboxParams::check`, the prover's constants and,
    with lifted rows, `compile` against the shape's requirements."""
    derived, checks = rustcheck.check_params(params, capacity)
    rustcheck.prover_checks(derived)
    lifting = rustcheck.full_lifting(lifting)
    if lifting is not None and shape.n_prime:
        req = dict(shape.requirements(), max_integer_coefficient=lifting["max_integer_coefficient"])
        rustcheck.check_compile(params, req, lifting["statement_modulus"], checks, lifting)
    return derived, checks


def _state(shape, policy, q, k=None):
    """(rank, n_msis, m2, log_sigma) of `solve` at the modulus value `q` (not necessarily prime)
    with $`\\gamma=D=0`$; raises "MSIS bound >= q" with the modulus that would be needed."""
    d = shape.d
    lext = l_ext_for(shape, lambda_for(q))
    if k is None:
        k = hardness.rank(d, q, policy)[0]
    logs = list(shape.log_sigma)
    n = 1
    while True:
        m2 = k + n + shape.l + lext
        logs[1] = rounded(shape.eta * mp.sqrt(m2 * d))
        b = msis_bound(shape, logs, n, m2, 0, 0)
        ok_delta = policy.msis_ok(hardness.delta_msis(b, n, d, q))
        if ok_delta and b < q:
            return k, n, m2, logs
        if ok_delta:
            raise DeriveError("MSIS bound >= q", need_q=int(mp.floor(b)) + 1)
        n += 1
        if n > 65535:
            raise DeriveError("MSIS search exhausted")


def _hard_with(shape, policy, q, gamma, st):
    _, n, m2, logs = st
    b = msis_bound(shape, logs, n, m2, 0, gamma)
    return b < q and policy.msis_ok(hardness.delta_msis(b, n, shape.d, q))


def _gamma0(shape, policy, q, st):
    """`solve`'s $`\\gamma_0`$: the largest power of two below q (and at most $`2^{63}`$)
    keeping MSIS hard."""
    g = min(2 ** (q.bit_length() - 1), 1 << 63)
    while g and not _hard_with(shape, policy, q, g, st):
        g //= 2
    return g


def _min_start(shape, policy, s, gamma):
    """The smallest integer $`q\\ge s`$ at which `gamma` keeps MSIS hard, or None below $`2s`$.
    Uses that hardness grows with q: the rank is fixed where it is equal at both ends."""
    def hard(q, k=None):
        try:
            st = _state(shape, policy, q, k)
        except DeriveError:
            return False, None
        return _hard_with(shape, policy, q, gamma, st), st[0]
    ok, k_s = hard(s)
    if ok:
        return s
    step = max(1, s >> 24)
    while True:
        hi = s + step
        ok, k_hi = hard(hi)
        if ok:
            break
        if hi > 2 * s:
            return None
        step *= 2
    lo = s
    fixed = k_s is not None and k_s == k_hi and lambda_for(s) == lambda_for(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if hard(mid, k_hi if fixed else None)[0]:
            hi = mid
        else:
            lo = mid
    return hi


def limit(capacity, max_bits=None):
    """The search's bound on q, as (bits, description): the Rust capacity, or the request's
    `max_bits` when that is smaller."""
    cap = min(capacity["prime_bits"], capacity["q_bits"])
    if max_bits is not None and max_bits < cap:
        return max_bits, f"the request's max_bits (q < 2^{max_bits})"
    return cap, f"the Rust capacity 2^{cap}"


def _attempt(shape, policy, start, capacity, lifting, trace, max_bits=None):
    """The best set whose modulus is at or above `start`: the MSIS fixed point for
    $`\\gamma=0`$, then for a few divisors $`\\gamma`$ around $`\\gamma_0`$ the smallest prime of
    its class at which $`\\gamma`$ keeps MSIS hard; the smallest estimated proof wins."""
    min_gamma = shape.omega * shape.d // 2
    cap, cap_text = limit(capacity, max_bits)
    s = start
    with mp.workprec(PREC):
        for _ in range(64):
            try:
                st = _state(shape, policy, s)
                break
            except DeriveError as e:
                if e.need_q is None:
                    raise
                trace.append({"probe": s, "event": e.kind, "next_start": max(e.need_q, s + 1)})
                s = max(e.need_q, s + 1)
                if s >= 1 << cap:
                    raise DeriveError("capacity", f"the MSIS bound needs q >= 2^{math.log2(s):.2f}"
                                      f", above {cap_text}")
        else:
            raise DeriveError("search exhausted", "MSIS fixed point")
        g0 = _gamma0(shape, policy, s, st)
    top = max(g0, 2 << (min_gamma.bit_length() - 1))
    cands = set()
    for j in range(max(top.bit_length() - 2, min_gamma.bit_length()), top.bit_length() + 2):
        g = modulus.top_gamma(1 << (j - 1), min_gamma)
        if g is not None and g < rustcheck.U64:
            cands.add(g)
    if not cands:
        raise DeriveError("no admissible gamma", f"none above omega*d/2={min_gamma}")
    results, over_cap = [], []
    for gamma in sorted(cands, reverse=True):
        with mp.workprec(PREC):
            s_g = _min_start(shape, policy, s, gamma)
        event = {"start": s, "gamma": gamma, "gamma0_at_start": g0}
        trace.append(event)
        if s_g is None:
            event["event"] = "gamma keeps MSIS hard only above 2*start"
            continue
        rejected = []
        for _ in range(16):
            q, tries = modulus.construct_prime(s_g, gamma)
            event.update(q=q, candidates_tested=tries)
            if q >= 1 << cap:
                # Only this gamma is out of range; a smaller one may give a prime below the cap.
                why = f"capacity: q = 2^{math.log2(q):.2f} exceeds {cap_text}"
                rejected.append({"q": q, "why": why})
                event["event"] = why
                over_cap.append(f"gamma {gamma}: {why}")
                break
            try:
                params, info = solve(shape, [q], policy, gamma=gamma)
            except DeriveError as e:
                if e.kind == "gamma too large" or e.need_q is not None:
                    rejected.append({"q": q, "why": e.kind})
                    s_g = q + 1
                    continue
                raise
            params["id"], params["estimator"] = "probe", "probe"
            try:
                derived, checks = check(params, shape, lifting, capacity)
            except rustcheck.RustCheckError as e:
                if e.rust_message not in Q_DEPENDENT:
                    raise DeriveError(e.rust_message, "refused by the emulated Rust check at "
                                      "every modulus", final=True) from None
                rejected.append({"q": q, "why": f"rust check: {e.rust_message}"})
                s_g = q + 1
                continue
            tight = checks.tight()
            if tight:
                rejected.append({"q": q, "why": f"within guard band: {tight}"})
                s_g = q + 1
                continue
            event.update(event="accepted", estimated_proof_bytes=derived["estimated_proof_bytes"])
            results.append((derived["estimated_proof_bytes"], q, params, info, derived, checks))
            break
        else:
            event["event"] = "no prime of the class passed in 16 tries"
        if rejected:
            event["rejected"] = rejected
    if not results:
        if over_cap:
            raise DeriveError("capacity", f"no gamma gives a q below the cap from {s}: "
                              f"{'; '.join(over_cap)}")
        raise DeriveError("no admissible gamma", f"no candidate divisor works from {s}")
    best = min(results, key=lambda r: (r[0], r[1]))
    return best[2], best[3], best[4], best[5]


def search(shape, policy, lifting=None, window_bits=4, min_bits=None,
           capacity=rustcheck.CAPACITY, max_bits=None):
    """Size the modulus, keeping $`q<2^{\\text{max\\_bits}}`$ when `max_bits` is given: returns
    (params, info, derived, checks, search record)."""
    if shape.n_prime and lifting is None:
        raise DeriveError("missing statement", "a range block needs the statement modulus and "
                          "max_integer_coefficient for the lifting bound")
    cap, cap_text = limit(capacity, max_bits)
    if min_bits and min_bits - 1 >= cap:
        # Before 2^(min_bits-1) is formed: a huge min_bits would not fit in memory.
        raise DeriveError("capacity", f"min_bits needs q >= 2^{min_bits - 1}, above {cap_text}")
    bounds = lower_bounds(shape, lifting, min_bits)
    q_lo = max(list(bounds.values()) + [1 << 16])
    binding = max(bounds, key=bounds.get) if bounds else "none"
    if q_lo >= 1 << cap:
        raise DeriveError("capacity", f"{binding} needs q >= 2^{math.log2(q_lo):.2f}, above "
                          f"{cap_text}")
    starts = [(bits, max(q_lo, 1 << (bits - 1)))
              for bits in range(q_lo.bit_length(), q_lo.bit_length() + window_bits + 1)]
    default_q_lo = _default_width_start(shape, lifting, min_bits)
    if default_q_lo is not None and default_q_lo > q_lo:
        # The per-slot width lowers the lifting bound, and so q_lo, but not the MSIS fixed
        # point: the window may then collapse onto that point. The default width's window as
        # well keeps every candidate of the default rule, now with the narrower sigma_4.
        starts = sorted(set(starts) | {
            (bits, max(default_q_lo, 1 << (bits - 1)))
            for bits in range(default_q_lo.bit_length(),
                              default_q_lo.bit_length() + window_bits + 1)},
            key=lambda x: (x[1], x[0]))
    candidates, trace, capacity_error = [], [], None
    for bits, start in starts:
        if start >= 1 << cap:
            break
        local = []
        try:
            params, info, derived, checks = _attempt(shape, policy, start, capacity, lifting,
                                                     local, max_bits)
        except DeriveError as e:
            if e.final:
                raise
            trace.append({"bits": bits, "start": start, "failed": str(e), "steps": local})
            if e.kind == "capacity":
                capacity_error = e
                break
            continue
        trace.append({"bits": bits, "start": start, "q": params["prime_factors"][0],
                      "estimated_proof_bytes": derived["estimated_proof_bytes"], "steps": local})
        candidates.append((derived["estimated_proof_bytes"], params["prime_factors"][0],
                           params, info, derived, checks))
    if not candidates:
        if capacity_error is not None:
            raise capacity_error
        reasons = sorted({t["failed"] for t in trace if "failed" in t})
        raise DeriveError("no feasible modulus", f"from 2^{math.log2(q_lo):.2f} within "
                          f"{window_bits} extra bits: {'; '.join(reasons)}")
    best = min(candidates, key=lambda c: (c[0], c[1]))
    q = best[1]
    # The binding condition at the chosen q: the smallest margin among the lower bounds and the
    # MSIS condition Bound < q.
    with mp.workprec(PREC):
        margins = {k: mp.log(mp.mpf(q) / v, 2) for k, v in bounds.items()}
        margins["MSIS bound < q"] = mp.log(q / best[3]["msis_bound"], 2)
    binding_at_q = min(margins, key=margins.get)
    record = {"lower_bounds": bounds, "q_lo": q_lo, "largest_lower_bound": binding,
              "binding_lower_bound": binding_at_q, "window_bits": window_bits,
              "max_bits": max_bits, "trace": trace}
    if default_q_lo is not None:
        record["default_width_q_lo"] = default_q_lo
    return best[2], best[3], best[4], best[5], record


def _default_width_start(shape, lifting, min_bits):
    """For a shape with a per-slot width, the search's starting modulus under the default
    width, or None (without a per-slot width, or when that width fails at every modulus, as the
    range limit on the extraction bound can). MSIS, gamma and the prime construction do not
    depend on `log_sigma[3]`, so from these starts the search meets every candidate of the
    default rule, each passing the same checks with a smaller lifting bound and a smaller
    estimate."""
    if shape.range_width is None:
        return None
    default = copy.copy(shape)
    default.log_sigma = shape.log_sigma[:3] + [shape.range_width["log_sigma3_default"]]
    default.range_width = None
    try:
        bounds = lower_bounds(default, lifting, min_bits)
    except DeriveError:
        return None
    return max(list(bounds.values()) + [1 << 16])
