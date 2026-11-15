"""Extraction sets of the straight-line extractable mode (Katsumata's transform, ePrint 2021/927).

The exact derivation of the LinHC parameters of `src/linhc.rs` (`derive`) and
`src/params/extraction.rs` (`ExtractionParams::modulus_for`), and the smallest LinHC rank that
a policy accepts. The crate derives the modulus again from the parameter set and the rank, and
refuses a set whose modulus differs, so a set that this module writes and the crate accepts
agrees with the Rust derivation exactly.
"""
import math

import sympy

from . import VERSION, hardness, jsonio, rustcheck

PHI = 53  # sigma' >= PHI * T, as `params::extraction::PHI`
MAX_LOG_SIGMA = 40  # as `linhc::MAX_LOG_SIGMA`
MODEL = ("LinHC commitment: MLWE of rank n over the proof degree modulo the LinHC modulus q', "
         "primal uSVP (ADPS16) with secret and error uniform in {-1,0,1}, any number of samples")


class ExtractionError(ValueError):
    pass


def ceil_sqrt(x):
    """The least r with r^2 >= x, as `linhc::ceil_sqrt`."""
    if x < 0:
        raise ValueError("negative")
    return 0 if x == 0 else math.isqrt(x - 1) + 1


def linhc(degree, rank, length, bound, eta, phi=PHI):
    """`linhc::derive`: the least mask exponent with sigma' >= phi T, p = 4B + 1 and the least
    prime q' above 2(pN + B), with the bounds the crate derives on the way."""
    if rank <= 0 or length <= 0 or bound <= 0:
        raise ExtractionError("LinHC dimensions")
    plaintext = 4 * bound + 1
    shift = eta * eta * (rank + 2 * length) * degree
    target = phi * phi * shift * 400
    t = 1
    while (961 << (2 * t)) < target:
        t += 1
        if t > MAX_LOG_SIGMA:
            raise ExtractionError("LinHC mask exponent")
    dims = (rank * degree, length * degree, length * degree)
    blocks = [((961 << (2 * t)) * dim) // 200 for dim in dims]
    beta = [ceil_sqrt(b) for b in blocks]
    noise = ceil_sqrt(dims[0]) * beta[0] + ceil_sqrt(dims[1]) * beta[1] + beta[2]
    q = (2 * (plaintext * noise + bound) + 1) | 1
    while not sympy.isprime(q):
        q += 2
    if q >= 1 << 127:
        raise ExtractionError("LinHC modulus")
    return {"plaintext": plaintext, "log_sigma": t, "shift_bound_squared": shift,
            "block_bounds_squared": blocks, "noise_bound": noise, "modulus": q}


def shape(params):
    """The LinHC message length m' = m1 + |l2_rows| + m2 - n_msis, the bound B on every
    coefficient of an accepted response, the challenges' eta and the degree, as the crate takes
    them from a checked parameter set (a parsed set as `rustcheck.from_json` returns it)."""
    derived, _ = rustcheck.check_params(params)
    length = params["m1"] + len(params["l2_rows"]) + params["m2"] - params["n_msis"]
    bound = max(ceil_sqrt(derived["z1_bound_squared"]), ceil_sqrt(derived["b_squared"]))
    return length, bound, derived["eta"], params["degree"]


def modulus_for(params, rank):
    """`ExtractionParams::modulus_for`."""
    length, bound, eta, degree = shape(params)
    if rank <= 0 or rank > length:
        raise ExtractionError("extraction rank")
    return linhc(degree, rank, length, bound, eta)["modulus"]


def search(params, policy):
    """The least LinHC rank n that `policy` accepts at the modulus q'(n) that n itself gives.

    q'(n) grows with n, and so does the least rank the policy accepts at q'(n), so the sequence
    n <- rank(q'(n)) from n = 1 never decreases and stops at the least fixed point. Returns the
    rank, its modulus and the estimate (block size, float delta) at that rank."""
    length, bound, eta, degree = shape(params)
    n = 1
    while True:
        q = linhc(degree, n, length, bound, eta)["modulus"]
        k, _, _ = hardness.rank(degree, q, policy)
        if k <= n:
            break
        n = k
        if n > length:
            raise ExtractionError(f"LinHC rank {n} above the message length {length}")
    beta, delta = hardness.mlwe(n, degree, q)
    if not policy.mlwe_ok(beta, delta):
        raise ExtractionError("LinHC rank search")
    return n, q, beta, delta


def extraction_set(params, policy):
    """The extraction set (the JSON of `ExtractionParams`) for a parsed parameter set."""
    n, q, beta, delta = search(params, policy)
    estimator = (f"jali-params {VERSION} extraction, policy {policy.label()}, "
                 f"LinHC MLWE uSVP (ADPS16) rank {n} beta {beta} delta {delta!r}")
    return {"model": "rom", "rank": n, "modulus": jsonio.encode_int(q), "mlwe_delta": delta,
            "estimator": estimator}
