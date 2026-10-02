"""Primes $`q\\equiv5\\pmod 8`$ with an even compression divisor $`\\gamma\\mid q-1`$.

The divisors come from the window $`(4\\gamma_0/5,\\gamma_0]`$ below the largest power of two
$`\\gamma_0`$ that keeps MSIS hard. The search fixes $`\\gamma`$ first:
$`q\\equiv5\\pmod8`$ means $`v_2(q-1)=2`$, so an even $`\\gamma\\mid q-1`$ has
$`v_2(\\gamma)\\in\\{1,2\\}`$; for such $`\\gamma`$ the primes $`q=1+\\gamma t`$ with $`t`$ odd
($`v_2(\\gamma)=2`$) or $`t\\equiv2\\pmod4`$ ($`v_2(\\gamma)=1`$) are exactly those
$`\\equiv5\\pmod8`$ with $`\\gamma\\mid q-1`$, and no factoring is needed. At a given prime,
`divisor_in_window` finds a divisor of $`q-1`$ in the window.

Primality: `sympy.isprime`, deterministic below $`2^{64}`$ and a strong BPSW probable-prime
test above (no known counterexample; not a proof).
"""
import sympy

# Above this gamma_0 the window scan is replaced by sympy's full factorization of q-1.
SCAN_LIMIT = 1 << 28


class ModulusError(ValueError):
    pass


def is_prime(n):
    return bool(sympy.isprime(n))


def primality_note(n):
    return ("deterministic (sympy.isprime, n < 2^64)" if n < 1 << 64
            else "probable prime (sympy.isprime: strong BPSW); not proven")


def rust_is_prime(n):
    """`jali::math::int::is_prime` (u64 Miller-Rabin with fixed bases), for the emulation."""
    if not 0 <= n < 1 << 64:
        raise ModulusError("Rust primality takes a u64")
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    s = ((n - 1) & -(n - 1)).bit_length() - 1
    d = (n - 1) >> s
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a % n, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(1, s):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def v2(n):
    return (n & -n).bit_length() - 1


def in_window(c, gamma0):
    """The window: $`4\\gamma_0/5<c\\le\\gamma_0`$, $`c`$ even (exact)."""
    return c % 2 == 0 and 5 * c > 4 * gamma0 and c <= gamma0


def divisor_in_window(n, gamma0, which):
    """The smallest or largest even divisor of `n` in the window, or None. Exact: a scan
    of the window for $`\\gamma_0\\le2^{28}`$, the divisors of a full factorization above."""
    if which not in ("smallest", "largest"):
        raise ValueError(which)
    if gamma0 < 2:
        return None
    if gamma0 > SCAN_LIMIT:
        found = [int(c) for c in sympy.divisors(n) if in_window(int(c), gamma0)]
        return (min(found) if which == "smallest" else max(found)) if found else None
    lo = 4 * gamma0 // 5 + 1
    lo += lo % 2
    hi = gamma0 - gamma0 % 2
    rng = range(lo, hi + 1, 2) if which == "smallest" else range(hi, lo - 1, -2)
    for c in rng:
        if n % c == 0:
            return c
    return None


def admissible_gammas(gamma0):
    """Even $`\\gamma`$ in the window with $`v_2(\\gamma)\\in\\{1,2\\}`$, largest first."""
    g = gamma0 - gamma0 % 2
    while 5 * g > 4 * gamma0:
        if g > 0 and v2(g) in (1, 2):
            yield g
        g -= 2


def top_gamma(gamma0, above=0):
    """The largest admissible $`\\gamma`$ in the window below $`\\gamma_0`$ that exceeds
    `above`, or None."""
    if gamma0 <= above:
        return None
    for g in admissible_gammas(gamma0):
        return g if g > above else None
    return None


def construct_prime(q_min, gamma, max_tries=1 << 22):
    """Smallest prime $`q\\ge`$ `q_min` with $`q\\equiv5\\pmod8`$ and $`\\gamma\\mid q-1`$.
    Returns (q, number of candidates tested)."""
    if gamma < 2 or v2(gamma) not in (1, 2):
        raise ModulusError(f"gamma={gamma} cannot divide q-1 for a prime q = 5 mod 8")
    step = 2 if v2(gamma) == 2 else 4  # t odd, or t = 2 mod 4
    first = 1 if step == 2 else 2
    t = max(1, -(-(q_min - 1) // gamma))
    t += (first - t) % step
    for tries in range(1, max_tries + 1):
        q = 1 + gamma * t
        if is_prime(q):
            if q % 8 != 5 or (q - 1) % gamma or q < q_min:
                raise AssertionError("construction invariant")
            return q, tries
        t += step
    raise ModulusError(f"no prime found in {max_tries} candidates")
