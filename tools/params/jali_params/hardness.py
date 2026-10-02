"""MLWE and MSIS hardness estimates, and the security policies built on them.

Every figure here comes from a cost model: it is an estimate under heuristics, not a proof.

MLWE, primal uSVP. MLWE of rank $`k`$ over degree $`d`$ is taken as LWE of dimension
$`n=kd`$ whose secret and error coefficients are uniform on $`\\{-\\nu,\\dots,\\nu\\}`$, of
standard deviation $`s=\\sqrt{\\nu(\\nu+1)/3}`$; the secret is in normal form and the attacker
may take any number $`m`$ of samples. Kannan's embedding with embedding coefficient $`s`$ gives
a lattice of dimension $`D=m+n+1`$ and volume $`s\\,q^m`$ that contains a vector of norm about
$`s\\sqrt D`$. The 2016 estimate (Alkim, Ducas, Pöppelmann and Schwabe, "Post-quantum key
exchange - a new hope", USENIX Security 2016; examined by Albrecht, Göpfert, Virdia and
Wunderer, ASIACRYPT 2017) takes BKZ with block size $`b`$ to find it when its projection on the
last $`b`$ Gram-Schmidt directions is shorter than the $`b`$-th last Gram-Schmidt vector under
the geometric series assumption:
$`s\\sqrt b\\le\\delta_0(b)^{2b-D}(s\\,q^m)^{1/D}`$, the form of Eq. (2) of Albrecht et al.
(Section 3.2 of ePrint 2017/815). Eq. (1) of the 2016 paper (Section 6.3 of ePrint 2015/1092)
has the exponent $`2b-D-1`$, which footnote 7 of Albrecht et al. takes for an error; as
$`\\delta_0(b)>1`$, the form used never gives a larger block size. The root Hermite factor is
$`\\delta_0(b)=((b/(2\\pi e))(\\pi b)^{1/b})^{1/(2(b-1))}`$ (Y. Chen's thesis, Paris 7, 2013),
used from $`b=50`$ on. In logarithms the right side is
$`g(D)=(2b-D)\\ln\\delta_0+\\ln q-c/D`$ with $`c=(n+1)\\ln q-\\ln s`$, concave in $`D`$ with its
maximum at $`D^*=\\sqrt{c/\\ln\\delta_0}`$, so the best $`m`$ is found at
$`\\lfloor D^*\\rfloor`$ or $`\\lceil D^*\\rceil`$, but at least $`D=n+1`$ (`usvp_max_g`). The
block size of an instance is the smallest $`b\\ge50`$ for which this succeeds; smaller ones are
reported as 50.

MSIS: the closed form $`\\delta=2^{(\\log_2B)^2/(4nd\\log_2q)}`$ of the standard methodology
(Micciancio and Regev 2009; Gama and Nguyen, EUROCRYPT 2008) that LNP22 Section 6.1 applies:
in the best subdimension, BKZ with root Hermite factor $`\\delta`$ finds vectors of length
$`2^{2\\sqrt{nd\\log_2q\\log_2\\delta}}`$, which is $`B`$ at this $`\\delta`$. Its block size is
the smallest $`b`$ with $`\\delta_0(b)\\le\\delta`$.

Core-SVP: $`2^{0.292b}`$ classical (Becker, Ducas, Gama and Laarhoven, SODA 2016) and
$`2^{0.265b}`$ quantum (Laarhoven's thesis, TU Eindhoven, 2015), the cost model of the 2016
estimate.
"""
import mpmath as mp

PREC = 128  # bits of the estimates; the decisions do not depend on it (see the tests)
with mp.workprec(512):
    DELTA_MAX = mp.mpf("1.0044")  # the ceiling of every policy and of `TboxParams::check`
BETA_FLOOR = 346  # the smallest block size b with delta_0(b) <= DELTA_MAX
MIN_BETA = 50  # the root Hermite factor formula is used from this block size on
CORE_SVP = {"classical": 0.292, "quantum": 0.265}


class HardnessError(ValueError):
    pass


def delta_0(b):
    """The root Hermite factor $`\\delta_0(b)`$ of BKZ with block size `b`, at `PREC` bits."""
    if b < MIN_BETA:
        raise ValueError(f"the root Hermite factor formula is used from b = {MIN_BETA} on")
    with mp.workprec(PREC):
        b = mp.mpf(b)
        return (b / (2 * mp.pi * mp.e) * (mp.pi * b) ** (1 / b)) ** (1 / (2 * (b - 1)))


def stddev(nu):
    """Standard deviation of the uniform distribution on $`\\{-\\nu,\\dots,\\nu\\}`$."""
    return mp.sqrt(mp.mpf(nu * (nu + 1)) / 3)


def usvp_max_g(b, n, q, nu=1, dims=None):
    """The largest $`g(D)`$ over the lattice dimensions `dims`, and the largest $`D`$ attaining
    it. By default `dims` are $`\\lfloor D^*\\rfloor`$ and $`\\lceil D^*\\rceil`$, each at least
    $`n+1`$; as $`g`$ is concave, they give the maximum over all $`D\\ge n+1`$ (the tests scan
    the others)."""
    with mp.workprec(PREC):
        log_q, log_s = mp.log(q), mp.log(stddev(nu))
        log_delta = mp.log(delta_0(b))
        c = (n + 1) * log_q - log_s
        if dims is None:
            best = mp.sqrt(c / log_delta)
            dims = {max(n + 1, int(mp.floor(best))), max(n + 1, int(mp.ceil(best)))}
        return max(((2 * b - dim) * log_delta + log_q - c / dim, dim) for dim in dims)


def usvp_succeeds(b, n, q, nu=1):
    """Whether the 2016 estimate finds the embedded vector with block size `b`, for LWE of
    dimension `n` and modulus `q` with the best number of samples."""
    with mp.workprec(PREC):
        return mp.log(stddev(nu)) + mp.log(b) / 2 <= usvp_max_g(b, n, q, nu)[0]


def usvp_block_size(n, q, nu=1):
    """The smallest block size in $`[50,2n]`$ for which `usvp_succeeds`, 50 for any smaller
    one; the estimate is undefined when $`2n`$ fails."""
    lo, hi = MIN_BETA, 2 * n
    if hi < lo or not usvp_succeeds(hi, n, q, nu):
        raise HardnessError(f"uSVP estimate undefined for n={n}: no block size in "
                            f"[{MIN_BETA}, 2n] succeeds")
    if usvp_succeeds(lo, n, q, nu):
        return lo
    while hi - lo > 1:  # succeeds at hi, not at lo
        mid = (lo + hi) // 2
        if usvp_succeeds(mid, n, q, nu):
            hi = mid
        else:
            lo = mid
    return hi


def mlwe(k, d, q, nu=1):
    """The uSVP estimate of MLWE with rank `k` over degree `d`: block size and float
    $`\\delta_0`$."""
    beta = usvp_block_size(k * d, q, nu)
    return beta, float(delta_0(beta))


def beta_from_delta(delta):
    """The smallest block size $`b\\ge50`$ with $`\\delta_0(b)\\le`$ `delta`."""
    if delta_0(MIN_BETA) <= delta:
        return MIN_BETA
    lo, hi = MIN_BETA, 2 * MIN_BETA
    while delta_0(hi) > delta:
        lo, hi = hi, 2 * hi
    while hi - lo > 1:  # delta_0(lo) > delta >= delta_0(hi)
        mid = (lo + hi) // 2
        if delta_0(mid) <= delta:
            hi = mid
        else:
            lo = mid
    return hi


def delta_msis(bound, n, d, q):
    """The MSIS closed form, at the caller's precision."""
    return mp.mpf(2) ** (mp.log(bound, 2) ** 2 / (4 * n * d * mp.log(q, 2)))


def core_svp_bits(beta):
    """Core-SVP exponents (heuristic), rounded to 0.1 bit."""
    return {k: round(c * beta, 1) for k, c in CORE_SVP.items()}


class Policy:
    """A security policy: which MLWE and MSIS ranks are acceptable.

    `delta`: uSVP $`\\delta_0\\le1.0044`$ for MLWE (equivalently block size at least 346) and
    closed-form $`\\delta<1.0044`$ for MSIS, a stricter ceiling than the $`\\delta<1.0045`$
    (block size 335) that LNP22 Section 6.1 aims for. `beta`: block size at least `beta_min`
    for both, and the ceiling as well.
    """

    def __init__(self, name, beta_min=None):
        if name == "delta":
            if beta_min is not None:
                raise ValueError("the delta policy takes no beta_min")
        elif name == "beta":
            if not isinstance(beta_min, int) or isinstance(beta_min, bool) \
                    or beta_min < BETA_FLOOR:
                raise ValueError(f"beta policy needs an integer beta_min >= {BETA_FLOOR}")
        else:
            raise ValueError(f"unknown policy {name!r}")
        self.name = name
        self.beta_min = beta_min

    def describe(self):
        if self.name == "delta":
            return {"name": self.name, "mlwe": "uSVP delta_0 <= 1.0044", "msis": "delta < 1.0044"}
        return {"name": self.name, "beta_min": self.beta_min,
                "mlwe": f"uSVP beta >= {self.beta_min}",
                "msis": f"beta >= {self.beta_min} and delta < 1.0044"}

    def label(self):
        return self.name if self.name == "delta" else f"beta>={self.beta_min}"

    def mlwe_ok(self, beta, delta):
        if mp.mpf(delta) > DELTA_MAX:
            return False
        return self.name == "delta" or beta >= self.beta_min

    def msis_ok(self, delta):
        """`delta` at the tool's precision; strict, as `TboxParams::check` is."""
        if not delta < DELTA_MAX:
            return False
        return self.name == "delta" or beta_from_delta(delta) >= self.beta_min


def rank(d, q, policy, nu=1, limit=1 << 16):
    """The smallest MLWE rank that `policy` accepts: an exponential search from 1, then
    bisection, since the block size grows with the rank. Returns (rank, beta, float delta)."""
    estimates = {}

    def ok(k):
        estimates[k] = mlwe(k, d, q, nu)
        return policy.mlwe_ok(*estimates[k])
    hi = 1
    while not ok(hi):
        hi *= 2
        if hi > limit:
            raise HardnessError("MLWE rank search exhausted")
    lo = hi // 2  # not accepted, unless it is 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return (hi, *estimates[hi])
