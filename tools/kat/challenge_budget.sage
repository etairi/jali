# The weight of the fallback of `rand::challenge_from_bytes` on CHALLENGE_BYTES = 4032 uniform
# bytes, exactly, in rational arithmetic: sage tools/kat/challenge_budget.sage
#
# The decoder reads W = floor(8 N / k) words of k = bitlen(2 omega) bits, keeps each with
# probability p = (2 omega + 1) / 2^k, groups the kept ones into draws of d/2 coefficients, and
# falls back to 1 if every whole draw fails the eta test. With r an upper bound on the
# probability that a draw fails it, the fallback has probability at most
# sum_a Bin(W, p)(a) r^floor(a / (d/2)). The draws are independent and uniform on the ambient
# set, so the bound holds for any fixed failure rate up to r.
def p_fail(W, p, half, r):
    return sum(binomial(W, a) * p**a * (1 - p)**(W - a) * r**(a // half) for a in range(W + 1))

N = 4032
for d, omega in [(64, 8), (128, 2)]:
    k = ZZ(2 * omega).nbits()
    p = QQ(2 * omega + 1) / 2**k
    W = 8 * N // k
    for r in [QQ(1) / 4, QQ(1) / 50]:
        print(f"d={d} omega={omega} words={W} r<={r}: log2 P(fallback) = "
              f"{RR(log(p_fail(W, p, d // 2, r), 2)):.2f}")
