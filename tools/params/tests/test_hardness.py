"""The uSVP estimate against its recorded grid and its own premises, and the policy arithmetic."""
import math
import random
import unittest

import mpmath as mp

import common
import mlwe_grid
from jali_params import hardness


class Grid(unittest.TestCase):
    """The recorded rank searches (`mlwe_grid.py`, tests/data)."""

    @classmethod
    def setUpClass(cls):
        cls.grid = common.grid()

    def test_grid_shape(self):
        self.assertEqual(len(self.grid), 54)
        self.assertEqual(sum(len(r["evals"]) for r in self.grid), 562)

    def test_every_recorded_evaluation(self):
        for r in self.grid:
            for e in r["evals"]:
                with self.subTest(d=r["d"], log2q=r["log2q"], k=e["k"]):
                    self.assertEqual(hardness.mlwe(e["k"], r["d"], int(r["q"])),
                                     (e["beta"], e["delta"]))

    def test_the_file_is_current(self):
        # The same searches, ranks evaluated in the same order, and the same selected ranks.
        self.assertEqual(common.read("tools/params/tests/data/mlwe_grid.jsonl"), mlwe_grid.text())

    def test_the_selected_rank_is_the_smallest_accepted(self):
        pol = hardness.Policy("delta")
        for r in self.grid:
            d, q, k = r["d"], int(r["q"]), r["rank"]
            with self.subTest(d=d, log2q=r["log2q"]):
                self.assertTrue(pol.mlwe_ok(*hardness.mlwe(k, d, q)))
                self.assertFalse(pol.mlwe_ok(*hardness.mlwe(k - 1, d, q)))

    def test_decisions_do_not_depend_on_the_precision(self):
        # At 192 bits every recorded block size and selected rank is the same.
        old = hardness.PREC
        try:
            hardness.PREC = 192
            for r in self.grid[::5]:
                d, q = r["d"], int(r["q"])
                with self.subTest(d=d, log2q=r["log2q"]):
                    self.assertEqual([hardness.usvp_block_size(e["k"] * d, q) for e in r["evals"]],
                                     [e["beta"] for e in r["evals"]])
                    self.assertEqual(hardness.rank(d, q, hardness.Policy("delta"))[0], r["rank"])
        finally:
            hardness.PREC = old


class Estimate(unittest.TestCase):
    def test_success_is_monotone_in_the_block_size(self):
        # The premise of the bisection: from the block size on, every larger one succeeds.
        for d, log2q, k in ((64, 40, 26), (128, 64, 17), (64, 128, 40)):
            n, q = k * d, 1 << log2q
            beta = hardness.usvp_block_size(n, q)
            with self.subTest(d=d, log2q=log2q, k=k):
                self.assertFalse(hardness.usvp_succeeds(beta - 1, n, q))
                self.assertTrue(all(hardness.usvp_succeeds(b, n, q)
                                    for b in range(beta, min(2 * n, beta + 400) + 1, 7)))

    def test_the_best_number_of_samples(self):
        # The maximum over m of the right side lies at floor(D*) or ceil(D*): a scan over every
        # lattice dimension D = m + n + 1 up to 3 D* finds no larger value.
        n, q, b = 26 * 64, 1 << 40, 361
        with mp.workprec(hardness.PREC):
            s = hardness.stddev(1)
            log_delta = mp.log(hardness.delta_0(b))
            c = (n + 1) * mp.log(q) - mp.log(s)

            def g(dim):
                return (2 * b - dim) * log_delta + mp.log(q) - c / dim
            star = mp.sqrt(c / log_delta)
            best = max(g(int(mp.floor(star))), g(int(mp.ceil(star))))
            self.assertGreater(star, n + 1)
            self.assertLessEqual(max(g(dim) for dim in range(n + 1, int(3 * star))), best)
            # At this block size the estimate succeeds.
            self.assertLessEqual(mp.log(s) + mp.log(b) / 2, best)
        self.assertEqual(hardness.usvp_block_size(n, q), b)

    def test_the_candidates_give_the_maximum_over_all_dimensions(self):
        # `usvp_max_g` evaluates g only at floor(D*) and ceil(D*), each at least n+1. A scan of
        # every D in [n+1, 3D*] finds no larger value on random instances: half with D* above
        # n+1, half with q < delta_0(b)^(n-1), which puts D* below n+1, so that the clamp to n+1
        # decides. D* is recomputed here in floats.
        rng = random.Random(3)
        inside = clamped = 0
        for i in range(32):
            nu = rng.randint(1, 4)
            if i % 2:
                n = rng.choice((64, 128, 256, 512))
                b = rng.randrange(50, 2 * n)
                q = rng.randrange(1 << 16, 1 << rng.randrange(17, 64))
            else:
                n = rng.choice((512, 768, 1024))
                b = rng.randrange(50, 120)
                q = rng.randrange(2, int(float(hardness.delta_0(b)) ** (n - 1)))
            c = (n + 1) * math.log(q) - math.log(math.sqrt(nu * (nu + 1) / 3))
            star = math.sqrt(c / math.log(float(hardness.delta_0(b))))
            value, dim = hardness.usvp_max_g(b, n, q, nu)
            scan = hardness.usvp_max_g(b, n, q, nu, range(n + 1, max(n + 1, int(3 * star)) + 1))
            with self.subTest(b=b, n=n, q=q, nu=nu):
                self.assertEqual(value, scan[0])
                self.assertGreaterEqual(dim, n + 1)
            inside += star > n + 2
            clamped += star < n
        self.assertGreaterEqual(inside, 12)
        self.assertGreaterEqual(clamped, 12)

    def test_small_block_sizes_are_reported_as_50(self):
        self.assertEqual(hardness.mlwe(1, 64, 1 << 128), (50, float(hardness.delta_0(50))))
        with self.assertRaises(ValueError):
            hardness.delta_0(49)

    def test_an_undefined_estimate_is_refused(self):
        with self.assertRaises(hardness.HardnessError):
            hardness.usvp_block_size(16, 1 << 40)


class BlockSize(unittest.TestCase):
    def test_root_hermite_factors(self):
        # Block sizes of the root Hermite factors 1.0121, 1.0093 and 1.0024: 50, 100 and 808,
        # as the lattice-estimator's doctests of its block-size search give them.
        self.assertEqual(hardness.beta_from_delta(mp.mpf("1.0121")), 50)
        self.assertEqual(hardness.beta_from_delta(mp.mpf("1.0093")), 100)
        self.assertEqual(hardness.beta_from_delta(mp.mpf("1.0024")), 808)
        self.assertGreater(hardness.delta_0(50), mp.mpf("1.0120"))

    def test_the_ceiling(self):
        with mp.workprec(512):
            self.assertEqual(hardness.beta_from_delta(hardness.DELTA_MAX), hardness.BETA_FLOOR)
            self.assertEqual(hardness.BETA_FLOOR, 346)
            self.assertEqual(hardness.beta_from_delta(mp.mpf("1.0045")), 335)
            self.assertLessEqual(hardness.delta_0(346), hardness.DELTA_MAX)
            self.assertGreater(hardness.delta_0(345), hardness.DELTA_MAX)
            gap = min(abs(hardness.delta_0(b) - hardness.DELTA_MAX) for b in range(300, 400))
            self.assertAlmostEqual(float(gap), 8.34e-7, delta=0.01e-7)

    def test_delta_0_decreases(self):
        prev = hardness.delta_0(50)
        for b in range(51, 3001):
            cur = hardness.delta_0(b)
            self.assertLess(cur, prev, b)
            prev = cur

    def test_core_svp(self):
        self.assertEqual(hardness.core_svp_bits(346), {"classical": 101.0, "quantum": 91.7})
        self.assertEqual(hardness.core_svp_bits(439), {"classical": 128.2, "quantum": 116.3})


class Monotonicity(unittest.TestCase):
    def test_block_size_grows_with_rank_and_rank_with_q(self):
        rng = random.Random(7)
        for _ in range(12):
            d = rng.choice((64, 128))
            log2q = rng.randrange(24, 120)
            q = (1 << log2q) + rng.randrange(1 << (log2q - 2))
            ks = sorted(rng.sample(range(4, 70), 5))
            betas = [hardness.mlwe(k, d, q)[0] for k in ks]
            self.assertEqual(betas, sorted(betas), (d, log2q, ks))
            pol = hardness.Policy("delta")
            r1 = hardness.rank(d, q, pol)[0]
            r2 = hardness.rank(d, q << 8, pol)[0]
            self.assertLessEqual(r1, r2)


class Policies(unittest.TestCase):
    def test_policy_validation(self):
        with self.assertRaises(ValueError):
            hardness.Policy("beta", 300)
        with self.assertRaises(ValueError):
            hardness.Policy("delta", 439)
        with self.assertRaises(ValueError):
            hardness.Policy("estimator")

    def test_the_msis_ceiling_is_strict(self):
        # As in `TboxParams::check`, an MSIS delta of exactly 1.0044 is refused.
        with mp.workprec(512):
            below = hardness.DELTA_MAX - mp.mpf(2) ** -64
        for pol in (hardness.Policy("delta"), hardness.Policy("beta", hardness.BETA_FLOOR)):
            with self.subTest(policy=pol.label()):
                self.assertFalse(pol.msis_ok(hardness.DELTA_MAX))
                self.assertTrue(pol.msis_ok(below))

    def test_beta_policy_ranks(self):
        # Rank at beta >= 346 / >= 439 for d = 64 at 2^40 is 26 / 30.
        self.assertEqual(hardness.rank(64, 1 << 40, hardness.Policy("delta"))[0], 26)
        k, beta, _ = hardness.rank(64, 1 << 40, hardness.Policy("beta", 439))
        self.assertEqual(k, 30)
        self.assertGreaterEqual(beta, 439)
        self.assertLess(hardness.mlwe(k - 1, 64, 1 << 40)[0], 439)


if __name__ == "__main__":
    unittest.main()
