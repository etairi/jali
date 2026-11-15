"""The extraction sets of the straight-line extractable mode: the exact LinHC derivation, the
rank search, and the shipped sets, which `tests/param_sets.rs` loads and the crate checks."""
import json
import subprocess
import sys
import unittest

import common
from jali_params import extraction, hardness, rustcheck

TOOL = common.TOOL / "lnp_params.py"
SHIPPED = {"kyber1024-d64": "delta", "kyber1024-d128": "delta", "demo": "delta",
           "toy-d64": "delta"}


def params(name):
    p, _, _ = rustcheck.from_json(common.read(f"src/params/sets/{name}.json"))
    return p


class Derivation(unittest.TestCase):
    def test_ceil_sqrt(self):
        for x, r in [(0, 0), (1, 1), (2, 2), (4, 2), (5, 3), (10 ** 20, 10 ** 10),
                     (10 ** 20 + 1, 10 ** 10 + 1)]:
            self.assertEqual(extraction.ceil_sqrt(x), r, x)

    def test_the_moduli_are_least_and_prime(self):
        # Every derived modulus is the least prime above 2(pN + B), with p = 4B + 1, and the
        # mask exponent is the least with 961 4^t >= 400 phi^2 T^2.
        for name in SHIPPED:
            p = params(name)
            length, bound, eta, d = extraction.shape(p)
            rank = json.loads(common.read(f"src/params/sets/{name}.extraction.json"))["rank"]
            out = extraction.linhc(d, rank, length, bound, eta)
            self.assertEqual(out["plaintext"], 4 * bound + 1)
            floor = 2 * (out["plaintext"] * out["noise_bound"] + bound)
            self.assertGreater(out["modulus"], floor)
            q = out["modulus"] - 2
            while q > floor and not extraction.sympy.isprime(q):
                q -= 2
            self.assertLessEqual(q, floor, name)
            t = out["log_sigma"]
            target = extraction.PHI ** 2 * out["shift_bound_squared"] * 400
            self.assertGreaterEqual(961 << (2 * t), target)
            self.assertLess(961 << (2 * (t - 1)), target)

    def test_the_refusals(self):
        p = params("toy-d64")
        length, _, _, _ = extraction.shape(p)
        for rank in (0, length + 1):
            with self.assertRaises(extraction.ExtractionError):
                extraction.modulus_for(p, rank)


class Search(unittest.TestCase):
    def test_the_rank_is_the_least_fixed_point(self):
        policy = hardness.Policy("delta")
        for name in SHIPPED:
            p = params(name)
            n, q, beta, delta = extraction.search(p, policy)
            self.assertEqual(q, extraction.modulus_for(p, n))
            self.assertTrue(policy.mlwe_ok(beta, delta))
            # The rank below fails the policy at the modulus it gives.
            below = extraction.modulus_for(p, n - 1)
            self.assertFalse(policy.mlwe_ok(*hardness.mlwe(n - 1, p["degree"], below)), name)

    def test_the_shipped_sets_are_the_tools_output(self):
        res = subprocess.run([sys.executable, str(TOOL), "extraction", "--shipped", "--check"],
                             capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)

    def test_one_set_from_the_command_line(self):
        res = subprocess.run([sys.executable, str(TOOL), "extraction",
                              str(common.JALI / "src/params/sets/kyber1024-d64.json")],
                             capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stdout,
                         common.read("src/params/sets/kyber1024-d64.extraction.json"))


if __name__ == "__main__":
    unittest.main()
