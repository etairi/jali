"""Prime search: q = 5 mod 8, primality, an even gamma | q-1 in the window, minimality."""
import random
import unittest

import sympy

import common
from jali_params import modulus

Q_MINS = [2 ** 20, 2 ** 32 - 3, 2 ** 32 + 3, 2 ** 63 - 1, 2 ** 64, 2 ** 99, 2 ** 100 - 1, 2 ** 100,
          2 ** 128, 2 ** 200, 2 ** 256]
# gamma_0 values: the rounding bound needs gamma > omega*d/2 = 256 (d = 64) or 128 (d = 128).
GAMMA0S = [2 ** 9, 2 ** 16, 2 ** 19]


class Construction(unittest.TestCase):
    def check(self, q_min, gamma0):
        gamma = next(modulus.admissible_gammas(gamma0))
        q, tries = modulus.construct_prime(q_min, gamma)
        self.assertTrue(sympy.isprime(q))
        self.assertTrue(common.miller_rabin(q))
        self.assertEqual(q % 8, 5)
        self.assertEqual(gamma % 2, 0)
        self.assertEqual((q - 1) % gamma, 0)
        self.assertIn(modulus.v2(gamma), (1, 2))
        self.assertTrue(5 * gamma > 4 * gamma0 and gamma <= gamma0)
        self.assertGreaterEqual(q, q_min)
        self.assertEqual(modulus.construct_prime(q_min, gamma), (q, tries))
        return q, gamma

    def test_properties(self):
        for q_min in Q_MINS:
            for gamma0 in GAMMA0S:
                with self.subTest(q_min=q_min, gamma0=gamma0):
                    self.check(q_min, gamma0)

    def test_minimal_in_the_gamma_class(self):
        # Brute force below 2^24: no prime in [q_min, q) is 5 mod 8 with gamma | q-1.
        rng = random.Random(11)
        for _ in range(40):
            q_min = rng.randrange(2 ** 12, 2 ** 24)
            gamma0 = 1 << rng.randrange(5, 12)
            q, gamma = self.check(q_min, gamma0)
            for c in range(q_min, q):
                self.assertFalse(sympy.isprime(c) and c % 8 == 5 and (c - 1) % gamma == 0, c)

    def test_gamma_choice(self):
        self.assertEqual(next(modulus.admissible_gammas(16)), 14)
        self.assertEqual(next(modulus.admissible_gammas(32)), 30)
        self.assertEqual(next(modulus.admissible_gammas(12)), 12)
        # 8: the window (6.4, 8] holds only 8, whose 2-adic valuation 3 cannot divide q-1.
        self.assertEqual(list(modulus.admissible_gammas(8)), [])
        for g0 in range(10, 400):
            for g in modulus.admissible_gammas(g0):
                self.assertTrue(modulus.in_window(g, g0) and modulus.v2(g) in (1, 2))

    def test_no_admissible_gamma_at_the_rounding_bound(self):
        # The rounding check needs 2^(D-1) omega d < gamma, which fails at D = 0 when
        # gamma <= omega d / 2 (256 at d = 64): no gamma is admissible at or below it.
        self.assertIsNone(modulus.top_gamma(256, 256))
        self.assertIsNone(modulus.top_gamma(300, 300))
        self.assertEqual(modulus.top_gamma(300, 299), 300)
        self.assertEqual(modulus.top_gamma(512, 256), 510)
        self.assertEqual(modulus.top_gamma(1 << 64, 256), (1 << 64) - 2)

    def test_refusals(self):
        with self.assertRaises(modulus.ModulusError):
            modulus.construct_prime(2 ** 40, 8)  # v2 = 3
        with self.assertRaises(modulus.ModulusError):
            modulus.construct_prime(2 ** 40, 7)


class Divisors(unittest.TestCase):
    def test_window_scan_equals_divisor_enumeration(self):
        rng = random.Random(5)
        for _ in range(300):
            n = rng.randrange(2, 2 ** 48) * rng.choice((1, 4, 12, 60, 720))
            g0 = rng.randrange(2, 1 << rng.randrange(3, 20))
            divs = [int(x) for x in sympy.divisors(n) if modulus.in_window(int(x), g0)]
            self.assertEqual(modulus.divisor_in_window(n, g0, "smallest"),
                             min(divs) if divs else None)
            self.assertEqual(modulus.divisor_in_window(n, g0, "largest"),
                             max(divs) if divs else None)

    def test_factoring_path_above_the_scan_limit(self):
        g0 = 4 * modulus.SCAN_LIMIT
        n = (g0 - 2) * 7 * 1000003  # g0 - 2 = 2 (2^29 - 1) lies in the window
        divs = [int(x) for x in sympy.divisors(n) if modulus.in_window(int(x), g0)]
        self.assertIn(g0 - 2, divs)
        self.assertEqual(modulus.divisor_in_window(n, g0, "largest"), max(divs))
        self.assertEqual(modulus.divisor_in_window(n, g0, "smallest"), min(divs))
        self.assertIsNone(modulus.divisor_in_window(1000003 * 1000033, g0, "smallest"))

    def test_the_divisor_of_toy_d64(self):
        # q = 2^40 + 141, the smallest prime from 2^40 on that is 5 mod 8, and gamma_0 = 2^16:
        # the largest even divisor of q - 1 in the window is toy-d64's gamma, 65202.
        q = 1099511627917
        self.assertEqual(q, next(p for p in range(1 << 40, (1 << 40) + 1000)
                                 if p % 8 == 5 and modulus.is_prime(p)))
        self.assertEqual(modulus.divisor_in_window(q - 1, 1 << 16, "largest"), 65202)


class RustPrimality(unittest.TestCase):
    def test_rust_miller_rabin_equals_sympy(self):
        rng = random.Random(3)
        cases = [2, 3, 5, 25326001, 3215031751, 2152302898747, 3474749660383, 341550071728321,
                 3825123056546413051, 318665857834031151167461 % (1 << 64), 2 ** 61 - 1,
                 2 ** 64 - 59, 2 ** 64 - 1]
        cases += [rng.randrange(1 << 64) for _ in range(2000)]
        cases += [rng.randrange(1 << 62) | 1 for _ in range(2000)]
        for n in cases:
            self.assertEqual(modulus.rust_is_prime(n), sympy.isprime(n), n)
        with self.assertRaises(modulus.ModulusError):
            modulus.rust_is_prime(1 << 64)


if __name__ == "__main__":
    unittest.main()
