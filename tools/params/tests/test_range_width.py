"""The per-slot width rule for sigma_4: requirements that carry `approx_alpha_squared`
(`Statement::requirements`, a per-slot bound on the squared norm of the approximate-range vector)
get `log_sigma[3]` = max(width of the per-slot bound, guard), where the guard is the smallest
width whose rejection constant for n'd linf_bound^2 is 2. Requirements without the field keep
the default rule byte for byte. The prover's constant keeps n'd linf_bound^2, so the rule may
only narrow sigma_4 down to the guard; `rust_reference.py`'s kinds check it against Rust."""
import copy
import json
import math
import unittest

import mpmath as mp

import common
from jali_params import cli, derive, partition, request, rustcheck

from test_lifting import LINF, LINF_BLOCKS, statement


def guard(n_prime, d, linf):
    """The smallest t with ceil(exp(337 n'd linf^2 / (2 sigma_t^2)) + 1e-10) = 2, sigma_t =
    1.55 2^t, recomputed here in f64."""
    a = float(n_prime * d * linf * linf)
    for t in range(101):
        s = 1.55 * 2.0 ** t
        x = 337.0 * a / (2.0 * s * s)
        if x < 700 and math.ceil(math.exp(x) + 1e-10) == 2:
            return t
    raise AssertionError("no guard")


def slot(a):
    """rounded(5 sqrt(337) sqrt(a)), recomputed with mpmath."""
    with mp.workprec(300):
        v = 5 * mp.sqrt(337) * mp.sqrt(a)
        c = mp.mpf("1.55")
        t = max(0, int(mp.floor(mp.log(v / c, 2))))
        return t if abs(c * 2 ** t - v) <= abs(c * 2 ** (t + 1) - v) else t + 1


def shape(n_prime, d, linf, approx=None, **extra):
    req = {"m1": 10, "l": 1, "alpha_squared": 576, "n_bin": 0, "l2_rows": [8],
           "l2_bounds_squared": [64], "n_prime": n_prime, "linf_bound": linf}
    if approx is not None:
        req["approx_alpha_squared"] = approx
    req.update(extra)
    return derive.Shape(req, d)


class Reading(unittest.TestCase):
    def request(self, approx):
        r = statement(copy.deepcopy(LINF), LINF_BLOCKS)
        if approx is not None:
            r["statement"]["requirements"]["approx_alpha_squared"] = approx
        return r

    def test_the_field_is_read_after_the_requirements_it_depends_on(self):
        # LINF: n' = 5, d = 64, linf_bound = 70, so n'd linf^2 = 1,568,000.
        default = 5 * 64 * 70 * 70
        for spelling in (200000, "200000"):
            _, _, _, req, _, _ = request.read(json.dumps(self.request(spelling)))
            self.assertEqual(req["approx_alpha_squared"], 200000)
        _, _, _, req, _, _ = request.read(json.dumps(self.request(default - 1)))
        self.assertEqual(req["approx_alpha_squared"], default - 1)
        # Absent: the key stays absent (the default rule).
        _, _, _, req, _, _ = request.read(json.dumps(self.request(None)))
        self.assertNotIn("approx_alpha_squared", req)

    def test_refusals(self):
        default = 5 * 64 * 70 * 70
        for bad in (default, default + 1, 1 << 128, str(1 << 128), None, -1, True, 1.5, "01"):
            r = self.request(0)
            r["statement"]["requirements"]["approx_alpha_squared"] = bad
            with self.subTest(bad=bad), self.assertRaises(request.RequestError):
                request.read(json.dumps(r))
        # Without a range block Rust never writes it.
        r = statement({"m1": 10, "l": 0, "alpha_squared": 576, "n_bin": 0, "l2_rows": [8],
                       "l2_bounds_squared": [64], "n_prime": 0, "linf_bound": 0,
                       "approx_alpha_squared": 0}, p=156)
        with self.assertRaises(request.RequestError) as e:
            request.read(json.dumps(r))
        self.assertIn("range block", str(e.exception))


class Rule(unittest.TestCase):
    def test_without_the_field_the_default_rule(self):
        s = shape(5, 64, 70)
        self.assertIsNone(s.range_width)
        with mp.workprec(300):
            self.assertEqual(s.log_sigma[3], derive.rounded(5 * mp.sqrt(337)
                                                            * mp.sqrt(5 * 64) * 70))

    def test_the_larger_of_the_per_slot_width_and_the_guard(self):
        # Shapes of the Rust tests (tests/tag_preimage.rs, computed there by the Rust port): the
        # binary-pairs form at degree 128 keeps 33, the l_inf form narrows 32 to 31 at both
        # degrees; and synthetic ones where the guard binds.
        cases = [
            (9, 128, 3574883, 13086503388246863, (33, 33, 31)),
            (41, 128, 1270853, 1653828964216337, (31, 31, 30)),
            (81, 64, 1270853, 1653828964216337, (31, 31, 30)),
        ]
        for n, d, linf, a, (t, t_slot, t_guard) in cases:
            with self.subTest(n=n, d=d):
                s = shape(n, d, linf, a)
                w = s.range_width
                self.assertEqual((w["log_sigma3"], w["log_sigma3_per_slot"],
                                  w["log_sigma3_guard"]), (t, t_slot, t_guard))
                self.assertEqual((t_slot, t_guard), (slot(a), guard(n, d, linf)))
                self.assertEqual(s.log_sigma[3], t)
                self.assertEqual(w["binding"], "per-slot")
        # One lifted clause at degree 64: one slot of 64, so the guard binds.
        for linf in (1, 13, 3574883):
            s = shape(1, 64, linf, linf * linf)
            w = s.range_width
            self.assertEqual(w["binding"], "guard", linf)
            self.assertEqual(s.log_sigma[3], guard(1, 64, linf))
            self.assertGreater(guard(1, 64, linf), slot(linf * linf))

    def test_the_rejection_constant_stays_2_and_the_guard_is_minimal(self):
        for n, d, linf, a in [(5, 64, 70, 200000), (1, 64, 13, 169), (2, 128, 5, 25),
                              (41, 128, 1270853, 1653828964216337)]:
            s = shape(n, d, linf, a)
            default = n * d * linf * linf
            t = s.log_sigma[3]
            self.assertEqual(rustcheck.range_rejection_constant(t, default), 2)
            g = s.range_width["log_sigma3_guard"]
            self.assertEqual(rustcheck.range_rejection_constant(g, default), 2)
            try:
                narrower = rustcheck.range_rejection_constant(g - 1, default)
            except rustcheck.RustCheckError:
                narrower = None
            self.assertNotEqual(narrower, 2)
            # Never wider than the default rule.
            self.assertLessEqual(t, s.range_width["log_sigma3_default"])

    def test_invalid_values_are_refused(self):
        for bad in (-1, True, 5 * 64 * 70 * 70, 1 << 128):
            with self.subTest(bad=bad), self.assertRaises(derive.DeriveError):
                shape(5, 64, 70, bad)


class Derive(unittest.TestCase):
    def request(self, approx, modulus=None):
        r = statement(copy.deepcopy(LINF), LINF_BLOCKS)
        if modulus is not None:
            r["modulus"] = modulus
        if approx is not None:
            r["statement"]["requirements"]["approx_alpha_squared"] = approx
        return r

    def test_a_set_differs_only_in_its_width_and_the_report_names_the_rule(self):
        # The default rule's search sets q and gamma; the wider sigma_4 has the larger lifting
        # bound, so both requests derive at that modulus.
        searched, _ = cli.derive_request(json.dumps(self.request(None)), what_if=False,
                                         compare=False)
        s = json.loads(searched)
        fixed = {"prime_factors": s["prime_factors"], "gamma": s["gamma"]}
        base, base_rep = cli.derive_request(json.dumps(self.request(None, fixed)),
                                            what_if=False, compare=False)
        self.assertNotIn("range_width", base_rep)
        text, rep = cli.derive_request(json.dumps(self.request(200000, fixed)),
                                       what_if=False, compare=False)
        a, b = json.loads(base), json.loads(text)
        # The estimator string holds the request's hash, which differs.
        for x in (a, b):
            x.pop("estimator")
        self.assertEqual(b["log_sigma"][3], 15)
        self.assertEqual(a["log_sigma"][3], 16)
        a["log_sigma"][3] = 15
        self.assertEqual(a, b)
        w = rep["range_width"]
        self.assertEqual((w["log_sigma3"], w["log_sigma3_default"], w["binding"]),
                         (15, 16, "per-slot"))
        self.assertEqual(rep["repetitions"]["M4"], 2)
        self.assertEqual(rep["statement"]["requirements"]["approx_alpha_squared"], 200000)
        # The set passes the emulated checks, and check --request accepts it, with a smaller
        # size estimate (one step of sigma_4 in the range response).
        code, out, _ = common.check(text, self.request(200000, fixed))
        self.assertEqual(code, 0, out)
        code, base_out, _ = common.check(base, self.request(None, fixed))
        self.assertEqual(code, 0, base_out)
        self.assertLess(json.loads(out)["estimated_proof_bytes"],
                        json.loads(base_out)["estimated_proof_bytes"])

    def test_the_per_slot_set_is_never_larger_than_the_default_rules(self):
        # The narrower width lowers the lifting bound but not the MSIS fixed point, so the
        # search also starts where the default width would, and meets every default candidate
        # again with a smaller estimate. Without those starts the reference kind "linf" got
        # q = 2221208669 with gamma 1022 and 18,199 estimated bytes (17,520 measured) against
        # the default rule's 15,494 (14,993 measured): the window above its lower bound, 2^27.8,
        # collapsed onto the MSIS fixed point, 2^31.05.
        ref = json.loads((common.DATA / "rust_reference.json").read_text())["statements"]
        for kind, (narrower, same_q) in {"two-moduli": (1, True), "linf": (2, True),
                                         "packed": (0, True)}.items():
            req = ref[kind]["request"]
            text, rep = cli.derive_request(json.dumps(req), what_if=False, compare=False)
            base = copy.deepcopy(req)
            del base["statement"]["requirements"]["approx_alpha_squared"]
            btext, brep = cli.derive_request(json.dumps(base), what_if=False, compare=False)
            a, b = json.loads(text), json.loads(btext)
            with self.subTest(kind=kind):
                self.assertEqual(b["log_sigma"][3] - a["log_sigma"][3], narrower)
                self.assertEqual(a["prime_factors"] == b["prime_factors"], same_q)
                self.assertEqual(rep["derived"]["estimated_proof_bytes"] + 32 * narrower,
                                 brep["derived"]["estimated_proof_bytes"])
                self.assertIn("default_width_window_from", rep["modulus"])
                self.assertNotIn("default_width_window_from", brep["modulus"])

    def test_a_placement_keeps_the_field(self):
        req = dict(LINF, approx_alpha_squared=200000)
        placed = partition.placed_requirements(req, [(8, 64)], [0])
        self.assertEqual(placed["approx_alpha_squared"], 200000)
        self.assertEqual(derive.Shape(placed, 64).log_sigma[3],
                         derive.Shape(req, 64).log_sigma[3])


if __name__ == "__main__":
    unittest.main()
