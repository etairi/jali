"""Requirements with several lifted moduli and with variables bounded in l-infinity: how copied
requirements are read, the emulated checks of `statement::compile` at their boundaries and in
Rust's order, and the lower bounds the derivation takes from them. `test_rustcheck.py` checks the
same emulation against Rust's own decisions on statements of each kind."""
import copy
import json
import unittest

import common
from jali_params import cli, derive, jsonio, modulus, request, rustcheck

Q = 1099511627917  # prime, 5 mod 8: toy-d64's modulus


def toy():
    return jsonio.load_params(common.read("src/params/sets/toy-d64.json"))


def statement(requirements, blocks=None, p=156, movable=None):
    s = {"source": "requirements", "requirements": requirements, "statement_modulus": p}
    if blocks is not None:
        s["blocks"] = blocks
    if movable is not None:
        s["movable"] = movable
    return {"schema": request.SCHEMA, "id": "lifting-test", "degree": 64,
            "modulus": {"search": {"factors": 1, "window_bits": 1}},
            "hardness": {"policy": "delta"}, "statement": s}


# Requirements of the shape of the "linf" statement kind of rust_reference.py: s (4 polynomials,
# ||s||^2 <= 64) and y (linf <= 2) over degree 128, a clause modulo 13 and a linear constraint
# modulo 12, exported at proof degree 64. The numbers only need to be consistent here.
LINF = {"m1": 10, "l": 1, "alpha_squared": 64 + 2 * 64 * 4, "n_bin": 0, "l2_rows": [8],
        "l2_bounds_squared": [64], "n_prime": 5, "linf_bound": 70,
        "max_integer_coefficient": 830,
        "lifted_moduli": [{"modulus": 12, "max_integer_coefficient": 830},
                          {"modulus": 13, "max_integer_coefficient": 822}],
        "linf": {"lifting": [{"modulus": 13, "exact": 5, "linear": 273, "quadratic": 0},
                             {"modulus": 12, "exact": 26, "linear": 402, "quadratic": 0}],
                 "extraction_limit": None}}
LINF_BLOCKS = [{"name": "s", "rows": 8, "norm": {"l2_squared": 64}, "placement": "ajtai"},
               {"name": "y", "rows": 2, "norm": {"linf": 2}, "placement": "ajtai"}]


class Reading(unittest.TestCase):
    def test_lifted_moduli_and_linf(self):
        r = statement(LINF, LINF_BLOCKS)
        _, _, _, req, lifting, _ = request.read(json.dumps(r))
        self.assertEqual(req["n_prime"], 5)
        self.assertEqual(lifting["pairs"], [[12, 830], [13, 822]])
        self.assertEqual(lifting["linf"], [[13, 5, 273, 0], [12, 26, 402, 0]])
        self.assertEqual((lifting["has_linf"], lifting["betas"], lifting["lifted"],
                          lifting["extraction_limit"]), (True, [2], True, None))
        # linf_bound = max(1, ceil(830/12), ceil(822/13), beta 2) = 70.
        self.assertEqual(max(1, -(-830 // 12), -(-822 // 13), 2), 70)
        # Numbers from 2^64 on are decimal strings, and either spelling reads the same.
        wide = copy.deepcopy(r)
        wide["statement"]["requirements"]["linf"]["extraction_limit"] = str(2 ** 70)
        self.assertEqual(request.read(json.dumps(wide))[4]["extraction_limit"], 2 ** 70)

    def test_inconsistent_requirements_are_refused(self):
        def refused(change=None, blocks=LINF_BLOCKS, **kw):
            req = copy.deepcopy(LINF)
            if change:
                change(req)
            with self.assertRaises(request.RequestError) as e:
                request.read(json.dumps(statement(req, blocks, **kw)))
            return str(e.exception)

        def set_(key, value):
            return lambda req: req.__setitem__(key, value)
        # linf needs the blocks, which tell the linf rows and bounds.
        self.assertIn("blocks", refused(blocks=None))
        # linf_bound must be what Rust computes; F must be the largest lifted F_j.
        self.assertIn("linf_bound", refused(set_("linf_bound", 71)))
        self.assertIn("largest", refused(set_("max_integer_coefficient", 829)))
        # lifted_moduli ascending and at least 2; unknown fields; entries complete.
        self.assertIn("ascending", refused(lambda r: r["lifted_moduli"].reverse()))
        self.assertIn("modulus", refused(
            lambda r: r["lifted_moduli"][0].__setitem__("modulus", 1)))
        self.assertIn("unknown", refused(
            lambda r: r["linf"]["lifting"][0].__setitem__("cubic", 0)))
        self.assertIn("needs", refused(lambda r: r["linf"].pop("extraction_limit")))
        # The blocks must add up to the aggregate fields.
        self.assertIn("m1", refused(set_("m1", 11)))
        self.assertIn("alpha_squared", refused(set_("alpha_squared", 65)))
        bad = copy.deepcopy(LINF_BLOCKS)
        bad[1]["norm"] = {"linf": 3}
        self.assertIn("alpha_squared", refused(blocks=bad))
        bad[1]["norm"] = "binary"  # a binary block's share is rows * d
        self.assertIn("alpha_squared", refused(blocks=bad))
        bad = copy.deepcopy(LINF_BLOCKS)
        bad[1]["placement"] = "bdlop"
        self.assertIn("m1", refused(blocks=bad))
        # Rust's Norm and Placement forms only; an unbounded block is not in the Ajtai part.
        for norm in ({"linf": 0}, {"linf": 2 ** 64}, {"l2": 64}, "binary ", {"binary": True}):
            bad = copy.deepcopy(LINF_BLOCKS)
            bad[1]["norm"] = norm
            refused(blocks=bad)
        bad = copy.deepcopy(LINF_BLOCKS)
        bad[1]["placement"] = "search"
        self.assertIn("placement", refused(blocks=bad))
        bad = copy.deepcopy(LINF_BLOCKS) + [{"name": "u", "rows": 2, "norm": "unbounded",
                                             "placement": "ajtai"}]
        self.assertIn("BDLOP", refused(blocks=bad))
        # A lifted modulus without lifted rows.
        no_lift = copy.deepcopy(LINF)
        no_lift.update(n_prime=2, linf_bound=2, max_integer_coefficient=0, l=0)
        no_lift["linf"]["lifting"] = []
        request.read(json.dumps(statement({k: v for k, v in no_lift.items()
                                           if k != "lifted_moduli"}, LINF_BLOCKS)))
        with self.assertRaises(request.RequestError):
            request.read(json.dumps(statement(no_lift, LINF_BLOCKS)))

    def test_linf_without_lifted_constraints(self):
        # Only y's range rows: nothing is lifted, so linf_bound is beta and there is no
        # lifting bound, but 2E + 1 < q and the range limit still apply.
        req = {"m1": 10, "l": 0, "alpha_squared": 64 + 2 * 64 * 4, "n_bin": 0, "l2_rows": [8],
               "l2_bounds_squared": [64], "n_prime": 2, "linf_bound": 2,
               "max_integer_coefficient": 0,
               "linf": {"lifting": [], "extraction_limit": 6}}
        _, _, _, _, lifting, _ = request.read(json.dumps(statement(req, LINF_BLOCKS, p=13)))
        self.assertEqual((lifting["lifted"], lifting["pairs"], lifting["extraction_limit"]),
                         (False, [], 6))

    def test_export_modulus_pins_q(self):
        # Requirements exported at Q, as if a constraint over Z_Q were native there: a search
        # or another modulus is refused, Q derives, and check refuses a set at another q.
        r = statement(LINF, LINF_BLOCKS)
        r["statement"]["export_modulus"] = Q
        for mod in ({"search": {"factors": 1}}, {"prime_factors": [Q + 8]}):
            with self.subTest(modulus=mod):
                with self.assertRaises(request.RequestError) as e:
                    request.read(json.dumps(dict(r, modulus=mod)))
                self.assertIn("export_modulus", str(e.exception))
        # The window has no divisor of Q - 1 for this shape (gamma_0 = 2^15), so give one.
        pinned = dict(r, modulus={"prime_factors": [Q], "gamma": 21734})
        text, _ = cli.derive_request(json.dumps(pinned), what_if=False, compare=False)
        self.assertEqual(jsonio.load_params(text)["prime_factors"], [Q])
        code, out, err = common.check(text, pinned)
        self.assertEqual(code, 0, out + err)
        # The set the search derives without the field is at another q: refused against the
        # pinned request, accepted against the request it was derived for.
        searched = statement(LINF, LINF_BLOCKS)
        other, _ = cli.derive_request(json.dumps(searched), what_if=False, compare=False)
        self.assertNotEqual(jsonio.load_params(other)["prime_factors"], [Q])
        code, out, _ = common.check(other, pinned)
        self.assertEqual((code, json.loads(out)["rust_message"]), (1, "export modulus"))
        self.assertEqual(common.check(other, searched)[0], 0)
        # A modulus outside [2, 2^256), and the field on a lin-worst-case statement, are refused.
        for bad in (1, str(2 ** 256)):
            with self.assertRaises(request.RequestError):
                request.read(json.dumps(dict(pinned, statement=dict(pinned["statement"],
                                                                    export_modulus=bad))))
        lin = json.loads((common.TOOL / "sets" / "demo.request.json").read_text())
        lin["statement"]["export_modulus"] = Q
        with self.assertRaises(request.RequestError) as e:
            request.read(json.dumps(lin))
        self.assertIn("unknown statement fields", str(e.exception))


class Emulation(unittest.TestCase):
    def test_per_constraint_moduli_at_the_boundary(self):
        # The multi-modulus implementer's vectors: at q = 1099511627917, linf_bound 13 and
        # log_sigma[3] = 13, psi = ceil(28 * 1.55 * 2^13 / 13) = 27349. Each constraint with its
        # own modulus fits; one F for every modulus does not.
        p = dict(toy(), linf_bound=13, log_sigma=[16, 12, 10, 13])
        self.assertEqual(rustcheck.psi(p), 27349)
        pairs = [[2 ** 20, 10273886], [1546268, 13]]
        self.assertEqual(rustcheck.check_lifting_pairs(p, pairs, 10), 1099510971858)
        self.assertLess(1099510971858, Q)
        with self.assertRaises(rustcheck.RustCheckError) as e:
            rustcheck.check_lifting_pairs(p, [[1546268, 10273886]], 10)
        self.assertEqual(e.exception.rust_message, "modulus lifting bound")
        self.assertEqual(rustcheck.lifting_rhs(p, [[1546268, 10273886]]), 1099531519604)
        # The single-modulus form is the same check.
        self.assertEqual(rustcheck.check_lifting(p, 13, 1, 1546268),
                         rustcheck.lifting_rhs(p, [[1546268, 13]]))

    def test_linf_lifting_uses_the_extraction_bound(self):
        p = dict(toy(), linf_bound=13, log_sigma=[16, 12, 10, 13])
        e = rustcheck.extraction_bound(13)
        self.assertEqual(e, 2 * ((124 << 13) // 5))
        entry = [13, 5, 273, 7]
        self.assertEqual(rustcheck.linf_f(entry, e), 5 + 273 * e + 7 * e * e)
        slack = 13 * 27349
        self.assertEqual(rustcheck.lifting_rhs(p, [[12, 1]], [entry]),
                         max(2 * (1 + 12 * slack), 2 * (5 + 273 * e + 7 * e * e + 13 * slack)))

    def lifting(self, **change):
        _, _, _, req, lifting, _ = request.read(json.dumps(statement(LINF, LINF_BLOCKS)))
        lifting.update(change)
        return req, lifting

    def params_for(self, req):
        p = dict(toy(), m1=req["m1"], l=req["l"], alpha_squared=req["alpha_squared"],
                 l2_rows=req["l2_rows"], l2_bounds_squared=req["l2_bounds_squared"],
                 n_bin=0, n_prime=req["n_prime"], linf_bound=req["linf_bound"])
        return p

    def outcome(self, p, req, statement_modulus, lifting):
        try:
            rustcheck.check_compile(p, req, statement_modulus, lifting=lifting)
            return "ok"
        except rustcheck.RustCheckError as e:
            return e.rust_message

    def test_refusals_in_rusts_order(self):
        req, lifting = self.lifting()
        p = self.params_for(req)
        e = rustcheck.extraction_bound(p["log_sigma"][3])
        self.assertEqual(self.outcome(p, req, 156, lifting), "ok")
        # The inverses first: the statement modulus, then the constraint moduli.
        both = dict(lifting, pairs=lifting["pairs"] + [[3 * Q, 1]])
        self.assertEqual(self.outcome(p, req, 2 * Q, both), "noninvertible statement modulus")
        self.assertEqual(self.outcome(p, req, 156, both), "noninvertible constraint modulus")
        # A modulus equal to q would make the constraint native: the tool refuses to emulate.
        self.assertEqual(self.outcome(p, req, 156, dict(lifting, pairs=[[Q, 1]])), "shape")
        # Then beta against linf_bound, the range limit and 2E + 1 < q, then the lifting.
        self.assertEqual(self.outcome(dict(p, linf_bound=1), req, 156, lifting),
                         "approximate range bound")
        self.assertEqual(self.outcome(p, req, 156, dict(lifting, extraction_limit=e - 1)),
                         "variable range above statement modulus")
        self.assertEqual(self.outcome(p, req, 156, dict(lifting, extraction_limit=e)), "ok")
        t = next(t for t in range(100) if 2 * rustcheck.extraction_bound(t) + 1 >= Q)
        self.assertEqual(self.outcome(dict(p, log_sigma=[16, 12, 10, t]), req, 156, lifting),
                         "approximate extraction bound")
        self.assertEqual(self.outcome(dict(p, log_sigma=[16, 12, 10, t - 1]), req, 156,
                                      lifting), "modulus lifting bound")
        self.assertEqual(self.outcome(dict(p, linf_bound=2), req, 156, lifting),
                         "carry or quotient bound")


class LowerBounds(unittest.TestCase):
    def shape(self):
        return derive.Shape(LINF, 64)

    def test_the_largest_threshold_and_the_extraction_bound(self):
        _, _, _, _, lifting, _ = request.read(json.dumps(statement(LINF, LINF_BLOCKS)))
        shape = self.shape()
        bounds = derive.lower_bounds(shape, lifting)
        widths = {"log_sigma": shape.log_sigma, "linf_bound": shape.linf_bound}
        e = rustcheck.extraction_bound(shape.log_sigma[3])
        self.assertEqual(bounds["modulus lifting bound"],
                         rustcheck.lifting_rhs(widths, lifting["pairs"], lifting["linf"]) + 1)
        self.assertEqual(bounds["approximate extraction bound"], 2 * e + 2)
        # With the linf entries the threshold is set at E, far above the honest one.
        honest = rustcheck.lifting_rhs(widths, lifting["pairs"]) + 1
        self.assertGreater(bounds["modulus lifting bound"], honest)

    def test_a_range_limit_below_e_is_final(self):
        _, _, _, _, lifting, _ = request.read(json.dumps(statement(LINF, LINF_BLOCKS)))
        e = rustcheck.extraction_bound(self.shape().log_sigma[3])
        with self.assertRaises(derive.DeriveError) as err:
            derive.lower_bounds(self.shape(), dict(lifting, extraction_limit=e - 1))
        self.assertTrue(err.exception.final)
        self.assertEqual(err.exception.kind, "variable range above statement modulus")

    def test_a_derived_set_passes_the_emulated_compile(self):
        text, rep = cli.derive_request(json.dumps(statement(LINF, LINF_BLOCKS)), what_if=False,
                                       compare=False)
        params = jsonio.load_params(text)
        _, _, _, req, lifting, _ = request.read(json.dumps(statement(LINF, LINF_BLOCKS)))
        rustcheck.check_compile(params, req, 156, lifting=lifting)
        # The lifting at E binds: a prime of the set's gamma class at or below that bound fails
        # the emulated compile there.
        self.assertEqual(rep["modulus"]["largest_lower_bound"], "modulus lifting bound")
        bound = derive.lower_bounds(derive.Shape(req, 64), lifting)["modulus lifting bound"]
        g = params["gamma"]
        step = g * (2 if modulus.v2(g) == 2 else 4)
        below = params["prime_factors"][0]
        while below >= bound or not modulus.is_prime(below):
            below -= step
        with self.assertRaises(rustcheck.RustCheckError) as e:
            rustcheck.check_compile(dict(params, prime_factors=[below]), req, 156,
                                    lifting=lifting)
        self.assertEqual(e.exception.rust_message, "modulus lifting bound")
        # The report lists the per-constraint bounds and the conditions on E.
        names = [c["name"] for c in rep["checks"]]
        self.assertIn("modulus lifting bound (compile), modulus 12", names)
        self.assertIn("modulus lifting bound at E (compile), linf constraint 1, modulus 12",
                      names)
        self.assertIn("approximate extraction bound 2E+1 < q (compile)", names)
        self.assertEqual(rep["statement"]["lifted_moduli"][0], {"modulus": 12,
                                                                "max_integer_coefficient": 830})
        self.assertEqual(rep["statement"]["linf"]["betas"], [2])


if __name__ == "__main__":
    unittest.main()
