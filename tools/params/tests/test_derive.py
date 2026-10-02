"""Native derivation: parity with the positional form, the modulus search at its failure
boundary, the policies, and refusals of infeasible requests."""
import json
import math
import unittest

import common
from jali_params import cli, derive, hardness, jsonio, modulus, request, rustcheck

DELTA = hardness.Policy("delta")
BETA_439 = hardness.Policy("beta", 439)


def previous_in_class(q, gamma):
    step = gamma * (2 if modulus.v2(gamma) == 2 else 4)
    p = q - step
    while not modulus.is_prime(p):
        p -= step
    return p


class PositionalParity(unittest.TestCase):
    """`solve` with the external report reproduces what the positional form writes."""

    def derive(self, req_path, rep_path):
        req, rep = common.load(req_path), common.load(rep_path)
        shape = derive.Shape(req, req["degree"])
        params, _ = derive.solve(shape, req["prime_factors"],
                                 external={"rank": rep["rank"], "delta": rep["delta"]})
        params.update(id=req["id"], estimator=rep["provenance"])
        return params

    def test_fixtures_byte_identical(self):
        for req, rep, out in common.POSITIONAL:
            with self.subTest(fixture=out):
                self.assertEqual(jsonio.dump_params(self.derive(req, rep)), common.read(out))

    def test_no_divisor_refusal(self):
        with self.assertRaises(derive.DeriveError) as e:
            self.derive("tests/fixtures/params/tag-preimage-crt-d128-no-divisor.request.json",
                        "tests/fixtures/params/tag-preimage-crt-d128-no-divisor.report.json")
        self.assertEqual(str(e.exception),
                         "no suitable divisor; choose another prime and rerun the estimator")


TOY_NO_RANGE = dict(m1=10, l=2, alpha_squared=5760, n_bin=2, l2_rows=[2, 1],
                      l2_bounds_squared=[128, 64], n_prime=0, linf_bound=0)
BINARY_HEAVY = dict(m1=400, l=0, alpha_squared=400 * 64, n_bin=400, l2_rows=[],
                    l2_bounds_squared=[], n_prime=0, linf_bound=0)
BIG_NORM = dict(m1=20, l=0, alpha_squared=2 ** 30, n_bin=0, l2_rows=[20],
                l2_bounds_squared=[2 ** 30], n_prime=0, linf_bound=0)


class Boundary(unittest.TestCase):
    """Three shapes, each dominated by one bound: the chosen q passes the emulated Rust check
    and the previous prime of its gamma class fails exactly that bound."""

    def run_case(self, req, binding, rust_message, policy=DELTA):
        shape = derive.Shape(req, 64)
        params, info, derived, checks, rec = derive.search(shape, policy)
        q, gamma = params["prime_factors"][0], params["gamma"]
        self.assertEqual(rec["binding_lower_bound"], binding)
        self.assertEqual(checks.tight(), [])
        self.assertTrue(modulus.is_prime(q) and q % 8 == 5 and (q - 1) % gamma == 0)
        prev = previous_in_class(q, gamma)
        with self.assertRaises(rustcheck.RustCheckError) as e:
            rustcheck.check_params(dict(params, prime_factors=[prev]))
        self.assertEqual(e.exception.rust_message, rust_message)
        return params, info, derived

    def test_msis_bound(self):
        params, info, _ = self.run_case(TOY_NO_RANGE, "MSIS bound < q", "MSIS estimate")
        # The MSIS bound, not the ARP bounds near 2^28.2, sizes q.
        self.assertLess(math.log2(params["prime_factors"][0]) - math.log2(info["msis_bound"]),
                        0.001)
        # Re-deriving at the previous prime of the class fails too.
        prev = previous_in_class(params["prime_factors"][0], params["gamma"])
        with self.assertRaises(derive.DeriveError):
            derive.solve(derive.Shape(TOY_NO_RANGE, 64), [prev], DELTA,
                         gamma=params["gamma"])

    def test_arp_modulus_bound(self):
        self.run_case(BINARY_HEAVY, "ARP modulus bound", "ARP modulus bound")

    def test_exact_norm_lifting_bound(self):
        self.run_case(BIG_NORM, "exact norm lifting bound 0", "exact norm lifting bound",
                      BETA_439)

    def test_guard_band_is_respected(self):
        # Every f64 inequality of the chosen set holds with a relative margin of 1e-9 or more.
        shape = derive.Shape(TOY_NO_RANGE, 64)
        params, _, _, checks, _ = derive.search(shape, DELTA)
        for row in checks.rows:
            if row.get("float"):
                self.assertGreaterEqual(row["relative_margin"], rustcheck.GUARD, row["name"])

    def test_a_tight_candidate_is_skipped(self):
        # The MSIS bound sizes q here. The search's candidate for gamma = 510, the smallest
        # prime of its class where 510 keeps MSIS hard, holds "MSIS bound < q" with a relative
        # margin near 1.5e-6, the MSIS delta with about 5e-5. With a guard band of 1e-5 that
        # candidate counts as failing, and the search takes the next prime of the class, whose
        # f64 margins clear the band. (With the hint allowance of the size estimate, the
        # search chooses gamma = 1022, whose margins clear the band.)
        shape = derive.Shape(TOY_NO_RANGE, 64)
        _, _, _, checks, rec = derive.search(shape, DELTA)
        self.assertEqual(checks.tight(1e-5), [])

        def steps(record):
            return [s for t in record["trace"] for s in t["steps"] if s.get("gamma") == 510]
        tight = steps(rec)[0]
        self.assertEqual(tight["event"], "accepted")
        params, _ = derive.solve(shape, [tight["q"]], DELTA, gamma=510)
        params["id"] = params["estimator"] = "probe"
        self.assertEqual(rustcheck.check_params(params)[1].tight(1e-5), ["MSIS bound < q"])
        old = rustcheck.GUARD
        try:
            rustcheck.GUARD = 1e-5
            _, _, _, checks, rec = derive.search(shape, DELTA)
        finally:
            rustcheck.GUARD = old
        self.assertEqual(checks.tight(1e-5), [])
        skipped = [s for s in steps(rec) if {"q": tight["q"], "why": "within guard band: "
                                             "['MSIS bound < q']"} in s.get("rejected", [])]
        self.assertTrue(skipped)
        for s in skipped:
            self.assertEqual(s["event"], "accepted")
            self.assertGreater(s["q"], tight["q"])
            params, _ = derive.solve(shape, [s["q"]], DELTA, gamma=510)
            params["id"] = params["estimator"] = "probe"
            self.assertEqual(rustcheck.check_params(params)[1].tight(1e-5), [])


class NewSets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sets = {n: jsonio.load_params((common.TOOL / "sets" / f"{n}.json").read_text())
                    for n in common.NEW_SETS}
        cls.reports = {n: json.loads((common.TOOL / "sets" / f"{n}.report.json").read_text())
                       for n in common.NEW_SETS}

    def test_policy_delta(self):
        for name, p in self.sets.items():
            rep = self.reports[name]
            with self.subTest(set=name):
                q = p["prime_factors"][0]
                beta, delta = hardness.mlwe(p["mlwe_rank"], p["degree"], q)
                self.assertEqual(delta, p["mlwe_delta"])
                self.assertGreaterEqual(beta, 346)
                self.assertLess(hardness.mlwe(p["mlwe_rank"] - 1, p["degree"], q)[0], 346)
                self.assertLess(float(rep["hardness"]["msis"]["delta"]), 1.0044)
                self.assertEqual(rep["policy"]["name"], "delta")

    def test_policy_comparison_is_beta_439(self):
        for name, rep in self.reports.items():
            c = rep["policy_comparison"]
            with self.subTest(set=name):
                self.assertEqual(c["policy"]["beta_min"], 439)
                self.assertGreaterEqual(c["mlwe_beta"], 439)
                self.assertGreaterEqual(c["msis_beta"], 439)
                self.assertGreater(c["estimated_proof_bytes"],
                                   rep["summary"]["estimated_proof_bytes"])

    def test_json_numbers_and_capacity(self):
        for name, p in self.sets.items():
            text = (common.TOOL / "sets" / f"{name}.json").read_text()
            self.assertEqual(jsonio.dump_params(p), text)
            self.assertLess(p["prime_factors"][0], 1 << 64)
            self.assertIn(p["prime_factors"][0], json.loads(text)["prime_factors"])

    def test_deterministic(self):
        text = (common.TOOL / "sets" / "kyber1024-d128.request.json").read_text()
        a = cli.derive_request(text, what_if=False, compare=False)
        b = cli.derive_request(text, what_if=False, compare=False)
        self.assertEqual(a[0], b[0])
        self.assertEqual(json.dumps(a[1]), json.dumps(b[1]))

    def test_recorded_cross_checks_are_current(self):
        # sets/<name>.xcheck*.json come from the Sage cross-check; they must describe the
        # current set.
        import hashlib
        paths = sorted((common.TOOL / "sets").glob("*.xcheck*.json"))
        self.assertEqual(len(paths), 6)
        for path in paths:
            name = path.name.split(".xcheck")[0]
            x = json.loads(path.read_text())
            text = (common.TOOL / "sets" / f"{name}.json").read_text()
            self.assertEqual(x["params_sha256"], hashlib.sha256(text.encode()).hexdigest(), name)
            self.assertTrue(x["lattice_estimator"]["pinned"], name)

    def test_independent_of_global_precision(self):
        # lnp_params.py sets mpmath's global precision to 300 bits; the package must not care.
        import mpmath as mp
        text = (common.TOOL / "sets" / "demo.request.json").read_text()
        old = mp.mp.prec
        try:
            mp.mp.prec = 300
            a = cli.derive_request(text, what_if=False, compare=False)
            mp.mp.prec = 53
            b = cli.derive_request(text, what_if=False, compare=False)
        finally:
            mp.mp.prec = old
        self.assertEqual(a[0], b[0])
        self.assertEqual(json.dumps(a[1]), json.dumps(b[1]))

    def test_window_keeps_the_smallest_proof(self):
        for name, rep in self.reports.items():
            sizes = [t["estimated_proof_bytes"] for t in rep["modulus"]["search"]
                     if "estimated_proof_bytes" in t]
            self.assertEqual(min(sizes), rep["summary"]["estimated_proof_bytes"], name)


class RequestPaths(unittest.TestCase):
    def test_requirements_source_matches_the_lin_template(self):
        # Requirements Rust computed for the Kyber template give the same set as the template.
        ref = json.loads((common.DATA / "rust_reference.json").read_text())["sets"]
        for name in ("kyber1024-d64", "demo"):
            r = json.loads((common.TOOL / "sets" / f"{name}.request.json").read_text())
            r["statement"] = {"source": "requirements",
                              "statement_modulus": ref[name]["statement_modulus"],
                              "requirements": ref[name]["requirements"]}
            got = jsonio.load_params(cli.derive_request(json.dumps(r), what_if=False,
                                                        compare=False)[0])
            want = jsonio.load_params((common.TOOL / "sets" / f"{name}.json").read_text())
            got.pop("estimator"), want.pop("estimator")
            self.assertEqual(got, want, name)

    def test_external_policy_with_fixed_modulus_reproduces_a_fixture(self):
        old = common.load("tools/params/possession.request.json")
        rep = common.load("tools/params/toy-d64.report.json")
        req = {k: old[k] for k in ("m1", "l", "alpha_squared", "n_bin", "l2_rows",
                                   "l2_bounds_squared", "n_prime", "linf_bound")}
        # Statement::requirements of examples/possession.rs, as the example prints it
        # (`cargo run --release --features serde --example possession`).
        req["max_integer_coefficient"] = 162
        r = {"schema": request.SCHEMA, "id": old["id"], "degree": old["degree"],
             "modulus": {"prime_factors": old["prime_factors"]},
             "hardness": {"policy": "external", "mlwe_rank": rep["rank"],
                          "mlwe_delta": rep["delta"], "provenance": rep["provenance"]},
             "statement": {"source": "requirements", "requirements": req,
                           "statement_modulus": 13}}
        text, report = cli.derive_request(json.dumps(r))
        self.assertEqual(text, common.read("examples/fixtures/possession.json"))
        self.assertEqual(report["policy"]["name"], "external")

    def test_bad_requests(self):
        base = json.loads((common.TOOL / "sets" / "demo.request.json").read_text())
        for change in ({"schema": "v1"}, {"degree": 256}, {"mode": "compat"},
                       {"extra": 1}, {"hardness": {"policy": "external"}},
                       {"statement": {"source": "instance"}},
                       {"modulus": {"prime_factors": ["01"]}}):
            with self.subTest(change=change):
                with self.assertRaises(request.RequestError):
                    cli.derive_request(json.dumps(dict(base, **change)))

    def requirements_request(self, drop=()):
        ref = json.loads((common.DATA / "rust_reference.json").read_text())["sets"]
        r = json.loads((common.TOOL / "sets" / "kyber1024-d64.request.json").read_text())
        req = {k: v for k, v in ref["kyber1024-d64"]["requirements"].items() if k not in drop}
        r["statement"] = {"source": "requirements", "requirements": req,
                          "statement_modulus": 3329}
        return r

    def test_range_block_needs_max_integer_coefficient(self):
        # Without F the search's lower bound on q is too small: the tool picked
        # q = 2423955247093, below the true lifting bound 2423973316489, and compile refused it.
        r = self.requirements_request(drop=("max_integer_coefficient",))
        self.assertGreater(r["statement"]["requirements"]["n_prime"], 0)
        with self.assertRaises(request.RequestError) as e:
            cli.derive_request(json.dumps(r), what_if=False, compare=False)
        self.assertIn("max_integer_coefficient", str(e.exception))
        # Without a range block the field is not needed (Rust gives 0 there).
        no_range = dict(TOY_NO_RANGE)
        r["statement"] = {"source": "requirements", "requirements": no_range}
        _, _, _, req, lifting, _ = request.read(json.dumps(r))
        self.assertEqual((req["max_integer_coefficient"], lifting), (0, None))

    def test_malformed_requests_are_refused_cleanly(self):
        # Typos, two norm kinds, duplicate keys, NaN, missing keys and wrong types: each is a
        # RequestError, not an ignored field, a set, a KeyError or a TypeError.
        base = json.loads((common.TOOL / "sets" / "kyber1024-d64.request.json").read_text())
        blk = base["statement"]["blocks"][0]

        def changed(path, value=None, delete=False):
            r = json.loads(json.dumps(base))
            obj = r
            for key in path[:-1]:
                obj = obj[key]
            if delete:
                del obj[path[-1]]
            else:
                obj[path[-1]] = value
            return json.dumps(r, indent=2)
        text = json.dumps(base, indent=2)
        req = self.requirements_request()
        fixed = {"prime_factors": [2423991946973]}  # kyber1024-d64's q

        def with_requirements(**change):
            r = json.loads(json.dumps(req))
            r["statement"]["requirements"].update(change)
            return json.dumps(r)
        cases = {
            "statement typo": changed(["statement", "public_lnf"], 1),
            "block key typo": json.dumps(dict(base, statement=dict(
                base["statement"], blocks=[{"name": "w", "length": 8, "nrom": blk["norm"]}]))),
            "missing rows": changed(["statement", "rows"], delete=True),
            "missing block norm": changed(["statement", "blocks", 0, "norm"], delete=True),
            "two norm kinds": changed(["statement", "blocks", 0, "norm"],
                                      {"l2_squared": 2950, "binary": True}),
            "binary false": changed(["statement", "blocks", 0, "norm"], {"binary": False}),
            "norm typo": changed(["statement", "blocks", 0, "norm"], {"l2_sqared": 2950}),
            "empty blocks": changed(["statement", "blocks"], []),
            "duplicate degree": text.replace('"degree": 64,', '"degree": 64,\n  "degree": 128,'),
            "NaN": json.dumps(dict(base, modulus=fixed, hardness={
                "policy": "external", "mlwe_rank": 26, "mlwe_delta": float("nan"),
                "provenance": "x"})),
            "external rank string": json.dumps(dict(base, modulus=fixed, hardness={
                "policy": "external", "mlwe_rank": "x", "mlwe_delta": 1.004,
                "provenance": "x"})),
            "missing modulus": changed(["modulus"], delete=True),
            "missing hardness": changed(["hardness"], delete=True),
            "missing statement": changed(["statement"], delete=True),
            "statement not an object": changed(["statement"], "lin"),
            "degree float": changed(["degree"], 64.0),
            "not an object": "[]",
            "not JSON": "{",
            "requirements typo": with_requirements(n_primes=1),
            "requirements missing m1": json.dumps(self.requirements_request(drop=("m1",))),
            "l2_rows not a list": with_requirements(l2_rows=32),
            "prime_factors not a list": changed(["modulus"], {"prime_factors": 5}),
            # gamma 0 divided by zero (a traceback); 1 and odd values are no even divisor.
            "gamma 0": changed(["modulus"], dict(fixed, gamma=0)),
            "gamma '0'": changed(["modulus"], dict(fixed, gamma="0")),
            "gamma 1": changed(["modulus"], dict(fixed, gamma=1)),
            "gamma odd": changed(["modulus"], dict(fixed, gamma=262141)),
            "requirements statement_modulus 0": json.dumps(dict(req, statement=dict(
                req["statement"], statement_modulus=0))),
        }
        for name, bad in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(request.RequestError):
                    cli.derive_request(bad, what_if=False, compare=False)

    def test_statements_rust_cannot_build_are_refused(self):
        # Rust's Ring takes a modulus in [2, 2^256) and a degree that is a power of two from 64
        # to 1024, and Statement::var refuses repeated names, long names, empty blocks and more
        # than 32768 polynomials. The tool must refuse each, among them degree 2048, modulus 1
        # and two blocks named w, with the reason.
        base = json.loads((common.TOOL / "sets" / "kyber1024-d64.request.json").read_text())
        blk = base["statement"]["blocks"][0]

        def statement(**change):
            return json.dumps(dict(base, statement=dict(base["statement"], **change)))

        def blocks(*bs):
            return statement(blocks=[dict(blk, **b) for b in bs])
        cases = {
            "statement_modulus 0": statement(statement_modulus=0),
            "statement_modulus 1": statement(statement_modulus=1),
            "statement_modulus 2^256": statement(statement_modulus=str(2 ** 256)),
            "statement_degree 0": statement(statement_degree=0),
            "statement_degree 96": statement(statement_degree=96),
            "statement_degree 2048": statement(statement_degree=2048),
            "two blocks named w": blocks({"length": 4}, {"length": 4}),
            "a block of length 0": blocks({"length": 0}),
            "a name of 257 bytes": blocks({"name": "x" * 257}),
            "a name of 258 bytes in 129 characters": blocks({"name": "é" * 129}),
            "32769 polynomials": blocks({"length": 32768}, {"name": "v", "length": 1}),
        }
        for name, bad in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(request.RequestError):
                    cli.derive_request(bad, what_if=False, compare=False)
        # The limits themselves pass the request checks: a name of 256 bytes, a modulus of 2.
        for ok in (blocks({"name": "x" * 256}), statement(statement_modulus=2)):
            request.read(ok)

    def test_requirements_must_describe_one_statement(self):
        # With a range block, Statement::requirements gives linf_bound = max(1, ceil(F/p)) for
        # the statement modulus p. An F that does not match, such as 0 or 1 next to
        # kyber1024-d64's linf_bound, would give q = 2423955247093, which compile refuses.
        r = self.requirements_request()
        req = r["statement"]["requirements"]
        f, linf = int(req["max_integer_coefficient"]), int(req["linf_bound"])
        self.assertEqual(linf, -(-f // 3329))
        for bad_f in (0, 1, f - 3329):
            with self.subTest(max_integer_coefficient=bad_f):
                req["max_integer_coefficient"] = bad_f
                with self.assertRaises(request.RequestError) as e:
                    cli.derive_request(json.dumps(r), what_if=False, compare=False)
                self.assertIn("linf_bound", str(e.exception))
        # F = 0 with linf_bound 1 is what Rust gives for a lifted constraint whose form is
        # identically zero; within one ceiling step a wrong F cannot be told from the right one.
        for ok_f, ok_linf in ((0, 1), (f - 1, linf)):
            req.update(max_integer_coefficient=ok_f, linf_bound=ok_linf)
            lifting = request.read(json.dumps(r))[4]
            self.assertEqual((lifting["statement_modulus"], lifting["max_integer_coefficient"],
                              lifting["pairs"], lifting["lifted"]),
                             (3329, ok_f, [[3329, ok_f]], True))

    def test_no_equations_lift_nothing(self):
        # Statement::requirements lifts constraints only; without rows there is none, so n_prime,
        # linf_bound and F stay 0 (with linf_bound 3476 and F 11570096 but n_prime 0, derive
        # would refuse the set as an "approximate block").
        base = json.loads((common.TOOL / "sets" / "kyber1024-d64.request.json").read_text())
        base["statement"]["rows"] = 0
        _, _, _, req, lifting, _ = request.read(json.dumps(base))
        self.assertEqual((req["n_prime"], req["linf_bound"], req["max_integer_coefficient"],
                          lifting), (0, 0, 0, None))

    def test_fixed_modulus_gamma(self):
        # A fixed modulus takes the positional form's divisor (the largest even divisor of q-1 in
        # the window below gamma_0), so kyber1024-d64's own prime gives gamma 524284, not the
        # searched 262142; the request's `gamma` reproduces the set.
        text = (common.TOOL / "sets" / "kyber1024-d64.json").read_text()
        shipped = jsonio.load_params(text)
        r = json.loads((common.TOOL / "sets" / "kyber1024-d64.request.json").read_text())
        r["modulus"] = {"prime_factors": shipped["prime_factors"]}
        p, rep = cli.derive_request(json.dumps(r), what_if=False, compare=False)
        self.assertEqual(jsonio.load_params(p)["gamma"], 524284)
        self.assertIn("positional form", rep["modulus"]["gamma_rule"])
        r["modulus"]["gamma"] = shipped["gamma"]
        p, rep = cli.derive_request(json.dumps(r), what_if=False, compare=False)
        got = jsonio.load_params(p)
        self.assertEqual(rep["modulus"]["gamma_rule"], "given in the request")
        got.pop("estimator"), shipped.pop("estimator")
        self.assertEqual(got, shipped)
        # A gamma that does not divide q - 1 is refused.
        r["modulus"]["gamma"] = shipped["gamma"] + 2
        with self.assertRaises(derive.DeriveError):
            cli.derive_request(json.dumps(r), what_if=False, compare=False)


class RustLambda(unittest.TestCase):
    """$`\\lambda`$ as Rust computes it: log2 of the f64 nearest q1, which is 64.0 from
    $`2^{64}-46080`$ on, so $`\\lambda=2`$ there although the exact logarithm is below 64."""

    Q = 2 ** 64 - 899  # prime, 5 mod 8

    def test_boundary(self):
        self.assertTrue(modulus.is_prime(self.Q) and self.Q % 8 == 5)
        for q, lam in ((self.Q, 2), (2 ** 64 - 46080, 2), (2 ** 64 - 46081, 4), (2 ** 64 + 13, 2),
                       (2 ** 63, 4), (1099511627917, 4)):
            with self.subTest(q=q):
                self.assertEqual(derive.lambda_for(q), lam)
                self.assertEqual(rustcheck.rust_lambda(q), lam)

    def test_fixed_modulus_near_2_64(self):
        # With the exact logarithm the tool took lambda = 4 and wrote an m2 one larger than
        # Rust derives: "simulatability rank". Now the set passes the emulated check and compile.
        r = json.loads((common.TOOL / "sets" / "kyber1024-d64.request.json").read_text())
        r["modulus"] = {"prime_factors": [self.Q]}
        p, rep = cli.derive_request(json.dumps(r), what_if=False, compare=False)
        params = jsonio.load_params(p)
        derived, _ = rustcheck.check_params(params)
        # l_ext = 4 exact + 4 approximate slots + 1 + lambda/2 + 1 at degree 64.
        self.assertEqual((derived["lambda"], derived["l_ext"]), (2, 11))
        self.assertEqual(params["m2"], params["mlwe_rank"] + params["n_msis"] + params["l"] + 11)
        self.assertEqual(rep["derived"]["lambda"], 2)


class Refusals(unittest.TestCase):
    def search(self, req, lifting=None, capacity=rustcheck.CAPACITY):
        return derive.search(derive.Shape(req, 64), DELTA, lifting, capacity=capacity)

    def expect(self, kind, req, lifting=None, capacity=rustcheck.CAPACITY):
        with self.assertRaises(derive.DeriveError) as e:
            self.search(req, lifting, capacity)
        self.assertEqual(e.exception.kind, kind, str(e.exception))
        return e.exception

    def test_incomplete(self):
        self.expect("completeness dimensions", dict(TOY_NO_RANGE, m1=1, l2_rows=[1],
                                                    l2_bounds_squared=[64], n_bin=0))

    def test_width_capacity(self):
        # Reachable only with a limit of 40 on the exponents (NARROW_CAPACITY).
        self.expect("Gaussian width capacity", dict(TOY_NO_RANGE, alpha_squared=2 ** 64 - 1),
                    capacity=rustcheck.NARROW_CAPACITY)

    def test_dimensions(self):
        self.expect("dimensions", dict(TOY_NO_RANGE, m1=70000))

    def test_zero_bound(self):
        self.expect("exact norm blocks", dict(TOY_NO_RANGE, l2_bounds_squared=[128, 0]))

    def test_range_block_without_bound(self):
        self.expect("approximate block", dict(TOY_NO_RANGE, n_prime=2))

    def test_range_block_without_statement(self):
        req = dict(TOY_NO_RANGE, n_prime=2, linf_bound=4)
        self.expect("missing statement", req)

    def test_lifting_above_capacity(self):
        req = dict(TOY_NO_RANGE, n_prime=16, linf_bound=2 ** 20)
        e = self.expect("capacity", req, {"statement_modulus": 2 ** 40,
                                          "max_integer_coefficient": 2 ** 60},
                        rustcheck.NARROW_CAPACITY)
        self.assertIn("modulus lifting bound", str(e))
        e = self.expect("capacity", req, {"statement_modulus": 2 ** 230,
                                          "max_integer_coefficient": 2 ** 250})
        self.assertIn("2^256", str(e))

    def test_search_capacity(self):
        e = self.expect("capacity", TOY_NO_RANGE, None,
                        dict(rustcheck.CAPACITY, prime_bits=30))
        self.assertIn("2^30", str(e))

    def test_field_capacity(self):
        # TboxParams stores these bounds as u64.
        for change in ({"alpha_squared": 2 ** 64}, {"l2_bounds_squared": [128, 2 ** 64]},
                       {"n_prime": 1, "linf_bound": 2 ** 64}):
            with self.subTest(change=change):
                with self.assertRaises(derive.DeriveError) as e:
                    derive.Shape(dict(TOY_NO_RANGE, **change), 64)
                self.assertEqual(e.exception.kind, "field capacity")

    def test_max_bits_is_named(self):
        r = json.loads((common.TOOL / "sets" / "kyber1024-d64.request.json").read_text())
        r["modulus"]["search"]["max_bits"] = 40
        with self.assertRaises(derive.DeriveError) as e:
            cli.derive_request(json.dumps(r), what_if=False, compare=False)
        self.assertEqual(e.exception.kind, "capacity")
        self.assertIn("max_bits (q < 2^40)", str(e.exception))
        r["modulus"]["search"]["max_bit"] = 40
        with self.assertRaises(request.RequestError):
            cli.derive_request(json.dumps(r), what_if=False, compare=False)

    def test_max_bits_0_is_a_cap(self):
        # max_bits 0 asks for q < 1: a cap, not an absent field, as a number and as the
        # string "0"; the search refuses both.
        r = json.loads((common.TOOL / "sets" / "kyber1024-d64.request.json").read_text())
        for zero in (0, "0"):
            r["modulus"]["search"]["max_bits"] = zero
            with self.subTest(max_bits=zero):
                with self.assertRaises(derive.DeriveError) as e:
                    cli.derive_request(json.dumps(r), what_if=False, compare=False)
                self.assertEqual(e.exception.kind, "capacity")
                self.assertIn("max_bits (q < 2^0)", str(e.exception))
        # min_bits 0 asks for q >= 1/2, which every q meets: the same as no min_bits.
        del r["modulus"]["search"]["max_bits"]
        r["modulus"]["search"]["min_bits"] = 0
        self.assertIsNone(request.read(json.dumps(r))[5]["search"]["min_bits"])

    def test_a_huge_min_bits_is_refused_before_it_is_formed(self):
        # 2^(10^12) would not fit in memory; the capacity check comes first.
        r = json.loads((common.TOOL / "sets" / "kyber1024-d64.request.json").read_text())
        for bits, cap in ((10 ** 12, "the Rust capacity 2^256"), (257, "the Rust capacity 2^256"),
                          (41, "the request's max_bits (q < 2^40)")):
            r["modulus"]["search"].update(min_bits=bits, max_bits=40 if bits == 41 else 300)
            with self.subTest(min_bits=bits):
                with self.assertRaises(derive.DeriveError) as e:
                    cli.derive_request(json.dumps(r), what_if=False, compare=False)
                self.assertEqual(str(e.exception),
                                 f"capacity: min_bits needs q >= 2^{bits - 1}, above {cap}")

    def test_a_gamma_over_max_bits_is_skipped(self):
        # With min_bits = max_bits = 46 the largest gamma's prime is 2^46.16; the search skips
        # that gamma and takes the best of the smaller ones instead of giving up.
        r = json.loads((common.TOOL / "sets" / "kyber1024-d64.request.json").read_text())
        r["modulus"]["search"].update(min_bits=46, max_bits=46)
        p, rep = cli.derive_request(json.dumps(r), what_if=False, compare=False)
        got = jsonio.load_params(p)
        self.assertTrue(1 << 45 <= got["prime_factors"][0] < 1 << 46)
        self.assertEqual((got["gamma"], rep["summary"]["estimated_proof_bytes"]), (131070, 21926))
        steps = [s for t in rep["modulus"]["search"] for s in t["steps"]]
        over = [s for s in steps if s.get("event", "").startswith("capacity")]
        self.assertEqual([s["gamma"] for s in over], [524286])
        self.assertIn("exceeds the request's max_bits (q < 2^46)", over[0]["rejected"][0]["why"])
        accepted = [s["gamma"] for s in steps if s.get("event") == "accepted"]
        self.assertEqual(accepted, [262142, 131070, 65534])

    def test_a_gamma_over_max_bits_does_not_end_the_search(self):
        # Demo with max_bits 65: at the 65-bit start, gamma 1048574's prime is 2^65.01, above
        # the cap, so the smaller gammas of that start are tried, and gamma 524286 wins with
        # 24,438 B. (The shipped demo set, with max_bits 64, has gamma 65534 and 24,814 B.)
        r = json.loads((common.TOOL / "sets" / "demo.request.json").read_text())
        r["modulus"]["search"]["max_bits"] = 65
        p, rep = cli.derive_request(json.dumps(r), what_if=False, compare=False)
        got = jsonio.load_params(p)
        self.assertEqual((got["prime_factors"], got["gamma"], got["d_bits"]),
                         ([18446744073723182029], 524286, 10))
        self.assertEqual(rep["summary"]["estimated_proof_bytes"], 24438)
        by_bits = {t["bits"]: t for t in rep["modulus"]["search"]}
        shipped = jsonio.load_params((common.TOOL / "sets" / "demo.json").read_text())
        at61 = by_bits[61]
        self.assertEqual((jsonio.decode_int(at61["q"]), at61["estimated_proof_bytes"]),
                         (shipped["prime_factors"][0], 24814))
        last = by_bits[65]["steps"]
        self.assertEqual(last[0]["gamma"], 1048574)
        self.assertIn("exceeds the request's max_bits (q < 2^65)", last[0]["event"])
        self.assertEqual([s["gamma"] for s in last if s.get("event") == "accepted"],
                         [524286, 262142, 131070])

    def test_demo_below_2_64_and_the_alternative(self):
        # The demo request keeps q below 2^64; without max_bits the window reaches 65 bits,
        # where lambda drops from 4 to 2 and the estimate is smaller.
        text = (common.TOOL / "sets" / "demo.request.json").read_text()
        r = json.loads(text)
        self.assertEqual(r["modulus"]["search"]["max_bits"], 64)
        shipped = jsonio.load_params((common.TOOL / "sets" / "demo.json").read_text())
        del r["modulus"]["search"]["max_bits"]
        p, rep = cli.derive_request(json.dumps(r), what_if=False, compare=False)
        wide = jsonio.load_params(p)
        self.assertGreaterEqual(wide["prime_factors"][0], 2 ** 64)
        self.assertLess(shipped["prime_factors"][0], 2 ** 64)
        self.assertLess(rep["summary"]["estimated_proof_bytes"],
                        rustcheck.check_params(shipped)[0]["estimated_proof_bytes"])
        self.assertEqual(rustcheck.check_params(wide)[0]["lambda"], 2)
        self.assertIsInstance(json.loads(p)["prime_factors"][0], str)

    def test_two_primes(self):
        req = common.load("tools/params/toy-d64.request.json")
        shape = derive.Shape(req, 64)
        with self.assertRaises(derive.DeriveError) as e:
            derive.solve(shape, [13, 1099511627917], DELTA)
        self.assertEqual(e.exception.kind,
                         "binary, exact-norm and range blocks require a prime modulus")
        text = (common.TOOL / "sets" / "demo.request.json").read_text()
        r = json.loads(text)
        r["modulus"]["search"]["factors"] = 2
        with self.assertRaises(request.RequestError):
            cli.derive_request(json.dumps(r))


if __name__ == "__main__":
    unittest.main()
