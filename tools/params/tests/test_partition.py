"""Placement: exhaustive enumeration, a shape that gains, and the search of `derive` over the
blocks a request calls movable, whose result `Statement::var_placed` and `lin::compile_placed`
realize (`test_rustcheck.py` checks one against Rust)."""
import copy
import itertools
import json
import unittest

import common
from jali_params import cli, derive, hardness, jsonio, partition, request

POLICY = hardness.Policy("delta")
# One short block with a huge norm and a long block with a small one: moving the first to
# BDLOP removes it from alpha and shrinks sigma_1.
REQ = dict(m1=21, l=0, alpha_squared=2 ** 40 + 1000, n_bin=0, l2_rows=[1, 20],
           l2_bounds_squared=[2 ** 40, 1000], n_prime=0, linf_bound=0)
BLOCKS = [(1, 2 ** 40), (20, 1000)]


class Placements(unittest.TestCase):
    def test_enumeration(self):
        for n in range(6):
            got = list(partition.placements(n))
            self.assertEqual(sorted(got), sorted(itertools.chain.from_iterable(
                itertools.combinations(range(n), k) for k in range(n + 1))))
            self.assertEqual(len(got), 2 ** n)

    def test_requirements(self):
        r = partition.placed_requirements(REQ, BLOCKS, (0,))
        self.assertEqual((r["m1"], r["l"], r["alpha_squared"]), (20, 1, 1000))
        self.assertEqual(r["l2_rows"], REQ["l2_rows"])

    def test_a_placement_that_gains_is_found(self):
        res = partition.evaluate(REQ, BLOCKS, 64, POLICY, None)
        self.assertEqual(res["method"], "exhaustive")
        self.assertEqual(len(res["evaluated"]), 4)
        best = partition.best(res)
        self.assertEqual(best["bdlop_blocks"], [0])
        all_ajtai = res["evaluated"][0]
        self.assertLess(best["estimated_proof_bytes"], all_ajtai["estimated_proof_bytes"])
        # The exhaustive best is the minimum over every feasible placement.
        sizes = [e["estimated_proof_bytes"] for e in res["evaluated"]
                 if "estimated_proof_bytes" in e]
        self.assertEqual(best["estimated_proof_bytes"], min(sizes))

    def test_more_blocks_than_the_limit_are_refused(self):
        many = [(1, 1)] * (partition.EXHAUSTIVE_LIMIT + 1)
        for call in (lambda: partition.candidates(many),
                     lambda: partition.evaluate(REQ, many, 64, POLICY, None)):
            with self.assertRaises(derive.DeriveError) as e:
                call()
            self.assertEqual(e.exception.kind, "placement")
            self.assertTrue(e.exception.final)
        self.assertEqual(len(list(partition.candidates(many[1:]))),
                         2 ** partition.EXHAUSTIVE_LIMIT)

    def test_new_sets_have_nothing_to_move(self):
        # Their one block is the whole Ajtai part: moving it leaves the part empty.
        for name in common.NEW_SETS:
            rep = json.loads((common.TOOL / "sets" / f"{name}.report.json").read_text())
            best = partition.best(rep["partition"])
            self.assertEqual(best["bdlop_blocks"], [], name)
            self.assertIn("empty Ajtai part", rep["partition"]["evaluated"][1]["failed"])
            self.assertEqual(rep["partition"]["evaluated"][1]["bdlop_blocks"], ["w"])


def lin_request(movable=None, **w):
    """A lin-worst-case request over Z_12289, degree 128: w (one polynomial, ||w||^2 <= 2^24)
    and s (ten, ||s||^2 <= 1000), one equation with public coefficients of absolute value 6.
    `w` holds extra fields of w's block."""
    statement = {"source": "lin-worst-case", "statement_degree": 128,
                 "statement_modulus": 12289, "rows": 1, "public_linf": 6,
                 "blocks": [dict({"name": "w", "length": 1, "norm": {"l2_squared": 2 ** 24}},
                                 **w),
                            {"name": "s", "length": 10, "norm": {"l2_squared": 1000}}]}
    if movable is not None:
        statement["movable"] = movable
    return {"schema": request.SCHEMA, "id": "placement-test", "degree": 64,
            "modulus": {"search": {"factors": 1, "window_bits": 1}},
            "hardness": {"policy": "delta"}, "statement": statement}


def without_estimator(text):
    params = jsonio.load_params(text)
    params.pop("estimator")
    return params


class Search(unittest.TestCase):
    """`derive` with movable blocks keeps the placement with the smallest estimate."""

    @classmethod
    def setUpClass(cls):
        cls.text, cls.report = cli.derive_request(json.dumps(lin_request(["w", "s"])),
                                                  what_if=False, compare=False)

    def test_the_search_moves_the_block_with_the_large_bound(self):
        placement = self.report["placement"]
        self.assertEqual((placement["method"], placement["movable"]), ("exhaustive", ["w", "s"]))
        self.assertEqual(placement["chosen"], {"ajtai": ["s"], "bdlop": ["w"]})
        sizes = {tuple(e["bdlop_blocks"]): e.get("estimated_proof_bytes")
                 for e in placement["evaluated"]}
        self.assertEqual(list(sizes), [(), ("w",), ("s",), ("w", "s")])
        self.assertLess(sizes[("w",)], sizes[()])
        # s alone in the Ajtai part is too short for completeness; neither leaves it empty.
        self.assertEqual((sizes[("s",)], sizes[("w", "s")]), (None, None))
        params = jsonio.load_params(self.text)
        self.assertEqual((params["m1"], params["l"], params["alpha_squared"]), (20, 2, 1000))
        self.assertEqual(self.report["summary"]["estimated_proof_bytes"], sizes[("w",)])
        self.assertNotIn("partition", self.report)

    def test_check_assumes_the_placement_the_set_was_derived_for(self):
        # The request declares w in the Ajtai part; derive moved it to BDLOP. check must not
        # compare the set with the declared placement, which it would refuse ("bounded witness
        # norm budget"), but find the placement whose m1, l and alpha_squared are the set's.
        code, out, err = common.check(self.text, lin_request(["w", "s"]))
        self.assertEqual(code, 0, out + err)
        out = json.loads(out)
        self.assertTrue(out["accepted"])
        self.assertEqual(out["placement_assumed"], {"ajtai": ["s"], "bdlop": ["w"]})
        # Without movable blocks the declared placement is checked, and the set fails it.
        code, out, _ = common.check(self.text, lin_request())
        self.assertEqual((code, json.loads(out)["rust_message"]),
                         (1, "bounded witness norm budget"))
        # A set derived for the declared placement matches the empty one.
        text, _ = cli.derive_request(json.dumps(lin_request()), what_if=False, compare=False)
        code, out, err = common.check(text, lin_request(["w", "s"]))
        self.assertEqual(code, 0, out + err)
        self.assertEqual(json.loads(out)["placement_assumed"],
                         {"ajtai": ["w", "s"], "bdlop": []})
        # A set that fits no placement is refused as such.
        params = jsonio.load_params(self.text)
        params["alpha_squared"] += 1
        code, out, _ = common.check(jsonio.dump_params(params), lin_request(["w", "s"]))
        out = json.loads(out)
        self.assertEqual((code, out["accepted"], out["rust_message"]), (1, False, "placement"))
        self.assertIn("(20, 2, 1001) matches no placement", out["error"])

    def test_candidates_follow_the_search(self):
        # Every subset, in the order evaluate follows.
        self.assertEqual(list(partition.candidates(BLOCKS)), list(partition.placements(2)))
        evaluated = partition.evaluate(REQ, BLOCKS, 64, POLICY, None)["evaluated"]
        self.assertEqual([tuple(e["bdlop_blocks"]) for e in evaluated],
                         list(partition.candidates(BLOCKS)))

    def test_a_declared_placement_gives_the_same_set(self):
        text, report = cli.derive_request(json.dumps(lin_request(placement="bdlop")),
                                          what_if=False, compare=False)
        self.assertEqual(without_estimator(text), without_estimator(self.text))
        self.assertNotIn("placement", report)

    def test_copied_requirements_with_blocks_search_the_same(self):
        # Requirements and blocks as Rust exports them for w declared in the BDLOP part.
        _, _, _, req, _, _ = request.read(json.dumps(lin_request(placement="bdlop")))
        blocks = [{"name": "w", "rows": 2, "norm": {"l2_squared": 2 ** 24}, "placement": "bdlop"},
                  {"name": "s", "rows": 20, "norm": {"l2_squared": 1000}, "placement": "ajtai"}]
        r = lin_request()
        r["statement"] = {"source": "requirements", "statement_modulus": 12289,
                          "requirements": {k: jsonio.encode_wide(v) for k, v in req.items()},
                          "blocks": blocks, "movable": ["w"]}
        text, report = cli.derive_request(json.dumps(r), what_if=False, compare=False)
        self.assertEqual(report["placement"]["chosen"], {"ajtai": ["s"], "bdlop": ["w"]})
        self.assertEqual(without_estimator(text), without_estimator(self.text))

    def test_at_a_fixed_modulus(self):
        params = jsonio.load_params(self.text)
        r = lin_request(["w"])
        r["modulus"] = {"prime_factors": params["prime_factors"], "gamma": params["gamma"]}
        text, report = cli.derive_request(json.dumps(r), what_if=False, compare=False)
        self.assertEqual(report["placement"]["chosen"]["bdlop"], ["w"])
        self.assertEqual(without_estimator(text), without_estimator(self.text))

    def test_refusals(self):
        blocks = [{"name": "w", "rows": 2, "norm": {"l2_squared": 2 ** 24}, "placement": "ajtai"},
                  {"name": "s", "rows": 20, "norm": {"l2_squared": 1000}, "placement": "ajtai"},
                  {"name": "u", "rows": 2, "norm": "unbounded", "placement": "bdlop"}]
        _, _, _, req, _, _ = request.read(json.dumps(lin_request()))
        req = dict(req, l=2)
        copied = lin_request()
        copied["statement"] = {"source": "requirements", "statement_modulus": 12289,
                               "requirements": req, "blocks": blocks}
        for movable, statement in ((["v"], None), (["w", "w"], None), ([], None),
                                   (["u"], copied["statement"])):
            r = lin_request(movable)
            if statement is not None:
                r["statement"] = dict(statement, movable=movable)
            with self.subTest(movable=movable):
                with self.assertRaises(request.RequestError):
                    cli.derive_request(json.dumps(r), what_if=False, compare=False)
        # More movable blocks than the search tries.
        r = lin_request()
        r["statement"]["blocks"] = [{"name": f"b{i}", "length": 1, "norm": {"l2_squared": 1}}
                                    for i in range(partition.EXHAUSTIVE_LIMIT + 1)]
        r["statement"]["movable"] = [b["name"] for b in r["statement"]["blocks"]]
        with self.assertRaises(request.RequestError) as e:
            request.read(json.dumps(r))
        self.assertIn("13 movable blocks", str(e.exception))
        # Copied requirements need the blocks to move one.
        r = copy.deepcopy(copied)
        r["statement"].pop("blocks")
        r["statement"]["movable"] = ["w"]
        with self.assertRaises(request.RequestError):
            request.read(json.dumps(r))
        # No placement that works: w alone is too short for completeness in the Ajtai part,
        # and in the BDLOP part it leaves that part empty.
        r = lin_request(["w"])
        r["statement"]["blocks"] = r["statement"]["blocks"][:1]
        with self.assertRaises(derive.DeriveError) as e:
            cli.derive_request(json.dumps(r), what_if=False, compare=False)
        self.assertEqual(e.exception.kind, "placement")


if __name__ == "__main__":
    unittest.main()
