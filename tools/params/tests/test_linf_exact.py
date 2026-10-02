"""Blocks bounded exactly by bits (`{"linf_exact": b}`, `Norm::LinfExact`) and subring blocks
(`Statement::var_subring`), as `Statement::blocks` exports them. A block's `rows` are the
proof-ring polynomials it commits: the nonzero components of a subring block, the bits of an
exact one, which count as binary rows with a share of `rows * d` and nothing in `n_prime` or
`linf_bound`. The reader refuses what `Statement::var` refuses, and a placement search moves an
exact block's bit rows with it. The reference kinds `subring` and `linf-exact` of
`rust_reference.py` are Rust's exports; `test_rustcheck.py` also checks the tool's sets and its
emulation of `compile` on them against Rust's decisions."""
import copy
import json
import unittest

import common
import rust_reference
from jali_params import cli, partition, request

REF = rust_reference.load(common.DATA / "rust_reference.json")


def kind(name):
    return copy.deepcopy(REF["statements"][name]["request"])


class Norms(unittest.TestCase):
    def test_the_exact_norm_and_its_refusals(self):
        self.assertEqual(request._exported_norm({"linf_exact": 512}, "n"), ("linf_exact", 512))
        self.assertEqual(request._exported_norm({"linf_exact": 2 ** 63 - 1}, "n"),
                         ("linf_exact", 2 ** 63 - 1))
        # Statement::var refuses 0 and 2^63 on; a u64 bounds every norm; Rust's spelling only.
        for bad in ({"linf_exact": 0}, {"linf_exact": 2 ** 63}, {"linf_exact": 2 ** 64},
                    {"linf_exact": -1}, {"linf_exact": 1.5}, {"linfexact": 1},
                    {"linf_exact": 1, "linf": 1}, "linf_exact"):
            with self.subTest(norm=bad), self.assertRaises(request.RequestError):
                request._exported_norm(bad, "n")

    def test_shares(self):
        # Bits are binary coefficients: rows * d, whatever beta; linf pays beta^2.
        self.assertEqual(request.share(("linf_exact", 512), 22, 64), 22 * 64)
        self.assertEqual(request.share(("binary",), 22, 64), 22 * 64)
        self.assertEqual(request.share(("linf", 5), 2, 64), 2 * 64 * 25)


class ReferenceKinds(unittest.TestCase):
    def test_rusts_exports_read(self):
        # linf-exact: s (4 polynomials, 8 rows at degree 64), u (two bits per component, 4
        # rows) in the Ajtai part, v (a subring element of degree 64, one component of four
        # bits) in the BDLOP part; the bits of u and v are all the binary rows.
        entry = REF["statements"]["linf-exact"]
        blocks = {b["name"]: b for b in entry["blocks"]}
        self.assertEqual([(n, b["rows"], b["norm"], b["placement"]) for n, b in blocks.items()],
                         [("s", 8, {"l2_squared": 64}, "ajtai"),
                          ("u", 4, {"linf_exact": 1}, "ajtai"),
                          ("v", 4, {"linf_exact": 5}, "bdlop")])
        _, _, _, req, lifting, _ = request.read(json.dumps(entry["request"]))
        self.assertEqual((req["m1"], req["n_bin"], req["alpha_squared"]),
                         (12, 8, 64 + 4 * 64))
        # No linf section: nothing depends on E but the carries and quotients.
        self.assertEqual((lifting["has_linf"], lifting["betas"]), (False, []))
        # subring: x (binary) and y (linf 2), one component each at degree 64.
        entry = REF["statements"]["subring"]
        self.assertEqual([(b["name"], b["rows"]) for b in entry["blocks"]],
                         [("s", 8), ("x", 1), ("y", 1)])
        _, _, _, req, lifting, _ = request.read(json.dumps(entry["request"]))
        self.assertEqual((req["n_bin"], req["alpha_squared"]), (1, 64 + 64 + 64 * 4))
        self.assertEqual(lifting["betas"], [2])

    def test_inconsistent_blocks_are_refused(self):
        def refused(change):
            r = kind("linf-exact")
            change(r["statement"])
            with self.assertRaises(request.RequestError) as e:
                request.read(json.dumps(r))
            return str(e.exception)
        # As binary the bits add up the same, and so does another beta with as many bits (the
        # requirements see beta only through the integer bounds); as linf they are no binary
        # rows.
        for block, norm in ((1, "binary"), (2, {"linf_exact": 6})):
            r = kind("linf-exact")
            r["statement"]["blocks"][block]["norm"] = norm
            request.read(json.dumps(r))
        self.assertIn("n_bin", refused(
            lambda s: s["blocks"][1].__setitem__("norm", {"linf": 1})))
        # The bits are binary rows: n_bin must count them.
        self.assertIn("n_bin", refused(
            lambda s: s["requirements"].__setitem__("n_bin", s["requirements"]["n_bin"] - 4)))
        self.assertIn("refuses", refused(
            lambda s: s["blocks"][2].__setitem__("norm", {"linf_exact": 0})))

    def test_moving_an_exact_block_moves_its_bits(self):
        r = kind("linf-exact")
        r["statement"]["movable"] = ["u"]
        blocks, names = request.placement_of(r)
        _, _, _, req, lifting, _ = request.read(json.dumps(r))
        u = blocks[1]
        moved = partition.placed_requirements(req, [(u["rows"], u["share"])], (0,))
        self.assertEqual((moved["m1"], moved["l"], moved["alpha_squared"]),
                         (req["m1"] - 4, req["l"] + 4, req["alpha_squared"] - 4 * 64))
        self.assertEqual(moved["n_bin"], req["n_bin"])
        # The search evaluates both placements with those requirements: with u in BDLOP the
        # Ajtai part keeps s alone, 8 rows, below the check's (m1 + Z) d >= 640.
        _, rep = cli.derive_request(json.dumps(r), what_if=False, compare=False)
        evaluated = rep["placement"]["evaluated"]
        self.assertEqual([(e["bdlop_blocks"], e["m1"], e["l"], e["alpha_squared"])
                          for e in evaluated],
                         [([], 12, 7, 320), (["u"], 8, 11, 64)])
        self.assertIn("estimated_proof_bytes", evaluated[0])
        self.assertIn("completeness dimensions", evaluated[1]["failed"])
        self.assertEqual(rep["placement"]["chosen"]["ajtai"], ["s", "u"])


if __name__ == "__main__":
    unittest.main()
