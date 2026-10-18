"""Two-sided exact Euclidean bounds (`{"l2_squared_range": {"min": a, "max": b}}`,
`Norm::L2SquaredRange`), as `Statement::blocks` exports them. The block itself counts as an
exact-norm block with its upper bound; the compiler adds one binary lower-bound row to the Ajtai
part, wherever the block is placed, with a share of `d`, which the reader expects in `m1`,
`alpha_squared` and `n_bin`. The reader refuses what `Statement::var` refuses, and a placement
search moves the block's own rows but not the lower-bound row. The exports below are Rust's
(`Statement::requirements` and `Statement::blocks` at degree 64 and the probe modulus
1099511627917 for the statement of `tests/l2_range.rs`: `a`, five polynomials with
`l2_squared` 640, `s`, two with `l2_squared_range` 5 to 64, a binary `x`, and one lifted
constraint modulo 1313)."""
import copy
import json
import unittest

import common  # noqa: F401  (puts the tool on the path)
from jali_params import cli, partition, request

BLOCKS = [{"name": "a", "norm": {"l2_squared": 640}, "placement": "ajtai", "rows": 10},
          {"name": "s", "norm": {"l2_squared_range": {"max": 64, "min": 5}},
           "placement": "ajtai", "rows": 4},
          {"name": "x", "norm": "binary", "placement": "ajtai", "rows": 2}]
AJTAI = {"alpha_squared": 896, "l": 0, "l2_bounds_squared": [640, 64], "l2_rows": [10, 4],
         "linf_bound": 1, "m1": 17, "max_integer_coefficient": 20, "n_bin": 3, "n_prime": 2}
# The same statement with s placed in the BDLOP part.
BDLOP = {"alpha_squared": 832, "l": 4, "l2_bounds_squared": [640, 64], "l2_rows": [10, 4],
         "linf_bound": 1, "m1": 13, "max_integer_coefficient": 20, "n_bin": 3, "n_prime": 2}


def make_request(requirements=AJTAI, placement="ajtai"):
    blocks = copy.deepcopy(BLOCKS)
    blocks[1]["placement"] = placement
    return {"schema": "jali-params-request/2", "id": "l2-range", "degree": 64,
            "modulus": {"search": {"factors": 1, "window_bits": 4}},
            "hardness": {"policy": "delta"},
            "statement": {"source": "requirements", "requirements": copy.deepcopy(requirements),
                          "statement_modulus": 1313, "blocks": blocks}}


class Norms(unittest.TestCase):
    def test_the_range_norm_and_its_refusals(self):
        self.assertEqual(request._exported_norm({"l2_squared_range": {"min": 5, "max": 64}}, "n"),
                         ("l2_squared_range", 64, 5))
        self.assertEqual(request._exported_norm({"l2_squared_range": {"min": 5, "max": 5}}, "n"),
                         ("l2_squared_range", 5, 5))
        for bad in ({"l2_squared_range": {"min": 0, "max": 5}},
                    {"l2_squared_range": {"min": 6, "max": 5}},
                    {"l2_squared_range": {"min": 1, "max": 2 ** 64}},
                    {"l2_squared_range": {"min": 1}},
                    {"l2_squared_range": {"min": 1, "max": 2, "lower": 1}},
                    {"l2_squared_range": [1, 2]},
                    {"l2_squared_range": {"min": 1, "max": 2}, "l2_squared": 2}):
            with self.subTest(norm=bad), self.assertRaises(request.RequestError):
                request._exported_norm(bad, "n")

    def test_the_share_is_the_upper_bound(self):
        self.assertEqual(request.share(("l2_squared_range", 64, 5), 4, 64), 64)


class Exports(unittest.TestCase):
    def test_rusts_exports_read_with_the_lower_bound_row(self):
        for requirements, placement in ((AJTAI, "ajtai"), (BDLOP, "bdlop")):
            with self.subTest(placement=placement):
                _, _, _, req, _, _ = request.read(json.dumps(make_request(requirements,
                                                                          placement)))
                self.assertEqual((req["m1"], req["n_bin"], req["alpha_squared"]),
                                 (requirements["m1"], 3, requirements["alpha_squared"]))
                self.assertEqual(req["l2_bounds_squared"], [640, 64])

    def test_requirements_without_the_lower_bound_row_are_refused(self):
        for key, change in (("m1", -1), ("n_bin", -1), ("alpha_squared", -64)):
            r = make_request()
            r["statement"]["requirements"][key] += change
            with self.subTest(key=key), self.assertRaises(request.RequestError) as e:
                request.read(json.dumps(r))
            self.assertIn(key, str(e.exception))

    def test_moving_the_block_leaves_the_lower_bound_row(self):
        r = make_request()
        r["statement"]["movable"] = ["s"]
        blocks, names = request.placement_of(r)
        self.assertEqual(names, ["s"])
        _, _, _, req, _, _ = request.read(json.dumps(r))
        s = blocks[1]
        moved = partition.placed_requirements(req, [(s["rows"], s["share"])], (0,))
        # The Rust export with s in the BDLOP part.
        self.assertEqual((moved["m1"], moved["l"], moved["alpha_squared"], moved["n_bin"]),
                         (BDLOP["m1"], BDLOP["l"], BDLOP["alpha_squared"], BDLOP["n_bin"]))
        _, rep = cli.derive_request(json.dumps(r), what_if=False, compare=False)
        evaluated = {tuple(e["bdlop_blocks"]): e for e in rep["placement"]["evaluated"]}
        self.assertEqual((evaluated[()]["m1"], evaluated[("s",)]["m1"]), (17, 13))


if __name__ == "__main__":
    unittest.main()
