"""JSON conventions: existing files round-trip byte for byte; the number rule for wide integers."""
import json
import unittest

import common
from jali_params import jsonio


class RoundTrip(unittest.TestCase):
    def test_existing_parameter_files(self):
        files = common.PARAMS_FILES + [f"tools/params/sets/{n}.json" for n in common.NEW_SETS]
        for path in files:
            text = common.read(path)
            with self.subTest(path=path):
                self.assertEqual(jsonio.dump_params(jsonio.load_params(text)), text)

    def test_reports_and_requests_are_canonical(self):
        for name in common.NEW_SETS:
            for suffix in (".request.json", ".report.json"):
                text = (common.TOOL / "sets" / f"{name}{suffix}").read_text()
                self.assertEqual(jsonio.dumps(json.loads(text)), text, name + suffix)


class NumberRule(unittest.TestCase):
    def test_encode(self):
        self.assertEqual(jsonio.encode_int(2 ** 64 - 1), 2 ** 64 - 1)
        self.assertEqual(jsonio.encode_int(2 ** 64), "18446744073709551616")
        with self.assertRaises(TypeError):
            jsonio.encode_int(True)
        with self.assertRaises(TypeError):
            jsonio.encode_int(-1)

    def test_decode(self):
        self.assertEqual(jsonio.decode_int(5), 5)
        self.assertEqual(jsonio.decode_int(2 ** 64 - 1), 2 ** 64 - 1)
        self.assertEqual(jsonio.decode_int("18446744073709551616"), 2 ** 64)
        self.assertEqual(jsonio.decode_int("0"), 0)
        # A JSON number of 2^64 or more is refused, as the crate refuses it (src/json.rs).
        for bad in ("01", "-1", "1.0", " 1", "", "١", True, 1.0, -3, None, 2 ** 64):
            with self.assertRaises(ValueError, msg=repr(bad)):
                jsonio.decode_int(bad)

    def test_strict_decoding(self):
        text = common.read("src/params/sets/toy-d64.json")
        self.assertEqual(jsonio.load_params(text)["gamma"], 65202)
        delta = '"mlwe_delta": 1.0042736680871425'
        for bad in (text.replace('"gamma": 65202,', '"gamma": 65202,\n  "gamma": 65202,'),
                    text.replace(delta, '"mlwe_delta": NaN'),
                    text.replace(delta, '"mlwe_delta": Infinity'),
                    text.replace('"prime_factors": [\n    1099511627917\n  ]',
                                 '"prime_factors": [\n    18446744073709551629\n  ]'),
                    "[]"):
            self.assertNotEqual(bad, text)
            with self.assertRaises(ValueError):
                jsonio.load_params(bad)

    def test_wide_prime_is_a_string(self):
        p = jsonio.load_params(common.read("src/params/sets/toy-d64.json"))
        q = 2 ** 127 - 1
        p["prime_factors"] = [q]
        text = jsonio.dump_params(p)
        self.assertIn('"170141183460469231731687303715884105727"', text)
        self.assertEqual(jsonio.load_params(text)["prime_factors"], [q])
        mixed = dict(p, prime_factors=[13, q])
        self.assertEqual(json.loads(jsonio.dump_params(mixed))["prime_factors"], [13, str(q)])

    def test_every_tool_file_follows_the_rule(self):
        # Parameter files, requests, reports, cross-checks and the test data: no JSON number of
        # 2^64 or more anywhere (such values are decimal strings).
        paths = sorted(common.TOOL.glob("**/*.json")) + sorted(common.TOOL.glob("**/*.jsonl"))
        self.assertGreaterEqual(len(paths), 21)

        def wide(o):
            if isinstance(o, dict):
                return any(wide(v) for v in o.values())
            if isinstance(o, list):
                return any(wide(v) for v in o)
            return isinstance(o, int) and not isinstance(o, bool) and abs(o) >= jsonio.LIMIT
        for path in paths:
            text = path.read_text()
            docs = [json.loads(x) for x in text.splitlines() if x.strip()] \
                if path.suffix == ".jsonl" else [json.loads(text)]
            with self.subTest(path=str(path.relative_to(common.TOOL))):
                self.assertFalse(any(wide(d) for d in docs))

    def test_fields(self):
        text = common.read("src/params/sets/toy-d64.json")
        raw = json.loads(text)
        with self.assertRaises(ValueError):
            jsonio.load_params(json.dumps(dict(raw, extra=1)))
        del raw["gamma"]
        with self.assertRaises(ValueError):
            jsonio.load_params(json.dumps(raw))


if __name__ == "__main__":
    unittest.main()
