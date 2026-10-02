"""The optional cross-check with malb's lattice-estimator (JALI_LATTICE_ESTIMATOR_DIR, commit
53da598), which needs SageMath. Run with
`sage -python -m unittest discover -s tools/params/tests -p test_xcheck.py -v`; the tests skip,
with the reason, when the estimator is unavailable. `WithoutSage` runs everywhere."""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import common
from jali_params import cli, jsonio, xcheck

ESTIMATOR = xcheck.available("lattice-estimator")
ESTIMATOR_ENV = "JALI_LATTICE_ESTIMATOR_DIR"


def run_tool(*args, env=None):
    return subprocess.run([sys.executable, str(common.TOOL / "lnp_params.py"), *map(str, args)],
                          capture_output=True, text=True, env=env)


class WithoutSage(unittest.TestCase):
    """What runs without Sage: the number rule of the recorded costs and the target verdict."""

    def test_a_huge_sample_count_is_a_string(self):
        # BKW's sample count in the full estimate is near 2^292: a decimal string, not a number.
        out = xcheck._cost({"rop": 2.0 ** 308.8, "m": 2 ** 292 + 1, "beta": 343, "d": 3231})
        self.assertEqual(out, {"rop": 308.8, "beta": 343, "d": 3231, "m": str(2 ** 292 + 1)})

    def test_verdict_on_the_recorded_rough_estimates(self):
        for name in common.NEW_SETS:
            x = json.loads((common.TOOL / "sets" / f"{name}.xcheck.json").read_text())
            costs = {p: {a: (c["rop"], jsonio.decode_int(c["beta"])) for a, c in x[p].items()}
                     for p in ("mlwe", "msis")}
            with self.subTest(set=name):
                v = xcheck.verdict(costs, 128)
                self.assertFalse(v["reached"])
                self.assertEqual(v["cheapest"]["mlwe"]["attack"], "dual_hybrid")
                self.assertEqual(v["cheapest"]["mlwe"]["rop_log2"], x["mlwe_min_rop_log2"])
                self.assertEqual(v["cheapest"]["msis"]["rop_log2"], x["msis_min_rop_log2"])
                self.assertIn("MLWE dual_hybrid 2^", xcheck.shortfall(v))
                self.assertTrue(xcheck.verdict(costs, 100)["reached"])
                # At the cheapest cost itself the target is reached (>=, not >).
                lowest = min(x["mlwe_min_rop_log2"], x["msis_min_rop_log2"])
                self.assertTrue(xcheck.verdict(costs, lowest)["reached"])
                self.assertFalse(xcheck.verdict(costs, lowest + 0.01)["reached"])

    def test_an_unpinned_checkout_is_refused_unless_allowed(self):
        # What git reports is mocked. The target check gates a set, so it refuses a checkout
        # that is not at the pinned commit or has local changes; the recording modes only
        # record that state.
        pinned = xcheck.LATTICE_ESTIMATOR_COMMIT
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {ESTIMATOR_ENV: tmp}):
            for head, status, refused in ((pinned, "", False), ("0" * 40, "", True),
                                          (pinned, " M estimator/lwe.py", True),
                                          (pinned, "?? estimator/extra.py", True),
                                          ("", "", True)):
                def fake_run(cmd, **kwargs):
                    out = head if "rev-parse" in cmd else status
                    return subprocess.CompletedProcess(cmd, 0, stdout=out + "\n", stderr="")
                with self.subTest(head=head[:7], status=status), \
                        mock.patch.object(xcheck.subprocess, "run", fake_run), \
                        mock.patch.object(xcheck, "_sage", lambda: None):
                    meta = {"commit": head, "pinned": head == pinned, "dirty": bool(status)}
                    self.assertEqual(xcheck._checkout(ESTIMATOR_ENV, pinned), (tmp, meta))
                    if not refused:
                        self.assertEqual(xcheck._checkout(ESTIMATOR_ENV, pinned, False)[1],
                                         meta)
                        continue
                    for call in (lambda: xcheck._checkout(ESTIMATOR_ENV, pinned, False),
                                 lambda: xcheck.rough_check({}, 128)):
                        with self.assertRaises(xcheck.Unavailable) as e:
                            call()
                        self.assertIn("--allow-unpinned-estimator", str(e.exception))
                    self.assertIn("allow-unpinned",
                                  xcheck.available("lattice-estimator", allow_unpinned=False))
                    self.assertIsNone(xcheck.available("lattice-estimator"))

    def test_the_estimators_printout_goes_to_stderr(self):
        # A stand-in for lattice-estimator that prints its results as the real one does
        # (Logging.print); stdout must carry only the tool's JSON.
        class Estimate:
            def __call__(self, params):
                print("usvp                 :: rop: ≈2^100.0, red: ≈2^100.0, β: 343")
                return {"usvp": {"rop": 2.0 ** 100, "beta": 343}}

            def rough(self, params):
                return self(params)

        class Problem:
            estimate = Estimate()

            @staticmethod
            def Parameters(**kwargs):
                return kwargs
        fake = types.ModuleType("estimator")
        fake.LWE = fake.SIS = Problem
        fake.ND = types.SimpleNamespace(Uniform=lambda a, b: (a, b))
        sage, sage_all = types.ModuleType("sage"), types.ModuleType("sage.all")
        sage_all.RR, sage_all.version = float, lambda: "stand-in"
        modules = {"estimator": fake, "sage": sage, "sage.all": sage_all}
        meta = {"commit": xcheck.LATTICE_ESTIMATOR_COMMIT, "pinned": True, "dirty": False}
        inst = {"mlwe": {"n": 64, "q": 5, "m": 64},
                "msis": {"n": 64, "q": 5, "m": 128, "log2_bound": 1.0}}
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(xcheck, "_sage", lambda: None), \
                mock.patch.object(xcheck, "_checkout", lambda *a: ("stand-in", meta)), \
                mock.patch.object(sys, "path", list(sys.path)), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            v = xcheck.rough_check(inst, 100)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(err.getvalue().count("usvp"), 2)
        self.assertTrue(v["reached"])
        self.assertEqual(v["lattice_estimator"], meta)


@unittest.skipIf(ESTIMATOR, f"lattice-estimator unavailable: {ESTIMATOR}")
class LatticeEstimator(unittest.TestCase):
    def test_toy_d64_rough(self):
        # usvp 2^105.7 (beta 362), dual_hybrid 2^104.3, SIS 2^101.3 (beta 347) in the rough
        # model; the instances are those of toy-d64.
        inst = {"mlwe": {"n": 26 * 64, "q": 1099511627917, "m": 30 * 64},
                "msis": {"n": 16 * 64, "q": 1099511627917, "m": 68 * 64,
                         "log2_bound": 32.1974936500626}}
        out = xcheck.lattice_estimator(inst)
        self.assertTrue(out["lattice_estimator"]["pinned"], out["lattice_estimator"])
        self.assertAlmostEqual(out["mlwe"]["usvp"]["rop"], 105.7, delta=0.1)
        self.assertEqual(out["mlwe"]["usvp"]["beta"], 362)
        self.assertAlmostEqual(out["mlwe"]["dual_hybrid"]["rop"], 104.3, delta=0.1)
        self.assertAlmostEqual(out["msis"]["lattice"]["rop"], 101.3, delta=0.1)
        self.assertEqual(out["msis"]["lattice"]["beta"], 347)

    def test_beta_min_is_not_the_rough_estimate(self):
        # beta_min bounds the uSVP estimate's block size (any number of samples) and the MSIS
        # closed form's; the estimator's dual-hybrid attack on the proof's actual samples is
        # cheaper. At
        # beta_min 439 kyber1024-d64's MLWE instance is at 2^127.6 (beta 437). A larger
        # beta_min is no fix by itself: the search re-optimizes q and the ranks, and 440
        # reaches 2^128 while 441 does not.
        r = json.loads((common.TOOL / "sets" / "kyber1024-d64.request.json").read_text())
        results = {}
        for beta_min in (439, 440, 441, 444):
            r["hardness"] = {"policy": "beta", "beta_min": beta_min}
            _, rep = cli.derive_request(json.dumps(r), what_if=False, compare=False)
            results[beta_min] = xcheck.rough_check(cli.instances(rep), 128)
        v = results[439]
        self.assertEqual((v["cheapest"]["mlwe"]["attack"], v["cheapest"]["mlwe"]["beta"]),
                         ("dual_hybrid", 437))
        self.assertAlmostEqual(v["cheapest"]["mlwe"]["rop_log2"], 127.61, delta=0.01)
        self.assertTrue(v["cheapest"]["msis"]["reached"])
        self.assertEqual({b: x["reached"] for b, x in results.items()},
                         {439: False, 440: True, 441: False, 444: True})

    def test_targets_on_the_command_line(self):
        run = run_tool
        with tempfile.TemporaryDirectory() as tmp:
            r = json.loads((common.TOOL / "sets" / "kyber1024-d64.request.json").read_text())
            f = Path(tmp) / "r.json"
            for beta_min, code in ((444, 0), (439, 2)):
                r["hardness"] = {"policy": "beta", "beta_min": beta_min}
                f.write_text(json.dumps(r))
                p, rep = Path(tmp) / f"{beta_min}.json", Path(tmp) / f"{beta_min}.report.json"
                res = run("derive", f, "--no-what-if", "--estimator-target", 128,
                          "--params", p, "--report", rep)
                with self.subTest(beta_min=beta_min):
                    self.assertEqual(res.returncode, code, res.stderr)
                    if code:
                        self.assertIn("MLWE dual_hybrid 2^127.61 (beta 437)", res.stderr)
                        self.assertFalse(p.exists())  # no set below the target is written
                    else:
                        self.assertTrue(json.loads(rep.read_text())["estimator_check"]["reached"])
            # Without --params the set goes to stdout, and only the set: the estimator's own
            # lines go to stderr.
            r["hardness"] = {"policy": "beta", "beta_min": 444}
            f.write_text(json.dumps(r))
            res = run("derive", f, "--no-what-if", "--estimator-target", 128)
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertEqual(jsonio.load_params(res.stdout)["id"], r["id"])
            self.assertIn("dual_hybrid", res.stderr)
        # The shipped set, under the delta policy, is near 2^100.6 in the rough model.
        res = run("xcheck", "kyber1024-d64", "--target", 100)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertTrue(json.loads(res.stdout)["reached"])  # stdout is the verdict alone
        self.assertIn("dual_hybrid", res.stderr)
        res = run("xcheck", "kyber1024-d64", "--target", 128)
        self.assertEqual(res.returncode, 1)
        self.assertIn("below the target 2^128", res.stderr)
        self.assertFalse(json.loads(res.stdout)["reached"])

    def test_an_unpinned_estimator_needs_consent(self):
        # A copy of the estimator outside git is not the pinned checkout: the target checks
        # refuse it, before deriving, unless allowed; then the verdict records it.
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(Path(os.environ[ESTIMATOR_ENV]) / "estimator",
                            Path(tmp) / "estimator")
            env = dict(os.environ, **{ESTIMATOR_ENV: tmp})
            res = run_tool("xcheck", "kyber1024-d64", "--target", 100, env=env)
            self.assertEqual(res.returncode, 2, res.stderr)
            self.assertIn("not the pinned commit", res.stderr)
            res = run_tool("derive", common.TOOL / "sets" / "kyber1024-d64.request.json",
                           "--no-what-if", "--estimator-target", 100, env=env)
            self.assertEqual((res.returncode, res.stdout), (2, ""), res.stderr)
            self.assertIn("--estimator-target needs the lattice-estimator", res.stderr)
            res = run_tool("xcheck", "kyber1024-d64", "--target", 100,
                           "--allow-unpinned-estimator", env=env)
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertFalse(json.loads(res.stdout)["lattice_estimator"]["pinned"])

    def test_new_sets_rough(self):
        for name in common.NEW_SETS:
            rep = json.loads((common.TOOL / "sets" / f"{name}.report.json").read_text())
            h = rep["hardness"]
            inst = {"mlwe": {"n": h["mlwe"]["n"], "q": int(h["mlwe"]["q"]),
                             "m": h["mlwe"]["samples_m_for_cross_checks"]},
                    "msis": {"n": h["msis"]["n"], "q": int(h["msis"]["q"]), "m": h["msis"]["m"],
                             "log2_bound": h["msis"]["log2_bound"]}}
            out = xcheck.lattice_estimator(inst)
            with self.subTest(set=name):
                # Under the delta policy the rough (core-SVP) figures sit near 100 bits.
                self.assertGreater(out["mlwe_min_rop_log2"], 100.0)
                self.assertGreater(out["msis_min_rop_log2"], 100.0)
                self.assertGreaterEqual(out["msis"]["lattice"]["beta"], 346)


if __name__ == "__main__":
    unittest.main()
