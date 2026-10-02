"""The command line: the positional form reproduces every checked-in set and fixture byte for
byte, the subcommands work, and the generated sets regenerate exactly."""
import contextlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import common
from jali_params import cli, hardness, jsonio, rustcheck, xcheck

TOOL = common.TOOL / "lnp_params.py"


def run(*args, cwd=None):
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)], capture_output=True,
                          text=True, cwd=cwd)


class PositionalForm(unittest.TestCase):
    def test_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp:
            for req, rep, out in common.POSITIONAL:
                res = run(common.JALI / req, common.JALI / rep, "--output", Path(tmp) / "o.json")
                self.assertEqual(res.returncode, 0, res.stderr)
                self.assertEqual((Path(tmp) / "o.json").read_text(), common.read(out), out)

    def test_the_toy_report_is_the_tools_rank(self):
        # The MLWE report of toy-d64 and of the possession fixture, as `rank --mlwe-report`
        # writes it for their ring.
        res = run("rank", "--degree", 64, "--q", 1099511627917, "--mlwe-report")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stdout, common.read("tools/params/toy-d64.report.json"))
        report = json.loads(res.stdout)
        self.assertEqual((report["rank"], report["delta"]),
                         hardness.rank(64, 1099511627917, hardness.Policy("delta"))[::2])

    def test_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = "tests/fixtures/params/tag-preimage-crt-d128-no-divisor"
            res = run(common.JALI / f"{f}.request.json", common.JALI / f"{f}.report.json",
                      "--output", Path(tmp) / "r.json")
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("no suitable divisor", res.stderr)

    def test_lambda_near_2_64(self):
        # q1 = 2^64 - 899: Rust's lambda is 2 (log2 of the nearest f64 is 64.0). With the exact
        # logarithm the original form wrote an m2 that Rust refuses ("simulatability rank").
        q = 2 ** 64 - 899
        shipped = jsonio.load_params((common.TOOL / "sets" / "kyber1024-d64.json").read_text())
        req = {k: shipped[k] for k in ("m1", "l", "alpha_squared", "n_bin", "l2_rows",
                                       "l2_bounds_squared", "n_prime", "linf_bound")}
        req.update(id="test-lambda-edge", prime_factors=[q], degree=64)
        k, _, delta = hardness.rank(64, q, hardness.Policy("delta"))
        report = {"degree": 64, "modulus": q, "rank": k, "delta": delta,
                  "provenance": "test: the tool's uSVP estimate at this q"}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "req.json").write_text(json.dumps(req))
            (Path(tmp) / "rep.json").write_text(json.dumps(report))
            res = run(Path(tmp) / "req.json", Path(tmp) / "rep.json", "--output",
                      Path(tmp) / "out.json")
            self.assertEqual(res.returncode, 0, res.stderr)
            out = jsonio.load_params((Path(tmp) / "out.json").read_text())
        derived, _ = rustcheck.check_params(out)
        self.assertEqual(derived["lambda"], 2)

    def test_subcommand_names_agree(self):
        text = TOOL.read_text()
        names = set(re.findall(r"'(\w+)'", re.search(r"sys\.argv\[1\] in \(([^)]*)\)",
                                                      text).group(1)))
        self.assertEqual(names, cli.SUBCOMMANDS)


class Subcommands(unittest.TestCase):
    def test_regenerate_check(self):
        res = run("regenerate", "--check")
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        for name in common.NEW_SETS:
            self.assertIn(f"{name}: params identical, crate copy identical, report identical",
                          res.stdout)

    def test_regenerate_check_detects_a_change(self):
        # The failure path of `regenerate --check`, on copies of the sets: an altered parameter
        # file, an altered crate copy, then an altered report, each make it name the difference
        # and exit 1.
        old_sets, old_crate = cli.SETS, cli.CRATE_SETS
        with tempfile.TemporaryDirectory() as tmp:
            sets, crate = Path(tmp) / "sets", Path(tmp) / "crate"
            shutil.copytree(common.TOOL / "sets", sets)
            shutil.copytree(cli.CRATE_SETS, crate)
            cli.SETS, cli.CRATE_SETS = sets, crate
            try:
                good = (sets / "kyber1024-d128.json").read_text()
                bad = json.loads(good)
                bad["d_bits"] += 1
                (sets / "kyber1024-d128.json").write_text(json.dumps(bad, indent=2) + "\n")
                code, out = self.regenerate_check("kyber1024-d128")
                self.assertEqual(code, 1, out)
                # The crate's copy must equal the tool's, which no longer matches it.
                self.assertIn("kyber1024-d128: params DIFFER, crate copy DIFFERS from the tool "
                              "copy, report identical", out)
                (sets / "kyber1024-d128.json").write_text(good)
                (crate / "kyber1024-d128.json").write_text(good.replace("\n", "\r\n"))
                code, out = self.regenerate_check("kyber1024-d128")
                self.assertEqual(code, 1, out)
                self.assertIn("kyber1024-d128: params identical, crate copy DIFFERS from the "
                              "tool copy, report identical", out)
                (crate / "kyber1024-d128.json").write_text(good)
                rep = json.loads((sets / "kyber1024-d128.report.json").read_text())
                rep["summary"]["estimated_proof_bytes"] += 1
                (sets / "kyber1024-d128.report.json").write_text(json.dumps(rep, indent=2) + "\n")
                code, out = self.regenerate_check("kyber1024-d128")
                self.assertEqual(code, 1, out)
                self.assertIn("kyber1024-d128: params identical, crate copy identical, report "
                              "DIFFERS in summary", out)
            finally:
                cli.SETS, cli.CRATE_SETS = old_sets, old_crate

    def regenerate_check(self, name):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                cli.main(["regenerate", "--check", name])
                code = 0
            except SystemExit as e:
                code = e.code
        return code, buf.getvalue()

    def test_check(self):
        for name in common.NEW_SETS:
            res = run("check", common.TOOL / "sets" / f"{name}.json", "--request",
                      common.TOOL / "sets" / f"{name}.request.json")
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            out = json.loads(res.stdout)
            self.assertTrue(out["accepted"])
            self.assertEqual(out["tight"], [])
        res = run("check", common.JALI / "src/params/sets/toy-d64.json")
        self.assertEqual(json.loads(res.stdout)["estimated_proof_bytes"], 17896)

    def test_derive_to_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = run("derive", common.TOOL / "sets" / "kyber1024-d128.request.json", "--params",
                      Path(tmp) / "p.json", "--report", Path(tmp) / "r.json")
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertEqual((Path(tmp) / "p.json").read_text(),
                             (common.TOOL / "sets" / "kyber1024-d128.json").read_text())

    def test_prime_and_rank(self):
        res = run("prime", "--min", 2 ** 100, "--gamma0", 2 ** 20)
        out = json.loads(res.stdout)
        q = int(out["q"])
        self.assertTrue(q >= 2 ** 100 and q % 8 == 5 and (q - 1) % out["gamma"] == 0)
        self.assertIn("probable prime", out["primality"])
        res = run("rank", "--degree", 64, "--q", 2 ** 40)
        self.assertEqual(json.loads(res.stdout)["rank"], 26)
        res = run("rank", "--degree", 64, "--q", 2 ** 40, "--policy", "beta", "--beta-min", 439)
        self.assertEqual(json.loads(res.stdout)["rank"], 30)

    def test_errors_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = json.loads((common.TOOL / "sets" / "demo.request.json").read_text())
            r["hardness"] = {"policy": "beta", "beta_min": 100}
            f = Path(tmp) / "bad.request.json"
            f.write_text(json.dumps(r))
            res = run("derive", f)
            self.assertEqual(res.returncode, 2)
            self.assertIn("beta_min", res.stderr)

    def test_malformed_requests_exit_without_a_traceback(self):
        base = json.loads((common.TOOL / "sets" / "kyber1024-d64.request.json").read_text())
        no_rows = json.loads(json.dumps(base))
        del no_rows["statement"]["rows"]
        typo = json.loads(json.dumps(base))
        typo["statement"]["blocks"][0]["nrom"] = typo["statement"]["blocks"][0].pop("norm")
        # gamma 0 must be refused, not divided by.
        gamma0 = dict(base, modulus={"prime_factors": [2423991946973], "gamma": 0})
        with tempfile.TemporaryDirectory() as tmp:
            for name, r in (("rows", no_rows), ("nrom", typo), ("gamma", gamma0)):
                f = Path(tmp) / "bad.request.json"
                f.write_text(json.dumps(r))
                for args in (("derive", f, "--no-what-if"),
                             ("check", common.TOOL / "sets" / "kyber1024-d64.json",
                              "--request", f)):
                    with self.subTest(case=name, command=args[0]):
                        res = run(*args)
                        self.assertEqual(res.returncode, 2, res.stderr)
                        self.assertNotIn("Traceback", res.stderr)
                        self.assertIn(name, res.stderr)

    @unittest.skipIf(xcheck.available("lattice-estimator") is None,
                     "the lattice-estimator is available; test_xcheck runs the check itself")
    def test_estimator_target_needs_the_estimator(self):
        # Asked for and unable to run, the check refuses before deriving; it never passes
        # silently.
        res = run("derive", common.TOOL / "sets" / "kyber1024-d64.request.json",
                  "--estimator-target", 128, "--no-what-if")
        self.assertEqual(res.returncode, 2)
        self.assertIn("--estimator-target needs the lattice-estimator", res.stderr)
        self.assertEqual(res.stdout, "")

    def test_allow_unpinned_goes_with_target(self):
        res = run("xcheck", "kyber1024-d64", "--allow-unpinned-estimator")
        self.assertEqual(res.returncode, 1)
        self.assertIn("--allow-unpinned-estimator applies to --target only", res.stderr)
        self.assertEqual(res.stdout, "")


class Moduli(unittest.TestCase):
    def test_default_output_and_rust_table(self):
        rust = [[int(a), int(b)] for a, b in
                re.findall(r"\((\d+), (\d+)\)", common.read("src/params/moduli.rs"))]
        base = subprocess.run([sys.executable, str(common.TOOL / "moduli.py")],
                              capture_output=True, text=True)
        self.assertEqual(json.loads(base.stdout), rust[:4])
        full = subprocess.run([sys.executable, str(common.TOOL / "moduli.py"), "--count",
                               str(len(rust))], capture_output=True, text=True)
        self.assertEqual(json.loads(full.stdout), rust)


if __name__ == "__main__":
    unittest.main()
