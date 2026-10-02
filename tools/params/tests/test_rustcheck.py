"""The emulation of the Rust checks against values Rust produced, and its refusals.

`tests/data/rust_reference.json` and `rust_check_cases.json` hold Rust's results; regenerate
them with `python3 tools/params/rust_reference.py` (needs cargo) whenever the crate's checks
or the sets change.
"""
import copy
import hashlib
import json
import re
import unittest

import mpmath as mp

import common
import rust_reference
from jali_params import cli, derive, jsonio, request, rustcheck

REF = rust_reference.load(common.DATA / "rust_reference.json")
CASES = rust_reference.load(common.DATA / "rust_check_cases.json")
REGENERATE = "rerun tools/params/rust_reference.py"
# f64 outputs of libm functions are compared exactly on the platform that recorded them.
EXACT = CASES["provenance"]["platform"] == rust_reference.PLATFORM


def toy():
    return jsonio.load_params(common.read("src/params/sets/toy-d64.json"))


def message(rust_error):
    """The emulation's name for a Rust error's Debug form."""
    m = re.fullmatch(r'Parameter\("(.*)"\)', rust_error)
    return m.group(1) if m else rust_error.lower()


def assert_checked(test, derived, checks, rust):
    """Rust's CheckedParams equal the emulation's values; its f64 outputs bit for bit on the
    recording platform, within 4 ULP on others (`rust_reference.f64_close`)."""
    for key, value in rust["checked"].items():
        test.assertEqual(derived[key], value, key)
    for key, value in rust["f64"].items():
        test.assertTrue(rust_reference.f64_close(checks.floats[key], value, EXACT),
                        (key, checks.floats[key], value))


class AgainstRust(unittest.TestCase):
    def test_toy_d64_pins(self):
        derived, _ = rustcheck.check_params(toy())
        pin = REF["toy-d64"]
        for key in ("b_squared", "z3_bound_squared", "z4_bound", "estimated_proof_bytes"):
            self.assertEqual(derived[key], pin[key], key)
        self.assertEqual((derived["lambda"], derived["l_ext"], derived["n_ex"]), (4, 12, 7))
        self.assertEqual((derived["range_rejection_M3"], derived["range_rejection_M4"]), (2, 2))

    def test_new_sets_checked_values(self):
        self.assertEqual(sorted(REF["sets"]), sorted(common.NEW_SETS))
        for name, ref in REF["sets"].items():
            text = (common.TOOL / "sets" / f"{name}.json").read_text()
            with self.subTest(set=name):
                self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), ref["params_sha256"],
                                 f"the set changed: {REGENERATE}")
                _, derived, checks = rustcheck.from_json(text)
                assert_checked(self, derived, checks,
                               {"checked": ref["checked"], "f64": ref["checked_f64"]})
                self.assertEqual(checks.tight(), [])
                rustcheck.prover_checks(derived)

    def test_worst_case_requirements(self):
        for name, ref in REF["sets"].items():
            params = jsonio.load_params((common.TOOL / "sets" / f"{name}.json").read_text())
            req = rustcheck.lin_requirements(params["degree"], ref["checked"]["q"], 256,
                                             ref["statement_modulus"], 4,
                                             [("w", 8, ("l2_squared", ref["bound_squared"]))])
            with self.subTest(set=name):
                self.assertEqual(req, ref["requirements"])
                self.assertEqual((ref["worst_case_compile"], ref["worst_case_lin_compile"]),
                                 ("ok", "ok"))
                rustcheck.check_compile(params, req, ref["statement_modulus"])

    def test_rust_linf_bound_is_the_quotient_bound_of_f(self):
        # The identity that request.py checks on copied requirements: with every lifted
        # constraint at the statement modulus p, linf_bound = max(1, ceil(F/p)).
        for name, ref in REF["sets"].items():
            req, p = ref["requirements"], ref["statement_modulus"]
            with self.subTest(set=name):
                self.assertGreater(req["n_prime"], 0)
                self.assertEqual(req["linf_bound"],
                                 max(1, -(-req["max_integer_coefficient"] // p)))

    def test_previous_prime_fails_exactly_the_lifting_bound(self):
        for name, ref in REF["sets"].items():
            params = jsonio.load_params((common.TOOL / "sets" / f"{name}.json").read_text())
            before = ref["previous_prime_same_gamma_class"]
            prev = dict(params, prime_factors=[before["q"]])
            with self.subTest(set=name):
                derived, _ = rustcheck.check_params(prev)
                for key, value in before["checked"].items():
                    self.assertEqual(derived[key], value, key)
                with self.assertRaises(rustcheck.RustCheckError) as e:
                    rustcheck.check_compile(prev, ref["requirements"], ref["statement_modulus"])
                self.assertEqual(e.exception.rust_message, "modulus lifting bound")
                self.assertEqual(message(before["worst_case_compile"]), "modulus lifting bound")
                self.assertEqual(message(before["worst_case_lin_compile"]),
                                 "modulus lifting bound")

    def test_random_instances_proved(self):
        for name, ref in REF["sets"].items():
            r = ref["random_instance"]
            with self.subTest(set=name):
                self.assertTrue(r["verified"] and r["tampered_rejected"]
                                and r["other_context_rejected"])
                self.assertLess(r["proof_bytes"], ref["checked"]["estimated_proof_bytes"])


class AgainstRustCases(unittest.TestCase):
    """TboxParams::from_json and lin::compile on 264 perturbed sets (every refusal, JSON
    decoding, moduli up to 255 bits, the hint allowance at small gamma, seeded random
    mutations), as Rust decided them."""

    def test_the_hint_allowance_is_rusts(self):
        # The five cases at small gamma have sigma_2/gamma from 1.001 to 18.8, where the
        # allowance of params::hint_bits exceeds 2.25 bits: Rust accepted them, and its
        # estimate equals the emulation's.
        est = [c for c in CASES["cases"] if c["name"].startswith("est/")]
        self.assertEqual(len(est), 5)
        for c in est:
            with self.subTest(case=c["name"]):
                text = rust_reference.case_text(CASES["bases"][c["base"]], c)
                params, derived, _ = rustcheck.from_json(text)
                sigma2 = 1.55 * 2.0 ** params["log_sigma"][1]
                self.assertGreater(rustcheck.hint_bits(sigma2, params["gamma"]), 2.25)
                self.assertEqual(derived["estimated_proof_bytes"],
                                 c["rust"]["checked"]["estimated_proof_bytes"])

    def test_case_texts_are_the_ones_rust_saw(self):
        for c in CASES["cases"]:
            text = rust_reference.case_text(CASES["bases"][c["base"]], c)
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), c["sha256"], c["name"])

    def test_the_emulation_decides_every_case_as_rust(self):
        outcomes = set()
        for c in CASES["cases"]:
            text = rust_reference.case_text(CASES["bases"][c["base"]], c)
            rust = c["rust"]
            with self.subTest(case=c["name"]):
                if "error" in rust:
                    with self.assertRaises(rustcheck.RustCheckError) as e:
                        rustcheck.from_json(text)
                    self.assertEqual(e.exception.rust_message, message(rust["error"]))
                    outcomes.add(e.exception.rust_message)
                    continue
                params, derived, checks = rustcheck.from_json(text)
                assert_checked(self, derived, checks, rust)
                outcomes.add("ok")
                if "compile" in rust:
                    st = c["statement"]
                    req = rustcheck.lin_requirements(params["degree"], derived["q"], 256, st["p"],
                                                     4, [("w", 8, ("l2_squared", st["bsq"]))])
                    try:
                        rustcheck.check_compile(params, req, st["p"])
                        got = "ok"
                    except rustcheck.RustCheckError as e:
                        got = e.rust_message
                    want = "ok" if rust["compile"] == "ok" else message(rust["compile"])
                    self.assertEqual(got, want, "compile")
                    outcomes.add("compile: " + got)
        # Every refusal of check (reachable within the width cap) and of compile occurs.
        for m in ("encoding", "overflow", "parameter provenance", "proof modulus factors",
                  "proof degree", "compression", "compressed commitment bit width",
                  "exact norm blocks", "dimensions", "approximate block",
                  "Gaussian width capacity",
                  "binary, exact-norm and range blocks require a prime modulus",
                  "completeness dimensions", "simulatability rank", "MLWE estimate",
                  "compressed response bound capacity", "MSIS estimate", "rounding bound",
                  "range rejection M", "ARP modulus bound", "compile: ok",
                  "compile: bounded witness norm budget", "compile: compiled witness dimensions",
                  "compile: carry or quotient bound", "compile: lifting slack",
                  "compile: modulus lifting bound", "compile: dimension",
                  "compile: exact block bound"):
            self.assertIn(m, outcomes)

    def test_data_matches_the_current_sets(self):
        for name in common.NEW_SETS:
            text = (common.TOOL / "sets" / f"{name}.json").read_text()
            self.assertEqual(CASES["bases"][name], json.loads(text), f"{name}: {REGENERATE}")

    def test_cases_are_the_recorded_ones(self):
        # rust_reference.cases() still defines exactly the recorded case texts, including the
        # ones it stores as raw text because they hold a JSON number of 2^64 or more.
        bases, cases = rust_reference.cases()
        self.assertEqual([c["name"] for c in cases], [c["name"] for c in CASES["cases"]])
        for c, old in zip(cases, CASES["cases"]):
            text = rust_reference.case_text(bases[c["base"]], c)
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), old["sha256"], c["name"])
        raw = {c["name"] for c in cases if "raw" in c}
        self.assertLessEqual({"enc/prime-number-2^64", "enc/gamma-number-2^64",
                              "chk/factors-wide-prime", "chk/factors-wide-composite"}, raw)


class AgainstRustStatements(unittest.TestCase):
    """The statement kinds of `rust_reference.py`'s verifier: constraints with two moduli,
    variables bounded in l-infinity, a block placed in BDLOP, 65 constant-coefficient clauses
    whose carries fill two packed polynomials, subring variables, variables bounded exactly by
    bits, and the refusals of their compilation. Rust exported their requirements and blocks,
    the tool derived a set for each of the first six, Rust compiled it and proved and verified a
    seeded instance, and Rust decided every mutation of those sets (`statement_cases`)."""

    def read(self, kind):
        entry = REF["statements"][kind]
        _, _, _, req, lifting, _ = request.read(json.dumps(entry["request"]))
        return entry, req, lifting

    def test_the_requests_read_rusts_exports(self):
        for kind, entry in REF["statements"].items():
            with self.subTest(kind=kind):
                _, req, lifting = self.read(kind)
                exported = entry["requirements"]
                for key in request.REQUIREMENTS_KEYS:
                    want = exported[key]
                    want = [jsonio.decode_int(x) for x in want] if isinstance(want, list) \
                        else jsonio.decode_int(want)
                    self.assertEqual(req[key], want, key)
                # The per-slot bound, where Rust wrote one, reaches the derivation.
                self.assertEqual(req.get("approx_alpha_squared"),
                                 jsonio.decode_int(exported["approx_alpha_squared"])
                                 if "approx_alpha_squared" in exported else None)
                moduli = exported.get("lifted_moduli", [])
                if moduli:
                    self.assertEqual(lifting["pairs"], [
                        [jsonio.decode_int(m["modulus"]),
                         jsonio.decode_int(m["max_integer_coefficient"])] for m in moduli])
                self.assertEqual(lifting["has_linf"], "linf" in exported)
        statements = REF["statements"]
        self.assertEqual([m["modulus"] for m in
                          statements["two-moduli"]["requirements"]["lifted_moduli"]],
                         [12, 13, 156])
        self.assertIsNone(statements["linf"]["requirements"]["linf"]["extraction_limit"])
        self.assertEqual(statements["linf-limit"]["requirements"]["linf"]["extraction_limit"],
                         504)
        self.assertEqual([(b["name"], b["placement"]) for b in statements["placed"]["blocks"]],
                         [("w", "bdlop"), ("s", "ajtai")])
        # 65 clauses at degree 64: two packed carry polynomials, each a message with a range
        # row; kinds with a clause or rows of unequal bounds carry the per-slot bound.
        packed = statements["packed"]["requirements"]
        self.assertEqual((packed["l"], packed["n_prime"]), (2, 2))
        self.assertEqual(sorted(k for k, e in statements.items()
                                if "approx_alpha_squared" in e["requirements"]),
                         ["linf", "linf-exact", "linf-limit", "noninvertible-constraint",
                          "noninvertible-statement", "packed", "subring", "two-moduli"])
        # Subring blocks commit one component each at degree 64; exact blocks their bits,
        # which are binary rows, with no linf section.
        self.assertEqual([b["rows"] for b in statements["subring"]["blocks"]], [8, 1, 1])
        exact = statements["linf-exact"]
        self.assertEqual([(b["rows"], b["norm"]) for b in exact["blocks"]],
                         [(8, {"l2_squared": 64}), (4, {"linf_exact": 1}),
                          (4, {"linf_exact": 5})])
        self.assertEqual(exact["requirements"]["n_bin"], 8)
        self.assertNotIn("linf", exact["requirements"])

    def test_the_tool_derives_the_recorded_sets(self):
        derived = [k for k, e in REF["statements"].items() if "params" in e]
        self.assertEqual(derived, ["two-moduli", "linf", "placed", "packed", "subring",
                                   "linf-exact"])
        for kind in derived:
            entry = REF["statements"][kind]
            with self.subTest(kind=kind):
                text, rep = cli.derive_request(json.dumps(entry["request"]), what_if=False,
                                               compare=False)
                self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),
                                 entry["params_sha256"], f"{kind}: {REGENERATE}")
                self.assertEqual(json.loads(text), entry["params"])
                self.assertEqual(entry["compile"], "ok")
                r = entry["random_instance"]
                self.assertTrue(r["verified"] and r["tampered_rejected"]
                                and r["other_context_rejected"])
        # The search kept w in the BDLOP part, where Rust compiled it.
        placed = REF["statements"]["placed"]
        self.assertEqual((placed["placement"]["bdlop"], placed["compiled_kind"]), (["w"], "placed"))

    def test_the_emulation_decides_every_case_as_rust(self):
        outcomes = set()
        for c in REF["statement_cases"]:
            _, req, lifting = self.read(c["kind"])
            text = rust_reference.case_text(REF["statements"][c["base"]]["params"], c)
            with self.subTest(case=c["name"]):
                self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), c["sha256"])
                try:
                    params, _, _ = rustcheck.from_json(text)
                    rustcheck.check_compile(params, req, lifting["statement_modulus"],
                                            lifting=lifting)
                    got = "ok"
                except rustcheck.RustCheckError as e:
                    got = e.rust_message
                self.assertEqual(got, "ok" if c["rust"] == "ok" else message(c["rust"]))
                outcomes.add(got)
        # Every refusal the new requirements bring, and the older ones of compile, occur.
        for m in ("ok", "noninvertible statement modulus", "noninvertible constraint modulus",
                  "variable range above statement modulus", "approximate range bound",
                  "approximate extraction bound", "carry or quotient bound",
                  "modulus lifting bound", "bounded witness norm budget",
                  "compiled witness dimensions"):
            self.assertIn(m, outcomes)

    def test_the_tool_refuses_what_rust_refuses_at_every_modulus(self):
        entry = REF["statements"]["linf-limit"]
        self.assertTrue(entry["tool"].startswith("variable range above statement modulus"),
                        entry["tool"])
        rust = [c["rust"] for c in REF["statement_cases"] if c["kind"] == "linf-limit"]
        self.assertEqual([message(x) for x in rust], ["variable range above statement modulus"])


class DataEncoding(unittest.TestCase):
    """The data files follow the number rule, and the reader accepts both spellings."""

    def test_either_spelling_reads_the_same(self):
        q = 2 ** 64 + 13
        number = {"cases": [{"name": "x", "base": "b", "set": {"prime_factors": [q]},
                             "sha256": "s", "rust": {"checked": {"q": q, "lambda": 2}}}]}
        string = {"cases": [{"name": "x", "base": "b", "raw": "{}", "sha256": "s",
                             "rust": {"checked": {"q": str(q), "lambda": 2}}}]}
        a = rust_reference._comparable(rust_reference.decode(copy.deepcopy(number)))
        b = rust_reference._comparable(rust_reference.decode(copy.deepcopy(string)))
        self.assertEqual(a, b)
        self.assertEqual(b["cases"][0]["rust"]["checked"]["q"], q)
        with self.assertRaises(ValueError):
            rust_reference.decode({"cases": [{"rust": {"checked": {"q": "0x10"}}}]})

    def test_written_integers_follow_the_rule(self):
        checked, _ = rust_reference._checked({"q_hex": "1" + "0" * 16, "lambda": 2,
                                              "b_squared": str(2 ** 100), "arp_bound": "1.0",
                                              "msis_delta": "1.0"})
        self.assertEqual(checked, {"q": str(2 ** 64), "lambda": 2, "b_squared": str(2 ** 100)})

    def test_check_names_what_differs(self):
        # `rust_reference.py --check` names each statement case, set and statement kind that
        # differs, not only the key of the data file.
        old = copy.deepcopy(REF)
        new = copy.deepcopy(REF)
        name = new["statement_cases"][3]["name"]
        new["statement_cases"][3]["rust"] = 'Parameter("changed")'
        new["statement_cases"].append(dict(new["statement_cases"][0], name="extra"))
        new["sets"]["demo"]["random_instance"]["proof_bytes"] += 1
        new["statements"]["linf"]["compile"] = "changed"
        self.assertEqual(rust_reference.differences("statement_cases", new["statement_cases"],
                                                    old["statement_cases"], True),
                         [f"statement case {name} differs", "statement case extra is new"])
        self.assertEqual(rust_reference.differences("sets", new["sets"], old["sets"], True),
                         ["sets demo differs in random_instance"])
        self.assertEqual(rust_reference.differences("statements", new["statements"],
                                                    old["statements"], True),
                         ["statements linf differs in compile"])
        self.assertEqual(rust_reference.differences("toy-d64", 1, 2, True),
                         ["toy-d64 differs"])


class Refusals(unittest.TestCase):
    """Each check of TboxParams::check fires with its message under a perturbation."""

    def expect(self, message, capacity=rustcheck.CAPACITY, **changes):
        p = toy()
        p.update(changes)
        with self.assertRaises(rustcheck.RustCheckError) as e:
            rustcheck.check_params(p, capacity)
        self.assertEqual(e.exception.rust_message, message, changes)

    def test_messages(self):
        self.expect("parameter provenance", estimator="")
        self.expect("parameter provenance", id="x" * 257)
        self.expect("proof modulus factors", prime_factors=[1099511627917 + 8])
        self.expect("proof modulus factors", prime_factors=[1099511627917, 13])
        # 2^64 + 13 is prime and 5 mod 8: refused only under the narrow capacity (u64 factors).
        self.expect("proof modulus factors", rustcheck.NARROW_CAPACITY,
                    prime_factors=[(1 << 64) + 13])
        self.expect("compression", prime_factors=[(1 << 64) + 13])
        self.expect("proof modulus factors", prime_factors=[(1 << 64) + 21])
        self.expect("proof degree", degree=256)
        self.expect("compression", gamma=65203)
        self.expect("compression", gamma=8)
        self.expect("rounding bound", gamma=4)
        self.expect("exact norm blocks", l2_rows=[2, 0])
        self.expect("exact norm blocks", l2_bounds_squared=[128])
        self.expect("dimensions", m2=70000)
        self.expect("dimensions", alpha_squared=0)
        self.expect("approximate block", linf_bound=0)
        self.expect("Gaussian width capacity", log_sigma=[101, 12, 10, 8])
        self.expect("binary, exact-norm and range blocks require a prime modulus",
                    prime_factors=[13, 1099511627917], gamma=2, d_bits=0)
        self.expect("completeness dimensions", m1=0, l2_rows=[2], l2_bounds_squared=[128])
        self.expect("simulatability rank", mlwe_rank=25)
        self.expect("MLWE estimate", mlwe_delta=1.0045)
        self.expect("MSIS estimate", n_msis=10, m2=50, mlwe_rank=26)
        self.expect("range rejection M", log_sigma=[16, 12, 3, 8])

    def test_a_range_width_sized_from_linf_bound_alone_is_refused(self):
        # sigma_4 must cover the Euclidean bound sqrt(n'd) linf_bound of the range vector. Sized
        # from linf_bound alone, kyber1024-d64's rejection constant exceeds 2^16.
        p = jsonio.load_params((common.TOOL / "sets" / "kyber1024-d64.json").read_text())
        t = derive.rounded(5 * mp.sqrt(337) * p["linf_bound"])
        self.assertLess(t, p["log_sigma"][3])
        alpha_squared = p["n_prime"] * p["degree"] * p["linf_bound"] ** 2
        with self.assertRaises(rustcheck.RustCheckError) as e:
            rustcheck.range_rejection_constant(t, alpha_squared)
        self.assertEqual(e.exception.rust_message, "range rejection M")
        with self.assertRaises(rustcheck.RustCheckError) as e:
            rustcheck.check_params(dict(p, log_sigma=p["log_sigma"][:3] + [t]))
        self.assertEqual(e.exception.rust_message, "range rejection M")

    def test_lifting_refusals(self):
        p = toy()
        with self.assertRaises(rustcheck.RustCheckError) as e:
            rustcheck.check_lifting(p, 0, 5, 13)
        self.assertEqual(e.exception.rust_message, "carry or quotient bound")
        with self.assertRaises(rustcheck.RustCheckError) as e:
            rustcheck.check_lifting(p, 1 << 40, 4, 13)
        self.assertEqual(e.exception.rust_message, "modulus lifting bound")
        req = {"m1": 10, "l": 2, "alpha_squared": 5761}
        with self.assertRaises(rustcheck.RustCheckError) as e:
            rustcheck.check_compile(p, req, 13)
        self.assertEqual(e.exception.rust_message, "bounded witness norm budget")
        with self.assertRaises(rustcheck.RustCheckError) as e:
            rustcheck.check_compile(p, dict(req, alpha_squared=5760, l=3), 13)
        self.assertEqual(e.exception.rust_message, "compiled witness dimensions")

    def test_lifting_bound_of_2_256_or_more(self):
        # Rust compares in 1024 bits: a right-hand side above every q fails the lifting check,
        # not an overflow check.
        with self.assertRaises(rustcheck.RustCheckError) as e:
            rustcheck.check_lifting(toy(), 1 << 300, 4, 13)
        self.assertEqual(e.exception.rust_message, "modulus lifting bound")

    def test_field_types(self):
        # What JSON decoding refuses before check (Rust: Error::Encoding).
        for change in ({"gamma": 1 << 64}, {"alpha_squared": 1 << 64}, {"linf_bound": -1},
                       {"l2_bounds_squared": [128, 1 << 64]}, {"log_sigma": [16, 12, 10]},
                       {"log_sigma": [16, 12, 10, 1 << 32]}, {"d_bits": 1 << 32},
                       {"m1": True}, {"m1": 10.0}, {"prime_factors": [1 << 256]},
                       {"prime_factors": 1099511627917}, {"mlwe_delta": "1.0044"},
                       {"estimator": None}):
            with self.subTest(change=change):
                with self.assertRaises(rustcheck.RustCheckError) as e:
                    rustcheck.check_params(dict(toy(), **change))
                self.assertEqual(e.exception.rust_message, "encoding")

    def test_prover_constants(self):
        # A narrow sigma_2 passes check (it only shrinks the MSIS bound), but the prover's
        # Rej_2 constant exp(1/(2 g2^2)) overflows: the tool refuses such a set.
        p = dict(toy(), log_sigma=[16, 2, 10, 8])
        derived, _ = rustcheck.check_params(p)
        self.assertIsNone(derived["rejection_M2"])
        with self.assertRaises(rustcheck.RustCheckError) as e:
            rustcheck.prover_checks(derived)
        self.assertEqual(e.exception.rust_message, "rejection constant capacity")
        derived, _ = rustcheck.check_params(toy())
        rustcheck.prover_checks(derived)
        self.assertLess(max(derived["rejection_M1"], derived["rejection_M2"]), 65536)

    def test_unbounded_block_cannot_be_lifted(self):
        with self.assertRaises(rustcheck.RustCheckError) as e:
            rustcheck.lin_requirements(64, 1 << 40, 256, 3329, 4,
                                       [("w", 8, ("l2_squared", 10)), ("u", 1, ("unbounded",))])
        self.assertEqual(e.exception.rust_message, "unbounded variable in lifted relation")

    def test_rns_and_capacity(self):
        self.assertEqual(rustcheck.rns_primes_required(1099511627917, 64), 2)
        self.assertEqual(rustcheck.rns_primes_required((1 << 99) + 1, 128), 4)
        p = copy.deepcopy(toy())
        rustcheck.check_params(p, rustcheck.NARROW_CAPACITY)
        with self.assertRaises(rustcheck.RustCheckError):
            rustcheck.check_params(p, dict(rustcheck.NARROW_CAPACITY, prime_bits=40))
        # The narrow capacity's limit on the exponents: 40.
        with self.assertRaises(rustcheck.RustCheckError) as e:
            rustcheck.check_params(dict(p, log_sigma=[41, 12, 10, 8]), rustcheck.NARROW_CAPACITY)
        self.assertEqual(e.exception.rust_message, "Gaussian width capacity")


if __name__ == "__main__":
    unittest.main()
