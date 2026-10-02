"""Shared paths and helpers for the parameter-tool tests (stdlib unittest)."""
import contextlib
import io
import json
import random
import sys
import tempfile
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1]
JALI = TOOL.parents[1]
DATA = TOOL / "tests" / "data"
if str(TOOL) not in sys.path:
    sys.path.insert(0, str(TOOL))


def read(path):
    return (JALI / path).read_text()


def load(path):
    return json.loads(read(path))


def check(params_text, request_obj):
    """`lnp_params.py check PARAMS --request REQUEST` in this process: (exit status, standard
    output, standard error)."""
    from jali_params import cli
    with tempfile.TemporaryDirectory() as tmp:
        params, req = Path(tmp) / "params.json", Path(tmp) / "request.json"
        params.write_text(params_text)
        req.write_text(json.dumps(request_obj))
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                cli.main(["check", str(params), "--request", str(req)])
                code = 0
            except SystemExit as e:
                code = e.code
    return code, out.getvalue(), err.getvalue()


def grid():
    """The recorded rank searches of the uSVP estimate (`mlwe_grid.py`)."""
    lines = (DATA / "mlwe_grid.jsonl").read_text().splitlines()
    head = json.loads(lines[0])
    assert "provenance" in head
    return [json.loads(x) for x in lines[1:]]


def miller_rabin(n, rounds=40, seed=2026):
    """An independent probabilistic primality test with seeded random bases."""
    if n < 4:
        return n in (2, 3)
    if n % 2 == 0:
        return False
    rng = random.Random(seed)
    s, d = 0, n - 1
    while d % 2 == 0:
        s, d = s + 1, d // 2
    for _ in range(rounds):
        a = rng.randrange(2, n - 1)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


# Sets and fixtures of the positional form: (request, MLWE report, output).
POSITIONAL = [
    ("tools/params/toy-d64.request.json", "tools/params/toy-d64.report.json",
     "src/params/sets/toy-d64.json"),
    ("tools/params/possession.request.json", "tools/params/toy-d64.report.json",
     "examples/fixtures/possession.json"),
] + [
    (f"tests/fixtures/params/{n}.request.json", f"tests/fixtures/params/{n}.report.json",
     f"tests/fixtures/params/{n}.json")
    for n in ("tag-preimage-crt-d128", "tag-preimage-crt-d64", "dense-blocks-7", "decoder-ranges")
]
PARAMS_FILES = [x[2] for x in POSITIONAL]
NEW_SETS = ["kyber1024-d64", "kyber1024-d128", "demo"]
