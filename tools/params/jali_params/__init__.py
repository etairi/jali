"""Jali parameter tool: hardness estimates, modulus search and the derivation of parameter sets.

Pure Python with mpmath and sympy. Sage is needed only for the optional cross-checks in
`xcheck`. See `tools/README.md`.
"""
import hashlib
from pathlib import Path

# Bump when a derivation changes its output: the version enters the provenance string of every
# parameter set, and hence its transcript fingerprint.
VERSION = "3"

ROOT = Path(__file__).resolve().parent.parent

# Files whose bytes define the tool; their hash goes into every report.
TOOL_FILES = [
    "lnp_params.py",
    "jali_params/__init__.py",
    "jali_params/cli.py",
    "jali_params/derive.py",
    "jali_params/hardness.py",
    "jali_params/jsonio.py",
    "jali_params/modulus.py",
    "jali_params/partition.py",
    "jali_params/request.py",
    "jali_params/rustcheck.py",
    "jali_params/xcheck.py",
]


def tool_sha256():
    """SHA-256 over the tool files in `TOOL_FILES` order, each prefixed by its name and length."""
    h = hashlib.sha256()
    for name in TOOL_FILES:
        data = (ROOT / name).read_bytes()
        h.update(name.encode() + b"\0" + str(len(data)).encode() + b"\0" + data)
    return h.hexdigest()
