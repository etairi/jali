#!/usr/bin/env python3
"""Extracts the test vectors of a CFRG Internet-Draft (the text form of
draft-irtf-cfrg-fiat-shamir or draft-irtf-cfrg-sigma-protocols) into JSON.

The vectors are blocks of `Key = Value` lines, a value inline or on indented lines below its
key (wrapped hexadecimal is joined without a separator), a sequence one `- item` per line with
its continuation lines indented further. Page headers and footers interrupt blocks and are
dropped. A block starts at `Id = ...` and ends at the first line of prose. A value that is a
sequence is a list, every other value a string.

    python3 tools/kat/ietf_vectors.py draft-irtf-cfrg-fiat-shamir-04.txt \
        --output kat/ietf/fiat-shamir-04.json

The text is https://www.ietf.org/archive/id/<draft>.txt, and the output records its SHA-256.
"""
import argparse
import hashlib
import json
import re
import sys

PAGE = re.compile(r"^(Orrù & Yun\s+Expires|Internet-Draft\s)")
KEY = re.compile(r"^\s*([A-Z][A-Za-z]*) =(?: (.*))?$")
ITEM = re.compile(r"^\s*- (.*)$")


def vectors(lines):
    out, current, key, indent = [], None, None, 0

    def finish():
        nonlocal current, key
        if current is not None:
            out.append(current)
        current, key = None, None

    for raw in lines:
        line = raw.rstrip("\n")
        if PAGE.match(line) or not line.strip():
            continue
        depth = len(line) - len(line.lstrip(" "))
        m = KEY.match(line)
        if m and m.group(1) == "Id":
            finish()
            current, indent = {"Id": m.group(2).strip()}, depth
            continue
        if current is None:
            continue
        if m and depth == indent:
            key = m.group(1)
            current[key] = (m.group(2) or "").strip()
            continue
        item = ITEM.match(line)
        if item and key is not None and depth > indent \
                and (current[key] == "" or isinstance(current[key], list)):
            if current[key] == "":
                current[key] = []
            current[key].append(item.group(1).strip())
            continue
        # a continuation: more indented than the keys, and no space (wrapped hexadecimal or a
        # wrapped name), which no line of prose is
        part = line.strip()
        if key is not None and depth > indent and " " not in part:
            if isinstance(current[key], list) and current[key]:
                current[key][-1] += part
            else:
                current[key] += part[2:] if part.startswith("0x") and current[key] else part
            continue
        finish()
    finish()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("draft", help="the text form of the draft")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    data = open(a.draft, "rb").read()
    found = vectors(data.decode("utf-8").splitlines())
    ids = [v["Id"] for v in found]
    if len(ids) != len(set(ids)):
        sys.exit("repeated Id")
    name = a.draft.rsplit("/", 1)[-1]
    doc = {"source": {"file": name, "sha256": hashlib.sha256(data).hexdigest()},
           "vectors": found}
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"{len(found)} vectors from {name}")


if __name__ == "__main__":
    main()
