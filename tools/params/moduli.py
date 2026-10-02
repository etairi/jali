#!/usr/bin/env python3
"""Deterministically generate 62-bit NTT primes; Python standard library only.

`--count N` generates the first N primes (default 4); each prefix is the table of a smaller
count, so a ring that needs k primes selects the same ones whatever the table length.
"""
import argparse
import json

def prime(n):
    s, d = 0, n - 1
    while d % 2 == 0:
        s, d = s + 1, d // 2
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        x = pow(a % n, d, n)
        if a % n == 0 or x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

def generate(count=4):
    p = ((1 << 62) - 1) // 2048 * 2048 + 1
    rows = []
    while len(rows) < count:
        if prime(p):
            a = 2
            while True:
                root = pow(a, (p - 1) // 2048, p)
                if pow(root, 1024, p) == p - 1:
                    rows.append([p, root])
                    break
                a += 1
        p -= 2048
    return rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=4)
    print(json.dumps(generate(parser.parse_args().count), indent=2))
