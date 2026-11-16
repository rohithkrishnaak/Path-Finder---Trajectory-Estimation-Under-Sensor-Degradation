"""Strict checker for estimated_trajectory.txt against the Track 3 output contract (FAQ, Nov 2026).

Checks:
  * every non-comment line has exactly 8 space-separated numeric fields, all finite
  * LF line endings only, no tabs/commas, comment lines start with '#'
  * unit quaternions (|q| within 1e-6 of 1)
  * EXACTLY one pose per input row: same count, and pose i has input row i's timestamp (ns -> s, exact decimal
    compare). Appending poses beyond the input is forbidden ("DO NOT append extra poses").
The input is parsed with the same reader as main.py (any column order, timestamp unit, zip/folder/csv).
Exit code 0 = pass, 1 = fail.

Usage: python format_check.py estimated_trajectory.txt <input.zip|folder|csv>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from decimal import Decimal, InvalidOperation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from main import load  # noqa: E402


def check(est_path: str, input_path: str, verbose=True):
    errors: list[str] = []
    try:
        t_in = [int(x) for x in load(input_path).t_ns]
    except SystemExit as e:  # no rows at all
        t_in = []
        errors.append(str(e))

    with open(est_path, "rb") as f:
        raw = f.read()
    if b"\r" in raw:
        errors.append("CR characters found (must be LF line endings)")
    text = raw.decode("utf-8", errors="replace")
    rows = []
    for ln, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue
        if "," in line or "\t" in line:
            errors.append(f"line {ln}: separator must be a single space")
        tok = line.split()
        if len(tok) != 8:
            errors.append(f"line {ln}: expected 8 fields, got {len(tok)}")
            continue
        try:
            ts = Decimal(tok[0])
            vals = [float(x) for x in tok[1:]]
        except (InvalidOperation, ValueError):
            errors.append(f"line {ln}: non-numeric field")
            continue
        if not all(math.isfinite(v) for v in vals) or not ts.is_finite():
            errors.append(f"line {ln}: non-finite value")
            continue
        qn = math.sqrt(sum(v * v for v in vals[3:7]))
        if abs(qn - 1.0) > 1e-6:
            errors.append(f"line {ln}: quaternion norm {qn:.9f}")
        rows.append((ts, vals))
        if len(errors) > 50:
            break

    n_in, n_out = len(t_in), len(rows)
    if n_out != n_in:
        errors.append(f"pose count {n_out} != input rows {n_in} (must be exactly one pose per IMU row)")
    mism = 0
    for i in range(min(n_in, n_out)):
        if rows[i][0] * 1_000_000_000 != t_in[i]:  # exact decimal comparison
            mism += 1
            if mism <= 5:
                errors.append(f"pose {i}: timestamp {rows[i][0]} != input {t_in[i]} ns")
    if mism > 5:
        errors.append(f"... {mism} timestamp mismatches in total")
    ok = not errors
    if verbose:
        print(f"[format_check] {'PASS' if ok else 'FAIL'}  input_rows={n_in} output_rows={n_out}")
        for e in errors[:60]:
            print("  -", e)
    return ok, errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("est")
    ap.add_argument("input")
    a = ap.parse_args()
    ok, _ = check(a.est, a.input)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
