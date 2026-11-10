"""Strict checker for estimated_trajectory.txt against the Track 3 output contract.

Checks:
  * every non-comment line has exactly 8 space-separated numeric fields, all finite
  * LF line endings only, no tabs/commas, comment lines start with '#'
  * unit quaternions (|q| within 1e-6 of 1)
  * the first N timestamps equal the N input IMU timestamps EXACTLY (ns -> s, decimal compare), same order
  * any extra (appended blind) rows are strictly increasing, start after the last input row,
    and span <= blind_s + 1 sample; if --meta says the failure was truncated, they must cover >= blind_s - 1 sample
Exit code 0 = pass, 1 = fail.

Usage: python format_check.py estimated_trajectory.txt <input.zip|folder|csv> [--meta x.meta.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from decimal import Decimal, InvalidOperation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research"))
from common import load_imu  # noqa: E402


def check(est_path: str, input_path: str, meta_path: str | None = None, blind_s: float = 5.0, verbose=True):
    errors: list[str] = []
    imu = load_imu(input_path)
    t_in = [int(x) for x in imu.t_ns]

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
    if n_out < n_in:
        errors.append(f"too few poses: {n_out} < {n_in} input rows")
    mism = 0
    for i in range(min(n_in, n_out)):
        if rows[i][0] * 1_000_000_000 != t_in[i]:  # exact decimal comparison
            mism += 1
            if mism <= 5:
                errors.append(f"pose {i}: timestamp {rows[i][0]} != input {t_in[i]} ns")
    if mism > 5:
        errors.append(f"... {mism} timestamp mismatches in total")

    extra = rows[n_in:]
    if extra:
        last = Decimal(t_in[-1]) / Decimal(10 ** 9)
        prev = last
        for ts, _ in extra:
            if ts <= prev:
                errors.append(f"appended row {ts} not strictly increasing")
                break
            prev = ts
        span = float(extra[-1][0] - last)
        if span > blind_s + 0.011:
            errors.append(f"appended rows span {span:.3f}s > blind {blind_s}s")
    if meta_path:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("failure_truncated"):
            span = float(extra[-1][0] - Decimal(t_in[-1]) / Decimal(10 ** 9)) if extra else 0.0
            if span < blind_s - 0.011:
                errors.append(f"failure truncated but appended rows only span {span:.3f}s")
    ok = not errors
    if verbose:
        print(f"[format_check] {'PASS' if ok else 'FAIL'}  input_rows={n_in} output_rows={n_out} appended={len(extra)}")
        for e in errors[:60]:
            print("  -", e)
    return ok, errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("est")
    ap.add_argument("input")
    ap.add_argument("--meta")
    ap.add_argument("--blind-s", type=float, default=5.0)
    a = ap.parse_args()
    ok, _ = check(a.est, a.input, a.meta, a.blind_s)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
