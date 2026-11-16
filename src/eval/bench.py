"""Run src/main.py over a folder of test cases; print tiered ATE, CPU time and peak heap per case and the mean.

  python src/eval/bench.py --cases data/official [--glob "*_o42.zip"] [--set name=value ...] [--no-nn]
                           [--weights src/weights/vel.bin] [--mem] [--label x] [--mode se3]

Each case is <name>.zip + <name>.gt.txt + <name>.meta.json (made by research/make_official.py).
Also scores the "never moves" reference (Z0: every pose at the origin, identity rotation).

Budgets, as the FAQ defines them:
  * compute: CPU seconds of main.run() per second of IMU data (target <= 20 ms), measured without tracing.
  * memory:  --mem re-runs each case under tracemalloc (heap of Python objects and numpy buffers, i.e. the
             algorithm's data structures) and reports the peak from just before the weights load to the end.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import time
import tracemalloc

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, SRC)
import main as pf  # noqa: E402
from score import score  # noqa: E402


def zero_like(est_path, out_path):
    with open(est_path) as f, open(out_path, "w") as g:
        for line in f:
            if line.startswith("#"):
                g.write(line)
                continue
            g.write(f"{line.split()[0]} 0 0 0 0 0 0 1\n")


def run_case(zp, est, P, weights=None, no_nn=False, mem=True):
    """CPU seconds of an untraced run (model load included), then, if `mem`, the peak heap (MB) of a second
    run under tracemalloc. Returns (run statistics, cpu seconds, peak MB or NaN). Used by make_bundle.py too."""
    t0 = time.process_time()
    net = None if no_nn else pf.VelNet(weights)
    st = pf.run(zp, est, P, net)
    cpu = time.process_time() - t0
    peak = float("nan")
    if mem:
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        net = None if no_nn else pf.VelNet(weights)
        pf.run(zp, est, P, net)
        peak = (tracemalloc.get_traced_memory()[1] - base) / 2 ** 20
        del net
        tracemalloc.stop()
    return st, cpu, peak


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--glob", default="*.zip")
    ap.add_argument("--weights", help="external float32 weight file (default: the ensemble embedded in main.py)")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--label", default="")
    ap.add_argument("--mode", default="se3")
    ap.add_argument("--no-nn", action="store_true")
    ap.add_argument("--mem", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    P = pf.Params()
    for kv in a.set:
        P.set(kv)
    keys = ("ate_ckpt", "ate_blind", "ate_dropout", "rpe_t_1s_ckpt", "rpe_r_deg_1s_ckpt", "blind_rel_rmse")
    rows, zrows, perf, out = [], [], [], {}
    tmp = tempfile.mkdtemp()
    for zp in sorted(glob.glob(os.path.join(a.cases, a.glob))):
        base = zp[:-4]
        name = os.path.basename(base)
        est = os.path.join(tmp, name + ".txt")
        st, cpu, peak = run_case(zp, est, P, a.weights, a.no_nn, a.mem)
        with open(base + ".meta.json") as fh:
            meta = json.load(fh)
        r = score(base + ".gt.txt", est, meta, [a.mode])[a.mode]
        zp0 = est + ".z0"
        zero_like(est, zp0)
        zr = score(base + ".gt.txt", zp0, meta, [a.mode])[a.mode]
        ms = 1000.0 * cpu / st["imu_seconds"]
        rows.append([r[k] for k in keys]); zrows.append([zr[k] for k in keys]); perf.append([ms, peak])
        out[name] = dict(score={k: r[k] for k in keys}, z0={k: zr[k] for k in keys}, ms_per_imu_s=ms, peak_heap_mb=peak,
                         rows=st["rows"], poses=st["poses"], order=st["order"])
        if not a.quiet:
            print(f"{name:26s} " + " ".join(f"{r[k]:8.2f}" for k in keys) +
                  f" | {ms:5.2f} ms/s {peak:5.2f} MB | Z0 ckpt {zr['ate_ckpt']:.2f} blind {zr['ate_blind']:.2f}"
                  f" | {st['order']} rows {st['rows']} poses {st['poses']}", flush=True)
    m, zm = np.nanmean(rows, 0), np.nanmean(zrows, 0)
    print(f"{'':26s} " + " ".join(f"{k[:8]:>8}" for k in keys))
    print(f"{'MEAN ' + a.label:26s} " + " ".join(f"{x:8.2f}" for x in m) +
          f" | {np.mean([p[0] for p in perf]):5.2f} ms/s (max {np.max([p[0] for p in perf]):.2f})" +
          (f" {np.nanmax([p[1] for p in perf]):5.2f} MB max" if a.mem else ""))
    print(f"{'Z0 never-moves':26s} " + " ".join(f"{x:8.2f}" for x in zm))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(cases=out, mean=dict(zip(keys, map(float, m))), z0=dict(zip(keys, map(float, zm)))), f, indent=1)


if __name__ == "__main__":
    main()
