"""Sweep offline post-processing of main.py outputs (allowed: FAQ says the task is offline / non-causal).

  python src/research/postproc_sweep.py --cases data/official_ho --est-dir <dir with <case>.txt> [--glob "*.zip"]

Variants are applied to the position track only; orientation is untouched. Scored with eval/score.py (se3).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "eval"))
from score import score  # noqa: E402


def load(path):
    t, rest = [], []
    with open(path) as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            a = ln.split()
            t.append(a[0])
            rest.append([float(x) for x in a[1:]])
    return t, np.array(rest)


def centered_ma(p, t, win_s):
    """Centred moving average over +-win_s/2 seconds (time-based, handles gaps)."""
    if win_s <= 0:
        return p
    c = np.vstack([np.zeros((1, 3)), np.cumsum(p, 0)])
    lo = np.searchsorted(t, t - win_s / 2, "left")
    hi = np.searchsorted(t, t + win_s / 2, "right")
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def variant(t, p, tf, v):
    out = p.copy()
    pre = t <= tf
    if v.get("ma", 0) > 0:
        sm = centered_ma(p, t, v["ma"])
        if v.get("ma_pre_only"):
            out[pre] = sm[pre]
        else:
            out = sm
    a = v.get("shrink", 1.0)
    if a != 1.0:
        c = out[pre].mean(0)
        out = c + a * (out - c)
    az = v.get("shrink_z", 1.0)
    if az != 1.0:
        cz = out[pre, 2].mean()
        out[:, 2] = cz + az * (out[:, 2] - cz)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--est-dir", required=True)
    ap.add_argument("--glob", default="*.zip")
    a = ap.parse_args()
    variants = [dict(name="base")]
    for w in (0.25, 0.5, 1.0, 2.0, 3.0, 5.0):
        variants.append(dict(name=f"ma{w}", ma=w))
    for s in (0.9, 0.8, 0.7, 0.6):
        variants.append(dict(name=f"shrink{s}", shrink=s))
    for w, s in ((1.0, 0.8), (2.0, 0.8), (2.0, 0.7), (3.0, 0.8)):
        variants.append(dict(name=f"ma{w}+s{s}", ma=w, shrink=s))
    for sz in (0.5, 0.0):
        variants.append(dict(name=f"ma2+z{sz}", ma=2.0, shrink_z=sz))

    cases = sorted(glob.glob(os.path.join(a.cases, a.glob)))
    res = {v["name"]: [] for v in variants}
    tmp = tempfile.mkdtemp()
    for z in cases:
        base = z[:-4]
        name = os.path.basename(base)
        meta = json.load(open(base + ".meta.json"))
        ts, arr = load(os.path.join(a.est_dir, name + ".txt"))
        t = np.array([float(x) for x in ts])
        tf = meta["t_fail_ns"] * 1e-9
        for v in variants:
            p = variant(t, arr[:, :3], tf, v)
            out = os.path.join(tmp, "e.txt")
            with open(out, "w") as f:
                for i in range(len(t)):
                    f.write(f"{ts[i]} {p[i,0]:.6f} {p[i,1]:.6f} {p[i,2]:.6f} "
                            f"{arr[i,3]:.9f} {arr[i,4]:.9f} {arr[i,5]:.9f} {arr[i,6]:.9f}\n")
            r = score(base + ".gt.txt", out, meta, ["se3"])["se3"]
            res[v["name"]].append((r["ate_ckpt"], r["ate_blind"], r["ate_dropout"], r["rpe_r_deg_1s_ckpt"]))
    print(f"{'variant':16s} ckpt   blind  drop   rotRPE   (mean of {len(cases)} cases)")
    for k, v in res.items():
        m = np.nanmean(np.array(v), 0)
        print(f"{k:16s} {m[0]:.3f}  {m[1]:.3f}  {m[2]:.3f}  {m[3]:.3f}")


if __name__ == "__main__":
    main()
