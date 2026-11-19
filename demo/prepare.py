"""Prepare the Webots demo: align Path Finder's output to ground truth and write a compact replay file + figures.

  python demo/prepare.py --name MH_05_difficult --gt X.gt.txt --est X.est.txt --nonn X.nonn.txt --meta X.meta.json

Writes
  demo/webots/controllers/replay/<name>.replay.csv   20 Hz: t, ground truth pose, Path Finder pose, physics-only
                                                     position, phase (0 normal, 1 dropout, 2 blind)
  demo/figures/<name>_topdown.png                    top-down view, ground truth vs Path Finder vs physics only

Alignment is SE(3) Umeyama over the whole run, the same as the official evo evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "eval"))
sys.path.insert(0, os.path.join(HERE, "..", "src", "research"))
from score import align, associate  # noqa: E402
from common import load_tum, quats_to_rots  # noqa: E402


def rots_to_quats(R):
    """Rotation matrices (n,3,3) -> quaternions (n,4) as x y z w."""
    q = np.empty((len(R), 4))
    for i, m in enumerate(R):
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            q[i] = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q[i] = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q[i] = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q[i] = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def aligned(gt, path):
    e = load_tum(path)
    ie, ig = associate(e.t, gt.t)
    pa, Ra, _ = align("se3", e.p[ie], quats_to_rots(e.q[ie]), gt.p[ig], quats_to_rots(gt.q[ig]))
    return e.t[ie], ig, pa, Ra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--est", required=True)
    ap.add_argument("--nonn", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--hz", type=float, default=20.0)
    a = ap.parse_args()

    gt = load_tum(a.gt)
    meta = json.load(open(a.meta))
    t, ig, pe, Re = aligned(gt, a.est)
    t2, ig2, pn, _ = aligned(gt, a.nonn)
    pn_at = np.stack([np.interp(t, t2, pn[:, k]) for k in range(3)], 1)
    pg, Rg = gt.p[ig], quats_to_rots(gt.q[ig])
    # The EuRoC IMU frame has x pointing up. For display, re-map body axes so the drone model's z is the body
    # axis that points up on average (a fixed permutation; it does not change any error).
    u = np.einsum("nji,j->ni", Rg, np.array([0.0, 0.0, 1.0])).mean(0)
    k = int(np.argmax(np.abs(u)))
    c3 = np.zeros(3); c3[k] = np.sign(u[k])
    c1 = np.zeros(3); c1[[m for m in range(3) if m != k][0]] = 1.0
    C = np.column_stack([c1, np.cross(c3, c1), c3])
    qg = rots_to_quats(Rg @ C)
    qe = rots_to_quats(Re @ C)

    tf = meta["t_fail_ns"] * 1e-9
    phase = np.zeros(len(t), int)
    for d in meta["dropouts"]:
        s, e = d["start_ns"] * 1e-9, d["end_ns"] * 1e-9 + 1.0
        phase[(t >= s) & (t <= e)] = 1
    phase[t > tf] = 2

    # subsample to a fixed rate, always keeping the failure instant
    keep = [0]
    for i in range(1, len(t)):
        if t[i] - t[keep[-1]] >= 1.0 / a.hz - 1e-6 or (phase[i] != phase[keep[-1]]):
            keep.append(i)
    keep = np.array(keep)

    out_dir = os.path.join(HERE, "webots", "controllers", "replay")
    os.makedirs(out_dir, exist_ok=True)
    rows = np.column_stack([t[keep] - t[0], pg[keep], qg[keep], pe[keep], qe[keep], pn_at[keep], phase[keep]])
    hdr = "t gx gy gz gqx gqy gqz gqw ex ey ez eqx eqy eqz eqw nx ny nz phase"
    np.savetxt(os.path.join(out_dir, f"{a.name}.replay.csv"), rows, fmt="%.4f", header=hdr, comments="")

    err_e = np.linalg.norm(pe - pg, axis=1)
    err_n = np.linalg.norm(pn_at - pg, axis=1)
    ck = phase != 2
    stats = dict(
        name=a.name, duration_s=float(t[-1] - t[0]), t_fail_rel=float(tf - t[0]),
        ate_ckpt=float(np.sqrt((err_e[ck] ** 2).mean())), ate_blind=float(np.sqrt((err_e[~ck] ** 2).mean())),
        ate_ckpt_nonn=float(np.sqrt((err_n[ck] ** 2).mean())), ate_blind_nonn=float(np.sqrt((err_n[~ck] ** 2).mean())),
    )
    json.dump(stats, open(os.path.join(out_dir, f"{a.name}.stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(HERE, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    ax.plot(pg[:, 0], pg[:, 1], color="#2e7d32", lw=2.2, label="Ground truth (hidden from us)")
    ax.plot(pn_at[:, 0], pn_at[:, 1], color="#c62828", lw=1.2, alpha=0.7,
            label=f"Physics only: {stats['ate_ckpt_nonn']:.1f} m ATE")
    ax.plot(pe[:, 0], pe[:, 1], color="#ef6c00", lw=1.8, label=f"Path Finder: {stats['ate_ckpt']:.1f} m ATE")
    b = phase == 2
    ax.plot(pe[b, 0], pe[b, 1], color="#6a1b9a", lw=3, label="Path Finder, blind 5 s (IMU dead)")
    ax.scatter(pg[b][-1:, 0], pg[b][-1:, 1], marker="*", s=220, color="#2e7d32", zorder=5, label="Target")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{a.name}: held-out, degraded by the organisers' script (SE(3) aligned)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{a.name}_topdown.png"))


if __name__ == "__main__":
    main()
