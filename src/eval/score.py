"""Trajectory scoring that mirrors the evo suite (ATE RMSE + RPE) with Track 3 tier breakdown.

  * association: nearest ground-truth timestamp within --max-diff (evo default 0.01 s)
  * alignment modes: se3 (evo -a, Umeyama), sim3 (-as), origin (--align_origin), none
  * tiers (need the degrade.py meta file):
       ckpt    = all poses before the IMU failure            (proxy for "waypoints", 30 pts)
       blind   = poses after the failure (5 s dead reckoning) (20 pts)
       dropout = poses inside a dropout and the 1 s after it  (10 pts)
  * RPE: translation [m] and rotation [deg] for delta = 1 frame (evo default) and delta = 1 s

Usage:
  python score.py --gt X.gt.txt --est est.txt [--meta X.meta.json] [--modes se3,sim3,origin,none] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research"))
from common import load_tum, quats_to_rots, rot_angle  # noqa: E402

MODES = ("se3", "sim3", "origin", "none")


def associate(t_est, t_gt, max_diff=0.01):
    idx = np.clip(np.searchsorted(t_gt, t_est), 1, len(t_gt) - 1)
    left, right = t_gt[idx - 1], t_gt[idx]
    j = np.where(np.abs(t_est - left) <= np.abs(right - t_est), idx - 1, idx)
    ok = np.abs(t_gt[j] - t_est) <= max_diff
    return np.nonzero(ok)[0], j[ok]


def umeyama(src, dst, with_scale):
    """Find s,R,t minimising ||dst - (s R src + t)||."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum() if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def align(mode, p_est, R_est, p_gt, R_gt):
    if mode in ("se3", "sim3"):
        s, R, t = umeyama(p_est, p_gt, mode == "sim3")
    elif mode == "origin":
        R = R_gt[0] @ R_est[0].T
        s, t = 1.0, p_gt[0] - R @ p_est[0]
    else:
        s, R, t = 1.0, np.eye(3), np.zeros(3)
    return s * p_est @ R.T + t, np.einsum("ij,njk->nik", R, R_est), s


def rpe(p_e, R_e, p_g, R_g, step):
    if len(p_e) <= step:
        return float("nan"), float("nan")
    i, j = np.arange(len(p_e) - step), np.arange(step, len(p_e))
    # relative motions in the first pose's frame
    dpe = np.einsum("nji,nj->ni", R_e[i], p_e[j] - p_e[i])
    dpg = np.einsum("nji,nj->ni", R_g[i], p_g[j] - p_g[i])
    dRe = np.einsum("nji,njk->nik", R_e[i], R_e[j])
    dRg = np.einsum("nji,njk->nik", R_g[i], R_g[j])
    # error E = rel_gt^-1 * rel_est
    t_err = np.einsum("nji,nj->ni", dRg, dpe - dpg)
    r_err = rot_angle(np.einsum("nji,njk->nik", dRg, dRe))
    return float(np.sqrt((t_err ** 2).sum(1).mean())), float(np.degrees(np.sqrt((r_err ** 2).mean())))


def tier_masks(t, meta):
    m = {"all": np.ones(len(t), bool)}
    if meta:
        tf = meta["t_fail_ns"] * 1e-9
        m["ckpt"] = t <= tf
        m["blind"] = t > tf
        d = np.zeros(len(t), bool)
        for dr in meta["dropouts"]:
            d |= (t >= dr["start_ns"] * 1e-9) & (t <= dr["end_ns"] * 1e-9 + 1.0)
        m["dropout"] = d
    return m


def score(gt_path, est_path, meta=None, modes=MODES, max_diff=0.01):
    gt, est = load_tum(gt_path), load_tum(est_path)
    ie, ig = associate(est.t, gt.t, max_diff)
    p_e, R_e = est.p[ie], quats_to_rots(est.q[ie])
    p_g, R_g = gt.p[ig], quats_to_rots(gt.q[ig])
    t = est.t[ie]
    fs = 1.0 / np.median(np.diff(t)) if len(t) > 1 else 200.0
    masks = tier_masks(t, meta)
    out = {"n_est": len(est.t), "n_matched": int(len(ie)), "n_gt": len(gt.t)}
    for mode in modes:
        pa, Ra, s = align(mode, p_e, R_e, p_g, R_g)
        err = np.linalg.norm(pa - p_g, axis=1)
        rerr = np.degrees(rot_angle(np.einsum("nji,njk->nik", R_g, Ra)))
        r = {"scale": s}
        for k, mk in masks.items():
            r[f"ate_{k}"] = float(np.sqrt((err[mk] ** 2).mean())) if mk.any() else float("nan")
            r[f"are_deg_{k}"] = float(np.sqrt((rerr[mk] ** 2).mean())) if mk.any() else float("nan")
        r["rpe_t_1f"], r["rpe_r_deg_1f"] = rpe(pa, Ra, p_g, R_g, 1)
        r["rpe_t_1s"], r["rpe_r_deg_1s"] = rpe(pa, Ra, p_g, R_g, max(1, int(round(fs))))
        if "ckpt" in masks and masks["ckpt"].any():  # RPE before the IMU failure only
            mc = masks["ckpt"]
            r["rpe_t_1s_ckpt"], r["rpe_r_deg_1s_ckpt"] = rpe(pa[mc], Ra[mc], p_g[mc], R_g[mc], max(1, int(round(fs))))
        if meta:  # blind error measured relative to the true pose at failure (independent of global drift)
            mb = masks["blind"]
            if mb.any() and (~mb).any():
                k0 = np.nonzero(~mb)[0][-1]
                de = np.einsum("ij,nj->ni", Ra[k0].T, pa[mb] - pa[k0])
                dg = np.einsum("ij,nj->ni", R_g[k0].T, p_g[mb] - p_g[k0])
                r["blind_rel_rmse"] = float(np.sqrt(((de - dg) ** 2).sum(1).mean()))
        out[mode] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--est", required=True)
    ap.add_argument("--meta")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--max-diff", type=float, default=0.01)
    ap.add_argument("--json")
    a = ap.parse_args()
    meta = json.load(open(a.meta)) if a.meta else None
    res = score(a.gt, a.est, meta, a.modes.split(","), a.max_diff)
    print(f"matched {res['n_matched']}/{res['n_est']} est poses (gt {res['n_gt']})")
    keys = [k for k in res[a.modes.split(',')[0]] if k != "scale"]
    print(f"{'metric':>16} " + " ".join(f"{m:>10}" for m in a.modes.split(",")))
    for k in keys:
        print(f"{k:>16} " + " ".join(f"{res[m].get(k, float('nan')):10.4f}" for m in a.modes.split(",")))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
