"""Train the learned velocity model (Track 3 Path Finder).

Input : last 1 s of IMU (200 samples at 200 Hz) in the BODY frame plus the gravity direction in the body frame
        (from the filter orientation), block-averaged 4:1 -> 50 steps x 9 = 450 numbers.
Output: velocity in the body frame (3) + log-variance (3). The EKF uses it as a velocity measurement.

Training data (EuRoC ground truth):
  * clean IMU with on-the-fly augmentation (white noise, constant bias, tilt error of the filter attitude);
  * OFFICIAL-STYLE copies (--official K): each sequence is run through the organisers' degrade_imu.degrade()
    K times (different seeds, max level and ramp), in both column orders (documented `t, accel, gyro`, and the
    raw EuRoC file fed in unchanged, which puts accel-sized noise on the gyro). Each copy is then put back on
    the original time grid exactly as main.py sees it: invalid rows (zero, stuck, duplicated) are dropped and
    every gap is filled by linear interpolation of the IMU (main.py bridges gaps <= 0.5 s the same way).
    Windows that reach into the zeroed failure window are excluded.

  python train_vel.py --data data/euroc --val MH_05_difficult,V1_02_medium,V2_03_difficult --out weights/x.npz --bin weights/x.bin
  python train_vel.py --data data/euroc --val none --out weights/vel.npz --bin weights/vel.bin   # final fit
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "official"))
from common import find_gt_csv, load_imu, official_degrade, quats_to_rots, slerp_quats  # noqa: E402

W = 200          # window samples
BLK = 4
T = W // BLK     # 50 steps
IN = T * 9
SA, SG = 0.2, 0.5   # input scaling (accel, gyro)
ORDER = ("l1.weight", "l1.bias", "l2.weight", "l2.bias", "l3.weight", "l3.bias")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_seq(d):
    """Full IMU stream plus GT (rotation, velocity) on the IMU rows inside the GT span [i0, i1).
    EuRoC ground truth carries velocity; motion-capture ground truth (TUM-VI) only has poses, so velocity is a
    central difference over +-3 samples (50 ms at 120 Hz). `gt_ok` is False near motion-capture gaps."""
    imu = load_imu(d)
    gt = np.loadtxt(find_gt_csv(d), delimiter=",", comments="#")
    tg = gt[:, 0] * 1e-9
    t = imu.t_ns * 1e-9
    inside = np.nonzero((t >= tg[0]) & (t <= tg[-1]))[0]
    i0, i1 = int(inside[0]), int(inside[-1]) + 1
    ti = t[i0:i1]
    q = gt[:, 4:8]
    R = quats_to_rots(slerp_quats(tg, q / np.linalg.norm(q, axis=1, keepdims=True), ti))
    bad = np.zeros(len(tg), bool)                       # GT samples next to a gap
    for k in np.nonzero(np.diff(tg) > 0.03)[0]:
        bad[max(0, k - 6):k + 8] = True
    if gt.shape[1] >= 11:
        vg = gt[:, 8:11]
    else:
        h = 3
        vg = np.zeros((len(tg), 3))
        vg[h:-h] = (gt[2 * h:, 1:4] - gt[:-2 * h, 1:4]) / (tg[2 * h:] - tg[:-2 * h])[:, None]
        bad[:h] = bad[-h:] = True
    v = np.column_stack([np.interp(ti, tg, vg[:, i]) for i in range(3)])
    j = np.clip(np.searchsorted(tg, ti), 1, len(tg) - 1)
    gt_ok = ~(bad[j - 1] | bad[j])
    return dict(t_ns=imu.t_ns, g=imu.gyro, a=imu.acc, i0=i0, i1=i1, R=R, v=v, gt_ok=gt_ok,
                name=os.path.basename(os.path.normpath(d)))


def official_copy(s, seed, order, max_level, gamma):
    """Degrade the full stream with the organisers' script, then rebuild it on the original grid the way
    main.py bridges short gaps: drop invalid rows (non-finite, zero, stuck, duplicated) and interpolate the IMU
    linearly. (main.py bridges gaps up to 0.5 s only; the official script's gaps are at most ~0.25 s, so every
    gap here would be bridged.) Returns (g, a, ok) on [i0, i1)."""
    t_ns, g, a = s["t_ns"], s["g"], s["a"]
    out, _ = official_degrade(t_ns, g, a, order, seed, max_level, gamma)
    ts = out["timestamp"].to_numpy(np.int64)
    V = out[["c1", "c2", "c3", "c4", "c5", "c6"]].to_numpy(np.float64)
    ok = np.all(np.isfinite(V), 1) & np.any(V != 0, 1)
    ok[1:] &= ~np.all(V[1:] == V[:-1], 1)
    tv, Vv = (ts[ok] - t_ns[0]) * 1e-9, V[ok]
    tq = (t_ns - t_ns[0]) * 1e-9
    Vi = np.column_stack([np.interp(tq, tv, Vv[:, c]) for c in range(6)])
    gd, ad = (Vi[:, 3:6], Vi[:, 0:3]) if order == "script" else (Vi[:, 0:3], Vi[:, 3:6])
    alive = tq < tv[-1] - 0.05            # before the IMU failure
    i0, i1 = s["i0"], s["i1"]
    return gd[i0:i1], ad[i0:i1], alive[i0:i1]


def build(dirs, n_official, seed0, fixed_seed=None):
    """Concatenate segments: per sequence one clean segment + n_official degraded copies.
    Returns tensors G, A (rows), BI (row -> index into R/V), R, V, and window-end indices for clean / official."""
    G, A, BI, Rl, Vl, idx_clean, idx_off = [], [], [], [], [], [], []
    rows = base = 0
    rng = np.random.default_rng(seed0)
    for d in dirs:
        s = load_seq(d)
        n = s["i1"] - s["i0"]
        Rl.append(s["R"]); Vl.append(s["v"])
        segs = [(s["g"][s["i0"]:s["i1"]], s["a"][s["i0"]:s["i1"]], s["gt_ok"].copy(), False)]
        for k in range(n_official):
            order = "script" if k % 2 == 0 else "euroc"
            sd = fixed_seed if fixed_seed is not None else int(rng.integers(1000, 10 ** 6))
            ml = 1.0 if fixed_seed is not None else float(rng.uniform(0.7, 1.3))
            gm = 1.7 if fixed_seed is not None else float(rng.uniform(1.3, 2.1))
            gd, ad, ok = official_copy(s, sd, order, ml, gm)
            segs.append((gd, ad, ok & s["gt_ok"], True))
        for g, a, ok, off in segs:
            G.append(g); A.append(a); BI.append(base + np.arange(n))
            bad_cum = np.cumsum(~ok)                      # a window is usable only if every row in it is ok
            end_ok = np.zeros(n, bool)
            end_ok[W - 1:] = (bad_cum[W - 1:] - np.concatenate([[0], bad_cum[:-W]])) == 0
            (idx_off if off else idx_clean).append(rows + np.nonzero(end_ok)[0])
            rows += n
        base += n
        print(f"  {s['name']}: {n} rows, {n_official} official copies", flush=True)
    f32 = lambda x: torch.tensor(np.concatenate(x), dtype=torch.float32, device=DEV)
    cat_i = lambda x: np.concatenate(x) if x else np.zeros(0, np.int64)
    return (f32(G), f32(A), torch.tensor(np.concatenate(BI), device=DEV), f32(Rl), f32(Vl),
            cat_i(idx_clean), cat_i(idx_off))


VAL_DEFAULT = "MH_05_difficult,V1_02_medium,V2_03_difficult,room3,room6"


def list_seqs(data, val):
    """All sequence folders under the comma-separated roots, split into (train, validation) paths."""
    dirs = [os.path.join(r, d) for r in data.split(",") for d in sorted(os.listdir(r)) if os.path.isdir(os.path.join(r, d))]
    return [d for d in dirs if os.path.basename(d) not in val], [d for d in dirs if os.path.basename(d) in val]


def exp_so3(w):
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    k = w / th
    K = torch.zeros(w.shape[:-1] + (3, 3), device=w.device)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    th = th[..., None]
    return torch.eye(3, device=w.device) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


AUG = dict(tilt=2.0, accn=2.5, accb=0.3, gyrb=0.05)


def make_batch(D, idx, augment, extra_aug=True):
    """augment: tilt error always; extra_aug: synthetic noise/bias (used on clean rows only)."""
    G, A, BI, R, V = D[:5]
    B = len(idx)
    ar = torch.arange(-W + 1, 1, device=DEV)
    it = torch.as_tensor(idx, device=DEV)
    ii = it[:, None] + ar[None]
    g, a = G[ii], A[ii]
    Rk, ve = R[BI[ii]], V[BI[it]]
    if augment:
        if extra_aug:
            u = torch.rand(B, 1, 1, device=DEV)
            a = a + torch.randn(B, 1, 3, device=DEV) * (torch.rand(B, 1, 1, device=DEV) * AUG["accb"]) + \
                torch.randn(B, W, 3, device=DEV) * (u ** 2 * AUG["accn"])
            u2 = torch.rand(B, 1, 1, device=DEV)
            g = g + torch.randn(B, 1, 3, device=DEV) * (torch.rand(B, 1, 1, device=DEV) * AUG["gyrb"]) + \
                torch.randn(B, W, 3, device=DEV) * (u2 ** 2 * 0.05)
        tilt = torch.randn(B, 3, device=DEV) * np.deg2rad(AUG["tilt"])
        tilt[:, 2] = 0
        Rk = exp_so3(tilt)[:, None] @ Rk
    Re = Rk[:, -1]                                              # attitude at the window end
    gb = Rk[..., 2, :]                                          # R^T e_z in body frame = third row of R
    x = torch.cat([a * SA, g * SG, gb], -1).reshape(B, T, BLK, 9).mean(2).reshape(B, IN)
    y = (Re.transpose(1, 2) @ ve[..., None])[..., 0]            # velocity in the body frame
    return x, y


class Net(nn.Module):
    def __init__(self, h1=256, h2=128, drop=0.1):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(IN, h1), nn.Linear(h1, h2), nn.Linear(h2, 6)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(torch.relu(self.l1(x)))
        x = self.drop(torch.relu(self.l2(x)))
        o = self.l3(x)
        return o[:, :3], o[:, 3:].clamp(-6, 4)


def batch_mixed(D, p_off, B, rng):
    ic, io = D[5], D[6]
    n_off = int(round(B * p_off)) if len(io) else 0
    xs, ys = [], []
    if B - n_off:
        x, y = make_batch(D, rng.choice(ic, B - n_off), True, True); xs.append(x); ys.append(y)
    if n_off:
        x, y = make_batch(D, rng.choice(io, n_off), True, False); xs.append(x); ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="one or more folders of sequences, comma-separated")
    ap.add_argument("--val", default=VAL_DEFAULT)
    ap.add_argument("--out", required=True, help="npz with the state dict")
    ap.add_argument("--bin", help="also write the flat float32 file read by main.py / embed_weights.py")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--tilt", type=float, default=2.0)
    ap.add_argument("--accn", type=float, default=2.5)
    ap.add_argument("--accb", type=float, default=0.3)
    ap.add_argument("--gyrb", type=float, default=0.05)
    ap.add_argument("--official", type=int, default=8, help="official-style degraded copies per sequence")
    ap.add_argument("--p-off", type=float, default=0.7, help="fraction of each batch from the official copies")
    a = ap.parse_args()
    AUG.update(tilt=a.tilt, accn=a.accn, accb=a.accb, gyrb=a.gyrb)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    val = [] if a.val == "none" else a.val.split(",")
    tr, va = list_seqs(a.data, val)
    print("device", DEV, "| train", [os.path.basename(x) for x in tr], "val", [os.path.basename(x) for x in va], flush=True)
    Dt = build(tr, a.official, 1000 + a.seed)
    if va:
        Dv = build(va, 2, 0, fixed_seed=42)  # the organisers' seed, both column orders
        iv_c, iv_o = Dv[5][::10], Dv[6][::10]
    net = Net(drop=a.dropout).to(DEV)   # main.py's VelNet expects exactly this shape (450-256-128-6)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=a.epochs * a.steps)
    for ep in range(a.epochs):
        net.train(); t0 = time.time(); tot = 0
        for _ in range(a.steps):
            x, y = batch_mixed(Dt, a.p_off, 512, rng)
            m, lv = net(x)
            mse = (m - y) ** 2
            loss = mse.mean() if ep < 3 else (0.5 * (mse * torch.exp(-lv) + lv)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        msg = f"ep {ep:02d} loss {tot / a.steps:.4f}"
        if va:
            net.eval()
            with torch.no_grad():
                for tag, iv, aug, extra in (("clean", iv_c, False, False), ("synth", iv_c, True, True),
                                            ("official", iv_o, True, False)):
                    torch.manual_seed(1)
                    x, y = make_batch(Dv, iv, aug, extra)
                    m, lv = net(x)
                    rm = ((m - y) ** 2).mean().sqrt().item()
                    rz_ = (y ** 2).mean().sqrt().item()
                    z = ((m - y) ** 2 / torch.exp(lv)).mean().item()
                    msg += f" | {tag} rmse {rm:.3f} (zero {rz_:.3f}) z2 {z:.2f}"
        print(msg, f"{time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    np.savez(a.out, **sd)
    if a.bin:
        np.concatenate([sd[k].astype("<f4").ravel() for k in ORDER]).tofile(a.bin)
    print("saved", a.out, a.bin or "")


if __name__ == "__main__":
    main()
