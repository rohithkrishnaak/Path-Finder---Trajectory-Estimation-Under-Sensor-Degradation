"""Train the streaming GRU velocity model (Track 3 Path Finder).

The model reads one 9-feature block per 20 ms (mean of 4 IMU samples: accel*0.2, gyro*0.5, gravity direction
in the body frame from the filter attitude) and carries its hidden state along the flight. After every block
it outputs body-frame velocity (3) + log-variance (3). main.py steps it at 50 Hz and uses every 10th output.

Training: segments of --len blocks from EuRoC (clean + synthetic augmentation, and official-style copies made
by the organisers' degrade_imu.py, see train_vel.py), hidden state starts at zero, loss on every step after a
--warm block warm-up. Validation also runs LONG segments to check that the carried state does not drift.

  python train_gru.py --data data/euroc --hidden 96 --out weights/gru/ho_s0.npz            # held-out split
  python train_gru.py --data data/euroc --val none --hidden 96 --out weights/gru/fin_s0.npz
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

import train_vel as tv

DEV = tv.DEV
BLK = 4


def seg_batch(D, idx, L, augment, extra_aug):
    """idx: window END rows (segments of L*BLK rows). Returns x (B, L, 9), y (B, L, 3)."""
    G, A, BI, R, V = D[:5]
    B, n = len(idx), L * BLK
    it = torch.as_tensor(idx, device=DEV)
    ii = it[:, None] + torch.arange(-n + 1, 1, device=DEV)[None]
    g, a = G[ii], A[ii]
    ie = ii[:, BLK - 1::BLK]                                   # last row of every block
    Rk, Re, ve = R[BI[ii]], R[BI[ie]], V[BI[ie]]
    if augment:
        if extra_aug:
            u = torch.rand(B, 1, 1, device=DEV)
            a = a + torch.randn(B, 1, 3, device=DEV) * (torch.rand(B, 1, 1, device=DEV) * tv.AUG["accb"]) + \
                torch.randn(B, n, 3, device=DEV) * (u ** 2 * tv.AUG["accn"])
            u2 = torch.rand(B, 1, 1, device=DEV)
            g = g + torch.randn(B, 1, 3, device=DEV) * (torch.rand(B, 1, 1, device=DEV) * tv.AUG["gyrb"]) + \
                torch.randn(B, n, 3, device=DEV) * (u2 ** 2 * 0.05)
        tilt = torch.randn(B, 3, device=DEV) * np.deg2rad(tv.AUG["tilt"])   # filter attitude error (constant)
        tilt[:, 2] = 0
        E = tv.exp_so3(tilt)[:, None]
        Rk, Re = E @ Rk, E @ Re
    gb = Rk[..., 2, :]
    x = torch.cat([a * tv.SA, g * tv.SG, gb], -1).reshape(B, L, BLK, 9).mean(2)
    y = (Re.transpose(-1, -2) @ ve[..., None])[..., 0]          # velocity in the (filter's) body frame
    return x, y, Re


class Net(nn.Module):
    def __init__(self, hidden=96, drop=0.2):
        super().__init__()
        self.gru = nn.GRU(9, hidden, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.out = nn.Linear(hidden, 6)

    def forward(self, x):
        o = self.out(self.drop(self.gru(x)[0]))
        return o[..., :3], o[..., 3:].clamp(-6, 4)


def ends(idx_all, n_rows_needed, rng, k):
    return rng.choice(idx_all, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default="MH_05_difficult,V1_02_medium,V2_03_difficult")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--len", type=int, default=300, help="training segment length in blocks (50 blocks = 1 s)")
    ap.add_argument("--warm", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--official", type=int, default=8)
    ap.add_argument("--p-off", type=float, default=0.7)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--disp", type=float, default=0.0, help="weight of the 5 s displacement (drift) loss")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    seqs = sorted(d for d in os.listdir(a.data) if os.path.isdir(os.path.join(a.data, d)))
    val = [] if a.val == "none" else a.val.split(",")
    L = a.len
    tv.W = L * BLK                       # build() then only returns window ends with a full segment behind them
    Dt = tv.build([os.path.join(a.data, s) for s in seqs if s not in val], a.official, 1000 + a.seed)
    if val:
        LV = 2000                        # 40 s segments: does the carried state stay healthy?
        tv.W = LV * BLK
        Dv = tv.build([os.path.join(a.data, s) for s in seqs if s in val], 2, 0, fixed_seed=42)
        iv = Dv[6][::400]
    net = Net(a.hidden, a.drop).to(DEV)
    print("params", sum(p.numel() for p in net.parameters()), flush=True)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=a.epochs * a.steps)
    n_off = int(round(a.batch * a.p_off))
    for ep in range(a.epochs):
        net.train(); t0 = time.time(); tr = 0.0
        for _ in range(a.steps):
            xc, yc, Rc = seg_batch(Dt, rng.choice(Dt[5], a.batch - n_off), L, True, True)
            xo, yo, Ro = seg_batch(Dt, rng.choice(Dt[6], n_off), L, True, False)
            x, y, Rw = torch.cat([xc, xo]), torch.cat([yc, yo]), torch.cat([Rc, Ro])
            m, lv = net(x)
            m, lv, y, Rw = m[:, a.warm:], lv[:, a.warm:], y[:, a.warm:], Rw[:, a.warm:]
            mse = (m - y) ** 2
            loss = mse.mean() if ep < 2 else (0.5 * (mse * torch.exp(-lv) + lv)).mean()
            if a.disp > 0:
                # drift loss: world-frame displacement error over 5 s chunks (what ATE actually punishes)
                ew = (Rw @ (m - y)[..., None])[..., 0] * 0.02
                n5 = (ew.shape[1] // 250) * 250
                d = ew[:, :n5].reshape(ew.shape[0], -1, 250, 3).sum(2)
                loss = loss + a.disp * (d ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step(); tr += mse.mean().item()
        msg = f"ep {ep:02d} train rmse {np.sqrt(tr / a.steps):.3f}"
        if val:
            net.eval()
            with torch.no_grad():
                torch.manual_seed(1)
                x, y, Rw = seg_batch(Dv, iv, LV, True, False)
                m, lv = net(x)
                e = (m - y) ** 2
                ew = (Rw @ (m - y)[..., None])[..., 0] * 0.02
                d10 = ew[:, 500:2000].reshape(len(iv), -1, 500, 3).sum(2).norm(dim=-1).mean().item()
                d10z = ((Rw @ y[..., None])[..., 0] * 0.02)[:, 500:2000].reshape(len(iv), -1, 500, 3).sum(2).norm(dim=-1).mean().item()
                r = lambda s: e[:, s].mean().sqrt().item()
                z = (e[:, 50:] / torch.exp(lv[:, 50:])).mean().item()
                msg += (f" | official val rmse: 1-3 s {r(slice(50, 150)):.3f}  3-10 s {r(slice(150, 500)):.3f}"
                        f"  10-40 s {r(slice(500, None)):.3f} (zero {(y ** 2).mean().sqrt().item():.3f}) z2 {z:.2f}"
                        f" | 10 s drift {d10:.2f} m (zero {d10z:.2f})")
        print(msg, f"{time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    np.savez(a.out, **sd)
    print("saved", a.out)


if __name__ == "__main__":
    main()
