"""Architecture exploration for the velocity model (held-out velocity RMSE on official-style copies).

  python arch_search.py --data data/euroc --arch mlp|mlp_wide|cnn|gru|mlp2s [--epochs 15]

Shares data building and augmentation with train_vel.py. A candidate must beat the shipped MLP clearly to be
worth porting to numpy inference in main.py.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

import train_vel as tv


class CNN(nn.Module):
    """1-D CNN over the 50 block-averaged steps (9 channels)."""
    def __init__(self, drop):
        super().__init__()
        self.c = nn.Sequential(nn.Conv1d(9, 48, 5, padding=2), nn.ReLU(), nn.Conv1d(48, 64, 5, stride=2, padding=2), nn.ReLU(),
                               nn.Conv1d(64, 96, 5, stride=2, padding=2), nn.ReLU())
        self.h = nn.Sequential(nn.Dropout(drop), nn.Linear(96 * 13, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, 6))

    def forward(self, x):
        o = self.h(self.c(x.view(-1, tv.T, 9).transpose(1, 2)).flatten(1))
        return o[:, :3], o[:, 3:].clamp(-6, 4)


class GRU(nn.Module):
    def __init__(self, drop):
        super().__init__()
        self.g = nn.GRU(9, 96, batch_first=True)
        self.h = nn.Sequential(nn.Dropout(drop), nn.Linear(96, 6))

    def forward(self, x):
        o = self.h(self.g(x.view(-1, tv.T, 9))[0][:, -1])
        return o[:, :3], o[:, 3:].clamp(-6, 4)


class MLP(nn.Module):
    def __init__(self, drop, h1=256, h2=128):
        super().__init__()
        self.n = nn.Sequential(nn.Linear(tv.IN, h1), nn.ReLU(), nn.Dropout(drop), nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(drop),
                               nn.Linear(h2, 6))

    def forward(self, x):
        o = self.n(x)
        return o[:, :3], o[:, 3:].clamp(-6, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arch", default="mlp")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--window", type=int, default=200, help="samples; blocks stay at 50 (block = window/50)")
    ap.add_argument("--official", type=int, default=8)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--wd", type=float, default=0.02)
    a = ap.parse_args()
    tv.W, tv.BLK = a.window, a.window // 50
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    val = ["MH_05_difficult", "V1_02_medium", "V2_03_difficult"]
    seqs = sorted(d for d in os.listdir(a.data) if os.path.isdir(os.path.join(a.data, d)))
    Dt = tv.build([os.path.join(a.data, s) for s in seqs if s not in val], a.official, 1000)
    Dv = tv.build([os.path.join(a.data, s) for s in val], 2, 0, fixed_seed=42)
    iv_c, iv_o = Dv[5][::10], Dv[6][::10]
    net = {"mlp": lambda: MLP(a.drop), "mlp_wide": lambda: MLP(a.drop, 512, 256), "cnn": lambda: CNN(a.drop),
           "gru": lambda: GRU(a.drop)}[a.arch]().to(tv.DEV)
    print(a.arch, "params", sum(p.numel() for p in net.parameters()), flush=True)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=a.wd)
    steps = 300
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=a.epochs * steps)
    for ep in range(a.epochs):
        net.train(); t0 = time.time(); tr = 0.0
        for _ in range(steps):
            x, y = tv.batch_mixed(Dt, 0.7, 512, rng)
            m, lv = net(x)
            mse = (m - y) ** 2
            loss = mse.mean() if ep < 3 else (0.5 * (mse * torch.exp(-lv) + lv)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tr += mse.mean().item()
        net.eval()
        msg = f"ep {ep:02d} train rmse {np.sqrt(tr / steps):.3f}"
        with torch.no_grad():
            for tag, iv, aug in (("clean", iv_c, False), ("official", iv_o, True)):
                torch.manual_seed(1)
                x, y = tv.make_batch(Dv, iv, aug, False)
                m, _ = net(x)
                msg += f" | {tag} {((m - y) ** 2).mean().sqrt().item():.3f} (zero {(y ** 2).mean().sqrt().item():.3f})"
        print(msg, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
