"""Synthetic trajectory + IMU generator (EuRoC layout) for testing without real data.

Produces <out>/mav0/imu0/data.csv and <out>/mav0/state_groundtruth_estimate0/data.csv with exactly
consistent kinematics, so filter bugs can be separated from data issues.

Usage:
  python sim_traj.py --out data/sim/SIM_01 --duration 120 --mode quad --seed 1
Modes:
  quad      attitude tilts with acceleration (thrust-aligned), yaw wanders - multirotor-like
  handheld  independent free attitude (larger roll/pitch) - TUM-VI-like
  static    never moves (filter sanity test)
  rotate    rotates in place, no translation (orientation test)
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

from common import G, rot_to_quat, write_imu_csv

# ADIS16448 (EuRoC sensor.yaml) continuous-time densities
GYRO_ND, GYRO_RW = 1.6968e-4, 1.9393e-5   # rad/s/sqrt(Hz), rad/s^2/sqrt(Hz)
ACC_ND, ACC_RW = 2.0e-3, 3.0e-3           # m/s^2/sqrt(Hz), m/s^3/sqrt(Hz)


def smoothstep5(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


def euler_zyx_to_rot(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    R = np.empty((len(yaw), 3, 3))
    R[:, 0, 0] = cy * cp; R[:, 0, 1] = cy * sp * sr - sy * cr; R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp; R[:, 1, 1] = sy * sp * sr + cy * cr; R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp;     R[:, 2, 1] = cp * sr;                R[:, 2, 2] = cp * cr
    return R


def random_signal(rng, t, n_terms, amp, fmax):
    out = np.zeros_like(t)
    for _ in range(n_terms):
        f = rng.uniform(0.02, fmax)
        out += rng.uniform(0.3, 1.0) * amp / n_terms * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return out


def generate(duration=120.0, mode="quad", seed=0, fs=200.0, static_head=2.0, ramp=2.0,
             pos_amp=3.0, fmax=0.25, noise=True, bias=True):
    rng = np.random.default_rng(seed)
    up = 5  # fine-grid oversampling for numerical derivatives
    dt_f = 1.0 / (fs * up)
    n_f = int(round(duration * fs * up)) + 2 * up + 1
    t_f = (np.arange(n_f) - up) * dt_f  # padded by one coarse sample each side
    s = smoothstep5((t_f - static_head) / ramp)
    tm = t_f - static_head

    if mode in ("static", "rotate"):
        p = np.zeros((n_f, 3))
    else:
        p = np.column_stack([s * random_signal(rng, tm, 4, pos_amp, fmax),
                             s * random_signal(rng, tm, 4, pos_amp, fmax),
                             s * random_signal(rng, tm, 3, 0.4 * pos_amp, fmax)])
    v = np.gradient(p, dt_f, axis=0)
    a = np.gradient(v, dt_f, axis=0)

    if mode == "static":
        yaw = np.full(n_f, rng.uniform(-np.pi, np.pi)); pitch = np.full(n_f, rng.uniform(-.05, .05)); roll = np.full(n_f, rng.uniform(-.05, .05))
        R = euler_zyx_to_rot(yaw, pitch, roll)
    elif mode == "quad":
        yaw = rng.uniform(-np.pi, np.pi) + s * random_signal(rng, tm, 3, 2.0, fmax)
        thrust = a - np.array([0.0, 0.0, -G])
        zb = thrust / np.linalg.norm(thrust, axis=1, keepdims=True)
        xc = np.column_stack([np.cos(yaw), np.sin(yaw), np.zeros(n_f)])
        yb = np.cross(zb, xc); yb /= np.linalg.norm(yb, axis=1, keepdims=True)
        xb = np.cross(yb, zb)
        R = np.stack([xb, yb, zb], axis=2)
    else:  # handheld / rotate
        yaw = rng.uniform(-np.pi, np.pi) + s * random_signal(rng, tm, 3, 3.0, fmax)
        pitch = s * random_signal(rng, tm, 3, 0.6, fmax)
        roll = s * random_signal(rng, tm, 3, 0.6, fmax)
        R = euler_zyx_to_rot(yaw, pitch, roll)

    # body angular rate: w_B(t) = vee(log(R(t-h)^T R(t+h))) / 2h
    w = np.zeros((n_f, 3))
    M = np.einsum("nji,njk->nik", R[:-2], R[2:])  # R(t-h)^T R(t+h), small rotations
    vee = 0.5 * np.column_stack([M[:, 2, 1] - M[:, 1, 2], M[:, 0, 2] - M[:, 2, 0], M[:, 1, 0] - M[:, 0, 1]])
    ang = np.arccos(np.clip((np.trace(M, axis1=1, axis2=2) - 1) / 2, -1, 1))
    scale = np.where(ang > 1e-9, ang / np.maximum(np.sin(ang), 1e-300), 1.0)
    w[1:-1] = vee * scale[:, None] / (2 * dt_f)
    f_b = np.einsum("nji,nj->ni", R, a - np.array([0.0, 0.0, -G]))

    idx = np.arange(up, n_f - up, up)[: int(round(duration * fs))]
    t = t_f[idx]
    n = len(idx)
    gyro, acc = w[idx].copy(), f_b[idx].copy()

    bg = np.zeros((n, 3)); ba = np.zeros((n, 3))
    if bias:
        bg0 = rng.normal(0, 0.003, 3); ba0 = rng.normal(0, 0.05, 3)
        bg = bg0 + np.cumsum(rng.normal(0, GYRO_RW * math.sqrt(1 / fs), (n, 3)), axis=0)
        ba = ba0 + np.cumsum(rng.normal(0, ACC_RW * math.sqrt(1 / fs), (n, 3)), axis=0)
        gyro += bg; acc += ba
    if noise:
        gyro += rng.normal(0, GYRO_ND * math.sqrt(fs), (n, 3))
        acc += rng.normal(0, ACC_ND * math.sqrt(fs), (n, 3))

    q = np.array([rot_to_quat(r) for r in R[idx]])
    for i in range(1, n):  # hemisphere continuity
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    t0_ns = 1403636579758555392 + int(rng.integers(0, 10**9))
    t_ns = t0_ns + np.round(t * 1e9).astype(np.int64) - int(round(t[0] * 1e9))
    return dict(t_ns=t_ns, gyro=gyro, acc=acc, p=p[idx], v=v[idx], q=q, bg=bg, ba=ba)


def write_euroc(out_dir, d):
    write_imu_csv(os.path.join(out_dir, "mav0", "imu0", "data.csv"), d["t_ns"], d["gyro"], d["acc"])
    gt_path = os.path.join(out_dir, "mav0", "state_groundtruth_estimate0", "data.csv")
    os.makedirs(os.path.dirname(gt_path), exist_ok=True)
    with open(gt_path, "w", newline="\n") as f:
        f.write("#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], "
                "v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], "
                "b_w_RS_S_z [rad s^-1], b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]\n")
        for i in range(len(d["t_ns"])):
            row = [*d["p"][i], *d["q"][i], *d["v"][i], *d["bg"][i], *d["ba"][i]]
            f.write(str(int(d["t_ns"][i])) + "," + ",".join(f"{x:.9g}" for x in row) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--mode", default="quad", choices=["quad", "handheld", "static", "rotate"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--static-head", type=float, default=2.0)
    ap.add_argument("--pos-amp", type=float, default=3.0)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--no-bias", action="store_true")
    a = ap.parse_args()
    d = generate(a.duration, a.mode, a.seed, static_head=a.static_head, pos_amp=a.pos_amp,
                 noise=not a.no_noise, bias=not a.no_bias)
    write_euroc(a.out, d)
    print(f"wrote {len(d['t_ns'])} samples to {a.out}")


if __name__ == "__main__":
    main()
