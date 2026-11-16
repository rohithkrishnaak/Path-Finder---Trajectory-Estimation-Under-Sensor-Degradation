"""Python reference implementation of the Path Finder runtime (baseline B1).

This file is the numpy twin of src/main.cpp + src/core/*.hpp. Both must produce the same output
(parity test: pose difference < 1e-6). Keep the algorithm, constants and operation order identical.

Pipeline (see docs/plan.md section 3):
  pass 1  scan: row validity flags, nominal dt, last valid row, static-window initialisation
  pass 2  ESKF (15-state error state) with midpoint IMU propagation, 1-sample-lookahead spike repair,
          gap propagation (decaying constant-velocity model) for invalid rows / timestamp gaps,
          static detection -> ZUPT + gravity-tilt + zero-rate updates, adaptive IMU noise,
          blind 5 s forecaster (decaying constant velocity + decaying turn rate) after the last valid row.

Usage: python ref_pipeline.py <input.zip|folder|csv> -o estimated_trajectory.txt [--blind auto|append|none]
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field

import numpy as np

from common import G, load_imu, quat_exp, quat_mul, quat_to_rot, skew


@dataclass
class Params:
    # validity
    w_max: float = 35.0            # rad/s, beyond any plausible reading -> invalid
    a_max: float = 160.0           # m/s^2
    gap_factor: float = 2.5        # dt > gap_factor * nominal dt -> treat as dropout
    # spike repair (needs |x_i - x_prev| and |x_i - x_next| > T while |x_next - x_prev| < T/2)
    spike_k: float = 10.0
    spike_floor_g: float = 0.3     # rad/s
    spike_floor_a: float = 3.0     # m/s^2
    # noise (per-sample std at the nominal rate; adaptive estimate can only raise these)
    sig_g: float = 2.5e-3
    sig_a: float = 3.0e-2
    sig_bg_rw: float = 2.0e-4      # rad/s/sqrt(s)  (inflated vs datasheet to absorb jamming drift)
    sig_ba_rw: float = 3.0e-2      # m/s^2/sqrt(s)
    noise_ema_tau: float = 1.0     # s
    # initialisation
    init_buf_s: float = 10.0
    static_win: int = 40           # samples
    init_win: int = 100            # samples
    p0_att: float = 0.02           # rad (roll/pitch)
    p0_yaw: float = 1e-4
    p0_v: float = 0.05
    p0_ba: float = 0.1
    p0_bg: float = 0.01
    # static detector + updates
    static_acc_std: float = 0.10   # std of |a| over window [m/s^2]
    static_gyr_mean: float = 0.04  # mean |w - bg| over window [rad/s]
    static_every: int = 10         # apply static updates every N static samples
    sig_zupt: float = 0.02         # m/s
    sig_tilt: float = 0.05         # m/s^2 (window-mean accel)
    sig_zrate: float = 0.002       # rad/s (window-mean gyro)
    # gap / blind propagation
    tau_v_gap: float = 2.0         # s, velocity decay during dropouts
    tau_w_gap: float = 0.5         # s, angular-rate decay during dropouts
    sig_v_gap: float = 1.0         # m/s/sqrt(s) process noise during dropouts
    sig_th_gap: float = 0.2        # rad/sqrt(s)
    tau_v_blind: float = 2.0
    tau_w_blind: float = 1.0
    w_hist_s: float = 0.5          # window for the last angular-rate estimate
    blind_s: float = 5.0
    # weak velocity prior (pseudo-measurement v = 0, applied every vprior_every valid samples); 0 = off
    sig_vprior: float = 0.5       # m/s; velocity-only update, swept on sims (EXP-003); re-tune on EuRoC
    vprior_every: int = 10


# ----------------------------------------------------------------------------- pass 1
@dataclass
class Scan:
    n: int = 0
    valid: np.ndarray = None
    last_valid: int = -1
    dt_ns: int = 5_000_000
    q0: np.ndarray = None
    bg0: np.ndarray = None
    static_init: bool = False


def row_valid(g, a, prev, P: Params):
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(a))):
        return False
    if np.all(g == 0.0) and np.all(a == 0.0):
        return False
    if np.any(np.abs(g) > P.w_max) or np.any(np.abs(a) > P.a_max):
        return False
    if prev is not None and np.array_equal(g, prev[0]) and np.array_equal(a, prev[1]):
        return False  # frozen / repeated sample
    return True


def gravity_to_quat(f):
    """Roll/pitch from a specific-force vector (body frame), yaw = 0. Returns q_WB (wxyz)."""
    roll = math.atan2(f[1], f[2])
    pitch = math.atan2(-f[0], math.sqrt(f[1] ** 2 + f[2] ** 2))
    cr, sr, cp, sp = math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    # q = Ry(pitch) * Rx(roll)
    return np.array([cp * cr, cp * sr, sp * cr, -sp * sr])


def scan(t_ns, gyro, acc, P: Params) -> Scan:
    s = Scan(n=len(t_ns))
    s.valid = np.zeros(s.n, bool)
    prev = None
    for i in range(s.n):
        s.valid[i] = row_valid(gyro[i], acc[i], prev, P)
        prev = (gyro[i], acc[i])
    vi = np.nonzero(s.valid)[0]
    s.last_valid = int(vi[-1]) if len(vi) else -1
    d = np.diff(t_ns[:2001])
    d = d[d > 0]
    s.dt_ns = int(np.median(d)) if len(d) else 5_000_000
    # initialisation buffer: valid rows within the first init_buf_s seconds
    t_lim = t_ns[0] + int(P.init_buf_s * 1e9)
    b = vi[t_ns[vi] <= t_lim] if len(vi) else vi
    gb, ab = gyro[b], acc[b]
    W = P.init_win
    s.q0, s.bg0 = None, np.zeros(3)
    for k in range(0, len(b) - W + 1, W // 4):
        an = np.linalg.norm(ab[k:k + W], axis=1)
        gn = np.linalg.norm(gb[k:k + W] - gb[k:k + W].mean(0), axis=1)
        if an.std() < P.static_acc_std and gn.mean() < P.static_gyr_mean and abs(an.mean() - G) < 0.5:
            s.q0 = gravity_to_quat(ab[k:k + W].mean(0))
            s.bg0 = gb[k:k + W].mean(0)
            s.static_init = True
            break
    if s.q0 is None:
        s.q0 = gravity_to_quat(ab.mean(0)) if len(b) else np.array([1.0, 0, 0, 0])
    return s


# ----------------------------------------------------------------------------- ESKF
IP, IV, ITH, IBA, IBG = 0, 3, 6, 9, 12


class ESKF:
    def __init__(self, q0, bg0, P: Params, static_init: bool):
        self.prm = P
        self.p = np.zeros(3); self.v = np.zeros(3); self.q = q0.copy()
        self.ba = np.zeros(3); self.bg = bg0.copy()
        d = np.zeros(15)
        d[IP:IP + 3] = 1e-6
        d[IV:IV + 3] = (P.p0_v if static_init else 1.0) ** 2
        d[ITH:ITH + 2] = P.p0_att ** 2; d[ITH + 2] = P.p0_yaw ** 2
        d[IBA:IBA + 3] = P.p0_ba ** 2; d[IBG:IBG + 3] = P.p0_bg ** 2
        self.P = np.diag(d)
        self.gvec = np.array([0.0, 0.0, -G])

    def propagate_imu(self, g0, a0, g1, a1, dt, sig_g, sig_a):
        R0 = quat_to_rot(self.q)
        w = 0.5 * (g0 + g1) - self.bg
        q1 = quat_mul(self.q, quat_exp(w * dt)); q1 /= np.linalg.norm(q1)
        R1 = quat_to_rot(q1)
        aW = 0.5 * (R0 @ (a0 - self.ba) + R1 @ (a1 - self.ba)) + self.gvec
        self.p = self.p + self.v * dt + 0.5 * aW * dt * dt
        self.v = self.v + aW * dt
        self.q = q1
        ab = 0.5 * (a0 + a1) - self.ba
        F = np.eye(15)
        RA = R0 @ skew(ab)
        F[IP:IP + 3, IV:IV + 3] = np.eye(3) * dt
        F[IP:IP + 3, ITH:ITH + 3] = -0.5 * RA * dt * dt
        F[IP:IP + 3, IBA:IBA + 3] = -0.5 * R0 * dt * dt
        F[IV:IV + 3, ITH:ITH + 3] = -RA * dt
        F[IV:IV + 3, IBA:IBA + 3] = -R0 * dt
        F[ITH:ITH + 3, ITH:ITH + 3] = quat_to_rot(quat_exp(w * dt)).T
        F[ITH:ITH + 3, IBG:IBG + 3] = -np.eye(3) * dt
        Qd = np.zeros(15)
        Qd[IV:IV + 3] = (sig_a * dt) ** 2
        Qd[ITH:ITH + 3] = (sig_g * dt) ** 2
        Qd[IBA:IBA + 3] = self.prm.sig_ba_rw ** 2 * dt
        Qd[IBG:IBG + 3] = self.prm.sig_bg_rw ** 2 * dt
        self.P = F @ self.P @ F.T + np.diag(Qd)

    def propagate_gap(self, w_body, dt, tau_v, sig_v, sig_th):
        """No IMU: decaying constant velocity, rotate with (already decayed) body rate."""
        dv = math.exp(-dt / tau_v)
        self.p = self.p + self.v * dt
        self.v = self.v * dv
        self.q = quat_mul(self.q, quat_exp(w_body * dt)); self.q /= np.linalg.norm(self.q)
        F = np.eye(15)
        F[IP:IP + 3, IV:IV + 3] = np.eye(3) * dt
        F[IV:IV + 3, IV:IV + 3] = np.eye(3) * dv
        Qd = np.zeros(15)
        Qd[IV:IV + 3] = sig_v ** 2 * dt
        Qd[ITH:ITH + 3] = sig_th ** 2 * dt
        Qd[IBA:IBA + 3] = self.prm.sig_ba_rw ** 2 * dt
        Qd[IBG:IBG + 3] = self.prm.sig_bg_rw ** 2 * dt
        self.P = F @ self.P @ F.T + np.diag(Qd)

    def update(self, H, r, Rm, only=None):
        """EKF update. `only` = (start, stop) restricts the correction to those error states
        (Schmidt-style); the Joseph form keeps P consistent for any gain."""
        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)
        if only is not None:
            Km = np.zeros_like(K); Km[only[0]:only[1]] = K[only[0]:only[1]]; K = Km
        dx = K @ r
        IKH = np.eye(15) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Rm @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.p = self.p + dx[IP:IP + 3]
        self.v = self.v + dx[IV:IV + 3]
        self.q = quat_mul(self.q, quat_exp(dx[ITH:ITH + 3])); self.q /= np.linalg.norm(self.q)
        self.ba = self.ba + dx[IBA:IBA + 3]
        self.bg = self.bg + dx[IBG:IBG + 3]

    def static_updates(self, a_mean, g_mean, P: Params):
        H = np.zeros((3, 15)); H[:, IV:IV + 3] = np.eye(3)
        self.update(H, -self.v, np.eye(3) * P.sig_zupt ** 2)
        R = quat_to_rot(self.q)
        fb = R.T @ np.array([0.0, 0.0, G])
        H = np.zeros((3, 15)); H[:, ITH:ITH + 3] = skew(fb); H[:, IBA:IBA + 3] = np.eye(3)
        self.update(H, a_mean - (self.ba + fb), np.eye(3) * P.sig_tilt ** 2)
        H = np.zeros((3, 15)); H[:, IBG:IBG + 3] = np.eye(3)
        self.update(H, g_mean - self.bg, np.eye(3) * P.sig_zrate ** 2)


# ----------------------------------------------------------------------------- pass 2
def run(t_ns, gyro, acc, P: Params = Params(), blind_mode="auto"):
    """Returns (t_ns_out int64 array, pos (N,3), quat wxyz (N,4))."""
    sc = scan(t_ns, gyro, acc, P)
    n = sc.n
    gyro = gyro.copy(); acc = acc.copy()
    f = ESKF(sc.q0, sc.bg0, P, sc.static_init)
    dt_nom = sc.dt_ns * 1e-9
    gap_dt = P.gap_factor * dt_nom
    alpha = 1.0 - math.exp(-dt_nom / P.noise_ema_tau)
    var_g = P.sig_g ** 2; var_a = P.sig_a ** 2

    out_t, out_p, out_q = [], [], []
    last = None            # last valid (gyro, acc) used for propagation
    w_hist = []            # recent bias-corrected angular rates
    w_hist_n = max(1, int(round(P.w_hist_s / dt_nom)))
    w_gap = np.zeros(3)
    in_gap = False
    win_g, win_a = [], []  # static-detector window
    static_count = 0
    good = (f.p.copy(), f.v.copy(), f.q.copy())

    def emit(i_t):
        q = f.q / np.linalg.norm(f.q)
        if out_q and np.dot(q, out_q[-1]) < 0:
            q = -q
        out_t.append(i_t); out_p.append(f.p.copy()); out_q.append(q)

    blind_start = sc.last_valid + 1
    for i in range(n):
        valid = bool(sc.valid[i])
        # spike repair with one-sample lookahead (only between valid neighbours)
        if valid and last is not None and i + 1 < n and sc.valid[i + 1]:
            for arr, floor, var in ((gyro, P.spike_floor_g, var_g), (acc, P.spike_floor_a, var_a)):
                T = max(P.spike_k * math.sqrt(var), floor)
                prev_v = last[0] if arr is gyro else last[1]
                for c in range(3):
                    x, xp, xn = arr[i, c], prev_v[c], arr[i + 1, c]
                    if abs(x - xp) > T and abs(x - xn) > T and abs(xn - xp) < 0.5 * T:
                        arr[i, c] = 0.5 * (xp + xn)
        if i == 0:
            if valid:
                last = (gyro[0].copy(), acc[0].copy())
            emit(t_ns[0])
            continue
        dt = (t_ns[i] - t_ns[i - 1]) * 1e-9
        if i >= blind_start:
            # ---- blind forecaster (after the IMU died)
            if i == blind_start:
                w_gap = np.mean(w_hist, axis=0) if w_hist else np.zeros(3)
            if dt > 0:
                w_gap = w_gap * math.exp(-dt / P.tau_w_blind)
                f.propagate_gap(w_gap, dt, P.tau_v_blind, P.sig_v_gap, P.sig_th_gap)
            emit(t_ns[i])
            continue
        if dt <= 0:
            emit(t_ns[i])
            continue
        if valid and last is not None and dt <= gap_dt and not in_gap:
            g1, a1 = gyro[i], acc[i]
            # adaptive noise from first differences (var(diff) = 2 sigma^2 for white noise)
            var_g = max(P.sig_g ** 2, (1 - alpha) * var_g + alpha * 0.5 * float(np.mean((g1 - last[0]) ** 2)))
            var_a = max(P.sig_a ** 2, (1 - alpha) * var_a + alpha * 0.5 * float(np.mean((a1 - last[1]) ** 2)))
            f.propagate_imu(last[0], last[1], g1, a1, dt, math.sqrt(var_g), math.sqrt(var_a))
        else:
            # ---- dropout: invalid row, or a timestamp gap
            if not in_gap:
                w_gap = np.mean(w_hist, axis=0) if w_hist else np.zeros(3)
                in_gap = True
            w_gap = w_gap * math.exp(-dt / P.tau_w_gap)
            f.propagate_gap(w_gap, dt, P.tau_v_gap, P.sig_v_gap, P.sig_th_gap)
            if valid:  # first valid sample after a gap: restart IMU integration from here
                in_gap = False
                win_g.clear(); win_a.clear(); static_count = 0
        if valid:
            last = (gyro[i].copy(), acc[i].copy())
            wc = gyro[i] - f.bg
            w_hist.append(wc)
            if len(w_hist) > w_hist_n:
                w_hist.pop(0)
            if P.sig_vprior > 0 and i % P.vprior_every == 0:
                H = np.zeros((3, 15)); H[:, IV:IV + 3] = np.eye(3)
                f.update(H, -f.v, np.eye(3) * P.sig_vprior ** 2, only=(IV, IV + 3))
            win_g.append(gyro[i].copy()); win_a.append(acc[i].copy())
            if len(win_g) > P.static_win:
                win_g.pop(0); win_a.pop(0)
            if len(win_g) == P.static_win:
                ga, aa = np.array(win_g), np.array(win_a)
                an = np.linalg.norm(aa, axis=1)
                gn = np.linalg.norm(ga - f.bg, axis=1)
                if an.std() < P.static_acc_std and gn.mean() < P.static_gyr_mean:
                    static_count += 1
                    if static_count % P.static_every == 0:
                        f.static_updates(aa.mean(0), ga.mean(0), P)
                else:
                    static_count = 0
        if not (np.all(np.isfinite(f.p)) and np.all(np.isfinite(f.q)) and np.all(np.isfinite(f.P))):
            f.p, f.v, f.q = good[0].copy(), good[1].copy(), good[2].copy()
            f.P = np.eye(15)
        else:
            good = (f.p.copy(), f.v.copy(), f.q.copy())
        emit(t_ns[i])

    # append the 5 s blind window when the file simply ends (no trailing invalid rows)
    if blind_mode == "append" or (blind_mode == "auto" and blind_start >= n and n > 0):
        w_gap = np.mean(w_hist, axis=0) if w_hist else np.zeros(3)
        k = 1
        dt = sc.dt_ns * 1e-9
        while k * sc.dt_ns <= int(P.blind_s * 1e9):
            w_gap = w_gap * math.exp(-dt / P.tau_w_blind)
            f.propagate_gap(w_gap, dt, P.tau_v_blind, P.sig_v_gap, P.sig_th_gap)
            emit(int(t_ns[-1]) + k * sc.dt_ns)
            k += 1
    return np.array(out_t, dtype=np.int64), np.array(out_p), np.array(out_q)


def write_output(path, t_ns, p, q):
    with open(path, "w", newline="\n") as fo:
        fo.write("# timestamp tx ty tz qx qy qz qw\n")
        for t, pp, qq in zip(t_ns, p, q):
            t = int(t)
            fo.write(f"{t // 1_000_000_000}.{t % 1_000_000_000:09d} {pp[0]:.6f} {pp[1]:.6f} {pp[2]:.6f} "
                     f"{qq[1]:.9f} {qq[2]:.9f} {qq[3]:.9f} {qq[0]:.9f}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default="estimated_trajectory.txt")
    ap.add_argument("--blind", default="auto", choices=["auto", "append", "none"])
    a = ap.parse_args()
    imu = load_imu(a.input)
    t, p, q = run(imu.t_ns, imu.gyro, imu.acc, Params(), a.blind)
    write_output(a.output, t, p, q)
    print(f"wrote {len(t)} poses to {a.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
