"""Degradation simulator modelled on the Track 3 description (organizer script not yet available).

Turns a clean EuRoC-layout sequence into the hackathon input:
  progressive noise growth, escalating bias drift, dropouts (~2 s) in several encodings,
  spikes, optional saturation, and a terminal IMU failure followed by a 5 s blind window.

Outputs (in --out-dir):
  <name>.zip            zipped folder containing .../imu0/data.csv  (the evaluation input)
  <name>.gt.txt         TUM ground truth over the whole span incl. the blind window
  <name>.meta.json      every random choice (dropout intervals, failure time, styles, params)

Every parameter is randomised from --seed (domain randomisation, see devlog D-007) unless fixed by a flag.

Usage:
  python degrade.py --seq data/euroc/MH_01_easy --out-dir data/degraded --seed 3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import zipfile

import numpy as np

from common import Traj, find_gt_csv, load_euroc_gt, load_imu, write_imu_csv, write_tum

STYLES = ["missing", "nan", "empty", "zero", "frozen"]
BASE_SIG_G = 1.6968e-4 * math.sqrt(200.0)   # per-sample gyro noise at 200 Hz (EuRoC ADIS16448)
BASE_SIG_A = 2.0e-3 * math.sqrt(200.0)


def sample_params(rng: np.random.Generator, overrides: dict) -> dict:
    p = dict(
        noise_alpha_g=float(rng.uniform(2, 30)),      # final noise = base * (1 + alpha)
        noise_alpha_a=float(rng.uniform(2, 30)),
        noise_pow=float(rng.uniform(1.0, 3.0)),
        bias_ramp_g=float(rng.uniform(0.0, 0.05)),    # rad/s at failure
        bias_ramp_a=float(rng.uniform(0.0, 0.5)),     # m/s^2 at failure
        bias_pow=float(rng.uniform(1.0, 3.0)),
        bias_rw_g=float(10 ** rng.uniform(-5, -3.3)),  # rad/s^2/sqrt(Hz) final intensity
        bias_rw_a=float(10 ** rng.uniform(-3.5, -2)),
        n_dropouts=int(rng.poisson(rng.uniform(2, 6))),
        dropout_min=0.3, dropout_max=2.5,
        dropout_styles=STYLES,
        spike_rate=float(rng.uniform(0, 0.5)),        # spikes/s at failure
        saturate=bool(rng.random() < 0.25),
        sat_gyro=float(rng.uniform(3, 8)),             # rad/s
        sat_acc=float(rng.uniform(3, 8) * 9.81),
        failure_style=str(rng.choice(STYLES)),
        blind_s=5.0,
        crlf=bool(rng.random() < 0.2),
        header=bool(rng.random() < 0.9),
        zip_root=str(rng.choice(["mav0/imu0", "imu0", "{name}/mav0/imu0", "{name}/imu0"])),
    )
    p.update({k: v for k, v in overrides.items() if v is not None})
    return p


def place_dropouts(rng, t, t_lo, t_hi, n, dmin, dmax, styles):
    """Non-overlapping intervals, biased towards later (more jammed) mission time."""
    out = []
    for _ in range(50 * max(n, 1)):
        if len(out) >= n:
            break
        u = rng.beta(2.0, 1.2)
        start = t_lo + u * (t_hi - t_lo)
        dur = rng.uniform(dmin, dmax)
        end = min(start + dur, t_hi)
        if end - start < dmin * 0.5:
            continue
        if any(not (end + 1.0 < s or start > e + 1.0) for s, e, _ in out):
            continue
        out.append((start, end, str(rng.choice(styles))))
    return sorted(out)


def degrade(seq_dir: str, out_dir: str, seed: int, name: str | None = None, **overrides) -> dict:
    rng = np.random.default_rng(seed)
    imu = load_imu(seq_dir)
    gt = load_euroc_gt(find_gt_csv(seq_dir))
    name = name or f"{os.path.basename(os.path.normpath(seq_dir))}_s{seed}"
    P = sample_params(rng, overrides)

    # keep only IMU rows covered by ground truth
    t = imu.t_ns * 1e-9
    keep = (t >= gt.t[0]) & (t <= gt.t[-1])
    t_ns, gyro, acc = imu.t_ns[keep], imu.gyro[keep].copy(), imu.acc[keep].copy()
    t = t_ns * 1e-9
    n = len(t_ns)
    fs = 1.0 / np.median(np.diff(t))
    t0, t_end = t[0], t[-1]
    t_fail = t_end - P["blind_s"]
    u = np.clip((t - t0) / (t_fail - t0), 0.0, 1.0)

    # progressive noise: total std = base * (1 + alpha * u^pow)
    for arr, base, alpha in ((gyro, BASE_SIG_G, P["noise_alpha_g"]), (acc, BASE_SIG_A, P["noise_alpha_a"])):
        mult = 1.0 + alpha * u ** P["noise_pow"]
        extra = base * np.sqrt(np.maximum(mult ** 2 - 1.0, 0.0))
        arr += rng.normal(0, 1, (n, 3)) * extra[:, None]

    # escalating bias drift: deterministic ramp + random walk with growing intensity
    for arr, ramp, rw in ((gyro, P["bias_ramp_g"], P["bias_rw_g"]), (acc, P["bias_ramp_a"], P["bias_rw_a"])):
        d = rng.normal(0, 1, 3); d /= np.linalg.norm(d)
        drift = ramp * (u ** P["bias_pow"])[:, None] * d
        steps = rng.normal(0, 1, (n, 3)) * (rw * u * math.sqrt(1.0 / fs))[:, None]
        arr += drift + np.cumsum(steps, axis=0)

    # spikes (single-sample outliers)
    spikes = []
    lam = P["spike_rate"] * u / fs
    for i in np.nonzero(rng.random(n) < lam)[0]:
        ch = int(rng.integers(0, 6))
        mag = rng.uniform(10, 100) * (BASE_SIG_G if ch < 3 else BASE_SIG_A) * rng.choice([-1, 1])
        (gyro if ch < 3 else acc)[i, ch % 3] += mag
        spikes.append(int(i))

    if P["saturate"]:
        np.clip(gyro, -P["sat_gyro"], P["sat_gyro"], out=gyro)
        np.clip(acc, -P["sat_acc"], P["sat_acc"], out=acc)

    # dropouts (not in the first 5 s, which we leave for initialisation like a real mission start)
    drops = place_dropouts(rng, t, t0 + 5.0, t_fail - 1.0, P["n_dropouts"], P["dropout_min"], P["dropout_max"],
                           P["dropout_styles"])
    remove = np.zeros(n, bool)
    blank = np.zeros(n, bool)

    def apply(mask, style):
        if style == "missing":
            remove[mask] = True
        elif style == "nan":
            gyro[mask] = np.nan; acc[mask] = np.nan
        elif style == "empty":
            blank[mask] = True
        elif style == "zero":
            gyro[mask] = 0.0; acc[mask] = 0.0
        elif style == "frozen":
            idx = np.nonzero(mask)[0]
            if len(idx):
                j = max(idx[0] - 1, 0)
                gyro[mask] = gyro[j]; acc[mask] = acc[j]

    for s, e, style in drops:
        apply((t >= s) & (t < e), style)
    fail_mask = t > t_fail
    apply(fail_mask, P["failure_style"])

    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp()
    try:
        csv = os.path.join(tmp, "data.csv")
        k = ~remove
        write_imu_csv(csv, t_ns[k], gyro[k], acc[k], header=P["header"], crlf=P["crlf"], blank_mask=blank[k])
        root = P["zip_root"].format(name=name)
        zpath = os.path.join(out_dir, f"{name}.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv, f"{root}/data.csv")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ground truth (TUM) over the whole span, interpolated at IMU timestamps for exact association
    from scipy.spatial.transform import Rotation, Slerp
    tg = np.clip(t, gt.t[0], gt.t[-1])
    pos = np.column_stack([np.interp(tg, gt.t, gt.p[:, i]) for i in range(3)])
    rot = Slerp(gt.t, Rotation.from_quat(gt.q[:, [1, 2, 3, 0]]))(tg).as_quat()
    write_tum(os.path.join(out_dir, f"{name}.gt.txt"), Traj(t_ns * 1e-9, pos, rot[:, [3, 0, 1, 2]]))

    meta = dict(
        name=name, seq=os.path.basename(os.path.normpath(seq_dir)), seed=seed, params=P,
        fs=float(fs), n_rows_clean=int(n), n_rows_out=int((~remove).sum()),
        t0_ns=int(t_ns[0]), t_end_ns=int(t_ns[-1]),
        t_fail_ns=int(round(t_fail * 1e9)),
        failure_truncated=P["failure_style"] == "missing",
        dropouts=[dict(start_ns=int(round(s * 1e9)), end_ns=int(round(e * 1e9)), style=st) for s, e, st in drops],
        spikes=len(spikes),
    )
    with open(os.path.join(out_dir, f"{name}.meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True, help="EuRoC-layout sequence folder (contains mav0/imu0 + ground truth)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name")
    ap.add_argument("--failure-style", choices=STYLES)
    ap.add_argument("--n-dropouts", type=int)
    ap.add_argument("--clean", action="store_true", help="no noise/bias/spikes/dropouts (still blind tail)")
    a = ap.parse_args()
    ov = dict(failure_style=a.failure_style, n_dropouts=a.n_dropouts)
    if a.clean:
        ov.update(noise_alpha_g=0.0, noise_alpha_a=0.0, bias_ramp_g=0.0, bias_ramp_a=0.0, bias_rw_g=0.0,
                  bias_rw_a=0.0, n_dropouts=0, spike_rate=0.0, saturate=False)
    m = degrade(a.seq, a.out_dir, a.seed, a.name, **ov)
    print(json.dumps({k: m[k] for k in ("name", "n_rows_out", "failure_truncated", "dropouts")}, indent=1))


if __name__ == "__main__":
    main()
