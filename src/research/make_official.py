"""Build evaluation-style test cases with the OFFICIAL organiser degradation script (official/degrade_imu.py).

For each EuRoC sequence this writes, in --out-dir:
  <name>.zip        imu0/data.csv degraded by degrade_imu.degrade() (the input our main.py receives)
  <name>.gt.txt     TUM ground truth interpolated at every output row timestamp (inside the GT span)
  <name>.meta.json  t_fail_ns (start of the zeroed 5 s window), detected dropouts (timestamp gaps), script stats

The official script is imported unchanged; only the seed can be varied (the organisers' runs use SEED = 42).

Column order (the FAQ does not say which one the evaluation file uses, so we test both):
  --order script   timestamp, accel xyz, gyro xyz  (the order documented in degrade_imu.py)       [default]
  --order euroc    the raw EuRoC file fed straight in (timestamp, gyro xyz, accel xyz). The script then
                   treats the gyro columns as accel and vice versa, so the noise lands on the "wrong" sensor.

Usage:
  python make_official.py --euroc data/euroc --out-dir data/official [--seqs MH_01_easy,V1_02_medium]
                          [--seed 42] [--order script|euroc] [--crop]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile

import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (IMU_HEADER, Traj, find_gt_csv, interp_traj, load_euroc_gt, load_imu,  # noqa: E402
                    official_degrade, write_tum)

HDR_EUROC = IMU_HEADER.split(",")                       # t, gyro xyz, accel xyz
HDR_SCRIPT = [HDR_EUROC[0]] + HDR_EUROC[4:7] + HDR_EUROC[1:4]   # t, accel xyz, gyro xyz


def make_case(seq_dir, out_dir, seed=42, order="script", crop=False, max_level=None, gamma=None, name=None):
    seq = os.path.basename(os.path.normpath(seq_dir))
    imu = load_imu(seq_dir)
    gt = load_euroc_gt(find_gt_csv(seq_dir))
    t_ns, g, a = imu.t_ns, imu.gyro, imu.acc
    if crop:
        t = t_ns * 1e-9
        k = (t >= gt.t[0]) & (t <= gt.t[-1])
        t_ns, g, a = t_ns[k], g[k], a[k]
    cols = HDR_SCRIPT if order == "script" else HDR_EUROC
    out, stats = official_degrade(t_ns, g, a, order, seed, max_level, gamma, cols)
    cfg = stats["config"]

    name = name or f"{seq}_{'o' if order == 'script' else 'e'}{seed}"
    os.makedirs(out_dir, exist_ok=True)
    zpath = os.path.join(out_dir, f"{name}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{name}/mav0/imu0/data.csv", out.to_csv(index=False, header=True, lineterminator="\n"))

    t_out = out[cols[0]].to_numpy(dtype=np.int64)
    ts = t_out * 1e-9
    m, pos, quat = interp_traj(gt, ts)
    write_tum(os.path.join(out_dir, f"{name}.gt.txt"), Traj(ts[m], pos, quat))

    t0 = int(t_ns[0])
    T = (int(t_ns[-1]) - t0) * 1e-9
    t_fail_ns = t0 + int(round((T - cfg["failure_seconds"]) * 1e9))
    d = np.diff(t_out)
    dt_nom = float(np.median(d[d > 0]))
    gaps = np.nonzero(d > 2.5 * dt_nom)[0]
    drops = [dict(start_ns=int(t_out[i]), end_ns=int(t_out[i + 1]), style="missing") for i in gaps]
    meta = dict(name=name, seq=seq, seed=seed, order=order, crop=crop, source="official degrade_imu.py",
                t0_ns=t0, t_end_ns=int(t_ns[-1]), t_fail_ns=t_fail_ns,
                n_rows_out=int(len(out)), n_gt_rows=int(m.sum()), dropouts=drops,
                longest_gap_s=float(d.max() * 1e-9), stats=stats)
    with open(os.path.join(out_dir, f"{name}.meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--euroc", required=True, help="folder with the EuRoC sequences (each has mav0/imu0)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seqs", default="all")
    ap.add_argument("--seed", type=int, default=42, help="42 = the organisers' seed")
    ap.add_argument("--order", choices=["script", "euroc"], default="script")
    ap.add_argument("--crop", action="store_true", help="keep only IMU rows inside the ground-truth span")
    ap.add_argument("--max-level", type=float)
    ap.add_argument("--gamma", type=float)
    a = ap.parse_args()
    seqs = sorted(s for s in os.listdir(a.euroc) if os.path.isdir(os.path.join(a.euroc, s))) if a.seqs == "all" \
        else a.seqs.split(",")
    for s in seqs:
        m = make_case(os.path.join(a.euroc, s), a.out_dir, a.seed, a.order, a.crop, a.max_level, a.gamma)
        st = m["stats"]
        print(f"{m['name']:26s} rows {st['rows_in']}->{st['rows_out']}  dropped {st['dropped_total']}"
              f" (bursts {st['dropped_burst_gaps']})  gaps>2.5dt {len(m['dropouts'])}  longest gap {m['longest_gap_s']:.3f}s"
              f"  dup {st['duplicated_rows']}  stuck {st['stuck_samples']}  spikes {st['spikes']}", flush=True)


if __name__ == "__main__":
    main()
