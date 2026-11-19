"""Torture tests for main.py: odd encodings and edge cases must never break the output contract.

Every case runs `python main.py <input> -o out.txt` as the evaluator would, then:
  * format_check must pass (exactly one pose per input row, exact timestamps, unit quaternions, LF, finite);
  * cases that only re-encode the reference input (column order, units, timestamp unit, separators, header,
    containers) must give the same trajectory as the reference (max position difference < 1 mm);
  * the out-of-core input path used for long recordings must reproduce the reference byte for byte.

Usage (from the repo root):  python src/tests/torture.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "research", "official"))
from common import G, IMU_HEADER, load_tum, official_degrade  # noqa: E402
from format_check import check  # noqa: E402
import sim_traj  # noqa: E402

MAIN = os.path.join(ROOT, "main.py")
HDR_GA = IMU_HEADER
HDR_AG = "timestamp,ax,ay,az,gx,gy,gz"


def write_csv(path, t, g, a, header=HDR_GA, order="ga", crlf=False, sep=",", blank=None, nan=None, tfmt=str):
    nl = "\r\n" if crlf else "\n"
    with open(path, "w", newline="") as f:
        if header:
            f.write(header + nl)
        for i in range(len(t)):
            ts = tfmt(t[i])
            if blank is not None and blank[i]:
                f.write(ts + sep * 6 + nl)
                continue
            vals = (list(g[i]) + list(a[i])) if order == "ga" else (list(a[i]) + list(g[i]))
            if nan is not None and nan[i]:
                vals = [float("nan")] * 6
            f.write(ts + sep + sep.join(f"{v:.6f}" for v in vals) + nl)


def cases(tmp):
    d = sim_traj.generate(40.0, "quad", seed=11)
    t, g, a = d["t_ns"].astype(np.int64), d["gyro"], d["acc"]
    n = len(t)
    same, other = {}, {}

    def mk(name, **kw):
        p = os.path.join(tmp, name + ".csv")
        write_csv(p, kw.pop("t", t), kw.pop("g", g), kw.pop("a", a), **kw)
        return p

    ref = mk("ref")
    # --- re-encodings of the reference: the trajectory must not change
    same["accel_first_header"] = mk("ag", header=HDR_AG, order="ag")
    same["accel_first_noheader"] = mk("agnh", header=None, order="ag")
    same["crlf_noheader"] = mk("crlf", header=None, crlf=True)
    same["semicolon"] = mk("semi", sep=";")
    same["accel_in_g"] = mk("g", a=a / G, order="ag", header=HDR_AG)
    same["ts_seconds"] = mk("tsec", tfmt=lambda x: f"{int(x) // 10**9}.{int(x) % 10**9:09d}")
    same["ts_microseconds"] = mk("tus", t=t // 1000 * 1000, tfmt=lambda x: str(int(x) // 1000))
    z = os.path.join(tmp, "nested.zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(ref, "team/MH_XX/mav0/imu0/data.csv")
        zf.writestr("team/MH_XX/mav0/imu0/sensor.yaml", "rate_hz: 200\n")
    same["nested_zip"] = z
    folder = os.path.join(tmp, "folder_in")
    os.makedirs(os.path.join(folder, "mav0", "imu0"))
    shutil.copy(ref, os.path.join(folder, "mav0", "imu0", "data.csv"))
    same["folder"] = folder
    # --- degradations: only the contract is checked
    keep = np.ones(n, bool); keep[2000:2040] = False; keep[3001] = False; keep[5000:5400] = False
    other["missing_rows"] = mk("missing", t=t[keep], g=g[keep], a=a[keep])
    g3, a3 = g.copy(), a.copy(); g3[4000:4008] = g3[3999]; a3[4000:4008] = a3[3999]
    other["stuck_rows"] = mk("stuck", g=g3, a=a3)
    rep = np.ones(n, int); rep[[100, 2500, 6000, n - 10]] = 2
    idx = np.repeat(np.arange(n), rep)
    other["duplicated_rows"] = mk("dup", t=t[idx], g=g[idx], a=a[idx])
    g4, a4 = g.copy(), a.copy(); g4[n - 1000:] = 0; a4[n - 1000:] = 0
    other["zero_tail_5s"] = mk("zerotail", g=g4, a=a4)
    g5 = g.copy(); g5[1000::97, 1] += 1.0
    a5 = a.copy(); a5[1500::89, 0] += 6.0
    other["spikes"] = mk("spikes", g=g5, a=a5)
    blank = np.zeros(n, bool); blank[3000:3030] = True
    other["blank_fields"] = mk("blank", blank=blank)
    nan = np.zeros(n, bool); nan[6000:6100] = True
    other["nan_rows"] = mk("nan", nan=nan)
    t6 = t.copy(); t6[5001], t6[5002] = t6[5002], t6[5001]
    other["swapped_ts"] = mk("swap", t=t6)
    other["single_row"] = mk("single", t=t[:1], g=g[:1], a=a[:1])
    other["all_invalid"] = mk("allnan", nan=np.ones(n, bool))
    dm = sim_traj.generate(20.0, "quad", seed=12, static_head=0.0)
    other["no_static_start"] = mk("nostatic", t=dm["t_ns"].astype(np.int64), g=dm["gyro"], a=dm["acc"])
    try:  # the organisers' script, unmodified
        out, _ = official_degrade(t, g, a, "script", cols=("timestamp", "ax", "ay", "az", "gx", "gy", "gz"))
        p = os.path.join(tmp, "official.csv")
        out.to_csv(p, index=False, header=True, lineterminator="\n")
        other["official_degrade_imu"] = p
    except ImportError:
        print("note: pandas not installed, skipping the official degrade_imu.py case")
    return ref, same, other


def run(inp, out):
    r = subprocess.run([sys.executable, MAIN, inp, "-o", out], capture_output=True, text=True)
    return r.returncode, r.stderr.strip()


def main():
    tmp = tempfile.mkdtemp(prefix="pf_torture_")
    fails = 0
    try:
        ref, same, other = cases(tmp)
        ref_out = os.path.join(tmp, "ref.txt")
        rc, err = run(ref, ref_out)
        if rc != 0:
            sys.exit(f"reference run failed: {err}")
        R = load_tum(ref_out)
        for group, items in (("same", same), ("other", other)):
            for name, path in items.items():
                out = os.path.join(tmp, name + ".txt")
                rc, err = run(path, out)
                ok_fmt, errs = check(out, path, verbose=False) if rc == 0 else (False, [err[-300:]])
                note = ""
                ok_same = True
                if group == "same" and ok_fmt:
                    A = load_tum(out)
                    if len(A.t) != len(R.t):
                        ok_same, note = False, f"rows {len(A.t)} vs {len(R.t)}"
                    else:
                        dp = float(np.abs(A.p - R.p).max())
                        ok_same, note = dp < 1e-3, f"max |dp| vs reference {dp:.1e} m"
                ok = ok_fmt and ok_same
                fails += not ok
                print(f"{'PASS' if ok else 'FAIL'}  {name:22s} format={'ok' if ok_fmt else errs[:2]}  {note}")
        # out-of-core path (long recordings): must give exactly the same file as the in-RAM path
        sys.path.insert(0, ROOT)
        import main as pf
        st = pf.run(ref, os.path.join(tmp, "spill.txt"), pf.Params(), pf.VelNet(), spill_rows=4096,
                    la_net=pf.load_la())
        same = open(os.path.join(tmp, "spill.txt"), "rb").read() == open(ref_out, "rb").read()
        ok = same and st["out_of_core"]
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {'out_of_core_identical':22s} memory-mapped input reproduces the reference byte for byte")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{fails} failing case(s)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
