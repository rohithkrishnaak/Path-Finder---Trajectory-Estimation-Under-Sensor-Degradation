"""Torture tests: odd input encodings and edge cases must never break the output contract,
and the C++ binary must match the Python reference.

Usage (from repo root):  python src/tests/torture.py [--bin src/pathfinder.exe]
"""
from __future__ import annotations

import argparse
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
from common import IMU_HEADER, load_tum  # noqa: E402
from format_check import check  # noqa: E402
import sim_traj  # noqa: E402

REF = os.path.join(ROOT, "research", "ref_pipeline.py")


def write_csv(path, t, g, a, header=True, crlf=False, sep=",", blank=None, nan=None):
    nl = "\r\n" if crlf else "\n"
    with open(path, "w", newline="") as f:
        if header:
            f.write(IMU_HEADER + nl)
        for i in range(len(t)):
            if blank is not None and blank[i]:
                f.write(f"{t[i]}" + sep * 6 + nl)
                continue
            vals = list(g[i]) + list(a[i])
            if nan is not None and nan[i]:
                vals = [float("nan")] * 6
            f.write(str(int(t[i])) + sep + sep.join(f"{v:.10g}" for v in vals) + nl)


def cases(tmp):
    d = sim_traj.generate(30.0, "quad", seed=11)
    t, g, a = d["t_ns"], d["gyro"], d["acc"]
    n = len(t)
    out = {}

    def mk(name, **kw):
        p = os.path.join(tmp, name + ".csv")
        write_csv(p, kw.pop("t", t), kw.pop("g", g), kw.pop("a", a), **kw)
        return p

    out["plain_csv"] = mk("plain")
    out["crlf_noheader"] = mk("crlf", header=False, crlf=True)
    out["semicolon"] = mk("semi", sep=";")
    blank = np.zeros(n, bool); blank[3000:3400] = True
    out["blank_rows"] = mk("blank", blank=blank)
    nan = np.zeros(n, bool); nan[n - 1000:] = True
    out["nan_tail"] = mk("nantail", nan=nan)
    keep = np.ones(n, bool); keep[2000:2400] = False
    out["missing_rows"] = mk("missing", t=t[keep], g=g[keep], a=a[keep])
    g2, a2 = g.copy(), a.copy(); g2[4000:4300] = 0; a2[4000:4300] = 0
    out["zero_rows"] = mk("zero", g=g2, a=a2)
    g3, a3 = g.copy(), a.copy(); g3[4000:4300] = g3[3999]; a3[4000:4300] = a3[3999]
    out["frozen_rows"] = mk("frozen", g=g3, a=a3)
    t4 = t.copy(); t4[5000] = t4[4999]; t4[5001], t4[5002] = t4[5002], t4[5001]
    out["dup_and_swap_ts"] = mk("dupswap", t=t4)
    g5 = g.copy(); g5[1000::97, 1] += 25.0
    out["spikes"] = mk("spikes", g=g5)
    out["single_row"] = mk("single", t=t[:1], g=g[:1], a=a[:1])
    out["all_invalid"] = mk("allnan", nan=np.ones(n, bool))
    dm = sim_traj.generate(20.0, "quad", seed=12, static_head=0.0)
    out["no_static_start"] = mk("nostatic", t=dm["t_ns"], g=dm["gyro"], a=dm["acc"])
    # container variants
    z = os.path.join(tmp, "nested.zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out["plain_csv"], "team/MH_XX/mav0/imu0/data.csv")
        zf.writestr("team/MH_XX/mav0/imu0/sensor.yaml", "rate_hz: 200\n")
    out["nested_zip"] = z
    folder = os.path.join(tmp, "folder_in")
    os.makedirs(os.path.join(folder, "mav0", "imu0"))
    shutil.copy(out["plain_csv"], os.path.join(folder, "mav0", "imu0", "data.csv"))
    out["folder"] = folder
    return out


LEGACY = ["--set", "sig_vprior=0.5", "--set", "tau_v_gap=2", "--set", "tau_v_blind=2"]


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stderr.strip()


def main():
    ap = argparse.ArgumentParser()
    default_bin = os.path.join(ROOT, "pathfinder.exe" if os.name == "nt" else "pathfinder")
    ap.add_argument("--bin", default=default_bin)
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="pf_torture_")
    fails = 0
    try:
        for name, path in cases(tmp).items():
            oc, op = os.path.join(tmp, name + ".cpp.txt"), os.path.join(tmp, name + ".py.txt")
            # parity is against the numpy twin of the baseline filter: learned velocity off, legacy parameters
            rc1, e1 = run([a.bin, path, "-o", oc, "--no-nn"] + LEGACY)
            on = os.path.join(tmp, name + ".nn.txt")
            rc3, e3 = run([a.bin, path, "-o", on])
            ok_nn, errs_nn = check(on, path, verbose=False) if rc3 == 0 else (False, [e3])
            rc2, e2 = run([sys.executable, REF, path, "-o", op])
            ok_fmt, errs = check(oc, path, verbose=False) if rc1 == 0 else (False, [e1])
            par = "n/a"
            ok_par = False
            if rc1 == 0 and rc2 == 0:
                A, B = load_tum(oc), load_tum(op)
                if len(A.t) == len(B.t):
                    dp = float(np.abs(A.p - B.p).max()) if len(A.t) else 0.0
                    dq = float(np.abs(A.q - B.q).max()) if len(A.t) else 0.0
                    par = f"dp={dp:.1e} dq={dq:.1e}"
                    ok_par = dp < 1e-4 and dq < 1e-4
                else:
                    par = f"rows {len(A.t)} vs {len(B.t)}"
            ok = ok_fmt and ok_par and ok_nn
            fails += not ok
            print(f"{'PASS' if ok else 'FAIL'}  {name:18s} format={'ok' if ok_fmt else errs[:2]} nn-format={'ok' if ok_nn else errs_nn[:2]}  parity: {par}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{fails} failing case(s)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
