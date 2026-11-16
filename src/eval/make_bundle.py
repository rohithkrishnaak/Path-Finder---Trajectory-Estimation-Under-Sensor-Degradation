"""Build the Track 3 submission bundle: <team>.zip with estimated_trajectory.txt, metadata.json and src/.

  python src/eval/make_bundle.py <input.zip|folder|imu0/data.csv> --team my_team [--out my_team.zip]

Runs src/main.py (self-contained, embedded weights) on the given input, validates the output with format_check
(exactly one pose per input row), measures CPU time per IMU-second and the peak heap (tracemalloc, the FAQ's
"algorithm / heap level"), and writes EXACTLY the layout of the Track 3 slide "Submission Bundle":

  team_name.zip
  ├── estimated_trajectory.txt
  ├── metadata.json
  └── src/
      ├── main.py
      └── README.md

The bundle's main.py is then re-run from a clean temporary folder (nothing else next to it) and its output
must be byte-identical to estimated_trajectory.txt.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)
import main as pf  # noqa: E402
from bench import run_case  # noqa: E402
from format_check import check  # noqa: E402

EXPECTED = ("estimated_trajectory.txt", "metadata.json", "src/main.py", "src/README.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--team", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    out_zip = a.out or f"{a.team}.zip"
    tmp = tempfile.mkdtemp()
    est = os.path.join(tmp, "estimated_trajectory.txt")
    st, cpu, peak_mb = run_case(a.input, est, pf.Params())   # same measurement as bench.py
    ok, _ = check(est, a.input)
    if not ok:
        sys.exit("format check failed; not building the bundle")
    meta = {
        "team": a.team,
        "entry": "python src/main.py <input.zip> -o estimated_trajectory.txt  (Python 3.9+, numpy; weights embedded)",
        "rows": st["rows"],
        "poses": st["poses"],
        "imu_seconds": round(st["imu_seconds"], 3),
        "runtime_ms_per_imu_second": round(1000.0 * cpu / st["imu_seconds"], 3),
        "peak_heap_mb_tracemalloc": round(peak_mb, 3),
        "threads": 1,
        "input_detected": {k: st[k] for k in ("order", "timestamp_unit", "accel_unit", "gyro_unit")},
        "machine": f"{platform.platform()} / {platform.processor()} / Python {platform.python_version()}",
    }
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(est, "estimated_trajectory.txt")
        z.writestr("metadata.json", json.dumps(meta, indent=1))
        z.write(os.path.join(SRC, "main.py"), "src/main.py")
        z.write(os.path.join(SRC, "README.md"), "src/README.md")
    # verify: exact layout, and the bundled main.py alone reproduces the trajectory
    with zipfile.ZipFile(out_zip) as z:
        names = sorted(z.namelist())
        assert names == sorted(EXPECTED), names
        clean = tempfile.mkdtemp()
        z.extractall(clean)
    rerun = os.path.join(clean, "rerun.txt")
    subprocess.run([sys.executable, os.path.join(clean, "src", "main.py"), os.path.abspath(a.input), "-o", rerun],
                   check=True, cwd=clean, capture_output=True)
    if open(rerun, "rb").read() != open(os.path.join(clean, "estimated_trajectory.txt"), "rb").read():
        sys.exit("bundled main.py does not reproduce estimated_trajectory.txt")
    print("layout:", ", ".join(names), "| clean re-run of src/main.py reproduces the trajectory: yes")
    print(json.dumps(meta, indent=1))
    print("wrote", out_zip)


if __name__ == "__main__":
    main()
