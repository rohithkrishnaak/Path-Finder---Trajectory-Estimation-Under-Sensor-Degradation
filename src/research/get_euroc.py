"""Extract only what we need (IMU + ground truth) from downloaded EuRoC MAV zips.

The original ETH host (robotics.ethz.ch/~asl-datasets) is offline; download the zips from the
ETH Research Collection EuRoC entry (machine_hall.zip, vicon_room1.zip, vicon_room2.zip, which
contain per-sequence zips), or any mirror, into data/raw/. Then:

  python get_euroc.py --raw data/raw --out data/euroc

Result: data/euroc/<SEQ>/mav0/{imu0,state_groundtruth_estimate0}/data.csv  (a few MB per sequence).
Nested zips are spooled to a temp file (not RAM); images are never extracted.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
import zipfile

WANT = ("imu0/data.csv", "imu0/sensor.yaml", "state_groundtruth_estimate0/data.csv")
SEQ_RE = re.compile(r"(MH_0[1-5]_\w+|V[12]_0[1-3]_\w+)")


def extract_from(zf: zipfile.ZipFile, out: str, label: str):
    for name in zf.namelist():
        if name.lower().endswith(".zip"):
            m = SEQ_RE.search(name)
            # spool the (GB-sized) inner zip to disk instead of RAM
            with zf.open(name) as inner, tempfile.TemporaryFile() as tmp:
                shutil.copyfileobj(inner, tmp, 1 << 20)
                tmp.seek(0)
                extract_from(zipfile.ZipFile(tmp), out, m.group(1) if m else label)
            continue
        norm = name.replace("\\", "/")
        if any(norm.endswith(w) for w in WANT):
            m = SEQ_RE.search(norm)
            seq = m.group(1) if m else label
            rel = norm[norm.index("mav0/"):] if "mav0/" in norm else norm
            dst = os.path.join(out, seq, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with zf.open(name) as src, open(dst, "wb") as f:
                f.write(src.read())
            print("  ", dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for fn in sorted(os.listdir(a.raw)):
        if fn.lower().endswith(".zip"):
            print(fn)
            label = os.path.splitext(fn)[0]
            with zipfile.ZipFile(os.path.join(a.raw, fn)) as zf:
                extract_from(zf, a.out, label)


if __name__ == "__main__":
    main()
