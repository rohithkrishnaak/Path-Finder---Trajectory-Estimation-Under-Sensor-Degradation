"""Fetch only IMU + ground truth from TUM-VI sequences (EuRoC export), without the camera images.

Each archive is a 1.5-3 GB tar that is mostly images. It ends with a small dso/ folder holding the calibrated
IMU (dso/imu.txt) and the motion-capture ground truth of the IMU frame (dso/gt_imu.csv). The server supports
HTTP range requests, so this downloads only the last --tail-mb of the archive and extracts those two files.

  python fetch_tumvi.py --out data/tumvi [--seqs room1,room2,...]

Result, in the EuRoC layout the rest of the tooling reads:
  data/tumvi/<seq>/mav0/imu0/data.csv     #timestamp [ns], w xyz [rad/s], a xyz [m/s^2]
  data/tumvi/<seq>/mav0/mocap0/data.csv   #timestamp [ns], p xyz, q wxyz
TUM-VI is handheld (BMI160 at 200 Hz, mocap at 120 Hz). Rooms 1-6 have ground truth for the whole sequence.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

BASE = "https://vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-{s}_512_16.tar"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMU_HEADER  # noqa: E402


def fetch(seq, out, tail_mb):
    p_imu = os.path.join(out, seq, "mav0", "imu0", "data.csv")
    p_gt = os.path.join(out, seq, "mav0", "mocap0", "data.csv")
    if os.path.exists(p_imu) and os.path.exists(p_gt):
        print(seq, "already present", flush=True)
        return
    url = BASE.format(s=seq)
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        url, size = r.geturl(), int(r.headers["Content-Length"])
    start = max(0, (size - tail_mb * (1 << 20)) // 512 * 512)       # tar headers sit on 512-byte boundaries
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{size - 1}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        buf = r.read()
    got = {}
    for off in range(0, len(buf) - 512, 512):
        h = buf[off:off + 512]
        if h[257:262] != b"ustar":
            continue
        name = h[:100].split(b"\0", 1)[0].decode(errors="replace")
        n = int(h[124:136].split(b"\0", 1)[0].strip() or b"0", 8)
        for key in ("dso/imu.txt", "dso/gt_imu.csv"):
            if name.endswith(key) and off + 512 + n <= len(buf):
                got[key] = buf[off + 512:off + 512 + n].decode()
    if len(got) < 2:
        print(f"{seq}: FAILED, found only {sorted(got)} in the last {tail_mb} MB", file=sys.stderr, flush=True)
        return
    os.makedirs(os.path.dirname(p_imu), exist_ok=True)
    os.makedirs(os.path.dirname(p_gt), exist_ok=True)
    rows = [ln.split() for ln in got["dso/imu.txt"].splitlines() if ln and not ln.startswith("#")]
    with open(p_imu, "w", newline="\n") as f:
        f.write(IMU_HEADER + "\n")
        f.writelines(",".join(r) + "\n" for r in rows)
    with open(p_gt, "w", newline="\n") as f:
        f.write(got["dso/gt_imu.csv"].replace("\r\n", "\n"))
    print(f"{seq}: {len(rows)} IMU rows, {got['dso/gt_imu.csv'].count(chr(10)) - 1} ground-truth rows", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seqs", default="room1,room2,room3,room4,room5,room6")
    ap.add_argument("--tail-mb", type=int, default=24)
    a = ap.parse_args()
    for s in a.seqs.split(","):
        try:
            fetch(s, a.out, a.tail_mb)
        except Exception as e:
            print(f"{s}: FAILED {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
