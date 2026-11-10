"""Fetch only the EuRoC IMU + ground-truth CSVs from the ETH Research Collection.

  python fetch_euroc.py --out data/euroc [--only MH_01]

The outer archives (machine_hall 12.7 GB, vicon_room1/2 6 GB) are read with HTTP range requests:
only their central directory and each sequence's inner zip are transferred, never the ROS .bag
files (about half the archive). The inner zips are deflate-compressed, so each one (1-2.5 GB, mostly
images) is streamed to a temp file, the 3 needed members are extracted, and the temp file is deleted.
If an inner zip were stored uncompressed, its members would be range-read directly instead.
Result: data/euroc/<SEQ>/mav0/{imu0,state_groundtruth_estimate0}/... (a few MB per sequence).
"""
from __future__ import annotations

import argparse
import io
import os
import re
import time
import urllib.error
import urllib.request
import zipfile
import zlib

ARCHIVES = {  # ETH Research Collection, EuRoC MAV dataset (DOI 10.3929/ethz-b-000690084)
    "machine_hall.zip": "https://www.research-collection.ethz.ch/server/api/core/bitstreams/7b2419c1-62b5-4714-b7f8-485e5fe3e5fe/content",
    "vicon_room1.zip": "https://www.research-collection.ethz.ch/server/api/core/bitstreams/02ecda9a-298f-498b-970c-b7c44334d880/content",
    "vicon_room2.zip": "https://www.research-collection.ethz.ch/server/api/core/bitstreams/ea12bc01-3677-4b4c-853d-87c7870b8c44/content",
}
WANT = ("mav0/imu0/data.csv", "mav0/imu0/sensor.yaml", "mav0/state_groundtruth_estimate0/data.csv")
SEQ_RE = re.compile(r"((?:MH_0[1-5])|(?:V[12]_0[1-3]))_\w+")
UA = {"User-Agent": "pathfinder-research/0.1"}


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, with a small block cache."""

    def __init__(self, url, size=None, offset=0, block=1 << 20):
        self.url, self.base, self.block = url, offset, block
        self.pos = 0
        self.cache = {}
        self.bytes_fetched = 0
        if size is None:
            for attempt in range(8):
                try:
                    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        size = int(r.headers["Content-Range"].split("/")[-1])
                    break
                except Exception as e:
                    print(f"    size probe retry in {30 * (attempt + 1)}s ({e})")
                    time.sleep(30 * (attempt + 1))
            else:
                raise IOError("size probe failed repeatedly")
        self.size = size

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _get(self, start, end):  # inclusive range, absolute within this view
        for attempt in range(8):
            try:
                req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={self.base + start}-{self.base + end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                self.bytes_fetched += len(data)
                return data
            except Exception as e:  # rate limiting / transient errors
                wait = 30 * (attempt + 1)
                if isinstance(e, urllib.error.HTTPError) and e.headers.get("Retry-After", "").isdigit():
                    wait = int(e.headers["Retry-After"]) + 1
                print(f"    retry in {wait}s ({e})")
                time.sleep(wait)
        raise IOError("range request failed repeatedly")

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            b = self.pos // self.block
            if b not in self.cache:
                s = b * self.block
                self.cache[b] = self._get(s, min(s + self.block, self.size) - 1)
                if len(self.cache) > 64:
                    self.cache.pop(next(iter(self.cache)))
            chunk = self.cache[b][self.pos - b * self.block:][:n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def member_data_offset(f: HttpRangeFile, info: zipfile.ZipInfo) -> int:
    f.seek(info.header_offset)
    h = f.read(30)
    name_len, extra_len = int.from_bytes(h[26:28], "little"), int.from_bytes(h[28:30], "little")
    return info.header_offset + 30 + name_len + extra_len


def stream_member(url: str, start: int, csize: int, method: int, dst: str):
    """One HTTP range request for the member's compressed bytes, inflated on the fly to dst."""
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{start + csize - 1}"})
            got = 0
            dec = zlib.decompressobj(-15) if method == zipfile.ZIP_DEFLATED else None
            with urllib.request.urlopen(req, timeout=300) as r, open(dst, "wb") as fo:
                while chunk := r.read(1 << 22):
                    got += len(chunk)
                    fo.write(dec.decompress(chunk) if dec else chunk)
                if dec:
                    fo.write(dec.flush())
            if got != csize:
                raise IOError(f"short read {got}/{csize}")
            return
        except Exception as e:
            wait = 30 * (attempt + 1)
            if isinstance(e, urllib.error.HTTPError) and e.headers.get("Retry-After", "").isdigit():
                wait = int(e.headers["Retry-After"]) + 1
            print(f"    stream retry in {wait}s ({e})")
            time.sleep(wait)
    raise IOError("streaming failed repeatedly")


def extract_wanted(zf: zipfile.ZipFile, seq: str, out: str):
    for info in zf.infolist():
        n = info.filename.replace("\\", "/")
        for w in WANT:
            if n.endswith(w):
                dst = os.path.join(out, seq, w)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with zf.open(info) as src, open(dst, "wb") as fo:
                    fo.write(src.read())
                print(f"    {dst}  ({os.path.getsize(dst) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/euroc")
    ap.add_argument("--only", help="regex on sequence name, e.g. 'MH_01'")
    a = ap.parse_args()
    total = 0
    os.makedirs(a.out, exist_ok=True)
    for arch, url in ARCHIVES.items():
        outer = HttpRangeFile(url)
        print(f"{arch}: {outer.size / 1e9:.1f} GB remote")
        zo = zipfile.ZipFile(outer)
        for info in zo.infolist():
            m = SEQ_RE.search(os.path.basename(info.filename))
            if not (info.filename.endswith(".zip") and m) or os.path.basename(info.filename).startswith("._"):
                continue
            seq = m.group(0)
            if a.only and not re.search(a.only, seq):
                continue
            if os.path.exists(os.path.join(a.out, seq, WANT[2])):
                print(f"  {seq}: already present")
                continue
            print(f"  {seq}: inner zip {info.file_size / 1e9:.2f} GB, compress_type={info.compress_type}")
            if info.compress_type == zipfile.ZIP_STORED:
                inner = HttpRangeFile(url, size=info.file_size, offset=member_data_offset(outer, info))
                extract_wanted(zipfile.ZipFile(inner), seq, a.out)
                total += inner.bytes_fetched
            else:  # compressed inner zip: stream it to a temp file (images included), extract, delete
                t0 = time.time()
                stream_member(url, member_data_offset(outer, info), info.compress_size, info.compress_type,
                              os.path.join(a.out, f"_{seq}.zip"))
                with zipfile.ZipFile(os.path.join(a.out, f"_{seq}.zip")) as zi:
                    extract_wanted(zi, seq, a.out)
                os.remove(os.path.join(a.out, f"_{seq}.zip"))
                total += info.compress_size
                print(f"    streamed {info.compress_size / 1e9:.2f} GB in {time.time() - t0:.0f}s")
        total += outer.bytes_fetched
    print(f"done; ~{total / 1e6:.1f} MB transferred")


if __name__ == "__main__":
    main()
