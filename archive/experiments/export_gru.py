"""Pack trained GRU members (npz from train_gru.py) into the flat float32 file main.py reads (*.gru.bin).

  python export_gru.py weights/gru/a.npz weights/gru/b.npz ... --out weights/x.gru.bin
"""
import argparse

import numpy as np

KEYS = ("gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0", "out.weight", "out.bias")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    parts, H = [], None
    for p in a.npz:
        d = np.load(p)
        h = d["gru.weight_hh_l0"].shape[1]
        assert H in (None, h), "members must share the hidden size"
        H = h
        parts += [d[k].astype("<f4").ravel() for k in KEYS]
    flat = np.concatenate([np.array([len(a.npz), H], "<f4")] + parts)
    flat.tofile(a.out)
    print(f"{len(a.npz)} members, hidden {H}, {flat.nbytes} bytes -> {a.out}")


if __name__ == "__main__":
    main()
