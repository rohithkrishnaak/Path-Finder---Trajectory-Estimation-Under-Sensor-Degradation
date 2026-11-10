#!/usr/bin/env python3
"""Path Finder entry point (standard library only).

Runs the compiled C++ runtime (building it first if needed), which is what meets the
<= 8 MB peak-RAM and <= 20 ms per IMU-second budgets. `--engine python` runs the numpy
reference implementation instead; it produces identical output but is not budget-compliant.

  python main.py <input.zip|folder|imu0/data.csv> [-o estimated_trajectory.txt] [--blind auto|append|none]
                 [--metrics metrics.json] [--engine cpp|python]
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "pathfinder.exe" if os.name == "nt" else "pathfinder")


def build():
    sh = shutil.which("sh") or shutil.which("bash")
    if not sh:
        sys.exit("error: need 'sh' and g++ to build the runtime (see README.md), or use --engine python")
    subprocess.run([sh, os.path.join(HERE, "build.sh")], check=True)


def main(argv):
    engine = "cpp"
    if "--engine" in argv:
        k = argv.index("--engine")
        engine = argv[k + 1] if k + 1 < len(argv) else "cpp"
        argv = argv[:k] + argv[k + 2:]
    if engine == "python":
        ref = os.path.join(HERE, "research", "ref_pipeline.py")
        args = [a for a in argv if a != "--metrics"]
        return subprocess.call([sys.executable, ref] + args)
    if not os.path.exists(BIN):
        build()
    return subprocess.call([BIN] + argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
