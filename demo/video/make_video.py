"""Assemble the 2-3 minute pipeline walkthrough video (slide 14 deliverable) from cards + the Webots replay.

  python demo/video/make_video.py [--webots demo/video/build/webots_V1_02.mp4] [--out demo/video/PathFinder_demo.mp4]

Needs ffmpeg. By default the copy that ships with Webots is used (with Webots' DLL folders on PATH);
set FFMPEG=<path> to use another one. The Webots clip comes from demo/webots (see demo/README.md).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BUILD = os.path.join(HERE, "build")
W, H, FPS = 1280, 720, 30
BG, FG, MUTED, ACC, GOOD, BAD, PURP = "#0f1720", "#f1f5f9", "#94a3b8", "#ff9f1c", "#5cff7a", "#ff5050", "#d39bff"

# Held-out results (EXP-023, devlog.md): ensembles trained without these flights (weights/mix/ho_ens.bin, weights/la/ho120_ens.bin),
# mean over both input column orders, scored with src/eval/score.py (se3).
RESULTS = {
    "EuRoC drone (MH_05, V1_02, V2_03)": dict(pf=(2.63, 2.17, 2.50), phys=(8.36, 6.01, 9.52), z0=(3.60, 2.55, 3.51)),
    "TUM-VI handheld (room3, room6)": dict(pf=(1.11, 0.41, 0.87), phys=(2.91, 1.90, 1.69), z0=(1.16, 0.51, 1.09)),
}


def ffmpeg():
    exe = os.environ.get("FFMPEG")
    env = dict(os.environ)
    if not exe:
        w = r"C:\Program Files\Webots"
        exe = os.path.join(w, "msys64", "mingw64", "bin", "ffmpeg.exe")
        env["PATH"] = os.pathsep.join([os.path.join(w, "msys64", "mingw64", "bin"), os.path.join(w, "msys64", "usr", "bin"),
                                       os.path.join(w, "lib", "webots"), os.path.join(w, "msys64", "mingw64", "bin", "cpp"),
                                       r"C:\Windows\system32", r"C:\Windows"])  # nothing else: other DLLs clash
    return exe, env


def run_ff(args):
    exe, env = ffmpeg()
    subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-y"] + args, check=True, env=env)


def canvas():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    return fig, ax


def header(ax, kicker, title):
    ax.text(70, 70, kicker.upper(), color=ACC, fontsize=15, fontweight="bold")
    ax.text(70, 125, title, color=FG, fontsize=25, fontweight="bold")
    ax.plot([70, 160], [150, 150], color=ACC, lw=4)


def bullets(ax, items, x, y, width=60, size=17, gap=50, color=FG):
    for it in items:
        lines = textwrap.wrap(it, width)
        ax.text(x, y, "\u2022", color=ACC, fontsize=size, fontweight="bold")
        for j, ln in enumerate(lines):
            ax.text(x + 28, y + j * size * 1.55, ln, color=color, fontsize=size)
        y += gap + (len(lines) - 1) * size * 1.55
    return y


def save(fig, name):
    p = os.path.join(BUILD, name + ".png")
    fig.savefig(p, facecolor=BG)
    plt.close(fig)
    return p


def card_title():
    fig, ax = canvas()
    ax.text(W / 2, 250, "PATH FINDER", color=FG, fontsize=60, fontweight="bold", ha="center")
    ax.text(W / 2, 330, "Trajectory Estimation Under Sensor Degradation", color=ACC, fontsize=26, ha="center")
    ax.text(W / 2, 410, "Samsung PRISM XR Hackathon 2026  \u00b7  Track 3", color=MUTED, fontsize=19, ha="center")
    ax.text(W / 2, 470, "IMU-only 6-DoF trajectory + 5 s dead reckoning, in one Python file", color=FG, fontsize=18,
            ha="center")
    ax.text(W / 2, 560, "Team tan-x  ·  VIT, Vellore", color=FG, fontsize=20, fontweight="bold", ha="center")
    ax.text(W / 2, 640, "Pipeline walkthrough", color=MUTED, fontsize=15, ha="center")
    return save(fig, "c01_title")


def card_problem():
    fig, ax = canvas()
    header(ax, "The problem", "GPS is jammed. Then the IMU degrades. Then it dies.")
    bullets(ax, [
        "Only input: a degraded IMU CSV (EuRoC format, 200 Hz). Noise and bias drift grow, spikes, stuck values.",
        "Dropouts: rows are simply missing, so the timestamps jump.",
        "The last 5 s: the IMU has failed. We must still give a pose for every row, to reach the target.",
        "Plain double integration drifts as t\u00b2: a 0.05 m/s\u00b2 bias is 250 m of error after 100 s.",
    ], 70, 230, width=82)
    ax.text(70, 590, "Budgets", color=ACC, fontsize=18, fontweight="bold")
    ax.text(70, 635, "\u2264 8 MB heap   \u00b7   \u2264 20 ms per 1 s of IMU   \u00b7   one CPU thread, offline   \u00b7   "
            "one pose per IMU row", color=FG, fontsize=18)
    return save(fig, "c02_problem")


def card_pipeline():
    fig, ax = canvas()
    header(ax, "How it works", "src/main.py: one file, numpy only, model embedded")
    boxes = [
        ("READ", "zip / folder / csv\ncolumn order + units\nauto-detected"),
        ("CLEAN", "invalid, stuck and\nduplicate rows;\nfind the failure"),
        ("ESKF", "15-state error-state\nKalman filter, spike\nrepair, ZUPT"),
        ("LEARNED\nVELOCITY", "3-MLP ensemble, 1 s\nof IMU \u2192 velocity\n+ its uncertainty"),
        ("PRIOR", "adaptive workspace\nprior bounds the\nslow drift"),
        ("BLIND 5 s", "decaying constant-\nvelocity + turn-rate\nmodel"),
    ]
    bw, gap, y0 = 180, 22, 230
    for i, (t, d) in enumerate(boxes):
        x = 55 + i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, y0), bw, 250, boxstyle="round,pad=4,rounding_size=14",
                                    fc="#1e293b", ec=ACC if i == 3 else "#334155", lw=2.5))
        ax.text(x + bw / 2, y0 + 50, t, color=ACC if i == 3 else FG, fontsize=16, fontweight="bold", ha="center",
                va="center")
        ax.text(x + bw / 2, y0 + 165, d, color=MUTED, fontsize=12.5, ha="center", va="center", linespacing=1.5)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + bw + gap - 2, y0 + 125), xytext=(x + bw + 4, y0 + 125),
                        arrowprops=dict(arrowstyle="-|>", color=ACC, lw=2))
    bullets(ax, [
        "Offline RTS smoother with a look-ahead velocity net (0.4 s before, 0.6 s after); drone / handheld auto-detected.",
        "The net is trained on EuRoC (drone) + TUM-VI (handheld), degraded with the organisers' own script.",
        "Output: estimated_trajectory.txt in TUM format, exactly one pose per input row.",
    ], 70, 545, width=95, size=16, gap=40)
    return save(fig, "c03_pipeline")


def card_replay_intro():
    fig, ax = canvas()
    header(ax, "Live replay in Webots", "A flight the model never saw: EuRoC V1_02")
    bullets(ax, [
        "The drone flies the hidden ground truth.",
        "Orange: Path Finder's estimate from the degraded IMU only, aligned with SE(3) like the evo evaluator.",
        "Red: the same filter without the learned velocity.",
        "After the IMU fails, the replay slows down: purple is the 5 s dead-reckoned prediction.",
    ], 70, 230, width=85)
    return save(fig, "c04_replay")


def card_results():
    fig, ax = canvas()
    header(ax, "Results", "Held-out flights, official degradation (ATE, m)")
    cols = ["", "Before failure\n(waypoints, 30)", "Blind 5 s\n(20)", "Dropouts\n(10)"]
    xs = [70, 600, 830, 1040]
    for x, c in zip(xs, cols):
        ax.text(x, 215, c, color=MUTED, fontsize=14, va="center", linespacing=1.3)
    y = 290
    for ds, r in RESULTS.items():
        ax.text(70, y, ds, color=ACC, fontsize=17, fontweight="bold")
        y += 45
        for key, lab, col in (("pf", "Path Finder", GOOD), ("phys", "Physics-only filter", BAD),
                              ("z0", "A pose that never moves", MUTED)):
            ax.text(100, y, lab, color=col, fontsize=16, fontweight="bold" if key == "pf" else "normal")
            for x, v in zip(xs[1:], r[key]):
                ax.text(x + 40, y, f"{v:.2f}", color=col, fontsize=16, fontweight="bold" if key == "pf" else "normal")
            y += 38
        y += 25
    ax.text(70, 670, "Rotation RPE 1.1 deg/s on EuRoC vs 23 for a fixed pose.  Path Finder beats it on every tier "
            "of both datasets.", color=FG, fontsize=15)
    return save(fig, "c05_results")


def card_topdown():
    fig, ax = canvas()
    header(ax, "Hardest held-out flight", "EuRoC MH_05 difficult: 111 s in a machine hall")
    img = plt.imread(os.path.join(ROOT, "demo", "figures", "MH_05_difficult_topdown.png"))
    ih, iw = img.shape[:2]
    h = 520
    w = iw * h / ih
    ax.imshow(img, extent=(W / 2 - w / 2, W / 2 + w / 2, 175 + h, 175))
    return save(fig, "c06_topdown")


def card_budget(stats):
    fig, ax = canvas()
    header(ax, "Inside the budget", "Measured, not estimated")
    tiles = [("4.9 MB", "peak heap (tracemalloc)\nlimit 8 MB"), ("~5.5 ms", "per 1 s of IMU, one thread\nlimit 20 ms"),
             ("1 : 1", "one pose per input row,\nexact timestamps"), ("22 / 22", "torture tests pass\n(format, units, order)")]
    for i, (big, small) in enumerate(tiles):
        x = 70 + i * 290
        ax.add_patch(FancyBboxPatch((x, 230), 260, 230, boxstyle="round,pad=4,rounding_size=14", fc="#1e293b",
                                    ec="#334155", lw=2))
        ax.text(x + 130, 310, big, color=ACC, fontsize=34, fontweight="bold", ha="center", va="center")
        ax.text(x + 130, 400, small, color=MUTED, fontsize=14, ha="center", va="center", linespacing=1.5)
    bullets(ax, [
        "Beyond 5 minutes the input and the pose buffer are memory-mapped, so the heap stays bounded.",
        "The bundled src/main.py is re-run from a clean folder and must reproduce the trajectory byte for byte.",
    ], 70, 540, width=95, size=16, gap=45)
    return save(fig, "c07_budget")


def card_run():
    fig, ax = canvas()
    header(ax, "Run it yourself", "Clone, pip install numpy, run")
    code = ("pip install numpy\n"
            "python src/main.py input.zip -o estimated_trajectory.txt\n\n"
            "python src/eval/make_bundle.py input.zip --team VIT_tan-x\n"
            "python src/eval/score.py --gt gt.txt --est estimated_trajectory.txt\n"
            "python src/tests/torture.py")
    ax.add_patch(FancyBboxPatch((70, 210), 1140, 260, boxstyle="round,pad=4,rounding_size=12", fc="#020617",
                                ec="#334155", lw=2))
    ax.text(100, 240, code, color=GOOD, fontsize=17, family="monospace", va="top", linespacing=1.6)
    ax.text(70, 540, "Limitations", color=ACC, fontsize=18, fontweight="bold")
    bullets(ax, [
        "IMU-only dead reckoning still drifts about 3 m per 10 s; margins over a fixed pose are honest but modest.",
        "The blind 5 s gains least: after the failure there is no future data for the smoother to use.",
    ], 70, 590, width=100, size=15, gap=40)
    return save(fig, "c08_run")


def card_end():
    fig, ax = canvas()
    ax.text(W / 2, 300, "Thank you", color=FG, fontsize=50, fontweight="bold", ha="center")
    ax.text(W / 2, 380, "Path Finder  \u00b7  Track 3  \u00b7  Samsung PRISM XR Hackathon 2026", color=ACC, fontsize=20,
            ha="center")
    ax.text(W / 2, 440, "github.com/rohithkrishnaak/Path-Finder---Trajectory-Estimation-Under-Sensor-Degradation",
            color=MUTED, fontsize=14, ha="center")
    return save(fig, "c09_end")


def still_clip(png, seconds, out):
    run_ff(["-loop", "1", "-framerate", str(FPS), "-t", f"{seconds}", "-i", png,
            "-vf", f"scale={W}:{H},format=yuv420p,fade=t=in:st=0:d=0.4,fade=t=out:st={seconds - 0.4}:d=0.4",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-r", str(FPS), "-video_track_timescale", "15360", out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--webots", default=os.path.join(BUILD, "webots_V1_02.mp4"))
    ap.add_argument("--out", default=os.path.join(HERE, "PathFinder_demo.mp4"))
    a = ap.parse_args()
    os.makedirs(BUILD, exist_ok=True)
    stats = json.load(open(os.path.join(ROOT, "demo", "webots", "controllers", "replay", "V1_02_medium.stats.json")))

    plan = [(card_title(), 6), (card_problem(), 14), (card_pipeline(), 17), (card_replay_intro(), 9),
            ("WEBOTS", None), (card_results(), 15), (card_topdown(), 9), (card_budget(stats), 11), (card_run(), 12),
            (card_end(), 5)]
    parts = []
    for i, (src, sec) in enumerate(plan):
        out = os.path.join(BUILD, f"part{i:02d}.mp4")
        if src == "WEBOTS":
            run_ff(["-i", a.webots, "-vf", f"scale={W}:{H},fps={FPS},format=yuv420p,fade=t=in:st=0:d=0.4",
                    "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-video_track_timescale", "15360", out])
        else:
            still_clip(src, sec, out)
        parts.append(out)
    lst = os.path.join(BUILD, "parts.txt")
    with open(lst, "w") as f:
        for p in parts:
            f.write(f"file '{p.replace(os.sep, '/')}'\n")
    run_ff(["-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-r", str(FPS), "-movflags", "+faststart", a.out])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
