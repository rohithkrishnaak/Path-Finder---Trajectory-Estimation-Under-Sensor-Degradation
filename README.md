# Path Finder: Trajectory Estimation Under Sensor Degradation

**Samsung PRISM XR Hackathon 2026, Track 3.** IMU-only 6-DoF trajectory estimation, with 5 s of dead reckoning after the IMU fails.

The submission is one Python script, [`src/main.py`](src/main.py). It needs only numpy and has the learned model embedded. It stays inside both budgets: at most 4.9 MB of heap on the longest EuRoC flight (limit 8 MB) and about 5-6 ms of compute per IMU-second (limit 20 ms).

## Submission contents

| Checklist item | Where |
|---|---|
| **Submission package** (Track 3 slide 17: `team_name.zip` with `estimated_trajectory.txt`, `metadata.json`, `src/main.py`, `src/README.md`) | [`submission/VIT_tan-x.zip`](submission/), built by [`src/eval/make_bundle.py`](src/eval/make_bundle.py) |
| **Source code** | [`src/`](src/): `main.py` (submission), `eval/` (scorer, format checker, bench, bundler), `research/` (training, data tools), `tests/` |
| **Presentation** | [`VIT_tan-x_Submission.pptx`](VIT_tan-x_Submission.pptx) and [`.pdf`](VIT_tan-x_Submission.pdf) |
| **Video** (2-3 min) | [`demo/video/PathFinder_demo.mp4`](demo/video/PathFinder_demo.mp4): pipeline walkthrough with a Webots replay of a held-out flight ([how it is made](demo/README.md)) |
| **README** with reproducible setup, Docker files and requirements | this file, [`src/README.md`](src/README.md), [`src/Dockerfile`](src/Dockerfile), [`src/requirements.txt`](src/requirements.txt) |
| APK / SDK | Not applicable: Track 3 is a Python script |
| AI disclosure | To be added by the team |

## Repository layout

```
├── submission/VIT_tan-x.zip          the Track 3 bundle (slide 17)
│   ├── estimated_trajectory.txt      one TUM pose per input IMU row
│   ├── metadata.json                 runtime, peak heap, detected input layout, machine
│   └── src/main.py, src/README.md    entry point (model embedded) + how to run it
├── VIT_tan-x_Submission.pptx / .pdf  presentation (slide 24 naming)
├── demo/                             walkthrough video + Webots replay that produces it
├── src/                              main.py, Dockerfile, requirements, eval/, research/, tests/
├── weights/mix/ho_ens.bin            held-out ensembles behind every reported number
│   weights/la/ho120_ens.bin          (causal and look-ahead; trained without the five test flights)
├── docs/                             submission checklist, plan; organisers/ holds their FAQ, slides and degrade_imu.py
├── devlog.md                         every decision with alternatives and evidence
└── archive/                          earlier C++ runtime, rejected experiments
```

## Quick start

```sh
pip install numpy                                       # Python 3.9+, the only runtime dependency
python src/main.py input.zip -o estimated_trajectory.txt
```

The input can be the organisers' zipped folder (`imu0/data.csv`), a folder or the CSV itself. The output has exactly one TUM-format pose (`timestamp tx ty tz qx qy qz qw`) per input IMU row.

Docker, limited to a single CPU:
```sh
docker build -t pathfinder -f src/Dockerfile src
docker run --rm --cpus=1 -v "$PWD/data:/data" pathfinder /data/input.zip -o /data/estimated_trajectory.txt
```

Build the submission bundle from the official evaluation input:
```sh
python src/eval/make_bundle.py <official_input.zip> --team VIT_tan-x
```
The script writes the zip, checks its layout and the output format, then re-runs the bundled `main.py` from a clean folder. That re-run must reproduce the trajectory byte for byte.

## How it works

1. **Read and clean.** Column order, units and timestamp scale are auto-detected. Invalid, stuck and duplicate rows are flagged, and the IMU failure is located.
2. **Error-state Kalman filter** with 15 states. It uses midpoint integration, spike repair, online noise estimation and zero-velocity updates.
3. **Learned velocity.** A 3-member MLP ensemble reads the last 1 s of IMU and predicts the velocity and its uncertainty every 0.2 s. It was trained on EuRoC (drone) and TUM-VI (handheld) data degraded with the organisers' `degrade_imu.py`.
4. **Adaptive workspace prior.** It bounds slow drift. Short dropouts are bridged with interpolated IMU.
5. **Offline RTS smoother.** The FAQ allows non-causal processing, so after the forward pass a Rauch-Tung-Striebel smoother replays every velocity and prior update backwards. Each learned-velocity update then also corrects the path before it. Inside the smoother, a second, **look-ahead** network replaces each velocity measurement; it sees 0.4 s before and 0.6 s after the instant it estimates.
   - **Drone or handheld.** Rotor vibration in the accelerometer tells them apart. Handheld runs get a tighter workspace prior, which suits small rooms.
6. **Blind 5 s.** A decaying constant-velocity and turn-rate model runs from the final filter state.

Details are in [`src/README.md`](src/README.md). Every design decision, with the alternatives and the evidence, is in [`devlog.md`](devlog.md).

## Evaluation and results

Scoring follows the organisers' setup: the evo suite with **SE(3) Umeyama alignment**, then ATE RMSE and RPE. [`src/eval/score.py`](src/eval/score.py) reproduces it and splits ATE into the scored tiers:
- **Waypoints / checkpoints (30 pts):** every pose before the IMU fails.
- **Blind 5 s (20 pts):** the poses after the failure.
- **Dropouts (10 pts):** each gap plus the 1 s after it.

The other 40 points are for peak memory and runtime.

The table shows held-out sequences only: the ensembles used were trained without these flights (`weights/mix/ho_ens.bin`, look-ahead `weights/la/ho120_ens.bin`). The degradation is the official one, and each value is the mean over both input column orders (EXP-023). Values are ATE in metres.

| Dataset | Method | Checkpoints | Blind 5 s | Dropouts |
|---|---|---|---|---|
| EuRoC (MH_05, V1_02, V2_03) | **Path Finder** | **2.63** | **2.17** | **2.50** |
| | Before the smoother, look-ahead and platform switch (EXP-020) | 2.76 | 2.21 | 2.61 |
| | Physics-only filter (`--no-nn`) | 8.36 | 6.01 | 9.52 |
| | A pose that never moves | 3.60 | 2.55 | 3.51 |
| TUM-VI (room3, room6) | **Path Finder** | **1.11** | **0.41** | **0.87** |
| | Before the smoother, look-ahead and platform switch (EXP-020) | 1.33 | 0.65 | 1.09 |
| | Physics-only filter (`--no-nn`) | 2.91 | 1.90 | 1.69 |
| | A pose that never moves | 1.16 | 0.51 | 1.09 |

- **Both datasets:** Path Finder beats the fixed pose on every tier.
- **Rotation:** 1 s RPE is 1.1 deg/s on EuRoC, against 23 deg/s for the fixed pose.
- **Across all 30 held-out cases** (20 of them extra degradation seeds), the tier-weighted error falls from 341 to 312 (-8.5 %): the offline smoother (D-052), the look-ahead velocity (D-054) and the drone/handheld switch (D-055).

To reproduce:
```sh
python src/eval/bench.py --cases data/official_ho --weights weights/mix/ho_ens.bin --la-weights weights/la/ho120_ens.bin --la-lag 120   # held-out tiered ATE
python src/eval/bench.py --cases data/official --glob "MH_01*_o42.zip" --mem            # peak heap, ms per IMU-second
python src/tests/torture.py                                      # 22 edge cases
```

## More documentation
- [`docs/submission_checklist.md`](docs/submission_checklist.md): every rule from the slides and FAQ, how it is met, and what is left for the team.
- [`docs/plan.md`](docs/plan.md): plan and architecture.
- [`demo/README.md`](demo/README.md): the Webots replay and the video.
- [`archive/`](archive/README.md): the earlier C++ runtime, no longer allowed under the FAQ's Python-only rule.

## License
MIT ([`LICENSE`](LICENSE)). `src/research/official/degrade_imu.py` belongs to the hackathon organisers.
