# Path Finder: IMU-only 6-DoF trajectory under sensor degradation

Samsung PRISM XR Hackathon 2026, **Track 3**.

The input is a degraded IMU stream produced by the organisers' `degrade_imu.py`:
- noise, bias drift and scale-factor drift that grow over the run;
- spikes and stuck values;
- missing rows (dropouts) and duplicated rows;
- a final 5 s in which the IMU has failed (values zeroed, timestamps kept).

Path Finder estimates the 6-DoF pose at every input row, including the blind 5 s. It is a single Python script, `main.py`, as the Track 3 FAQ requires.

| Budget (official) | Limit | Measured (longest EuRoC sequence, MH_01, 35.6k rows) |
|---|---|---|
| Peak memory (algorithm / heap, `tracemalloc`) | ≤ 8 MB | **2.77 MB**, including the 0.45 MB embedded model |
| Compute | ≤ 20 ms per 1 s of IMU | **3.7 ms** (Linux, one pinned core); 4.6-6.3 ms (Windows) |
| Long recordings | same limits | a 23-minute file peaks at 4.2 MB (rows beyond 5 minutes are memory-mapped from a temp file) |
| Threads | 1 | 1 |
| Network | offline | offline |

## Quick start
```sh
pip install numpy                                   # the only dependency (Python 3.9+)
python src/main.py <input.zip|folder|data.csv> -o estimated_trajectory.txt
```
`main.py` is self-contained: the learned model is embedded in it (int8 weights as base64, about 0.6 MB of the file). No other file is needed next to it.

The script prints one line to stderr: the number of poses, the time per IMU-second, the detected column order and whether the model loaded. Options:
- `--metrics m.json`: writes the run statistics.
- `--no-nn`: physics-only filter. Not recommended: it drifts badly.
- `--weights path`: load an external float32 model file instead of the embedded one.
- `--order auto|ag|ga`, `--accel-unit auto|ms2|g`, `--gyro-unit auto|rad|deg`: force the input layout. Normally auto-detected.
- `--set name=value`: override a filter parameter (see `Params` in `main.py`).

Docker (the `Dockerfile` is in the repository's `src/`; its whole content is shown here, so it can also be recreated from this README alone):
```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir "numpy>=1.24"
WORKDIR /app
COPY main.py README.md ./
ENTRYPOINT ["python", "main.py"]
```
```sh
docker build -t pathfinder -f src/Dockerfile src
docker run --rm --cpus=1 -v "$PWD/data:/data" pathfinder /data/input.zip -o /data/estimated_trajectory.txt
```

## Submission bundle (Track 3 slide "Submission Bundle & Evaluation")
`python src/eval/make_bundle.py <input.zip> --team CollegeName_TeamName` writes exactly:
```
CollegeName_TeamName.zip
├── estimated_trajectory.txt     # the generated trajectory
├── metadata.json                # runtime, peak heap, detected input layout, machine
└── src/
    ├── main.py                  # entry point (self-contained, model embedded)
    └── README.md                # this file
```
It then checks the layout, runs the format check (one pose per input row, exact timestamps), and re-runs the bundled `src/main.py` alone from a clean folder. That re-run must reproduce `estimated_trajectory.txt` byte for byte.

## Input / output contract
- **Input**: a zip (any nesting; the shortest path ending in `imu0/data.csv` is used, otherwise any `.csv`), a folder, or the CSV itself.
  - Seven columns: a timestamp, then accelerometer and gyroscope triples in either order. `degrade_imu.py` documents `t, accel, gyro`; raw EuRoC is `t, gyro, accel`. The accelerometer is the triple whose median norm is about g; header names break ties.
  - Accel in m/s² or g; gyro in rad/s or deg/s; timestamps in ns, µs, ms or s.
  - The header is optional, with or without `#`. CRLF, `;`/tab/space separators and empty or NaN fields are tolerated.
- **Output**: `estimated_trajectory.txt` in TUM format, `timestamp tx ty tz qx qy qz qw`, timestamps in seconds.
  - **Exactly one pose per input row, in input order.** Duplicated rows and the invalid blind window are included. Nothing is appended (FAQ).
  - Timestamps are printed from integer nanoseconds, so they are exact.
  - The quaternion is a unit `q_WB` with hemisphere continuity.
  - The world frame is gravity-aligned (z up), with its origin at the first pose and initial yaw 0. The evaluator aligns with SE(3) Umeyama.

## How it works
1. **Read** the CSV into float32 / int64 arrays (32 bytes per row). Detect the column order and units. Past 60,000 rows (5 minutes) the parsed rows are written to temp files and memory-mapped, so the heap stays flat for long recordings; the output is byte-identical either way.
2. **Validity.** A row is invalid if it is non-finite, all zero (the failed IMU), out of range, or identical to the previous row (a stuck or duplicated sample). The blind window starts after the last valid row.
3. **Initialise** roll, pitch and gyro bias from the first static second. Yaw starts at 0.
4. **Error-state Kalman filter** (15 states: position, velocity, attitude, accel bias, gyro bias):
   - Midpoint IMU integration; the covariance is propagated every 8 steps.
   - Spike repair using one row of lookahead.
   - IMU noise estimated online.
   - **Short gaps (≤ 0.5 s) are bridged by integrating linearly interpolated IMU.**
   - Static detection triggers zero-velocity, gravity-tilt and zero-rate updates.
   - **A learned body-frame velocity measurement every 0.2 s.** This is a 3-member MLP ensemble (450-256-128-6) that reads the last 1 s of IMU and the gravity direction, and predicts velocity plus its uncertainty. It is trained on EuRoC (drone) and TUM-VI room (handheld) ground truth with IMU degraded by the organisers' `degrade_imu.py` (16 random seeds, severities and ramps per sequence, both column orders), rebuilt on the time grid exactly as `main.py` bridges gaps.
   - **An adaptive bounded-workspace prior.** Once per second the filter looks at the path so far: its centroid, and its spread on each axis. After a 20 s warm-up it adds a pseudo-measurement "position = that centroid", with a per-axis standard deviation equal to the spread (between 0.3 m and 5 m). This bounds the slow drift of dead reckoning and adapts to the size of the space, from a 3 m room to a 10 m hall. `--set pos_adapt=0` keeps a fixed 5 m prior at the start position; `--set sig_pos=0` turns the prior off.
5. **Long gaps and the blind 5 s**: a decaying constant-velocity / turn-rate model, started from the filter state.

The decisions, alternatives and evidence are in `../devlog.md`. D-030 to D-035 cover the move to Python and the official degradation; D-036 to D-039 cover the embedded model and the retraining; D-040 to D-043 cover the int8 model, the workspace prior and the ideas that were tested and rejected; D-044 to D-046 cover the TUM-VI training data and the adaptive prior; D-047 to D-049 cover the outside review (single-pass prior, out-of-core input, the look-ahead model that was rejected).

## Results (official `degrade_imu.py`, seed 42, SE(3) alignment, metres)
Held-out sequences only: three EuRoC drone flights (MH_05, V1_02, V2_03) and two TUM-VI handheld rooms (room3, room6). The models scored here were trained with exactly the shipped recipe but without these five sequences. Each number is the mean over both input variants (documented `t, accel, gyro` order, and the raw EuRoC-format file fed straight into the script).

| Dataset | Model | Before failure | Blind 5 s | Around dropouts | Rotation RPE (deg/s) |
|---|---|---|---|---|---|
| EuRoC (drone) | **Path Finder (current)** | **2.76** | 2.21 | **2.61** | 1.13 |
| EuRoC (drone) | Trained on EuRoC only, fixed 5 m prior | 2.99 | 2.02 | 3.18 | 1.18 |
| EuRoC (drone) | A pose that never moves | 3.60 | 2.55 | 3.51 | 23.5 |
| TUM-VI (handheld) | **Path Finder (current)** | **1.34** | **0.70** | **1.10** | **2.64** |
| TUM-VI (handheld) | Trained on EuRoC only, fixed 5 m prior | 9.56 | 13.12 | 9.42 | 2.58 |
| TUM-VI (handheld) | A pose that never moves | 1.16 | 0.51 | 1.09 | 41.1 |

- Training on both datasets is what makes the handheld case work at all: a drone-only model is off by about 9.5 m in a 3 m room.
- On EuRoC, Path Finder beats the fixed pose on every tier. On the small TUM-VI rooms its position error is still slightly above a fixed pose, while its rotation is 15 times better.
- IMU-only dead reckoning drifts about 3 m in 10 s on these sequences even with the learned velocity (D-041), which is why the margins over a fixed pose are modest.

## Repository layout
```
src/
  main.py                    the submission: complete pipeline (numpy only), model embedded
  weights/vel.bin            the same ensemble as a float32 file (source for research/embed_weights.py)
  requirements.txt           numpy (runtime); research extras are in research/requirements.txt
  Dockerfile
  research/
    official/degrade_imu.py  the organisers' degradation script (unmodified)
    make_official.py         builds test cases with it (zip + interpolated ground truth + meta)
    common.py                I/O + rotation utilities for the tooling
    train_vel.py             training the velocity network (PyTorch, CPU is enough) incl. official-style copies
    embed_weights.py         writes a weight file into main.py (int8 + per-row scale, base64)
    sim_traj.py, degrade.py  synthetic trajectories and our earlier randomised simulator
    get_euroc.py, fetch_euroc.py      EuRoC download / extraction
    fetch_tumvi.py           TUM-VI rooms: IMU + ground truth only (24 MB per sequence instead of 1.7 GB)
  eval/
    bench.py                 runs main.py over a case folder: tiered ATE, ms per IMU-second, peak heap
    score.py                 ATE/RPE identical to evo (verified), all alignment modes, per-tier metrics
    format_check.py          strict output-contract checker (exactly one pose per input row)
    make_bundle.py           builds <team>.zip (estimated_trajectory.txt, metadata.json, src/)
  tests/torture.py           22 edge cases: format, invariance to column order / units / encodings, out-of-core path
../archive/                  the earlier C++17 runtime and the rejected-experiment scripts, kept for reference
```

## Reproduce
```sh
pip install -r src/research/requirements.txt         # numpy + pandas are enough for the steps below
python src/research/fetch_tumvi.py --out data/tumvi
python src/research/make_official.py --euroc data/euroc --out-dir data/official
python src/research/make_official.py --euroc data/tumvi --out-dir data/official_tumvi
python src/eval/bench.py --cases data/official --glob "*_o42.zip" --mem
python src/tests/torture.py
python src/eval/make_bundle.py data/official/MH_01_easy_o42.zip --team CollegeName_TeamName
# retrain + re-embed (about 2 min per member on a laptop CPU):
python src/research/train_vel.py --data data/euroc,data/tumvi --val none --steps 400 --seed 10 --official 16 --p-off 0.9 --epochs 20 --out a.npz --bin a.bin
python src/research/train_vel.py --data data/euroc,data/tumvi --val none --steps 400 --seed 11 --official 16 --p-off 0.9 --epochs 20 --out b.npz --bin b.bin
python src/research/train_vel.py --data data/euroc,data/tumvi --val none --steps 400 --seed 12 --official 8 --p-off 0.7 --epochs 15 --out c.npz --bin c.bin
cat a.bin b.bin c.bin > src/weights/vel.bin && python src/research/embed_weights.py src/weights/vel.bin
```
EuRoC is expected under `data/euroc/<sequence>/mav0/...` (see `research/fetch_euroc.py`).

## Licenses
- Project: MIT (see `../LICENSE`).
- `research/official/degrade_imu.py` belongs to the hackathon organisers.
