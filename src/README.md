# Path Finder: IMU-only 6-DoF trajectory under sensor degradation

Samsung PRISM XR Hackathon 2026, **Track 3**.

The input is a degraded EuRoC IMU stream: progressive noise, bias drift, ~2 s dropouts, then total IMU failure. Path Finder estimates the 6-DoF trajectory through the mission and dead-reckons the 5 s after the failure.

| Budget (official) | Limit | Current (B1 baseline) |
|---|---|---|
| Peak RAM | ≤ 8 MB | **1.98 MB** on Linux (static binary, `/usr/bin/time`); 5.3 MB Windows working set incl. system DLLs |
| Compute | ≤ 20 ms per 1 s of IMU | **0.7–1.2 ms** per IMU-second (Linux, one pinned core, median of 5) |
| Threads | 1 | 1 |
| Network | offline | offline |

## Quick start
```sh
sh src/build.sh                                 # needs gcc/g++ (C++17); no other dependencies
src/pathfinder <input.zip|folder|imu0/data.csv> -o estimated_trajectory.txt
# or, through the standard-library Python wrapper (builds on first use):
python src/main.py <input> -o estimated_trajectory.txt
```
Options:
- `--blind auto|append|none`. `auto`: if the file ends with valid data, 5 s of poses are appended at the nominal rate; trailing invalid rows are always forecast.
- `--metrics m.json`: writes runtime and peak-memory figures.

Docker:
```sh
docker build -t pathfinder -f src/Dockerfile src
docker run --rm --cpus=1 -v "$PWD/data:/data" pathfinder /data/input.zip -o /data/estimated_trajectory.txt
```

## Input / output contract
- **Input**: a zip (any nesting; the shortest path ending in `imu0/data.csv` is used), a folder, or the CSV itself. EuRoC columns: `timestamp[ns], w_x,w_y,w_z [rad/s], a_x,a_y,a_z [m/s²]`. The parser tolerates CRLF, a missing header, `;`/tab/space separators, empty or NaN fields, and scientific-notation timestamps.
- **Output**: `estimated_trajectory.txt` in TUM format, `timestamp tx ty tz qx qy qz qw`.
  - **Exactly one pose per input row, in input order.** Timestamps are printed from the integer nanoseconds, so they are exact.
  - The quaternion is a unit `q_WB` with hemisphere continuity.
  - The world frame is gravity-aligned (z up), with its origin at the first pose and initial yaw 0.

## Architecture (current: baseline B1)
The program streams the input twice and never holds the whole sequence in memory (RAM is O(1) in sequence length).
1. **Scan**: classify rows (NaN/empty, all-zero, out-of-range, frozen-repeat → invalid), estimate the nominal rate, find the last valid row, and initialise from the first static window (roll/pitch from gravity, gyro bias).
2. **Filter**: a 15-state error-state Kalman filter (ESKF).
   - Midpoint IMU integration.
   - Spike repair using one sample of lookahead.
   - IMU noise estimated online from first differences.
   - Static detection triggers zero-velocity, gravity-tilt and zero-rate updates.
   - A weak zero-velocity prior that only corrects velocity, so it cannot bend the attitude.
3. **Dropouts** (invalid rows or timestamp gaps): a decaying constant-velocity / turn-rate motion model with inflated covariance.
4. **Blind 5 s** after the last valid row: the same decaying model, started from the final state.

The planned upgrades (learned velocity network, rotor-drag physics, offline two-filter smoother, learned forecaster) are in `../docs/plan.md`. Every decision is in `../devlog.md`.

## Repository layout
```
src/
  main.cpp, core/            C++17 runtime (Eigen 3.4.0 + miniz 3.0.2 vendored in third_party/)
  main.py                    stdlib wrapper; --engine python runs the numpy reference
  build.sh, Dockerfile
  research/
    common.py                I/O + rotation utilities shared by all Python tools
    ref_pipeline.py          numpy twin of the C++ runtime (parity-tested, byte-identical output)
    sim_traj.py              synthetic trajectory + IMU generator (EuRoC layout)
    degrade.py               randomised degradation simulator (noise growth, bias drift, dropouts, failure)
    get_euroc.py             extracts IMU + ground truth from downloaded EuRoC zips
  eval/
    score.py                 ATE/RPE identical to evo (verified), all alignment modes, per-tier metrics
    format_check.py          strict output-contract checker
    budget.sh                Linux peak-RSS + ms/IMU-second measurement (taskset, /usr/bin/time)
  tests/torture.py           15 edge cases: format + C++/Python parity
```

## Reproduce the tests
```sh
pip install -r src/research/requirements.txt
python src/research/sim_traj.py --out data/sim/SIM_Q1 --duration 90 --mode quad --seed 1
python src/research/degrade.py --seq data/sim/SIM_Q1 --out-dir data/degraded --seed 7
src/pathfinder data/degraded/SIM_Q1_s7.zip -o out/est.txt
python src/eval/format_check.py out/est.txt data/degraded/SIM_Q1_s7.zip --meta data/degraded/SIM_Q1_s7.meta.json
python src/eval/score.py --gt data/degraded/SIM_Q1_s7.gt.txt --est out/est.txt --meta data/degraded/SIM_Q1_s7.meta.json
python src/tests/torture.py
```

## Licenses
- Project: MIT (see `../LICENSE`).
- Eigen: MPL2.
- miniz: MIT.
