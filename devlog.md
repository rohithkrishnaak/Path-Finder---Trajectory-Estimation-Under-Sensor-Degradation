# Path Finder (Track 3) — Devlog

Every decision is logged here: what was decided, why, which alternatives were not taken and why not.
Full plan: [`docs/plan.md`](docs/plan.md).

**Entry format:** `ID · date · decision · context · reason/evidence (EXP link) · alternatives not taken + why · status`
**Status values:** `accepted` · `provisional` (pending experiment) · `superseded by D-xxx` · `rejected`

---

## Constraints snapshot (from the hackathon PDF, 2026-11-10)
- Input: EuRoC `imu0/data.csv` (ns timestamps, gyro rad/s, accel m/s² incl. gravity) inside a zip.
- Output: `estimated_trajectory.txt`, TUM format `t tx ty tz qx qy qz qw`, t in seconds, **one pose per IMU frame**.
- Budgets: ≤ 8 MB peak RAM, ≤ 20 ms compute per 1 s IMU, single CPU thread, offline.
- Score: memory 20 · runtime 20 · checkpoints 30 · blind 5 s 20 · dropout 10. Evaluated with `evo` (ATE RMSE, RPE).
- Deadline: 18 Nov 2026 23:59, no extension. Internal target 18 Nov 18:00. Code freeze 17 Nov 22:00.
- Team: 2 people (A = ML/research, B = systems/C++/eval).

---

## Decisions

### D-001 · 2026-11-10 · C++17 runtime, Python/PyTorch for research — accepted
- **Context:** The 8 MB peak-RAM and 20 ms/s single-thread budgets make up 40% of the score.
- **Reason:** Importing numpy alone takes a Python process to about 25–30 MB RSS. A per-sample Python EKF at 200 Hz would need about 100 ms per IMU-second. A static C++ binary should fit in < 4 MB and use < 5 ms/s. The user confirmed this choice.
- **Not taken:** Pure Python, which fails the budgets if RSS is measured. Rust, because of team familiarity and build friction with an 8-day deadline. Python + Numba/Cython, because the JIT and runtime memory still exceed 8 MB.

### D-002 · 2026-11-10 · Error-state EKF backbone fused with learned measurements — accepted
- **Reason:** Plain double integration error grows quadratically. An EKF can fuse any number of measurement sources, each with its own uncertainty, which is how the "hybrid" gets built. It also degrades gracefully when a learned model is wrong, because gating plus covariance lets it downweight the bad input.
- **Not taken:** End-to-end network pose regression, which is unstable over long horizons, gives no uncertainty, and would be hard to fit in the CPU budget. A pure physics filter is kept only as the fallback (B1/M0).

### D-003 · 2026-11-10 · Offline two-filter smoother with disk spill — provisional (EXP pending)
- **Reason:** Processing is offline, so data after a dropout can bridge it. That is the main lever for T3, and smoothing also lowers ATE. Spilling compact float32 records to a temp file keeps RAM O(1) in sequence length.
- **Not taken:** In-RAM RTS, which needs about 70 MB for long sequences. Forward-only filtering, which throws away the offline advantage.
- **Fallback:** If temp disk use is disallowed, keep 10 Hz keyframes in RAM (≤ 3 MB cap) plus fixed-lag smoothing.

### D-004 · 2026-11-10 · Body-frame velocity network (AirIO-style) + learned log-variance — provisional
- **Reason:** Drone IMU motion patterns are most consistent in the body frame. Yaw is unobservable, so global-frame inputs mix in arbitrary rotations. The learned variance becomes the EKF measurement R.
- **Not taken:** Global-frame RoNIN/TLIO inputs, which were designed for pedestrians and are worse for drones. These still run as bake-off candidate M1 for evidence.

### D-005 · 2026-11-10 · Multirotor rotor-drag measurement model — provisional
- **Reason:** In a multirotor, horizontal body-frame acceleration is proportional to −k·v, so the accelerometer gives velocity information without any learning. It costs almost nothing, generalises well, and serves as the physics fallback.
- **Not taken:** Ignoring aerodynamics, which leaves velocity unobservable between ZUPTs.

### D-006 · 2026-11-10 · Exact integer ns → s timestamp formatting (`%lld.%09lld`) — accepted
- **Reason:** "Misaligned rows incur heavy penalties." Converting to double and printing can introduce rounding error.
- **Not taken:** `printf("%.6f", ns*1e-9)`.

### D-007 · 2026-11-10 · Our own randomised degradation simulator — accepted
- **Context:** The organizer degradation script is "provided" but we don't have it yet.
- **Reason:** We can't wait with only 8 days. Randomising over noise growth, bias drift, dropout styles (missing rows, NaN, zeros, frozen values), spikes, saturation and terminal failure makes the models robust to whatever the real script does. The organizer script becomes one simulator mode when it arrives.
- **Not taken:** Waiting for the script. Training on a single fixed degradation, which would overfit.

### D-008 · 2026-11-10 · Leave-one-sequence-out CV + end-to-end proxy score J — accepted
- **Reason:** Random window splits leak information between overlapping windows of the same sequence. Component losses (velocity MSE, for example) don't map reliably onto ATE, so selection runs on the full pipeline output. J weights the tiers as the rubric does and is reported for each evo alignment mode, choosing on the worst case.
- **Not taken:** A random split; selection on validation loss alone.

### D-009 · 2026-11-10 · Vendored fixed-size Eigen + single-file miniz — accepted
- **Reason:** Header-only, with no heap allocations for fixed sizes, and well tested. miniz lets us read the zip input directly. Vendored versions: Eigen 3.4.0 (tarball sha256 `8586084f…1c72`) and miniz 3.0.2.
- **Not taken:** Hand-rolled matrix code (higher bug risk); requiring the evaluator to unzip first (one more way to fail).

### D-010 · 2026-11-10 · Weights embedded as a generated `weights.h` — accepted
- **Reason:** A single self-contained binary, with no file-path or loader failures.
- **Not taken:** A separate weights file plus a loader; ONNX Runtime/TFLite, which have megabytes of runtime overhead and break the 8 MB budget.

### D-011 · 2026-11-10 · One output row per input row, in input order, + `--blind auto` — accepted
- **Reason:** This follows the "exactly one pose per IMU frame" rule. Invalid trailing rows get forecasts. If the file simply ends, 5 s of rows are appended at the median dt; evo association ignores unmatched rows.
- **Not taken:** Resampling to a uniform grid, which breaks row correspondence.
- **Open:** Confirm with the organizers (Q3).

### D-012 · 2026-11-10 · No fingerprinting or memorisation of public sequences — accepted
- **Reason:** The hidden test may reuse public EuRoC/TUM-VI sequences. Recognising them and replaying ground truth would break the spirit of the contest and risk disqualification. Final models train on all public data after honest LOSO selection, which is standard practice.

### D-013 · 2026-11-10 · Demo video ≤ 3 min — accepted
- **Reason:** The sources conflict: the track says 2–3 min, the general rules say max 3 min, the timeline says 5 min. A 2:45 video satisfies all three.

### D-014 · 2026-11-10 · `main.py` = stdlib-only wrapper; `--engine python` = numpy reference twin — accepted
- **Reason:** The deliverable text says "standalone Python script", while the bundle allows `main.py / main.cpp`. A wrapper satisfies both readings, and the numpy twin gives parity tests and transparency. Verified: both engines produce byte-identical output (EXP-006).

### D-015 · 2026-11-10 · Present the work for "research / IP / commercialisation" and XR relevance — accepted
- **Context:** The launch poster says solutions should be "innovative, practical, measurable" with "potential for research, intellectual property creation, and commercialization". Track 3 is framed as "reliable tracking and navigation when sensor measurements become degraded or unreliable". No new technical rules.
- **Reason:** The final demo is judged by a jury, not only by evo. The deck and video will include an XR-relevance slide: the same pipeline keeps a headset or handheld device tracked when its cameras fail. They will also include a slide on what is novel and could be patented, with measured evidence (ablation tables, budget numbers).
- **Effect on the plan:** TUM-VI (handheld, like an XR device) moves from "cut early" to "evaluate even if not used for training". It gives a cross-domain generalisation result for the deck. The scope-cut order becomes: M6 → I3 → TUM-VI *training* → G2 → F6.
- **Not taken:** Treating the deck as an afterthought. The top 15 present live, and the jury's view decides the final ranking.

### D-016 · 2026-11-10 · Midpoint (trapezoidal) IMU integration — accepted
- **Evidence:** EXP-001. On clean synthetic data, 60 s of integration gave 0.206 m position error with first-order Euler and 0.049 m with midpoint, and orientation error of 5.5e-6 rad.
- **Not taken:** first-order Euler, which is 4x worse at the same cost. RK4 needs the IMU signal between samples, which we don't have.

### D-017 · 2026-11-10 · Synthetic trajectory generator + simulator-first development — accepted
- **Context:** The EuRoC files are not downloaded yet, and the original ETH host (robotics.ethz.ch) is offline.
- **Reason:** `sim_traj.py` produces kinematically consistent IMU and ground truth (quad, handheld, static, rotate modes). This let us build and verify the whole toolchain on day 0 and separate filter bugs from data issues.
- **Not taken:** waiting for the data. A simulator is also the only source of exact ground truth for unit tests.

### D-018 · 2026-11-10 · Weak zero-velocity prior, applied as a velocity-only (Schmidt) update — provisional
- **Evidence:** EXP-002 and EXP-003.
  - With no prior, pure IMU integration of degraded data explodes: ATE 236–504 m over 90 s.
  - A full-state prior (σ = 1 m/s) brings ATE down to about 1.1 m, but it **corrupts the attitude**. The filter explains the velocity residual by tilting, so pre-failure rotation RPE on clean data rose from 0.019°/s to 0.507°/s.
  - A velocity-only update keeps the attitude exact (0.019°/s), with ATE 2.4–3.5 m on degraded data at σ = 0.5 m/s.
- **Decision:** velocity-only update with σ = 0.5 m/s, applied every 10 samples. To be re-tuned on EuRoC.
- **Not taken:** the full-state prior (it looks better on ATE but is wrong for the right reason, and a correct attitude is a prerequisite for the body-frame velocity network). No prior at all (the error explodes).

### D-019 · 2026-11-10 · Sanity baseline Z0 ("never moves") is mandatory in every comparison — accepted
- **Evidence:** EXP-004. On synthetic data, B1's *position* ATE is about the same as outputting the origin for every pose (Z0 ≈ 1.03 m, the RMS spread of the trajectory). B1 only beats Z0 on orientation.
- **Reason:** IMU-only position estimation carries almost no information without a motion model. A method only adds value if it beats Z0. This is the gap the learned velocity model (M1–M5) must close.

### D-020 · 2026-11-10 · Scoring reports pre-failure RPE separately — accepted
- **Evidence:** EXP-005. The rotation RPE looked like 9°/s on clean data. The per-frame breakdown showed it was entirely the 5 s blind window (the simulated drone yaws at about 65°/s, which no forecaster can know). Inside the tracked portion, the error was 0.0009° per frame.
- **Reason:** A single number hid where the error came from. `score.py` now adds `rpe_*_1s_ckpt`.

### D-021 · 2026-11-10 · Spike repair with one-sample lookahead (not invalidation) — accepted
- **Reason:** Isolated outliers are repaired to the mean of their neighbours instead of being dropped, so the IMU chain isn't broken into gaps. A spike must differ from *both* neighbours by T = max(10σ̂, floor) while the neighbours agree to within T/2. A real step change doesn't satisfy that.
- **Not taken:** a Hampel filter over a wide window (more state and more latency); dropping the sample (creates a gap).

### D-022 · 2026-11-10 · Repo layout: git root = project folder, all code under `src/` — accepted
- **Reason:** The submission zip needs `src/` holding the complete source, so research, eval and tests live under `src/` too. The organizer PDF, data and outputs are git-ignored.

### D-023 · 2026-11-10 · All Python file edits use explicit UTF-8 — accepted (process)
- **Context:** An edit script used Python's default Windows encoding (cp1252). It failed on "σ" after the file had already been opened for writing, which emptied this devlog. The devlog was rebuilt from the session record.
- **Rule:** Always pass `encoding="utf-8"`. Write to a temp file, then rename. Commit the devlog often so git keeps a copy.

### D-024 · 2026-11-10 · Fetch EuRoC via HTTP range requests (`fetch_euroc.py`) — accepted
- **Context:** The original ETH host is offline. The ETH Research Collection hosts 3 archives totalling 24.7 GB (machine_hall 12.7 GB, vicon_room1 6.0 GB, vicon_room2 6.0 GB). Each holds a ROS `.bag` and a deflate-compressed ASL `.zip` per sequence.
- **Reason:** The server supports `Range` (HTTP 206). Reading the zip central directory remotely lets us skip all `.bag` files (about half the bytes) and keep only 3 small CSV/YAML files per sequence. The deflated inner zips still have to be streamed.
- **Not taken:** Downloading the 24.7 GB archives in full (slower, uses about 25 GB of disk); downloading the `.bag` files (they need ROS tooling).

---

## Candidate bake-off register (filled as experiments run)
| Stage | Candidates | Winner | Evidence |
|---|---|---|---|
| Baseline | Z0 never-moves · B0 naive integration · B1 ESKF + static init + ZUPT + velocity-only prior | B1 (shippable floor) | EXP-002…004 |
| Velocity | M0 drag · M1 ResNet1D · M2 TCN · M3 GRU · M4 AirIO-CNN/MLP · M5 LightGBM · M6 Transformer (teacher) | — | — |
| Gyro corrector | G0 none · G1 Brossard CNN · G2 GRU | — | — |
| Dropout bridging | I0 ZOH · I1 linear interp · I2 smoother-only · I3 BiGRU imputer | — | — |
| Blind 5 s | F0 hold · F1 CV · F2 CV-decay · F3 CA · F4 CTRV-3D · F5 GRU/MLP · F6 kNN primitives · F7 blend | — | — |

## Experiment log
| EXP | Date | Config | Data / seeds | ATE ckpt | ATE blind | ATE dropout | RPE | ms per IMU-s | Peak RSS | J | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 001 | 11-10 | Euler vs midpoint integration, clean IMU, true init | SIM quad 60 s | 0.206 vs 0.049 m (end error) | — | — | rot 5.5e-6 rad | — | — | — | midpoint (D-016) |
| 002 | 11-10 | B1 no prior | SIM_Q1 s7 / s8 (degraded) | 504 / 236 m | 807 / 768 | 522 / 221 | — | — | — | — | explodes |
| 003 | 11-10 | B1 + velocity-only prior σ 0 / 1.0 / 0.5 | SIM_Q1 clean / s7 / s8 | clean 8.3/1.06/1.04; s7 504/5.2/3.5; s8 236/3.5/2.4 | s8 768/10.1/6.2 | — | rot RPE 0.019°/s (unchanged) | — | — | — | σ = 0.5 (D-018) |
| 004 | 11-10 | Z0 vs B1 | SIM_Q1 clean / s7 / s8 | Z0 1.03 / 1.04 / 1.03 | Z0 1.06 | Z0 0.82–1.02 | — | — | — | — | B1 ≈ Z0 on position (D-019) |
| 005 | 11-10 | score.py vs evo 1.x | SIM_Q1 s7, clean | se3 / sim3 / origin ATE and RPE (t, deg) identical to 4+ decimals | | | | | | | score.py verified |
| 007 | 11-10 | **First real data**: B1 vs Z0, EuRoC MH_05 (clean+blind tail / degraded seed 1: 2 dropouts, frozen failure) | MH05_clean / MH05_s1 | B1 6.96 / 18.47 vs Z0 6.99 (se3) | B1 10.1 / 52.3 vs Z0 3.88 | B1 23.3 vs Z0 6.35 | rot RPE 0.09 / 1.51°/s (Z0 10.4) | 3.8–4.0 (Windows) | 5.3 MB | | On real data B1 position is **worse than Z0** once degraded; orientation is good. The learned velocity model is the critical path; the 2 s-decay CV forecaster overshoots (blind 52 m) |
| 008 | 11-10 | **Official-style budget measurement** on Linux (WSL2 Ubuntu, g++ 15.2, static binary, `taskset -c 0`, `/usr/bin/time`, median of 5) | MH_01_s3 (182 s, longest), V1_02_s3, V2_03_s3, MH05_s1 | — | — | — | — | **0.71–1.21** | **1.98 MB** | | Both budgets met with 4x/16x headroom (8 MB / 20 ms). The Linux output is byte-identical to the Windows build. Note: MH_01 found no static window in its first 10 s (static_init=0); check the init on EuRoC |
| 006 | 11-10 | C++ B1 runtime vs Python twin; 15 torture cases | sims + torture set | byte-identical output | | | | 1.8–2.0 (Windows) | 5.3 MB (Windows working set) | | parity and budgets OK; Linux RSS pending |

---

## Open questions sent to the organizers (prism@samsung.com) — 2026-11-10
1. How is peak RAM measured (process RSS or Python tracemalloc)? Is a compiled C++ entry point (`main.cpp`) accepted, as the bundle tree suggests?
2. How are dropouts represented in the CSV: missing rows, NaN/empty fields, zeros, or frozen values?
3. For the post-failure 5 s: are rows with timestamps present but invalid, or does the file end so that we generate timestamps? At what rate?
4. Which evo alignment is used (`-a`, `-as`, `--align_origin`, none), and how are "waypoints/checkpoints" defined?
5. Do you run our code on hidden data, or do we submit `estimated_trajectory.txt` for data you provide? When will the evaluation zip be released?
6. Can the degradation script and its parameter ranges be shared? Is the initial pose given?
7. Is temporary disk use allowed? What CPU is the reference hardware?
8. Demo video length: 2–3 min (track), 5 min (timeline) or max 3 min (rules)?

| Q | Answer | Date | Impact |
|---|---|---|---|
| Q1–Q8 | Email sent; no reply yet. Proceeding with the "handle every reading" defaults from §2 of the plan | — | None until a reply arrives |
