# Path Finder (Track 3) — Devlog

Every decision is logged here: what was decided, why, which alternatives were not taken and why not.
Full plan: [`docs/plan.md`](docs/plan.md).

**Entry format:** `ID · date · decision · context · reason/evidence (EXP link) · alternatives not taken + why · status`
**Status values:** `accepted` · `provisional` (pending experiment) · `superseded by D-xxx` · `rejected`

---

## Constraints snapshot (from the hackathon PDF, 2026-11-10)
- Input: EuRoC `imu0/data.csv` (ns timestamps, gyro rad/s, accel m/s² incl. gravity) inside a zip.
- Output: `estimated_trajectory.txt`, TUM format `t tx ty tz qx qy qz qw`, t in seconds, **one pose per IMU frame**.
- Budgets: ≤ 8 MB peak RAM, ≤ 20 ms compute per 1 s IMU, single CPU thread, offline. **FAQ (16 Nov): Python only (`main.py`); the 8 MB is measured at the algorithm / heap level, not process RSS.**
- Score: memory 20 · runtime 20 · checkpoints 30 · blind 5 s 20 · dropout 10. Evaluated with `evo` (ATE RMSE, RPE).
- Deadline: ~~18 Nov~~ **22 Nov 2026** (FAQ, 16 Nov). Submission via a Google Form announced in the official channel.
- Team: 2 people (A = ML/research, B = systems/C++/eval).

---

## Decisions

### D-001 · 2026-11-10 · C++17 runtime, Python/PyTorch for research — superseded by D-030
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

### D-007 · 2026-11-10 · Our own randomised degradation simulator — superseded by D-032 for evaluation
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

### D-011 · 2026-11-10 · One output row per input row, in input order, + `--blind auto` — amended by D-031 (never append)
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

### D-025 · 2026-11-16 · Learned body-frame velocity measurement fused in the ESKF — accepted
- **Context:** EXP-009. The baseline B1 diverges on degraded EuRoC (mean checkpoint ATE 66.9 m on 9 held-out runs; a never-moves pose gets 3.6 m). Accelerometer bias is unobservable, so velocity drifts.
- **Decision:** an MLP (450-256-128-6) reads the last 1 s of IMU in the **body frame** (50 blocks of 4 samples: accel, gyro, gravity direction from the filter attitude) and predicts body-frame velocity plus a log-variance. The variance, scaled by 2, becomes the measurement noise of a velocity update applied every 0.2 s. Ground truth for training is EuRoC `state_groundtruth_estimate0`, with on-the-fly degradation (white noise, bias, 2° tilt error).
- **Evidence:** held-out velocity RMSE 0.41 m/s vs 0.57 for predicting zero. A heading-frame version (rotating IMU into a yaw-free world frame) only reached 0.49, so the body-frame input is the useful one (a multirotor's drag shows up in body-frame accel).
- **Not taken:** global-frame RoNIN/TLIO inputs (worse here); end-to-end pose regression.
- **Status:** accepted.

### D-026 · 2026-11-16 · Update all error states, drop the zero-velocity prior — accepted
- **Evidence:** EXP-009. Velocity-only (Schmidt) updates with the old 0.5 m/s prior: 66.9 m. Full updates + prior: 17.2 m. Full updates, prior off: **3.75 m**. With full updates the filter learns the accelerometer bias from the velocity measurements, which is what stops the drift. The prior fought the network.
- **Also:** gap and blind velocity decay tau 2 s -> 0.5 s (checkpoint ATE 3.75 -> 3.50).
- **Not taken:** inflating the measurement noise with the estimated noise level (`nn_noise_pow`): 0.3 gave 3.60, 0.6 gave 5.82, 1.0 gave 31.2. Less trust in the network made drift worse.

### D-027 · 2026-11-16 · 3-member ensemble, bfloat16 weights in RAM, weights file next to the binary — accepted
- **Reason:** one member is noisy (checkpoint ATE 3.37-3.68 across seeds, one run reached only 0.55 m/s RMSE); averaging 4 members gave blind ATE 4.15 vs 4.4-5.0. Shipped model: 3 members trained on all 11 sequences (dropout 0.3, weight decay 0.02, 15 epochs; longer training overfit: val z² rose from 1.9 to 2.9). Weights are stored as bfloat16 (halves RAM, no accuracy change) and widened on the fly.
- **Why a file, not a header:** an embedded float array would be about 25 MB of source. `src/weights/vel.bin` (1.8 MB) is found next to the executable; `--weights` and `--no-nn` override it.
- **Cost:** Linux peak RSS 2.8 MB, about 1.4 ms per IMU-second. Windows working set 6.2 MB (it includes the C runtime).

### D-028 · 2026-11-16 · Honest limits of the learned model — recorded
- On held-out runs the blind-window ATE (4.2 m) is still above a fixed pose (2.5 m). Under the global alignment it is dominated by drift accumulated under the heaviest noise at the end of the run, not by the 5 s forecast itself (blind error relative to the pose at failure is 1.05 vs 1.00 m for a fixed pose).
- Trained on EuRoC (drone) only. A handheld source (TUM-VI) is out of distribution and would need retraining.
- The C++ and numpy twins are byte-identical only for the baseline (`--no-nn` with the legacy parameters, used by `torture.py`). The learned path exists only in C++.

---

### D-029 · 2026-11-16 · Efficiency pass: update the net every 0.2 s, keep -O2 — accepted
- **Evidence:** EXP-010. Calling the 3-member net every 0.1 / 0.2 / 0.4 s gave checkpoint ATE 3.44 / 3.41 / 3.44 m, so 0.2 s is free. Linux compute fell from about 2.0 to **1.4 ms per IMU-second**, peak RSS 2.8 MB. `-O3` was no faster than `-O2` and used 0.1 MB more.
- **Also tried:** wider training augmentation (tilt 5 deg, noise 4, bias 0.6): blind ATE 3.98 vs 4.00-4.47 for the default, within seed noise. The shipped ensemble mixes two default members and one wide member for diversity (held-out mix: 3.41 / 4.09 / 3.23 m).
- **Not taken:** AVX2 builds (the evaluator CPU is unknown); reusing the first layer across the sliding window (it shares only half its inputs).

---

### D-030 · 2026-11-16 · Python-only runtime: `src/main.py` (numpy), C++ archived — accepted
- **Context:** The official Track 3 FAQ: "we are requiring all teams to stick exclusively to Python ... submit a standalone Python script (`main.py`)". `main.cpp` is not accepted. The 8 MB limit is measured at the algorithm / heap level (data structures, buffers, estimation logic), not the interpreter's RSS.
- **Decision:** port the full C++ pipeline (ESKF, learned velocity ensemble, static updates, gap and blind models) to one standalone `src/main.py` that needs only numpy. The C++ runtime and its parity tooling move to `archive/cpp_runtime/`.
- **How it meets the budgets in Python:**
  - State propagation in scalar Python floats; the 15x15 covariance is propagated once per 4 IMU steps (`cov_every=4`; accuracy identical to every step in EXP-012, 3.13 m both, at 5.3 vs 9.3 ms/s).
  - Input held as float32 (the CSV has 6 decimals) and timestamps as int64: 32 B per row. Validity checks in 4096-row chunks; rows converted to Python floats 1024 at a time; output written in 512-line batches.
  - Weights rounded to bfloat16 as validated, stored as float16 (exact for all but a few hundred weights below 1e-4) and widened per call: 0.86 MB instead of 1.8 MB. Loaded layer by layer.
  - `np.median` is avoided because it lazily imports `numpy.ma` (about 0.6 MB of heap); a partition-based median is used.
- **Evidence:** EXP-011. Peak heap 3.47 MB (tracemalloc, weights + run) on the longest sequence (MH_01, 35.6k rows), 6.3 ms per IMU-second on Windows, 6.1 ms on Linux (WSL, one pinned core). Linux and Windows outputs agree to 1e-6 m.
- **Not taken:** keeping `main.py` as a wrapper around the binary (now disallowed); pure stdlib Python (the 3-member MLP would cost about 100 ms per IMU-second without numpy).
- **Risk recorded:** importing numpy itself allocates about 7.7 MB under tracemalloc. If the organisers trace from interpreter start, numpy alone would breach 8 MB. Their wording ("not the base OS-level RSS of the runtime") suggests they trace the algorithm, which is how `eval/bench.py --mem` measures.

### D-031 · 2026-11-16 · Never append poses; the blind window is inside the input — accepted
- **Context:** FAQ: "DO NOT append extra poses beyond the input timestamps ... The evaluation dataset will contain timestamps during the blind 5-second window ... the sensor values will be unusable/invalid." In `degrade_imu.py` the last 5 s are set to exactly 0.0.
- **Decision:** `--blind append/auto` removed. Every input row gets exactly one pose, nothing more. All-zero rows are invalid, so the blind window starts after the last valid row. `format_check.py` now fails on any count mismatch.

### D-032 · 2026-11-16 · Evaluate on the official degradation script — accepted
- **Context:** The student package ships `degrade_imu.py`. It differs from our simulator:
  - Drops are *missing rows*: single rows with probability up to 6 %, and bursts of 5-50 rows, so gaps are at most about 0.25 s, not 2 s.
  - Stuck values (2-10 repeats), duplicated rows (same timestamp), and spikes on all axes.
  - White noise, bias random walk and scale-factor random walk ramp as 0.02 + 0.98 t^1.7.
  - The first 2 s are clean, the last 5 s are zeroed, and the seed is fixed (42).
- **Decision:** `research/official/degrade_imu.py` is kept byte-identical to the organisers' copy. `research/make_official.py` imports it and writes the zip, ground truth interpolated at every output row, and a meta file (failure time, plus every timestamp gap > 2.5 dt as a dropout). Test set `data/official/` holds the 11 EuRoC sequences at seed 42. The held-out subset `data/official_ho/` holds MH_05, V1_02 and V2_03, which `weights/mix3_val.bin` never saw.
- **Column order is ambiguous:** the script documents `timestamp, accel, gyro`, but EuRoC files are `timestamp, gyro, accel`. If the organisers fed EuRoC files in unchanged, the script would put accelerometer noise on the gyro columns. `main.py` detects the order (the triple whose median norm is about g is the accelerometer, with header names as a tie-break), the accel unit (g or m/s^2), the gyro unit and the timestamp unit. Both variants are tested (`*_o42` and `*_e42`).

### D-033 · 2026-11-16 · Bridge short gaps by interpolating the IMU — accepted
- **Context:** With the official degradation, a gap starts every second or so late in the run (missing rows, stuck values). The C++ logic sent every gap > 2.5 dt to the motion model and **reset the 1 s window of the learned model**, so the network almost never ran in the degraded half.
- **Decision:** gaps up to `bridge_s = 0.5 s` are integrated with linearly interpolated IMU in sub-steps of one nominal dt. Invalid rows inside such a gap get a pose from the interpolated IMU. The learned window keeps filling with the interpolated samples, and extra noise (1 m/s^2, 0.05 rad/s) goes into the covariance. Longer gaps still use the decaying motion model.
- **Evidence:** EXP-012 (held-out): checkpoint ATE 5.46 -> **3.13 m**, blind 11.46 -> **4.09 m**, dropout 7.80 -> **3.34 m**.

### D-034 · 2026-11-16 · Blind-window velocity decay — kept at 0.5 s
- **Evidence:** EXP-012. tau_v_blind 0.1 / 0.25 / 0.5 / 1.0 s gives blind ATE 4.08 / 4.08 / 4.09 / 4.13 m. Relative to the pose at failure, the forecast is about as good as holding still (0.59-0.64 vs 0.60 m). The blind tier is dominated by drift accumulated before the failure, so effort should go into the pre-failure estimate.

### D-035 · 2026-11-16 · Honest status on official-style data — recorded
- Held-out (EXP-012): checkpoint ATE 3.13 m vs 3.60 m for a never-moving pose; dropout 3.34 vs 3.51; blind 4.09 vs **2.55** (still worse than Z0); rotation RPE 0.86 vs 23.5 deg/s.
- If the evaluation file is the EuRoC file fed in unchanged (`*_e42`, accelerometer noise on the gyro), checkpoint ATE is 3.63 m, about the same as Z0 (3.60).
- The shipped `weights/vel.bin` was trained on all 11 sequences, so scores on `data/official/` are in-sample (2.97 m checkpoint, EXP-011) and must not be quoted as generalisation.
- Next levers: retrain the velocity net with the official degradation as augmentation (done in D-037), and an offline backward pass that uses the data after each gap.

---

### D-036 · 2026-11-16 · Model embedded in `main.py`; bundle is exactly `estimated_trajectory.txt`, `metadata.json`, `src/{main.py, README.md}` — accepted
- **Context:** The Track 3 slide shows `team_name.zip` with exactly those entries ("Evaluators run the wrapper directly - non-standard layouts fail"). The FAQ asks for a *standalone* `main.py`, and the general rules disqualify teams that do not follow the guideline.
- **Decision:** `research/embed_weights.py` writes the ensemble into `main.py` between markers: float16 weights (bfloat16-rounded, as validated), float32 biases, base64. The adjacent bytes literals become one constant at compile time, and `binascii` decodes it in C, so loading creates no Python objects per weight. `--weights` still loads an external file for research.
- **`make_bundle.py`** writes only the four entries and asserts the layout. It re-runs the bundled `src/main.py` alone from a clean folder, and the output must be byte-identical. The Dockerfile content is reproduced in the README, so the bundle's README alone is enough to rebuild the container.
- **Evidence:** the embedded and file-loaded outputs are byte-identical. Peak heap is unchanged (3.47 MB); `main.py` is 1.33 MB. The bundle re-runs on Linux (WSL) from the extracted zip.

### D-037 · 2026-11-16 · Retrain the velocity net on official-style degradation — accepted
- **Context:** The previous net was trained with our own synthetic noise. Slide 15 says the evaluation input is EuRoC format (gyro columns first). That makes it likely the organisers fed EuRoC files straight into `degrade_imu.py`, which documents accel first and so puts accelerometer-sized noise on the gyro.
- **Decision:** `train_vel.py --official K` runs `degrade_imu.degrade()` K times per sequence:
  - random seeds (never 42), max level U(0.7, 1.3), gamma U(1.3, 2.1);
  - alternating the documented order and the EuRoC-fed-in order;
  - each copy rebuilt on the original grid exactly as `main.py` sees it (invalid rows dropped, gaps linearly interpolated), with windows touching the failure excluded.
  A fraction `--p-off` of every batch comes from these copies; the rest are clean rows with the old synthetic augmentation. Training runs on the CPU (about 5 s per epoch); a CUDA build was not needed.
- **Evidence:** EXP-014 (held-out, 3 sequences, both orders, `nn_sig_scale` 1.0):
  - Old net: before failure 3.13 / 3.63, blind 4.09 / 2.91.
  - New (8 copies, p_off 0.7): 2.82 / 3.19, blind 3.00 / 2.09.
  - New (16 copies, p_off 0.9, 20 epochs): 2.76 / 3.11, blind 3.22 / 2.41.
  The two recipes are within noise, so the shipped ensemble mixes them: two members of the 16/0.9 recipe and one of the 8/0.7 recipe, trained on all 11 sequences (seeds 10-12).
- **Not taken:** a 6-member ensemble (combined score 3.38 vs 3.44: +1.5 % for +0.9 MB heap and +1.5 ms/s); a CUDA build (the 3 GB download is unnecessary for a 150k-parameter MLP).

### D-038 · 2026-11-16 · `nn_sig_scale` 2.0 -> 1.0 — accepted
- **Evidence:** EXP-014, new held-out ensemble, combined score 0.3·ckpt + 0.2·blind + 0.1·dropout summed over both orders: 2.0 -> 3.78, 1.5 -> 3.60, 1.25 -> 3.48, 1.0 -> 3.46, 0.75 -> 3.44. The net trained on official-style data is trustworthy enough to weight more. 1.0 was chosen over 0.75 as the less aggressive of two values within noise. `nn_sig_floor` 0.05 vs 0.10: identical.

### D-039 · 2026-11-16 · Noise-scaled bias random walk (`bias_rw_k`) — implemented, off by default
- **Idea:** in `degrade_imu.py` the bias random walk and the white noise both scale with severity (ratio about 0.04), so the filter's bias process noise could follow the online noise estimate.
- **Evidence:** EXP-014. With k = 0.02-0.08, the EuRoC-fed-in checkpoint ATE improves (3.33 -> 2.92-3.01), but the blind ATE gets worse in both orders (2.47 -> 3.46) and rotation RPE rises. The combined score did not improve, so it stays off.

---

### D-040 · 2026-11-16 · int8 weights with per-row scales; covariance every 8 steps — accepted
- **Context:** profiling showed the float16 -> float32 widening of the first layer was the largest single cost (18 % of the run), and the model was a quarter of the heap.
- **Decision:** each weight row is stored as int8 with one float32 scale (`VelNet.quantise`); biases stay float32. A float32 file given with `--weights` is quantised the same way on load, so the embedded and file paths are byte-identical. `cov_every` 4 -> 8, and the per-row finiteness check no longer touches the covariance.
- **Evidence:** EXP-016. Held-out combined score 3.246 -> 3.240 (no change). Peak heap 3.47 -> **2.76 MB**. Compute 5.6 -> **3.8 ms per IMU-second** (Linux, one pinned core). `main.py` 1.33 MB -> 0.69 MB. cov_every 4 / 8 / 16: 3.240 / 3.241 / 3.241.
- **Not taken:** calling the net every 0.3 / 0.4 s instead of 0.2 s (3.268 / 3.302 vs 3.241).

### D-041 · 2026-11-16 · Stronger velocity models tested and rejected — recorded
- **Per-step velocity error on the held-out split (official-style):** MLP 0.413 m/s, wide MLP (512-256) 0.389, 1-D CNN 0.369, GRU-96 over the same 1 s window 0.344, streaming GRU-64/96/128 about 0.37 (predicting zero: 0.58-0.60). Longer MLP windows (2 s, 4 s) did not help.
- **End to end the ranking flips.** Single streaming GRUs scored 4.32-4.39 against 3.56 for a single MLP member. They overfit (train 0.22-0.27 vs validation 0.37, z² up to 5), and their errors are correlated in time, which is what produces drift. A drift loss (world-frame displacement error over 5 s chunks) improved the GRU to 4.22, still worse than the MLP.
- **Wide MLP ensemble:** 3.251 vs 3.240, a tie, for 1.8x the compute. Rejected.
- **Lesson:** per-step velocity RMSE does not predict ATE; the 10 s displacement error does (3.1-3.5 m for every model, 5.0 m for predicting zero). Model selection stays end to end (D-008). `arch_search.py`, `train_gru.py` and `export_gru.py` are kept so the result can be reproduced.

### D-042 · 2026-11-16 · Weak bounded-workspace prior (`sig_pos = 5 m`, once per second) — accepted
- **Context:** with 3 m of drift per 10 s, a fixed pose (Z0) is a strong reference because the flights stay inside a room or hall. The dead-reckoned estimate was worse than Z0 in the blind window.
- **Decision:** a pseudo-measurement "position = start position" with σ = 5 m, applied once per second. It is weak next to the velocity updates over short spans and only bounds the slow drift.
- **Evidence:** EXP-016, held-out, both column orders. Combined score 3.464 -> **3.246**; σ = 3 / 4 / 5 / 10 / 20 m gives 3.217 / 3.231 / 3.246 / 3.278 / 3.360. Blind ATE 3.00 -> 2.29 m (documented order) and 2.09 -> 1.75 m (EuRoC fed in); the pre-failure and dropout tiers are unchanged or slightly better. 5 m was chosen as the weakest value on the flat part of the curve.
- **Limit:** it assumes a room-scale flight (true for EuRoC and the TUM-VI room sequences). `--set sig_pos=0` turns it off.

### D-043 · 2026-11-16 · Submission checklist — recorded
- `docs/submission_checklist.md` maps every rule in the slides and the FAQ to how it is met and how it is verified, and lists what only the team can do (team name, video, deck, form, repository visibility).

---

### D-044 · 2026-11-16 · Add the TUM-VI room sequences to training — accepted
- **Context:** the slides name EuRoC MAV *and* TUM-VI as datasets. The net had only ever seen drone flights. On official-style TUM-VI cases the shipped model scored 8.52 m before the failure against 1.11 m for a never-moving pose (EXP-017). A hidden TUM-VI test sequence would have been a disaster.
- **Data:** `research/fetch_tumvi.py` downloads only the last 24 MB of each 1.7 GB archive with an HTTP range request. That tail holds the calibrated IMU (`dso/imu.txt`) and the motion-capture pose of the IMU frame (`dso/gt_imu.csv`). Rooms 1-6: 14.8 minutes of handheld motion at 200 Hz with full ground truth. (Walking the tar headers one range request at a time was tried first and was 50x slower.)
- **Training changes (`train_vel.py`):** several data roots; motion-capture ground truth has no velocity, so it is a central difference over ±3 samples (50 ms); rows near motion-capture gaps (> 30 ms) are masked, and a window is used only if all 200 rows are valid. Default validation split: MH_05, V1_02, V2_03, room3, room6.
- **Evidence:** EXP-017, held-out, means over both column orders. TUM-VI before failure 9.56 -> **2.03 m**, blind 13.12 -> 2.71 m. EuRoC also improves: 2.99 -> **2.55 m** before failure (blind 2.02 -> 2.37 m). More and more varied data helped both domains.
- **Shipped model:** 3 members trained on all 17 sequences (seeds 10-12, 400 steps per epoch; two members with 16 official copies and p_off 0.9, one with 8 copies and 0.7).

### D-045 · 2026-11-16 · Adaptive workspace prior in two passes — accepted (supersedes the fixed 5 m of D-042)
- **Context:** the right prior strength depends on the size of the space. A fixed σ of 1 m suits the TUM-VI rooms (combined score 0.777) but hurts EuRoC (1.582); 5 m is the reverse (1.369 / 1.519).
- **Decision:** pass 1 runs the filter with cheap settings (covariance every 16 steps, net every 0.4 s, fixed 5 m prior at the start position) and only accumulates the centroid and the per-axis spread of the path. Pass 2 is the real run, with the prior centred on that centroid and σ per axis = 0.7 × spread, clipped to [0.3, 5] m. The vertical axis gets a much tighter prior than the horizontal ones, which matches both datasets.
- **Evidence:** EXP-017. Combined score (EuRoC / TUM-VI): fixed 5 m 1.519 / 1.369; fixed 2 m 1.435 / 1.048; adaptive 0.5 → 1.513 / 0.636; **0.7 → 1.467 / 0.714**; 0.85 → 1.469 / 0.775. 0.7 has the best sum that does not give up EuRoC.
- **Cost:** compute 3.8 -> 5.8 ms per IMU-second (Linux, one pinned core), still 3.4x under the limit. Memory is unchanged because pass 1 stores nothing per row.
- **Not taken:** `nn_sig_scale` 1.25 (sum 2.142 vs 2.181 for 1.0: within noise, and worse on EuRoC).

### D-046 · 2026-11-16 · Honest status after the TUM-VI round — recorded
- EuRoC held-out: 2.70 / 1.99 / 2.59 m (before failure / blind / dropout) vs 3.60 / 2.55 / 3.51 m for a fixed pose.
- TUM-VI held-out: 1.41 / 0.86 / 1.19 m vs 1.16 / 0.51 / 1.09 m for a fixed pose. Position is still slightly worse than a fixed pose in a 3 m room; rotation RPE is 2.6 vs 41 deg/s.
- Only two rooms and three flights are held out, so differences of a few percent between settings are within noise.
- Motions outside both domains (vehicles, long corridors, outdoor walks) are untested. The TUM-VI corridor, magistrale, outdoor and slide sequences have ground truth only at the start and end, so they cannot be used for training or scoring.

---

### D-047 · 2026-11-16 · Outside review: single-pass adaptive prior replaces the two-pass one — accepted (supersedes D-045)
- **Finding:** the two-pass design ran the whole filter twice only to learn two numbers per axis (centroid and spread). That cost 2 ms per IMU-second and an extra code path.
- **Decision:** one pass. Once per second the running centroid and per-axis spread of the path so far are updated. After 20 s (`pos_warm`) the prior is centred on the running centroid with σ = 1.0 × running spread, clipped to [0.3, 5] m; before that it is the fixed 5 m prior at the start. The two-pass code (`SpreadMeter`, pass-1 parameters) was removed.
- **Evidence:** EXP-018, held-out, combined score EuRoC + TUM-VI: two-pass 1.467 + 0.714 = 2.181; single pass (20 s, k = 1.0) 1.531 + 0.652 = **2.183**. Other settings (warm 10-40 s, k 0.7-1.3) are within 2.18-2.26. Compute 5.8 -> **3.7 ms per IMU-second** (Linux, one pinned core).

### D-048 · 2026-11-16 · Outside review: out-of-core input for long recordings — accepted
- **Finding:** heap grew by 37 bytes per row, so a recording longer than about 15 minutes would have broken the 8 MB limit (a 23-minute file measured 11.9 MB). The evaluation length is not stated anywhere.
- **Decision:** past 60,000 rows (5 minutes at 200 Hz) the parser writes rows to two temp files in 4096-row chunks and memory-maps them; the filter code is unchanged because it already reads rows in chunks. The temp files are removed when the run ends. If no temp space is available the loader stays in RAM. `initialise` no longer builds an index over all rows.
- **Evidence:** 288,640 rows (22.7 min): peak heap 11.89 MB in RAM, **4.24 MB** out of core, byte-identical output, no temp files left. The remaining growth is 5 bytes per row (validity flag and next-valid index). EuRoC and TUM-VI room files are below the threshold, so their numbers are unchanged. `tests/torture.py` forces the path and checks byte equality.
- **Open:** whether temp files are allowed was asked in Q7 and not answered directly. Below 5 minutes nothing is written to disk.

### D-049 · 2026-11-16 · Outside review: non-causal (look-ahead) velocity model — rejected
- **Idea:** the task is offline, so the net could see IMU after the instant it predicts. `train_vel.py --lag/--window` trains it.
- **Evidence:** per-step held-out error: causal 1 s 0.387 m/s; centred 1 s 0.378; centred 2 s 0.353. End to end (single member, filter applying the residual `lag` later against the stored past estimate): combined EuRoC / TUM-VI 1.526 / 0.746 causal; 1.767 / 0.697 centred 1 s; 2.179 / 0.996 centred 2 s. The delayed update assumes the velocity error is constant over the look-ahead, and the last second before the failure gets no update at all, which hurts the blind tier most.
- **Decision:** rejected and removed from `main.py`. A proper version needs a fixed-lag smoother (state cloning); the 2-9 % per-step gain does not justify that complexity given D-041 (per-step error does not predict ATE).

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
| 009 | 11-16 | Learned body-frame velocity + full-state updates, 9 held-out degraded runs (MH_05, V1_02, V2_03 x seeds 21-23), SE3 | held-out, nets never saw them | B1 66.87 -> **3.41** (Z0 3.62) | B1 102.75 -> **4.09** (Z0 2.45) | B1 67.30 -> **3.23** (Z0 3.48) | rot RPE 1 s 5.2 -> 5.0 deg/s (Z0 23) | 1.4 (Linux) | 2.8 MB (Linux) | | Position drift removed; blind-window ATE still above a fixed pose (D-028). Origin alignment: 113.8 -> 4.06 m |
| 006 | 11-10 | C++ B1 runtime vs Python twin; 15 torture cases | sims + torture set | byte-identical output | | | | 1.8–2.0 (Windows) | 5.3 MB (Windows working set) | | parity and budgets OK; Linux RSS pending |

| 011 | 11-16 | **Python port** (`src/main.py`, shipped `vel.bin`), official `degrade_imu.py`, seed 42, all 11 sequences (in-sample for the net), SE3 | data/official *_o42 | **2.97** (Z0 3.46) | 4.67 (Z0 3.09) | 3.22 (Z0 3.34) | rot RPE 1 s 1.44 deg/s (Z0 18.4) | 6.3 Win / 6.1 Linux | **3.47 MB heap** (tracemalloc) | | Budgets met in Python + numpy; the first version (before the float32/fp16 savings) peaked at 6.8 MB |
| 012 | 11-16 | Held-out ablations, `mix3_val.bin` (never saw MH_05, V1_02, V2_03), official degradation, SE3 | data/official_ho *_o42 | default **3.13**; no bridging 5.46; no NN 6783; cov_every=1 3.13 | 4.09; 11.46; 16052; 4.09 (Z0 2.55) | 3.34; 7.80; 9606; 3.34 (Z0 3.51) | rot 0.86 deg/s (Z0 23.5) | 5.3 (cov_every=1: 9.3) | | | Bridging (D-033) is the big win; the NN is essential; tau_v_blind 0.1-1.0 makes no difference (D-034) |
| 013 | 11-16 | Same as 012, EuRoC file fed straight into the script (gyro/accel noise swapped) | data/official_ho *_e42 | 3.63 (Z0 3.60) | 2.91 (Z0 2.55) | 4.00 (Z0 3.51) | rot 1.32 deg/s | 5.7 | | | Order auto-detected (ga); accuracy degrades because the gyro gets 0.2 rad/s of noise |
| 014 | 11-16 | **Official-style retraining** (held-out nets, val MH_05/V1_02/V2_03) + filter sweeps | data/official_ho, *_o42 / *_e42 | old net 3.13 / 3.63 -> **2.82 / 3.19** (sig 1.0) | 4.09 / 2.91 -> **3.00 / 2.09** | 3.34 / 4.00 -> **2.95 / 3.48** | rot 0.98 / 1.40 deg/s | 5-6 | | 3.46 (sum of both orders) | D-037, D-038; bias_rw_k rejected (D-039); 6 members not worth it |
| 015 | 11-16 | **Shipped**: final ensemble (all 11 sequences) embedded in main.py, sig 1.0, all 15 official cases (in-sample) | data/official all | old shipped 3.45 -> **2.70** (Z0 3.54) | 5.59 -> **4.20** (Z0 3.11) | 4.00 -> **3.08** (Z0 3.40) | rot 1.67 deg/s (Z0 18.8) | 5.2-6.7 Win, 5.6 Linux | **3.47 MB heap** | | In-sample sanity check, not generalisation. MH_01 (the longest run) still has a large blind error (8.8 / 13.7 m) |
| 016 | 11-16 | **Efficiency + prior**: int8 weights, cov_every 8, sig_pos 5 m; rejected: GRU, CNN, wide MLP, slower NN rate | data/official_ho, *_o42 / *_e42 | **2.79 / 3.20** | **2.29 / 1.75** (Z0 2.55) | **2.91 / 3.45** | rot 0.97 / 1.40 deg/s | **3.8 Linux**, 4.1-4.8 Win | **2.76 MB heap** | **3.241** (was 3.464) | D-040 to D-042. In-sample on all 15 cases: 2.63 / 2.87 / 2.76 vs Z0 3.54 / 3.11 / 3.40 |
| 017 | 11-16 | **TUM-VI training data + adaptive two-pass prior**; held-out MH_05, V1_02, V2_03 and room3, room6; mean over both column orders | data/official_ho, data/official_ho2 room* | EuRoC 2.99 -> **2.70**; TUM-VI 9.56 -> **1.41** (Z0 3.60 / 1.16) | 2.02 -> **1.99**; 13.12 -> **0.86** (Z0 2.55 / 0.51) | 3.18 -> **2.59**; 9.42 -> **1.19** (Z0 3.51 / 1.09) | rot 1.17 / 2.64 deg/s (Z0 23.5 / 41.1) | **5.8 Linux**, 7-8.5 Win | **2.77 MB heap** | 1.467 / 0.714 | D-044, D-045. In-sample, shipped model: EuRoC 15 cases 2.80 / 2.74 / 2.65; TUM-VI 12 cases 0.72 / 0.58 / 0.67 (was 8.52 / 6.58 / 7.42) |
| 018 | 11-16 | **Outside review**: single-pass adaptive prior, out-of-core input; rejected: look-ahead model | data/official_ho, data/official_ho2 room* (mean of both orders) | EuRoC **2.76**; TUM-VI **1.34** | 2.21; **0.70** | 2.61; **1.10** | rot 1.13 / 2.64 deg/s | **3.7 Linux**, 4.6-6.3 Win | **2.77 MB** (23-min file: 4.24 MB) | 1.531 / 0.652 | D-047 to D-049. In-sample, shipped model: EuRoC 2.85 / 2.87 / 2.66; TUM-VI 0.74 / 0.57 / 0.68 |
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
| Q1 | Python only (`main.py`); 8 MB measured at the algorithm / heap level | 16 Nov (FAQ) | D-030: C++ archived, numpy port |
| Q2 | Dropouts are physically missing rows (timestamp jumps); see `degrade_imu.py` | 16 Nov | D-032, D-033 |
| Q3 | Blind 5 s rows are present with invalid values; never append poses | 16 Nov | D-031 |
| Q4 | evo with SE(3) Umeyama alignment; waypoints are hidden and used only for scoring | 16 Nov | Model selection on `se3` |
| Q5 | Not answered directly; the pipeline must use only the degraded IMU file | 16 Nov | — |
| Q6 | The degradation script is shared (`degrade_imu.py`, seed 42); initial pose not given | 16 Nov | D-032 |
| Q7 | Offline: the whole CSV may be read into memory and smoothed non-causally, within 8 MB | 16 Nov | A future backward pass is allowed |
| Q8 | Video 2-3 min, strict maximum 3 min | 16 Nov | D-013 unchanged |
