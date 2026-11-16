# Track 3: Path Finder (IMU-only 6-DoF trajectory under sensor degradation). Implementation plan

> **Update 2026-11-16 (official Track 3 FAQ + `degrade_imu.py`).** This changes several parts of the plan below; `devlog.md` D-030 to D-035 has the details.
> - **Python only.** Submit a standalone `main.py`. The C++ runtime (§3, §4) is archived in `archive/cpp_runtime/`, and `src/main.py` is a numpy port of it.
> - **The 8 MB is measured at the algorithm / heap level**, not process RSS. Measured: 3.47 MB peak heap, 6.1 ms per IMU-second.
> - **Never append poses.** The blind 5 s rows are in the input, with zeroed values.
> - **Dropouts are missing rows**: frequent and short (≤ 0.25 s with the official script), plus stuck and duplicated rows. Short gaps are now bridged by interpolating the IMU (D-033).
> - **Evaluation:** evo with SE(3) Umeyama alignment. Waypoints are hidden.
> - **Offline batch smoothing is allowed** (the whole CSV may be held in memory within 8 MB).
> - **Deadline: 22 Nov 2026.** Video 2-3 min, strict maximum 3 min.
> - **Test data:** `src/research/make_official.py` runs the official script (seed 42) over EuRoC. Our own `degrade.py` is kept only for augmentation ideas.

## Context
Samsung PRISM XR Hackathon 2026, Track 3. We get a degraded EuRoC-format IMU CSV with progressive noise, bias drift, dropouts of about 2 s, unknown gravity alignment, and finally a complete IMU failure. We must output a 6-DoF trajectory with **exactly one pose per IMU frame**, plus a 5 s dead-reckoned path after the failure.
- **Team:** 2 people. **Today:** Tue 10 Nov 2026. **Hard deadline:** 18 Nov 23:59 (no extensions). Our internal submit target is 18 Nov 18:00.
- **Decided with the user:** a C++ runtime, with Python/PyTorch for research and training. No organizer assets exist yet, so we build our own degradation simulator and swap in theirs if it arrives.
- **Machine:** RTX 5060 8 GB with torch 2.14+CUDA, Python 3.14, MSYS2 g++, WSL available. Docker and cmake are not installed.

## 1. Rules digest (from the PDF; the scoring source of truth)
| Item | Requirement |
|---|---|
| Input | Zipped folder, `imu0/data.csv`, comma-separated, header `#timestamp [ns], w_RS_S_x..z [rad/s], a_RS_S_x..z [m/s², incl. gravity]` |
| Output | `estimated_trajectory.txt`, space-separated, `timestamp tx ty tz qx qy qz qw`. Timestamp in **seconds** (ns×1e-9). Unit quaternion in the world frame. Optional `#` header. **One pose per IMU frame**: dropped or misaligned rows are heavily penalised or fail |
| Budgets | ≤ 8 MB peak RAM · ≤ 20 ms of compute per 1 s of IMU · single CPU thread · offline |
| Scoring (100) | Peak memory 20 · Runtime 20 · Waypoints/checkpoints (T1) 30 · Blind 5 s (T2) 20 · Dropout resilience (T3) 10 |
| Evaluator | `evo` against hidden high-accuracy GT, using ATE RMSE (global) and RPE (local drift and rotation per unit time) |
| Bundle | `team_name.zip/{estimated_trajectory.txt, metadata.json (opt), src/{main.py or main.cpp, README.md}}` |
| General rules | Public or shared GitHub repo · README with reproducible setup **and Docker files** · demo video ≤ 3 min (the track says 2–3 min walkthrough) · PPT/PDF named `CollegeName_TeamName` (theme ID, title, team, problem in our own words, architecture diagram, stack, innovation, results, limitations) · one Google Form submission · any deviation = disqualification |
| Data | EuRoC MAV (11 sequences, ADIS16448 at 200 Hz, Vicon/Leica GT) and TUM-VI (BMI160 at 200 Hz, mocap GT in rooms) |

## 2. Ambiguities and how the design covers all readings
On day 0, send the questions below to prism@samsung.com. Until answers arrive, the code handles every interpretation through flags with safe defaults.
1. **Peak-RAM measurement** (process RSS or Python tracemalloc). C++ satisfies both, and `main.py` wraps the binary.
2. **Dropout representation** (missing rows, NaN/empty fields, zeros, or frozen repeated values). The detector handles all four. Output = one row per *input* row, in input order, with duplicates preserved.
3. **Post-failure 5 s** (trailing rows with timestamps but invalid data, or a file that just ends). `--blind auto`: invalid trailing rows get forecasts, and if the file simply ends we append 5 s at the median dt. Both modes are tested.
4. **evo alignment mode** (`-a`, `-as`, `--align_origin`, none). We output a gravity-aligned world frame with origin at the first pose and yaw = 0, and the harness scores every mode, so model choice must be robust across them.
5. **Who runs the code.** Do they run it on hidden data, or do we submit an output file for given data? Either way the CLI accepts a zip, a folder or a CSV.
6. Degradation script and parameter ranges; whether the initial pose is given; whether temp-disk use is allowed; the reference CPU; and the conflict over video length (2–3 vs 5 vs ≤ 3 min). **We use ≤ 3 min**, which satisfies all three.

## 3. Runtime architecture (C++17, single thread, target < 4 MB RSS and < 5 ms per 1 s of IMU)
```
 zip/folder/csv ─► [1 Ingest & Sanitize] ─► [2 Init] ─► FORWARD PASS ───────────────────────────► temp file (compact states+cov, float32)
                   streaming parser        static/gravity   [3 Learned IMU corrector (gyro denoise)]                 │
                   gap/NaN/zero/frozen/    alignment,       [4 ESKF 15-state propagate @200 Hz]                      ▼
                   spike/saturation flags  gyro-bias init   [5 Learned body-velocity net @10 Hz + log-var] ─► EKF upd  BACKWARD PASS
                   rate estimate                            [6 Physics updates: ZUPT, rotor-drag, gravity-tilt]     (two-filter smoother,
                   degradation-level est.                   [7 Adaptive Q/R from degradation level; gap mode]        reads fwd file in reverse
                                                                                                                      chunks, fuses)
                                                                                                                         │
                                                       [8 Blind-5s forecaster on smoothed end state] ◄───────────────────┤
                                                                                                                         ▼
                                                       [9 Writer: exact ns→s string, unit quat, sign continuity, NaN guard] ─► estimated_trajectory.txt
```
Component details:
1. **Ingest.** A hand-written `fgets` + `strtoll`/`strtod` parser (no iostream). It tolerates CRLF, spaces, scientific notation, and a missing or variant header. Zip input uses vendored single-file **miniz**, streamed. Each row gets flags {valid, gap_before, nan, zero, frozen, spike, saturated}. Per-segment degradation level σ̂(t) comes from a high-pass residual MAD over a 1 s window.
2. **Init.** Find the first static window (gyro and accel variance thresholds). Roll and pitch come from the mean accel, the gyro bias from the mean gyro, and yaw = 0. If no static window exists, use a long-window mean accel (a hover averages to gravity).
3. **IMU corrector.** A bake-off candidate (§5). It is kept only if it improves end-to-end orientation RPE.
4. **ESKF.** State: q, v, p, b_g, b_a (error state 15). Double precision. Fixed-size **Eigen** (vendored, header-only, no heap) instead of hand-rolled matrix code, to cut bug risk.
5. **Learned velocity pseudo-measurement.** The input is the last 1 s of bias-corrected body-frame IMU decimated to 50 Hz, plus the body-frame gravity direction from the filter (the AirIO insight: a body-frame representation suits drones). The output is body velocity (3) plus log-variance (3), applied as an EKF update. A Mahalanobis gate rejects outliers.
6. **Physics measurements.** ZUPT and zero angular rate when static. The **multirotor rotor-drag model** a_xy,body ≈ −k·v_xy,body, with k fitted on GT, which turns the accelerometer into a velocity sensor; it is a bake-off candidate and doubles as the physics fallback. The gravity-tilt update is used only when |a|≈g and the motion is low-dynamic.
7. **Adaptive noise and gaps.** Q and R scale with σ̂(t). Inside a dropout, propagate with decayed last rates and inflated covariance, and do no NN updates. The backward pass bridges the gap with post-gap data, which is the main lever for T3.
8. **Blind forecaster.** The winner of the §5 bake-off. It runs on the smoothed state at the last valid sample. Orientation uses a decaying rate hold.
9. **Writer.** The timestamp is printed from integer ns as `%lld.%09lld` (exact, no float rounding). Quaternions are normalised and kept hemisphere-continuous. A final guard replaces any non-finite pose with the last valid one, so the output never contains a NaN.

**Memory strategy.** Streaming in and out. The forward pass spills compact float32 records (state plus upper-triangular P, about 0.5 KB at 50 Hz keyframes) to a temp file. The backward pass reads in reverse chunks of 64 KB and writes a reversed temp file, which a final chunked reversal turns into the output. RAM is O(1) in sequence length. If temp disk is disallowed, the fallback is RAM keyframes at 10 Hz with a ≤ 3 MB cap, plus fixed-lag smoothing for longer sequences.
**Weights** are embedded as a generated `weights.h` (float32 constexpr): one self-contained binary with no runtime file I/O. The NN gets ≤ 2 ms per 1 s of IMU, so a streaming GRU or TCN with cached state and ≤ 100 k params is the target.

## 4. Repository layout (the GitHub repo, which is also the zip's `src/`)
```
src/
  main.cpp, core/{ingest,init,eskf,nn,forecast,smoother,writer}.hpp, third_party/{eigen,miniz}
  weights.h (generated)  build.sh  Makefile  Dockerfile  .gitattributes (LF)
  main.py            # stdlib only: builds via g++ if needed, runs binary; --engine python = numpy reference
  README.md          # build/run, Docker, reproduce training, results
  research/          # Python 3.11-compatible, pinned requirements.txt
    data/ (download+extract only imu0 & state_groundtruth_estimate0), degrade.py, datasets.py,
    models/, train.py, bakeoff.py, forecast_bakeoff.py, export_weights.py, ref_pipeline.py (numpy twin of C++)
  eval/ score.py (evo wrapper + own fast ATE/RPE, all align modes, tier metrics), budget.sh (WSL: /usr/bin/time -v, taskset -c 0), format_check.py
devlog.md   metadata.json   estimated_trajectory.txt
```

## 5. ML protocol (test several candidates, pick the best or a hybrid)
**Data.** All 11 EuRoC sequences, plus TUM-VI room1–6 (full mocap GT) as extra training diversity. The split is **leave-one-sequence-out (LOSO)** CV, grouped by environment (MH vs V1 vs V2) to check generalisation. **Degradation simulator** (`degrade.py`) with domain randomisation:
- noise σ(t) = σ0·(1+αt^p)
- bias random walk with escalating intensity, plus a ramp
- Poisson dropouts of 0.2–2.5 s, rendered randomly as missing rows, NaN, zeros or frozen values
- spikes and saturation
- terminal failure with the last 5 s invalid or truncated
- several seeds per sequence

When the organizer script arrives, it becomes one mode of the simulator and the evaluation set is regenerated with it.
**The selection metric is always end-to-end, not the component loss.** Proxy J = 0.30·ATE_ckpt/ATE_B1 + 0.20·ATE_blind/ATE_B1 + 0.10·ATE_dropout/ATE_B1, computed for each evo alignment mode, with the worst-case mode reported. Ties are broken by RPE and runtime.

| Stage | Candidates | Notes |
|---|---|---|
| Baselines | B0 naive double integration; B1 ESKF + static init + ZUPT | B1 is the shippable floor from day 1 |
| Velocity model | M0 rotor-drag physics only · M1 ResNet1D (RoNIN/TLIO-lite) · M2 causal dilated TCN · M3 GRU-32/64 · M4 AirIO-style body-frame CNN/MLP · M5 LightGBM on window stats (deployed as C++ if-else codegen) · M6 small Transformer (teacher only, distilled if it wins) | Train MSE first, then Gaussian NLL for the covariance (TLIO recipe). Deploy gate: ≤ 2 ms/s, ≤ 1 MB |
| Gyro corrector | G0 none (EKF bias) · G1 Brossard dilated-CNN denoiser · G2 GRU corrector | Judged by orientation RPE |
| Dropout bridging | I0 zero-order hold · I1 linear IMU interpolation · I2 smoother-only · I3 bidirectional-GRU imputer | Offline processing lets us use future data |
| Blind 5 s | F0 hold position · F1 constant velocity · F2 CV with exponential decay τ · F3 constant acceleration · F4 3D constant turn-rate · F5 GRU/MLP forecaster on the last 2 s of state · F6 kNN motion-primitive retrieval · F7 horizon-weighted blend of the best ones | Trained on GT windows **with estimate-error noise injected**, plus real pipeline end states. Shrinkage toward zero velocity is expected to help because degradation peaks right before the failure |
| Hybrid | The EKF fuses every surviving measurement source under its learned covariance (a principled hybrid). The forecaster blend F7 | Ablation table: each component on/off |

Hyperparameter and covariance tuning uses Optuna on CV folds only, with a fixed budget, and the CV-vs-holdout gap is reported to detect overfitting. **Final weights** are trained on all sequences after selection is frozen. **Not taken:** fingerprinting or memorising public sequences in case the hidden test reuses them. It is against the spirit of the contest, carries disqualification risk, and gets logged in the devlog.

## 6. Schedule (A = ML/research, B = systems/C++/eval). Ship something shippable every day.
| Day | A | B |
|---|---|---|
| 10 Nov (Tue) | Email organizers; create devlog.md; download EuRoC (TUM-VI rooms in background) | Repo skeleton, .gitattributes, WSL toolchain, `format_check.py`, `score.py` (cross-checked against evo) |
| 11 Nov | `degrade.py` v1, dataset windows, B0/B1 Python reference | C++ ingest, writer, zip/folder/csv, blind-append, budget.sh. **B1 in C++ = submission v0** |
| 12 Nov | Train M1–M5, G1, fit drag k (M0) | ESKF C++ ↔ Python parity (Δpose < 1e-6), adaptive Q/R, gap mode |
| 13 Nov | Velocity and gyro bake-off round 1 (LOSO), NLL stage, keep the top 2 | Two-filter smoother with disk spill (Python reference, then C++); I0–I2 |
| 14 Nov | Forecaster bake-off F0–F7; I3 | C++ inference for the top-2 nets, `export_weights.py`, parity Δ < 1e-4 |
| 15 Nov | End-to-end C++ eval on all sequences, seeds and align modes; ablations; choose hybrid; Optuna tune | Integration; profiling under `taskset -c 0`; RSS < 4 MB |
| 16 Nov | Final training on all data → freeze model v1; results tables | Torture tests (§8); Dockerfile; README; metadata.json |
| 17 Nov | PPT `CollegeName_TeamName`, architecture diagram | ≤ 3 min video; clean-clone rebuild in fresh WSL; build the zip. **Code freeze 22:00** |
| 18 Nov | Buffer; submit via the Google Form by 18:00 | |

**Scope-cut order if we slip:** M6 → I3 → TUM-VI *training* data → G2 → F6. TUM-VI rooms are still used for evaluation as a handheld/XR generalisation result for the deck (D-015). Never cut: format tests, budgets, B1 fallback.

## 7. devlog.md (created on day 0; every decision appended)
Each entry has: `ID · date · decision · context · reason/evidence (EXP link) · alternatives not taken + why · status`. It also holds an **experiment table** (`EXP-### · config · data/seed · ATE/RPE per tier · runtime · RSS · verdict`).
Seed entries, taken from this plan:
- **D-001 C++ runtime + Python R&D.** Not taken: pure Python, since numpy alone is about 25 MB RSS and per-sample loops are too slow for 20 ms/s. Rust was also rejected (team familiarity and build friction).
- **D-002 ESKF backbone fused with learned measurements.** Not taken: end-to-end network pose regression, which is unstable and can't express uncertainty or degrade gracefully. Plain integration explodes quadratically.
- **D-003 Offline two-filter smoother with disk spill.** Not taken: in-RAM RTS (about 70 MB for long sequences) and forward-only filtering, which wastes the offline advantage, especially for gaps.
- **D-004 Body-frame velocity net (AirIO-style).** Not taken: global-frame RoNIN inputs, which are worse for drones because yaw is unobservable.
- **D-005 Rotor-drag physics measurement.** Not taken: ignoring the multirotor aerodynamics.
- **D-006 Exact integer ns→s formatting.** Not taken: double printf, which can produce rounding misalignment.
- **D-007 Our own randomised degradation simulator.** Not taken: waiting for the organizer script.
- **D-008 LOSO CV with end-to-end proxy J.** Not taken: random window splits, which leak sequence information.
- **D-009 Eigen fixed-size + miniz vendored.** Not taken: hand-rolled matrix code, because of bug risk.
- **D-010 Embedded weights.h.** Not taken: a weights file plus a loader.
- **D-011 One output row per input row + `--blind auto`.** Not taken: resampling to a uniform grid.
- **D-012 No sequence memorisation.**
- **D-013 Video ≤ 3 min.**

## 8. Red-team review (from the evaluator's, the judge's and Murphy's point of view)
- **Auto-eval format is the biggest single risk.** `format_check.py` asserts 8 columns, row count equal to input rows (+ blind rows), timestamps exactly equal to input ns/1e9, |q|=1, no NaN or inf, LF line endings, and that evo loads the file. It runs in CI on every commit and on every torture case: CRLF, no header, duplicate or out-of-order timestamps, NaN rows, zero rows, a 2.5 s gap at start, middle and end, no static start, 30-min sequences, a nested zip root, and a bare CSV.
- **Overfitting to our own simulator.** Wide randomisation; learned covariance plus gating lets the filter ignore a confused net; the physics-only fallback is always compiled in (`--no-nn`). Report results on a "surprise" degradation style held out of training.
- **The `main.py` "standalone Python script" wording.** `main.py` is stdlib-only: it auto-builds and runs the binary, and `--engine python` gives the numpy reference with identical output. The README explains that the efficiency figures refer to the binary.
- **Python 3.14 locally vs a probable 3.11 evaluator.** Research code is pinned to 3.11 and tested in WSL with a 3.11 venv. The C++ binary is independent of either.
- **Windows dev vs Linux eval.** All budget measurements happen in WSL. Builds are `-O2 -march=x86-64-v2` (portable), statically linked, with no `-ffast-math` (avoids NaN surprises).
- **Runtime counts startup, parsing and I/O.** This is measured as wall time over IMU duration, with the median of 5 runs on a single pinned core.
- **The judges at the 27 Nov demo reward explanation.** Devlog, ablation tables, live plots (`eval/plots`). Innovation highlights: learned-covariance fusion, rotor-drag physics, an O(1)-RAM offline smoother, a horizon-aware forecaster.
- **A worse score under some alignment mode** is caught because the harness always scores all modes and we pick on the worst case.

## 9. Verification (end-to-end)
1. `bash src/build.sh && ./pathfinder data/MH_03_degraded.zip -o est.txt` → `python eval/format_check.py est.txt data/.../imu0/data.csv` passes.
2. `python eval/score.py --gt ... --est est.txt --modes all` prints ATE/RPE per tier and J. The numbers match `evo_ape tum gt est -a` and `evo_rpe` within 1e-6.
3. `bash eval/budget.sh` (WSL, `taskset -c 0`, `/usr/bin/time -v`) reports peak RSS < 8 MB (target < 4) and ms per IMU-second < 20 (target < 5) on the longest sequence.
4. Parity: the C++ and Python reference outputs agree (pose Δ < 1e-4 with the NN, < 1e-6 without).
5. LOSO table: the final hybrid beats B1 and every single-component variant on J; the results are recorded in the devlog.
6. Rebuild from a clean clone in fresh WSL (and Docker if available), then rebuild the zip and check its layout against the bundle tree in the slides.
