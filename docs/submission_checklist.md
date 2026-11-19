# Track 3 submission checklist

Sources: the organisers' slides (14-17 and 24), `docs/organisers/Track3_FAQ_Responses.txt` and `docs/organisers/degrade_imu.py` (used unmodified by the tooling as `src/research/official/degrade_imu.py`). Deadline: **22 Nov 2026, 11:59 PM**. "Teams NOT following the submission guideline would lead to direct disqualification."

## Met and verified by the tooling

| Rule | How it is met | Verified by |
|---|---|---|
| Python only, standalone `main.py` (FAQ) | `src/main.py` is one file; the model is embedded; it needs only numpy | `make_bundle.py` re-runs the bundled file alone from a clean folder |
| Bundle layout: `estimated_trajectory.txt`, `metadata.json`, `src/main.py`, `src/README.md` (slide 17) | `make_bundle.py` writes exactly these four entries | layout assertion in `make_bundle.py` |
| Input: zipped folder, `imu0/data.csv`, EuRoC header (slide 15) | zip / folder / CSV accepted; column order and units auto-detected | `tests/torture.py` (22 cases) |
| Output: `estimated_trajectory.txt`, space-separated, 8 columns, `#` header, seconds, unit quaternion (slide 16) | `Writer` in `main.py`; timestamps printed from integer ns | `eval/format_check.py` |
| Exactly one pose per IMU frame; never append (slide 16, FAQ) | one pose per input row, duplicates and the blind rows included | `format_check.py` fails on any count or timestamp mismatch |
| Only the degraded IMU file is used; no waypoints (FAQ) | nothing else is read | code |
| ≤ 8 MB peak memory, algorithm / heap level (slide 14, FAQ) | 4.88 MB on the longest EuRoC sequence with the offline smoother and look-ahead model (D-052, D-054); the input and pose buffer are memory-mapped beyond 5 minutes | `eval/bench.py --mem` (tracemalloc) |
| ≤ 20 ms per 1 s of IMU, single CPU thread, offline (slide 14) | about 5-6 ms on a Windows laptop, one thread; no network or threads used | `bench.py`, WSL run with `taskset -c 0` |
| Dropouts are missing rows; blind 5 s rows carry invalid values (FAQ) | short gaps bridged, zero rows treated as the failure | `data/official*` cases built with the organisers' script |
| Datasets: EuRoC MAV and TUM-VI (slide 14) | the model is trained on both; both have held-out test cases | `bench.py` on `data/official` and `data/official_tumvi` |
| README with reproducible setup, Docker files and requirements (slide 24) | `src/README.md`, `src/Dockerfile`, `src/requirements.txt` | Docker itself is not installed on the dev machine, so the image build is untested |

## GitHub checklist (submission form)

| Form item | Status | Where |
|---|---|---|
| Submission package, as per theme | Done (built from a held-out test input; rebuild from the official input when it is released) | `submission/VIT_tan-x.zip` |
| Source code | Done | `src/` |
| Presentation | Done: rewritten for the Python runtime and the current results; team and college filled in | `VIT_tan-x_Submission.pptx` / `.pdf` |
| Video | Done: 2 min 47 s walkthrough with a Webots replay; upload it to YouTube or Drive and paste the link into slide 7 | `demo/video/PathFinder_demo.mp4`, `demo/README.md` |
| AI disclosure | **Team to write** | — |
| README | Done | `README.md`, `src/README.md`, `src/Dockerfile`, `src/requirements.txt` |
| APK / SDK | Not applicable (Python script) | — |

## Left for the team

| Item | Note |
|---|---|
| Team name | Done: `VIT_tan-x` (VIT, Vellore; team tan-x). Rebuild the bundle with `python src/eval/make_bundle.py <official input.zip> --team VIT_tan-x` |
| Deck placeholders | Slide 7: video link |
| Final `estimated_trajectory.txt` | Must come from the organisers' evaluation input once it is released. The bundle in `submission/` uses held-out MH_05 (`data/official_ho/MH_05_difficult_o42.zip`) |
| GitHub repository | Public or shared with the organisers; commit and push the current tree |
| AI disclosure | Team's own document |
| Google Form | One submission per team; the link is announced in the official channel |

## Open risks (worth asking the organisers)

| Risk | Detail |
|---|---|
| Where memory tracing starts | Importing numpy alone allocates about 7.7 MB under tracemalloc. Our 2.77 MB counts the algorithm (model + buffers), which matches the FAQ wording |
| Column order of the evaluation file | Slide 15 says gyro first; `degrade_imu.py` documents accel first. Both are handled and tested |
| Temp files for long inputs | Recordings over 5 minutes are parsed into temp files and memory-mapped. If temp files are not allowed, such a file would need about 37 bytes of heap per row instead |
| numpy availability | Assumed, because the organisers' own script needs numpy and pandas |
