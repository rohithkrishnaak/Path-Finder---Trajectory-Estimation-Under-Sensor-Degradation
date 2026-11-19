# Archive

`cpp_runtime/` is the C++17 runtime (Eigen + miniz) that the project used until 2026-11-16. The Track 3 FAQ
then required Python only ("submit a standalone Python script, main.py"), so `src/main.py` is now the single
runtime. The C++ code is kept for reference; it is not part of the submission.

- `main.cpp`, `core/`: the ESKF + learned-velocity pipeline that `src/main.py` was ported from.
- `research/ref_pipeline.py`: the numpy twin of the C++ baseline (B1) used for parity tests.
- `tests/torture_cpp_parity.py`, `eval/budget.sh`: C++/Python parity and the Linux RSS measurement.
- Build (if ever needed): `sh archive/cpp_runtime/build.sh`.

`experiments/` holds scripts for ideas that were tested and rejected (devlog D-041): `arch_search.py` (CNN / GRU /
wide-MLP comparison) and `train_gru.py` + `export_gru.py` (streaming GRU). They were written against the
`src/research/train_vel.py` of 2026-11-16 and are kept only so the recorded results can be traced.

`scratch_bench/` holds the five synthetic 50 s ground-truth clips (drone and handheld) that D-050 was measured on. Their IMU input files were never committed, so the cases cannot be re-run. D-051 explains why the evaluation uses real held-out EuRoC / TUM-VI flights instead.

`VIO-Edge.html` is an earlier exported design / overview page, kept for reference. It is not part of the submission.
