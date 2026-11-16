# Path Finder: Trajectory Estimation Under Sensor Degradation

Samsung PRISM XR Hackathon 2026, Track 3. IMU-only 6-DoF trajectory estimation with 5 s post-failure dead reckoning. The submission is one Python script (`src/main.py`, numpy only) that stays within 8 MB of heap (2.8 MB measured) and 20 ms of compute per IMU-second (about 4 ms measured).

It is an error-state Kalman filter fused with a small learned velocity model, which is trained on EuRoC (drone) and TUM-VI (handheld) data degraded by the organisers' own script and embedded in `main.py`. Short dropouts are bridged by interpolating the IMU, and an adaptive workspace prior bounds the drift. On held-out sequences degraded by the organisers' script, position error before the failure is 2.8 m on EuRoC (3.6 m for a pose that never moves) and 1.3 m on TUM-VI.

```sh
pip install numpy
python src/main.py input.zip -o estimated_trajectory.txt
```

- Run, contract, results, layout and tests: [`src/README.md`](src/README.md)
- Submission checklist (every rule, how it is met, what is left for the team): [`docs/submission_checklist.md`](docs/submission_checklist.md)
- Plan and architecture: [`docs/plan.md`](docs/plan.md)
- Decision log (every choice, the alternatives and the evidence): [`devlog.md`](devlog.md)
- Earlier C++ runtime (no longer allowed by the FAQ): [`archive/`](archive/README.md)
