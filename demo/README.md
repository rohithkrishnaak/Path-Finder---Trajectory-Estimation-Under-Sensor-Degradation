# Demo: Webots replay and walkthrough video

`demo/video/PathFinder_demo.mp4` (2 min 47 s, 1280x720) is the pipeline walkthrough video that the Track 3 slides ask for. It has no voice-over; on-screen captions explain each step.

It has three parts:
1. **Cards**: the problem, how `src/main.py` works, and what the replay shows.
2. **Webots replay** of EuRoC **V1_02_medium**, a flight the model never saw in training, degraded by the organisers' `degrade_imu.py`:
   - The DJI Mavic flies the hidden ground truth.
   - The **orange** ghost is Path Finder's estimate from the degraded IMU only, SE(3)-aligned exactly like the evo evaluator.
   - The **red** marker is the same filter without the learned velocity.
   - The HUD shows the live error and the running ATE, and flags dropouts.
   - When the IMU fails, the replay slows to 0.5x and the 5 s dead-reckoned prediction is drawn in **purple**.
3. **Cards**: held-out results, the hardest flight (MH_05), budgets, how to run it, and limitations.

## Rebuild it
Requires Webots R2025a (its bundled ffmpeg is used) and the repo's Python environment with numpy and matplotlib.

```sh
# 1. trajectories for the held-out flight (input built by src/research/make_official.py)
python src/main.py data/official_ho/V1_02_medium_o42.zip --weights weights/mix/ho_ens.bin -o demo/data/V1_02_medium.est.txt   # model trained without V1_02
python src/main.py data/official_ho/V1_02_medium_o42.zip --no-nn -o demo/data/V1_02_medium.nonn.txt
cp data/official_ho/V1_02_medium_o42.gt.txt   demo/data/V1_02_medium.gt.txt
cp data/official_ho/V1_02_medium_o42.meta.json demo/data/V1_02_medium.meta.json

# 2. align (SE(3), like evo) and write the 20 Hz replay file + top-down figure
python demo/prepare.py --name V1_02_medium --gt demo/data/V1_02_medium.gt.txt \
    --est demo/data/V1_02_medium.est.txt --nonn demo/data/V1_02_medium.nonn.txt --meta demo/data/V1_02_medium.meta.json

# 3. record the Webots replay (opens Webots, about 70 s in real time, then quits)
PF_MOVIE="$PWD/demo/video/build/webots_V1_02.mp4" webots --mode=realtime --batch demo/webots/worlds/pathfinder_demo.wbt

# 4. assemble cards + replay
python demo/video/make_video.py
```

You can also open `demo/webots/worlds/pathfinder_demo.wbt` in Webots and press play to watch the replay without recording. The controller's Python interpreter is set in `demo/webots/controllers/replay/runtime.ini`; change it to match your machine.

Environment variables for the controller:

| Variable | Meaning | Default |
|---|---|---|
| `PF_REPLAY` | replay file to play | `V1_02_medium.replay.csv` (`MH_05_difficult.replay.csv` is also provided) |
| `PF_MOVIE` | where to record the movie | not set: no recording |
| `PF_SPEED` | playback speed before the failure | 1.5 |

The replay files are small (20 Hz) and committed. `demo/data/` and `demo/video/build/` are regenerated and ignored by git.
