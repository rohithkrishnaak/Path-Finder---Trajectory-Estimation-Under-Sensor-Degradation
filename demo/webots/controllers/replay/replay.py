"""Webots supervisor: replay a held-out flight and record it as a movie.

The Mavic follows the hidden ground truth, the orange ghost follows Path Finder's IMU-only estimate (SE(3) aligned,
exactly what the evaluator does) and the red marker follows the physics-only filter. After the IMU fails the replay
slows down to show the 5 s of dead reckoning.

Environment (all optional):
  PF_REPLAY  replay file name in this folder (default V1_02_medium.replay.csv)
  PF_MOVIE   absolute path of the .mp4 to record (default: no recording)
  PF_SPEED   playback speed before the failure (default 1.5)
"""
import math
import os

from controller import Supervisor

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = os.environ.get("PF_REPLAY", "V1_02_medium.replay.csv")
MOVIE = os.environ.get("PF_MOVIE", "")
SPEED = float(os.environ.get("PF_SPEED", "1.5"))
SLOW = 0.5          # playback speed in the blind window
TRAIL_DT = 0.25     # seconds of flight between breadcrumbs

rows = []
with open(os.path.join(HERE, NAME)) as f:
    next(f)
    for line in f:
        rows.append([float(x) for x in line.split()])
T = [r[0] for r in rows]
t_fail = next(r[0] for r in rows if r[18] == 2)
seq = NAME.split(".")[0]

sup = Supervisor()
dt = int(sup.getBasicTimeStep())
gt, est, nonn = sup.getFromDef("GT"), sup.getFromDef("EST"), sup.getFromDef("NONN")
view = sup.getFromDef("VIEW")
trails = sup.getFromDef("TRAILS").getField("children")
gt_tr, gt_rot = gt.getField("translation"), gt.getField("rotation")
est_tr, est_rot = est.getField("translation"), est.getField("rotation")
nonn_tr = nonn.getField("translation")
v_pos, v_rot = view.getField("position"), view.getField("orientation")
sup.getFromDef("TARGET").getField("translation").setSFVec3f([rows[-1][1], rows[-1][2], 0.005])

cx = sum(r[1] for r in rows) / len(rows)
cy = sum(r[2] for r in rows) / len(rows)


def q_to_axis_angle(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-9:
        return [0, 0, 1, 0]
    return [x / n, y / n, z / n, 2 * math.atan2(n, w)]


def mat_to_axis_angle(m):
    tr = m[0][0] + m[1][1] + m[2][2]
    w = math.sqrt(max(0.0, 1 + tr)) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + m[0][0] - m[1][1] - m[2][2])) / 2, m[2][1] - m[1][2])
    y = math.copysign(math.sqrt(max(0.0, 1 - m[0][0] + m[1][1] - m[2][2])) / 2, m[0][2] - m[2][0])
    z = math.copysign(math.sqrt(max(0.0, 1 - m[0][0] - m[1][1] + m[2][2])) / 2, m[1][0] - m[0][1])
    return q_to_axis_angle(x, y, z, w)


def look_at(eye, target):
    d = [target[i] - eye[i] for i in range(3)]
    yaw = math.atan2(d[1], d[0])
    pitch = math.atan2(-d[2], math.hypot(d[0], d[1]))
    cy_, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    m = [[cy_ * cp, -sy, cy_ * sp], [sy * cp, cy_, sy * sp], [-sp, 0, cp]]   # Rz(yaw) * Ry(pitch)
    v_pos.setSFVec3f(eye)
    v_rot.setSFRotation(mat_to_axis_angle(m))


TRAIL_KEEP = 36     # breadcrumbs kept per trail (about 9 s of flight)
crumbs = {"gt": [], "est": [], "nonn": []}


def crumb(kind, p, rgb, r, keep=TRAIL_KEEP):
    trails.importMFNodeFromString(-1, (
        f"Pose {{ translation {p[0]:.3f} {p[1]:.3f} {p[2]:.3f} children [ Shape {{ appearance PBRAppearance {{ "
        f"baseColor {rgb} emissiveColor {rgb} metalness 0 roughness 1 }} geometry Sphere {{ radius {r} subdivision 1 }} }} ] }}"))
    crumbs[kind].append(trails.getMFNode(-1))
    while keep and len(crumbs[kind]) > keep:
        crumbs[kind].pop(0).remove()


def clear_trails():
    for lst in crumbs.values():
        while lst:
            lst.pop().remove()


def interp(t):
    i = 0
    lo, hi = 0, len(T) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if T[mid] <= t:
            lo = mid
        else:
            hi = mid
    i = lo
    a, b = rows[i], rows[min(i + 1, len(rows) - 1)]
    s = 0.0 if b[0] == a[0] else max(0.0, min(1.0, (t - a[0]) / (b[0] - a[0])))
    pos = [a[k] + s * (b[k] - a[k]) for k in range(len(a))]
    pos[4:8] = a[4:8] if s < 0.5 else b[4:8]       # quaternions: nearest sample (20 Hz, invisible)
    pos[11:15] = a[11:15] if s < 0.5 else b[11:15]
    pos[18] = a[18]
    return pos


def label(i, text, x, y, size, color):
    sup.setLabel(i, text, x, y, size, color, 0, "Arial")


WHITE, ORANGE, GREEN, RED, PURPLE, YELLOW = 0xFFFFFF, 0xFF9F1C, 0x5CFF7A, 0xFF5050, 0xD39BFF, 0xFFE14D
label(0, f"PATH FINDER  -  held-out EuRoC {seq} (official degradation)", 0.01, 0.01, 0.075, WHITE)
label(2, "DRONE = ground truth (hidden)", 0.70, 0.01, 0.07, GREEN)
label(3, "ORANGE = Path Finder (IMU only)", 0.70, 0.05, 0.07, ORANGE)
label(4, "RED = physics-only filter", 0.70, 0.09, 0.07, RED)

if MOVIE:
    sup.movieStartRecording(MOVIE, 1280, 720, 0, 95, 1, False)

t, wall, last_crumb, sq, n, sq_b, n_b = 0.0, 0.0, -1.0, 0.0, 0, 0.0, 0
hold_fail, hold_end, ang = 2.0, 4.0, 0.0
state, hold = "intro", 0.0
while sup.step(dt) != -1:
    wall += dt / 1000.0
    if state == "intro" and wall > 2.0:
        state = "fly"
    elif state == "fly":
        t += SPEED * dt / 1000.0
        if t >= t_fail:
            t, state, hold = t_fail, "fail", 0.0
            clear_trails()
    elif state == "fail":
        hold += dt / 1000.0
        if hold > hold_fail:
            state = "blind"
    elif state == "blind":
        t += SLOW * dt / 1000.0
        if t >= T[-1]:
            t, state, hold = T[-1], "end", 0.0
    elif state == "end":
        hold += dt / 1000.0
        if hold > hold_end:
            break

    r = interp(t)
    g, e, nn = r[1:4], r[8:11], r[15:18]
    gt_tr.setSFVec3f(g)
    gt_rot.setSFRotation(q_to_axis_angle(*r[4:8]))
    est_tr.setSFVec3f(e)
    est_rot.setSFRotation(q_to_axis_angle(*r[11:15]))
    nonn_tr.setSFVec3f(nn)
    gt.resetPhysics()

    err = math.dist(g, e)
    blind = state in ("blind", "end")
    step_crumb = 0.1 if blind else TRAIL_DT
    if state in ("fly", "blind") and t - last_crumb >= step_crumb:
        last_crumb = t
        crumb("gt", g, "0.1 0.8 0.3", 0.03, 0 if blind else TRAIL_KEEP)
        crumb("est", e, "0.7 0.3 1" if blind else "1 0.55 0.05", 0.035 if blind else 0.025, 0 if blind else TRAIL_KEEP)
        crumb("nonn", nn, "0.9 0.15 0.15", 0.018, 0 if blind else TRAIL_KEEP)
        if blind:
            sq_b += err * err; n_b += 1
        else:
            sq += err * err; n += 1

    # camera: slow orbit around the drone; closer during the blind window
    ang += (0.12 if state == "fly" else 0.05) * dt / 1000.0
    close = state in ("fail", "blind", "end")
    rad, hgt = (2.6, 1.9) if close else (4.8, 3.2)
    tgt = [0.7 * g[0] + 0.3 * e[0], 0.7 * g[1] + 0.3 * e[1], 0.5 * (g[2] + e[2])]
    look_at([tgt[0] + rad * math.cos(ang + 3.6), tgt[1] + rad * math.sin(ang + 3.6), hgt], tgt)

    rms = math.sqrt(sq / n) if n else 0.0
    if state in ("intro", "fly"):
        if r[18] == 1:
            label(1, f"t = {t:5.1f} s    DROPOUT: rows missing, bridging the gap", 0.01, 0.055, 0.08, YELLOW)
        else:
            label(1, f"t = {t:5.1f} s    tracking through noise, bias drift and spikes", 0.01, 0.055, 0.08, WHITE)
        label(5, f"error now {err:4.2f} m     ATE so far {rms:4.2f} m", 0.01, 0.10, 0.08, ORANGE)
    elif state == "fail":
        label(1, f"t = {t:5.1f} s    IMU FAILED - last checkpoint passed", 0.01, 0.055, 0.09, RED)
        label(5, f"ATE before failure {rms:4.2f} m.  Next: 5 s of pure dead reckoning", 0.01, 0.10, 0.08, ORANGE)
    else:
        rb = math.sqrt(sq_b / n_b) if n_b else 0.0
        label(1, f"t = {t:5.1f} s    BLIND 5 s (slow motion): no sensor data", 0.01, 0.055, 0.08, PURPLE)
        label(5, f"error now {err:4.2f} m     blind ATE {rb:4.2f} m     (purple = prediction)", 0.01, 0.10, 0.08, PURPLE)

if MOVIE:
    sup.movieStopRecording()
    while not sup.movieIsReady():
        if sup.step(dt) == -1:
            break
    print("movie status", sup.movieFailed())
sup.simulationQuit(0)
