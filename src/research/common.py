"""Shared I/O and rotation utilities for the research code (numpy only).

Conventions (identical to the C++ runtime):
  * World frame W: z up, gravity g_W = [0, 0, -G].
  * Quaternions are Hamilton, stored as [w, x, y, z] internally; q represents R_WB
    (rotates body-frame vectors into the world frame).
  * Accelerometer measures specific force f_B = R_WB^T (a_W - g_W).
  * TUM output order is "t tx ty tz qx qy qz qw".
"""
from __future__ import annotations

import io
import math
import os
import zipfile
from dataclasses import dataclass

import numpy as np

G = 9.81  # m/s^2 (also used in C++; EuRoC is ~9.81 in Zurich)


# ----------------------------------------------------------------------------- rotations
def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_exp(phi):
    """Rotation vector -> unit quaternion [w,x,y,z]."""
    ang = math.sqrt(phi[0] ** 2 + phi[1] ** 2 + phi[2] ** 2)
    if ang < 1e-12:
        q = np.array([1.0, 0.5 * phi[0], 0.5 * phi[1], 0.5 * phi[2]])
        return q / np.linalg.norm(q)
    s = math.sin(0.5 * ang) / ang
    return np.array([math.cos(0.5 * ang), phi[0] * s, phi[1] * s, phi[2] * s])


def quat_log(q):
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    v = q[1:]
    n = np.linalg.norm(v)
    if n < 1e-12:
        return 2.0 * v
    return 2.0 * math.atan2(n, q[0]) * v / n


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def quats_to_rots(Q):
    """(N,4) wxyz -> (N,3,3). Normalises first (non-unit q gives a non-orthonormal R)."""
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    w, x, y, z = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    R = np.empty((len(Q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rots_to_quats(R):
    return np.array([rot_to_quat(r) for r in R])


def rot_angle(R):
    """Geodesic angle of rotation matrix/matrices (rad). atan2 form: accurate near 0 and pi."""
    tr = np.trace(R, axis1=-2, axis2=-1)
    vx = R[..., 2, 1] - R[..., 1, 2]
    vy = R[..., 0, 2] - R[..., 2, 0]
    vz = R[..., 1, 0] - R[..., 0, 1]
    return np.arctan2(0.5 * np.sqrt(vx * vx + vy * vy + vz * vz), 0.5 * (tr - 1.0))


# ----------------------------------------------------------------------------- EuRoC / TUM I/O
IMU_HEADER = ("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
              "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]")


@dataclass
class ImuRows:
    """Raw rows exactly as in the CSV. Missing/unparseable fields are NaN."""
    t_ns: np.ndarray      # int64 (N,)
    gyro: np.ndarray      # float64 (N,3)
    acc: np.ndarray       # float64 (N,3)


def find_imu_csv_in_zip(zf: zipfile.ZipFile) -> str:
    names = [n for n in zf.namelist() if n.replace("\\", "/").lower().endswith("imu0/data.csv")]
    if not names:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if not names:
        raise FileNotFoundError("no imu0/data.csv in zip")
    return sorted(names, key=len)[0]


def resolve_imu_text(path: str) -> str:
    """Accept a zip, a folder (searched for imu0/data.csv) or a CSV file; return CSV text."""
    if os.path.isdir(path):
        for root, _dirs, files in os.walk(path):
            if os.path.basename(root) == "imu0" and "data.csv" in files:
                with open(os.path.join(root, "data.csv"), "r", newline="") as f:
                    return f.read()
        raise FileNotFoundError(f"no imu0/data.csv under {path}")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            return zf.read(find_imu_csv_in_zip(zf)).decode("utf-8", errors="replace")
    with open(path, "r", newline="") as f:
        return f.read()


def _to_float(s: str) -> float:
    s = s.strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_imu_text(text: str) -> ImuRows:
    """Tolerant parser mirroring the C++ one: skips blank/comment/header lines,
    accepts ',', ';', whitespace separators, CRLF; missing fields -> NaN."""
    ts, gy, ac = [], [], []
    for line in io.StringIO(text):
        line = line.strip()
        if not line or line[0] == "#":
            continue
        parts = [p for p in line.replace(";", ",").replace("\t", ",").split(",")]
        if len(parts) == 1:
            parts = line.split()
        try:
            t = int(parts[0].strip())
        except ValueError:
            try:
                t = int(round(float(parts[0].strip())))  # e.g. scientific notation
            except ValueError:
                continue  # header line without '#'
        vals = [_to_float(p) for p in parts[1:7]]
        vals += [float("nan")] * (6 - len(vals))
        ts.append(t)
        gy.append(vals[0:3])
        ac.append(vals[3:6])
    return ImuRows(np.array(ts, dtype=np.int64), np.array(gy, dtype=float).reshape(-1, 3),
                   np.array(ac, dtype=float).reshape(-1, 3))


def load_imu(path: str) -> ImuRows:
    return parse_imu_text(resolve_imu_text(path))


def write_imu_csv(path: str, t_ns, gyro, acc, header: bool = True, crlf: bool = False,
                  blank_mask=None, fmt: str = "%.10g"):
    """Write EuRoC IMU CSV. Rows where blank_mask is True get empty value fields."""
    nl = "\r\n" if crlf else "\n"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        if header:
            f.write(IMU_HEADER + nl)
        for i in range(len(t_ns)):
            if blank_mask is not None and blank_mask[i]:
                f.write(f"{int(t_ns[i])},,,,,," + nl)
                continue
            vals = list(gyro[i]) + list(acc[i])
            f.write(str(int(t_ns[i])) + "," + ",".join("nan" if not np.isfinite(v) else (fmt % v) for v in vals) + nl)


@dataclass
class Traj:
    t: np.ndarray   # seconds (float64)
    p: np.ndarray   # (N,3)
    q: np.ndarray   # (N,4) wxyz


def load_tum(path: str) -> Traj:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(x) for x in line.split()])
    a = np.array(rows, dtype=float).reshape(-1, 8)
    return Traj(a[:, 0], a[:, 1:4], np.column_stack([a[:, 7], a[:, 4:7]]))


def write_tum(path: str, traj: Traj):
    with open(path, "w", newline="\n") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for t, p, q in zip(traj.t, traj.p, traj.q):
            f.write(f"{t:.9f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} {q[0]:.9f}\n")


def load_euroc_gt(path: str) -> Traj:
    """EuRoC state_groundtruth_estimate0/data.csv (ns, p xyz, q wxyz, ...) or TUM-VI mocap0/data.csv."""
    a = np.loadtxt(path, delimiter=",", comments="#", usecols=range(8))
    return Traj(a[:, 0] * 1e-9, a[:, 1:4], a[:, 4:8] / np.linalg.norm(a[:, 4:8], axis=1, keepdims=True))


def find_gt_csv(seq_dir: str) -> str:
    for sub in ("state_groundtruth_estimate0", "mocap0", "vicon0", "leica0"):
        for root, dirs, files in os.walk(seq_dir):
            if os.path.basename(root) == sub and "data.csv" in files:
                return os.path.join(root, "data.csv")
    raise FileNotFoundError(f"no ground truth under {seq_dir}")
