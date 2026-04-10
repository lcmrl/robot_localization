"""Run the self-contained Python EKF on IMU CSV data and save an output trajectory.

Inputs (defaults follow the user's request):
  - CSV:  ./src/data.csv
  - YAML: ./sensor.yaml
    - ABS:  ./src/absolute_sensor.csv

This repo currently ships example files under ./example/data/; if the default
paths do not exist, this script will fall back to:
  - ./example/data/data.csv
  - ./example/data/sensor.yaml
    - ./example/data/absolute_sensor.csv

Output:
  - ./src/output/trajectory.csv

Notes / assumptions:
- The EKF state matches robot_localization's 15-state EKF.
- IMU angular velocity (w_RS_S_*) is mapped to state angular velocity
  (Vroll, Vpitch, Vyaw) as a simple axis correspondence (x->roll-rate, etc.).
- IMU linear acceleration (a_RS_S_*) is mapped to state acceleration (Ax, Ay, Az).
- IMPORTANT: Most IMUs report *specific force* (includes gravity). If you feed that
    directly into the EKF as linear acceleration, you will get unrealistic
    translations quickly. Use --remove_gravity to subtract gravity in the body frame
    using the filter's current roll/pitch estimate.
- absolute_sensor.csv provides absolute pose; only the first 8 columns are used:
    [timestamp_ns, x, y, z, qw, qx, qy, qz]. All remaining columns are ignored.
- relative.csv provides *relative* pose increments; expected columns:
        timestamp_ns, dx, dy, dz, dqw, dqx, dqy, dqz
    These increments are accumulated into a pose stream and fused like an absolute
    pose measurement. If both absolute and relative are enabled, the relative pose
    stream is automatically aligned to the absolute pose (first shared timestamp)
    and used only at timestamps where an absolute measurement is not present.
- Measurement covariances are derived from noise densities in sensor.yaml using
  sigma_sample = noise_density * sqrt(rate_hz) (i.e., noise_density / sqrt(dt)).

This script uses only the Python standard library plus the local src/python_ekf.py.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import math
import os
from pathlib import Path
import re
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


# Local, repo-contained EKF implementation
from src.python_ekf import (
    EKF,
    Measurement,
    StateMemberAx,
    StateMemberAy,
    StateMemberAz,
    StateMemberPitch,
    StateMemberRoll,
    StateMemberVpitch,
    StateMemberVroll,
    StateMemberVyaw,
    StateMemberVx,
    StateMemberVy,
    StateMemberVz,
    StateMemberX,
    StateMemberY,
    StateMemberYaw,
    StateMemberZ,
)


def _gravity_body_from_rp(roll: float, pitch: float, g: float) -> Tuple[float, float, float]:
    """Gravity vector expressed in the body frame, given roll/pitch.

    Assumes world +Z is up and R = Rz(yaw)*Ry(pitch)*Rx(roll).
    gravity_body = R^T * [0,0,g] = g * [-sin(p), cos(p)*sin(r), cos(p)*cos(r)]
    """

    sp = math.sin(pitch)
    cp = math.cos(pitch)
    sr = math.sin(roll)
    cr = math.cos(roll)
    return (-g * sp, g * cp * sr, g * cp * cr)


def _parse_sensor_yaml_for_noise(path: Path) -> Dict[str, float]:
    """Parse only the keys we need from a YAML file (no PyYAML dependency).

    Expected keys (as in example/data/sensor.yaml):
      - rate_hz
      - gyroscope_noise_density
      - accelerometer_noise_density

    Returns dict with those keys as floats.
    """

    text = path.read_text(encoding="utf-8", errors="replace").splitlines()

    wanted = {
        "rate_hz": None,
        "gyroscope_noise_density": None,
        "accelerometer_noise_density": None,
    }

    # Simple "key: value" scanner. Ignores comments and nested structures.
    key_re = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*([^#]+?)\s*(?:#.*)?$")

    for line in text:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = key_re.match(line)
        if not m:
            continue
        key = m.group(1)
        val = m.group(2).strip()
        if key in wanted:
            try:
                wanted[key] = float(val)
            except ValueError as exc:
                raise ValueError(f"Could not parse {key} value '{val}' in {path}") from exc

    missing = [k for k, v in wanted.items() if v is None]
    if missing:
        raise ValueError(f"Missing keys in {path}: {', '.join(missing)}")

    return {k: float(v) for k, v in wanted.items()}  # type: ignore[arg-type]


def _canonicalize_header(name: str) -> str:
    # Remove units annotations like " [ns]" and trim leading '#'
    s = name.strip()
    if s.startswith("#"):
        s = s[1:].strip()
    # Drop bracket units
    s = re.sub(r"\s*\[.*?\]\s*", "", s)
    return s.strip()


def _parse_timestamp_ns(value: str) -> int:
    """Parse a nanosecond timestamp without losing precision.

    Many datasets store ns timestamps around 1e18. Parsing via float will lose
    integer precision, so prefer int/Decimal.
    """

    s = value.strip()
    try:
        return int(s)
    except ValueError:
        pass

    try:
        return int(Decimal(s))
    except (InvalidOperation, ValueError):
        pass

    # Last resort (may lose precision for large ns values)
    return int(float(s))


def _load_imu_csv(path: Path) -> Iterable[Tuple[int, float, float, float, float, float, float]]:
    """Yield IMU rows: (timestamp_ns, wx, wy, wz, ax, ay, az)."""

    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return

        cols = {_canonicalize_header(h): i for i, h in enumerate(header)}

        def idx(*candidates: str) -> int:
            for c in candidates:
                if c in cols:
                    return cols[c]
            raise KeyError(f"Missing expected column; tried: {candidates}. Found: {list(cols.keys())}")

        i_t = idx("timestamp")
        i_wx = idx("w_RS_S_x")
        i_wy = idx("w_RS_S_y")
        i_wz = idx("w_RS_S_z")
        i_ax = idx("a_RS_S_x")
        i_ay = idx("a_RS_S_y")
        i_az = idx("a_RS_S_z")

        for row in reader:
            if not row or all(not x.strip() for x in row):
                continue

            try:
                t_ns = _parse_timestamp_ns(row[i_t])
                wx = float(row[i_wx])
                wy = float(row[i_wy])
                wz = float(row[i_wz])
                ax = float(row[i_ax])
                ay = float(row[i_ay])
                az = float(row[i_az])
            except (ValueError, IndexError):
                # Skip malformed line
                continue

            yield (t_ns, wx, wy, wz, ax, ay, az)


def _load_absolute_pose_csv(path: Path) -> Iterator[Tuple[int, float, float, float, float, float, float, float]]:
    """Yield absolute pose rows: (timestamp_ns, x, y, z, qw, qx, qy, qz).

    File is expected to have no header and potentially many columns; only the
    first 8 are used: [t, x, y, z, qw, qx, qy, qz].
    """

    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 8:
                continue
            try:
                t_ns = _parse_timestamp_ns(row[0])
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
                qw = float(row[4])
                qx = float(row[5])
                qy = float(row[6])
                qz = float(row[7])
            except ValueError:
                continue

            qw, qx, qy, qz = _quat_normalize(qw, qx, qy, qz)
            yield (t_ns, x, y, z, qw, qx, qy, qz)


def _load_relative_pose_csv(path: Path) -> Iterator[Tuple[int, float, float, float, float, float, float, float]]:
    """Yield relative pose increments: (timestamp_ns, dx, dy, dz, dqw, dqx, dqy, dqz).

    Accepts files with or without a header.
    """

    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 8:
                continue

            try:
                t_ns = _parse_timestamp_ns(row[0])
                dx = float(row[1])
                dy = float(row[2])
                dz = float(row[3])
                dqw = float(row[4])
                dqx = float(row[5])
                dqy = float(row[6])
                dqz = float(row[7])
            except ValueError:
                # likely header or malformed row
                continue

            dqw, dqx, dqy, dqz = _quat_normalize(dqw, dqx, dqy, dqz)
            yield (t_ns, dx, dy, dz, dqw, dqx, dqy, dqz)


def _quat_normalize(qw: float, qx: float, qy: float, qz: float) -> Tuple[float, float, float, float]:
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n <= 0.0 or not math.isfinite(n):
        return (1.0, 0.0, 0.0, 0.0)
    inv = 1.0 / n
    return (qw * inv, qx * inv, qy * inv, qz * inv)


def _quat_conj(qw: float, qx: float, qy: float, qz: float) -> Tuple[float, float, float, float]:
    return (qw, -qx, -qy, -qz)


def _quat_mul(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_rotate_vec(q: Tuple[float, float, float, float], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    # v' = q ⊗ (0,v) ⊗ q*
    qw, qx, qy, qz = q
    vx, vy, vz = v
    t2 = qw * qx
    t3 = qw * qy
    t4 = qw * qz
    t5 = -qx * qx
    t6 = qx * qy
    t7 = qx * qz
    t8 = -qy * qy
    t9 = qy * qz
    t10 = -qz * qz
    rx = 2.0 * ((t8 + t10) * vx + (t6 - t4) * vy + (t3 + t7) * vz) + vx
    ry = 2.0 * ((t4 + t6) * vx + (t5 + t10) * vy + (t9 - t2) * vz) + vy
    rz = 2.0 * ((t7 - t3) * vx + (t2 + t9) * vy + (t5 + t8) * vz) + vz
    return (rx, ry, rz)


def _quat_to_rpy(qw: float, qx: float, qy: float, qz: float) -> Tuple[float, float, float]:
    """Convert quaternion (qw,qx,qy,qz) to roll/pitch/yaw (rad).

    Assumes scalar-first quaternion ordering as specified by the user.
    """

    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n <= 0.0 or not math.isfinite(n):
        return (0.0, 0.0, 0.0)
    qw /= n
    qx /= n
    qy /= n
    qz /= n

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    if sinp >= 1.0:
        pitch = math.pi / 2.0
    elif sinp <= -1.0:
        pitch = -math.pi / 2.0
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)


def _make_measurement_covariance(rate_hz: float, gyro_nd: float, accel_nd: float) -> List[List[float]]:
    # Discrete-time stddev approximation from noise density (per sqrt(Hz)).
    # A common approximation is sigma_sample = nd * sqrt(BW), with BW ~= fs/2.
    bw_hz = max(rate_hz * 0.5, 1.0)
    gyro_var = (gyro_nd * math.sqrt(bw_hz)) ** 2
    accel_var = (accel_nd * math.sqrt(bw_hz)) ** 2

    # Order: [Vroll, Vpitch, Vyaw, Ax, Ay, Az]
    cov = [[0.0 for _ in range(6)] for _ in range(6)]
    for i in range(3):
        cov[i][i] = gyro_var
    for i in range(3, 6):
        cov[i][i] = accel_var
    return cov


def main() -> int:
    parser = argparse.ArgumentParser(description="Run python_ekf.py on IMU data.csv")
    parser.add_argument("--data", default="./example/data.csv", help="Path to IMU CSV (default: src/data.csv)")
    parser.add_argument("--sensor", default="./example/data/sensor.yaml", help="Path to sensor.yaml (default: sensor.yaml)")
    parser.add_argument(
        "--absolute",
        default="./example/data/absolute_sensor.csv",
        help="Path to absolute_sensor.csv (ignored if --no_absolute)",
    )
    parser.add_argument(
        "--use_absolute",
        action="store_true",
        default=False,
        help="Use absolute pose measurements from absolute_sensor.csv (default: true)",
    )
    parser.add_argument(
        "--no_absolute",
        action="store_false",
        dest="use_absolute",
        help="Disable absolute pose measurements",
    )
    parser.add_argument(
        "--relative",
        default="./example/data/relative.csv",
        help="Path to relative.csv (ignored if --no_relative)",
    )
    parser.add_argument(
        "--use_relative",
        action="store_true",
        default=True,
        help="Use relative pose increments from relative.csv (default: false)",
    )
    parser.add_argument(
        "--no_relative",
        action="store_false",
        dest="use_relative",
        help="Disable relative pose increments",
    )
    parser.add_argument("--output_dir", default="./example/output", help="Output directory (default: src/output)")
    parser.add_argument("--output_name", default="trajectory.csv", help="Output filename (default: trajectory.csv)")
    parser.add_argument("--mahalanobis", type=float, default=float("inf"), help="Mahalanobis gate (sigmas)")
    parser.add_argument(
        "--remove_gravity",
        action="store_true",
        help="Subtract gravity from accelerometer using current roll/pitch estimate",
        default=True,
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=9.80665,
        help="Gravity magnitude to subtract when --remove_gravity is set (m/s^2)",
    )
    parser.add_argument(
        "--abs_pos_sigma",
        type=float,
        default=0.05,
        help="Absolute pose position measurement standard deviation (meters)",
    )
    parser.add_argument(
        "--abs_ori_sigma",
        type=float,
        default=0.1,
        help="Absolute pose orientation measurement standard deviation (radians)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent

    data_path = (repo_root / args.data).resolve() if not os.path.isabs(args.data) else Path(args.data)
    sensor_path = (repo_root / args.sensor).resolve() if not os.path.isabs(args.sensor) else Path(args.sensor)
    abs_path = (repo_root / args.absolute).resolve() if not os.path.isabs(args.absolute) else Path(args.absolute)
    rel_path = (repo_root / args.relative).resolve() if not os.path.isabs(args.relative) else Path(args.relative)

    # Fallback to example paths if defaults weren't found
    if not data_path.exists():
        fallback = repo_root / "example" / "data" / "data.csv"
        if fallback.exists():
            data_path = fallback
        else:
            raise FileNotFoundError(f"Could not find data CSV at {data_path} (and no fallback at {fallback})")

    if not sensor_path.exists():
        fallback = repo_root / "example" / "data" / "sensor.yaml"
        if fallback.exists():
            sensor_path = fallback
        else:
            raise FileNotFoundError(f"Could not find sensor.yaml at {sensor_path} (and no fallback at {fallback})")

    if args.use_absolute:
        if not abs_path.exists():
            fallback = repo_root / "example" / "data" / "absolute_sensor.csv"
            if fallback.exists():
                abs_path = fallback
            else:
                raise FileNotFoundError(
                    f"Could not find absolute sensor CSV at {abs_path} (and no fallback at {fallback})"
                )

    if args.use_relative:
        if not rel_path.exists():
            fallback = repo_root / "example" / "data" / "relative.csv"
            if fallback.exists():
                rel_path = fallback
            else:
                raise FileNotFoundError(
                    f"Could not find relative CSV at {rel_path} (and no fallback at {fallback})"
                )

    noise = _parse_sensor_yaml_for_noise(sensor_path)
    rate_hz = noise["rate_hz"]
    gyro_nd = noise["gyroscope_noise_density"]
    accel_nd = noise["accelerometer_noise_density"]

    cov6 = _make_measurement_covariance(rate_hz, gyro_nd, accel_nd)
    # Absolute pose sensor covariance: position + orientation
    pos_sigma = float(args.abs_pos_sigma)
    ori_sigma = float(args.abs_ori_sigma)
    if pos_sigma <= 0.0 or ori_sigma <= 0.0:
        raise ValueError("abs_pos_sigma and abs_ori_sigma must be > 0")

    pos_var = pos_sigma * pos_sigma
    ori_var = ori_sigma * ori_sigma
    cov6_abs = [
        [pos_var, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, pos_var, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, pos_var, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, ori_var, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, ori_var, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, ori_var],
    ]

    ekf = EKF()

    out_dir = (repo_root / args.output_dir).resolve() if not os.path.isabs(args.output_dir) else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name

    # Output columns: timestamp_ns + full state vector
    header = [
        "timestamp_ns",
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
        "vx",
        "vy",
        "vz",
        "vroll",
        "vpitch",
        "vyaw",
        "ax",
        "ay",
        "az",
    ]

    # Build update mapping indices for this measurement
    update_indices = [StateMemberVroll, StateMemberVpitch, StateMemberVyaw, StateMemberAx, StateMemberAy, StateMemberAz]

    with out_path.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)

        # Stream and time-merge IMU + (optional) absolute pose + (optional) relative pose increments
        imu_iter = iter(_load_imu_csv(data_path))
        abs_iter = iter(_load_absolute_pose_csv(abs_path)) if args.use_absolute else iter(())
        rel_iter = iter(_load_relative_pose_csv(rel_path)) if args.use_relative else iter(())

        imu_next = next(imu_iter, None)
        abs_next = next(abs_iter, None)
        rel_next = next(rel_iter, None)

        if imu_next is None and abs_next is None and rel_next is None:
            raise ValueError("No measurements found in enabled inputs")

        first_candidates = []
        if imu_next is not None:
            first_candidates.append(imu_next[0])
        if abs_next is not None:
            first_candidates.append(abs_next[0])
        if rel_next is not None:
            first_candidates.append(rel_next[0])
        first_t_ns = min(first_candidates)

        # Relative pose accumulator (in the relative stream's own frame)
        rel_p = (0.0, 0.0, 0.0)
        rel_q = (1.0, 0.0, 0.0, 0.0)
        rel_seen = False
        rel_aligned = not args.use_absolute  # if no absolute, treat rel frame as world
        align_p = (0.0, 0.0, 0.0)
        align_q = (1.0, 0.0, 0.0, 0.0)

        while imu_next is not None or abs_next is not None or rel_next is not None:
            next_ts: int
            next_ts = min(
                ts
                for ts in (
                    imu_next[0] if imu_next is not None else None,
                    abs_next[0] if abs_next is not None else None,
                    rel_next[0] if rel_next is not None else None,
                )
                if ts is not None
            )

            t_sec = (next_ts - first_t_ns) * 1e-9

            # Process all relative increments at this timestamp (update accumulator)
            rel_updated_this_ts = False
            while rel_next is not None and rel_next[0] == next_ts:
                _, dx, dy, dz, dqw, dqx, dqy, dqz = rel_next
                rel_next = next(rel_iter, None)
                rel_updated_this_ts = True
                rel_seen = True

                rel_p = (rel_p[0] + dx, rel_p[1] + dy, rel_p[2] + dz)
                rel_q = _quat_mul(rel_q, (dqw, dqx, dqy, dqz))
                rel_q = _quat_normalize(*rel_q)

            # Process all absolute measurements at this timestamp
            abs_present_this_ts = False
            last_abs_pose: Optional[Tuple[float, float, float, float, float, float, float]] = None
            while abs_next is not None and abs_next[0] == next_ts:
                abs_present_this_ts = True
                _, ax_p, ay_p, az_p, qw, qx, qy, qz = abs_next
                abs_next = next(abs_iter, None)

                roll, pitch, yaw = _quat_to_rpy(qw, qx, qy, qz)
                last_abs_pose = (ax_p, ay_p, az_p, qw, qx, qy, qz)

                meas = Measurement.from_subset(
                    time=t_sec,
                    update_indices=[
                        StateMemberX,
                        StateMemberY,
                        StateMemberZ,
                        StateMemberRoll,
                        StateMemberPitch,
                        StateMemberYaw,
                    ],
                    values=[ax_p, ay_p, az_p, roll, pitch, yaw],
                    covariance=cov6_abs,
                    mahalanobis_thresh=args.mahalanobis,
                )
                ekf.process_measurement(meas)

            # If both streams are enabled, align the relative frame to the absolute pose once.
            # (Does not require an exact timestamp match; uses the latest accumulated rel pose.)
            if (not rel_aligned) and rel_seen and (last_abs_pose is not None):
                ax_p, ay_p, az_p, qw, qx, qy, qz = last_abs_pose
                q_abs = (qw, qx, qy, qz)
                q_rel = rel_q
                align_q = _quat_mul(q_abs, _quat_conj(*q_rel))
                align_q = _quat_normalize(*align_q)

                rel_p_rot = _quat_rotate_vec(align_q, rel_p)
                align_p = (ax_p - rel_p_rot[0], ay_p - rel_p_rot[1], az_p - rel_p_rot[2])
                rel_aligned = True

            # Fuse relative pose as an observation (but avoid double-counting when absolute is present at same timestamp)
            if args.use_relative and rel_updated_this_ts and rel_aligned and (not abs_present_this_ts):
                if args.use_absolute:
                    p_fused = _quat_rotate_vec(align_q, rel_p)
                    p_fused = (p_fused[0] + align_p[0], p_fused[1] + align_p[1], p_fused[2] + align_p[2])
                    q_fused = _quat_mul(align_q, rel_q)
                    q_fused = _quat_normalize(*q_fused)
                else:
                    p_fused = rel_p
                    q_fused = rel_q

                roll, pitch, yaw = _quat_to_rpy(*q_fused)
                meas = Measurement.from_subset(
                    time=t_sec,
                    update_indices=[
                        StateMemberX,
                        StateMemberY,
                        StateMemberZ,
                        StateMemberRoll,
                        StateMemberPitch,
                        StateMemberYaw,
                    ],
                    values=[p_fused[0], p_fused[1], p_fused[2], roll, pitch, yaw],
                    covariance=cov6_abs,
                    mahalanobis_thresh=args.mahalanobis,
                )
                ekf.process_measurement(meas)

            # Process all IMU measurements at this timestamp
            while imu_next is not None and imu_next[0] == next_ts:
                _, wx, wy, wz, ax, ay, az = imu_next
                imu_next = next(imu_iter, None)

                if args.remove_gravity:
                    roll = ekf.state[StateMemberRoll]
                    pitch = ekf.state[StateMemberPitch]
                    g_bx, g_by, g_bz = _gravity_body_from_rp(roll, pitch, float(args.gravity))
                    ax -= g_bx
                    ay -= g_by
                    az -= g_bz

                meas = Measurement.from_subset(
                    time=t_sec,
                    update_indices=update_indices,
                    values=[wx, wy, wz, ax, ay, az],
                    covariance=cov6,
                    mahalanobis_thresh=args.mahalanobis,
                )
                ekf.process_measurement(meas)

            # Write one trajectory row per unique timestamp
            s = ekf.state
            writer.writerow(
                [
                    next_ts,
                    s[StateMemberX],
                    s[StateMemberY],
                    s[StateMemberZ],
                    s[StateMemberRoll],
                    s[StateMemberPitch],
                    s[StateMemberYaw],
                    s[StateMemberVx],
                    s[StateMemberVy],
                    s[StateMemberVz],
                    s[StateMemberVroll],
                    s[StateMemberVpitch],
                    s[StateMemberVyaw],
                    s[StateMemberAx],
                    s[StateMemberAy],
                    s[StateMemberAz],
                ]
            )

    print(f"Wrote trajectory to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
