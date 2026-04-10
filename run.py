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
- absolute_sensor.csv provides absolute position; only the first 4 columns are used:
    timestamp_ns, x, y, z. All remaining columns are ignored.
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
from typing import Dict, Iterable, List, Tuple


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


def _load_absolute_position_csv(path: Path) -> Iterable[Tuple[int, float, float, float]]:
    """Yield absolute position rows: (timestamp_ns, x, y, z).

    File is expected to have no header and potentially many columns; only the
    first 4 are used.
    """

    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 4:
                continue
            try:
                t_ns = _parse_timestamp_ns(row[0])
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
            except ValueError:
                continue
            yield (t_ns, x, y, z)


def _make_measurement_covariance(rate_hz: float, gyro_nd: float, accel_nd: float) -> List[List[float]]:
    # Discrete-time stddev approximation: sigma = nd * sqrt(rate_hz)
    gyro_var = (gyro_nd * math.sqrt(rate_hz)) ** 2
    accel_var = (accel_nd * math.sqrt(rate_hz)) ** 2

    # Order: [Vroll, Vpitch, Vyaw, Ax, Ay, Az]
    cov = [[0.0 for _ in range(6)] for _ in range(6)]
    for i in range(3):
        cov[i][i] = gyro_var
    for i in range(3, 6):
        cov[i][i] = accel_var
    return cov


def main() -> int:
    parser = argparse.ArgumentParser(description="Run python_ekf.py on IMU data.csv")
    parser.add_argument("--data", default="src/data.csv", help="Path to IMU CSV (default: src/data.csv)")
    parser.add_argument("--sensor", default="sensor.yaml", help="Path to sensor.yaml (default: sensor.yaml)")
    parser.add_argument(
        "--absolute",
        default="src/absolute_sensor.csv",
        help="Path to absolute_sensor.csv (default: src/absolute_sensor.csv)",
    )
    parser.add_argument("--output_dir", default="src/output", help="Output directory (default: src/output)")
    parser.add_argument("--output_name", default="trajectory.csv", help="Output filename (default: trajectory.csv)")
    parser.add_argument("--mahalanobis", type=float, default=float("inf"), help="Mahalanobis gate (sigmas)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent

    data_path = (repo_root / args.data).resolve() if not os.path.isabs(args.data) else Path(args.data)
    sensor_path = (repo_root / args.sensor).resolve() if not os.path.isabs(args.sensor) else Path(args.sensor)
    abs_path = (repo_root / args.absolute).resolve() if not os.path.isabs(args.absolute) else Path(args.absolute)

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

    if not abs_path.exists():
        fallback = repo_root / "example" / "data" / "absolute_sensor.csv"
        if fallback.exists():
            abs_path = fallback
        else:
            raise FileNotFoundError(
                f"Could not find absolute sensor CSV at {abs_path} (and no fallback at {fallback})"
            )

    noise = _parse_sensor_yaml_for_noise(sensor_path)
    rate_hz = noise["rate_hz"]
    gyro_nd = noise["gyroscope_noise_density"]
    accel_nd = noise["accelerometer_noise_density"]

    cov6 = _make_measurement_covariance(rate_hz, gyro_nd, accel_nd)
    # Absolute position sensor covariance: sigma = 0.05 m
    pos_sigma = 0.05
    pos_var = pos_sigma * pos_sigma
    cov3 = [[pos_var, 0.0, 0.0], [0.0, pos_var, 0.0], [0.0, 0.0, pos_var]]

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

        # Stream and time-merge IMU + absolute position measurements
        imu_iter = iter(_load_imu_csv(data_path))
        abs_iter = iter(_load_absolute_position_csv(abs_path))

        imu_next = next(imu_iter, None)
        abs_next = next(abs_iter, None)

        if imu_next is None and abs_next is None:
            raise ValueError("No measurements found in either IMU or absolute sensor inputs")

        first_candidates = []
        if imu_next is not None:
            first_candidates.append(imu_next[0])
        if abs_next is not None:
            first_candidates.append(abs_next[0])
        first_t_ns = min(first_candidates)

        while imu_next is not None or abs_next is not None:
            next_ts: int
            if imu_next is None:
                next_ts = abs_next[0]
            elif abs_next is None:
                next_ts = imu_next[0]
            else:
                next_ts = imu_next[0] if imu_next[0] <= abs_next[0] else abs_next[0]

            t_sec = (next_ts - first_t_ns) * 1e-9

            # Process all absolute measurements at this timestamp
            while abs_next is not None and abs_next[0] == next_ts:
                _, x, y, z = abs_next
                abs_next = next(abs_iter, None)

                meas = Measurement.from_subset(
                    time=t_sec,
                    update_indices=[StateMemberX, StateMemberY, StateMemberZ],
                    values=[x, y, z],
                    covariance=cov3,
                    mahalanobis_thresh=args.mahalanobis,
                )
                ekf.process_measurement(meas)

            # Process all IMU measurements at this timestamp
            while imu_next is not None and imu_next[0] == next_ts:
                _, wx, wy, wz, ax, ay, az = imu_next
                imu_next = next(imu_iter, None)

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
