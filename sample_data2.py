"""Generate relative pose increments from an absolute pose CSV.

Input format (like ./example/data/absolute_sensor.csv):
  timestamp_ns, x, y, z, qw, qx, qy, qz, ... (extra columns ignored)

Output (./example/data/relative.csv by default):
  timestamp_ns, dx, dy, dz, dqw, dqx, dqy, dqz

Where the delta pose is computed w.r.t. the previous timestamp:
  dpos = pos_i - pos_{i-1}
  dq   = q_{i-1}^-1 ⊗ q_i

The first output row uses zeros and identity quaternion.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple


def _parse_timestamp_ns(value: str) -> int:
	s = value.strip()
	try:
		return int(s)
	except ValueError:
		pass
	try:
		return int(Decimal(s))
	except (InvalidOperation, ValueError):
		pass
	return int(float(s))


@dataclass(frozen=True)
class Pose:
	t_ns: int
	x: float
	y: float
	z: float
	qw: float
	qx: float
	qy: float
	qz: float


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


def _load_absolute_pose_csv(path: Path) -> Iterator[Pose]:
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
			yield Pose(t_ns=t_ns, x=x, y=y, z=z, qw=qw, qx=qx, qy=qy, qz=qz)


def _iter_relative_deltas(poses: Iterable[Pose]) -> Iterable[Tuple[int, float, float, float, float, float, float, float]]:
	prev: Optional[Pose] = None
	for p in poses:
		if prev is None:
			yield (p.t_ns, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
			prev = p
			continue

		dx = p.x - prev.x
		dy = p.y - prev.y
		dz = p.z - prev.z

		q_prev = (prev.qw, prev.qx, prev.qy, prev.qz)
		q_curr = (p.qw, p.qx, p.qy, p.qz)
		q_rel = _quat_mul(_quat_conj(*q_prev), q_curr)
		q_rel = _quat_normalize(*q_rel)

		yield (p.t_ns, dx, dy, dz, q_rel[0], q_rel[1], q_rel[2], q_rel[3])
		prev = p


def main() -> int:
	parser = argparse.ArgumentParser(description="Compute relative pose deltas from absolute_sensor.csv")
	parser.add_argument(
		"--input",
		default="example/data/absolute_sensor.csv",
		help="Input absolute pose CSV (default: example/data/absolute_sensor.csv)",
	)
	parser.add_argument(
		"--output",
		default="example/data/relative.csv",
		help="Output relative CSV (default: example/data/relative.csv)",
	)
	args = parser.parse_args()

	input_path = Path(args.input)
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	poses = _load_absolute_pose_csv(input_path)
	deltas = _iter_relative_deltas(poses)

	with output_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["timestamp_ns", "dx", "dy", "dz", "dqw", "dqx", "dqy", "dqz"])
		for row in deltas:
			writer.writerow(row)

	print(f"Wrote: {output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

