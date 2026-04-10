"""Copy a text file while skipping the first N rows.

Example:
  python sample_data.py --input in.txt --output out.txt --skip 10
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile


def _copy_skipping_rows(input_path: Path, output_path: Path, skip_rows: int) -> None:
	if skip_rows < 0:
		raise ValueError("--skip must be >= 0")

	input_path = input_path.resolve()
	output_path = output_path.resolve()

	with input_path.open("r", encoding="utf-8", errors="replace", newline="") as f_in, output_path.open(
		"w", encoding="utf-8", newline=""
	) as f_out:
		for i, line in enumerate(f_in):
			if i % skip_rows == 0:
				f_out.write(line)


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Copy INPUT to OUTPUT, skipping the first N rows.",
	)
	parser.add_argument("--input", required=True, help="Input text file path")
	parser.add_argument("--output", required=True, help="Output text file path")
	parser.add_argument("--skip", type=int, default=0, help="Number of rows to skip from the top")
	args = parser.parse_args()

	_copy_skipping_rows(Path(args.input), Path(args.output), int(args.skip))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

