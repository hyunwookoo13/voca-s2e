#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from navigation_ablation import write_summary


def _run_spec(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("run must use non-empty LABEL=PATH")
    return label.strip(), path.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare VOCA navigation, waypoint-gate, and memory metrics."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=_run_spec,
        help="Repeatable experiment specification: LABEL=OUTPUT_DIR_OR_CSV",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    paths = write_summary(args.run, args.output_dir)
    print("wrote {}".format(paths["json"]))
    print("wrote {}".format(paths["csv"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
