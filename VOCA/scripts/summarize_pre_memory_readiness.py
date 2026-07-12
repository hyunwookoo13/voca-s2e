#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pre_memory_readiness import write_pre_memory_readiness


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize VOCA Qwen pre-memory benchmark readiness.")
    parser.add_argument("output_dir", help="Benchmark output directory containing trajectory_* folders.")
    parser.add_argument("--metrics-csv", default="", help="Optional benchmark metrics CSV path.")
    args = parser.parse_args()

    paths = write_pre_memory_readiness(
        args.output_dir,
        args.metrics_csv if args.metrics_csv else None,
    )
    print("wrote {}".format(paths["json"]))
    print("wrote {}".format(paths["csv"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
