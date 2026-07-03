from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from goal_adapter import GoalAdapter
from goal_adapter.evaluation import evaluate_cases
from goal_adapter.fixtures import FIRST_STAGE_REFINEMENT_CASES


DEFAULT_OUTPUT_PATH = Path("reports") / "goal_adapter_eval.json"


def run_evaluation(output_path: str | Path = DEFAULT_OUTPUT_PATH) -> dict:
    summary = evaluate_cases(GoalAdapter(), FIRST_STAGE_REFINEMENT_CASES)
    report = summary.to_json()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the first-stage GoalAdapter fixture evaluation.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path where the JSON evaluation report will be written.",
    )
    args = parser.parse_args(argv)

    report = run_evaluation(args.output)
    metrics = report["metrics"]
    print(f"Wrote evaluation report to {args.output}")
    print(f"total_cases={report['total_cases']}")
    print(f"waypoint_success_rate={metrics['waypoint_success_rate']:.3f}")
    print(f"failure_repeat_rate={metrics['failure_repeat_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
