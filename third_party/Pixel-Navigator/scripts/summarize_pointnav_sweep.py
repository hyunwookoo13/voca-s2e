from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def collect_summaries(root: Path) -> list[dict[str, Any]]:
    rows = []
    for summary_path in sorted(root.glob("*/voca_memory_pointnav_benchmark_summary.json")):
        summary = _load_summary(summary_path)
        aggregate = summary.get("aggregate", {})
        run_dir = summary_path.parent
        config_path = run_dir / "run_config.json"
        config = _load_summary(config_path) if config_path.exists() else {}
        rows.append(
            {
                "run": run_dir.name,
                "episodes": aggregate.get("episodes", 0),
                "success": float(aggregate.get("success", 0.0) or 0.0),
                "spl": float(aggregate.get("spl", 0.0) or 0.0),
                "soft_spl": float(aggregate.get("soft_spl", 0.0) or 0.0),
                "mean_distance_to_goal": float(aggregate.get("mean_distance_to_goal", 0.0) or 0.0),
                "mean_distance_to_goal_delta": float(aggregate.get("mean_distance_to_goal_delta", 0.0) or 0.0),
                "mean_env_steps": float(aggregate.get("mean_env_steps", 0.0) or 0.0),
                "mean_agent_steps": float(aggregate.get("mean_agent_steps", 0.0) or 0.0),
                "pixelnav_policy": config.get("pixelnav_policy", ""),
                "vlm": config.get("vlm", ""),
                "heuristic_y_ratio": config.get("heuristic_y_ratio", ""),
                "bearing_rotate_threshold_deg": config.get("bearing_rotate_threshold_deg", ""),
                "max_pixelnav_steps": config.get("max_pixelnav_steps", ""),
                "summary_json": str(summary_path),
            }
        )
    rows.sort(
        key=lambda row: (
            row["success"],
            row["spl"],
            row["mean_distance_to_goal_delta"],
            -row["mean_distance_to_goal"],
        ),
        reverse=True,
    )
    return rows


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    headers = [
        "rank",
        "run",
        "episodes",
        "success",
        "spl",
        "soft_spl",
        "dist",
        "delta",
        "env_steps",
        "policy",
        "vlm",
        "y",
        "rot_thr",
        "pix_steps",
    ]
    lines = ["# PointNav Sweep Summary", "", "|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for index, row in enumerate(rows, start=1):
        values = [
            index,
            row["run"],
            row["episodes"],
            row["success"],
            row["spl"],
            row["soft_spl"],
            row["mean_distance_to_goal"],
            row["mean_distance_to_goal_delta"],
            row["mean_env_steps"],
            row["pixelnav_policy"],
            row["vlm"],
            row["heuristic_y_ratio"],
            row["bearing_rotate_threshold_deg"],
            row["max_pixelnav_steps"],
        ]
        lines.append("|" + "|".join(_fmt(value) for value in values) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize VOCA PointNav sweep outputs.")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    rows = collect_summaries(args.root)
    out_json = args.root / "pointnav_sweep_summary.json"
    out_md = args.root / "pointnav_sweep_summary.md"
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(rows, out_md)
    print(json.dumps({"runs": len(rows), "summary_json": str(out_json), "summary_md": str(out_md)}, ensure_ascii=False))
    if rows:
        print(json.dumps({"best": rows[0]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
