from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

from goal_adapter.habitat_pixelnav_closed_loop import load_memory_smoke_config_from_audit
from goal_adapter.habitat_pixelnav_real_vlm_tiny_e2e import run_habitat_pixelnav_real_vlm_tiny_e2e_from_env


RunOneFn = Callable[[Path, Path, bool], dict[str, Any]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a small batch of real VLM Habitat/PixelNav tiny E2E trials.")
    parser.add_argument("--audit-result", action="append", default=[], help="Step B audit result JSON. Can be repeated.")
    parser.add_argument("--audit-glob", action="append", default=[], help="Glob for audit result JSON files. Can be repeated.")
    parser.add_argument("--out", required=True, help="Output directory for batch summary and per-trial artifacts.")
    parser.add_argument("--force-front-view-waypoint", action="store_true")
    args = parser.parse_args(argv)

    audit_paths = _resolve_audit_paths(args.audit_result, args.audit_glob)
    if not audit_paths:
        parser.error("provide at least one --audit-result or --audit-glob")

    summary = run_real_vlm_tiny_e2e_batch(
        audit_paths,
        output_dir=args.out,
        force_front_view_waypoint=bool(args.force_front_view_waypoint),
    )
    print(json.dumps({"status": summary["status"], "summary_json": summary["summary_json"]}, ensure_ascii=False))
    return 0


def run_real_vlm_tiny_e2e_batch(
    audit_results: Iterable[str | Path],
    *,
    output_dir: str | Path,
    force_front_view_waypoint: bool,
    run_one: RunOneFn | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_one = run_one or _run_one_from_env
    audit_paths = [Path(path) for path in audit_results]

    records: list[dict[str, Any]] = []
    for index, audit_path in enumerate(audit_paths):
        trial_dir = output_dir / f"trial_{index:03d}_{audit_path.stem}"
        try:
            trial_summary = run_one(audit_path, trial_dir, bool(force_front_view_waypoint))
            record = _record_from_trial(index, audit_path, trial_dir, trial_summary)
        except Exception as exc:  # pragma: no cover - exercised by real runs
            trial_dir.mkdir(parents=True, exist_ok=True)
            error_payload = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            error_path = trial_dir / "trial_error.json"
            error_path.write_text(json.dumps(error_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            record = {
                "index": index,
                "audit_result": str(audit_path),
                "trial_dir": str(trial_dir),
                "status": "error",
                "tiny_e2e": {"passed": False, "failure_stage": "exception"},
                "error_json": str(error_path),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        records.append(record)

    summary = _aggregate_records(records)
    summary["records"] = records
    summary["output_dir"] = str(output_dir)
    summary_path = output_dir / "real_vlm_tiny_e2e_batch_summary.json"
    markdown_path = output_dir / "docmost_real_vlm_tiny_e2e_batch.md"
    summary["summary_json"] = str(summary_path)
    summary["docmost_markdown"] = str(markdown_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_docmost(summary), encoding="utf-8")
    return summary


def _run_one_from_env(audit_path: Path, trial_dir: Path, force_front_view_waypoint: bool) -> dict[str, Any]:
    config = load_memory_smoke_config_from_audit(
        audit_path,
        output_dir=trial_dir,
        max_agent_steps=1,
        force_front_view_waypoint=force_front_view_waypoint,
    )
    return run_habitat_pixelnav_real_vlm_tiny_e2e_from_env(config)


def _record_from_trial(index: int, audit_path: Path, trial_dir: Path, trial: dict[str, Any]) -> dict[str, Any]:
    tiny = trial.get("tiny_e2e") or {}
    execution = trial.get("execution") or {}
    outcome = execution.get("action_outcome") if isinstance(execution.get("action_outcome"), dict) else {}
    memory = trial.get("memory") if isinstance(trial.get("memory"), dict) else {}
    return {
        "index": index,
        "audit_result": str(audit_path),
        "trial_dir": str(trial_dir),
        "status": str(trial.get("status") or "unknown"),
        "summary_json": trial.get("summary_json"),
        "tiny_e2e": {
            "passed": bool(tiny.get("passed")),
            "endpoint_smoke_passed": bool(tiny.get("endpoint_smoke_passed")),
            "gated_execution_executed": bool(tiny.get("gated_execution_executed")),
            "memory_updated": bool(tiny.get("memory_updated")),
            "failure_stage": tiny.get("failure_stage"),
        },
        "execution": {
            "success": outcome.get("success"),
            "moved_distance_m": outcome.get("moved_distance_m"),
            "collision": outcome.get("collision"),
            "no_progress": outcome.get("no_progress"),
            "message": outcome.get("message"),
        },
        "memory": {
            "num_nodes": memory.get("num_nodes"),
            "num_edges": memory.get("num_edges"),
            "current_node_id": memory.get("current_node_id"),
        },
    }


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(record.get("status") or "unknown") for record in records)
    failure_stage_counts = Counter(
        str(record.get("tiny_e2e", {}).get("failure_stage"))
        for record in records
        if record.get("tiny_e2e", {}).get("failure_stage") is not None
    )
    moved_distances = [
        float(record.get("execution", {}).get("moved_distance_m"))
        for record in records
        if record.get("execution", {}).get("moved_distance_m") is not None
    ]
    total = len(records)
    passed = sum(1 for record in records if record.get("tiny_e2e", {}).get("passed"))
    return {
        "status": "passed" if total > 0 and passed == total else ("empty" if total == 0 else "mixed"),
        "total_trials": total,
        "passed_trials": passed,
        "pass_rate": (passed / total) if total else 0.0,
        "status_counts": dict(status_counts),
        "endpoint_smoke_passed": sum(1 for record in records if record.get("tiny_e2e", {}).get("endpoint_smoke_passed")),
        "gated_execution_executed": sum(1 for record in records if record.get("tiny_e2e", {}).get("gated_execution_executed")),
        "memory_updated": sum(1 for record in records if record.get("tiny_e2e", {}).get("memory_updated")),
        "failure_stage_counts": dict(failure_stage_counts),
        "mean_moved_distance_m": round(sum(moved_distances) / len(moved_distances), 6) if moved_distances else None,
        "collision_count": sum(1 for record in records if record.get("execution", {}).get("collision")),
        "no_progress_count": sum(1 for record in records if record.get("execution", {}).get("no_progress")),
    }


def _render_docmost(summary: dict[str, Any]) -> str:
    lines = [
        "# Real VLM Tiny E2E Batch",
        "",
        "## 1. Summary",
        "",
        f"- status: `{summary['status']}`",
        f"- total_trials: `{summary['total_trials']}`",
        f"- passed_trials: `{summary['passed_trials']}`",
        f"- pass_rate: `{summary['pass_rate']}`",
        f"- mean_moved_distance_m: `{summary['mean_moved_distance_m']}`",
        f"- collision_count: `{summary['collision_count']}`",
        f"- no_progress_count: `{summary['no_progress_count']}`",
        "",
        "## 2. Stage Counts",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| endpoint_smoke_passed | {summary['endpoint_smoke_passed']} |",
        f"| gated_execution_executed | {summary['gated_execution_executed']} |",
        f"| memory_updated | {summary['memory_updated']} |",
        "",
        "## 3. Failure Stages",
        "",
        "| stage | count |",
        "|---|---:|",
    ]
    if summary["failure_stage_counts"]:
        for stage, count in sorted(summary["failure_stage_counts"].items()):
            lines.append(f"| {stage} | {count} |")
    else:
        lines.append("| none | 0 |")
    lines.extend(
        [
            "",
            "## 4. Trials",
            "",
            "| index | status | passed | moved_m | collision | no_progress | summary |",
            "|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for record in summary["records"]:
        execution = record.get("execution") or {}
        lines.append(
            "| {index} | `{status}` | `{passed}` | `{moved}` | `{collision}` | `{no_progress}` | `{summary_path}` |".format(
                index=record.get("index"),
                status=record.get("status"),
                passed=record.get("tiny_e2e", {}).get("passed"),
                moved=execution.get("moved_distance_m"),
                collision=execution.get("collision"),
                no_progress=execution.get("no_progress"),
                summary_path=record.get("summary_json") or record.get("error_json"),
            )
        )
    lines.extend(
        [
            "",
            "## 5. Recommended Runtime Env",
            "",
            "```bash",
            'export QWEN_BASE_URL="http://server-01.cgv:8000/v1"',
            'export QWEN_API_KEY="EMPTY"',
            'export PIXELNAV_DEVICE="cpu"',
            'export QWEN_MODEL="qwen3-vl-32b-thinking"',
            'export QWEN_TIMEOUT_S="600"',
            'export QWEN_MAX_TOKENS="8192"',
            'export QWEN_MAX_JSON_RETRIES="1"',
            'export QWEN_EXTRA_PAYLOAD_JSON=\'{"response_format":{"type":"json_object"}}\'',
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _resolve_audit_paths(audit_results: list[str], audit_globs: list[str]) -> list[Path]:
    paths = [Path(path) for path in audit_results]
    for pattern in audit_globs:
        paths.extend(sorted(Path().glob(pattern)))
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


if __name__ == "__main__":
    raise SystemExit(main())
