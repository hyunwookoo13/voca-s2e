from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute Habitat/PixelNav benchmark metrics from tiny E2E or closed-loop summaries.",
    )
    parser.add_argument("--trial-summary", action="append", default=[], help="Path to real_vlm_tiny_e2e.json.")
    parser.add_argument("--batch-summary", action="append", default=[], help="Path to batch summary JSON.")
    parser.add_argument("--audit-result", action="append", default=[], help="Step B audit result JSON. Can be repeated.")
    parser.add_argument("--audit-glob", action="append", default=[], help="Glob for audit result JSON files. Can be repeated.")
    parser.add_argument("--out", required=True, help="Output directory for benchmark artifacts.")
    parser.add_argument("--force-front-view-waypoint", action="store_true")
    parser.add_argument("--success-distance-m", type=float, default=0.5)
    parser.add_argument("--min-progress-m", type=float, default=0.1)
    args = parser.parse_args(argv)

    summary_paths = _resolve_summary_paths(args.trial_summary, args.batch_summary)
    audit_paths = _resolve_audit_paths(args.audit_result, args.audit_glob)
    if audit_paths:
        result = run_habitat_pixelnav_benchmark_from_audits(
            audit_paths,
            output_dir=args.out,
            force_front_view_waypoint=bool(args.force_front_view_waypoint),
            success_distance_m=args.success_distance_m,
            min_progress_m=args.min_progress_m,
        )
    else:
        if not summary_paths:
            parser.error("provide at least one --trial-summary, --batch-summary, --audit-result, or --audit-glob")
        result = run_habitat_pixelnav_benchmark_from_summaries(
            summary_paths,
            output_dir=args.out,
            success_distance_m=args.success_distance_m,
            min_progress_m=args.min_progress_m,
        )
    print(json.dumps({"status": result["status"], "summary_json": result["summary_json"]}, ensure_ascii=False))
    return 0


def run_habitat_pixelnav_benchmark_from_audits(
    audit_paths: Iterable[str | Path],
    *,
    output_dir: str | Path,
    force_front_view_waypoint: bool,
    success_distance_m: float = 0.5,
    min_progress_m: float = 0.1,
    run_batch: Any | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = output_dir / "real_vlm_batch"
    run_batch = run_batch or _run_real_vlm_batch
    audit_list = [Path(path) for path in audit_paths]
    batch_summary = run_batch(audit_list, batch_dir, bool(force_front_view_waypoint))
    summary_paths = _resolve_summary_paths([], [str(batch_summary["summary_json"])])
    result = run_habitat_pixelnav_benchmark_from_summaries(
        summary_paths,
        output_dir=output_dir,
        success_distance_m=success_distance_m,
        min_progress_m=min_progress_m,
    )
    result["source"] = {
        "mode": "audit_real_vlm_batch",
        "audit_results": [str(path) for path in audit_list],
        "batch_summary_json": str(batch_summary["summary_json"]),
        "force_front_view_waypoint": bool(force_front_view_waypoint),
    }
    summary_path = Path(result["summary_json"])
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["source"] = result["source"]
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result.update(payload)
    Path(result["docmost_markdown"]).write_text(_render_docmost(payload), encoding="utf-8")
    return result


def run_habitat_pixelnav_benchmark_from_summaries(
    summary_paths: Iterable[str | Path],
    *,
    output_dir: str | Path,
    success_distance_m: float = 0.5,
    min_progress_m: float = 0.1,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, path in enumerate(summary_paths):
        summary_path = Path(path)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        episode_id = _episode_id_from_path(summary_path, index)
        record = compute_episode_benchmark_record(
            payload,
            episode_id=episode_id,
            success_distance_m=success_distance_m,
            min_progress_m=min_progress_m,
        )
        record["summary_json"] = str(summary_path)
        records.append(record)

    aggregate = _aggregate_benchmark_records(records)
    summary = {
        "status": "ready" if records else "empty",
        "benchmark_type": "pointnav_style_from_habitat_pixelnav_summaries",
        "metric_mode": {
            "success_distance_m": float(success_distance_m),
            "min_progress_m": float(min_progress_m),
            "shortest_path_source": "euclidean_proxy_until_navmesh_episode_loader_is_enabled",
            "spl_definition": "success * shortest_path_m / max(shortest_path_m, path_length_m)",
        },
        "total_episodes": len(records),
        "metrics": aggregate,
        "episodes": records,
    }
    summary_path = output_dir / "habitat_pixelnav_benchmark_summary.json"
    markdown_path = output_dir / "docmost_habitat_pixelnav_benchmark.md"
    summary["summary_json"] = str(summary_path)
    summary["docmost_markdown"] = str(markdown_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_docmost(summary), encoding="utf-8")
    return summary


def _run_real_vlm_batch(
    audit_paths: list[Path],
    output_dir: Path,
    force_front_view_waypoint: bool,
) -> dict[str, Any]:
    from goal_adapter.habitat_pixelnav_real_vlm_batch import run_real_vlm_tiny_e2e_batch

    return run_real_vlm_tiny_e2e_batch(
        audit_paths,
        output_dir=output_dir,
        force_front_view_waypoint=force_front_view_waypoint,
    )


def compute_episode_benchmark_record(
    trial_summary: dict[str, Any],
    *,
    episode_id: str,
    success_distance_m: float,
    min_progress_m: float,
) -> dict[str, Any]:
    config = trial_summary.get("config") if isinstance(trial_summary.get("config"), dict) else {}
    outcomes = _extract_action_outcomes(trial_summary)
    outcome = outcomes[-1] if outcomes else {}
    memory = trial_summary.get("memory") if isinstance(trial_summary.get("memory"), dict) else {}
    memory_update = trial_summary.get("memory_update") if isinstance(trial_summary.get("memory_update"), dict) else {}
    start_xyz = _as_xyz(config.get("start_position_xyz"))
    goal_xy = _as_xy(config.get("goal_map_xy"))
    final_xyz = _extract_final_position_xyz(outcome) or start_xyz
    start_xy = _xz_to_xy(start_xyz)
    final_xy = _xz_to_xy(final_xyz)
    initial_distance = _distance2(start_xy, goal_xy)
    final_distance = _distance2(final_xy, goal_xy)
    path_length = _extract_path_length(outcomes, start_xyz, final_xyz)
    shortest_path = max(0.0, initial_distance)
    goal_progress = initial_distance - final_distance

    go_outcomes = [item for item in outcomes if str(item.get("action") or "go") == "go" or item.get("moved_distance_m") is not None]
    local_execution_success = bool(go_outcomes) and all(bool(item.get("success")) for item in go_outcomes)
    pointnav_success = bool(final_distance <= float(success_distance_m))
    progress_success = bool(goal_progress >= float(min_progress_m))
    spl = 0.0
    if pointnav_success and shortest_path > 0.0:
        spl = shortest_path / max(shortest_path, path_length)

    temporal_edges = list(memory.get("temporal_edges") or [])
    edge_successes = [
        str(edge.get("status") or (edge.get("traversal") or {}).get("status")) == "success"
        for edge in temporal_edges
        if isinstance(edge, dict)
    ]
    deadlock_state = memory.get("deadlock_state") if isinstance(memory.get("deadlock_state"), dict) else {}
    return {
        "episode_id": episode_id,
        "scene_path": config.get("scene_path"),
        "status": trial_summary.get("status"),
        "local_execution_success": local_execution_success,
        "pointnav_success": pointnav_success,
        "progress_success": progress_success,
        "collision": any(bool(item.get("collision")) for item in outcomes),
        "no_progress": any(bool(item.get("no_progress")) for item in outcomes),
        "initial_distance_to_goal_m": round(initial_distance, 6),
        "final_distance_to_goal_m": round(final_distance, 6),
        "goal_progress_m": round(goal_progress, 6),
        "path_length_m": round(path_length, 6),
        "shortest_path_m": round(shortest_path, 6),
        "spl": round(spl, 6),
        "progress_efficiency": round(max(0.0, goal_progress) / max(shortest_path, path_length, 1e-9), 6),
        "start_xy": [round(start_xy[0], 6), round(start_xy[1], 6)],
        "final_xy": [round(final_xy[0], 6), round(final_xy[1], 6)],
        "goal_xy": [round(goal_xy[0], 6), round(goal_xy[1], 6)],
        "memory": {
            "schema_version": memory.get("schema_version"),
            "updated": bool(memory_update.get("updated") or memory.get("num_nodes")),
            "num_nodes": _int_or_zero(memory.get("num_nodes")),
            "num_edges": _int_or_zero(memory.get("num_edges")),
            "temporal_edge_count": len(temporal_edges),
            "temporal_edge_success_rate": _rate(edge_successes),
            "deadlock_status": deadlock_state.get("status"),
            "negative_node_count": len(memory.get("negative_nodes") or []),
        },
    }


def _extract_action_outcomes(summary: dict[str, Any]) -> list[dict[str, Any]]:
    execution = summary.get("execution") if isinstance(summary.get("execution"), dict) else {}
    outcome = execution.get("action_outcome")
    if isinstance(outcome, dict):
        return [outcome]
    outcomes: list[dict[str, Any]] = []
    steps = summary.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict) and isinstance(step.get("outcome"), dict):
                outcomes.append(step["outcome"])
    return outcomes


def _extract_final_position_xyz(outcome: dict[str, Any]) -> list[float] | None:
    raw = outcome.get("raw") if isinstance(outcome.get("raw"), dict) else {}
    rollout_result_json = raw.get("rollout_result_json")
    if rollout_result_json and Path(str(rollout_result_json)).exists():
        try:
            payload = json.loads(Path(str(rollout_result_json)).read_text(encoding="utf-8"))
            final = _as_xyz_or_none(payload.get("final_position_xyz"))
            if final is not None:
                return final
        except (OSError, json.JSONDecodeError):
            pass
    rollout = raw.get("rollout") if isinstance(raw.get("rollout"), dict) else {}
    steps = rollout.get("steps")
    if isinstance(steps, list):
        for step in reversed(steps):
            if isinstance(step, dict):
                final = _as_xyz_or_none(step.get("position_after_xyz"))
                if final is not None:
                    return final
    return None


def _extract_path_length(outcomes: list[dict[str, Any]], start_xyz: Sequence[float], final_xyz: Sequence[float]) -> float:
    total_length = 0.0
    found = False
    for outcome in outcomes:
        moved = outcome.get("moved_distance_m")
        if isinstance(moved, (int, float)):
            total_length += max(0.0, float(moved))
            found = True
            continue
        raw = outcome.get("raw") if isinstance(outcome.get("raw"), dict) else {}
        rollout = raw.get("rollout") if isinstance(raw.get("rollout"), dict) else {}
        total = rollout.get("total_moved_distance_m")
        if isinstance(total, (int, float)):
            total_length += max(0.0, float(total))
            found = True
    if found:
        return total_length
    return math.dist(_xz_to_xy(start_xyz), _xz_to_xy(final_xyz))


def _aggregate_benchmark_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {
            "local_execution_sr": 0.0,
            "pointnav_sr": 0.0,
            "progress_sr": 0.0,
            "spl": 0.0,
        }
    return {
        "local_execution_sr": _rate(record["local_execution_success"] for record in records),
        "pointnav_sr": _rate(record["pointnav_success"] for record in records),
        "progress_sr": _rate(record["progress_success"] for record in records),
        "spl": round(mean(float(record["spl"]) for record in records), 6),
        "mean_progress_efficiency": round(mean(float(record["progress_efficiency"]) for record in records), 6),
        "mean_goal_progress_m": round(mean(float(record["goal_progress_m"]) for record in records), 6),
        "mean_path_length_m": round(mean(float(record["path_length_m"]) for record in records), 6),
        "collision_rate": _rate(record["collision"] for record in records),
        "no_progress_rate": _rate(record["no_progress"] for record in records),
        "memory_update_rate": _rate(record["memory"]["updated"] for record in records),
        "mean_memory_nodes": round(mean(float(record["memory"]["num_nodes"]) for record in records), 6),
        "mean_memory_edges": round(mean(float(record["memory"]["num_edges"]) for record in records), 6),
        "mean_temporal_edge_success_rate": round(
            mean(float(record["memory"]["temporal_edge_success_rate"]) for record in records),
            6,
        ),
        "negative_memory_episode_rate": _rate(record["memory"]["negative_node_count"] > 0 for record in records),
    }


def _render_docmost(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    lines = [
        "# Habitat PixelNav Benchmark",
        "",
        "## 1. Summary",
        "",
        f"- benchmark_type: `{summary['benchmark_type']}`",
        f"- total_episodes: `{summary['total_episodes']}`",
        f"- success_distance_m: `{summary['metric_mode']['success_distance_m']}`",
        f"- shortest_path_source: `{summary['metric_mode']['shortest_path_source']}`",
        "",
        "## 2. Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in [
        "local_execution_sr",
        "pointnav_sr",
        "progress_sr",
        "spl",
        "mean_progress_efficiency",
        "mean_goal_progress_m",
        "mean_path_length_m",
        "collision_rate",
        "no_progress_rate",
        "memory_update_rate",
        "mean_memory_nodes",
        "mean_memory_edges",
        "mean_temporal_edge_success_rate",
        "negative_memory_episode_rate",
    ]:
        lines.append(f"| {key} | {metrics.get(key)} |")
    lines.extend(
        [
            "",
            "## 3. Episodes",
            "",
            "| episode | local_sr | pointnav_sr | progress_sr | SPL | progress_m | path_m | nodes | edges | summary |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for record in summary["episodes"]:
        lines.append(
            "| {episode} | `{local}` | `{pointnav}` | `{progress}` | {spl} | {progress_m} | {path_m} | {nodes} | {edges} | `{path}` |".format(
                episode=record.get("episode_id"),
                local=record.get("local_execution_success"),
                pointnav=record.get("pointnav_success"),
                progress=record.get("progress_success"),
                spl=record.get("spl"),
                progress_m=record.get("goal_progress_m"),
                path_m=record.get("path_length_m"),
                nodes=record.get("memory", {}).get("num_nodes"),
                edges=record.get("memory", {}).get("num_edges"),
                path=record.get("summary_json"),
            )
        )
    lines.extend(
        [
            "",
            "## 4. Interpretation",
            "",
            "- `local_execution_sr` measures whether the VLM-selected PixelNav waypoint executed successfully.",
            "- `pointnav_sr` is strict goal-threshold success. Single-step local rollouts may progress without reaching the final goal.",
            "- `spl` is standard success-gated SPL using Euclidean proxy shortest path until official Habitat episode geodesics are wired.",
            "- `progress_sr` and `mean_progress_efficiency` are the key early integration metrics before full multi-step PointNav episodes.",
            "",
        ]
    )
    return "\n".join(lines)


def _resolve_summary_paths(trial_summaries: list[str], batch_summaries: list[str]) -> list[Path]:
    paths = [Path(path) for path in trial_summaries]
    for batch_summary in batch_summaries:
        payload = json.loads(Path(batch_summary).read_text(encoding="utf-8"))
        for record in payload.get("records") or []:
            if isinstance(record, dict) and record.get("summary_json"):
                paths.append(Path(record["summary_json"]))
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _resolve_audit_paths(audit_results: list[str], audit_globs: list[str]) -> list[Path]:
    paths = [Path(path) for path in audit_results]
    for pattern in audit_globs:
        paths.extend(sorted(Path().glob(pattern)))
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _episode_id_from_path(path: Path, index: int) -> str:
    parent = path.parent.name
    if parent:
        return parent
    return f"episode_{index:04d}"


def _as_xyz(value: Any) -> list[float]:
    xyz = _as_xyz_or_none(value)
    if xyz is None:
        return [0.0, 0.0, 0.0]
    return xyz


def _as_xyz_or_none(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    return [float(value[0]), float(value[1]), float(value[2])]


def _as_xy(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return [0.0, 0.0]
    return [float(value[0]), float(value[1])]


def _xz_to_xy(value: Sequence[float]) -> list[float]:
    return [float(value[0]), float(value[2])]


def _distance2(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)


def _rate(values: Iterable[Any]) -> float:
    items = [bool(value) for value in values]
    if not items:
        return 0.0
    return round(sum(1 for value in items if value) / len(items), 6)


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
