import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from qwen_action_runner import (
    _memory_verdict_metrics,
    classify_go_failure,
    pitch_offset_from_actions,
)


PathLike = Union[str, Path]


def _read_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], bool]:
    if not path.is_file():
        return [], False
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows, True


def _read_json(path: Path) -> Tuple[Any, bool]:
    if not path.is_file():
        return None, False
    with path.open("r", encoding="utf-8") as f:
        return json.load(f), True


def _read_metrics(path: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    rows: Dict[int, Dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                episode = int(row.get("episode", len(rows)))
            except Exception:
                episode = len(rows)
            rows[episode] = dict(row)
    return rows


def _trajectory_episode(path: Path) -> int:
    try:
        return int(path.name.split("_")[-1])
    except Exception:
        return -1


def _trajectory_dirs(output_dir: Path) -> List[Path]:
    if not output_dir.exists():
        return []
    return sorted(
        [path for path in output_dir.iterdir() if path.is_dir() and path.name.startswith("trajectory_")],
        key=_trajectory_episode,
    )


def _call_action(call: Dict[str, Any]) -> str:
    output = call.get("vlm_output") if isinstance(call.get("vlm_output"), dict) else {}
    return str(output.get("action") or call.get("action") or "").strip().lower()


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _classification_for_go(go_execution: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    existing = go_execution.get("failure_classification")
    if isinstance(existing, dict) and existing.get("primary"):
        return dict(existing), False
    return classify_go_failure(go_execution), True


def _step_rows_from_json(steps: Any) -> List[Dict[str, Any]]:
    if isinstance(steps, list):
        return [dict(step) for step in steps if isinstance(step, dict)]
    if isinstance(steps, dict) and isinstance(steps.get("steps"), list):
        return [dict(step) for step in steps["steps"] if isinstance(step, dict)]
    return []


def _video_path(path: Path) -> str:
    return str(path) if path.is_file() else ""


def summarize_trajectory(trajectory_dir: PathLike, metrics_row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    trajectory_dir = Path(trajectory_dir)
    episode = _trajectory_episode(trajectory_dir)
    metrics_row = dict(metrics_row or {})
    qwen_calls, qwen_calls_present = _read_jsonl(trajectory_dir / "qwen_calls.jsonl")
    steps, steps_present = _read_json(trajectory_dir / "memory" / "steps.json")
    step_rows = _step_rows_from_json(steps)

    go_action_count = 0
    go_execution_count = 0
    classification_missing_count = 0
    classification_backfilled_count = 0
    pitch_reset_count = 0
    missing_pitch_reset_count = 0
    unneutralized_pitch_count = 0
    terminal_missing_pitch_reset_count = 0
    terminal_unneutralized_pitch_count = 0
    failure_counts: Counter = Counter()
    labels_counts: Counter = Counter()

    for call_idx, call in enumerate(qwen_calls):
        has_followup_observation = call_idx < len(qwen_calls) - 1
        if _call_action(call) == "go":
            go_action_count += 1
        go_execution = call.get("runner_go_execution")
        if not isinstance(go_execution, dict):
            continue
        go_execution_count += 1
        classification, backfilled = _classification_for_go(go_execution)
        if backfilled:
            classification_missing_count += 1
            classification_backfilled_count += 1
        primary = str(classification.get("primary") or "unknown")
        failure_counts[primary] += 1
        for label in classification.get("labels", []) or []:
            labels_counts[str(label)] += 1

        pitch_reset = go_execution.get("pitch_reset") if isinstance(go_execution.get("pitch_reset"), dict) else {}
        if pitch_reset:
            pitch_reset_count += 1
        policy_offset = _int_value(pitch_reset.get("policy_pitch_offset"), pitch_offset_from_actions(go_execution.get("policy_actions", []) or []))
        if policy_offset != 0 and not pitch_reset:
            if has_followup_observation:
                missing_pitch_reset_count += 1
            else:
                terminal_missing_pitch_reset_count += 1
        if pitch_reset and _int_value(pitch_reset.get("residual_pitch_offset"), 0) != 0:
            if has_followup_observation:
                unneutralized_pitch_count += 1
            else:
                terminal_unneutralized_pitch_count += 1

    steps_go_execution_count = 0
    for step in step_rows:
        runtime = step.get("runtime") if isinstance(step, dict) and isinstance(step.get("runtime"), dict) else {}
        if isinstance(runtime.get("go_execution"), dict):
            steps_go_execution_count += 1

    videos = {
        "fps_mp4": _video_path(trajectory_dir / "fps.mp4"),
        "metric_mp4": _video_path(trajectory_dir / "metric.mp4"),
    }
    llm_error_calls = _int_value(metrics_row.get("llm_error_calls"), 0)
    priors_context_present = "priors_context_item_count" in metrics_row
    priors_success_count = _int_value(metrics_row.get("priors_success_count"), 0)
    priors_parse_fail_count = _int_value(metrics_row.get("priors_parse_fail_count"), 0)
    priors_fallback_count = _int_value(metrics_row.get("priors_fallback_count"), 0)
    priors_context_item_count = _int_value(metrics_row.get("priors_context_item_count"), 0)
    priors_last_error = str(metrics_row.get("priors_last_error") or "")
    vlm_json_fallback_count = _int_value(metrics_row.get("vlm_json_fallback_count"), 0)
    vlm_json_last_error = str(metrics_row.get("vlm_json_last_error") or "")
    benchmark_valid = _int_value(metrics_row.get("benchmark_valid"), 1)
    benchmark_invalid_reason = str(
        metrics_row.get("benchmark_invalid_reason") or ""
    )
    point_verification_calls = _int_value(
        metrics_row.get("qwen_point_verification_calls"), 0
    )
    point_verification_passes = _int_value(
        metrics_row.get("qwen_point_verification_passes"), 0
    )
    point_verification_rejections = _int_value(
        metrics_row.get("qwen_point_verification_rejections"), 0
    )
    point_verification_errors = _int_value(
        metrics_row.get("qwen_point_verification_errors"), 0
    )
    missing_go_execution_count = max(0, go_action_count - go_execution_count)
    memory_verdict_metrics = _memory_verdict_metrics(qwen_calls)

    blockers: List[str] = []
    if not qwen_calls_present:
        blockers.append("missing_qwen_calls_jsonl")
    if qwen_calls_present and not qwen_calls:
        blockers.append("empty_qwen_calls_jsonl")
    if not videos["fps_mp4"]:
        blockers.append("missing_fps_mp4")
    if not videos["metric_mp4"]:
        blockers.append("missing_metric_mp4")
    if not steps_present:
        blockers.append("missing_steps_json")
    if go_action_count == 0:
        blockers.append("no_go_actions")
    if missing_go_execution_count > 0:
        blockers.append("missing_runner_go_execution")
    if go_execution_count > 0 and classification_missing_count > 0:
        blockers.append("missing_failure_classification")
    if go_execution_count > 0 and steps_go_execution_count == 0:
        blockers.append("missing_steps_runtime_go_execution")
    if missing_pitch_reset_count > 0:
        blockers.append("missing_pitch_reset")
    if unneutralized_pitch_count > 0:
        blockers.append("unneutralized_pitch_offset")
    if llm_error_calls > 0:
        blockers.append("llm_error_calls_present")
    if benchmark_valid == 0:
        blockers.append("benchmark_invalid_run")
    if memory_verdict_metrics["missing_calls"] > 0:
        blockers.append("missing_explicit_memory_verdict")
    if priors_context_present and qwen_calls and priors_context_item_count <= 0:
        blockers.append("empty_target_context_priors")
    if (
        priors_context_present
        and priors_parse_fail_count > 0
        and priors_fallback_count == 0
        and priors_success_count == 0
    ):
        blockers.append("unhandled_priors_parse_failure")

    return {
        "episode": episode,
        "trajectory_dir": str(trajectory_dir),
        "readiness_status": "ready" if not blockers else "needs_attention",
        "readiness_blockers": blockers,
        "metrics": metrics_row,
        "videos": videos,
        "qwen_call_count": len(qwen_calls),
        "go_action_count": go_action_count,
        "go_execution_count": go_execution_count,
        "missing_go_execution_count": missing_go_execution_count,
        "steps_go_execution_count": steps_go_execution_count,
        "failure_classification_missing_count": classification_missing_count,
        "failure_classification_backfilled_count": classification_backfilled_count,
        "pitch_reset_count": pitch_reset_count,
        "missing_pitch_reset_count": missing_pitch_reset_count,
        "unneutralized_pitch_count": unneutralized_pitch_count,
        "terminal_missing_pitch_reset_count": terminal_missing_pitch_reset_count,
        "terminal_unneutralized_pitch_count": terminal_unneutralized_pitch_count,
        "priors_success_count": priors_success_count,
        "priors_parse_fail_count": priors_parse_fail_count,
        "priors_fallback_count": priors_fallback_count,
        "priors_context_item_count": priors_context_item_count,
        "priors_last_error": priors_last_error,
        "vlm_json_fallback_count": vlm_json_fallback_count,
        "vlm_json_last_error": vlm_json_last_error,
        "benchmark_valid": benchmark_valid,
        "benchmark_invalid_reason": benchmark_invalid_reason,
        "point_verification_calls": point_verification_calls,
        "point_verification_passes": point_verification_passes,
        "point_verification_rejections": point_verification_rejections,
        "point_verification_errors": point_verification_errors,
        "memory_verdict_required_calls": memory_verdict_metrics["required_calls"],
        "memory_verdict_explicit_calls": memory_verdict_metrics["explicit_calls"],
        "memory_verdict_main_output_explicit_calls": memory_verdict_metrics[
            "main_output_explicit_calls"
        ],
        "memory_verdict_dedicated_verifier_explicit_calls": memory_verdict_metrics[
            "dedicated_verifier_explicit_calls"
        ],
        "memory_verdict_dedicated_verifier_unresolved_calls": memory_verdict_metrics[
            "dedicated_verifier_unresolved_calls"
        ],
        "memory_verdict_missing_calls": memory_verdict_metrics["missing_calls"],
        "memory_verdict_backend_deferred_calls": memory_verdict_metrics[
            "backend_deferred_calls"
        ],
        "memory_verdict_explicit_rate": memory_verdict_metrics["explicit_rate"],
        "memory_verdict_operation_counts": memory_verdict_metrics["operation_counts"],
        "failure_class_counts": dict(sorted(failure_counts.items())),
        "failure_label_counts": dict(sorted(labels_counts.items())),
        "llm_error_calls": llm_error_calls,
    }


def summarize_output_dir(output_dir: PathLike, metrics_csv: Optional[PathLike] = None) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    metrics_path = Path(metrics_csv) if metrics_csv else None
    metrics_by_episode = _read_metrics(metrics_path)
    episodes = [
        summarize_trajectory(path, metrics_by_episode.get(_trajectory_episode(path), {}))
        for path in _trajectory_dirs(output_dir)
    ]

    failure_counts: Counter = Counter()
    label_counts: Counter = Counter()
    blockers: Counter = Counter()
    for episode in episodes:
        failure_counts.update(episode.get("failure_class_counts", {}))
        label_counts.update(episode.get("failure_label_counts", {}))
        blockers.update(episode.get("readiness_blockers", []))

    totals = {
        "episode_count": len(episodes),
        "ready_episode_count": sum(1 for episode in episodes if episode.get("readiness_status") == "ready"),
        "qwen_call_count": sum(int(episode.get("qwen_call_count", 0)) for episode in episodes),
        "go_action_count": sum(int(episode.get("go_action_count", 0)) for episode in episodes),
        "go_execution_count": sum(int(episode.get("go_execution_count", 0)) for episode in episodes),
        "pitch_reset_count": sum(int(episode.get("pitch_reset_count", 0)) for episode in episodes),
        "missing_pitch_reset_count": sum(int(episode.get("missing_pitch_reset_count", 0)) for episode in episodes),
        "unneutralized_pitch_count": sum(int(episode.get("unneutralized_pitch_count", 0)) for episode in episodes),
        "terminal_missing_pitch_reset_count": sum(
            int(episode.get("terminal_missing_pitch_reset_count", 0)) for episode in episodes
        ),
        "terminal_unneutralized_pitch_count": sum(
            int(episode.get("terminal_unneutralized_pitch_count", 0)) for episode in episodes
        ),
        "priors_success_count": sum(int(episode.get("priors_success_count", 0)) for episode in episodes),
        "priors_parse_fail_count": sum(int(episode.get("priors_parse_fail_count", 0)) for episode in episodes),
        "priors_fallback_count": sum(int(episode.get("priors_fallback_count", 0)) for episode in episodes),
        "priors_context_item_count": sum(int(episode.get("priors_context_item_count", 0)) for episode in episodes),
        "vlm_json_fallback_count": sum(int(episode.get("vlm_json_fallback_count", 0)) for episode in episodes),
        "point_verification_calls": sum(
            int(episode.get("point_verification_calls", 0)) for episode in episodes
        ),
        "point_verification_passes": sum(
            int(episode.get("point_verification_passes", 0)) for episode in episodes
        ),
        "point_verification_rejections": sum(
            int(episode.get("point_verification_rejections", 0)) for episode in episodes
        ),
        "point_verification_errors": sum(
            int(episode.get("point_verification_errors", 0)) for episode in episodes
        ),
        "memory_verdict_required_calls": sum(
            int(episode.get("memory_verdict_required_calls", 0)) for episode in episodes
        ),
        "memory_verdict_explicit_calls": sum(
            int(episode.get("memory_verdict_explicit_calls", 0)) for episode in episodes
        ),
        "memory_verdict_main_output_explicit_calls": sum(
            int(episode.get("memory_verdict_main_output_explicit_calls", 0))
            for episode in episodes
        ),
        "memory_verdict_dedicated_verifier_explicit_calls": sum(
            int(episode.get("memory_verdict_dedicated_verifier_explicit_calls", 0))
            for episode in episodes
        ),
        "memory_verdict_dedicated_verifier_unresolved_calls": sum(
            int(episode.get("memory_verdict_dedicated_verifier_unresolved_calls", 0))
            for episode in episodes
        ),
        "memory_verdict_missing_calls": sum(
            int(episode.get("memory_verdict_missing_calls", 0)) for episode in episodes
        ),
        "memory_verdict_backend_deferred_calls": sum(
            int(episode.get("memory_verdict_backend_deferred_calls", 0))
            for episode in episodes
        ),
        "llm_error_calls": sum(int(episode.get("llm_error_calls", 0)) for episode in episodes),
        "failure_class_counts": dict(sorted(failure_counts.items())),
        "failure_label_counts": dict(sorted(label_counts.items())),
        "readiness_blocker_counts": dict(sorted(blockers.items())),
    }
    if not episodes:
        totals["readiness_blocker_counts"] = {"no_trajectories": 1}
    required_verdicts = int(totals.get("memory_verdict_required_calls", 0) or 0)
    totals["memory_verdict_explicit_rate"] = (
        float(totals.get("memory_verdict_explicit_calls", 0) or 0)
        / float(required_verdicts)
        if required_verdicts
        else 1.0
    )

    ready = bool(episodes) and all(episode.get("readiness_status") == "ready" for episode in episodes)
    return {
        "schema_version": "pre_memory_readiness_v1",
        "output_dir": str(output_dir),
        "metrics_csv": str(metrics_path) if metrics_path else "",
        "readiness_status": "ready" if ready else "needs_attention",
        "totals": totals,
        "episodes": episodes,
    }


def _episode_csv_rows(episodes: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for episode in episodes:
        rows.append(
            {
                "episode": episode.get("episode"),
                "readiness_status": episode.get("readiness_status"),
                "readiness_blockers": ";".join(episode.get("readiness_blockers", [])),
                "qwen_call_count": episode.get("qwen_call_count", 0),
                "go_action_count": episode.get("go_action_count", 0),
                "go_execution_count": episode.get("go_execution_count", 0),
                "missing_go_execution_count": episode.get("missing_go_execution_count", 0),
                "failure_classification_missing_count": episode.get("failure_classification_missing_count", 0),
                "pitch_reset_count": episode.get("pitch_reset_count", 0),
                "missing_pitch_reset_count": episode.get("missing_pitch_reset_count", 0),
                "unneutralized_pitch_count": episode.get("unneutralized_pitch_count", 0),
                "terminal_missing_pitch_reset_count": episode.get("terminal_missing_pitch_reset_count", 0),
                "terminal_unneutralized_pitch_count": episode.get("terminal_unneutralized_pitch_count", 0),
                "priors_success_count": episode.get("priors_success_count", 0),
                "priors_parse_fail_count": episode.get("priors_parse_fail_count", 0),
                "priors_fallback_count": episode.get("priors_fallback_count", 0),
                "priors_context_item_count": episode.get("priors_context_item_count", 0),
                "priors_last_error": episode.get("priors_last_error", ""),
                "vlm_json_fallback_count": episode.get("vlm_json_fallback_count", 0),
                "vlm_json_last_error": episode.get("vlm_json_last_error", ""),
                "point_verification_calls": episode.get("point_verification_calls", 0),
                "point_verification_passes": episode.get("point_verification_passes", 0),
                "point_verification_rejections": episode.get("point_verification_rejections", 0),
                "point_verification_errors": episode.get("point_verification_errors", 0),
                "memory_verdict_required_calls": episode.get(
                    "memory_verdict_required_calls", 0
                ),
                "memory_verdict_explicit_calls": episode.get(
                    "memory_verdict_explicit_calls", 0
                ),
                "memory_verdict_main_output_explicit_calls": episode.get(
                    "memory_verdict_main_output_explicit_calls", 0
                ),
                "memory_verdict_dedicated_verifier_explicit_calls": episode.get(
                    "memory_verdict_dedicated_verifier_explicit_calls", 0
                ),
                "memory_verdict_dedicated_verifier_unresolved_calls": episode.get(
                    "memory_verdict_dedicated_verifier_unresolved_calls", 0
                ),
                "memory_verdict_missing_calls": episode.get(
                    "memory_verdict_missing_calls", 0
                ),
                "memory_verdict_backend_deferred_calls": episode.get(
                    "memory_verdict_backend_deferred_calls", 0
                ),
                "memory_verdict_explicit_rate": episode.get(
                    "memory_verdict_explicit_rate", 1.0
                ),
                "memory_verdict_operation_counts": json.dumps(
                    episode.get("memory_verdict_operation_counts", {}), sort_keys=True
                ),
                "llm_error_calls": episode.get("llm_error_calls", 0),
                "failure_class_counts": json.dumps(episode.get("failure_class_counts", {}), sort_keys=True),
                "fps_mp4": episode.get("videos", {}).get("fps_mp4", ""),
                "metric_mp4": episode.get("videos", {}).get("metric_mp4", ""),
                "trajectory_dir": episode.get("trajectory_dir", ""),
            }
        )
    return rows


def write_pre_memory_readiness(output_dir: PathLike, metrics_csv: Optional[PathLike] = None) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_output_dir(output_dir, metrics_csv)
    json_path = output_dir / "pre_memory_readiness_summary.json"
    csv_path = output_dir / "pre_memory_readiness_summary.csv"

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    rows = _episode_csv_rows(summary.get("episodes", []))
    fieldnames = [
        "episode",
        "readiness_status",
        "readiness_blockers",
        "qwen_call_count",
        "go_action_count",
        "go_execution_count",
        "missing_go_execution_count",
        "failure_classification_missing_count",
        "pitch_reset_count",
        "missing_pitch_reset_count",
        "unneutralized_pitch_count",
        "terminal_missing_pitch_reset_count",
        "terminal_unneutralized_pitch_count",
        "priors_success_count",
        "priors_parse_fail_count",
        "priors_fallback_count",
        "priors_context_item_count",
        "priors_last_error",
        "vlm_json_fallback_count",
        "vlm_json_last_error",
        "point_verification_calls",
        "point_verification_passes",
        "point_verification_rejections",
        "point_verification_errors",
        "memory_verdict_required_calls",
        "memory_verdict_explicit_calls",
        "memory_verdict_main_output_explicit_calls",
        "memory_verdict_dedicated_verifier_explicit_calls",
        "memory_verdict_dedicated_verifier_unresolved_calls",
        "memory_verdict_missing_calls",
        "memory_verdict_backend_deferred_calls",
        "memory_verdict_explicit_rate",
        "memory_verdict_operation_counts",
        "llm_error_calls",
        "failure_class_counts",
        "fps_mp4",
        "metric_mp4",
        "trajectory_dir",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"json": str(json_path), "csv": str(csv_path)}
