import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union


PathLike = Union[str, Path]


COUNT_FIELDS = (
    "coarse_goal_guard_calls",
    "coarse_goal_distractor_rejections",
    "coarse_goal_stop_deferrals",
    "coarse_goal_direction_guard_calls",
    "coarse_goal_direction_rejections",
    "coarse_goal_direction_detour_exemptions",
    "llm_calls",
    "llm_error_calls",
    "priors_calls",
    "qwen_navigation_vlm_calls",
    "qwen_go_actions",
    "qwen_go_no_progress",
    "qwen_target_approach_limited_actions",
    "qwen_target_stop_refinement_actions",
    "qwen_target_anchor_reacquisition_actions",
    "qwen_target_lock_exhaustions",
    "qwen_target_approach_detour_triggers",
    "qwen_rotate_actions",
    "qwen_observation_requests",
    "qwen_stop_actions",
    "qwen_raw_go_decisions",
    "qwen_pixel_candidate_gate_passes",
    "qwen_pixel_candidate_gate_rejections",
    "qwen_point_verification_calls",
    "qwen_point_verification_passes",
    "qwen_point_verification_rejections",
    "qwen_point_verification_backend_calls",
    "qwen_point_verification_consistency_retries",
    "qwen_stop_verification_calls",
    "qwen_stop_verification_passes",
    "qwen_stop_verification_rejections",
    "qwen_stop_identity_critic_calls",
    "qwen_stop_identity_critic_passes",
    "qwen_stop_identity_critic_rejections",
    "qwen_stop_identity_critic_errors",
    "qwen_rgb_candidates_offered",
    "qwen_rgb_candidates_avoided",
    "qwen_rgb_candidates_verification_required",
    "qwen_selected_candidates_verification_required",
    "qwen_strategic_forced_exit_checks",
    "qwen_strategic_forced_exit_passes",
    "qwen_strategic_forced_exit_rejections",
    "qwen_strategic_force_leave_overrides",
    "qwen_strategic_forced_exit_failures",
    "qwen_strategic_full_sweep_requests",
    "qwen_strategic_memory_backtrack_overrides",
    "qwen_strategic_memory_backtrack_attempts",
    "qwen_strategic_memory_backtrack_executed",
    "qwen_strategic_memory_backtrack_point_rejections",
    "qwen_strategic_novel_sector_overrides",
    "qwen_strategic_novel_sector_attempts",
    "qwen_strategic_novel_sector_executed",
    "qwen_strategic_novel_sector_no_progress",
    "qwen_strategic_normal_backtrack_rejections",
    "qwen_strategic_exit_alias_repairs",
    "qwen_exit_floor_relaxations",
    "qwen_strategic_search_current_room",
    "qwen_strategic_leave_current_room",
    "qwen_strategic_traverse_gateway",
    "qwen_strategic_approach_target",
    "qwen_strategic_verify_target",
    "qwen_strategic_escape_deadlock",
    "memory_embedding_observations",
    "memory_embedding_views",
    "memory_revisit_queries",
    "memory_revisit_candidate_contexts",
    "memory_revisit_candidates_offered",
    "memory_ops_requested",
    "memory_ops_accepted",
    "memory_ops_rejected",
    "memory_revisits_confirmed",
    "memory_revisits_rejected",
    "memory_soft_merges",
    "memory_object_belief_updates",
    "memory_semantic_state_updates",
    "memory_room_category_updates",
    "memory_room_transitions",
)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _sum(rows: Sequence[Dict[str, Any]], key: str) -> float:
    return sum(_number(row.get(key)) for row in rows)


def _mean(rows: Sequence[Dict[str, Any]], key: str) -> float:
    return _sum(rows, key) / len(rows) if rows else 0.0


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = float(successes) / float(total)
    denominator = 1.0 + (z * z) / total
    center = (proportion + (z * z) / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            (proportion * (1.0 - proportion) / total)
            + (z * z) / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _mean_ci95(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    if len(values) <= 1:
        return mean, mean
    radius = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return mean - radius, mean + radius


def _weighted_mean(
    rows: Sequence[Dict[str, Any]],
    value_key: str,
    weight_key: str,
) -> float:
    total_weight = _sum(rows, weight_key)
    if total_weight <= 0:
        return _mean(rows, value_key)
    return sum(
        _number(row.get(value_key)) * _number(row.get(weight_key)) for row in rows
    ) / total_weight


def _navigation_vlm_mean(rows: Sequence[Dict[str, Any]]) -> float:
    instrumented = [
        row
        for row in rows
        if "qwen_navigation_vlm_avg_time_sec" in row
        and _number(row.get("qwen_navigation_vlm_calls")) > 0
    ]
    if instrumented:
        return _weighted_mean(
            instrumented,
            "qwen_navigation_vlm_avg_time_sec",
            "qwen_navigation_vlm_calls",
        )
    return _weighted_mean(rows, "llm_avg_time_sec", "llm_calls")


def _find_metrics_csv(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(
        candidate
        for candidate in path.glob("*.csv")
        if candidate.name.startswith("objnav_")
        and "readiness" not in candidate.name
    )
    if not candidates:
        raise FileNotFoundError("no objnav benchmark CSV found in {}".format(path))
    return candidates[0]


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _row_is_benchmark_valid(row: Dict[str, Any]) -> bool:
    raw = row.get("benchmark_valid")
    if raw is None or str(raw).strip() == "":
        return True
    return _number(raw, 1.0) >= 0.5


def _manifest_metadata(run_dir: Path) -> Dict[str, str]:
    manifests = sorted(run_dir.glob("trajectory_*/benchmark_manifest.json"))
    manifests.extend(
        sorted(run_dir.glob("trajectories/trajectory_*/benchmark_manifest.json"))
    )
    if not manifests:
        return {
            "qwen_model": "",
            "qwen_model_root": "",
            "qwen_base_url": "",
            "memory_schema": "",
            "experiment_fingerprint": "",
            "benchmark_seed": "",
            "policy_checkpoint_sha256": "",
            "git_commit": "",
            "memory_enabled": "",
            "pixelnav_candidate_scoring": "",
        }
    try:
        payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {
            "qwen_model": "",
            "qwen_model_root": "",
            "qwen_base_url": "",
            "memory_schema": "",
            "experiment_fingerprint": "",
            "benchmark_seed": "",
            "policy_checkpoint_sha256": "",
            "git_commit": "",
            "memory_enabled": "",
            "pixelnav_candidate_scoring": "",
        }
    reproducibility = (
        payload.get("reproducibility")
        if isinstance(payload.get("reproducibility"), dict)
        else {}
    )
    config = (
        reproducibility.get("config")
        if isinstance(reproducibility.get("config"), dict)
        else {}
    )
    checkpoint = (
        reproducibility.get("policy_checkpoint")
        if isinstance(reproducibility.get("policy_checkpoint"), dict)
        else {}
    )
    git_state = (
        reproducibility.get("git")
        if isinstance(reproducibility.get("git"), dict)
        else {}
    )
    return {
        "qwen_model": str(payload.get("qwen_model") or ""),
        "qwen_model_root": str(
            payload.get("qwen_model_root")
            or config.get("VOCA_QWEN_MODEL_ROOT")
            or ""
        ),
        "qwen_base_url": str(payload.get("qwen_base_url") or ""),
        "memory_schema": str(payload.get("memory_schema") or ""),
        "experiment_fingerprint": str(
            reproducibility.get("experiment_fingerprint_sha256") or ""
        ),
        "benchmark_seed": str(payload.get("benchmark_seed") or ""),
        "policy_checkpoint_sha256": str(checkpoint.get("sha256") or ""),
        "git_commit": str(git_state.get("commit") or ""),
        "memory_enabled": str(config.get("VOCA_QWEN_MEMORY_SIDECAR") or ""),
        "pixelnav_candidate_scoring": str(
            config.get("VOCA_PIXELNAV_CANDIDATE_SCORING") or ""
        ),
    }


def summarize_run(label: str, path: PathLike) -> Dict[str, Any]:
    requested_path = Path(path).resolve()
    metrics_csv = _find_metrics_csv(requested_path)
    run_dir = metrics_csv.parent
    rows = _read_csv(metrics_csv)
    valid_rows = [row for row in rows if _row_is_benchmark_valid(row)]
    invalid_rows = [row for row in rows if not _row_is_benchmark_valid(row)]
    metadata = _manifest_metadata(run_dir)
    counts = {key: int(round(_sum(valid_rows, key))) for key in COUNT_FIELDS}
    start_distance = _mean(valid_rows, "start_distance_to_goal")
    final_distance = _mean(valid_rows, "final_distance_to_goal")
    success_count = int(round(_sum(valid_rows, "success")))
    stop_rows = [
        row for row in valid_rows if _number(row.get("qwen_stop_actions")) > 0
    ]
    false_positive_stop_rows = [
        row for row in stop_rows if _number(row.get("success")) < 0.5
    ]
    no_stop_failure_rows = [
        row
        for row in valid_rows
        if _number(row.get("success")) < 0.5
        and _number(row.get("qwen_stop_actions")) <= 0
    ]
    near_miss_failure_rows = []
    for row in valid_rows:
        success_distance = max(
            0.0,
            _number(row.get("success_distance_m"), 0.1),
        )
        if (
            _number(row.get("success")) < 0.5
            and _number(row.get("final_distance_to_goal"))
            <= max(0.2, 2.0 * success_distance) + 1e-6
        ):
            near_miss_failure_rows.append(row)
    success_ci_low, success_ci_high = _wilson_interval(
        success_count,
        len(valid_rows),
    )
    spl_values = [_number(row.get("spl")) for row in valid_rows]
    soft_spl_values = [_number(row.get("soft_spl")) for row in valid_rows]
    final_distance_values = [
        _number(row.get("final_distance_to_goal")) for row in valid_rows
    ]
    category_stats: Dict[str, Dict[str, int]] = {}
    for row in valid_rows:
        category = str(row.get("object_goal") or "unknown")
        item = category_stats.setdefault(category, {"episodes": 0, "successes": 0})
        item["episodes"] += 1
        item["successes"] += int(_number(row.get("success")) >= 0.5)
    memory_backends = sorted(
        {
            str(row.get("memory_embedding_backend") or "")
            for row in valid_rows
            if row.get("memory_embedding_backend")
        }
    )
    summary: Dict[str, Any] = {
        "label": str(label),
        "run_dir": str(run_dir),
        "metrics_csv": str(metrics_csv),
        "episode_count": len(rows),
        "valid_episode_count": len(valid_rows),
        "invalid_episode_count": len(invalid_rows),
        "benchmark_complete": int(not invalid_rows and bool(valid_rows)),
        "invalid_reasons_json": json.dumps(
            [
                {
                    "episode": row.get("episode"),
                    "reason": str(row.get("benchmark_invalid_reason") or "unknown"),
                }
                for row in invalid_rows
            ],
            sort_keys=True,
        ),
        "success_count": success_count,
        "success_rate": _ratio(success_count, len(valid_rows)),
        "success_ci95_low": success_ci_low,
        "success_ci95_high": success_ci_high,
        "stop_action_episode_count": len(stop_rows),
        "successful_stop_episode_count": len(stop_rows)
        - len(false_positive_stop_rows),
        "false_positive_stop_episode_count": len(false_positive_stop_rows),
        "stop_episode_precision": _ratio(
            len(stop_rows) - len(false_positive_stop_rows),
            len(stop_rows),
        ),
        "no_stop_failure_episode_count": len(no_stop_failure_rows),
        "near_miss_failure_episode_count": len(near_miss_failure_rows),
        "spl_mean": _mean(valid_rows, "spl"),
        "spl_std": _sample_std(spl_values),
        "soft_spl_mean": _mean(valid_rows, "soft_spl"),
        "soft_spl_std": _sample_std(soft_spl_values),
        "start_distance_mean_m": start_distance,
        "final_distance_mean_m": final_distance,
        "final_distance_std_m": _sample_std(final_distance_values),
        "distance_reduction_mean_m": start_distance - final_distance,
        "episode_time_mean_sec": _mean(valid_rows, "episode_time_sec"),
        "num_steps_mean": _mean(valid_rows, "num_steps"),
        "collision_count_mean": _mean(valid_rows, "collision_count"),
        "max_step_truncation_count": int(
            round(_sum(valid_rows, "truncated_by_max_steps"))
        ),
        "planning_cycle_limit_count": int(
            round(_sum(valid_rows, "planning_cycle_limit_reached"))
        ),
        "category_results_json": json.dumps(category_stats, sort_keys=True),
        "llm_decision_time_weighted_mean_sec": _navigation_vlm_mean(valid_rows),
        "priors_time_weighted_mean_sec": _weighted_mean(
            valid_rows, "priors_avg_time_sec", "priors_calls"
        ),
        "go_no_progress_rate": _ratio(
            counts["qwen_go_no_progress"], counts["qwen_go_actions"]
        ),
        "pixel_candidate_gate_rejection_rate": _ratio(
            counts["qwen_pixel_candidate_gate_rejections"],
            counts["qwen_raw_go_decisions"],
        ),
        "point_verifier_rejection_rate": _ratio(
            counts["qwen_point_verification_rejections"],
            counts["qwen_point_verification_calls"],
        ),
        "strategic_forced_exit_pass_rate": _ratio(
            counts["qwen_strategic_forced_exit_passes"],
            counts["qwen_strategic_forced_exit_checks"],
        ),
        "strategic_forced_exit_rejection_rate": _ratio(
            counts["qwen_strategic_forced_exit_rejections"],
            counts["qwen_strategic_forced_exit_checks"],
        ),
        "stop_verifier_pass_rate": _ratio(
            counts["qwen_stop_verification_passes"],
            counts["qwen_stop_verification_calls"],
        ),
        "memory_op_acceptance_rate": _ratio(
            counts["memory_ops_accepted"], counts["memory_ops_requested"]
        ),
        "memory_place_nodes_mean": _mean(valid_rows, "memory_place_nodes"),
        "memory_failed_frontier_nodes_mean": _mean(
            valid_rows, "memory_failed_frontier_nodes"
        ),
        "memory_negative_edges_mean": _mean(valid_rows, "memory_negative_edges"),
        "memory_same_place_edges_mean": _mean(valid_rows, "memory_same_place_edges"),
        "memory_embedding_backend": ";".join(memory_backends),
        **metadata,
        **counts,
    }
    return summary


def _mcnemar_exact_p_value(baseline_only: int, treatment_only: int) -> float:
    discordant = int(baseline_only) + int(treatment_only)
    if discordant <= 0:
        return 1.0
    lower = min(int(baseline_only), int(treatment_only))
    tail = sum(math.comb(discordant, index) for index in range(lower + 1))
    return min(1.0, 2.0 * tail / float(2**discordant))


def _paired_comparison(
    baseline_label: str,
    baseline_rows: Sequence[Dict[str, Any]],
    treatment_label: str,
    treatment_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    baseline = {
        str(row.get("episode")): row
        for row in baseline_rows
        if _row_is_benchmark_valid(row)
    }
    treatment = {
        str(row.get("episode")): row
        for row in treatment_rows
        if _row_is_benchmark_valid(row)
    }
    keys = sorted(set(baseline).intersection(treatment))
    baseline_only = 0
    treatment_only = 0
    spl_deltas: List[float] = []
    final_distance_deltas: List[float] = []
    time_deltas: List[float] = []
    for key in keys:
        base = baseline[key]
        variant = treatment[key]
        base_success = _number(base.get("success")) >= 0.5
        variant_success = _number(variant.get("success")) >= 0.5
        baseline_only += int(base_success and not variant_success)
        treatment_only += int(variant_success and not base_success)
        spl_deltas.append(_number(variant.get("spl")) - _number(base.get("spl")))
        final_distance_deltas.append(
            _number(variant.get("final_distance_to_goal"))
            - _number(base.get("final_distance_to_goal"))
        )
        time_deltas.append(
            _number(variant.get("episode_time_sec"))
            - _number(base.get("episode_time_sec"))
        )
    spl_ci_low, spl_ci_high = _mean_ci95(spl_deltas)
    paired_count = len(keys)
    return {
        "baseline": baseline_label,
        "treatment": treatment_label,
        "paired_episode_count": paired_count,
        "success_rate_delta": _ratio(treatment_only - baseline_only, paired_count),
        "baseline_only_successes": baseline_only,
        "treatment_only_successes": treatment_only,
        "mcnemar_exact_p": _mcnemar_exact_p_value(baseline_only, treatment_only),
        "spl_delta_mean": statistics.mean(spl_deltas) if spl_deltas else 0.0,
        "spl_delta_ci95_low": spl_ci_low,
        "spl_delta_ci95_high": spl_ci_high,
        "final_distance_delta_mean_m": (
            statistics.mean(final_distance_deltas) if final_distance_deltas else 0.0
        ),
        "episode_time_delta_mean_sec": (
            statistics.mean(time_deltas) if time_deltas else 0.0
        ),
    }


def summarize_runs(runs: Iterable[Tuple[str, PathLike]]) -> Dict[str, Any]:
    run_specs = list(runs)
    summaries = [summarize_run(label, path) for label, path in run_specs]
    rows_by_label = {
        str(label): _read_csv(_find_metrics_csv(Path(path).resolve()))
        for label, path in run_specs
    }
    pairwise = []
    if run_specs:
        baseline_label = str(run_specs[0][0])
        for label, _path in run_specs[1:]:
            pairwise.append(
                _paired_comparison(
                    baseline_label,
                    rows_by_label[baseline_label],
                    str(label),
                    rows_by_label[str(label)],
                )
            )
    return {
        "schema_version": "voca_navigation_ablation_v2",
        "run_count": len(summaries),
        "runs": summaries,
        "paired_comparisons": pairwise,
    }


def write_summary(
    runs: Iterable[Tuple[str, PathLike]],
    output_dir: PathLike,
) -> Dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = summarize_runs(runs)
    json_path = output / "navigation_ablation_summary.json"
    csv_path = output / "navigation_ablation_summary.csv"
    pairwise_csv_path = output / "navigation_ablation_paired_comparisons.csv"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = payload["runs"]
    fieldnames = list(rows[0].keys()) if rows else ["label", "episode_count"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    pairwise_rows = payload["paired_comparisons"]
    pairwise_fields = (
        list(pairwise_rows[0].keys())
        if pairwise_rows
        else ["baseline", "treatment", "paired_episode_count"]
    )
    with pairwise_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=pairwise_fields)
        writer.writeheader()
        writer.writerows(pairwise_rows)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "pairwise_csv": str(pairwise_csv_path),
    }
