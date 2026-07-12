import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from navigation_supervisor import evaluate_execution_progress
from pixel_candidate_gate import (
    build_pixel_candidates,
    partition_pixel_candidates,
    validate_pixel_candidate_selection,
)


def evaluate_revisit_pairs(path: Path, threshold: float) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    tp = fp = tn = fn = 0
    for row in rows:
        expected = str(row.get("label")) == "same_place"
        predicted = float(row.get("similarity", 0.0)) >= float(threshold)
        tp += int(expected and predicted)
        fp += int(not expected and predicted)
        tn += int(not expected and not predicted)
        fn += int(expected and not predicted)
    precision = tp / float(tp + fp) if tp + fp else 1.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    false_positive_rate = fp / float(fp + tn) if fp + tn else 0.0
    return {
        "suite": "revisit_retrieval",
        "threshold": float(threshold),
        "samples": len(rows),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "false_positive_rate": round(false_positive_rate, 6),
        "passed": bool(fp == 0 and precision >= 0.98),
        "criterion": "zero labeled false merges and precision >= 0.98",
    }


def deterministic_guard_cases() -> List[Dict[str, Any]]:
    views = [{"view_id": 0, "view_type": "front", "relative_heading_deg": 0}]
    memory = {
        "candidate_refs": {
            "exits": [
                {
                    "candidate_ref": "failed_front",
                    "edge_id": "edge_failed_front",
                    "view_type_hint": "front",
                    "avoid": True,
                    "status": "blocked",
                    "reason": "prior collision",
                }
            ]
        }
    }
    generated = build_pixel_candidates(views, (480, 640, 3), memory)
    executable, excluded = partition_pixel_candidates(generated)
    gate_input = {
        "pixel_candidates": {
            "candidates": executable,
            "excluded_candidates": excluded,
        }
    }
    excluded_selection = validate_pixel_candidate_selection(
        {"action": "go", "selected_candidate_ref": excluded[0]["candidate_ref"]},
        gate_input,
    )
    same_sector = evaluate_execution_progress(
        start_position_xyz=[0.0, 0.0, 0.0],
        final_position_xyz=[0.4, 0.0, 0.0],
        collision_count=0,
        controller_reached_waypoint=False,
        supervisor_mode="escape_deadlock",
        min_translation_m=0.05,
        strategic_evidence={"sector_departure_selected": False},
    )
    new_sector = evaluate_execution_progress(
        start_position_xyz=[0.0, 0.0, 0.0],
        final_position_xyz=[0.4, 0.0, 0.0],
        collision_count=0,
        controller_reached_waypoint=False,
        supervisor_mode="escape_deadlock",
        min_translation_m=0.05,
        strategic_evidence={"sector_departure_selected": True},
    )
    first_stop = evaluate_execution_progress(
        start_position_xyz=[],
        final_position_xyz=[],
        collision_count=0,
        controller_reached_waypoint=False,
        supervisor_mode="verify_target",
        min_translation_m=0.05,
        strategic_evidence={"target_verification_stable": False},
    )
    confirmed_stop = evaluate_execution_progress(
        start_position_xyz=[],
        final_position_xyz=[],
        collision_count=0,
        controller_reached_waypoint=False,
        supervisor_mode="verify_target",
        min_translation_m=0.05,
        strategic_evidence={"target_verification_stable": True},
    )
    return [
        {
            "suite": "deadlock_guard",
            "case": "known_failed_candidates_not_offered",
            "passed": len(generated) == 6 and not executable and len(excluded) == 6,
            "detail": "generated={} executable={} excluded={}".format(
                len(generated), len(executable), len(excluded)
            ),
        },
        {
            "suite": "deadlock_guard",
            "case": "excluded_ref_fails_closed",
            "passed": not excluded_selection["passed"]
            and excluded_selection["reason"] == "selected_candidate_excluded",
            "detail": excluded_selection["reason"],
        },
        {
            "suite": "strategic_progress",
            "case": "same_failed_sector_not_escape",
            "passed": not same_sector["strategic_progress"],
            "detail": same_sector["strategic_rule_id"],
        },
        {
            "suite": "strategic_progress",
            "case": "new_sector_is_escape_progress",
            "passed": new_sector["strategic_progress"],
            "detail": new_sector["strategic_rule_id"],
        },
        {
            "suite": "stop_temporal_guard",
            "case": "single_observation_cannot_complete",
            "passed": not first_stop["strategic_progress"],
            "detail": first_stop["strategic_rule_id"],
        },
        {
            "suite": "stop_temporal_guard",
            "case": "stable_confirmation_can_complete",
            "passed": confirmed_stop["strategic_progress"],
            "detail": confirmed_stop["strategic_rule_id"],
        },
    ]


def summarize_stop_cases(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    labeled = [case for case in cases if case.get("expected_pass") is not None]
    positives = [case for case in labeled if bool(case.get("expected_pass"))]
    negatives = [case for case in labeled if not bool(case.get("expected_pass"))]
    true_positive = sum(bool(case.get("actual_pass")) for case in positives)
    false_positive = sum(bool(case.get("actual_pass")) for case in negatives)
    tpr = true_positive / float(len(positives)) if positives else 0.0
    fpr = false_positive / float(len(negatives)) if negatives else 0.0
    return {
        "suite": "stop_verifier",
        "samples": len(labeled),
        "positive_samples": len(positives),
        "negative_samples": len(negatives),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_positive_rate": round(tpr, 6),
        "false_positive_rate": round(fpr, 6),
        "passed": bool(negatives and false_positive == 0 and (not positives or tpr >= 0.5)),
        "criterion": "zero false stops and >= 0.5 positive recall",
    }


def write_stress_outputs(
    output_dir: Path,
    *,
    summaries: Sequence[Dict[str, Any]],
    cases: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "voca_navigation_stress_suite_v1",
        "metadata": dict(metadata),
        "summaries": [dict(item) for item in summaries],
        "all_required_suites_passed": all(
            bool(item.get("passed")) for item in summaries if item.get("required", True)
        ),
    }
    summary_path = output_dir / "stress_suite_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    case_path = output_dir / "stress_suite_cases.jsonl"
    with case_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    return {"summary_json": str(summary_path), "cases_jsonl": str(case_path)}
