from __future__ import annotations

from typing import Any

from goal_adapter.evaluation import evaluate_output
from goal_adapter.schema import ActionType, Confidence, ControllerAction, GoalAdapterOutput


EVALUATOR_DIAGNOSTIC_CASES = [
    {
        "name": "diagnostic_wrong_waypoint",
        "failure_mode": "wrong_waypoint",
        "case": {
            "name": "diagnostic_wrong_waypoint",
            "input": {
                "target_type": "gps",
                "progress_state": "normal",
                "memory_summary": {},
            },
            "expected_goal_xy": [1.0, 0.0],
            "expected_action_type": "NAVIGATE",
            "reasoning_terms": ["coarse"],
        },
        "output": GoalAdapterOutput(
            refined_goal_xy=[9.0, 9.0],
            selected_image_point=None,
            action_type=ActionType.NAVIGATE,
            reasoning="The coarse target is incorrectly sent far away.",
            confidence=Confidence.LOW,
        ),
        "expected_failed_checks": ["waypoint_success"],
    },
    {
        "name": "diagnostic_wrong_action_type",
        "failure_mode": "wrong_action_type",
        "case": {
            "name": "diagnostic_wrong_action_type",
            "input": {
                "target_type": "missing_point",
                "progress_state": "tracking_loss",
                "memory_summary": {},
            },
            "expected_goal_xy": [0.5, 0.5],
            "expected_action_type": "LOOK_AROUND",
            "reasoning_terms": ["tracking"],
        },
        "output": GoalAdapterOutput(
            refined_goal_xy=[0.5, 0.5],
            selected_image_point=None,
            action_type=ActionType.NAVIGATE,
            reasoning="Tracking loss was ignored even though tracking is uncertain.",
            confidence=Confidence.LOW,
        ),
        "expected_failed_checks": ["action_type_correct", "recovery_success"],
    },
    {
        "name": "diagnostic_repeated_failed_goal",
        "failure_mode": "repeated_failed_goal",
        "case": {
            "name": "diagnostic_repeated_failed_goal",
            "input": {
                "target_type": "language",
                "progress_state": "blocked",
                "memory_summary": {
                    "failed_goal_xy": [[1.0, 0.0]],
                },
            },
            "expected_goal_xy": [0.2, 1.2],
            "expected_action_type": "RESELECT_GOAL",
            "reasoning_terms": ["blocked"],
        },
        "output": GoalAdapterOutput(
            refined_goal_xy=[1.0, 0.0],
            selected_image_point=[320, 220],
            action_type=ActionType.RESELECT_GOAL,
            reasoning="The blocked direction is selected again.",
            confidence=Confidence.LOW,
        ),
        "expected_failed_checks": [
            "waypoint_success",
            "failure_repeated",
            "recovery_success",
        ],
    },
    {
        "name": "diagnostic_wrong_controller_action",
        "failure_mode": "wrong_controller_action",
        "case": {
            "name": "diagnostic_wrong_controller_action",
            "input": {
                "target_type": "language",
                "progress_state": "blocked",
                "s2e_status": "no_valid_trajectory",
                "memory_summary": {},
            },
            "expected_action_type": "DIRECT_CONTROL",
            "expected_controller_action": "MOVE_BACK",
            "reasoning_terms": ["s2e", "controller"],
        },
        "output": GoalAdapterOutput(
            refined_goal_xy=None,
            selected_image_point=None,
            action_type=ActionType.DIRECT_CONTROL,
            controller_action=ControllerAction.TURN_LEFT,
            reasoning="S2E failed, so the adapter requests a controller recovery action.",
            confidence=Confidence.LOW,
        ),
        "expected_failed_checks": ["controller_action_correct", "recovery_success"],
    },
]


def run_evaluator_diagnostics() -> dict[str, Any]:
    diagnostics = [_evaluate_diagnostic_case(case) for case in EVALUATOR_DIAGNOSTIC_CASES]
    detected_failures = sum(1 for diagnostic in diagnostics if diagnostic["detected"])
    total = len(diagnostics)
    return {
        "total_diagnostics": total,
        "detected_failures": detected_failures,
        "detection_rate": detected_failures / total if total else 0.0,
        "diagnostics": diagnostics,
    }


def _evaluate_diagnostic_case(diagnostic_case: dict[str, Any]) -> dict[str, Any]:
    result = evaluate_output(
        diagnostic_case["case"],
        diagnostic_case["output"],
    )
    result_json = result.to_json()
    failed_checks = _failed_checks(result_json)
    expected_failed_checks = diagnostic_case["expected_failed_checks"]
    detected = all(check in failed_checks for check in expected_failed_checks)

    return {
        "name": diagnostic_case["name"],
        "failure_mode": diagnostic_case["failure_mode"],
        "detected": detected,
        "expected_failed_checks": list(expected_failed_checks),
        "actual_failed_checks": failed_checks,
        "evaluation_result": result_json,
    }


def _failed_checks(result_json: dict[str, Any]) -> list[str]:
    failed_checks = [
        field
        for field in (
            "goal_xy_valid",
            "waypoint_success",
            "action_type_correct",
            "controller_action_correct",
            "reasoning_correct",
            "recovery_success",
        )
        if result_json.get(field) is False
    ]
    if result_json.get("failure_repeated") is True:
        failed_checks.append("failure_repeated")
    return failed_checks
