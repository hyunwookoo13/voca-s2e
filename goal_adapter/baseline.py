from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence

from goal_adapter.schema import (
    ActionType,
    Confidence,
    ControllerAction,
    GoalAdapterInput,
    GoalAdapterOutput,
)


@dataclass(frozen=True)
class CandidateWaypoint:
    goal_xy: list[float]
    image_point: list[int] | None
    kind: str
    semantic_score: float


def refine_with_baseline_rules(adapter_input: GoalAdapterInput, default_goal_xy: list[float]) -> GoalAdapterOutput:
    candidate = _select_candidate(adapter_input)
    if candidate is not None:
        action_type = _action_for_candidate(adapter_input)
        return GoalAdapterOutput(
            refined_goal_xy=candidate.goal_xy,
            selected_image_point=candidate.image_point,
            action_type=action_type,
            reasoning=_reasoning_for_candidate(adapter_input, candidate, action_type),
            confidence=Confidence.MEDIUM,
        )

    if _s2e_needs_direct_recovery(adapter_input):
        return GoalAdapterOutput(
            refined_goal_xy=None,
            selected_image_point=None,
            action_type=ActionType.DIRECT_CONTROL,
            controller_action=ControllerAction.MOVE_BACK,
            reasoning=(
                "S2E did not provide a valid trajectory and no alternative local "
                "goal candidate is available, so the adapter requests a direct "
                "controller recovery action."
            ),
            confidence=Confidence.LOW,
        )

    fallback_goal_xy = (
        adapter_input.current_goal_xy
        or adapter_input.previous_waypoint
        or list(default_goal_xy)
    )
    return GoalAdapterOutput(
        refined_goal_xy=fallback_goal_xy,
        selected_image_point=None,
        action_type=ActionType.NAVIGATE,
        reasoning=(
            "No valid candidate waypoint was provided, so the adapter keeps the "
            "current S2E local goal."
        ),
        confidence=Confidence.LOW,
    )


def _select_candidate(adapter_input: GoalAdapterInput) -> CandidateWaypoint | None:
    failed_goal_xy = _failed_goal_xy(adapter_input.memory_summary)
    candidates = [
        candidate
        for candidate in _candidate_waypoints(adapter_input.memory_summary)
        if not _matches_any(candidate.goal_xy, failed_goal_xy)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.semantic_score)


def _candidate_waypoints(memory_summary: Mapping[str, Any]) -> list[CandidateWaypoint]:
    raw_candidates = memory_summary.get("candidate_waypoints", [])
    if not isinstance(raw_candidates, list):
        return []

    candidates: list[CandidateWaypoint] = []
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, Mapping):
            continue
        if raw_candidate.get("navigable") is False:
            continue
        goal_xy = _parse_xy(raw_candidate.get("goal_xy"))
        if goal_xy is None:
            continue
        candidates.append(
            CandidateWaypoint(
                goal_xy=goal_xy,
                image_point=_parse_image_point(raw_candidate.get("image_point")),
                kind=str(raw_candidate.get("kind", "candidate waypoint")),
                semantic_score=_parse_score(raw_candidate.get("semantic_score")),
            )
        )
    return candidates


def _failed_goal_xy(memory_summary: Mapping[str, Any]) -> list[list[float]]:
    failed_entries = []
    for key in ("failed_goal_xy", "failed"):
        value = memory_summary.get(key, [])
        if isinstance(value, list):
            failed_entries.extend(value)
    return [goal_xy for goal_xy in (_parse_xy(entry) for entry in failed_entries) if goal_xy is not None]


def _action_for_candidate(adapter_input: GoalAdapterInput) -> ActionType:
    if adapter_input.target_type.value == "missing_point" or adapter_input.progress_state.value == "tracking_loss":
        return ActionType.LOOK_AROUND
    if adapter_input.progress_state.value in {"low_progress", "blocked", "repeated_view"}:
        return ActionType.RESELECT_GOAL
    return ActionType.NAVIGATE


def _s2e_needs_direct_recovery(adapter_input: GoalAdapterInput) -> bool:
    return adapter_input.s2e_status.value in {
        "failed",
        "collision",
        "no_valid_trajectory",
    }


def _reasoning_for_candidate(
    adapter_input: GoalAdapterInput,
    candidate: CandidateWaypoint,
    action_type: ActionType,
) -> str:
    if action_type is ActionType.LOOK_AROUND:
        return (
            "Tracking loss or a missing target point was detected, so the adapter "
            f"uses look-around evidence and selects the navigable {candidate.kind}."
        )
    if action_type is ActionType.RESELECT_GOAL:
        return (
            f"The current direction is {adapter_input.progress_state.value} and may "
            "repeat a failed goal, so the adapter reselects the navigable "
            f"{candidate.kind}."
        )
    if adapter_input.target_type.value == "gps":
        return (
            "The coarse GPS target is refined to a closer navigable waypoint: "
            f"{candidate.kind}."
        )
    if adapter_input.target_type.value == "object_point":
        return (
            "The object point is treated as a semantic target, and the adapter "
            f"selects an approach waypoint near it: {candidate.kind}."
        )
    return f"The adapter refines the local goal to the navigable {candidate.kind}."


def _matches_any(goal_xy: list[float], other_goals: list[list[float]]) -> bool:
    return any(_same_xy(goal_xy, other_goal) for other_goal in other_goals)


def _same_xy(left: list[float], right: list[float], tolerance: float = 1e-6) -> bool:
    return abs(left[0] - right[0]) <= tolerance and abs(left[1] - right[1]) <= tolerance


def _parse_xy(value: Any) -> list[float] | None:
    if _is_string_like(value) or not isinstance(value, Sequence) or len(value) != 2:
        return None
    if not all(isinstance(component, Real) for component in value):
        return None
    return [float(value[0]), float(value[1])]


def _parse_image_point(value: Any) -> list[int] | None:
    if value is None:
        return None
    if _is_string_like(value) or not isinstance(value, Sequence) or len(value) != 2:
        return None
    if not all(isinstance(component, int) for component in value):
        return None
    return [int(value[0]), int(value[1])]


def _parse_score(value: Any) -> float:
    if isinstance(value, Real):
        return float(value)
    return 0.0


def _is_string_like(value: Any) -> bool:
    return isinstance(value, (str, bytes, bytearray))
