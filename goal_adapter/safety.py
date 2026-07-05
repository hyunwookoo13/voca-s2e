from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence

from goal_adapter.schema import ActionType, GoalAdapterInput, GoalAdapterOutput


@dataclass(frozen=True)
class SafeNavigationCandidate:
    goal_xy: list[float]
    image_point: list[int] | None
    score: float
    rationale: str


def enforce_navigation_safety(
    adapter_input: GoalAdapterInput,
    output: GoalAdapterOutput,
) -> GoalAdapterOutput:
    if output.action_type not in {ActionType.NAVIGATE, ActionType.RESELECT_GOAL}:
        return output

    candidates = _safe_candidates(adapter_input.memory_summary)
    if not candidates:
        return output

    selected_candidate = _select_safe_candidate(candidates, output)
    if selected_candidate is None or _same_goal(output.refined_goal_xy, selected_candidate.goal_xy):
        return output

    return GoalAdapterOutput(
        refined_goal_xy=selected_candidate.goal_xy,
        selected_image_point=selected_candidate.image_point,
        action_type=output.action_type,
        controller_action=output.controller_action,
        reasoning=(
            f"{output.reasoning} Safety guard snapped the final goal to a "
            f"navmesh-validated navigable candidate: {selected_candidate.rationale}."
        ),
        confidence=output.confidence,
    )


def _select_safe_candidate(
    candidates: list[SafeNavigationCandidate],
    output: GoalAdapterOutput,
) -> SafeNavigationCandidate | None:
    if output.selected_image_point is not None:
        nearest = min(
            candidates,
            key=lambda candidate: _image_distance(output.selected_image_point, candidate.image_point),
        )
        if _image_distance(output.selected_image_point, nearest.image_point) <= 12.0:
            return nearest

    if output.refined_goal_xy is not None:
        nearest = min(
            candidates,
            key=lambda candidate: _goal_distance(output.refined_goal_xy, candidate.goal_xy),
        )
        if _goal_distance(output.refined_goal_xy, nearest.goal_xy) <= 0.75:
            return nearest

    return max(candidates, key=lambda candidate: candidate.score)


def _safe_candidates(memory_summary: Mapping[str, Any]) -> list[SafeNavigationCandidate]:
    raw_candidates = memory_summary.get("candidate_waypoints", [])
    if not isinstance(raw_candidates, list):
        return []

    candidates: list[SafeNavigationCandidate] = []
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, Mapping):
            continue
        if raw_candidate.get("navigable") is not True:
            continue
        goal_xy = _parse_xy(raw_candidate.get("goal_xy"))
        if goal_xy is None:
            continue
        candidates.append(
            SafeNavigationCandidate(
                goal_xy=goal_xy,
                image_point=_parse_image_point(raw_candidate.get("image_point")),
                score=_parse_score(raw_candidate),
                rationale=str(raw_candidate.get("rationale") or raw_candidate.get("kind") or "candidate waypoint"),
            )
        )
    return candidates


def _parse_score(candidate: Mapping[str, Any]) -> float:
    for key in ("semantic_score", "score"):
        value = candidate.get(key)
        if isinstance(value, Real):
            return float(value)
    return 0.0


def _parse_xy(value: Any) -> list[float] | None:
    if _is_string_like(value) or not isinstance(value, Sequence) or len(value) != 2:
        return None
    if not all(isinstance(component, Real) and math.isfinite(float(component)) for component in value):
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


def _image_distance(left: Sequence[int], right: Sequence[int] | None) -> float:
    if right is None:
        return math.inf
    dx = int(left[0]) - int(right[0])
    dy = int(left[1]) - int(right[1])
    return math.sqrt((dx * dx) + (dy * dy))


def _goal_distance(left: Sequence[float], right: Sequence[float]) -> float:
    dx = float(left[0]) - float(right[0])
    dy = float(left[1]) - float(right[1])
    return math.sqrt((dx * dx) + (dy * dy))


def _same_goal(left: Sequence[float] | None, right: Sequence[float]) -> bool:
    if left is None:
        return False
    return _goal_distance(left, right) <= 1e-6


def _is_string_like(value: Any) -> bool:
    return isinstance(value, (str, bytes, bytearray))
