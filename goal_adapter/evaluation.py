from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Iterable, Mapping, Protocol, Sequence

from goal_adapter.schema import GoalAdapterOutput


class RefinesGoal(Protocol):
    def refine(self, input_json: Mapping[str, Any]) -> GoalAdapterOutput:
        ...


@dataclass(frozen=True)
class EvaluationCaseResult:
    name: str
    goal_xy_valid: bool
    waypoint_success: bool
    action_type_correct: bool
    controller_action_correct: bool
    reasoning_correct: bool
    recovery_case: bool
    recovery_success: bool | None
    failure_repeated: bool

    @property
    def passed(self) -> bool:
        required_checks = [
            self.goal_xy_valid,
            self.waypoint_success,
            self.action_type_correct,
            self.controller_action_correct,
            self.reasoning_correct,
            not self.failure_repeated,
        ]
        if self.recovery_case:
            required_checks.append(bool(self.recovery_success))
        return all(required_checks)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "goal_xy_valid": self.goal_xy_valid,
            "waypoint_success": self.waypoint_success,
            "action_type_correct": self.action_type_correct,
            "controller_action_correct": self.controller_action_correct,
            "reasoning_correct": self.reasoning_correct,
            "recovery_case": self.recovery_case,
            "recovery_success": self.recovery_success,
            "failure_repeated": self.failure_repeated,
        }


@dataclass(frozen=True)
class EvaluationSummary:
    case_results: list[EvaluationCaseResult]

    @property
    def total_cases(self) -> int:
        return len(self.case_results)

    @property
    def goal_xy_validity_rate(self) -> float:
        return _rate(result.goal_xy_valid for result in self.case_results)

    @property
    def waypoint_success_rate(self) -> float:
        return _rate(result.waypoint_success for result in self.case_results)

    @property
    def action_type_accuracy(self) -> float:
        return _rate(result.action_type_correct for result in self.case_results)

    @property
    def reasoning_correctness_rate(self) -> float:
        return _rate(result.reasoning_correct for result in self.case_results)

    @property
    def recovery_success_rate(self) -> float:
        return _rate(
            bool(result.recovery_success)
            for result in self.case_results
            if result.recovery_case
        )

    @property
    def failure_repeat_rate(self) -> float:
        return _rate(
            result.failure_repeated
            for result in self.case_results
            if result.recovery_case
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "metrics": {
                "goal_xy_validity_rate": self.goal_xy_validity_rate,
                "waypoint_success_rate": self.waypoint_success_rate,
                "action_type_accuracy": self.action_type_accuracy,
                "reasoning_correctness_rate": self.reasoning_correctness_rate,
                "recovery_success_rate": self.recovery_success_rate,
                "failure_repeat_rate": self.failure_repeat_rate,
            },
            "case_results": [result.to_json() for result in self.case_results],
        }


def evaluate_cases(
    adapter: RefinesGoal,
    cases: Iterable[Mapping[str, Any]],
    waypoint_tolerance: float = 1e-6,
) -> EvaluationSummary:
    results = [
        evaluate_output(case, adapter.refine(case["input"]), waypoint_tolerance=waypoint_tolerance)
        for case in cases
    ]
    return EvaluationSummary(case_results=results)


def evaluate_output(
    case: Mapping[str, Any],
    output: GoalAdapterOutput,
    waypoint_tolerance: float = 1e-6,
) -> EvaluationCaseResult:
    goal_required = _goal_xy_required(case, output)
    goal_xy_valid = _valid_xy(output.refined_goal_xy) if goal_required else _optional_valid_xy(output.refined_goal_xy)
    waypoint_success = goal_xy_valid and _waypoint_success(
        case,
        output.refined_goal_xy,
        waypoint_tolerance,
    )
    action_type_correct = _action_value(output) == case.get("expected_action_type")
    controller_action_correct = _controller_action_correct(case, output)
    reasoning_correct = _reasoning_correct(case, output.reasoning)
    recovery_case = _is_recovery_case(case)
    failure_repeated = _failure_repeated(case, output.refined_goal_xy)
    recovery_success = None
    if recovery_case:
        recovery_success = (
            waypoint_success
            and action_type_correct
            and controller_action_correct
            and not failure_repeated
        )

    return EvaluationCaseResult(
        name=str(case.get("name", "unnamed_case")),
        goal_xy_valid=goal_xy_valid,
        waypoint_success=waypoint_success,
        action_type_correct=action_type_correct,
        controller_action_correct=controller_action_correct,
        reasoning_correct=reasoning_correct,
        recovery_case=recovery_case,
        recovery_success=recovery_success,
        failure_repeated=failure_repeated,
    )


def _rate(values: Iterable[bool]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def _action_value(output: GoalAdapterOutput) -> str:
    return output.action_type.value


def _controller_action_correct(case: Mapping[str, Any], output: GoalAdapterOutput) -> bool:
    expected = case.get("expected_controller_action")
    if expected is None:
        return True
    if output.controller_action is None:
        return False
    return output.controller_action.value == expected


def _goal_xy_required(case: Mapping[str, Any], output: GoalAdapterOutput) -> bool:
    if case.get("expected_goal_xy") is not None:
        return True
    if isinstance(case.get("acceptable_goal_regions"), list):
        return True
    return output.action_type.value in {"NAVIGATE", "RESELECT_GOAL"}


def _waypoint_success(
    case: Mapping[str, Any],
    goal_xy: Sequence[float] | None,
    default_tolerance: float,
) -> bool:
    if case.get("expected_goal_xy") is None and not isinstance(case.get("acceptable_goal_regions"), list):
        return True

    regions = case.get("acceptable_goal_regions")
    if isinstance(regions, list):
        return any(_inside_region(goal_xy, region, default_tolerance) for region in regions)

    expected_goal_xy = case.get("expected_goal_xy")
    tolerance = case.get("waypoint_tolerance", default_tolerance)
    return _same_xy(goal_xy, expected_goal_xy, tolerance)


def _inside_region(goal_xy: Sequence[float], region: Any, default_tolerance: float) -> bool:
    if not isinstance(region, Mapping):
        return False
    center = region.get("center")
    radius = region.get("radius", default_tolerance)
    if not isinstance(radius, Real) or radius < 0:
        return False
    return _distance_lte(goal_xy, center, float(radius))


def _reasoning_correct(case: Mapping[str, Any], reasoning: str) -> bool:
    terms = case.get("reasoning_terms", [])
    if not isinstance(terms, list):
        return False
    normalized_reasoning = reasoning.lower()
    return all(str(term).lower() in normalized_reasoning for term in terms)


def _is_recovery_case(case: Mapping[str, Any]) -> bool:
    adapter_input = case.get("input", {})
    if not isinstance(adapter_input, Mapping):
        return False
    target_type = adapter_input.get("target_type")
    progress_state = adapter_input.get("progress_state")
    expected_action_type = case.get("expected_action_type")
    return (
        target_type == "missing_point"
        or progress_state in {"tracking_loss", "blocked", "low_progress", "repeated_view"}
        or expected_action_type in {"LOOK_AROUND", "RESELECT_GOAL", "DIRECT_CONTROL"}
    )


def _failure_repeated(case: Mapping[str, Any], goal_xy: Sequence[float]) -> bool:
    adapter_input = case.get("input", {})
    if not isinstance(adapter_input, Mapping):
        return False
    memory_summary = adapter_input.get("memory_summary", {})
    if not isinstance(memory_summary, Mapping):
        return False

    failed_entries = []
    for key in ("failed_goal_xy", "failed"):
        value = memory_summary.get(key, [])
        if isinstance(value, list):
            failed_entries.extend(value)

    return any(_same_xy(goal_xy, failed_goal, 1e-6) for failed_goal in failed_entries)


def _valid_xy(value: Any) -> bool:
    return (
        not _is_string_like(value)
        and isinstance(value, Sequence)
        and len(value) == 2
        and all(isinstance(component, Real) and isfinite(float(component)) for component in value)
    )


def _optional_valid_xy(value: Any) -> bool:
    return value is None or _valid_xy(value)


def _same_xy(left: Any, right: Any, tolerance: float) -> bool:
    if not _valid_xy(left) or not _valid_xy(right):
        return False
    return (
        abs(float(left[0]) - float(right[0])) <= tolerance
        and abs(float(left[1]) - float(right[1])) <= tolerance
    )


def _distance_lte(left: Any, right: Any, radius: float) -> bool:
    if not _valid_xy(left) or not _valid_xy(right):
        return False
    dx = float(left[0]) - float(right[0])
    dy = float(left[1]) - float(right[1])
    return (dx * dx + dy * dy) <= radius * radius


def _is_string_like(value: Any) -> bool:
    return isinstance(value, (str, bytes, bytearray))
