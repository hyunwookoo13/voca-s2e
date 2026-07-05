from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import shutil
import time
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from goal_adapter.decision_parser import parse_vlm_decision
from goal_adapter.ollama_provider import OllamaVLMConfig, OllamaVLMProvider
from goal_adapter.safety import enforce_navigation_safety
from goal_adapter.schema import ActionType, ControllerAction, GoalAdapterInput, GoalAdapterOutput
from goal_adapter.visualization import CaseVisualizationArtifacts, render_case_visualizations


class DecisionProvider(Protocol):
    def decide(self, adapter_input: GoalAdapterInput) -> GoalAdapterOutput:
        ...


@dataclass(frozen=True)
class VLMDecisionBenchmarkCase:
    case_id: str
    category: str
    input_json: dict[str, Any]
    expected_action_type: ActionType
    expected_goal_xy: list[float] | None = None
    acceptable_goal_regions: list[dict[str, Any]] = field(default_factory=list)
    expected_controller_action: ControllerAction | None = None
    forbidden_goal_xy: list[list[float]] = field(default_factory=list)
    reasoning_terms: list[str] = field(default_factory=list)
    waypoint_tolerance: float = 0.75

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, Any],
        base_dir: str | Path | None = None,
    ) -> "VLMDecisionBenchmarkCase":
        case_id = _required_text(payload, "case_id")
        category = _required_text(payload, "category")
        input_json = _require_mapping(payload.get("input"), "input")
        expected = _require_mapping(payload.get("expected"), "expected")
        return cls(
            case_id=case_id,
            category=category,
            input_json=_resolve_input_image_paths(input_json, base_dir),
            expected_action_type=_parse_action_type(expected.get("action_type")),
            expected_goal_xy=_optional_xy(expected.get("goal_xy"), "expected.goal_xy"),
            acceptable_goal_regions=_optional_regions(expected.get("acceptable_goal_regions")),
            expected_controller_action=_optional_controller_action(expected.get("controller_action")),
            forbidden_goal_xy=_xy_list(expected.get("forbidden_goal_xy", []), "expected.forbidden_goal_xy"),
            reasoning_terms=_text_list(expected.get("reasoning_terms", []), "expected.reasoning_terms"),
            waypoint_tolerance=_optional_positive_float(
                expected.get("waypoint_tolerance", 0.75),
                "expected.waypoint_tolerance",
            ),
        )

    def to_json(self) -> dict[str, Any]:
        expected: dict[str, Any] = {
            "action_type": self.expected_action_type.value,
            "goal_xy": self.expected_goal_xy,
            "acceptable_goal_regions": self.acceptable_goal_regions,
            "controller_action": self.expected_controller_action.value
            if self.expected_controller_action is not None
            else None,
            "forbidden_goal_xy": self.forbidden_goal_xy,
            "reasoning_terms": self.reasoning_terms,
            "waypoint_tolerance": self.waypoint_tolerance,
        }
        return {
            "case_id": self.case_id,
            "category": self.category,
            "input": self.input_json,
            "expected": expected,
        }

    @property
    def expects_waypoint(self) -> bool:
        return self.expected_goal_xy is not None or bool(self.acceptable_goal_regions)

    @property
    def checks_forbidden_goal(self) -> bool:
        return bool(self.forbidden_goal_xy)

    @property
    def checks_reasoning(self) -> bool:
        return bool(self.reasoning_terms)

    @property
    def checks_controller_action(self) -> bool:
        return self.expected_controller_action is not None


@dataclass(frozen=True)
class VLMDecisionCaseResult:
    case_id: str
    category: str
    status: str
    passed: bool
    parse_success: bool
    action_type_correct: bool
    waypoint_success: bool | None
    forbidden_goal_checked: bool
    forbidden_goal_repeated: bool
    reasoning_correct: bool | None
    controller_action_correct: bool | None
    selected_point_navigable: bool | None
    attempts: int
    elapsed_seconds: float
    output_json: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "status": self.status,
            "passed": self.passed,
            "parse_success": self.parse_success,
            "action_type_correct": self.action_type_correct,
            "waypoint_success": self.waypoint_success,
            "forbidden_goal_checked": self.forbidden_goal_checked,
            "forbidden_goal_repeated": self.forbidden_goal_repeated,
            "reasoning_correct": self.reasoning_correct,
            "controller_action_correct": self.controller_action_correct,
            "selected_point_navigable": self.selected_point_navigable,
            "attempts": self.attempts,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "output_json": self.output_json,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class VLMDecisionBenchmarkSummary:
    case_results: list[VLMDecisionCaseResult]
    model_name: str
    output_dir: Path

    @property
    def total_cases(self) -> int:
        return len(self.case_results)

    @property
    def metrics(self) -> dict[str, float | None]:
        return _metrics_for(self.case_results)

    @property
    def category_metrics(self) -> dict[str, dict[str, float | None]]:
        categories = sorted({result.category for result in self.case_results})
        return {
            category: _metrics_for(
                result for result in self.case_results if result.category == category
            )
            for category in categories
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "output_dir": str(self.output_dir),
            "total_cases": self.total_cases,
            "metrics": self.metrics,
            "category_metrics": self.category_metrics,
            "case_results": [result.to_json() for result in self.case_results],
        }


def load_vlm_benchmark_cases(cases_dir: str | Path) -> list[VLMDecisionBenchmarkCase]:
    cases_dir = Path(cases_dir)
    cases: list[VLMDecisionBenchmarkCase] = []
    for path in sorted(cases_dir.rglob("*.json")):
        payload = _read_json_object(path)
        cases.append(VLMDecisionBenchmarkCase.from_json(payload, base_dir=path.parent))
    return cases


def evaluate_vlm_benchmark_case(
    case: VLMDecisionBenchmarkCase,
    output: GoalAdapterOutput,
    attempts: int = 1,
    elapsed_seconds: float = 0.0,
) -> VLMDecisionCaseResult:
    action_type_correct = output.action_type == case.expected_action_type
    waypoint_success = _waypoint_success(case, output.refined_goal_xy) if case.expects_waypoint else None
    forbidden_goal_repeated = _forbidden_goal_repeated(case, output.refined_goal_xy)
    reasoning_correct = _reasoning_correct(case, output.reasoning) if case.checks_reasoning else None
    controller_action_correct = (
        _controller_action_correct(case, output) if case.checks_controller_action else None
    )
    selected_point_navigable = _selected_point_navigable(case, output.selected_image_point)
    passed = (
        action_type_correct
        and waypoint_success is not False
        and not forbidden_goal_repeated
        and reasoning_correct is not False
        and controller_action_correct is not False
        and selected_point_navigable is not False
    )
    return VLMDecisionCaseResult(
        case_id=case.case_id,
        category=case.category,
        status="success",
        passed=passed,
        parse_success=True,
        action_type_correct=action_type_correct,
        waypoint_success=waypoint_success,
        forbidden_goal_checked=case.checks_forbidden_goal,
        forbidden_goal_repeated=forbidden_goal_repeated,
        reasoning_correct=reasoning_correct,
        controller_action_correct=controller_action_correct,
        selected_point_navigable=selected_point_navigable,
        attempts=attempts,
        elapsed_seconds=elapsed_seconds,
        output_json=output.to_json(),
    )


def run_vlm_decision_benchmark(
    cases: Iterable[VLMDecisionBenchmarkCase],
    output_dir: str | Path = Path("reports") / "vlm_benchmark" / "latest",
    decision_provider: DecisionProvider | None = None,
    model_name: str = "gemma4:26b",
    timeout_seconds: float = 420.0,
    max_attempts: int = 1,
    raise_on_error: bool = False,
) -> VLMDecisionBenchmarkSummary:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = decision_provider or OllamaVLMProvider(
        OllamaVLMConfig(model=model_name, timeout_seconds=timeout_seconds)
    )

    results: list[VLMDecisionCaseResult] = []
    for case in list(cases):
        result = _run_one_case(
            case=case,
            output_dir=output_dir,
            provider=provider,
            max_attempts=max_attempts,
            raise_on_error=raise_on_error,
        )
        results.append(result)

    summary = VLMDecisionBenchmarkSummary(
        case_results=results,
        model_name=model_name,
        output_dir=output_dir,
    )
    _write_json(output_dir / "summary.json", summary.to_json())
    _write_summary_csv(output_dir / "summary.csv", summary.case_results)
    (output_dir / "index.html").write_text(_render_index_html(summary), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an S2E-free VLM decision benchmark and write static reports.",
    )
    parser.add_argument("--cases-dir", required=True, help="Directory containing benchmark case JSON files.")
    parser.add_argument(
        "--output-dir",
        default=str(Path("reports") / "vlm_benchmark" / "latest"),
        help="Directory where benchmark artifacts will be written.",
    )
    parser.add_argument("--model", default="gemma4:26b", help="Ollama VLM model name.")
    parser.add_argument("--timeout-seconds", type=float, default=420.0, help="Ollama request timeout.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Provider/parser attempts per case before recording failure.",
    )
    parser.add_argument(
        "--raise-on-error",
        action="store_true",
        help="Raise provider/parser errors instead of recording failed case reports.",
    )
    parser.add_argument(
        "--fail-on-case-failure",
        action="store_true",
        help="Return exit code 1 when any benchmark case fails.",
    )
    args = parser.parse_args(argv)

    cases = load_vlm_benchmark_cases(args.cases_dir)
    summary = run_vlm_decision_benchmark(
        cases=cases,
        output_dir=args.output_dir,
        model_name=args.model,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
        raise_on_error=args.raise_on_error,
    )
    print(f"Wrote VLM benchmark report to {summary.output_dir}")
    print(f"total_cases={summary.total_cases}")
    print(f"case_success_rate={summary.metrics['case_success_rate']}")
    print(f"html={summary.output_dir / 'index.html'}")
    if args.fail_on_case_failure and summary.metrics["case_success_rate"] != 1.0:
        return 1
    return 0


def _run_one_case(
    case: VLMDecisionBenchmarkCase,
    output_dir: Path,
    provider: DecisionProvider,
    max_attempts: int,
    raise_on_error: bool,
) -> VLMDecisionCaseResult:
    case_dir = output_dir / "cases" / _safe_path_segment(case.case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    raw_response = ""
    started_at = time.time()
    attempts = 0
    visualization_artifacts: CaseVisualizationArtifacts | None = None

    try:
        prepared_input = _prepare_case_input(case, case_dir)
        adapter_input = GoalAdapterInput.from_json(prepared_input)
        _write_json(case_dir / "goal_adapter_input.json", adapter_input.to_json())
        output = None
        last_exc: Exception | None = None
        for attempts in range(1, max(1, max_attempts) + 1):
            try:
                raw_response = _generate_raw_or_decide(provider, adapter_input)
                output = parse_vlm_decision(raw_response)
                break
            except Exception as exc:
                last_exc = exc
                if attempts >= max(1, max_attempts):
                    raise
        if output is None:
            raise RuntimeError(f"VLM decision failed without output: {last_exc}")
        output = enforce_navigation_safety(adapter_input, output)
        decision_payload = output.to_json()
        visualization_case = VLMDecisionBenchmarkCase(
            case_id=case.case_id,
            category=case.category,
            input_json=adapter_input.to_json(),
            expected_action_type=case.expected_action_type,
            expected_goal_xy=case.expected_goal_xy,
            acceptable_goal_regions=case.acceptable_goal_regions,
            expected_controller_action=case.expected_controller_action,
            forbidden_goal_xy=case.forbidden_goal_xy,
            reasoning_terms=case.reasoning_terms,
            waypoint_tolerance=case.waypoint_tolerance,
        )
        visualization_artifacts = render_case_visualizations(visualization_case, output, case_dir)
        result = evaluate_vlm_benchmark_case(
            case=case,
            output=output,
            attempts=attempts,
            elapsed_seconds=time.time() - started_at,
        )
    except Exception as exc:
        if raise_on_error:
            raise
        decision_payload = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        result = _error_case_result(
            case=case,
            exc=exc,
            attempts=max(1, attempts),
            elapsed_seconds=time.time() - started_at,
        )

    (case_dir / "vlm_raw_response.txt").write_text(str(raw_response), encoding="utf-8")
    _write_json(case_dir / "vlm_decision.json", decision_payload)
    _write_json(case_dir / "result.json", result.to_json())
    (case_dir / "index.html").write_text(
        _render_case_html(
            case=case,
            result=result,
            decision=decision_payload,
            raw_response=str(raw_response),
            visualization_artifacts=visualization_artifacts,
        ),
        encoding="utf-8",
    )
    return result


def _prepare_case_input(case: VLMDecisionBenchmarkCase, case_dir: Path) -> dict[str, Any]:
    payload = dict(case.input_json)
    current_rgb = payload.get("current_rgb")
    if not isinstance(current_rgb, str) or not current_rgb:
        raise ValueError(f"{case.case_id} input must include current_rgb image path")
    payload["current_rgb"] = str(_copy_image(current_rgb, case_dir, "current_rgb").resolve())

    lookaround_images = []
    for index, image_path in enumerate(payload.get("optional_lookaround_images", [])):
        if isinstance(image_path, str) and image_path:
            lookaround_images.append(
                str(_copy_image(image_path, case_dir, f"lookaround_{index:02d}").resolve())
            )
    payload["optional_lookaround_images"] = lookaround_images
    return payload


def _copy_image(source_path: str, destination_dir: Path, stem: str) -> Path:
    source = Path(source_path)
    if not source.exists():
        raise ValueError(f"benchmark image does not exist: {source}")
    suffix = source.suffix if source.suffix else ".png"
    destination = destination_dir / f"{stem}{suffix}"
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    return destination


def _generate_raw_or_decide(provider: DecisionProvider, adapter_input: GoalAdapterInput) -> str:
    generate_raw = getattr(provider, "generate_raw_decision", None)
    if callable(generate_raw):
        return str(generate_raw(adapter_input))
    output = provider.decide(adapter_input)
    return json.dumps(output.to_json(), ensure_ascii=False)


def _error_case_result(
    case: VLMDecisionBenchmarkCase,
    exc: Exception,
    attempts: int,
    elapsed_seconds: float,
) -> VLMDecisionCaseResult:
    return VLMDecisionCaseResult(
        case_id=case.case_id,
        category=case.category,
        status="error",
        passed=False,
        parse_success=False,
        action_type_correct=False,
        waypoint_success=False if case.expects_waypoint else None,
        forbidden_goal_checked=case.checks_forbidden_goal,
        forbidden_goal_repeated=False,
        reasoning_correct=False if case.checks_reasoning else None,
        controller_action_correct=False if case.checks_controller_action else None,
        selected_point_navigable=None,
        attempts=attempts,
        elapsed_seconds=elapsed_seconds,
        output_json=None,
        error_type=type(exc).__name__,
        error_message=str(exc),
    )


def _metrics_for(results: Iterable[VLMDecisionCaseResult]) -> dict[str, float | None]:
    results = list(results)
    return {
        "parse_success_rate": _rate(result.parse_success for result in results),
        "case_success_rate": _rate(result.passed for result in results),
        "action_type_accuracy": _rate(result.action_type_correct for result in results),
        "waypoint_success_rate": _rate(
            result.waypoint_success for result in results if result.waypoint_success is not None
        ),
        "reasoning_correctness_rate": _rate(
            result.reasoning_correct for result in results if result.reasoning_correct is not None
        ),
        "controller_action_accuracy": _rate(
            result.controller_action_correct
            for result in results
            if result.controller_action_correct is not None
        ),
        "selected_point_navigability_rate": _rate(
            result.selected_point_navigable
            for result in results
            if result.selected_point_navigable is not None
        ),
        "forbidden_goal_repeat_rate": _rate(
            result.forbidden_goal_repeated for result in results if result.forbidden_goal_checked
        ),
    }


def _rate(values: Iterable[bool]) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _waypoint_success(case: VLMDecisionBenchmarkCase, goal_xy: Sequence[float] | None) -> bool:
    if case.expected_goal_xy is not None:
        return _distance_lte(goal_xy, case.expected_goal_xy, case.waypoint_tolerance)
    return any(_inside_region(goal_xy, region, case.waypoint_tolerance) for region in case.acceptable_goal_regions)


def _inside_region(
    goal_xy: Sequence[float] | None,
    region: Mapping[str, Any],
    default_tolerance: float,
) -> bool:
    center = region.get("center")
    radius = region.get("radius", default_tolerance)
    if not isinstance(radius, Real) or float(radius) < 0:
        return False
    return _distance_lte(goal_xy, center, float(radius))


def _forbidden_goal_repeated(
    case: VLMDecisionBenchmarkCase,
    goal_xy: Sequence[float] | None,
) -> bool:
    return any(_distance_lte(goal_xy, forbidden_xy, case.waypoint_tolerance) for forbidden_xy in case.forbidden_goal_xy)


def _reasoning_correct(case: VLMDecisionBenchmarkCase, reasoning: str) -> bool:
    normalized_reasoning = _normalize_reasoning_text(reasoning)
    return all(_normalize_reasoning_text(term) in normalized_reasoning for term in case.reasoning_terms)


def _normalize_reasoning_text(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def _controller_action_correct(case: VLMDecisionBenchmarkCase, output: GoalAdapterOutput) -> bool:
    return output.controller_action == case.expected_controller_action


def _selected_point_navigable(
    case: VLMDecisionBenchmarkCase,
    selected_image_point: Sequence[int] | None,
    threshold_pixels: float = 12.0,
) -> bool | None:
    if not _valid_image_point(selected_image_point):
        return None
    candidates = []
    for candidate in _candidate_waypoints(case):
        image_point = candidate.get("image_point")
        if _valid_image_point(image_point):
            candidates.append((candidate, image_point))
    if not candidates:
        return None

    closest_candidate, closest_point = min(
        candidates,
        key=lambda item: _pixel_distance(selected_image_point, item[1]),
    )
    if _pixel_distance(selected_image_point, closest_point) > threshold_pixels:
        return False
    return bool(closest_candidate.get("navigable", False))


def _candidate_waypoints(case: VLMDecisionBenchmarkCase) -> list[Mapping[str, Any]]:
    memory_summary = case.input_json.get("memory_summary", {})
    if not isinstance(memory_summary, Mapping):
        return []
    candidates = memory_summary.get("candidate_waypoints", [])
    if not isinstance(candidates, list):
        return []
    return [candidate for candidate in candidates if isinstance(candidate, Mapping)]


def _pixel_distance(left: Sequence[int], right: Sequence[int]) -> float:
    dx = int(left[0]) - int(right[0])
    dy = int(left[1]) - int(right[1])
    return math.sqrt((dx * dx) + (dy * dy))


def _valid_image_point(value: Any) -> bool:
    return (
        not isinstance(value, (str, bytes, bytearray))
        and isinstance(value, Sequence)
        and len(value) == 2
        and all(isinstance(component, int) for component in value)
    )


def _distance_lte(left: Any, right: Any, radius: float) -> bool:
    if not _valid_xy(left) or not _valid_xy(right):
        return False
    dx = float(left[0]) - float(right[0])
    dy = float(left[1]) - float(right[1])
    return (dx * dx + dy * dy) <= radius * radius


def _valid_xy(value: Any) -> bool:
    return (
        not isinstance(value, (str, bytes, bytearray))
        and isinstance(value, Sequence)
        and len(value) == 2
        and all(isinstance(component, Real) and math.isfinite(float(component)) for component in value)
    )


def _parse_action_type(value: Any) -> ActionType:
    try:
        return ActionType(value)
    except ValueError as exc:
        allowed = ", ".join(action.value for action in ActionType)
        raise ValueError(f"expected.action_type must be one of: {allowed}") from exc


def _optional_controller_action(value: Any) -> ControllerAction | None:
    if value is None:
        return None
    try:
        return ControllerAction(value)
    except ValueError as exc:
        allowed = ", ".join(action.value for action in ControllerAction)
        raise ValueError(f"expected.controller_action must be one of: {allowed}") from exc


def _optional_regions(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("expected.acceptable_goal_regions must be a list")
    regions = []
    for index, item in enumerate(value):
        region = _require_mapping(item, f"expected.acceptable_goal_regions[{index}]")
        regions.append(dict(region))
    return regions


def _resolve_input_image_paths(
    input_json: Mapping[str, Any],
    base_dir: str | Path | None,
) -> dict[str, Any]:
    payload = dict(input_json)
    if base_dir is None:
        return payload
    base_dir = Path(base_dir)
    current_rgb = payload.get("current_rgb")
    if isinstance(current_rgb, str) and current_rgb and not Path(current_rgb).is_absolute():
        payload["current_rgb"] = str((base_dir / current_rgb).resolve())

    resolved_lookaround_images = []
    for image_path in payload.get("optional_lookaround_images", []):
        if isinstance(image_path, str) and image_path and not Path(image_path).is_absolute():
            resolved_lookaround_images.append(str((base_dir / image_path).resolve()))
        else:
            resolved_lookaround_images.append(image_path)
    if "optional_lookaround_images" in payload:
        payload["optional_lookaround_images"] = resolved_lookaround_images
    memory_summary = payload.get("memory_summary")
    if isinstance(memory_summary, Mapping):
        resolved_memory = dict(memory_summary)
        metric_map = resolved_memory.get("metric_map")
        if isinstance(metric_map, Mapping):
            resolved_metric_map = dict(metric_map)
            metric_map_path = resolved_metric_map.get("path")
            if (
                isinstance(metric_map_path, str)
                and metric_map_path
                and not Path(metric_map_path).is_absolute()
            ):
                resolved_metric_map["path"] = str((base_dir / metric_map_path).resolve())
            resolved_memory["metric_map"] = resolved_metric_map
        payload["memory_summary"] = resolved_memory
    return payload


def _required_text(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _text_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return list(value)


def _xy_list(value: Any, field_name: str) -> list[list[float]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return [_xy(item, f"{field_name}[{index}]") for index, item in enumerate(value)]


def _optional_xy(value: Any, field_name: str) -> list[float] | None:
    if value is None:
        return None
    return _xy(value, field_name)


def _xy(value: Any, field_name: str) -> list[float]:
    if not _valid_xy(value):
        raise ValueError(f"{field_name} must be a numeric [x, y] list")
    return [float(value[0]), float(value[1])]


def _optional_positive_float(value: Any, field_name: str) -> float:
    if not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a number")
    value = float(value)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


def _read_json_object(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _require_mapping(payload, str(path))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_summary_csv(path: Path, results: Sequence[VLMDecisionCaseResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "case_id",
                "category",
                "status",
                "passed",
                "parse_success",
                "action_type_correct",
                "waypoint_success",
                "forbidden_goal_checked",
                "forbidden_goal_repeated",
                "reasoning_correct",
                "controller_action_correct",
                "selected_point_navigable",
                "attempts",
                "elapsed_seconds",
                "error_type",
                "error_message",
            ],
        )
        writer.writeheader()
        for result in results:
            row = result.to_json()
            row.pop("output_json", None)
            writer.writerow(row)


def _render_index_html(summary: VLMDecisionBenchmarkSummary) -> str:
    rows = "\n".join(_case_row(result) for result in summary.case_results)
    category_rows = "\n".join(
        _category_row(category, metrics)
        for category, metrics in summary.category_metrics.items()
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VLM 의사결정 벤치마크</title>
  <style>
    :root {{
      --ink: #1f2933;
      --muted: #5f6f82;
      --line: #d8dee8;
      --panel: #f7f9fc;
      --ok: #0f766e;
      --bad: #b91c1c;
    }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    header {{
      padding: 20px 24px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }}
    main {{
      padding: 20px 24px 32px;
      display: grid;
      gap: 22px;
    }}
    h1, h2 {{
      margin: 0;
      letter-spacing: 0;
    }}
    h1 {{ font-size: 24px; }}
    h2 {{ font-size: 18px; }}
    .muted {{ color: var(--muted); }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      background: var(--panel);
    }}
    .metric strong {{
      display: block;
      font-size: 22px;
      margin-top: 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
      background: var(--panel);
    }}
    a {{ color: #0b5cad; text-decoration: none; }}
    .pass {{ color: var(--ok); font-weight: 700; }}
    .fail {{ color: var(--bad); font-weight: 700; }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>VLM Decision Benchmark</h1>
      <div class="muted">모델: {html.escape(summary.model_name)} · 케이스: {summary.total_cases}</div>
    </div>
    <div class="{_pass_class(summary.metrics.get('case_success_rate') == 1.0)}">
      case_success_rate={_format_metric(summary.metrics.get("case_success_rate"))}
    </div>
  </header>
  <main>
    <section class="metrics">
      {_metric_card("파싱 성공률", summary.metrics.get("parse_success_rate"))}
      {_metric_card("action_type 정확도", summary.metrics.get("action_type_accuracy"))}
      {_metric_card("waypoint 성공률", summary.metrics.get("waypoint_success_rate"))}
      {_metric_card("reasoning 정확도", summary.metrics.get("reasoning_correctness_rate"))}
      {_metric_card("선택 지점 주행 가능률", summary.metrics.get("selected_point_navigability_rate"))}
      {_metric_card("금지 goal 반복률", summary.metrics.get("forbidden_goal_repeat_rate"))}
    </section>
    <section>
      <h2>유형별 지표</h2>
      <table>
        <thead><tr><th>유형</th><th>케이스 성공</th><th>Action</th><th>Waypoint</th><th>Reasoning</th><th>선택 지점</th><th>금지 goal 반복</th></tr></thead>
        <tbody>{category_rows}</tbody>
      </table>
    </section>
    <section>
      <h2>케이스 목록</h2>
      <table>
        <thead><tr><th>케이스</th><th>유형</th><th>상태</th><th>Action</th><th>Waypoint</th><th>선택 지점</th><th>Reasoning</th><th>오류</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def _render_case_html(
    case: VLMDecisionBenchmarkCase,
    result: VLMDecisionCaseResult,
    decision: Mapping[str, Any],
    raw_response: str,
    visualization_artifacts: CaseVisualizationArtifacts | None = None,
) -> str:
    input_json = json.dumps(case.input_json, indent=2, ensure_ascii=False)
    expected_json = json.dumps(case.to_json()["expected"], indent=2, ensure_ascii=False)
    decision_json = json.dumps(decision, indent=2, ensure_ascii=False)
    image_gallery = _case_image_gallery(visualization_artifacts)
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(case.case_id)} · VLM 벤치마크 케이스</title>
  <style>
    :root {{
      --ink: #1f2933;
      --muted: #5f6f82;
      --line: #d8dee8;
      --panel: #f7f9fc;
      --ok: #0f766e;
      --bad: #b91c1c;
    }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    header {{
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }}
    main {{
      padding: 20px 24px 32px;
      display: grid;
      grid-template-columns: minmax(260px, 460px) minmax(300px, 1fr);
      gap: 20px;
    }}
    h1, h2 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 22px; }}
    h2 {{ font-size: 17px; margin-bottom: 8px; }}
    img {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      background: var(--panel);
    }}
    .stack {{
      display: grid;
      gap: 12px;
    }}
    .status {{
      font-weight: 700;
    }}
    .pass {{ color: var(--ok); }}
    .fail {{ color: var(--bad); }}
    .muted {{ color: var(--muted); }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 12px;
      line-height: 1.45;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px;
      text-align: left;
    }}
    @media (max-width: 780px) {{
      main {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>{html.escape(case.case_id)}</h1>
      <div class="muted">{html.escape(case.category)}</div>
    </div>
    <div class="status {_pass_class(result.passed)}">{'통과' if result.passed else '실패'}</div>
  </header>
  <main>
    <section class="stack">
      {image_gallery}
      <div class="panel">
        <h2>검증 항목</h2>
        <table>
          <tbody>
            <tr><th>파싱</th><td>{_format_bool(result.parse_success)}</td></tr>
            <tr><th>Action</th><td>{_format_bool(result.action_type_correct)}</td></tr>
            <tr><th>Waypoint</th><td>{_format_optional_bool(result.waypoint_success)}</td></tr>
            <tr><th>선택 지점 주행 가능</th><td>{_format_optional_bool(result.selected_point_navigable)}</td></tr>
            <tr><th>금지 goal 반복 없음</th><td>{_format_bool(not result.forbidden_goal_repeated)}</td></tr>
            <tr><th>Reasoning</th><td>{_format_optional_bool(result.reasoning_correct)}</td></tr>
          </tbody>
        </table>
      </div>
    </section>
    <section class="stack">
      <div class="panel">
        <h2>기대값</h2>
        <pre>{html.escape(expected_json)}</pre>
      </div>
      <div class="panel">
        <h2>VLM 결정</h2>
        <pre>{html.escape(decision_json)}</pre>
      </div>
      <div class="panel">
        <h2>원본 VLM 응답</h2>
        <pre>{html.escape(raw_response)}</pre>
      </div>
      <div class="panel">
        <h2>입력 컨텍스트</h2>
        <pre>{html.escape(input_json)}</pre>
      </div>
    </section>
  </main>
</body>
</html>
"""


def _case_image_gallery(visualization_artifacts: CaseVisualizationArtifacts | None) -> str:
    if visualization_artifacts is None:
        return '<img src="current_rgb.png" alt="현재 RGB 프레임">'
    return (
        '<div class="panel">'
        "<h2>RGB 오버레이</h2>"
        '<img src="current_rgb_overlay.png" alt="VLM 선택 지점과 기대 지점이 표시된 RGB 프레임">'
        '<p class="muted">파란 십자: VLM이 RGB 이미지에서 선택한 지점. 초록 링: 기대 이미지 지점. 주황 링: coarse 이미지 지점.</p>'
        "</div>"
        '<div class="panel">'
        "<h2>탑다운 오버레이</h2>"
        '<img src="topdown_overlay.png" alt="로봇과 목표 지점이 표시된 탑다운 metric map">'
        '<p class="muted">시야 범위: 연한 파란색. 최근 이동 궤적: 밝은 초록색. 로봇 위치와 방향: 노란 원과 검은 선. VLM refined goal: 파란 링. 기대 goal: 초록 링. 주행 가능 후보: 하늘색 링. 막힘/주행 불가능/실패 goal: 빨간 표시. coarse/current goal: 주황 링.</p>'
        "</div>"
    )


def _case_row(result: VLMDecisionCaseResult) -> str:
    error = result.error_message or ""
    return (
        "<tr>"
        f"<td><a href=\"cases/{html.escape(_safe_path_segment(result.case_id))}/index.html\">"
        f"{html.escape(result.case_id)}</a></td>"
        f"<td>{html.escape(result.category)}</td>"
        f"<td class=\"{_pass_class(result.passed)}\">{'통과' if result.passed else '실패'}</td>"
        f"<td>{_format_bool(result.action_type_correct)}</td>"
        f"<td>{_format_optional_bool(result.waypoint_success)}</td>"
        f"<td>{_format_optional_bool(result.selected_point_navigable)}</td>"
        f"<td>{_format_optional_bool(result.reasoning_correct)}</td>"
        f"<td>{html.escape(error)}</td>"
        "</tr>"
    )


def _category_row(category: str, metrics: Mapping[str, float | None]) -> str:
    return (
        "<tr>"
        f"<td>{html.escape(category)}</td>"
        f"<td>{_format_metric(metrics.get('case_success_rate'))}</td>"
        f"<td>{_format_metric(metrics.get('action_type_accuracy'))}</td>"
        f"<td>{_format_metric(metrics.get('waypoint_success_rate'))}</td>"
        f"<td>{_format_metric(metrics.get('reasoning_correctness_rate'))}</td>"
        f"<td>{_format_metric(metrics.get('selected_point_navigability_rate'))}</td>"
        f"<td>{_format_metric(metrics.get('forbidden_goal_repeat_rate'))}</td>"
        "</tr>"
    )


def _metric_card(label: str, value: float | None) -> str:
    return (
        '<div class="metric">'
        f'<span class="muted">{html.escape(label)}</span>'
        f"<strong>{_format_metric(value)}</strong>"
        "</div>"
    )


def _format_bool(value: bool) -> str:
    css_class = _pass_class(value)
    return f'<span class="{css_class}">{"예" if value else "아니오"}</span>'


def _format_optional_bool(value: bool | None) -> str:
    if value is None:
        return '<span class="muted">해당 없음</span>'
    return _format_bool(value)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "해당 없음"
    return f"{value:.3f}"


def _pass_class(value: bool) -> str:
    return "pass" if value else "fail"


def _safe_path_segment(value: str) -> str:
    safe_value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe_value or "case"


if __name__ == "__main__":
    raise SystemExit(main())
