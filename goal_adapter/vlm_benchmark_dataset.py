from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from goal_adapter.schema import GoalAdapterInput


def write_sample_vlm_benchmark_cases(
    base_input_path: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    base_input = GoalAdapterInput.from_json(_read_json_object(base_input_path)).to_json()
    image_path = _copy_base_image(base_input, assets_dir)
    base_input["current_rgb"] = str(Path("assets") / image_path.name)
    base_input["optional_lookaround_images"] = []

    cases = _sample_cases(base_input)
    written_paths = []
    for index, case in enumerate(cases, start=1):
        path = output_dir / f"{index:02d}_{case['case_id']}.json"
        path.write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written_paths.append(path)
    return written_paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create four S2E-free VLM benchmark cases from one Habitat smoke GoalAdapter input.",
    )
    parser.add_argument(
        "--base-input",
        required=True,
        help="Path to a GoalAdapter input JSON that includes current_rgb.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("reports") / "vlm_benchmark" / "cases_smoke"),
        help="Directory where case JSON files and assets will be written.",
    )
    args = parser.parse_args(argv)

    written_paths = write_sample_vlm_benchmark_cases(args.base_input, args.output_dir)
    print(f"Wrote {len(written_paths)} VLM benchmark cases to {Path(args.output_dir)}")
    for path in written_paths:
        print(path)
    return 0


def _sample_cases(base_input: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": "coarse_gps_open_area",
            "category": "coarse_gps",
            "input": _case_input(
                base_input,
                target_type="gps",
                high_level_target={
                    "coarse_goal_xy": [11.0, -6.4],
                    "instruction": "Refine the coarse GPS target into a nearby open local waypoint.",
                },
                current_goal_xy=[11.0, -6.4],
                memory_summary={
                    "candidate_waypoints": [
                        {
                            "goal_xy": [10.5, -6.1],
                            "image_point": [82, 62],
                            "navigable": True,
                            "score": 0.92,
                            "rationale": "open floor ahead toward the coarse GPS target",
                        },
                        {
                            "goal_xy": [9.6, -7.5],
                            "image_point": [18, 78],
                            "navigable": False,
                            "score": 0.22,
                            "rationale": "side area is visually narrow",
                        },
                    ],
                    "visited_directions": ["back"],
                    "blocked_directions": [],
                },
            ),
            "expected": {
                "action_type": "NAVIGATE",
                "goal_xy": [10.5, -6.1],
                "waypoint_tolerance": 0.75,
                "reasoning_terms": ["open"],
            },
        },
        {
            "case_id": "coarse_object_front_floor",
            "category": "coarse_object_point",
            "input": _case_input(
                base_input,
                target_type="object_point",
                high_level_target={
                    "object_label": "target object",
                    "coarse_image_point": [88, 48],
                    "rule": "Select a navigable floor point in front of the object, not the object center.",
                },
                current_goal_xy=[10.8, -6.3],
                memory_summary={
                    "candidate_waypoints": [
                        {
                            "goal_xy": [10.5, -6.1],
                            "image_point": [88, 70],
                            "navigable": True,
                            "score": 0.9,
                            "rationale": "floor point in front of the coarse object point",
                        },
                        {
                            "goal_xy": [10.9, -6.4],
                            "image_point": [88, 48],
                            "navigable": False,
                            "score": 0.18,
                            "rationale": "visual object center is not a floor waypoint",
                        },
                    ],
                    "object_front_rule": "Use navigable floor around the object, not the object center.",
                },
            ),
            "expected": {
                "action_type": "NAVIGATE",
                "goal_xy": [10.5, -6.1],
                "waypoint_tolerance": 0.75,
                "reasoning_terms": ["floor"],
            },
        },
        {
            "case_id": "tracking_loss_lookaround",
            "category": "tracking_loss",
            "input": _case_input(
                base_input,
                target_type="missing_point",
                high_level_target={
                    "instruction": "The target point is currently missing after tracking loss.",
                },
                current_goal_xy=None,
                progress_state="tracking_loss",
                memory_summary={
                    "last_selected_image_point": [84, 60],
                    "last_goal_xy": [10.5, -6.1],
                    "trajectory_outcome": "tracking_loss",
                    "candidate_waypoints": [],
                },
            ),
            "expected": {
                "action_type": "LOOK_AROUND",
                "goal_xy": None,
                "waypoint_tolerance": 0.75,
                "reasoning_terms": ["tracking_loss"],
            },
        },
        {
            "case_id": "deadlock_reselect_goal",
            "category": "deadlock",
            "input": _case_input(
                base_input,
                target_type="gps",
                high_level_target={
                    "coarse_goal_xy": [11.0, -6.4],
                    "instruction": "Previous local goal is blocked; choose a different viable waypoint.",
                },
                current_goal_xy=[10.5, -6.1],
                progress_state="blocked",
                s2e_status="collision",
                memory_summary={
                    "failed_goal_xy": [[10.5, -6.1]],
                    "blocked_directions": ["front"],
                    "visited_directions": ["front", "front-right"],
                    "trajectory_outcome": "collision_then_no_progress",
                    "candidate_waypoints": [
                        {
                            "goal_xy": [10.5, -6.1],
                            "image_point": [82, 62],
                            "navigable": False,
                            "score": 0.12,
                            "rationale": "repeats blocked goal",
                        },
                        {
                            "goal_xy": [11.25, -5.35],
                            "image_point": [126, 52],
                            "navigable": True,
                            "score": 0.84,
                            "rationale": "different open direction that avoids the blocked path",
                        },
                    ],
                },
            ),
            "expected": {
                "action_type": "RESELECT_GOAL",
                "goal_xy": [11.25, -5.35],
                "forbidden_goal_xy": [[10.5, -6.1]],
                "waypoint_tolerance": 0.75,
                "reasoning_terms": ["blocked"],
            },
        },
    ]


def _case_input(
    base_input: Mapping[str, Any],
    target_type: str,
    high_level_target: Any,
    current_goal_xy: list[float] | None,
    memory_summary: Mapping[str, Any],
    progress_state: str = "normal",
    s2e_status: str = "unknown",
) -> dict[str, Any]:
    payload = dict(base_input)
    payload.update(
        {
            "target_type": target_type,
            "high_level_target": high_level_target,
            "current_goal_xy": current_goal_xy,
            "progress_state": progress_state,
            "s2e_status": s2e_status,
            "memory_summary": dict(memory_summary),
        }
    )
    return payload


def _copy_base_image(base_input: Mapping[str, Any], assets_dir: Path) -> Path:
    current_rgb = base_input.get("current_rgb")
    if not isinstance(current_rgb, str) or not current_rgb:
        raise ValueError("base input must include current_rgb")
    source = Path(current_rgb)
    if not source.exists():
        raise ValueError(f"base current_rgb image does not exist: {source}")
    suffix = source.suffix if source.suffix else ".png"
    destination = assets_dir / f"current_rgb{suffix}"
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    return destination


def _read_json_object(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
