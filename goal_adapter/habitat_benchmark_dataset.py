from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

from goal_adapter.habitat_sim_smoke import _create_simulator, _import_habitat_sim, _pose_from_position
from goal_adapter.habitat_smoke import write_rgb_png


CATEGORIES = ("coarse_gps", "coarse_object_point", "tracking_loss", "deadlock")
NAVIGABLE_MAP_COLOR = [72, 76, 82]
UNNAVIGABLE_MAP_COLOR = [255, 255, 255]


@dataclass(frozen=True)
class HabitatBenchmarkConfig:
    scene_paths: Sequence[str | Path]
    output_dir: str | Path
    cases_per_category: int = 100
    image_width: int = 320
    image_height: int = 240
    sensor_height: float = 0.88
    image_hfov: float = 79.0
    seed: int = 13
    waypoint_tolerance: float = 0.75
    map_meters_per_pixel: float = 0.1
    map_fov_range_meters: float = 3.0


@dataclass(frozen=True)
class HabitatBenchmarkDataset:
    output_dir: Path
    case_paths: list[Path]

    @property
    def total_cases(self) -> int:
        return len(self.case_paths)


def generate_habitat_benchmark_cases(
    config: HabitatBenchmarkConfig,
    habitat_sim_module: Any | None = None,
) -> HabitatBenchmarkDataset:
    if config.cases_per_category < 1:
        raise ValueError("cases_per_category must be >= 1")
    if not config.scene_paths:
        raise ValueError("scene_paths must include at least one scene")
    if config.map_meters_per_pixel <= 0:
        raise ValueError("map_meters_per_pixel must be > 0")
    if config.map_fov_range_meters <= 0:
        raise ValueError("map_fov_range_meters must be > 0")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    habitat_sim = habitat_sim_module or _import_habitat_sim()
    rng = random.Random(config.seed)
    case_paths: list[Path] = []
    case_index = 1

    for scene_index, scene_path in enumerate(config.scene_paths):
        simulator = _create_simulator(_sim_config(config, scene_path, output_dir), habitat_sim)
        try:
            pathfinder = _pathfinder(simulator)
            _seed_pathfinder(pathfinder, config.seed + scene_index)
            metric_map = _write_metric_map(
                pathfinder=pathfinder,
                assets_dir=assets_dir,
                output_dir=output_dir,
                scene_index=scene_index,
                config=config,
            )
            agent = simulator.initialize_agent(0)
            for category in CATEGORIES:
                for category_index in range(config.cases_per_category):
                    case_id = f"{category}_{case_index:04d}"
                    payload = _build_case(
                        case_id=case_id,
                        category=category,
                        scene_id=str(scene_path),
                        category_index=category_index,
                        simulator=simulator,
                        agent=agent,
                        pathfinder=pathfinder,
                        assets_dir=assets_dir,
                        output_dir=output_dir,
                        config=config,
                        rng=rng,
                        metric_map=metric_map,
                    )
                    path = output_dir / f"{case_index:04d}_{case_id}.json"
                    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                    case_paths.append(path)
                    case_index += 1
        finally:
            close = getattr(simulator, "close", None)
            if callable(close):
                close()

    return HabitatBenchmarkDataset(output_dir=output_dir, case_paths=case_paths)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Habitat-Sim pathfinder-labeled VLM benchmark cases.",
    )
    parser.add_argument("--scene", action="append", required=True, help="Habitat scene .glb path. Repeatable.")
    parser.add_argument("--output-dir", required=True, help="Directory where case JSON and assets will be written.")
    parser.add_argument("--cases-per-category", type=int, default=100, help="Cases to generate per category.")
    parser.add_argument("--width", type=int, default=320, help="RGB sensor width.")
    parser.add_argument("--height", type=int, default=240, help="RGB sensor height.")
    parser.add_argument("--sensor-height", type=float, default=0.88, help="RGB sensor height in meters.")
    parser.add_argument("--hfov", type=float, default=79.0, help="RGB sensor horizontal field of view.")
    parser.add_argument("--seed", type=int, default=13, help="Dataset sampling seed.")
    parser.add_argument(
        "--map-meters-per-pixel",
        type=float,
        default=0.1,
        help="Metric top-down map resolution.",
    )
    parser.add_argument(
        "--map-fov-range-meters",
        type=float,
        default=3.0,
        help="Range of the diagnostic top-down camera FOV cone.",
    )
    args = parser.parse_args(argv)

    dataset = generate_habitat_benchmark_cases(
        HabitatBenchmarkConfig(
            scene_paths=args.scene,
            output_dir=args.output_dir,
            cases_per_category=args.cases_per_category,
            image_width=args.width,
            image_height=args.height,
            sensor_height=args.sensor_height,
            image_hfov=args.hfov,
            seed=args.seed,
            map_meters_per_pixel=args.map_meters_per_pixel,
            map_fov_range_meters=args.map_fov_range_meters,
        )
    )
    print(f"Wrote Habitat benchmark dataset to {dataset.output_dir}")
    print(f"total_cases={dataset.total_cases}")
    return 0


def _build_case(
    case_id: str,
    category: str,
    scene_id: str,
    category_index: int,
    simulator: Any,
    agent: Any,
    pathfinder: Any,
    assets_dir: Path,
    output_dir: Path,
    config: HabitatBenchmarkConfig,
    rng: random.Random,
    metric_map: Mapping[str, Any],
) -> dict[str, Any]:
    robot_position = _sample_navigable_point(pathfinder)
    _set_agent_position(agent, robot_position)
    observations = simulator.get_sensor_observations()
    floor_image_points = _floor_image_points_from_depth(observations.get("depth"), config, count=3)
    image_filename = f"{case_id}_rgb.png"
    image_path = assets_dir / image_filename
    write_rgb_png(observations["rgb"], image_path)

    robot_xy = _position_xy(robot_position)
    local_goal = _sample_navigable_xy(pathfinder)
    alternate_goal = _sample_navigable_xy(pathfinder)
    coarse_goal = _sample_navigable_xy(pathfinder)
    heading = _heading_for(category_index, rng)
    base_input = {
        "current_rgb": str(Path("assets") / image_filename),
        "optional_lookaround_images": [],
        "current_pose": _pose_from_position(robot_position),
        "heading": heading,
        "previous_waypoint": None,
        "s2e_status": "unknown",
        "system_prompt": None,
    }
    memory_base = {
        "scene_id": scene_id,
        "robot_xy": robot_xy,
        "generator": "habitat_pathfinder",
        "sample_index": category_index,
        "metric_map": dict(metric_map),
        "camera_fov": {
            "hfov_degrees": config.image_hfov,
            "range_meters": config.map_fov_range_meters,
        },
        "trajectory_xy": _synthetic_trajectory_xy(robot_xy, heading),
    }

    if category == "coarse_gps":
        candidate_waypoints = [
            _candidate(local_goal, floor_image_points[0], True, 0.92, "best local navigable waypoint toward coarse GPS target"),
            _candidate(alternate_goal, floor_image_points[1], True, 0.58, "secondary navigable waypoint"),
        ]
        adapter_input = {
            **base_input,
            "target_type": "gps",
            "high_level_target": {
                "coarse_goal_xy": coarse_goal,
                "instruction": "Refine the coarse GPS target into a nearby local waypoint.",
            },
            "current_goal_xy": coarse_goal,
            "progress_state": "normal",
            "memory_summary": {
                **memory_base,
                "candidate_waypoints": candidate_waypoints,
                "visited_directions": [],
                "blocked_directions": [],
            },
        }
        expected = _expected("NAVIGATE", local_goal, ["navigable"], config)
    elif category == "coarse_object_point":
        object_center = [config.image_width // 2, int(config.image_height * 0.42)]
        object_front_floor = floor_image_points[0]
        candidate_waypoints = [
            _candidate(local_goal, object_front_floor, True, 0.94, "object-front navigable floor point"),
            _candidate(coarse_goal, object_center, False, 0.12, "object visual center is not a floor waypoint"),
        ]
        adapter_input = {
            **base_input,
            "target_type": "object_point",
            "high_level_target": {
                "object_label": "pathfinder_object_proxy",
                "coarse_image_point": object_center,
                "rule": "Select the object-front navigable floor point, not the object center.",
            },
            "current_goal_xy": coarse_goal,
            "progress_state": "normal",
            "memory_summary": {
                **memory_base,
                "candidate_waypoints": candidate_waypoints,
                "object_front_rule": "Use navigable floor in front of the visual object proxy.",
            },
        }
        expected = _expected("NAVIGATE", local_goal, ["floor"], config)
    elif category == "tracking_loss":
        adapter_input = {
            **base_input,
            "target_type": "missing_point",
            "high_level_target": {
                "instruction": "The target point is missing after tracking loss.",
            },
            "current_goal_xy": None,
            "progress_state": "tracking_loss",
            "memory_summary": {
                **memory_base,
                "last_selected_image_point": [config.image_width // 2, int(config.image_height * 0.6)],
                "last_goal_xy": local_goal,
                "trajectory_outcome": "tracking_loss",
                "candidate_waypoints": [],
            },
        }
        expected = _expected("LOOK_AROUND", None, ["tracking_loss"], config)
    elif category == "deadlock":
        failed_goal = local_goal
        candidate_waypoints = [
            _candidate(failed_goal, [config.image_width // 2, int(config.image_height * 0.52)], False, 0.1, "repeats blocked failed goal"),
            _candidate(alternate_goal, floor_image_points[0], True, 0.88, "different navigable waypoint away from blocked direction"),
        ]
        adapter_input = {
            **base_input,
            "target_type": "gps",
            "high_level_target": {
                "coarse_goal_xy": coarse_goal,
                "instruction": "Previous local goal is blocked; choose a different viable waypoint.",
            },
            "current_goal_xy": failed_goal,
            "progress_state": "blocked",
            "s2e_status": "collision",
            "memory_summary": {
                **memory_base,
                "failed_goal_xy": [failed_goal],
                "blocked_directions": ["front"],
                "visited_directions": ["front"],
                "trajectory_outcome": "collision_then_no_progress",
                "candidate_waypoints": candidate_waypoints,
            },
        }
        expected = _expected("RESELECT_GOAL", alternate_goal, ["blocked"], config)
        expected["forbidden_goal_xy"] = [failed_goal]
    else:
        raise ValueError(f"unsupported category: {category}")

    return {
        "case_id": case_id,
        "category": category,
        "input": adapter_input,
        "expected": expected,
    }


def _expected(
    action_type: str,
    goal_xy: list[float] | None,
    reasoning_terms: list[str],
    config: HabitatBenchmarkConfig,
) -> dict[str, Any]:
    return {
        "action_type": action_type,
        "goal_xy": goal_xy,
        "waypoint_tolerance": config.waypoint_tolerance,
        "reasoning_terms": reasoning_terms,
    }


def _candidate(
    goal_xy: list[float],
    image_point: list[int],
    navigable: bool,
    score: float,
    rationale: str,
) -> dict[str, Any]:
    return {
        "goal_xy": goal_xy,
        "image_point": image_point,
        "navigable": navigable,
        "score": score,
        "rationale": rationale,
    }


def _floor_image_point(config: HabitatBenchmarkConfig, width_fraction: float) -> list[int]:
    return [
        int(config.image_width * width_fraction),
        int(config.image_height * 0.82),
    ]


def _floor_image_points_from_depth(
    depth: Any,
    config: HabitatBenchmarkConfig,
    count: int,
) -> list[list[int]]:
    fallback = [
        _floor_image_point(config, 0.72),
        _floor_image_point(config, 0.5),
        _floor_image_point(config, 0.82),
    ]
    image = _depth_to_rows(depth)
    if image is None:
        return fallback[:count]

    height = len(image)
    width = len(image[0]) if image else 0
    if height < 2 or width < 2:
        return fallback[:count]

    min_y = int(height * 0.72)
    max_y = min(height - 1, int(height * 0.92))
    min_x = int(width * 0.08)
    max_x = max(min_x, int(width * 0.92))
    samples: list[tuple[float, int, int]] = []
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            value = _depth_value(image[y][x])
            if value is not None:
                samples.append((value, x, y))

    if not samples:
        return fallback[:count]

    max_depth = max(item[0] for item in samples)
    samples = [item for item in samples if item[0] >= max_depth * 0.8]
    samples.sort(key=lambda item: item[0], reverse=True)
    selected: list[list[int]] = []
    min_separation = max(8, int(width * 0.18))
    for _depth, x, y in samples:
        if all(abs(x - previous[0]) >= min_separation for previous in selected):
            selected.append([x, y])
        if len(selected) >= count:
            return selected

    for point in fallback:
        if all(point != existing for existing in selected):
            selected.append(point)
        if len(selected) >= count:
            break
    return selected[:count]


def _depth_to_rows(depth: Any) -> list[list[Any]] | None:
    if depth is None:
        return None
    if hasattr(depth, "tolist"):
        depth = depth.tolist()
    if not isinstance(depth, list) or not depth:
        return None
    rows = []
    for row in depth:
        if not isinstance(row, list) or not row:
            return None
        rows.append(row)
    return rows


def _depth_value(value: Any) -> float | None:
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    if not isinstance(value, Real) or not math.isfinite(float(value)):
        return None
    value = float(value)
    if value <= 0:
        return None
    return value


def _sample_navigable_point(pathfinder: Any) -> Any:
    get_random_point = getattr(pathfinder, "get_random_navigable_point", None)
    if not callable(get_random_point):
        raise RuntimeError("Habitat pathfinder must provide get_random_navigable_point")
    point = get_random_point()
    snap_point = getattr(pathfinder, "snap_point", None)
    if callable(snap_point):
        point = snap_point(point)
    return point


def _write_metric_map(
    pathfinder: Any,
    assets_dir: Path,
    output_dir: Path,
    scene_index: int,
    config: HabitatBenchmarkConfig,
) -> dict[str, Any]:
    min_x, max_x, min_y, max_y = _pathfinder_map_bounds(pathfinder)
    meters_per_pixel = config.map_meters_per_pixel
    width = max(2, int(math.ceil((max_x - min_x) / meters_per_pixel)) + 1)
    height = max(2, int(math.ceil((max_y - min_y) / meters_per_pixel)) + 1)
    floor_height = _point_component(_sample_navigable_point(pathfinder), 1)

    image = []
    for row_index in range(height):
        z = max_y - (row_index * meters_per_pixel)
        row = []
        for column_index in range(width):
            x = min_x + (column_index * meters_per_pixel)
            point = [x, floor_height, z]
            if _is_metric_map_point_navigable(pathfinder, point, meters_per_pixel):
                row.append(list(NAVIGABLE_MAP_COLOR))
            else:
                row.append(list(UNNAVIGABLE_MAP_COLOR))
        image.append(row)

    path = assets_dir / f"scene_{scene_index:03d}_metric_map.png"
    write_rgb_png(image, path)
    return {
        "path": str(path.relative_to(output_dir)),
        "meters_per_pixel": meters_per_pixel,
        "floor_height": round(floor_height, 3),
        "bounds": {
            "min_x": round(min_x, 3),
            "max_x": round(max_x, 3),
            "min_y": round(min_y, 3),
            "max_y": round(max_y, 3),
        },
    }


def _pathfinder_map_bounds(pathfinder: Any) -> tuple[float, float, float, float]:
    get_bounds = getattr(pathfinder, "get_bounds", None)
    if callable(get_bounds):
        bounds = get_bounds()
        if isinstance(bounds, Sequence) and len(bounds) == 2:
            lower, upper = bounds
            min_x = _point_component(lower, 0)
            max_x = _point_component(upper, 0)
            min_y = _point_component(lower, 2)
            max_y = _point_component(upper, 2)
            if min_x > max_x:
                min_x, max_x = max_x, min_x
            if min_y > max_y:
                min_y, max_y = max_y, min_y
            if not math.isclose(min_x, max_x) and not math.isclose(min_y, max_y):
                return (min_x, max_x, min_y, max_y)

    center = _sample_navigable_point(pathfinder)
    center_x = _point_component(center, 0)
    center_y = _point_component(center, 2)
    return (center_x - 10.0, center_x + 10.0, center_y - 10.0, center_y + 10.0)


def _is_metric_map_point_navigable(pathfinder: Any, point: Sequence[float], meters_per_pixel: float) -> bool:
    snap_point = getattr(pathfinder, "snap_point", None)
    snapped = snap_point(point) if callable(snap_point) else point
    if snapped is None:
        return False
    is_navigable = getattr(pathfinder, "is_navigable", None)
    if callable(is_navigable) and not bool(is_navigable(snapped)):
        return False
    return _horizontal_distance(point, snapped) <= max(meters_per_pixel * 1.5, 0.05)


def _horizontal_distance(left: Sequence[float], right: Any) -> float:
    dx = float(left[0]) - _point_component(right, 0)
    dz = float(left[2]) - _point_component(right, 2)
    return math.sqrt(dx * dx + dz * dz)


def _point_component(point: Any, index: int) -> float:
    if hasattr(point, "tolist"):
        point = point.tolist()
    try:
        component = point[index]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError("Habitat point must be a three-element coordinate") from exc
    except AttributeError as exc:
        raise ValueError("Habitat point must be a three-element coordinate") from exc
    if not isinstance(component, Real) or not math.isfinite(float(component)):
        raise ValueError("Habitat point components must be finite numbers")
    return float(component)


def _sample_navigable_xy(pathfinder: Any) -> list[float]:
    return _position_xy(_sample_navigable_point(pathfinder))


def _position_xy(position: Any) -> list[float]:
    if hasattr(position, "tolist"):
        position = position.tolist()
    return [round(float(position[0]), 3), round(float(position[2]), 3)]


def _synthetic_trajectory_xy(robot_xy: Sequence[float], heading: float) -> list[list[float]]:
    return [
        _offset_xy(robot_xy, heading + math.pi, 0.8),
        _offset_xy(robot_xy, heading + math.pi, 0.4),
        [round(float(robot_xy[0]), 3), round(float(robot_xy[1]), 3)],
    ]


def _offset_xy(origin_xy: Sequence[float], heading: float, distance: float) -> list[float]:
    return [
        round(float(origin_xy[0]) + math.sin(heading) * distance, 3),
        round(float(origin_xy[1]) + math.cos(heading) * distance, 3),
    ]


def _set_agent_position(agent: Any, position: Any) -> None:
    state = agent.get_state()
    state.position = position
    agent.set_state(state)


def _heading_for(index: int, rng: random.Random) -> float:
    return round(rng.uniform(-3.14159, 3.14159) + (index * 0.017), 6)


def _pathfinder(simulator: Any) -> Any:
    pathfinder = getattr(simulator, "pathfinder", None)
    if pathfinder is None or not getattr(pathfinder, "is_loaded", True):
        raise RuntimeError("Habitat simulator pathfinder is not loaded")
    return pathfinder


def _seed_pathfinder(pathfinder: Any, seed: int) -> None:
    seed_fn = getattr(pathfinder, "seed", None)
    if callable(seed_fn):
        seed_fn(seed)


def _sim_config(config: HabitatBenchmarkConfig, scene_path: str | Path, output_dir: Path) -> Any:
    from goal_adapter.habitat_sim_smoke import HabitatSimSmokeConfig

    return HabitatSimSmokeConfig(
        scene_path=scene_path,
        output_dir=output_dir,
        image_width=config.image_width,
        image_height=config.image_height,
        sensor_height=config.sensor_height,
        image_hfov=config.image_hfov,
        include_depth=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
