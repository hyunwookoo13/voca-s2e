from __future__ import annotations

import argparse
import base64
import json
import math
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from goal_adapter.habitat_benchmark_dataset import (
    HabitatBenchmarkConfig,
    _is_metric_map_point_navigable,
    _pathfinder_map_bounds,
    _point_component,
    _sample_navigable_point,
)
from goal_adapter.habitat_sim_smoke import HabitatSimSmokeConfig, _create_simulator, _import_habitat_sim
from goal_adapter.habitat_smoke import write_rgb_png
from goal_adapter.visualization import _read_png_rgb


Image = list[list[list[int]]]
Color = tuple[int, int, int]

BLACK: Color = (17, 24, 39)
BLUE: Color = (37, 99, 235)
GRAY: Color = (72, 76, 82)
LIGHT_GRAY: Color = (238, 242, 247)
RED: Color = (220, 38, 38)
WHITE: Color = (255, 255, 255)
YELLOW: Color = (250, 204, 21)


@dataclass(frozen=True)
class HabitatGoalAuditConfig:
    scene_path: str | Path
    output_dir: str | Path
    coarse_goal_xy: list[float] | None = None
    robot_position: list[float] | None = None
    heading: float = 0.0
    image_width: int = 640
    image_height: int = 480
    sensor_height: float = 0.88
    image_hfov: float = 79.0
    map_meters_per_pixel: float = 0.1
    endpoint: str = "http://localhost:8000/v1/chat/completions"
    model: str = "qwen3-vl-30b"
    max_projection_depth_m: float = 3.0


@dataclass(frozen=True)
class HabitatGoalAuditResult:
    output_dir: Path
    current_rgb_path: Path
    rgb_overlay_path: Path
    topdown_path: Path
    audit_path: Path
    result_json_path: Path
    selected_image_point: list[int] | None
    coarse_goal_xy: list[float]
    fine_goal_xy: list[float] | None


def run_habitat_goal_audit(
    config: HabitatGoalAuditConfig,
    habitat_sim_module: Any | None = None,
) -> HabitatGoalAuditResult:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    habitat_sim = habitat_sim_module or _import_habitat_sim()
    sim = _create_simulator(
        HabitatSimSmokeConfig(
            scene_path=config.scene_path,
            output_dir=output_dir,
            image_width=config.image_width,
            image_height=config.image_height,
            sensor_height=config.sensor_height,
            image_hfov=config.image_hfov,
            include_depth=True,
        ),
        habitat_sim,
    )
    try:
        agent = sim.initialize_agent(0)
        pathfinder = _pathfinder(sim)
        robot_position = _place_agent(agent, pathfinder, config)
        observations = sim.get_sensor_observations()
        current_rgb_path = output_dir / "current_rgb.png"
        write_rgb_png(observations["rgb"], current_rgb_path)

        coarse_goal_xy = config.coarse_goal_xy or _sample_far_goal_xy(pathfinder, robot_position)
        selected_point, raw_response = _query_qwen_for_point(
            image_path=current_rgb_path,
            config=config,
            robot_position=robot_position,
            coarse_goal_xy=coarse_goal_xy,
        )
        fine_goal_xy = _project_image_point_to_xy(
            selected_point=selected_point,
            depth=observations.get("depth"),
            robot_position=robot_position,
            heading=config.heading,
            hfov_degrees=config.image_hfov,
            image_width=config.image_width,
            image_height=config.image_height,
            max_depth_m=config.max_projection_depth_m,
            pathfinder=pathfinder,
        )

        map_image, transform = _metric_map_image_and_transform(pathfinder, config)
        _draw_topdown_layers(
            map_image,
            transform,
            robot_xy=_position_xy(robot_position),
            heading=config.heading,
            coarse_goal_xy=coarse_goal_xy,
            fine_goal_xy=fine_goal_xy,
        )
        topdown_path = output_dir / "topdown_goal_audit.png"
        write_rgb_png(map_image, topdown_path)

        rgb_overlay = _read_png_rgb(current_rgb_path)
        if selected_point is not None:
            _draw_cross(rgb_overlay, selected_point, RED, 16)
        rgb_overlay_path = output_dir / "current_rgb_fine_goal.png"
        write_rgb_png(rgb_overlay, rgb_overlay_path)

        audit_path = output_dir / "goal_audit_layout.png"
        write_rgb_png(_compose_layout(map_image, _read_png_rgb(current_rgb_path), rgb_overlay), audit_path)

        result_json_path = output_dir / "goal_audit_result.json"
        result_payload = {
            "scene_path": str(config.scene_path),
            "current_rgb": str(current_rgb_path),
            "topdown": str(topdown_path),
            "rgb_overlay": str(rgb_overlay_path),
            "audit_layout": str(audit_path),
            "robot_position_xyz": _point_list(robot_position),
            "robot_xy": _position_xy(robot_position),
            "heading": config.heading,
            "coarse_goal_xy": coarse_goal_xy,
            "selected_image_point": selected_point,
            "estimated_fine_goal_xy": fine_goal_xy,
            "raw_vlm_response": raw_response,
        }
        result_json_path.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    finally:
        close = getattr(sim, "close", None)
        if callable(close):
            close()

    return HabitatGoalAuditResult(
        output_dir=output_dir,
        current_rgb_path=current_rgb_path,
        rgb_overlay_path=rgb_overlay_path,
        topdown_path=topdown_path,
        audit_path=audit_path,
        result_json_path=result_json_path,
        selected_image_point=selected_point,
        coarse_goal_xy=coarse_goal_xy,
        fine_goal_xy=fine_goal_xy,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a Habitat coarse-to-fine VLM goal audit image.")
    parser.add_argument("--scene", required=True, help="Habitat/HM3D scene .glb path.")
    parser.add_argument("--output-dir", required=True, help="Directory where audit images are written.")
    parser.add_argument("--coarse-goal-x", type=float, default=None)
    parser.add_argument("--coarse-goal-y", type=float, default=None, help="Map z/y coordinate paired with x.")
    parser.add_argument("--robot-x", type=float, default=None)
    parser.add_argument("--robot-y", type=float, default=None, help="Habitat vertical y coordinate.")
    parser.add_argument("--robot-z", type=float, default=None)
    parser.add_argument("--heading", type=float, default=0.0, help="Robot heading in radians; 0 means +map-y/z.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=79.0)
    parser.add_argument("--sensor-height", type=float, default=0.88)
    parser.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", default="qwen3-vl-30b")
    args = parser.parse_args(argv)

    coarse_goal_xy = None
    if args.coarse_goal_x is not None or args.coarse_goal_y is not None:
        if args.coarse_goal_x is None or args.coarse_goal_y is None:
            parser.error("--coarse-goal-x and --coarse-goal-y must be provided together")
        coarse_goal_xy = [args.coarse_goal_x, args.coarse_goal_y]

    robot_position = None
    if args.robot_x is not None or args.robot_y is not None or args.robot_z is not None:
        if args.robot_x is None or args.robot_y is None or args.robot_z is None:
            parser.error("--robot-x, --robot-y, and --robot-z must be provided together")
        robot_position = [args.robot_x, args.robot_y, args.robot_z]

    result = run_habitat_goal_audit(
        HabitatGoalAuditConfig(
            scene_path=args.scene,
            output_dir=args.output_dir,
            coarse_goal_xy=coarse_goal_xy,
            robot_position=robot_position,
            heading=args.heading,
            image_width=args.width,
            image_height=args.height,
            sensor_height=args.sensor_height,
            image_hfov=args.hfov,
            endpoint=args.endpoint,
            model=args.model,
        )
    )
    print(f"wrote audit={result.audit_path}")
    print(f"selected_image_point={result.selected_image_point}")
    print(f"coarse_goal_xy={result.coarse_goal_xy}")
    print(f"fine_goal_xy={result.fine_goal_xy}")
    return 0


def _query_qwen_for_point(
    image_path: Path,
    config: HabitatGoalAuditConfig,
    robot_position: Any,
    coarse_goal_xy: Sequence[float],
) -> tuple[list[int] | None, str]:
    robot_xy = _position_xy(robot_position)
    bearing = _relative_bearing_degrees(robot_xy, coarse_goal_xy, config.heading)
    distance = math.dist(robot_xy, [float(coarse_goal_xy[0]), float(coarse_goal_xy[1])])
    prompt = (
        "You are given the current RGB observation of an indoor mobile robot in Habitat-Sim.\n"
        "The robot has a coarse GPS goal that is not necessarily visible in the image.\n"
        "Your task is not to click the final hidden goal. Select exactly one visible navigable floor point "
        "in the current image that is the best next local waypoint toward the coarse GPS goal.\n\n"
        f"Image size: width={config.image_width}, height={config.image_height}.\n"
        f"Robot map position [x, y]: {robot_xy}.\n"
        f"Robot heading radians: {config.heading:.4f}.\n"
        f"Coarse GPS goal [x, y]: {[round(float(coarse_goal_xy[0]), 3), round(float(coarse_goal_xy[1]), 3)]}.\n"
        f"Relative goal bearing in robot frame: {bearing:.1f} degrees. Negative is left, positive is right.\n"
        f"Coarse goal distance: {distance:.2f} meters.\n\n"
        f"Coordinate constraints are mandatory: u must be 0..{config.image_width - 1}, "
        f"v must be 0..{config.image_height - 1}. Coordinates must refer to the current RGB image only.\n"
        "Avoid walls, furniture, object surfaces, doors, image borders, and points too close to the robot.\n"
        "Prefer open visible floor that leads toward the relative bearing direction.\n\n"
        "Return only valid compact JSON exactly like: "
        "{\"selected_image_point\":[u,v],\"reasoning\":\"floor toward goal\",\"confidence\":\"high\"}. "
        "Keep reasoning under 8 words."
    )
    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(image_path.read_bytes()).decode("ascii")
                        },
                    },
                ],
            }
        ],
        "max_tokens": 256,
        "temperature": 0,
    }
    request = urllib.request.Request(
        config.endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        raw = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"]
    parsed = _extract_json(raw)
    point = parsed.get("selected_image_point")
    if _valid_image_point(point, config.image_width, config.image_height):
        return [int(point[0]), int(point[1])], raw
    point = _extract_selected_image_point(raw)
    if _valid_image_point(point, config.image_width, config.image_height):
        return [int(point[0]), int(point[1])], raw
    return None, raw


def _project_image_point_to_xy(
    selected_point: Sequence[int] | None,
    depth: Any,
    robot_position: Any,
    heading: float,
    hfov_degrees: float,
    image_width: int,
    image_height: int,
    max_depth_m: float,
    pathfinder: Any,
) -> list[float] | None:
    if selected_point is None:
        return None
    depth_value = _depth_at(depth, selected_point)
    if depth_value is None:
        depth_value = max_depth_m
    depth_value = min(depth_value, max_depth_m)

    u, _v = int(selected_point[0]), int(selected_point[1])
    normalized_x = ((u + 0.5) / image_width) - 0.5
    relative_angle = normalized_x * math.radians(hfov_degrees)
    world_heading = heading + relative_angle
    robot_xy = _position_xy(robot_position)
    target = [
        robot_xy[0] + math.sin(world_heading) * depth_value,
        robot_xy[1] + math.cos(world_heading) * depth_value,
    ]
    floor_y = _point_component(robot_position, 1)
    snap_point = getattr(pathfinder, "snap_point", None)
    if callable(snap_point):
        snapped = snap_point([target[0], floor_y, target[1]])
        if snapped is not None and _is_metric_map_point_navigable(pathfinder, snapped, 0.1):
            return _position_xy(snapped)
    return [round(target[0], 3), round(target[1], 3)]


def _metric_map_image_and_transform(
    pathfinder: Any,
    config: HabitatGoalAuditConfig,
) -> tuple[Image, "_MapTransform"]:
    min_x, max_x, min_y, max_y = _pathfinder_map_bounds(pathfinder)
    meters_per_pixel = config.map_meters_per_pixel
    width = max(2, int(math.ceil((max_x - min_x) / meters_per_pixel)) + 1)
    height = max(2, int(math.ceil((max_y - min_y) / meters_per_pixel)) + 1)
    floor_height = _point_component(_sample_navigable_point(pathfinder), 1)
    image: Image = []
    for row_index in range(height):
        z = max_y - (row_index * meters_per_pixel)
        row = []
        for column_index in range(width):
            x = min_x + (column_index * meters_per_pixel)
            point = [x, floor_height, z]
            row.append(list(GRAY if _is_metric_map_point_navigable(pathfinder, point, meters_per_pixel) else WHITE))
        image.append(row)
    return image, _MapTransform(bounds=(min_x, max_x, min_y, max_y), width=width, height=height)


def _draw_topdown_layers(
    image: Image,
    transform: "_MapTransform",
    robot_xy: Sequence[float],
    heading: float,
    coarse_goal_xy: Sequence[float],
    fine_goal_xy: Sequence[float] | None,
) -> None:
    _draw_ring(image, transform.to_pixel(coarse_goal_xy), YELLOW, 10)
    if fine_goal_xy is not None:
        _draw_ring(image, transform.to_pixel(fine_goal_xy), RED, 9)
    robot_px = transform.to_pixel(robot_xy)
    _draw_filled_circle(image, robot_px, BLACK, 8)
    arrow = [
        robot_px[0] + int(math.sin(heading) * 28),
        robot_px[1] - int(math.cos(heading) * 28),
    ]
    _draw_line(image, robot_px, arrow, BLACK, thickness=3)


def _compose_layout(topdown: Image, rgb: Image, rgb_overlay: Image) -> Image:
    topdown_panel = _resize_nearest(topdown, 640, 960)
    rgb_panel = _resize_nearest(rgb, 640, 480)
    overlay_panel = _resize_nearest(rgb_overlay, 640, 480)
    right = rgb_panel + overlay_panel
    return [
        topdown_row + right_row
        for topdown_row, right_row in zip(topdown_panel, right)
    ]


def _place_agent(agent: Any, pathfinder: Any, config: HabitatGoalAuditConfig) -> Any:
    position = config.robot_position or _sample_navigable_point(pathfinder)
    state = agent.get_state()
    state.position = position
    state.rotation = _rotation_from_heading(config.heading)
    agent.set_state(state)
    return position


def _rotation_from_heading(heading: float) -> Any:
    import numpy as np
    from habitat_sim.utils import common

    return common.quat_from_angle_axis(math.pi + heading, np.array([0.0, 1.0, 0.0]))


def _sample_far_goal_xy(pathfinder: Any, robot_position: Any) -> list[float]:
    robot_xy = _position_xy(robot_position)
    best = None
    best_distance = -1.0
    for _ in range(32):
        point = _sample_navigable_point(pathfinder)
        xy = _position_xy(point)
        distance = math.dist(robot_xy, xy)
        if distance > best_distance:
            best = xy
            best_distance = distance
    if best is None:
        raise RuntimeError("failed to sample coarse goal")
    return best


def _pathfinder(sim: Any) -> Any:
    pathfinder = getattr(sim, "pathfinder", None)
    if pathfinder is None or not getattr(pathfinder, "is_loaded", True):
        raise RuntimeError("Habitat simulator pathfinder is not loaded")
    return pathfinder


@dataclass(frozen=True)
class _MapTransform:
    bounds: tuple[float, float, float, float]
    width: int
    height: int

    def to_pixel(self, xy: Sequence[float]) -> list[int]:
        min_x, max_x, min_y, max_y = self.bounds
        x = (float(xy[0]) - min_x) / max(1e-9, max_x - min_x) * (self.width - 1)
        y = self.height - 1 - ((float(xy[1]) - min_y) / max(1e-9, max_y - min_y) * (self.height - 1))
        return [int(round(x)), int(round(y))]


def _extract_json(raw: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, Mapping) else {}
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            payload = json.loads(raw[start : end + 1])
            return payload if isinstance(payload, Mapping) else {}
    return {}


def _extract_selected_image_point(raw: str) -> list[int] | None:
    match = re.search(r'"selected_image_point"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]', raw)
    if not match:
        return None
    return [int(match.group(1)), int(match.group(2))]


def _relative_bearing_degrees(robot_xy: Sequence[float], goal_xy: Sequence[float], heading: float) -> float:
    dx = float(goal_xy[0]) - float(robot_xy[0])
    dy = float(goal_xy[1]) - float(robot_xy[1])
    goal_heading = math.atan2(dx, dy)
    return math.degrees(_wrap_angle(goal_heading - heading))


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= math.tau
    while angle < -math.pi:
        angle += math.tau
    return angle


def _depth_at(depth: Any, point: Sequence[int]) -> float | None:
    if depth is None:
        return None
    if hasattr(depth, "tolist"):
        depth = depth.tolist()
    if not isinstance(depth, Sequence) or not depth:
        return None
    x, y = int(point[0]), int(point[1])
    if y < 0 or y >= len(depth):
        return None
    row = depth[y]
    if not isinstance(row, Sequence) or x < 0 or x >= len(row):
        return None
    value = row[x]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        value = value[0] if value else None
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
        return None
    return float(value)


def _valid_image_point(point: Any, width: int, height: int) -> bool:
    return (
        isinstance(point, Sequence)
        and not isinstance(point, (str, bytes, bytearray))
        and len(point) == 2
        and all(isinstance(component, int) for component in point)
        and 0 <= int(point[0]) < width
        and 0 <= int(point[1]) < height
    )


def _position_xy(position: Any) -> list[float]:
    return [round(_point_component(position, 0), 3), round(_point_component(position, 2), 3)]


def _point_list(position: Any) -> list[float]:
    return [
        round(_point_component(position, 0), 3),
        round(_point_component(position, 1), 3),
        round(_point_component(position, 2), 3),
    ]


def _resize_nearest(image: Image, width: int, height: int) -> Image:
    src_h = len(image)
    src_w = len(image[0])
    output = []
    for y in range(height):
        src_y = min(src_h - 1, int(y * src_h / height))
        row = []
        for x in range(width):
            src_x = min(src_w - 1, int(x * src_w / width))
            row.append(list(image[src_y][src_x][:3]))
        output.append(row)
    return output


def _draw_ring(image: Image, center: Sequence[int], color: Color, radius: int) -> None:
    _draw_filled_circle(image, center, WHITE, radius + 2)
    _draw_filled_circle(image, center, color, radius)
    _draw_filled_circle(image, center, WHITE, max(1, radius - 4))


def _draw_cross(image: Image, center: Sequence[int], color: Color, radius: int) -> None:
    _draw_ring(image, center, color, radius)
    _draw_line(image, [center[0] - radius, center[1]], [center[0] + radius, center[1]], color, thickness=3)
    _draw_line(image, [center[0], center[1] - radius], [center[0], center[1] + radius], color, thickness=3)


def _draw_filled_circle(image: Image, center: Sequence[int], color: Color, radius: int) -> None:
    cx, cy = int(center[0]), int(center[1])
    radius_sq = radius * radius
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius_sq:
                _set_pixel(image, x, y, color)


def _draw_line(image: Image, start: Sequence[int], end: Sequence[int], color: Color, thickness: int = 1) -> None:
    x0, y0 = int(start[0]), int(start[1])
    x1, y1 = int(end[0]), int(end[1])
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        for oy in range(-(thickness // 2), thickness // 2 + 1):
            for ox in range(-(thickness // 2), thickness // 2 + 1):
                _set_pixel(image, x0 + ox, y0 + oy, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _set_pixel(image: Image, x: int, y: int, color: Color) -> None:
    if 0 <= y < len(image) and 0 <= x < len(image[0]):
        image[y][x] = [color[0], color[1], color[2]]


if __name__ == "__main__":
    raise SystemExit(main())
