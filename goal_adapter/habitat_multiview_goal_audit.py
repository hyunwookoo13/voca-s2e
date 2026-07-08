from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image as PilImage

from goal_adapter.habitat_benchmark_dataset import (
    _is_metric_map_point_navigable,
    _pathfinder_map_bounds,
    _point_component,
    _sample_navigable_point,
)
from goal_adapter.habitat_goal_audit import (
    BLACK,
    GRAY,
    RED,
    WHITE,
    YELLOW,
    Color,
    Image,
    _MapTransform,
    _depth_at,
    _draw_cross,
    _draw_filled_circle,
    _draw_line,
    _draw_ring,
    _pathfinder,
    _point_list,
    _position_xy,
    _project_image_point_to_xy,
    _read_png_rgb,
    _relative_bearing_degrees,
    _resize_nearest,
    _rotation_from_heading,
    _sample_far_goal_xy,
    _valid_image_point,
)
from goal_adapter.habitat_sim_smoke import HabitatSimSmokeConfig, _create_simulator, _import_habitat_sim
from goal_adapter.habitat_smoke import write_rgb_png


VIEW_OFFSETS = {
    "front": 0.0,
    "left": -math.pi / 2.0,
    "right": math.pi / 2.0,
    "back": math.pi,
}


@dataclass(frozen=True)
class MultiViewGoalAuditConfig:
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
    max_tokens: int = 256
    max_projection_depth_m: float = 3.0
    waypoint_policy: str = "best_local"
    disable_thinking: bool = False


def run_multiview_goal_audit(config: MultiViewGoalAuditConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    habitat_sim = _import_habitat_sim()
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
        robot_position = config.robot_position or _sample_navigable_point(pathfinder)
        coarse_goal_xy = config.coarse_goal_xy or _sample_far_goal_xy(pathfinder, robot_position)
        views = _capture_views(sim, agent, robot_position, config, output_dir)
        selected_view, selected_point, raw_response = _query_qwen_for_multiview_point(
            views=views,
            config=config,
            robot_position=robot_position,
            coarse_goal_xy=coarse_goal_xy,
        )
        selected_heading = config.heading + VIEW_OFFSETS.get(selected_view or "front", 0.0)
        selected_depth = views[selected_view]["depth"] if selected_view in views else None
        fine_goal_xy = _project_image_point_to_xy(
            selected_point=selected_point,
            depth=selected_depth,
            robot_position=robot_position,
            heading=selected_heading,
            hfov_degrees=config.image_hfov,
            image_width=config.image_width,
            image_height=config.image_height,
            max_depth_m=config.max_projection_depth_m,
            pathfinder=pathfinder,
        )

        for view_name, view in views.items():
            overlay = _read_png_rgb(view["rgb_path"])
            if view_name == selected_view and selected_point is not None:
                _draw_cross(overlay, selected_point, RED, 16)
            overlay_path = output_dir / f"{view_name}_rgb_fine_goal.png"
            write_rgb_png(overlay, overlay_path)
            view["overlay_path"] = overlay_path

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

        audit_path = output_dir / "multiview_goal_audit_layout.png"
        write_rgb_png(_compose_multiview_layout(map_image, views), audit_path)

        result = {
            "scene_path": str(config.scene_path),
            "waypoint_policy": config.waypoint_policy,
            "robot_position_xyz": _point_list(robot_position),
            "robot_xy": _position_xy(robot_position),
            "heading": config.heading,
            "coarse_goal_xy": coarse_goal_xy,
            "relative_goal_bearing_degrees": _relative_bearing_degrees(
                _position_xy(robot_position), coarse_goal_xy, config.heading
            ),
            "selected_view": selected_view,
            "selected_image_point": selected_point,
            "estimated_fine_goal_xy": fine_goal_xy,
            "topdown": str(topdown_path),
            "audit_layout": str(audit_path),
            "views": {
                name: {
                    "relative_heading_degrees": math.degrees(VIEW_OFFSETS[name]),
                    "rgb": str(view["rgb_path"]),
                    "overlay": str(view["overlay_path"]),
                }
                for name, view in views.items()
            },
            "raw_vlm_response": raw_response,
        }
        result_json_path = output_dir / "multiview_goal_audit_result.json"
        result_json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        result["result_json"] = str(result_json_path)
        return result
    finally:
        close = getattr(sim, "close", None)
        if callable(close):
            close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render 4-view Habitat VLM goal audits.")
    parser.add_argument("--scene-root", help="Directory containing Habitat .glb/.basis.glb scenes.")
    parser.add_argument("--scene", help="Single Habitat scene path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", default="qwen3-vl-30b")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=79.0)
    parser.add_argument("--sensor-height", type=float, default=0.88)
    parser.add_argument(
        "--waypoint-policy",
        choices=["best_local", "farthest_visible"],
        default="best_local",
        help="Waypoint selection policy used in the VLM prompt.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass chat_template_kwargs.enable_thinking=false to Qwen thinking models.",
    )
    args = parser.parse_args(argv)

    scene_paths = [Path(args.scene)] if args.scene else _discover_scene_paths(Path(args.scene_root or ""))
    if not scene_paths:
        raise RuntimeError("no Habitat scenes found")
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    layout_paths = []
    for index in range(args.count):
        scene_path = rng.choice(scene_paths)
        heading = rng.uniform(-math.pi, math.pi)
        trial_dir = output_dir / f"trial_{index:03d}_{scene_path.stem}"
        print(f"[{index + 1}/{args.count}] scene={scene_path} heading={heading:.3f}", flush=True)
        result = run_multiview_goal_audit(
            MultiViewGoalAuditConfig(
                scene_path=scene_path,
                output_dir=trial_dir,
                heading=heading,
                image_width=args.width,
                image_height=args.height,
                sensor_height=args.sensor_height,
                image_hfov=args.hfov,
                endpoint=args.endpoint,
                model=args.model,
                max_tokens=args.max_tokens,
                waypoint_policy=args.waypoint_policy,
                disable_thinking=args.disable_thinking,
            )
        )
        records.append(result)
        layout_paths.append(Path(result["audit_layout"]))
    contact_sheet = output_dir / "multiview_contact_sheet.png"
    _write_contact_sheet(layout_paths, contact_sheet, columns=2)
    summary_path = output_dir / "multiview_batch_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "available_scene_count": len(scene_paths),
                "requested_count": args.count,
                "seed": args.seed,
                "contact_sheet": str(contact_sheet),
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote contact_sheet={contact_sheet}")
    print(f"wrote summary={summary_path}")
    return 0


def _capture_views(
    sim: Any,
    agent: Any,
    robot_position: Any,
    config: MultiViewGoalAuditConfig,
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    views = {}
    for view_name, offset in VIEW_OFFSETS.items():
        state = agent.get_state()
        state.position = robot_position
        state.rotation = _rotation_from_heading(config.heading + offset)
        agent.set_state(state)
        observations = sim.get_sensor_observations()
        rgb_path = output_dir / f"{view_name}_rgb.png"
        write_rgb_png(observations["rgb"], rgb_path)
        views[view_name] = {
            "rgb_path": rgb_path,
            "depth": observations.get("depth"),
        }
    return views


def _query_qwen_for_multiview_point(
    views: Mapping[str, Mapping[str, Any]],
    config: MultiViewGoalAuditConfig,
    robot_position: Any,
    coarse_goal_xy: Sequence[float],
) -> tuple[str | None, list[int] | None, str]:
    robot_xy = _position_xy(robot_position)
    prompt = _build_multiview_prompt(config, robot_xy, coarse_goal_xy)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for view_name in ("front", "left", "right", "back"):
        content.append({"type": "text", "text": f"{view_name} view:"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,"
                    + base64.b64encode(Path(views[view_name]["rgb_path"]).read_bytes()).decode("ascii")
                },
            }
        )
    payload = _build_chat_completion_payload(content, config)
    request = urllib.request.Request(
        config.endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        message = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]
    raw = message.get("content") or message.get("reasoning") or ""
    parsed = _extract_json(raw)
    selected_view = parsed.get("selected_view")
    point = parsed.get("selected_image_point")
    if selected_view not in VIEW_OFFSETS:
        selected_view = _extract_selected_view(raw)
    if not _valid_image_point(point, config.image_width, config.image_height):
        point = _extract_selected_image_point(raw)
    if selected_view in VIEW_OFFSETS and _valid_image_point(point, config.image_width, config.image_height):
        return str(selected_view), [int(point[0]), int(point[1])], raw
    return None, None, raw


def _build_chat_completion_payload(
    content: list[dict[str, Any]],
    config: MultiViewGoalAuditConfig,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": config.max_tokens,
        "temperature": 0,
    }
    if config.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def _build_multiview_prompt(
    config: MultiViewGoalAuditConfig,
    robot_xy: Sequence[float],
    coarse_goal_xy: Sequence[float],
) -> str:
    bearing = _relative_bearing_degrees(robot_xy, coarse_goal_xy, config.heading)
    distance = math.dist(robot_xy, [float(coarse_goal_xy[0]), float(coarse_goal_xy[1])])
    policy_instruction = _waypoint_policy_instruction(config.waypoint_policy)
    return (
        "You are given four RGB observations from the same indoor mobile robot pose: "
        "front, left, right, and back.\n"
        "Choose the view and exactly one visible navigable floor point that is the best next local waypoint "
        "toward the hidden coarse GPS goal.\n\n"
        f"Each image size: width={config.image_width}, height={config.image_height}.\n"
        f"Robot map position [x, y]: {robot_xy}.\n"
        f"Robot heading radians for front view: {config.heading:.4f}.\n"
        f"Coarse GPS goal [x, y]: {[round(float(coarse_goal_xy[0]), 3), round(float(coarse_goal_xy[1]), 3)]}.\n"
        f"Relative goal bearing from front view: {bearing:.1f} degrees. Negative is left, positive is right, "
        "near 180/-180 means behind.\n"
        f"Coarse goal distance: {distance:.2f} meters.\n\n"
        "View headings relative to front: front=0, left=-90, right=+90, back=180 degrees.\n"
        "Use the relative goal bearing to pick the most relevant view before selecting a point. "
        f"{policy_instruction}\n"
        "Avoid walls, furniture, object surfaces, and image borders.\n"
        "Return only compact JSON: "
        "{\"selected_view\":\"front|left|right|back\",\"selected_image_point\":[u,v],"
        "\"reasoning\":\"short\",\"confidence\":\"high\"}."
    )


def _waypoint_policy_instruction(waypoint_policy: str) -> str:
    if waypoint_policy == "farthest_visible":
        return (
            "Select the farthest visible navigable floor point that the robot can reasonably drive toward. "
            "Do not choose a nearby floor point if a farther safe floor point exists. "
            "Prefer the far end of visible free space, corridor, doorway, or opening. "
            "In perspective RGB, far floor is usually higher in the image near the floor-wall boundary; "
            "avoid the lower foreground unless all farther floor is blocked. "
            "This point will be used as a local subgoal for S2E/PixelNav, so it should maximize useful progress."
        )
    return "Prefer a safe visible navigable floor point that makes progress toward the coarse goal."


def _metric_map_image_and_transform(
    pathfinder: Any,
    config: MultiViewGoalAuditConfig,
) -> tuple[Image, _MapTransform]:
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
    transform: _MapTransform,
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
    for offset, color, length in ((0.0, BLACK, 30), (-math.pi / 2.0, (107, 114, 128), 18), (math.pi / 2.0, (107, 114, 128), 18), (math.pi, (107, 114, 128), 18)):
        arrow_heading = heading + offset
        arrow = [
            robot_px[0] + int(math.sin(arrow_heading) * length),
            robot_px[1] - int(math.cos(arrow_heading) * length),
        ]
        _draw_line(image, robot_px, arrow, color, thickness=3 if offset == 0.0 else 2)


def _compose_multiview_layout(topdown: Image, views: Mapping[str, Mapping[str, Any]]) -> Image:
    topdown_panel = _resize_nearest(topdown, 640, 960)
    cells = []
    for view_name in ("front", "left", "right", "back"):
        image = _read_png_rgb(Path(views[view_name]["overlay_path"]))
        cells.append(_resize_nearest(image, 320, 240))
    right_top = [cells[0][y] + cells[1][y] for y in range(240)]
    right_bottom = [cells[2][y] + cells[3][y] for y in range(240)]
    right = _resize_nearest(right_top + right_bottom, 640, 960)
    return [topdown_row + right_row for topdown_row, right_row in zip(topdown_panel, right)]


def _discover_scene_paths(scene_root: Path) -> list[Path]:
    scene_paths = []
    for root, _dirs, filenames in os.walk(scene_root, followlinks=True):
        for filename in filenames:
            if filename.endswith(".glb") or filename.endswith(".basis.glb"):
                scene_paths.append(Path(root) / filename)
    return sorted(scene_paths)


def _write_contact_sheet(layout_paths: Sequence[Path], output_path: Path, columns: int) -> None:
    images = [PilImage.open(path).convert("RGB") for path in layout_paths]
    if not images:
        PilImage.new("RGB", (1, 1), "white").save(output_path)
        return
    width, height = images[0].size
    rows = (len(images) + columns - 1) // columns
    sheet = PilImage.new("RGB", (width * columns, height * rows), "white")
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns) * width, (index // columns) * height))
    sheet.save(output_path)


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


def _extract_selected_view(raw: str) -> str | None:
    match = re.search(r'"selected_view"\s*:\s*"([^"]+)"', raw)
    return match.group(1) if match else None


def _extract_selected_image_point(raw: str) -> list[int] | None:
    match = re.search(r'"selected_image_point"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]', raw)
    if not match:
        return None
    return [int(match.group(1)), int(match.group(2))]


if __name__ == "__main__":
    raise SystemExit(main())
