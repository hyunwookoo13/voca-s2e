from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from goal_adapter.habitat_smoke import PNG_SIGNATURE, write_rgb_png


Color = tuple[int, int, int]
Image = list[list[list[int]]]

BLUE: Color = (37, 99, 235)
GREEN: Color = (22, 163, 74)
ORANGE: Color = (217, 119, 6)
RED: Color = (220, 38, 38)
GRAY: Color = (107, 114, 128)
BLACK: Color = (17, 24, 39)
WHITE: Color = (255, 255, 255)
BACKGROUND: Color = (247, 249, 252)
GRID: Color = (219, 226, 236)
FOV_COLOR: Color = (191, 219, 254)
TRAJECTORY_COLOR: Color = (34, 197, 94)
NAVIGABLE_CANDIDATE_COLOR: Color = (14, 165, 233)
BLOCKED_CANDIDATE_COLOR: Color = (239, 68, 68)
ROBOT_COLOR: Color = (250, 204, 21)


@dataclass(frozen=True)
class CaseVisualizationArtifacts:
    rgb_overlay_path: Path
    topdown_overlay_path: Path


@dataclass(frozen=True)
class _WorldMarker:
    xy: list[float]
    color: Color
    radius: int


def render_case_visualizations(case: Any, output: Any, case_dir: str | Path) -> CaseVisualizationArtifacts:
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    rgb_overlay_path = case_dir / "current_rgb_overlay.png"
    topdown_overlay_path = case_dir / "topdown_overlay.png"

    rgb_image = _read_png_rgb(Path(case.input_json["current_rgb"]))
    _draw_rgb_markers(rgb_image, case, output)
    write_rgb_png(rgb_image, rgb_overlay_path)

    topdown_image, transform, draw_grid = _topdown_canvas(case, case_dir)
    _draw_topdown(topdown_image, case, output, transform, draw_grid)
    write_rgb_png(topdown_image, topdown_overlay_path)

    return CaseVisualizationArtifacts(
        rgb_overlay_path=rgb_overlay_path,
        topdown_overlay_path=topdown_overlay_path,
    )


def _draw_rgb_markers(image: Image, case: Any, output: Any) -> None:
    high_level_target = _mapping(case.input_json.get("high_level_target", {}))
    coarse_image_point = high_level_target.get("coarse_image_point")
    if _valid_image_point(coarse_image_point):
        _draw_ring(image, [int(coarse_image_point[0]), int(coarse_image_point[1])], ORANGE, radius=6)

    expected_image_point = _expected_image_point(case)
    if expected_image_point is not None:
        _draw_ring(image, expected_image_point, GREEN, radius=6)

    selected_image_point = getattr(output, "selected_image_point", None)
    if _valid_image_point(selected_image_point):
        _draw_cross(image, [int(selected_image_point[0]), int(selected_image_point[1])], BLUE, radius=7)


def _topdown_canvas(case: Any, case_dir: Path) -> tuple[Image, "_WorldTransform | None", bool]:
    metric_map = _metric_map(case, case_dir)
    if metric_map is not None:
        image, bounds = metric_map
        return (
            image,
            _WorldTransform(
                bounds=bounds,
                width=len(image[0]),
                height=len(image),
                margin=0,
                preserve_aspect=False,
            ),
            False,
        )
    return _blank_image(420, 420, BACKGROUND), None, True


def _metric_map(case: Any, case_dir: Path) -> tuple[Image, tuple[float, float, float, float]] | None:
    memory_summary = _mapping(case.input_json.get("memory_summary", {}))
    metric_map = _mapping(memory_summary.get("metric_map", {}))
    path_value = metric_map.get("path")
    bounds_value = _mapping(metric_map.get("bounds", {}))
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute() and not path.exists():
        path = case_dir / path
    if not path.exists():
        return None
    bounds = _metric_map_bounds(bounds_value)
    if bounds is None:
        return None
    return _read_png_rgb(path), bounds


def _metric_map_bounds(bounds: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    values = [
        bounds.get("min_x"),
        bounds.get("max_x"),
        bounds.get("min_y"),
        bounds.get("max_y"),
    ]
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
        return None
    min_x, max_x, min_y, max_y = [float(value) for value in values]
    if math.isclose(min_x, max_x) or math.isclose(min_y, max_y):
        return None
    if min_x > max_x:
        min_x, max_x = max_x, min_x
    if min_y > max_y:
        min_y, max_y = max_y, min_y
    return (min_x, max_x, min_y, max_y)


def _draw_topdown(
    image: Image,
    case: Any,
    output: Any,
    transform: "_WorldTransform | None" = None,
    draw_grid: bool = True,
) -> None:
    if draw_grid:
        _draw_grid(image, step=42)
    markers = _world_markers(case, output)
    if transform is None:
        bounds = _world_bounds(markers, case)
        transform = _WorldTransform(bounds=bounds, width=len(image[0]), height=len(image), margin=36)

    _draw_fov_cone(image, case, transform)
    _draw_trajectory(image, case, transform)
    _draw_candidate_waypoints(image, case, transform)

    robot_xy = _robot_xy(case)
    if robot_xy is not None:
        robot_px = transform.to_pixel(robot_xy)
        _draw_filled_circle(image, robot_px, BLACK, 8)
        _draw_filled_circle(image, robot_px, ROBOT_COLOR, 6)
        heading = case.input_json.get("heading")
        if isinstance(heading, (int, float)):
            arrow_end = [
                robot_px[0] + int(math.sin(float(heading)) * 30),
                robot_px[1] - int(math.cos(float(heading)) * 30),
            ]
            _draw_line(image, robot_px, arrow_end, BLACK, thickness=2)
            _draw_filled_circle(image, arrow_end, BLACK, 3)

    for marker in markers:
        _draw_ring(image, transform.to_pixel(marker.xy), marker.color, marker.radius)


def _draw_fov_cone(image: Image, case: Any, transform: "_WorldTransform") -> None:
    robot_xy = _robot_xy(case)
    if robot_xy is None:
        return
    heading = case.input_json.get("heading")
    if not isinstance(heading, (int, float)):
        return
    fov = _mapping(_memory_summary(case).get("camera_fov", {}))
    hfov_degrees = fov.get("hfov_degrees", 79.0)
    range_meters = fov.get("range_meters", 3.0)
    if not isinstance(hfov_degrees, (int, float)) or not isinstance(range_meters, (int, float)):
        return
    if range_meters <= 0:
        return

    half_fov = math.radians(float(hfov_degrees)) / 2.0
    heading = float(heading)
    points = [
        robot_xy,
        _project_heading(robot_xy, heading - half_fov, float(range_meters)),
        _project_heading(robot_xy, heading + half_fov, float(range_meters)),
    ]
    pixels = [transform.to_pixel(point) for point in points]
    _draw_filled_polygon(image, pixels, FOV_COLOR)
    _draw_line(image, pixels[0], pixels[1], FOV_COLOR, thickness=1)
    _draw_line(image, pixels[0], pixels[2], FOV_COLOR, thickness=1)


def _project_heading(origin_xy: Sequence[float], heading: float, distance: float) -> list[float]:
    return [
        float(origin_xy[0]) + math.sin(heading) * distance,
        float(origin_xy[1]) + math.cos(heading) * distance,
    ]


def _draw_trajectory(image: Image, case: Any, transform: "_WorldTransform") -> None:
    trajectory = _trajectory_xy(case)
    if len(trajectory) < 2:
        return
    pixels = [transform.to_pixel(point) for point in trajectory]
    for start, end in zip(pixels, pixels[1:]):
        _draw_line(image, start, end, TRAJECTORY_COLOR, thickness=2)


def _trajectory_xy(case: Any) -> list[list[float]]:
    trajectory = _memory_summary(case).get("trajectory_xy", [])
    if not isinstance(trajectory, list):
        return []
    points = [_xy(point) for point in trajectory if _valid_xy(point)]
    robot_xy = _robot_xy(case)
    if robot_xy is not None and (not points or not _same_xy(points[-1], robot_xy, 0.001)):
        points.append(robot_xy)
    return points


def _draw_candidate_waypoints(image: Image, case: Any, transform: "_WorldTransform") -> None:
    for candidate in _candidate_waypoints(case):
        goal_xy = candidate.get("goal_xy")
        if not _valid_xy(goal_xy):
            continue
        pixel = transform.to_pixel(_xy(goal_xy))
        navigable = candidate.get("navigable")
        if navigable is False:
            _draw_cross(image, pixel, BLOCKED_CANDIDATE_COLOR, radius=7)
        elif navigable is True:
            _draw_filled_circle(image, pixel, NAVIGABLE_CANDIDATE_COLOR, 13)
            _draw_filled_circle(image, pixel, WHITE, 8)
        else:
            _draw_filled_circle(image, pixel, GRAY, 5)


def _world_markers(case: Any, output: Any) -> list[_WorldMarker]:
    markers: list[_WorldMarker] = []
    high_level_target = _mapping(case.input_json.get("high_level_target", {}))
    coarse_goal_xy = high_level_target.get("coarse_goal_xy")
    if _valid_xy(coarse_goal_xy):
        markers.append(_WorldMarker(_xy(coarse_goal_xy), ORANGE, 8))

    current_goal_xy = case.input_json.get("current_goal_xy")
    if _valid_xy(current_goal_xy):
        markers.append(_WorldMarker(_xy(current_goal_xy), ORANGE, 6))

    expected_goal_xy = getattr(case, "expected_goal_xy", None)
    if _valid_xy(expected_goal_xy):
        markers.append(_WorldMarker(_xy(expected_goal_xy), GREEN, 8))

    refined_goal_xy = getattr(output, "refined_goal_xy", None)
    if _valid_xy(refined_goal_xy):
        markers.append(_WorldMarker(_xy(refined_goal_xy), BLUE, 8))

    for failed_goal_xy in _failed_goal_xy(case):
        markers.append(_WorldMarker(failed_goal_xy, RED, 7))
    return markers


def _world_bounds(markers: Sequence[_WorldMarker], case: Any) -> tuple[float, float, float, float]:
    points = [marker.xy for marker in markers]
    robot_xy = _robot_xy(case)
    if robot_xy is not None:
        points.append(robot_xy)
    for candidate in _candidate_waypoints(case):
        goal_xy = candidate.get("goal_xy")
        if _valid_xy(goal_xy):
            points.append(_xy(goal_xy))
    if not points:
        points = [[0.0, 0.0], [1.0, 1.0]]

    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    if math.isclose(min_x, max_x):
        min_x -= 1.0
        max_x += 1.0
    if math.isclose(min_y, max_y):
        min_y -= 1.0
        max_y += 1.0
    return (min_x, max_x, min_y, max_y)


@dataclass(frozen=True)
class _WorldTransform:
    bounds: tuple[float, float, float, float]
    width: int
    height: int
    margin: int
    preserve_aspect: bool = True

    def to_pixel(self, xy: Sequence[float]) -> list[int]:
        min_x, max_x, min_y, max_y = self.bounds
        usable_width = max(1, self.width - (self.margin * 2) - 1)
        usable_height = max(1, self.height - (self.margin * 2) - 1)
        x_scale = usable_width / (max_x - min_x)
        y_scale = usable_height / (max_y - min_y)
        if self.preserve_aspect:
            scale = min(x_scale, y_scale)
            x_scale = scale
            y_scale = scale
        x = self.margin + (float(xy[0]) - min_x) * x_scale
        y = self.height - 1 - self.margin - (float(xy[1]) - min_y) * y_scale
        return [int(round(x)), int(round(y))]


def _expected_image_point(case: Any) -> list[int] | None:
    expected_goal_xy = getattr(case, "expected_goal_xy", None)
    if not _valid_xy(expected_goal_xy):
        return None
    for candidate in _candidate_waypoints(case):
        if _same_xy(candidate.get("goal_xy"), expected_goal_xy, getattr(case, "waypoint_tolerance", 0.75)):
            image_point = candidate.get("image_point")
            if _valid_image_point(image_point):
                return [int(image_point[0]), int(image_point[1])]
    return None


def _candidate_waypoints(case: Any) -> list[Mapping[str, Any]]:
    memory_summary = _memory_summary(case)
    candidates = memory_summary.get("candidate_waypoints", [])
    if not isinstance(candidates, list):
        return []
    return [candidate for candidate in candidates if isinstance(candidate, Mapping)]


def _memory_summary(case: Any) -> Mapping[str, Any]:
    return _mapping(case.input_json.get("memory_summary", {}))


def _failed_goal_xy(case: Any) -> list[list[float]]:
    failed: list[list[float]] = []
    for goal_xy in getattr(case, "forbidden_goal_xy", []):
        if _valid_xy(goal_xy):
            failed.append(_xy(goal_xy))
    memory_summary = _mapping(case.input_json.get("memory_summary", {}))
    for key in ("failed_goal_xy", "failed"):
        value = memory_summary.get(key, [])
        if isinstance(value, list):
            for goal_xy in value:
                if _valid_xy(goal_xy):
                    failed.append(_xy(goal_xy))
    return failed


def _robot_xy(case: Any) -> list[float] | None:
    pose = _mapping(case.input_json.get("current_pose", {}))
    if isinstance(pose.get("x"), (int, float)) and isinstance(pose.get("y"), (int, float)):
        return [float(pose["x"]), float(pose["y"])]
    return None


def _draw_grid(image: Image, step: int) -> None:
    height = len(image)
    width = len(image[0])
    for x in range(0, width, step):
        _draw_line(image, [x, 0], [x, height - 1], GRID)
    for y in range(0, height, step):
        _draw_line(image, [0, y], [width - 1, y], GRID)


def _draw_ring(image: Image, center: Sequence[int], color: Color, radius: int) -> None:
    _draw_filled_circle(image, center, WHITE, radius + 2)
    _draw_filled_circle(image, center, color, radius)
    _draw_filled_circle(image, center, WHITE, max(1, radius - 3))


def _draw_cross(image: Image, center: Sequence[int], color: Color, radius: int) -> None:
    _draw_filled_circle(image, center, WHITE, radius + 2)
    _draw_line(image, [center[0] - radius, center[1]], [center[0] + radius, center[1]], color, thickness=2)
    _draw_line(image, [center[0], center[1] - radius], [center[0], center[1] + radius], color, thickness=2)
    _draw_filled_circle(image, center, color, 3)


def _draw_filled_circle(image: Image, center: Sequence[int], color: Color, radius: int) -> None:
    cx, cy = int(center[0]), int(center[1])
    radius_sq = radius * radius
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius_sq:
                _set_pixel(image, x, y, color)


def _draw_filled_polygon(image: Image, points: Sequence[Sequence[int]], color: Color) -> None:
    if len(points) < 3:
        return
    min_x = max(0, min(int(point[0]) for point in points))
    max_x = min(len(image[0]) - 1, max(int(point[0]) for point in points))
    min_y = max(0, min(int(point[1]) for point in points))
    max_y = min(len(image) - 1, max(int(point[1]) for point in points))
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if _point_in_polygon(x, y, points):
                _set_pixel(image, x, y, color)


def _point_in_polygon(x: int, y: int, points: Sequence[Sequence[int]]) -> bool:
    inside = False
    previous = points[-1]
    for current in points:
        x1, y1 = int(previous[0]), int(previous[1])
        x2, y2 = int(current[0]), int(current[1])
        crosses = (y1 > y) != (y2 > y)
        if crosses:
            slope_x = (x2 - x1) * (y - y1) / max(1e-9, y2 - y1) + x1
            if x < slope_x:
                inside = not inside
        previous = current
    return inside


def _draw_line(
    image: Image,
    start: Sequence[int],
    end: Sequence[int],
    color: Color,
    thickness: int = 1,
) -> None:
    x0, y0 = int(start[0]), int(start[1])
    x1, y1 = int(end[0]), int(end[1])
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        for offset_y in range(-(thickness // 2), thickness // 2 + 1):
            for offset_x in range(-(thickness // 2), thickness // 2 + 1):
                _set_pixel(image, x0 + offset_x, y0 + offset_y, color)
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
    if y < 0 or y >= len(image) or x < 0 or x >= len(image[0]):
        return
    image[y][x] = [color[0], color[1], color[2]]


def _blank_image(width: int, height: int, color: Color) -> Image:
    return [[[color[0], color[1], color[2]] for _x in range(width)] for _y in range(height)]


def _read_png_rgb(path: Path) -> Image:
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"not a PNG image: {path}")
    chunks = _png_chunks(data[len(PNG_SIGNATURE) :])
    ihdr = chunks.get(b"IHDR")
    if ihdr is None:
        raise ValueError(f"PNG missing IHDR: {path}")
    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack("!IIBBBBB", ihdr)
    if bit_depth != 8 or compression != 0 or filter_method != 0 or interlace != 0:
        raise ValueError("only non-interlaced 8-bit PNG images are supported")
    if color_type not in {2, 6}:
        raise ValueError("only RGB/RGBA PNG images are supported")
    channels = 3 if color_type == 2 else 4
    compressed = b"".join(value for key, value in _png_chunk_sequence(data[len(PNG_SIGNATURE) :]) if key == b"IDAT")
    raw = zlib.decompress(compressed)
    stride = width * channels
    rows: list[bytes] = []
    previous = bytes(stride)
    cursor = 0
    for _row in range(height):
        filter_type = raw[cursor]
        cursor += 1
        scanline = bytearray(raw[cursor : cursor + stride])
        cursor += stride
        recon = _unfilter_scanline(filter_type, scanline, previous, channels)
        rows.append(bytes(recon))
        previous = bytes(recon)

    image: Image = []
    for row in rows:
        pixels = []
        for index in range(0, len(row), channels):
            pixels.append([int(row[index]), int(row[index + 1]), int(row[index + 2])])
        image.append(pixels)
    return image


def _unfilter_scanline(filter_type: int, scanline: bytearray, previous: bytes, bpp: int) -> bytearray:
    recon = bytearray(scanline)
    for index, value in enumerate(scanline):
        left = recon[index - bpp] if index >= bpp else 0
        up = previous[index] if previous else 0
        up_left = previous[index - bpp] if previous and index >= bpp else 0
        if filter_type == 0:
            recon[index] = value
        elif filter_type == 1:
            recon[index] = (value + left) & 0xFF
        elif filter_type == 2:
            recon[index] = (value + up) & 0xFF
        elif filter_type == 3:
            recon[index] = (value + ((left + up) // 2)) & 0xFF
        elif filter_type == 4:
            recon[index] = (value + _paeth(left, up, up_left)) & 0xFF
        else:
            raise ValueError(f"unsupported PNG filter type: {filter_type}")
    return recon


def _paeth(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    pa = abs(estimate - left)
    pb = abs(estimate - up)
    pc = abs(estimate - up_left)
    if pa <= pb and pa <= pc:
        return left
    if pb <= pc:
        return up
    return up_left


def _png_chunks(data: bytes) -> dict[bytes, bytes]:
    return {key: value for key, value in _png_chunk_sequence(data)}


def _png_chunk_sequence(data: bytes) -> Iterable[tuple[bytes, bytes]]:
    cursor = 0
    while cursor < len(data):
        if cursor + 8 > len(data):
            break
        length = struct.unpack("!I", data[cursor : cursor + 4])[0]
        cursor += 4
        chunk_type = data[cursor : cursor + 4]
        cursor += 4
        chunk_data = data[cursor : cursor + length]
        cursor += length + 4
        yield chunk_type, chunk_data
        if chunk_type == b"IEND":
            break


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _valid_image_point(value: Any) -> bool:
    return (
        not isinstance(value, (str, bytes, bytearray))
        and isinstance(value, Sequence)
        and len(value) == 2
        and all(isinstance(component, int) for component in value)
    )


def _valid_xy(value: Any) -> bool:
    return (
        not isinstance(value, (str, bytes, bytearray))
        and isinstance(value, Sequence)
        and len(value) == 2
        and all(isinstance(component, (int, float)) and math.isfinite(float(component)) for component in value)
    )


def _xy(value: Sequence[float]) -> list[float]:
    return [float(value[0]), float(value[1])]


def _same_xy(left: Any, right: Any, tolerance: float) -> bool:
    if not _valid_xy(left) or not _valid_xy(right):
        return False
    dx = float(left[0]) - float(right[0])
    dy = float(left[1]) - float(right[1])
    return (dx * dx + dy * dy) <= tolerance * tolerance
