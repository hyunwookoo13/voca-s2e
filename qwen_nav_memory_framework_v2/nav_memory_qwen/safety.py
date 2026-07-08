"""Validation and safety wrappers for VLM outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math

from .schema import (
    ALLOWED_ACTIONS,
    ALLOWED_OBSERVATION_MODES,
    ALLOWED_STEP_DEG,
    ALLOWED_VIEWS,
    make_go_output,
    make_observation_request_output,
    make_rotate_output,
    normalize_angle_deg,
)


class VLMOutputError(ValueError):
    """Raised when the VLM output cannot be safely interpreted."""


def _point_ok(point: Any, width: int, height: int) -> bool:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return False
    try:
        u, v = int(point[0]), int(point[1])
    except Exception:
        return False
    return 0 <= u < width and 0 <= v < height


def _valid_view(view_type: Any, view_id: Any, available_views: List[Dict[str, Any]]) -> bool:
    view_types = {v.get("view_type") for v in available_views}
    view_ids = {v.get("view_id") for v in available_views}
    return view_type in view_types and view_id in view_ids


def sanitize_vlm_output(output: Dict[str, Any], vlm_input: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Validate a model output and repair with conservative fallbacks.

    Returns:
        (safe_output, warnings)
    """
    warnings: List[str] = []
    if not isinstance(output, dict):
        raise VLMOutputError("VLM output is not a JSON object")

    obs = vlm_input.get("observation", {})
    width = int(obs.get("image_width", 640) or 640)
    height = int(obs.get("image_height", 480) or 480)
    views = list(obs.get("views", []) or [])
    action = output.get("action")

    if action not in ALLOWED_ACTIONS:
        warnings.append(f"invalid action {action!r}; fallback to observation request")
        goal_bearing = float(vlm_input.get("task", {}).get("coarse_goal", {}).get("relative_bearing_deg", 0.0) or 0.0)
        return make_observation_request_output(
            mode="directed_sweep",
            center_yaw_deg=goal_bearing,
            step_deg=45,
            num_views=3,
            yaw_offsets_deg=[normalize_angle_deg(goal_bearing - 45), goal_bearing, normalize_angle_deg(goal_bearing + 45)],
            reason="invalid_model_action_need_scan",
            confidence="low",
        ), warnings

    if action == "go":
        point = output.get("selected_image_point") or output.get("fine_goal", {}).get("point_px")
        view_type = output.get("selected_view_type") or output.get("fine_goal", {}).get("view_type")
        view_id = output.get("selected_view_id")
        if view_id is None:
            view_id = output.get("fine_goal", {}).get("view_id")
        if not _point_ok(point, width, height):
            warnings.append("invalid selected_image_point; fallback rotate")
            return make_rotate_output(45, reason="R02_NO_VISIBLE_NAVIGABLE_FLOOR", confidence="low"), warnings
        if not _valid_view(view_type, view_id, views):
            warnings.append("selected view not in current observation; using first available view")
            first = views[0] if views else {"view_id": 0, "view_type": "front"}
            view_id = int(first.get("view_id", 0))
            view_type = str(first.get("view_type", "front"))
        u, v = int(point[0]), int(point[1])
        safe = make_go_output(
            view_id=int(view_id),
            view_type=str(view_type),
            point_px=(u, v),
            width=width,
            height=height,
            decision_reason=str(output.get("reasoning", {}).get("decision_reason", "G02_VISIBLE_FLOOR_TOWARD_GOAL")),
            goal_reason=str(output.get("reasoning", {}).get("goal_reason", "F02_VISIBLE_FLOOR_TOWARD_GOAL")),
            short_text=str(output.get("reasoning", {}).get("short_text", "sanitized go output")),
            confidence=str(output.get("confidence", "medium")),
        )
        if "memory_ops" in output:
            safe["memory_ops"] = output["memory_ops"]
        return safe, warnings

    if action == "rotate":
        yaw = output.get("control", {}).get("rotate_yaw_deg", output.get("rotate_yaw_deg", 45))
        try:
            yaw = float(yaw)
        except Exception:
            yaw = 45.0
        if abs(yaw) < 1e-3:
            yaw = 45.0
        yaw = max(-180.0, min(180.0, yaw))
        safe = make_rotate_output(yaw, reason=str(output.get("reasoning", {}).get("decision_reason", "R02_NO_VISIBLE_NAVIGABLE_FLOOR")), confidence=str(output.get("confidence", "medium")))
        if "memory_ops" in output:
            safe["memory_ops"] = output["memory_ops"]
        return safe, warnings

    if action == "request_observation":
        req = output.get("observation_request") or {}
        mode = req.get("mode") or "directed_sweep"
        if mode not in ALLOWED_OBSERVATION_MODES:
            warnings.append("invalid observation mode; fallback directed_sweep")
            mode = "directed_sweep"
        step = int(req.get("step_deg") or 45)
        if step not in ALLOWED_STEP_DEG:
            step = 45
        center = float(req.get("center_yaw_deg", req.get("center_heading_deg", 0.0)) or 0.0)
        if mode == "full_sweep":
            offsets = [-180, -135, -90, -45, 0, 45, 90, 135]
        else:
            num = int(req.get("num_views") or 3)
            num = max(1, min(num, 8))
            if "yaw_offsets_deg" in req and isinstance(req.get("yaw_offsets_deg"), list):
                offsets = [float(x) for x in req["yaw_offsets_deg"]][:8]
            elif num == 1:
                offsets = [center]
            else:
                half = (num - 1) / 2.0
                offsets = [normalize_angle_deg(center + (i - half) * step) for i in range(num)]
        safe = make_observation_request_output(
            mode=mode,
            center_yaw_deg=center,
            step_deg=step,
            num_views=len(offsets),
            yaw_offsets_deg=offsets,
            reason=str(req.get("reason") or output.get("reasoning", {}).get("short_text", "model requested more context")),
            confidence=str(output.get("confidence", "medium")),
        )
        if "memory_ops" in output:
            safe["memory_ops"] = output["memory_ops"]
        return safe, warnings

    # stop is allowed. Caller decides whether it means success or safety stop.
    if action == "stop":
        if "schema_version" not in output:
            output["schema_version"] = "nav_vlm_waypoint_v1"
        return output, warnings

    raise VLMOutputError(f"unhandled action: {action}")
