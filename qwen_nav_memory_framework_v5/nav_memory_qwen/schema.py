"""Schema helpers for ``nav_vlm_waypoint_v1``.

The framework deliberately preserves the user's original VLM envelope:

Input fields:
    task, coordinate_frame, robot_state, observation, memory, constraints

Output fields:
    schema_version, action, selected_view_id/type, selected_image_point,
    fine_goal, observation_request, reasoning, control, confidence

The only extension is that ``memory`` may contain richer graph-derived context,
and VLM outputs may optionally include ``memory_ops``. If ``memory_ops`` is
absent, the backend derives memory updates from action outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import time


ALLOWED_ACTIONS = ["go", "rotate", "stop", "request_observation"]
ALLOWED_OBSERVATION_MODES = ["current_only", "directed_view", "directed_sweep", "full_sweep"]
ALLOWED_VIEWS = ["front", "left", "right", "back"]
ALLOWED_STEP_DEG = [30, 45, 60, 90]


def now_ms() -> int:
    """Return current wall-clock time in milliseconds."""
    return int(time.time() * 1000)


def normalize_angle_deg(angle: float) -> float:
    """Normalize an angle to [-180, 180)."""
    return ((float(angle) + 180.0) % 360.0) - 180.0


def normalize_angle_rad(angle: float) -> float:
    """Normalize an angle to [-pi, pi)."""
    return ((float(angle) + math.pi) % (2.0 * math.pi)) - math.pi


def view_type_to_heading_deg(view_type: str) -> float:
    """Map a semantic view name to robot-relative yaw degrees."""
    mapping = {"front": 0.0, "left": -90.0, "right": 90.0, "back": 180.0}
    return mapping.get(view_type, 0.0)


def nearest_view_type(yaw_deg: float) -> str:
    """Return the closest allowed view type for a robot-relative yaw."""
    yaw = normalize_angle_deg(yaw_deg)
    candidates = [(abs(normalize_angle_deg(yaw - view_type_to_heading_deg(v))), v) for v in ALLOWED_VIEWS]
    return min(candidates, key=lambda x: x[0])[1]


@dataclass(frozen=True)
class RelativePose2D:
    """Relative SE(2) pose stored on graph edges.

    ``dx_m`` and ``dy_m`` are expressed in the source node's local frame.
    ``dyaw_deg`` rotates from source heading to destination heading.
    """

    dx_m: float = 0.0
    dy_m: float = 0.0
    dyaw_deg: float = 0.0
    covariance_diag: Tuple[float, float, float] = (0.05, 0.05, 4.0)

    def compose(self, other: "RelativePose2D") -> "RelativePose2D":
        """Compose this transform with another transform.

        The result is equivalent to applying ``self`` followed by ``other``.
        """
        theta = math.radians(self.dyaw_deg)
        c, s = math.cos(theta), math.sin(theta)
        dx = self.dx_m + c * other.dx_m - s * other.dy_m
        dy = self.dy_m + s * other.dx_m + c * other.dy_m
        yaw = normalize_angle_deg(self.dyaw_deg + other.dyaw_deg)
        cov = tuple(a + b for a, b in zip(self.covariance_diag, other.covariance_diag))
        return RelativePose2D(dx, dy, yaw, cov)  # type: ignore[arg-type]

    def inverse(self) -> "RelativePose2D":
        """Return the inverse relative transform."""
        theta = math.radians(self.dyaw_deg)
        c, s = math.cos(theta), math.sin(theta)
        # inverse of R,t is R^T,-R^T t
        dx = -(c * self.dx_m + s * self.dy_m)
        dy = -(-s * self.dx_m + c * self.dy_m)
        return RelativePose2D(dx, dy, normalize_angle_deg(-self.dyaw_deg), self.covariance_diag)

    def distance_m(self) -> float:
        return math.hypot(self.dx_m, self.dy_m)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dx_m": round(float(self.dx_m), 4),
            "dy_m": round(float(self.dy_m), 4),
            "dyaw_deg": round(float(normalize_angle_deg(self.dyaw_deg)), 3),
            "covariance_diag": [round(float(x), 6) for x in self.covariance_diag],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RelativePose2D":
        cov = tuple(float(x) for x in d.get("covariance_diag", (0.05, 0.05, 4.0)))
        if len(cov) != 3:
            cov = (0.05, 0.05, 4.0)
        return cls(float(d.get("dx_m", 0.0)), float(d.get("dy_m", 0.0)), float(d.get("dyaw_deg", 0.0)), cov)  # type: ignore[arg-type]


@dataclass
class RobotState:
    """Robot state required by the original schema."""

    map_xy: Tuple[float, float]
    heading_rad: float
    position_xyz: Optional[Tuple[float, float, float]] = None

    def to_schema(self) -> Dict[str, Any]:
        x, y = self.map_xy
        pos = self.position_xyz if self.position_xyz is not None else (x, 0.0, y)
        return {
            "position_xyz": [round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4)],
            "map_xy": [round(float(x), 4), round(float(y), 4)],
            "heading_rad": round(float(normalize_angle_rad(self.heading_rad)), 6),
        }


def relative_pose_between_robot_states(src: "RobotState", dst: "RobotState") -> RelativePose2D:
    """Return the live relative pose from a node anchor robot state to current robot state.

    The node itself does not store a global pose. The caller may keep a transient
    robot-state anchor for the latest/current node and use this helper to compute
    ``T_latest_node_to_current_robot`` from odometry/GPS/simulator pose at runtime.
    ``dx_m`` and ``dy_m`` are expressed in the anchor/latest-node local frame.
    """
    sx, sy = src.map_xy
    dxw = float(dst.map_xy[0]) - float(sx)
    dyw = float(dst.map_xy[1]) - float(sy)
    theta = float(src.heading_rad)
    c, st = math.cos(theta), math.sin(theta)
    # Rotate world delta into the source/anchor robot frame: R(-theta) * delta.
    dx = c * dxw + st * dyw
    dy = -st * dxw + c * dyw
    dyaw = normalize_angle_deg(math.degrees(float(dst.heading_rad) - float(src.heading_rad)))
    return RelativePose2D(dx_m=dx, dy_m=dy, dyaw_deg=dyaw)


@dataclass
class CoarseGoal:
    """Coarse PointNav or ObjNav goal."""

    task_mode: str = "PointNav"
    goal_type: str = "gps"
    map_xy: Optional[Tuple[float, float]] = None
    relative_bearing_deg: float = 0.0
    distance_m: float = 0.0
    instruction: str = "move toward the coarse GPS goal"
    target_object: Optional[str] = None
    camera_relation: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_object_goal(cls, target_object: str, robot_state: RobotState, instruction: Optional[str] = None) -> "CoarseGoal":
        """Build an ObjNav-style coarse goal without a metric goal point.

        The VLM must use RGB observations, object beliefs, and memory context to
        select observation requests or local fine waypoints. ``distance_m`` is set
        to a large sentinel because no GPS goal is available.
        """
        return cls(
            task_mode="ObjNav",
            goal_type="object",
            map_xy=None,
            relative_bearing_deg=0.0,
            distance_m=999.0,
            instruction=instruction or f"find the target object: {target_object}",
            target_object=str(target_object),
            camera_relation={
                "in_front_camera_frame": False,
                "bearing_in_fov": False,
                "estimated_visible": False,
                "preprocess_method": "object_goal_has_no_metric_bearing",
            },
        )

    @classmethod
    def from_map_goal(cls, goal_map_xy: Tuple[float, float], robot_state: RobotState) -> "CoarseGoal":
        """Compute bearing and distance from a map-aligned goal and robot state."""
        rx, ry = robot_state.map_xy
        gx, gy = goal_map_xy
        dx, dy = gx - rx, gy - ry
        distance = math.hypot(dx, dy)
        world_bearing_rad = math.atan2(dy, dx)
        rel_bearing = normalize_angle_deg(math.degrees(world_bearing_rad - robot_state.heading_rad))
        in_front = abs(rel_bearing) <= 90.0
        bearing_in_fov = abs(rel_bearing) <= 45.0
        return cls(
            task_mode="PointNav",
            goal_type="gps",
            map_xy=(float(gx), float(gy)),
            relative_bearing_deg=rel_bearing,
            distance_m=distance,
            instruction="move toward the coarse GPS goal",
            target_object=None,
            camera_relation={
                "in_front_camera_frame": bool(in_front),
                "bearing_in_fov": bool(bearing_in_fov),
                "estimated_visible": bool(bearing_in_fov and distance < 5.0),
                "preprocess_method": "transform_goal_to_camera_frame",
            },
        )

    def to_schema(self) -> Dict[str, Any]:
        coarse_goal: Dict[str, Any] = {
            "type": self.goal_type,
            "map_xy": list(self.map_xy) if self.map_xy is not None else None,
            "relative_bearing_deg": round(float(self.relative_bearing_deg), 3),
            "distance_m": round(float(self.distance_m), 3),
            "camera_relation": self.camera_relation or {
                "in_front_camera_frame": abs(self.relative_bearing_deg) <= 90,
                "bearing_in_fov": abs(self.relative_bearing_deg) <= 45,
                "estimated_visible": False,
                "preprocess_method": "transform_goal_to_camera_frame",
            },
        }
        return {
            "task_mode": self.task_mode,
            "instruction": self.instruction,
            "target_object": self.target_object,
            "coarse_goal": coarse_goal,
        }


@dataclass
class ObservationView:
    """One RGB view provided to the VLM."""

    view_id: int
    view_type: str
    relative_heading_deg: float
    image: str  # file path, data URI, or backend-specific handle
    timestamp_ms: int = field(default_factory=now_ms)

    def to_schema(self, placeholder: Optional[str] = None) -> Dict[str, Any]:
        return {
            "view_id": int(self.view_id),
            "view_type": self.view_type,
            "timestamp_ms": int(self.timestamp_ms),
            "relative_heading_deg": round(float(self.relative_heading_deg), 3),
            "image": placeholder if placeholder is not None else self.image,
        }


@dataclass
class Observation:
    """Observation block for the VLM input schema."""

    mode: str
    sequence_id: str
    frame_index: int
    image_width: int
    image_height: int
    views: List[ObservationView]
    timestamp_ms: int = field(default_factory=now_ms)

    def to_schema(self, image_placeholders: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
        image_placeholders = image_placeholders or {}
        return {
            "mode": self.mode,
            "sequence_id": self.sequence_id,
            "timestamp_ms": int(self.timestamp_ms),
            "frame_index": int(self.frame_index),
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "views": [v.to_schema(image_placeholders.get(v.view_id)) for v in self.views],
        }


def default_constraints() -> Dict[str, Any]:
    """Return the constraints block from the original v1 schema."""
    return {
        "allowed_actions": list(ALLOWED_ACTIONS),
        "allowed_observation_modes": list(ALLOWED_OBSERVATION_MODES),
        "allowed_views": list(ALLOWED_VIEWS),
        "allowed_step_deg": list(ALLOWED_STEP_DEG),
        "pixel_coordinate_rule": "u in [0,width-1], v in [0,height-1]",
        "must_select_visible_navigable_floor": True,
    }


def build_vlm_input_v1(
    *,
    task: CoarseGoal,
    robot_state: RobotState,
    observation: Observation,
    memory: Dict[str, Any],
    pose_noise_enabled: bool = True,
    pose_noise_xy_std_m: float = 0.10,
    pose_noise_heading_std_deg: float = 2.0,
    image_placeholders: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Build a schema-compatible VLM input.

    The memory object may contain graph-context extensions, but the top-level
    envelope is exactly ``nav_vlm_waypoint_v1``.
    """
    return {
        "schema_version": "nav_vlm_waypoint_v1",
        "task": task.to_schema(),
        "coordinate_frame": {
            "map_frame": "habitat_world_xy_or_gps_aligned_local_map",
            "robot_pose_source": "gps_or_sim_gt_with_noise_or_robot_backend",
            "heading_source": "imu_or_sim_gt_with_noise_or_robot_backend",
            "pose_noise": {
                "enabled": bool(pose_noise_enabled),
                "xy_std_m": float(pose_noise_xy_std_m),
                "heading_std_deg": float(pose_noise_heading_std_deg),
            },
        },
        "robot_state": robot_state.to_schema(),
        "observation": observation.to_schema(image_placeholders=image_placeholders),
        "memory": memory,
        "constraints": default_constraints(),
    }


def make_empty_fine_goal(width: int = 640, height: int = 480, navigability: str = "unknown") -> Dict[str, Any]:
    """Construct an invalid fine goal block."""
    return {
        "valid": False,
        "view_id": None,
        "view_type": None,
        "point_px": None,
        "point_norm": None,
        "projected_map_xy": None,
        "navigability": navigability,
    }


def make_go_output(
    *,
    view_id: int,
    view_type: str,
    point_px: Tuple[int, int],
    width: int,
    height: int,
    decision_reason: str,
    goal_reason: str,
    short_text: str,
    confidence: str = "medium",
) -> Dict[str, Any]:
    """Construct a valid v1 ``go`` output for testing or fallback policies."""
    u, v = int(point_px[0]), int(point_px[1])
    return {
        "schema_version": "nav_vlm_waypoint_v1",
        "action": "go",
        "selected_view_id": int(view_id),
        "selected_view_type": view_type,
        "selected_image_point": [u, v],
        "fine_goal": {
            "valid": True,
            "view_id": int(view_id),
            "view_type": view_type,
            "point_px": [u, v],
            "point_norm": [round(u / max(1, width - 1), 4), round(v / max(1, height - 1), 4)],
            "projected_map_xy": None,
            "navigability": "likely_free",
        },
        "observation_request": {"valid": False, "mode": None, "center_yaw_deg": None, "step_deg": None, "num_views": None, "yaw_offsets_deg": None, "reason": None},
        "reasoning": {
            "decision_reason": decision_reason,
            "goal_reason": goal_reason,
            "failure_mode": None,
            "short_text": short_text,
        },
        "control": {"vlm_control_mode": "resume_async_navigation", "rotate_yaw_deg": 0, "ttl_ms": 1000},
        "confidence": confidence,
    }


def make_rotate_output(yaw_deg: float, reason: str = "R02_NO_VISIBLE_NAVIGABLE_FLOOR", confidence: str = "medium") -> Dict[str, Any]:
    """Construct a valid v1 ``rotate`` output."""
    return {
        "schema_version": "nav_vlm_waypoint_v1",
        "action": "rotate",
        "selected_view_id": None,
        "selected_view_type": None,
        "selected_image_point": None,
        "fine_goal": make_empty_fine_goal(navigability="unknown"),
        "observation_request": None,
        "reasoning": {
            "decision_reason": reason,
            "goal_reason": "F08_NONE_ROTATE_OR_STOP",
            "failure_mode": "no_visible_navigable_floor",
            "short_text": "rotate to find navigable floor or exit",
        },
        "control": {"vlm_control_mode": "pause_until_rotation_done", "rotate_yaw_deg": round(float(yaw_deg), 3), "ttl_ms": 1000},
        "confidence": confidence,
    }


def make_observation_request_output(
    *,
    mode: str,
    center_yaw_deg: float,
    step_deg: int,
    num_views: int,
    yaw_offsets_deg: Sequence[float],
    reason: str,
    confidence: str = "medium",
) -> Dict[str, Any]:
    """Construct a valid v1 ``request_observation`` output."""
    return {
        "schema_version": "nav_vlm_waypoint_v1",
        "action": "request_observation",
        "selected_view_id": None,
        "selected_view_type": None,
        "selected_image_point": None,
        "fine_goal": make_empty_fine_goal(navigability="unknown"),
        "observation_request": {
            "valid": True,
            "mode": mode,
            "center_yaw_deg": round(float(center_yaw_deg), 3),
            "step_deg": int(step_deg),
            "num_views": int(num_views),
            "yaw_offsets_deg": [round(float(x), 3) for x in yaw_offsets_deg],
            "reason": reason,
        },
        "reasoning": {
            "decision_reason": "R03_LOW_CONFIDENCE_NEED_SCAN",
            "goal_reason": "F08_NONE_ROTATE_OR_STOP",
            "failure_mode": "insufficient_visual_context",
            "short_text": reason,
        },
        "control": {"vlm_control_mode": "pause_for_observation", "rotate_yaw_deg": 0, "ttl_ms": 1000},
        "confidence": confidence,
    }


def make_stop_output(reason: str = "S02_TARGET_REACHED_OR_TASK_DONE", confidence: str = "high") -> Dict[str, Any]:
    """Construct a valid v1 ``stop`` output."""
    return {
        "schema_version": "nav_vlm_waypoint_v1",
        "action": "stop",
        "selected_view_id": None,
        "selected_view_type": None,
        "selected_image_point": None,
        "fine_goal": make_empty_fine_goal(navigability="blocked"),
        "observation_request": None,
        "reasoning": {
            "decision_reason": reason,
            "goal_reason": "F08_NONE_ROTATE_OR_STOP",
            "failure_mode": None if reason == "S02_TARGET_REACHED_OR_TASK_DONE" else "hazard_or_abort",
            "short_text": "stop condition satisfied",
        },
        "control": {"vlm_control_mode": "hard_stop", "rotate_yaw_deg": 0, "ttl_ms": 2000},
        "confidence": confidence,
    }


def dataclass_to_json_dict(obj: Any) -> Any:
    """Recursively convert dataclasses and tuples into JSON-safe dictionaries."""
    if hasattr(obj, "__dataclass_fields__"):
        return dataclass_to_json_dict(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): dataclass_to_json_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_json_dict(v) for v in obj]
    return obj
