from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image


DEFAULT_VOCA_ROOT = Path("/home/icra/voca-s2e")
PIXELNAV_ACTION_NAMES = {
    0: "stop",
    1: "move_forward",
    2: "turn_left",
    3: "turn_right",
    4: "look_up",
    5: "look_down",
}


def ensure_voca_imports(voca_root: str | Path = DEFAULT_VOCA_ROOT) -> None:
    """Expose VOCA memory packages without requiring installation."""
    root = Path(voca_root)
    qwen_root = root / "qwen_nav_memory_framework_v5"
    for path in (root, qwen_root):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


@dataclass(frozen=True)
class BenchmarkRunConfig:
    output_dir: Path
    eval_episodes: int = 1
    max_agent_steps: int = 20
    max_env_steps: int | None = 250
    max_pixelnav_steps: int = 12
    mask_radius: int = 5
    turn_angle_deg: float = 30.0
    success_distance_m: float = 1.0
    force_front_view_waypoint: bool = True
    write_videos: bool = False
    video_fps: int = 4
    memory_video_fps: int = 2
    voca_root: Path = DEFAULT_VOCA_ROOT


class EpisodeVideoRecorder:
    """Write first-person and top-down Habitat videos for one episode."""

    def __init__(self, episode_dir: str | Path, *, enabled: bool = False, fps: int = 4):
        self.episode_dir = Path(episode_dir)
        self.enabled = bool(enabled)
        self.fps = max(1, int(fps))
        self.fps_path = self.episode_dir / "fps.mp4"
        self.metric_path = self.episode_dir / "metric.mp4"
        self._fps_writer = None
        self._metric_writer = None
        self.frame_count = 0
        self.metric_frame_count = 0

    def append(self, obs: dict[str, Any], metrics: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        import imageio

        rgb = _rgb_from_obs(obs)
        if self._fps_writer is None:
            self._fps_writer = imageio.get_writer(str(self.fps_path), fps=self.fps)
        self._fps_writer.append_data(rgb)
        self.frame_count += 1

        topdown = _topdown_frame_from_metrics(metrics or {})
        if topdown is not None:
            if self._metric_writer is None:
                self._metric_writer = imageio.get_writer(str(self.metric_path), fps=self.fps)
            self._metric_writer.append_data(topdown)
            self.metric_frame_count += 1

    def close(self) -> None:
        for writer in (self._fps_writer, self._metric_writer):
            if writer is not None:
                writer.close()
        self._fps_writer = None
        self._metric_writer = None

    def artifacts(self) -> dict[str, Any]:
        return {
            "fps_mp4": str(self.fps_path) if self.fps_path.exists() else None,
            "metric_mp4": str(self.metric_path) if self.metric_path.exists() else None,
            "fps_frames": self.frame_count,
            "metric_frames": self.metric_frame_count,
        }


class HabitatEnvMemoryBackend:
    """`nav_memory_qwen.RobotBackend` wrapper around an official Habitat env.

    The benchmark env remains authoritative: all movement goes through
    `habitat_env.step(...)`, and official SR/SPL are read from
    `habitat_env.get_metrics()`.
    """

    def __init__(
        self,
        *,
        env: Any,
        initial_obs: dict[str, Any],
        output_dir: str | Path,
        pixelnav_policy: Any,
        max_pixelnav_steps: int = 12,
        mask_radius: int = 5,
        turn_angle_deg: float = 30.0,
        sequence_id: str = "official_habitat_episode",
        max_env_steps: int | None = None,
        video_recorder: EpisodeVideoRecorder | None = None,
        voca_root: str | Path = DEFAULT_VOCA_ROOT,
    ):
        ensure_voca_imports(voca_root)
        self.env = env
        self._last_obs = dict(initial_obs)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pixelnav_policy = pixelnav_policy
        self.max_pixelnav_steps = int(max_pixelnav_steps)
        self.mask_radius = int(mask_radius)
        self.turn_angle_deg = float(turn_angle_deg)
        self.sequence_id = sequence_id
        self.frame_index = 0
        self.rollout_index = 0
        self.supported_action_ids = _supported_action_ids(env)
        self.max_env_steps = None if max_env_steps is None else max(0, int(max_env_steps))
        self.env_step_count = 0
        self.env_step_budget_exhausted = False
        self.video_recorder = video_recorder
        self.pointnav_goal_distance_m: float | None = None
        self.pointnav_success_distance_m = 0.20
        self.low_level_forward_step_m = 0.25
        if hasattr(self.pixelnav_policy, "bind_env"):
            self.pixelnav_policy.bind_env(env)

    def _distance_aware_rollout_step_limit(self) -> int:
        if self.pointnav_goal_distance_m is None:
            return self.max_pixelnav_steps
        try:
            distance = float(self.pointnav_goal_distance_m)
        except (TypeError, ValueError):
            return self.max_pixelnav_steps
        if not math.isfinite(distance):
            return self.max_pixelnav_steps
        remaining_clearance = distance - self.pointnav_success_distance_m
        if remaining_clearance <= 0.0:
            return 0
        near_goal_limit = max(1, int(math.floor(remaining_clearance / self.low_level_forward_step_m)))
        return min(self.max_pixelnav_steps, near_goal_limit)

    def _can_step_env(self) -> bool:
        if self.max_env_steps is None:
            return True
        if self.env_step_count < self.max_env_steps:
            return True
        self.env_step_budget_exhausted = True
        return False

    def _step_env(self, action: int) -> dict[str, Any] | None:
        if not self._can_step_env():
            return None
        obs = dict(self.env.step(int(action)))
        self.env_step_count += 1
        self._last_obs = obs
        if self.video_recorder is not None:
            self.video_recorder.append(obs, dict(self.env.get_metrics()))
        return obs

    def get_robot_state(self) -> Any:
        from nav_memory_qwen.schema import RobotState

        state = self.env.sim.get_agent_state()
        position = _position_xyz(state)
        heading = _heading_rad_from_agent_state(state)
        return RobotState(
            map_xy=(float(position[0]), float(position[2])),
            heading_rad=float(heading),
            position_xyz=(float(position[0]), float(position[1]), float(position[2])),
        )

    def get_observation(self) -> Any:
        return self.capture_views([0.0], mode="current_only")

    def capture_views(self, yaw_offsets_deg: Sequence[float], mode: str = "directed_sweep") -> Any:
        from nav_memory_qwen.schema import Observation, ObservationView, nearest_view_type, now_ms

        self.frame_index += 1
        frame_dir = self.output_dir / "observations" / f"frame_{self.frame_index:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        rgb = _rgb_from_obs(self._last_obs)
        views = []
        used_view_types: set[str] = set()
        for index, yaw in enumerate(yaw_offsets_deg):
            view_type = nearest_view_type(float(yaw))
            if view_type in used_view_types:
                continue
            used_view_types.add(view_type)
            image_path = frame_dir / f"view_{index:02d}_{view_type}.png"
            _write_rgb_png(rgb, image_path)
            views.append(
                ObservationView(
                    view_id=index,
                    view_type=view_type,
                    relative_heading_deg=float(yaw),
                    image=str(image_path),
                    timestamp_ms=now_ms(),
                )
            )
        return Observation(
            mode=mode,
            sequence_id=self.sequence_id,
            frame_index=self.frame_index,
            image_width=int(rgb.shape[1]),
            image_height=int(rgb.shape[0]),
            views=views,
        )

    def rotate(self, yaw_deg: float) -> Any:
        from nav_memory_qwen.robot_backend import ActionOutcome
        from nav_memory_qwen.schema import RelativePose2D

        yaw_deg = max(-180.0, min(180.0, float(yaw_deg)))
        if abs(yaw_deg) < 1e-6:
            return ActionOutcome(
                action="rotate",
                success=True,
                rotated_deg=0.0,
                odom_delta=RelativePose2D(),
                message="No rotation requested.",
            )

        before = _position_xyz(self.env.sim.get_agent_state())
        turn_action = 2 if yaw_deg > 0 else 3
        requested_step_count = max(1, int(round(abs(yaw_deg) / max(self.turn_angle_deg, 1e-6))))
        actual_step_count = 0
        for _ in range(requested_step_count):
            if getattr(self.env, "episode_over", False):
                break
            if self._step_env(turn_action) is None:
                break
            actual_step_count += 1
        actual_yaw = math.copysign(actual_step_count * self.turn_angle_deg, yaw_deg)
        after = _position_xyz(self.env.sim.get_agent_state())
        moved = _distance(before, after)
        return ActionOutcome(
            action="rotate",
            success=actual_step_count > 0 or abs(yaw_deg) < 1e-6,
            collision=bool(getattr(self.env.sim, "previous_step_collided", False)),
            moved_distance_m=round(moved, 6),
            rotated_deg=float(actual_yaw),
            odom_delta=RelativePose2D(0.0, 0.0, actual_yaw),
            message=f"Habitat env rotate via action {turn_action} x{actual_step_count}/{requested_step_count}.",
        )

    def stop(self) -> dict[str, Any]:
        obs = self._step_env(0)
        if obs is None:
            return self._last_obs
        return self._last_obs

    def execute_waypoint(self, *, view_type: str, view_id: int, point_px: tuple[int, int], ttl_ms: int) -> Any:
        from nav_memory_qwen.robot_backend import ActionOutcome
        from nav_memory_qwen.schema import RelativePose2D

        rollout_dir = self.output_dir / "rollouts" / f"rollout_{self.rollout_index:04d}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_index += 1

        goal_rgb = _rgb_from_obs(self._last_obs)
        goal_mask = _make_goal_mask(goal_rgb.shape[:2], point_px, radius=self.mask_radius)
        _write_rgb_png(goal_rgb, rollout_dir / "goal_rgb.png")
        _write_rgb_png(_overlay_point(goal_rgb, point_px), rollout_dir / "goal_rgb_selected_point.png")
        self.pixelnav_policy.reset(goal_rgb, goal_mask)

        start_position = _position_xyz(self.env.sim.get_agent_state())
        steps = []
        stopped_by_policy = False
        unsupported_action = None

        rollout_step_limit = self._distance_aware_rollout_step_limit()
        for step_index in range(rollout_step_limit):
            if getattr(self.env, "episode_over", False):
                break
            if not self._can_step_env():
                break
            obs_rgb = _rgb_from_obs(self._last_obs)
            collision_before = bool(getattr(self.env.sim, "previous_step_collided", False))
            position_before = _position_xyz(self.env.sim.get_agent_state())
            if hasattr(self.pixelnav_policy, "step_from_observation"):
                action, overlay = self.pixelnav_policy.step_from_observation(
                    self._last_obs,
                    obs_rgb,
                    collide=collision_before,
                )
            else:
                action, overlay = self.pixelnav_policy.step(obs_rgb, collide=collision_before)
            action = int(action)
            overlay_path = rollout_dir / f"overlay_{step_index:03d}.png"
            _write_rgb_png(_rgb_array(overlay), overlay_path)

            if action == 0:
                stopped_by_policy = True
                position_after = position_before
                sim_action = None
            elif action not in PIXELNAV_ACTION_NAMES or action not in self.supported_action_ids:
                unsupported_action = action
                position_after = position_before
                sim_action = None
            else:
                sim_action = PIXELNAV_ACTION_NAMES[action]
                try:
                    obs = self._step_env(action)
                    if obs is None:
                        position_after = position_before
                        sim_action = None
                        break
                    position_after = _position_xyz(self.env.sim.get_agent_state())
                except Exception:
                    unsupported_action = action
                    position_after = position_before
                    sim_action = None

            steps.append(
                {
                    "step_index": step_index,
                    "pixelnav_action": action,
                    "sim_action": sim_action,
                    "collision_before": collision_before,
                    "position_before_xyz": _float_list(position_before),
                    "position_after_xyz": _float_list(position_after),
                    "moved_distance_m": round(_distance(position_before, position_after), 6),
                    "overlay_path": str(overlay_path),
                }
            )
            if stopped_by_policy or unsupported_action is not None:
                break

        final_position = _position_xyz(self.env.sim.get_agent_state())
        total_moved = round(sum(float(step["moved_distance_m"]) for step in steps), 6)
        collision_count = sum(1 for step in steps if step["collision_before"])
        collision = bool(collision_count > 0 or getattr(self.env.sim, "previous_step_collided", False))
        success = bool(steps and unsupported_action is None and not collision and total_moved >= 0.20)
        rollout_payload = {
            "request": {
                "selected_view": view_type,
                "selected_view_id": int(view_id),
                "selected_image_point": [int(point_px[0]), int(point_px[1])],
                "ttl_ms": int(ttl_ms),
            },
            "rollout": {
                "steps": steps,
                "total_moved_distance_m": total_moved,
                "stopped_by_policy": stopped_by_policy,
                "unsupported_action": unsupported_action,
                "collision_count": collision_count,
                "env_step_budget_exhausted": self.env_step_budget_exhausted,
                "env_steps": self.env_step_count,
                "max_env_steps": self.max_env_steps,
                "rollout_step_limit": rollout_step_limit,
                "pointnav_goal_distance_m": self.pointnav_goal_distance_m,
            },
            "start_position_xyz": _float_list(start_position),
            "final_position_xyz": _float_list(final_position),
            "final_position_delta_m": round(_distance(start_position, final_position), 6),
            "pixelnav_goal": {
                "mask_radius": self.mask_radius,
                "mask_nonzero_pixels": int(np.count_nonzero(goal_mask)),
            },
        }
        rollout_json = rollout_dir / "pixelnav_rollout_result.json"
        rollout_json.write_text(
            json.dumps(_json_safe(rollout_payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        message = (
            f"PixelNav official-env rollout: moved {total_moved:.3f} m, "
            f"steps={len(steps)}, collision={int(collision)}."
        )
        return ActionOutcome(
            action="go",
            success=success,
            collision=collision,
            moved_distance_m=total_moved,
            rotated_deg=0.0,
            odom_delta=RelativePose2D(dx_m=total_moved, dy_m=0.0, dyaw_deg=0.0),
            message=message,
            raw={
                "source": "pixel_navigator_official_env",
                "rollout_result_json": str(rollout_json),
                "selected_view": view_type,
                "selected_view_id": int(view_id),
                "selected_image_point": [int(point_px[0]), int(point_px[1])],
                "rollout": rollout_payload["rollout"],
                "env_step_budget_exhausted": self.env_step_budget_exhausted,
                "env_steps": self.env_step_count,
                "max_env_steps": self.max_env_steps,
            },
        )


class FallbackVLMClient:
    """Use a deterministic fallback if the primary VLM call/parsing fails."""

    def __init__(self, primary: Any, fallback: Any):
        self.primary = primary
        self.fallback = fallback
        self.fallback_count = 0
        self.fallback_errors: list[str] = []

    def decide(self, vlm_input: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.primary.decide(vlm_input)
        except Exception as exc:
            self.fallback_count += 1
            reason = f"{type(exc).__name__}: {exc}"
            self.fallback_errors.append(reason)
            output = dict(self.fallback.decide(vlm_input))
            reasoning = dict(output.get("reasoning") or {})
            reasoning["short_text"] = "heuristic fallback after VLM parse/call failure"
            output["reasoning"] = reasoning
            output["runtime_fallback"] = {
                "used": True,
                "reason": reason,
            }
            return output


class PixelNavFriendlyHeuristicVLMClient:
    """Heuristic VLM with a goal point ratio that matches PixelNav rollouts better."""

    def __init__(self, base: Any | None = None, *, y_ratio: float = 0.625, voca_root: str | Path = DEFAULT_VOCA_ROOT):
        ensure_voca_imports(voca_root)
        if base is None:
            from nav_memory_qwen.vlm_client import HeuristicVLMClient

            base = HeuristicVLMClient()
        self.base = base
        self.y_ratio = float(y_ratio)

    def decide(self, vlm_input: dict[str, Any]) -> dict[str, Any]:
        output = dict(self.base.decide(vlm_input))
        if output.get("action") != "go":
            return output
        obs = vlm_input.get("observation", {}) if isinstance(vlm_input.get("observation"), dict) else {}
        width = int(obs.get("image_width", 640) or 640)
        height = int(obs.get("image_height", 480) or 480)
        u = width // 2
        v = max(0, min(height - 1, int(round(height * self.y_ratio))))
        output["selected_image_point"] = [u, v]
        fine_goal = dict(output.get("fine_goal") or {})
        fine_goal["point_px"] = [u, v]
        fine_goal["selected_image_point"] = [u, v]
        fine_goal["point_norm"] = [round(u / max(width - 1, 1), 4), round(v / max(height - 1, 1), 4)]
        output["fine_goal"] = fine_goal
        reasoning = dict(output.get("reasoning") or {})
        short_text = str(reasoning.get("short_text") or "")
        if "pixelnav-friendly" not in short_text:
            reasoning["short_text"] = (short_text + "; " if short_text else "") + "pixelnav-friendly mid-floor point"
        output["reasoning"] = reasoning
        return output


class PointNavBearingHeuristicVLMClient:
    """PointNav diagnostic VLM that prioritizes GPS bearing before memory exits."""

    def __init__(
        self,
        *,
        y_ratio: float = 0.625,
        rotate_threshold_deg: float = 25.0,
        max_rotate_deg: float = 90.0,
        stop_distance_m: float = 0.2,
        fallback: Any | None = None,
        voca_root: str | Path = DEFAULT_VOCA_ROOT,
    ):
        ensure_voca_imports(voca_root)
        self.y_ratio = float(y_ratio)
        self.rotate_threshold_deg = float(rotate_threshold_deg)
        self.max_rotate_deg = float(max_rotate_deg)
        self.stop_distance_m = float(stop_distance_m)
        self.fallback = fallback or PixelNavFriendlyHeuristicVLMClient(y_ratio=y_ratio, voca_root=voca_root)
        self.collision_recovery_turn_index = 0
        self.collision_recovery_forward_steps = 0

    def decide(self, vlm_input: dict[str, Any]) -> dict[str, Any]:
        from nav_memory_qwen.schema import make_go_output, make_rotate_output, make_stop_output, normalize_angle_deg

        task = vlm_input.get("task", {}) if isinstance(vlm_input.get("task"), dict) else {}
        if task.get("task_mode") != "PointNav":
            return self.fallback.decide(vlm_input)

        coarse = task.get("coarse_goal", {}) if isinstance(task.get("coarse_goal"), dict) else {}
        bearing = normalize_angle_deg(float(coarse.get("relative_bearing_deg", 0.0) or 0.0))
        distance = float(coarse.get("distance_m", 999.0) or 999.0)
        if distance <= self.stop_distance_m:
            output = make_stop_output(reason="S02_TARGET_REACHED_OR_TASK_DONE", confidence="high")
            output["reasoning"]["short_text"] = f"PointNav goal within {self.stop_distance_m:.2f} m stop distance"
            return output

        obs = vlm_input.get("observation", {}) if isinstance(vlm_input.get("observation"), dict) else {}
        width = int(obs.get("image_width", 640) or 640)
        height = int(obs.get("image_height", 480) or 480)
        front_view = next(
            (view for view in (obs.get("views", []) or []) if view.get("view_type") == "front"),
            {"view_id": 0, "view_type": "front"},
        )
        u = width // 2
        v = max(0, min(height - 1, int(round(height * self.y_ratio))))

        def make_front_go(short_text: str, confidence: str = "medium") -> dict[str, Any]:
            return make_go_output(
                view_id=int(front_view.get("view_id", 0)),
                view_type="front",
                point_px=(u, v),
                width=width,
                height=height,
                decision_reason="G02_VISIBLE_FLOOR_TOWARD_GOAL",
                goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
                short_text=short_text,
                confidence=confidence,
            )

        memory = vlm_input.get("memory", {}) if isinstance(vlm_input.get("memory"), dict) else {}
        runtime_state = memory.get("runtime_state", {}) if isinstance(memory.get("runtime_state"), dict) else {}
        last_outcome = (
            runtime_state.get("last_action_outcome", {})
            if isinstance(runtime_state.get("last_action_outcome"), dict)
            else {}
        )
        last_collision = bool(last_outcome.get("collision"))
        last_moved = float(last_outcome.get("moved_distance_m", 999.0) or 0.0)
        if self.collision_recovery_forward_steps > 0:
            self.collision_recovery_forward_steps -= 1
            return make_front_go(
                (
                    f"collision recovery forward after escape turn: "
                    f"bearing {bearing:.1f} deg, point y_ratio={self.y_ratio:.3f}"
                ),
                confidence="low",
            )
        if last_collision and last_moved < 0.20:
            yaw = 60.0 if self.collision_recovery_turn_index % 2 == 0 else -60.0
            self.collision_recovery_turn_index += 1
            self.collision_recovery_forward_steps = 1
            output = make_rotate_output(yaw, reason="R03_COLLISION_RECOVERY", confidence="low")
            output["reasoning"]["short_text"] = (
                f"collision recovery turn after low-progress collision: moved {last_moved:.3f} m"
            )
            return output

        if abs(bearing) > self.rotate_threshold_deg:
            yaw = max(-self.max_rotate_deg, min(self.max_rotate_deg, bearing))
            output = make_rotate_output(yaw, reason="R01_GOAL_OUTSIDE_CURRENT_VIEW", confidence="medium")
            output["reasoning"]["short_text"] = (
                f"PointNav bearing-first rotate: goal bearing {bearing:.1f} deg exceeds "
                f"{self.rotate_threshold_deg:.1f} deg threshold"
            )
            return output

        return make_front_go(
            (
                f"PointNav bearing-aligned front goal: bearing {bearing:.1f} deg, "
                f"point y_ratio={self.y_ratio:.3f}"
            )
        )


class ForwardOnlyPixelPolicy:
    """Cheap local controller for PointNav ablation: keep moving forward."""

    def reset(self, goal_image: np.ndarray, goal_mask: np.ndarray) -> None:
        self.goal_image_shape = tuple(goal_image.shape)

    def step(self, image: np.ndarray, collide: bool = False) -> tuple[int, np.ndarray]:
        return 1, _rgb_array(image)


class ReactiveForwardPixelPolicy:
    """Forward controller with a simple turn-on-collision recovery."""

    def __init__(self):
        self.collision_turn_count = 0

    def reset(self, goal_image: np.ndarray, goal_mask: np.ndarray) -> None:
        self.collision_turn_count = 0

    def step(self, image: np.ndarray, collide: bool = False) -> tuple[int, np.ndarray]:
        if collide:
            action = 2 if self.collision_turn_count % 2 == 0 else 3
            self.collision_turn_count += 1
            return action, _rgb_array(image)
        return 1, _rgb_array(image)


class PointGoalReactivePixelPolicy:
    """PointNav ablation controller that reads the official pointgoal sensor."""

    def __init__(self, *, success_distance_m: float = 0.2, turn_threshold_deg: float = 15.0):
        self.success_distance_m = float(success_distance_m)
        self.turn_threshold_deg = float(turn_threshold_deg)
        self.collision_turn_count = 0

    def reset(self, goal_image: np.ndarray, goal_mask: np.ndarray) -> None:
        self.collision_turn_count = 0

    def step_from_observation(self, obs: dict[str, Any], image: np.ndarray, collide: bool = False) -> tuple[int, np.ndarray]:
        pointgoal = obs.get("pointgoal_with_gps_compass")
        if pointgoal is None:
            return 1, _rgb_array(image)
        arr = np.asarray(pointgoal, dtype=np.float32).reshape(-1)
        if arr.size < 2:
            return 1, _rgb_array(image)
        distance = float(arr[0])
        bearing_deg = math.degrees(float(arr[1]))
        if distance <= self.success_distance_m:
            return 0, _rgb_array(image)
        if collide:
            action = 2 if self.collision_turn_count % 2 == 0 else 3
            self.collision_turn_count += 1
            return action, _rgb_array(image)
        if abs(bearing_deg) > self.turn_threshold_deg:
            # In this Habitat config, turn_left decreases the pointgoal bearing
            # and turn_right increases it.
            return (2 if bearing_deg > 0.0 else 3), _rgb_array(image)
        return 1, _rgb_array(image)

    def step(self, image: np.ndarray, collide: bool = False) -> tuple[int, np.ndarray]:
        return 1, _rgb_array(image)


class ShortestPathFollowerPixelPolicy:
    """Oracle PointNav upper-bound policy using Habitat's navmesh follower."""

    def __init__(self, *, goal_radius: float = 0.2):
        self.goal_radius = float(goal_radius)
        self.env = None
        self.goal_position = None
        self._follower = None

    def bind_env(self, env: Any) -> None:
        self.env = env
        self._follower = None

    def set_goal_position(self, goal_position: Sequence[float]) -> None:
        self.goal_position = np.asarray(goal_position, dtype=np.float32)
        self._follower = None

    def reset(self, goal_image: np.ndarray, goal_mask: np.ndarray) -> None:
        pass

    def _ensure_follower(self) -> Any | None:
        if self.env is None or self.goal_position is None:
            return None
        if self._follower is None:
            from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

            self._follower = ShortestPathFollower(
                self.env.sim,
                goal_radius=self.goal_radius,
                return_one_hot=False,
            )
        return self._follower

    def step(self, image: np.ndarray, collide: bool = False) -> tuple[int, np.ndarray]:
        follower = self._ensure_follower()
        if follower is None:
            return 1, _rgb_array(image)
        action = follower.get_next_action(self.goal_position)
        if action is None:
            action = 0
        return int(action), _rgb_array(image)


def run_objnav_memory_benchmark(
    *,
    env: Any,
    output_dir: str | Path,
    eval_episodes: int,
    max_agent_steps: int = 20,
    max_env_steps: int | None = None,
    max_pixelnav_steps: int = 12,
    pixelnav_policy_factory: Callable[[], Any],
    vlm_client_factory: Callable[[], Any],
    mask_radius: int = 5,
    turn_angle_deg: float = 30.0,
    success_distance_m: float = 1.0,
    force_front_view_waypoint: bool = True,
    write_videos: bool = False,
    video_fps: int = 4,
    memory_video_fps: int = 2,
    voca_root: str | Path = DEFAULT_VOCA_ROOT,
) -> dict[str, Any]:
    ensure_voca_imports(voca_root)
    from nav_memory_qwen.agent import NavAgentConfig, NavMemoryAgent

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for episode_index in range(int(eval_episodes)):
        episode_dir = output_dir / f"episode_{episode_index:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        initial_obs = dict(env.reset())
        video_recorder = EpisodeVideoRecorder(episode_dir, enabled=write_videos, fps=video_fps)
        if write_videos:
            video_recorder.append(initial_obs, dict(env.get_metrics()))
        initial_metrics = _json_safe(dict(env.get_metrics()))
        initial_distance_to_goal = _metric_float(initial_metrics, "distance_to_goal")
        episode = getattr(env, "current_episode", None)
        target_object = str(getattr(episode, "object_category", "unknown_object"))
        episode_id = str(getattr(episode, "episode_id", f"episode_{episode_index:04d}"))
        scene_id = str(getattr(episode, "scene_id", "unknown_scene"))
        backend = HabitatEnvMemoryBackend(
            env=env,
            initial_obs=initial_obs,
            output_dir=episode_dir / "backend",
            pixelnav_policy=pixelnav_policy_factory(),
            max_pixelnav_steps=max_pixelnav_steps,
            mask_radius=mask_radius,
            turn_angle_deg=turn_angle_deg,
            sequence_id=episode_id,
            max_env_steps=max_env_steps,
            video_recorder=video_recorder,
            voca_root=voca_root,
        )
        vlm_client = vlm_client_factory()
        agent = NavMemoryAgent(
            robot=backend,
            vlm_client=vlm_client,
            config=NavAgentConfig(
                success_distance_m=success_distance_m,
                max_steps=max_agent_steps,
                force_front_view_waypoint=force_front_view_waypoint,
            ),
        )
        stop_sent = False
        for step_index in range(int(max_agent_steps)):
            if getattr(env, "episode_over", False):
                break
            step_result = agent.step(target_object=target_object, step_index=step_index)
            if step_result.action == "stop" and not getattr(env, "episode_over", False):
                backend.stop()
                stop_sent = True
            if backend.env_step_budget_exhausted:
                break
            if step_result.done or step_result.error:
                break

        memory_dir = episode_dir / "memory"
        agent.save_run(memory_dir)
        memory_video_result = None
        if write_videos:
            memory_video_result = _render_memory_graph_video_artifact(
                memory_dir / "memory_graph.json",
                episode_dir,
                fps=memory_video_fps,
                voca_root=voca_root,
            )
        video_recorder.close()
        video_artifacts = video_recorder.artifacts()
        if memory_video_result:
            video_artifacts["memory_graph_video"] = memory_video_result.get("video_path")
            video_artifacts["memory_graph_video_summary_json"] = memory_video_result.get("summary_json")
        metrics = _compact_habitat_metrics(dict(env.get_metrics()))
        final_distance_to_goal = _metric_float(metrics, "distance_to_goal")
        record = {
            "episode_index": episode_index,
            "episode_id": episode_id,
            "scene_id": scene_id,
            "target_object": target_object,
            "habitat_metrics": metrics,
            "success": _metric_float(metrics, "success"),
            "spl": _metric_float(metrics, "spl"),
            "soft_spl": _metric_float(metrics, "soft_spl"),
            "distance_to_goal": final_distance_to_goal,
            "initial_distance_to_goal": initial_distance_to_goal,
            "distance_to_goal_delta": round(initial_distance_to_goal - final_distance_to_goal, 6),
            "episode_over": bool(getattr(env, "episode_over", False)),
            "stop_sent": bool(stop_sent),
            "agent_steps": len(agent.step_logs),
            "env_steps": backend.env_step_count,
            "max_env_steps": max_env_steps,
            "env_step_budget_exhausted": backend.env_step_budget_exhausted,
            "memory_nodes": len(agent.memory.nodes),
            "memory_edges": len(agent.memory.edges),
            "memory_graph_json": str(memory_dir / "memory_graph.json"),
            "steps_json": str(memory_dir / "steps.json"),
            "video_artifacts": video_artifacts,
            "vlm_fallback_count": int(getattr(vlm_client, "fallback_count", 0) or 0),
            "vlm_fallback_errors": list(getattr(vlm_client, "fallback_errors", []) or []),
        }
        (episode_dir / "episode_summary.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        records.append(record)
    return write_benchmark_summary(output_dir, records, benchmark_type="official_habitat_objnav_with_voca_memory")


def run_pointnav_memory_benchmark(
    *,
    env: Any,
    output_dir: str | Path,
    eval_episodes: int,
    max_agent_steps: int = 20,
    max_env_steps: int | None = None,
    max_pixelnav_steps: int = 12,
    pixelnav_policy_factory: Callable[[], Any],
    vlm_client_factory: Callable[[], Any],
    mask_radius: int = 5,
    turn_angle_deg: float = 30.0,
    success_distance_m: float = 0.5,
    force_front_view_waypoint: bool = True,
    write_videos: bool = False,
    video_fps: int = 4,
    memory_video_fps: int = 2,
    pointnav_goal_source: str = "sensor",
    voca_root: str | Path = DEFAULT_VOCA_ROOT,
) -> dict[str, Any]:
    ensure_voca_imports(voca_root)
    from nav_memory_qwen.agent import NavAgentConfig, NavMemoryAgent

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for episode_index in range(int(eval_episodes)):
        episode_dir = output_dir / f"episode_{episode_index:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        initial_obs = dict(env.reset())
        video_recorder = EpisodeVideoRecorder(episode_dir, enabled=write_videos, fps=video_fps)
        if write_videos:
            video_recorder.append(initial_obs, dict(env.get_metrics()))
        initial_metrics = _json_safe(dict(env.get_metrics()))
        initial_distance_to_goal = _metric_float(initial_metrics, "distance_to_goal")
        episode = getattr(env, "current_episode", None)
        goal_position = _pointnav_goal_position(episode)
        episode_goal_map_xy = (float(goal_position[0]), float(goal_position[2]))
        episode_id = str(getattr(episode, "episode_id", f"episode_{episode_index:04d}"))
        scene_id = str(getattr(episode, "scene_id", "unknown_scene"))
        backend = HabitatEnvMemoryBackend(
            env=env,
            initial_obs=initial_obs,
            output_dir=episode_dir / "backend",
            pixelnav_policy=pixelnav_policy_factory(),
            max_pixelnav_steps=max_pixelnav_steps,
            mask_radius=mask_radius,
            turn_angle_deg=turn_angle_deg,
            sequence_id=episode_id,
            max_env_steps=max_env_steps,
            video_recorder=video_recorder,
            voca_root=voca_root,
        )
        if hasattr(backend.pixelnav_policy, "set_goal_position"):
            backend.pixelnav_policy.set_goal_position(goal_position)
        vlm_client = vlm_client_factory()
        agent = NavMemoryAgent(
            robot=backend,
            vlm_client=vlm_client,
            config=NavAgentConfig(
                success_distance_m=success_distance_m,
                max_steps=max_agent_steps,
                force_front_view_waypoint=force_front_view_waypoint,
            ),
        )
        stop_sent = False
        for step_index in range(int(max_agent_steps)):
            if getattr(env, "episode_over", False):
                break
            state_for_goal = backend.get_robot_state()
            if pointnav_goal_source == "sensor":
                goal_map_xy = _pointnav_goal_map_xy_from_observation(
                    backend._last_obs,
                    state_for_goal,
                    fallback_goal_map_xy=episode_goal_map_xy,
                )
            elif pointnav_goal_source == "episode":
                goal_map_xy = episode_goal_map_xy
            else:
                raise ValueError(f"unsupported pointnav_goal_source: {pointnav_goal_source}")
            backend.pointnav_goal_distance_m = math.hypot(
                float(goal_map_xy[0]) - float(state_for_goal.map_xy[0]),
                float(goal_map_xy[1]) - float(state_for_goal.map_xy[1]),
            )
            step_result = agent.step(goal_map_xy=goal_map_xy, step_index=step_index)
            if step_result.action == "stop" and not getattr(env, "episode_over", False):
                backend.stop()
                stop_sent = True
            if backend.env_step_budget_exhausted:
                break
            if step_result.done or step_result.error:
                break

        memory_dir = episode_dir / "memory"
        agent.save_run(memory_dir)
        memory_video_result = None
        if write_videos:
            memory_video_result = _render_memory_graph_video_artifact(
                memory_dir / "memory_graph.json",
                episode_dir,
                fps=memory_video_fps,
                voca_root=voca_root,
            )
        video_recorder.close()
        video_artifacts = video_recorder.artifacts()
        if memory_video_result:
            video_artifacts["memory_graph_video"] = memory_video_result.get("video_path")
            video_artifacts["memory_graph_video_summary_json"] = memory_video_result.get("summary_json")
        metrics = _compact_habitat_metrics(dict(env.get_metrics()))
        final_distance_to_goal = _metric_float(metrics, "distance_to_goal")
        record = {
            "episode_index": episode_index,
            "episode_id": episode_id,
            "scene_id": scene_id,
            "goal_position_xyz": _float_list(goal_position),
            "goal_map_xy": [float(episode_goal_map_xy[0]), float(episode_goal_map_xy[1])],
            "pointnav_goal_source": pointnav_goal_source,
            "habitat_metrics": metrics,
            "success": _metric_float(metrics, "success"),
            "spl": _metric_float(metrics, "spl"),
            "soft_spl": _metric_float(metrics, "soft_spl"),
            "distance_to_goal": final_distance_to_goal,
            "initial_distance_to_goal": initial_distance_to_goal,
            "distance_to_goal_delta": round(initial_distance_to_goal - final_distance_to_goal, 6),
            "episode_over": bool(getattr(env, "episode_over", False)),
            "stop_sent": bool(stop_sent),
            "agent_steps": len(agent.step_logs),
            "env_steps": backend.env_step_count,
            "max_env_steps": max_env_steps,
            "env_step_budget_exhausted": backend.env_step_budget_exhausted,
            "memory_nodes": len(agent.memory.nodes),
            "memory_edges": len(agent.memory.edges),
            "memory_graph_json": str(memory_dir / "memory_graph.json"),
            "steps_json": str(memory_dir / "steps.json"),
            "video_artifacts": video_artifacts,
            "vlm_fallback_count": int(getattr(vlm_client, "fallback_count", 0) or 0),
            "vlm_fallback_errors": list(getattr(vlm_client, "fallback_errors", []) or []),
        }
        (episode_dir / "episode_summary.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        records.append(record)
    return write_benchmark_summary(output_dir, records, benchmark_type="official_habitat_pointnav_with_voca_memory")


def write_benchmark_summary(
    output_dir: str | Path,
    records: Iterable[dict[str, Any]],
    *,
    benchmark_type: str = "official_habitat_objnav_with_voca_memory",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = list(records)
    aggregate = {
        "episodes": len(episodes),
        "success": _mean(record["success"] for record in episodes),
        "spl": _mean(record["spl"] for record in episodes),
        "soft_spl": _mean(record["soft_spl"] for record in episodes),
        "mean_distance_to_goal": _mean(record["distance_to_goal"] for record in episodes),
        "mean_distance_to_goal_delta": _mean(record["distance_to_goal_delta"] for record in episodes),
        "mean_agent_steps": _mean(record["agent_steps"] for record in episodes),
        "mean_env_steps": _mean(record.get("env_steps", 0.0) for record in episodes),
        "mean_memory_nodes": _mean(record["memory_nodes"] for record in episodes),
        "mean_memory_edges": _mean(record["memory_edges"] for record in episodes),
    }
    summary = {
        "benchmark_type": benchmark_type,
        "created_at_unix_s": int(time.time()),
        "aggregate": aggregate,
        "episodes": episodes,
    }
    stem = "voca_memory_pointnav_benchmark" if "pointnav" in benchmark_type else "voca_memory_objnav_benchmark"
    summary_path = output_dir / f"{stem}_summary.json"
    csv_path = output_dir / f"{stem}_metrics.csv"
    summary["summary_json"] = str(summary_path)
    summary["metrics_csv"] = str(csv_path)
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_metrics_csv(csv_path, episodes)
    return summary


def create_official_objnav_env(dataset: str, eval_episodes: int) -> Any:
    import habitat
    from config_utils import hm3d_config, mp3d_config

    if dataset == "hm3d":
        config = hm3d_config(stage="val", episodes=eval_episodes)
    elif dataset == "mp3d":
        config = mp3d_config(stage="val", episodes=eval_episodes)
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    return habitat.Env(config)


def create_official_pointnav_test_env(eval_episodes: int) -> Any:
    import habitat
    from habitat.config.default_structured_configs import LookDownActionConfig, LookUpActionConfig
    from habitat.config.read_write import read_write

    _ensure_habitat_test_scene_alias()
    config_path = (
        "/home/icra/micromamba-root/envs/habitat/lib/python3.9/site-packages/"
        "habitat/config/benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    )
    config = habitat.get_config(config_path)
    with read_write(config):
        config.habitat.dataset.split = "val"
        config.habitat.dataset.data_path = (
            "/home/icra/habitat_data/versioned_data/"
            "habitat_test_pointnav_dataset_1.0/v1/{split}/{split}.json.gz"
        )
        config.habitat.dataset.scenes_dir = "/home/icra/habitat_data/versioned_data"
        config.habitat.environment.iterator_options.num_episode_sample = int(eval_episodes)
        config.habitat.task.actions["look_up"] = LookUpActionConfig()
        config.habitat.task.actions["look_down"] = LookDownActionConfig()
        rgb_sensor = config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
        depth_sensor = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
        rgb_sensor.width = 640
        rgb_sensor.height = 480
        rgb_sensor.hfov = 79
        depth_sensor.width = 640
        depth_sensor.height = 480
        depth_sensor.hfov = 79
        depth_sensor.max_depth = 5.0
        depth_sensor.normalize_depth = False
        config.habitat.simulator.forward_step_size = 0.25
        config.habitat.simulator.turn_angle = 30
    return habitat.Env(config)


def create_official_pointnav_env(dataset: str, eval_episodes: int, *, split: str = "val") -> Any:
    import habitat

    if dataset == "mp3d" and not Path("/home/icra/habitat_data/scene_datasets/mp3d").exists():
        raise FileNotFoundError(
            "MP3D PointNav episodes are present, but /home/icra/habitat_data/scene_datasets/mp3d is missing. "
            "Download the licensed Matterport3D habitat scene assets before running MP3D PointNav."
        )
    return habitat.Env(_official_pointnav_config(dataset, eval_episodes=eval_episodes, split=split))


def _official_pointnav_config(dataset: str, *, eval_episodes: int, split: str = "val") -> Any:
    import habitat
    from habitat.config.default_structured_configs import (
        CollisionsMeasurementConfig,
        FogOfWarConfig,
        LookDownActionConfig,
        LookUpActionConfig,
        TopDownMapMeasurementConfig,
    )
    from habitat.config.read_write import read_write

    if dataset not in {"hm3d", "mp3d"}:
        raise ValueError(f"unsupported pointnav dataset: {dataset}")

    config_path = (
        "/home/icra/micromamba-root/envs/habitat/lib/python3.9/site-packages/"
        f"habitat/config/benchmark/nav/pointnav/pointnav_{dataset}.yaml"
    )
    config = habitat.get_config(config_path)
    habitat_data = Path("/home/icra/habitat_data")
    with read_write(config):
        config.habitat.dataset.split = split
        config.habitat.dataset.data_path = str(
            habitat_data / "datasets" / "pointnav" / dataset / "v1" / "{split}" / "{split}.json.gz"
        )
        config.habitat.dataset.scenes_dir = str(habitat_data / "scene_datasets")
        if dataset == "hm3d":
            config.habitat.simulator.scene_dataset = str(
                habitat_data / "scene_datasets" / "hm3d" / "hm3d_annotated_basis.scene_dataset_config.json"
            )
        else:
            config.habitat.simulator.scene_dataset = str(
                habitat_data / "scene_datasets" / "mp3d" / "mp3d.scene_dataset_config.json"
            )
        config.habitat.environment.iterator_options.num_episode_sample = int(eval_episodes)
        config.habitat.task.actions["look_up"] = LookUpActionConfig()
        config.habitat.task.actions["look_down"] = LookDownActionConfig()
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=False,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5.0, fov=90),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )
        rgb_sensor = config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
        depth_sensor = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
        rgb_sensor.width = 640
        rgb_sensor.height = 480
        rgb_sensor.hfov = 79
        depth_sensor.width = 640
        depth_sensor.height = 480
        depth_sensor.hfov = 79
        depth_sensor.max_depth = 5.0
        depth_sensor.normalize_depth = False
        config.habitat.simulator.forward_step_size = 0.25
        config.habitat.simulator.turn_angle = 30
        config.habitat.simulator.habitat_sim_v0.allow_sliding = True
        config.habitat.task.measurements.success.success_distance = 0.2
    return config


def make_pixelnav_policy_factory(
    checkpoint: str | Path | None,
    device: str | None,
    *,
    kind: str = "checkpoint",
) -> Callable[[], Any]:
    if kind == "forward":
        return lambda: ForwardOnlyPixelPolicy()
    if kind == "reactive-forward":
        return lambda: ReactiveForwardPixelPolicy()
    if kind == "pointgoal-reactive":
        return lambda: PointGoalReactivePixelPolicy()
    if kind == "shortest-path":
        return lambda: ShortestPathFollowerPixelPolicy()
    if kind != "checkpoint":
        raise ValueError(f"unsupported pixelnav policy kind: {kind}")

    def factory() -> Any:
        from constants import POLICY_CHECKPOINT
        from policy_agent import Policy_Agent

        model_path = str(checkpoint or POLICY_CHECKPOINT)
        return Policy_Agent(model_path=model_path, device=device or os.getenv("PIXELNAV_DEVICE", "cuda:0"))

    return factory


def make_vlm_client_factory(
    kind: str,
    voca_root: str | Path = DEFAULT_VOCA_ROOT,
    *,
    y_ratio: float = 0.625,
    bearing_rotate_threshold_deg: float = 25.0,
) -> Callable[[], Any]:
    ensure_voca_imports(voca_root)
    if kind == "heuristic":
        return lambda: PixelNavFriendlyHeuristicVLMClient(y_ratio=y_ratio, voca_root=voca_root)
    if kind == "pointnav-bearing":
        return lambda: PointNavBearingHeuristicVLMClient(
            y_ratio=y_ratio,
            rotate_threshold_deg=bearing_rotate_threshold_deg,
            voca_root=voca_root,
        )
    if kind == "qwen":
        from nav_memory_qwen.vlm_client import OpenAICompatibleVLMClient

        return lambda: FallbackVLMClient(
            OpenAICompatibleVLMClient.from_env(),
            PixelNavFriendlyHeuristicVLMClient(y_ratio=y_ratio, voca_root=voca_root),
        )
    raise ValueError(f"unsupported vlm client kind: {kind}")


def run_dry_run_fake_env(output_dir: str | Path, *, task: str = "objnav") -> dict[str, Any]:
    env = _DryRunHabitatEnv()
    kwargs = {
        "env": env,
        "output_dir": output_dir,
        "eval_episodes": 1,
        "max_agent_steps": 2,
        "max_env_steps": 250,
        "max_pixelnav_steps": 4,
        "pixelnav_policy_factory": _DryRunPixelPolicy,
        "vlm_client_factory": _DryRunVLM,
        "force_front_view_waypoint": True,
    }
    if task in {"pointnav", "pointnav-test"}:
        env.current_episode = _DryRunPointNavEpisode()
        return run_pointnav_memory_benchmark(**kwargs)
    return run_objnav_memory_benchmark(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Official Habitat benchmark with VOCA VLM-memory integration.")
    parser.add_argument("--task", choices=["objnav", "pointnav", "pointnav-test"], default="objnav")
    parser.add_argument("--dataset", choices=["hm3d", "mp3d"], default="hm3d")
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--max-agent-steps", type=int, default=20)
    parser.add_argument("--max-env-steps", type=int, default=250)
    parser.add_argument("--max-pixelnav-steps", type=int, default=12)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--pixelnav-device", default=os.getenv("PIXELNAV_DEVICE"))
    parser.add_argument(
        "--pixelnav-policy",
        choices=["checkpoint", "forward", "reactive-forward", "pointgoal-reactive", "shortest-path"],
        default="checkpoint",
    )
    parser.add_argument("--vlm", choices=["qwen", "heuristic", "pointnav-bearing"], default="qwen")
    parser.add_argument("--heuristic-y-ratio", type=float, default=0.625)
    parser.add_argument("--bearing-rotate-threshold-deg", type=float, default=25.0)
    parser.add_argument("--success-distance-m", type=float, default=None)
    parser.add_argument("--pointnav-goal-source", choices=["sensor", "episode"], default="sensor")
    parser.add_argument("--write-videos", action="store_true")
    parser.add_argument("--video-fps", type=int, default=4)
    parser.add_argument("--memory-video-fps", type=int, default=2)
    parser.add_argument("--voca-root", default=str(DEFAULT_VOCA_ROOT))
    parser.add_argument("--dry-run-fake-env", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run_fake_env:
        result = run_dry_run_fake_env(args.out, task=args.task)
    else:
        if args.task == "pointnav-test":
            env = create_official_pointnav_test_env(args.eval_episodes)
            result = run_pointnav_memory_benchmark(
                env=env,
                output_dir=args.out,
                eval_episodes=args.eval_episodes,
                max_agent_steps=args.max_agent_steps,
                max_env_steps=args.max_env_steps,
                max_pixelnav_steps=args.max_pixelnav_steps,
                pixelnav_policy_factory=make_pixelnav_policy_factory(
                    args.checkpoint,
                    args.pixelnav_device,
                    kind=args.pixelnav_policy,
                ),
                vlm_client_factory=make_vlm_client_factory(
                    args.vlm,
                    args.voca_root,
                    y_ratio=args.heuristic_y_ratio,
                    bearing_rotate_threshold_deg=args.bearing_rotate_threshold_deg,
                ),
                success_distance_m=args.success_distance_m if args.success_distance_m is not None else 0.2,
                write_videos=args.write_videos,
                video_fps=args.video_fps,
                memory_video_fps=args.memory_video_fps,
                pointnav_goal_source=args.pointnav_goal_source,
                voca_root=args.voca_root,
            )
        elif args.task == "pointnav":
            env = create_official_pointnav_env(args.dataset, args.eval_episodes)
            result = run_pointnav_memory_benchmark(
                env=env,
                output_dir=args.out,
                eval_episodes=args.eval_episodes,
                max_agent_steps=args.max_agent_steps,
                max_env_steps=args.max_env_steps,
                max_pixelnav_steps=args.max_pixelnav_steps,
                pixelnav_policy_factory=make_pixelnav_policy_factory(
                    args.checkpoint,
                    args.pixelnav_device,
                    kind=args.pixelnav_policy,
                ),
                vlm_client_factory=make_vlm_client_factory(
                    args.vlm,
                    args.voca_root,
                    y_ratio=args.heuristic_y_ratio,
                    bearing_rotate_threshold_deg=args.bearing_rotate_threshold_deg,
                ),
                success_distance_m=args.success_distance_m if args.success_distance_m is not None else 0.2,
                write_videos=args.write_videos,
                video_fps=args.video_fps,
                memory_video_fps=args.memory_video_fps,
                pointnav_goal_source=args.pointnav_goal_source,
                voca_root=args.voca_root,
            )
        else:
            env = create_official_objnav_env(args.dataset, args.eval_episodes)
            result = run_objnav_memory_benchmark(
                env=env,
                output_dir=args.out,
                eval_episodes=args.eval_episodes,
                max_agent_steps=args.max_agent_steps,
                max_env_steps=args.max_env_steps,
                max_pixelnav_steps=args.max_pixelnav_steps,
                pixelnav_policy_factory=make_pixelnav_policy_factory(
                    args.checkpoint,
                    args.pixelnav_device,
                    kind=args.pixelnav_policy,
                ),
                vlm_client_factory=make_vlm_client_factory(
                    args.vlm,
                    args.voca_root,
                    y_ratio=args.heuristic_y_ratio,
                    bearing_rotate_threshold_deg=args.bearing_rotate_threshold_deg,
                ),
                success_distance_m=args.success_distance_m if args.success_distance_m is not None else 1.0,
                write_videos=args.write_videos,
                video_fps=args.video_fps,
                memory_video_fps=args.memory_video_fps,
                voca_root=args.voca_root,
            )
        close = getattr(env, "close", None)
        if callable(close):
            close()
    print(json.dumps({"summary_json": result["summary_json"], "aggregate": result["aggregate"]}, ensure_ascii=False))
    return 0


def _write_metrics_csv(path: Path, episodes: list[dict[str, Any]]) -> None:
    fieldnames = [
        "episode_index",
        "episode_id",
        "scene_id",
        "target_object",
        "success",
        "spl",
        "soft_spl",
        "distance_to_goal",
        "initial_distance_to_goal",
        "distance_to_goal_delta",
        "agent_steps",
        "env_steps",
        "max_env_steps",
        "env_step_budget_exhausted",
        "memory_nodes",
        "memory_edges",
        "memory_graph_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in episodes:
            writer.writerow({key: record.get(key) for key in fieldnames})


def _supported_action_ids(env: Any) -> set[int]:
    action_space = getattr(env, "action_space", None)
    n = getattr(action_space, "n", None)
    if isinstance(n, int) and n > 0:
        return set(range(n))
    spaces = getattr(action_space, "spaces", None)
    if isinstance(spaces, dict):
        ids = set()
        for key in spaces:
            try:
                ids.add(int(key))
            except (TypeError, ValueError):
                pass
        if ids:
            return ids
    return {0, 1, 2, 3}


def _pointnav_goal_position(episode: Any) -> np.ndarray:
    goals = getattr(episode, "goals", None) or []
    if not goals:
        raise ValueError("PointNav episode has no goals")
    goal = goals[0]
    position = getattr(goal, "position", None)
    if position is None and isinstance(goal, dict):
        position = goal.get("position")
    if position is None:
        raise ValueError("PointNav goal has no position")
    return np.asarray(position, dtype=np.float64).reshape(3)


def _pointnav_goal_map_xy_from_observation(
    obs: dict[str, Any],
    robot_state: Any,
    *,
    fallback_goal_map_xy: tuple[float, float],
) -> tuple[float, float]:
    pointgoal = obs.get("pointgoal_with_gps_compass") if isinstance(obs, dict) else None
    if pointgoal is None:
        return fallback_goal_map_xy
    values = np.asarray(pointgoal, dtype=np.float64).reshape(-1)
    if values.size < 2:
        return fallback_goal_map_xy
    distance = float(values[0])
    relative_bearing_rad = float(values[1])
    if not np.isfinite(distance) or not np.isfinite(relative_bearing_rad):
        return fallback_goal_map_xy
    robot_x, robot_y = robot_state.map_xy
    world_bearing = float(robot_state.heading_rad) + relative_bearing_rad
    return (
        float(robot_x + distance * math.cos(world_bearing)),
        float(robot_y + distance * math.sin(world_bearing)),
    )


def _ensure_habitat_test_scene_alias() -> None:
    versioned = Path("/home/icra/habitat_data/versioned_data")
    src = versioned / "habitat_test_scenes"
    dst = versioned / "habitat-test-scenes"
    if dst.exists() or not src.exists():
        return
    try:
        dst.symlink_to(src.name)
    except FileExistsError:
        pass


def _metric_float(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, 0.0)
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0


def _compact_habitat_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in metrics.items():
        if key == "top_down_map" and isinstance(value, dict):
            raw_map = value.get("map")
            compact_map = {
                "present": True,
                "map_shape": list(np.asarray(raw_map).shape) if raw_map is not None else None,
                "agent_map_coord": _json_safe(value.get("agent_map_coord")),
                "agent_angle": _json_safe(value.get("agent_angle")),
            }
            compact[key] = compact_map
        else:
            compact[key] = _json_safe(value)
    return compact


def _topdown_frame_from_metrics(metrics: dict[str, Any]) -> np.ndarray | None:
    top_down_map = metrics.get("top_down_map") if isinstance(metrics, dict) else None
    if not top_down_map:
        return None
    try:
        import cv2
        from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height

        frame = colorize_draw_agent_and_fit_to_height(top_down_map, 1024)
        return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception:
        raw_map = top_down_map.get("map") if isinstance(top_down_map, dict) else top_down_map
        if raw_map is None:
            return None
        map_array = np.asarray(raw_map)
        if map_array.ndim == 2:
            map_array = np.stack([map_array, map_array, map_array], axis=-1)
        if map_array.ndim != 3:
            return None
        if map_array.shape[2] > 3:
            map_array = map_array[:, :, :3]
        return np.ascontiguousarray(map_array.astype(np.uint8))


def _render_memory_graph_video_artifact(
    graph_path: str | Path,
    episode_dir: str | Path,
    *,
    fps: int = 2,
    voca_root: str | Path = DEFAULT_VOCA_ROOT,
) -> dict[str, Any]:
    ensure_voca_imports(voca_root)
    from goal_adapter.memory_graph_visualizer import render_memory_graph_video

    return render_memory_graph_video(
        graph_path,
        output_dir=episode_dir,
        output_video="memory_graph_reconstruction.mp4",
        fps=fps,
        hold_frames=4,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _rgb_from_obs(obs: dict[str, Any]) -> np.ndarray:
    if "rgb" not in obs:
        raise KeyError("Habitat observation is missing 'rgb'")
    return _rgb_array(obs["rgb"])


def _rgb_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("RGB image must have shape (H, W, 3+)")
    return np.ascontiguousarray(array[:, :, :3].astype(np.uint8))


def _write_rgb_png(rgb: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_rgb_array(rgb)).save(path)


def _make_goal_mask(image_hw: Sequence[int], point_px: Sequence[int], radius: int) -> np.ndarray:
    height, width = int(image_hw[0]), int(image_hw[1])
    u = max(0, min(width - 1, int(point_px[0])))
    v = max(0, min(height - 1, int(point_px[1])))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[max(0, v - radius) : min(height, v + radius + 1), max(0, u - radius) : min(width, u + radius + 1)] = 255
    return mask


def _overlay_point(rgb: np.ndarray, point_px: Sequence[int]) -> np.ndarray:
    image = _rgb_array(rgb).copy()
    height, width = image.shape[:2]
    u = max(0, min(width - 1, int(point_px[0])))
    v = max(0, min(height - 1, int(point_px[1])))
    image[max(0, v - 3) : min(height, v + 4), max(0, u - 3) : min(width, u + 4)] = np.array([255, 0, 0], dtype=np.uint8)
    return image


def _position_xyz(agent_state: Any) -> np.ndarray:
    position = getattr(agent_state, "position", agent_state)
    return np.asarray(position, dtype=np.float64).reshape(3)


def _heading_rad_from_agent_state(agent_state: Any) -> float:
    if hasattr(agent_state, "heading_rad"):
        return float(agent_state.heading_rad)
    rotation = getattr(agent_state, "rotation", None)
    if rotation is None:
        return 0.0
    try:
        import quaternion

        matrix = quaternion.as_rotation_matrix(rotation)
        forward = matrix @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        return float(math.atan2(forward[2], forward[0]))
    except Exception:
        return 0.0


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(aa - bb))


def _float_list(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]


class _DryRunEpisode:
    object_category = "chair"
    episode_id = "dry_run_objnav_0001"
    scene_id = "dry_run_scene.glb"


class _DryRunGoal:
    position = [1.0, 0.0, 0.0]


class _DryRunPointNavEpisode:
    episode_id = "dry_run_pointnav_0001"
    scene_id = "dry_run_scene.glb"
    goals = [_DryRunGoal()]


class _DryRunAgentState:
    def __init__(self, position: Sequence[float], heading_rad: float):
        self.position = np.asarray(position, dtype=np.float32)
        self.heading_rad = float(heading_rad)
        self.rotation = None


class _DryRunSim:
    previous_step_collided = False

    def __init__(self):
        self.position = np.zeros(3, dtype=np.float32)
        self.heading_rad = 0.0

    def get_agent_state(self) -> _DryRunAgentState:
        return _DryRunAgentState(self.position.copy(), self.heading_rad)


class _DryRunHabitatEnv:
    def __init__(self):
        self.sim = _DryRunSim()
        self.current_episode = _DryRunEpisode()
        self.episode_over = False
        self.metrics = {"success": 0.0, "spl": 0.0, "distance_to_goal": 2.0}

    def reset(self) -> dict[str, Any]:
        self.sim.position[:] = 0.0
        self.episode_over = False
        self.metrics = {"success": 0.0, "spl": 0.0, "distance_to_goal": 2.0}
        return {"rgb": np.full((96, 128, 3), 140, dtype=np.uint8)}

    def step(self, action: int) -> dict[str, Any]:
        action = int(action)
        if action == 0:
            self.episode_over = True
            self.metrics.update(success=1.0, spl=0.75, distance_to_goal=0.2)
        elif action == 1:
            self.sim.position[0] += 0.25
            self.metrics["distance_to_goal"] = max(0.0, self.metrics["distance_to_goal"] - 0.25)
        elif action == 2:
            self.sim.heading_rad -= math.radians(30.0)
        elif action == 3:
            self.sim.heading_rad += math.radians(30.0)
        return {"rgb": np.full((96, 128, 3), 140, dtype=np.uint8)}

    def get_metrics(self) -> dict[str, Any]:
        return dict(self.metrics)


class _DryRunPixelPolicy:
    def __init__(self):
        self.calls = 0

    def reset(self, goal_image: np.ndarray, goal_mask: np.ndarray) -> None:
        self.calls = 0

    def step(self, image: np.ndarray, collide: bool = False) -> tuple[int, np.ndarray]:
        self.calls += 1
        if self.calls == 1:
            return 1, image
        return 0, image


class _DryRunVLM:
    def __init__(self):
        self.calls = 0

    def decide(self, vlm_input: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            obs = vlm_input.get("observation", {})
            return {
                "schema_version": "nav_vlm_waypoint_v1",
                "action": "go",
                "selected_view_id": 0,
                "selected_view_type": "front",
                "selected_image_point": [int(obs.get("image_width", 128)) // 2, int(obs.get("image_height", 96)) - 12],
                "fine_goal": {"valid": True},
                "reasoning": {"short_text": "dry-run visible floor"},
                "confidence": "medium",
            }
        return {
            "schema_version": "nav_vlm_waypoint_v1",
            "action": "stop",
            "reasoning": {"short_text": "dry-run stop"},
            "confidence": "medium",
        }


if __name__ == "__main__":
    raise SystemExit(main())
