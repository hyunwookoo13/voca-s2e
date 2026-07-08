from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from goal_adapter.habitat_pixelnav_execute import (
    build_action_outcome_from_rollout,
    draw_selected_point_overlay,
    run_pixelnav_rollout_loop,
)
from goal_adapter.habitat_smoke import write_rgb_png


@dataclass(frozen=True)
class HabitatPixelNavBackendConfig:
    scene_path: str | Path
    output_dir: str | Path
    start_position_xyz: list[float]
    start_heading_rad: float = 0.0
    sequence_id: str = "habitat_pixelnav"
    image_width: int = 640
    image_height: int = 480
    sensor_height: float = 0.88
    image_hfov: float = 79.0
    forward_step_size: float = 0.25
    turn_angle: float = 30.0
    max_steps: int = 12
    mask_radius: int = 5
    pixelnav_root: str | Path = "/home/icra/Pixel-Navigator"
    checkpoint_path: str | Path = "checkpoints/navigator.pth"


class HabitatPixelNavBackend:
    def __init__(
        self,
        config: HabitatPixelNavBackendConfig,
        *,
        runner: Any | None = None,
        runner_factory: Any | None = None,
        executor_factory: Any | None = None,
    ):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sim = None
        self._agent = None
        self._runner = runner or (runner_factory or self._make_default_runner)(config)
        self._executor_factory = executor_factory or self._make_default_executor
        self._heading_rad = float(config.start_heading_rad)
        self._frame_index = 0
        self._rollout_index = 0
        if self._runner is not None:
            self._runner.set_pose(config.start_position_xyz, self._heading_rad)

    def get_robot_state(self) -> Any:
        RobotState = _import_memory_type("RobotState")
        position = tuple(_float_list(self._runner.position_xyz()))
        return RobotState(
            map_xy=(float(position[0]), float(position[2])),
            heading_rad=float(self._heading_rad),
            position_xyz=position,
        )

    def get_observation(self) -> Any:
        return self.capture_views([0.0], mode="current_only")

    def capture_views(self, yaw_offsets_deg: Sequence[float], mode: str = "directed_sweep") -> Any:
        Observation = _import_memory_type("Observation")
        ObservationView = _import_memory_type("ObservationView")
        nearest_view_type = _import_memory_type("nearest_view_type")

        self._frame_index += 1
        frame_dir = self.output_dir / "observations" / f"frame_{self._frame_index:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        position = self._runner.position_xyz()
        original_heading = self._heading_rad
        views = []
        used_view_types: set[str] = set()
        for index, yaw_deg in enumerate(yaw_offsets_deg):
            view_type = nearest_view_type(float(yaw_deg))
            if view_type in used_view_types:
                continue
            used_view_types.add(view_type)
            capture_heading = original_heading + math.radians(float(yaw_deg))
            self._runner.set_pose(position, capture_heading)
            image_path = frame_dir / f"view_{index:02d}_{view_type}.png"
            write_rgb_png(_rgb3(self._runner.rgb()), image_path)
            views.append(
                ObservationView(
                    view_id=index,
                    view_type=view_type,
                    relative_heading_deg=float(yaw_deg),
                    image=str(image_path),
                )
            )
        self._runner.set_pose(position, original_heading)
        return Observation(
            mode=mode,
            sequence_id=self.config.sequence_id,
            frame_index=self._frame_index,
            image_width=self.config.image_width,
            image_height=self.config.image_height,
            views=views,
        )

    def rotate(self, yaw_deg: float) -> Any:
        ActionOutcome = _import_robot_backend_type("ActionOutcome")
        RelativePose2D = _import_memory_type("RelativePose2D")
        yaw_deg = max(-180.0, min(180.0, float(yaw_deg)))
        self._heading_rad += math.radians(yaw_deg)
        self._runner.set_pose(self._runner.position_xyz(), self._heading_rad)
        return ActionOutcome(
            action="rotate",
            success=True,
            rotated_deg=yaw_deg,
            odom_delta=RelativePose2D(0.0, 0.0, yaw_deg),
            message=f"HabitatPixelNavBackend rotated {yaw_deg:.1f} deg.",
        )

    def execute_waypoint(self, *, view_type: str, view_id: int, point_px: tuple[int, int], ttl_ms: int) -> Any:
        view_type_to_heading_deg = _import_memory_type("view_type_to_heading_deg")
        rollout_dir = self.output_dir / "rollouts" / f"rollout_{self._rollout_index:04d}"
        self._rollout_index += 1
        rollout_dir.mkdir(parents=True, exist_ok=True)

        start_position = self._runner.position_xyz()
        base_heading = self._heading_rad
        selected_heading = base_heading + math.radians(view_type_to_heading_deg(view_type))
        self._runner.set_pose(start_position, selected_heading)

        goal_rgb = _rgb3(self._runner.rgb())
        goal_rgb_path = rollout_dir / "goal_rgb.png"
        write_rgb_png(goal_rgb, goal_rgb_path)
        goal_overlay_path = rollout_dir / "goal_rgb_selected_point.png"
        write_rgb_png(draw_selected_point_overlay(goal_rgb, point_px), goal_overlay_path)

        executor = self._executor_factory()
        goal = executor.reset_from_point(goal_rgb, point_px, radius=self.config.mask_radius)
        overlays_dir = rollout_dir / "overlays"
        summary = run_pixelnav_rollout_loop(
            executor,
            self._runner,
            max_steps=self.config.max_steps,
            overlay_dir=overlays_dir,
        )
        final_position = self._runner.position_xyz()
        self._heading_rad = selected_heading

        result = {
            "request": {
                "scene_path": str(self.config.scene_path),
                "robot_position_xyz": _float_list(start_position),
                "base_heading": base_heading,
                "selected_view": view_type,
                "selected_view_id": int(view_id),
                "selected_image_point": [int(point_px[0]), int(point_px[1])],
                "rollout_heading": selected_heading,
                "ttl_ms": int(ttl_ms),
                "goal_rgb_path": str(goal_rgb_path),
            },
            "pixelnav_goal": {
                "mask_radius": self.config.mask_radius,
                "mask_nonzero_pixels": int(np.count_nonzero(goal.goal_mask)),
            },
            "rollout": summary.to_json(),
            "start_position_xyz": _float_list(start_position),
            "final_position_xyz": _float_list(final_position),
            "final_position_delta_m": round(_distance(start_position, final_position), 6),
            "artifacts": {
                "goal_rgb": str(goal_rgb_path),
                "goal_rgb_selected_point": str(goal_overlay_path),
                "overlays_dir": str(overlays_dir),
            },
        }
        result_path = rollout_dir / "pixelnav_rollout_result.json"
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        return build_action_outcome_from_rollout(
            summary,
            selected_view=view_type,
            selected_image_point=point_px,
            source_raw={
                "source": "habitat_pixelnav_backend",
                "rollout_result_json": str(result_path),
                "selected_view_id": int(view_id),
                "ttl_ms": int(ttl_ms),
                "pixelnav_goal": result["pixelnav_goal"],
                "final_position_delta_m": result["final_position_delta_m"],
            },
        )

    def _make_default_executor(self) -> Any:
        from goal_adapter.pixelnav_bridge import PixelNavPolicyExecutor

        return PixelNavPolicyExecutor(
            pixelnav_root=self.config.pixelnav_root,
            checkpoint_path=self.config.checkpoint_path,
        )

    def _make_default_runner(self, config: HabitatPixelNavBackendConfig) -> Any:
        from goal_adapter.habitat_pixelnav_execute import (
            HabitatPixelNavRolloutConfig,
            HabitatSimPixelNavRunner,
            _create_pixelnav_simulator,
        )
        from goal_adapter.habitat_sim_smoke import _import_habitat_sim

        habitat_sim = _import_habitat_sim()
        rollout_config = HabitatPixelNavRolloutConfig(
            audit_result_path="",
            output_dir=config.output_dir,
            max_steps=config.max_steps,
            mask_radius=config.mask_radius,
            pixelnav_root=config.pixelnav_root,
            checkpoint_path=config.checkpoint_path,
            image_width=config.image_width,
            image_height=config.image_height,
            sensor_height=config.sensor_height,
            image_hfov=config.image_hfov,
            forward_step_size=config.forward_step_size,
            turn_angle=config.turn_angle,
        )
        self._sim = _create_pixelnav_simulator(rollout_config, str(config.scene_path), habitat_sim)
        self._agent = self._sim.initialize_agent(0)
        return HabitatSimPixelNavRunner(self._sim, self._agent)

    def close(self) -> None:
        close = getattr(self._sim, "close", None)
        if callable(close):
            close()


def _import_memory_type(name: str) -> Any:
    try:
        from nav_memory_qwen import schema
    except ModuleNotFoundError:
        qwen_root = Path(__file__).resolve().parents[1] / "qwen_nav_memory_framework_v5"
        if str(qwen_root) not in sys.path:
            sys.path.insert(0, str(qwen_root))
        from nav_memory_qwen import schema
    return getattr(schema, name)


def _import_robot_backend_type(name: str) -> Any:
    try:
        from nav_memory_qwen import robot_backend
    except ModuleNotFoundError:
        qwen_root = Path(__file__).resolve().parents[1] / "qwen_nav_memory_framework_v5"
        if str(qwen_root) not in sys.path:
            sys.path.insert(0, str(qwen_root))
        from nav_memory_qwen import robot_backend
    return getattr(robot_backend, name)


def _float_list(values: Any) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _rgb3(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("RGB observation must have shape (H, W, 3+)")
    return np.ascontiguousarray(array[:, :, :3].astype(np.uint8))
