from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from goal_adapter.habitat_goal_audit import _draw_cross, _read_png_rgb, _rotation_from_heading
from goal_adapter.habitat_multiview_goal_audit import VIEW_OFFSETS
from goal_adapter.habitat_sim_smoke import HabitatSimSmokeConfig, _import_habitat_sim
from goal_adapter.habitat_smoke import write_rgb_png
from goal_adapter.pixelnav_bridge import PixelNavPolicyExecutor


PIXELNAV_ACTION_NAMES = {
    0: None,
    1: "move_forward",
    2: "turn_left",
    3: "turn_right",
    4: "look_up",
    5: "look_down",
}


@dataclass(frozen=True)
class PixelNavStep:
    step_index: int
    pixelnav_action: int
    sim_action: str | None
    collision_before: bool
    position_before_xyz: list[float]
    position_after_xyz: list[float]
    moved_distance_m: float
    overlay_path: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "pixelnav_action": self.pixelnav_action,
            "sim_action": self.sim_action,
            "collision_before": self.collision_before,
            "position_before_xyz": list(self.position_before_xyz),
            "position_after_xyz": list(self.position_after_xyz),
            "moved_distance_m": self.moved_distance_m,
            "overlay_path": self.overlay_path,
        }


@dataclass(frozen=True)
class PixelNavRolloutSummary:
    steps: list[PixelNavStep]
    total_moved_distance_m: float
    stopped_by_policy: bool
    unsupported_action: int | None
    collision_count: int

    def to_json(self) -> dict[str, Any]:
        return {
            "steps": [step.to_json() for step in self.steps],
            "total_moved_distance_m": self.total_moved_distance_m,
            "stopped_by_policy": self.stopped_by_policy,
            "unsupported_action": self.unsupported_action,
            "collision_count": self.collision_count,
        }


@dataclass(frozen=True)
class PixelNavRolloutRequest:
    scene_path: str
    robot_position_xyz: list[float]
    base_heading: float
    selected_view: str
    selected_image_point: list[int]
    rollout_heading: float
    goal_rgb_path: str

    def to_json(self) -> dict[str, Any]:
        return {
            "scene_path": self.scene_path,
            "robot_position_xyz": list(self.robot_position_xyz),
            "base_heading": self.base_heading,
            "selected_view": self.selected_view,
            "selected_image_point": list(self.selected_image_point),
            "rollout_heading": self.rollout_heading,
            "goal_rgb_path": self.goal_rgb_path,
        }


def build_action_outcome_from_rollout(
    summary: PixelNavRolloutSummary,
    *,
    action: str = "go",
    min_success_moved_distance: float = 0.20,
    selected_view: str | None = None,
    selected_image_point: Sequence[int] | None = None,
    source_raw: dict[str, Any] | None = None,
) -> Any:
    ActionOutcome, RelativePose2D = _import_memory_action_types()
    moved_distance_m = round(float(summary.total_moved_distance_m), 6)
    collision = bool(summary.collision_count > 0 or any(step.collision_before for step in summary.steps))
    has_steps = bool(summary.steps)
    unsupported_action = summary.unsupported_action
    success = (
        has_steps
        and not collision
        and unsupported_action is None
        and moved_distance_m >= float(min_success_moved_distance)
        and (not summary.stopped_by_policy or moved_distance_m >= float(min_success_moved_distance))
    )
    local_waypoint_reached = success and (
        summary.stopped_by_policy or moved_distance_m >= float(min_success_moved_distance)
    )

    raw: dict[str, Any] = {
        "rollout": summary.to_json(),
        "collision_count": int(summary.collision_count),
        "stopped_by_policy": bool(summary.stopped_by_policy),
        "unsupported_action": unsupported_action,
        "local_waypoint_reached": local_waypoint_reached,
        "min_success_moved_distance": float(min_success_moved_distance),
    }
    if selected_view is not None:
        raw["selected_view"] = selected_view
    if selected_image_point is not None:
        raw["selected_image_point"] = [int(selected_image_point[0]), int(selected_image_point[1])]
    if source_raw:
        raw.update(source_raw)

    message = _action_outcome_message(
        summary=summary,
        success=success,
        collision=collision,
        moved_distance_m=moved_distance_m,
        min_success_moved_distance=float(min_success_moved_distance),
    )
    return ActionOutcome(
        action=action,
        success=success,
        collision=collision,
        moved_distance_m=moved_distance_m,
        rotated_deg=0.0,
        odom_delta=RelativePose2D(dx_m=moved_distance_m, dy_m=0.0, dyaw_deg=0.0),
        message=message,
        raw=raw,
    )


def build_action_outcome_from_rollout_result(
    rollout_result: str | Path | dict[str, Any],
    *,
    action: str = "go",
    min_success_moved_distance: float = 0.20,
) -> Any:
    result_path: Path | None = None
    if isinstance(rollout_result, (str, Path)):
        result_path = Path(rollout_result)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    elif isinstance(rollout_result, dict):
        payload = rollout_result
    else:
        raise TypeError("rollout_result must be a path or JSON-like dict")

    summary = _rollout_summary_from_result_payload(payload)
    request = payload.get("request", {})
    selected_view = request.get("selected_view") if isinstance(request, dict) else None
    selected_point = request.get("selected_image_point") if isinstance(request, dict) else None
    source_raw: dict[str, Any] = {
        "source": "pixelnav_rollout_result",
    }
    if result_path is not None:
        source_raw["rollout_result_json"] = str(result_path)
    if "audit_result_path" in payload:
        source_raw["audit_result_path"] = payload["audit_result_path"]
    if "pixelnav_goal" in payload:
        source_raw["pixelnav_goal"] = payload["pixelnav_goal"]
    if "final_position_delta_m" in payload:
        source_raw["final_position_delta_m"] = float(payload["final_position_delta_m"])

    return build_action_outcome_from_rollout(
        summary,
        action=action,
        min_success_moved_distance=min_success_moved_distance,
        selected_view=str(selected_view) if selected_view is not None else None,
        selected_image_point=selected_point if _is_two_element_sequence(selected_point) else None,
        source_raw=source_raw,
    )


def action_outcome_to_json(outcome: Any) -> dict[str, Any]:
    odom_delta = outcome.odom_delta
    return {
        "action": outcome.action,
        "success": bool(outcome.success),
        "collision": bool(outcome.collision),
        "no_progress": bool(outcome.no_progress),
        "moved_distance_m": round(float(outcome.moved_distance_m), 6),
        "rotated_deg": round(float(outcome.rotated_deg), 6),
        "odom_delta": {
            "dx_m": round(float(odom_delta.dx_m), 6),
            "dy_m": round(float(odom_delta.dy_m), 6),
            "dyaw_deg": round(float(odom_delta.dyaw_deg), 6),
        },
        "message": str(outcome.message),
        "raw": outcome.raw,
    }


def load_rollout_request_from_audit(audit_result_path: str | Path) -> PixelNavRolloutRequest:
    payload = json.loads(Path(audit_result_path).read_text(encoding="utf-8"))
    selected_view = payload.get("selected_view")
    if selected_view not in VIEW_OFFSETS:
        raise ValueError("audit result must contain a valid selected_view")
    selected_point = payload.get("selected_image_point")
    if not isinstance(selected_point, list) or len(selected_point) != 2:
        raise ValueError("audit result must contain selected_image_point [u, v]")
    views = payload.get("views")
    if not isinstance(views, dict) or selected_view not in views:
        raise ValueError("audit result must contain selected view RGB path")
    selected_view_payload = views[selected_view]
    if not isinstance(selected_view_payload, dict) or "rgb" not in selected_view_payload:
        raise ValueError("selected view payload must contain rgb path")

    base_heading = float(payload.get("heading", 0.0))
    return PixelNavRolloutRequest(
        scene_path=str(payload["scene_path"]),
        robot_position_xyz=_float_list(payload["robot_position_xyz"]),
        base_heading=base_heading,
        selected_view=str(selected_view),
        selected_image_point=[int(selected_point[0]), int(selected_point[1])],
        rollout_heading=base_heading + VIEW_OFFSETS[str(selected_view)],
        goal_rgb_path=str(selected_view_payload["rgb"]),
    )


def pixelnav_action_to_sim_action(
    pixelnav_action: int,
    supported_action_names: set[str] | frozenset[str],
) -> str | None:
    action_name = PIXELNAV_ACTION_NAMES.get(int(pixelnav_action))
    if action_name is None:
        return None
    return action_name if action_name in supported_action_names else None


def run_pixelnav_rollout_loop(
    executor: Any,
    runner: Any,
    max_steps: int,
    overlay_dir: str | Path | None = None,
) -> PixelNavRolloutSummary:
    steps: list[PixelNavStep] = []
    stopped_by_policy = False
    unsupported_action = None
    overlay_dir = Path(overlay_dir) if overlay_dir is not None else None
    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    for step_index in range(max_steps):
        obs_rgb = _rgb3(runner.rgb())
        position_before = _float_list(runner.position_xyz())
        collision_before = bool(runner.previous_step_collided())
        policy_step = executor.step(obs_rgb, collide=collision_before)
        pixelnav_action = int(policy_step.action)

        overlay_path = None
        if overlay_dir is not None:
            overlay_path = str(overlay_dir / f"overlay_{step_index:03d}.png")
            write_rgb_png(_rgb3(policy_step.overlay_image), overlay_path)

        sim_action = pixelnav_action_to_sim_action(
            pixelnav_action,
            set(runner.supported_action_names()),
        )
        if pixelnav_action == 0:
            stopped_by_policy = True
            position_after = position_before
        elif sim_action is None:
            unsupported_action = pixelnav_action
            position_after = position_before
        else:
            runner.step(sim_action)
            position_after = _float_list(runner.position_xyz())

        steps.append(
            PixelNavStep(
                step_index=step_index,
                pixelnav_action=pixelnav_action,
                sim_action=sim_action,
                collision_before=collision_before,
                position_before_xyz=position_before,
                position_after_xyz=position_after,
                moved_distance_m=_distance(position_before, position_after),
                overlay_path=overlay_path,
            )
        )

        if stopped_by_policy or unsupported_action is not None:
            break

    return PixelNavRolloutSummary(
        steps=steps,
        total_moved_distance_m=round(sum(step.moved_distance_m for step in steps), 6),
        stopped_by_policy=stopped_by_policy,
        unsupported_action=unsupported_action,
        collision_count=sum(1 for step in steps if step.collision_before),
    )


@dataclass(frozen=True)
class HabitatPixelNavRolloutConfig:
    audit_result_path: str | Path
    output_dir: str | Path
    max_steps: int = 12
    mask_radius: int = 5
    pixelnav_root: str | Path = "/home/icra/Pixel-Navigator"
    checkpoint_path: str | Path = "checkpoints/navigator.pth"
    image_width: int = 640
    image_height: int = 480
    sensor_height: float = 0.88
    image_hfov: float = 79.0
    forward_step_size: float = 0.25
    turn_angle: float = 30.0


class HabitatSimPixelNavRunner:
    def __init__(self, sim: Any, agent: Any):
        self.sim = sim
        self.agent = agent

    def set_pose(self, position_xyz: Sequence[float], heading: float) -> None:
        state = self.agent.get_state()
        state.position = list(position_xyz)
        state.rotation = _rotation_from_heading(heading)
        self.agent.set_state(state)

    def rgb(self) -> np.ndarray:
        return _rgb3(self.sim.get_sensor_observations()["rgb"])

    def previous_step_collided(self) -> bool:
        return bool(getattr(self.sim, "previous_step_collided", False))

    def position_xyz(self) -> list[float]:
        return _float_list(self.agent.get_state().position)

    def supported_action_names(self) -> set[str]:
        return set(self.agent.agent_config.action_space.keys())

    def step(self, action_name: str) -> None:
        self.sim.step(action_name)


def run_habitat_pixelnav_rollout_from_audit(
    config: HabitatPixelNavRolloutConfig,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    request = load_rollout_request_from_audit(config.audit_result_path)
    habitat_sim = _import_habitat_sim()
    sim = _create_pixelnav_simulator(config, request.scene_path, habitat_sim)
    try:
        agent = sim.initialize_agent(0)
        runner = HabitatSimPixelNavRunner(sim, agent)
        runner.set_pose(request.robot_position_xyz, request.rollout_heading)

        initial_rgb = runner.rgb()
        initial_rgb_path = output_dir / "initial_selected_view_rgb.png"
        write_rgb_png(initial_rgb, initial_rgb_path)

        goal_rgb = np.asarray(Image.open(request.goal_rgb_path).convert("RGB"))
        goal_overlay = draw_selected_point_overlay(goal_rgb, request.selected_image_point)
        goal_overlay_path = output_dir / "goal_rgb_selected_point.png"
        write_rgb_png(goal_overlay, goal_overlay_path)

        executor = PixelNavPolicyExecutor(
            pixelnav_root=config.pixelnav_root,
            checkpoint_path=config.checkpoint_path,
        )
        goal = executor.reset_from_point(
            goal_rgb,
            request.selected_image_point,
            radius=config.mask_radius,
        )
        overlays_dir = output_dir / "overlays"
        summary = run_pixelnav_rollout_loop(
            executor,
            runner,
            max_steps=config.max_steps,
            overlay_dir=overlays_dir,
        )
        final_position = runner.position_xyz()
    finally:
        close = getattr(sim, "close", None)
        if callable(close):
            close()

    contact_sheet_path = output_dir / "rollout_contact_sheet.png"
    _write_rollout_contact_sheet(
        goal_overlay_path=goal_overlay_path,
        initial_rgb_path=initial_rgb_path,
        overlay_paths=[Path(step.overlay_path) for step in summary.steps if step.overlay_path],
        output_path=contact_sheet_path,
    )

    result = {
        "audit_result_path": str(config.audit_result_path),
        "request": request.to_json(),
        "pixelnav_goal": {
            "mask_radius": config.mask_radius,
            "mask_nonzero_pixels": int(np.count_nonzero(goal.goal_mask)),
        },
        "rollout": summary.to_json(),
        "start_position_xyz": request.robot_position_xyz,
        "final_position_xyz": final_position,
        "final_position_delta_m": round(_distance(request.robot_position_xyz, final_position), 6),
        "artifacts": {
            "initial_selected_view_rgb": str(initial_rgb_path),
            "goal_rgb_selected_point": str(goal_overlay_path),
            "rollout_contact_sheet": str(contact_sheet_path),
            "overlays_dir": str(overlays_dir),
        },
    }
    result_path = output_dir / "pixelnav_rollout_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result["result_json"] = str(result_path)
    return result


def _create_pixelnav_simulator(
    config: HabitatPixelNavRolloutConfig,
    scene_path: str,
    habitat_sim: Any,
) -> Any:
    simulator_config = habitat_sim.SimulatorConfiguration()
    simulator_config.scene_id = str(scene_path)

    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "rgb"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [config.image_height, config.image_width]
    sensor_spec.position = [0.0, config.sensor_height, 0.0]
    if hasattr(sensor_spec, "hfov"):
        sensor_spec.hfov = config.image_hfov

    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = [sensor_spec]
    agent_config.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward",
            habitat_sim.agent.ActuationSpec(amount=config.forward_step_size),
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left",
            habitat_sim.agent.ActuationSpec(amount=config.turn_angle),
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right",
            habitat_sim.agent.ActuationSpec(amount=config.turn_angle),
        ),
        "look_up": habitat_sim.agent.ActionSpec(
            "look_up",
            habitat_sim.agent.ActuationSpec(amount=config.turn_angle),
        ),
        "look_down": habitat_sim.agent.ActionSpec(
            "look_down",
            habitat_sim.agent.ActuationSpec(amount=config.turn_angle),
        ),
    }
    return habitat_sim.Simulator(habitat_sim.Configuration(simulator_config, [agent_config]))


def _write_rollout_contact_sheet(
    goal_overlay_path: Path,
    initial_rgb_path: Path,
    overlay_paths: Sequence[Path],
    output_path: Path,
) -> None:
    labeled_images = [
        ("goal image + VLM point", goal_overlay_path),
        ("rollout initial RGB", initial_rgb_path),
    ]
    labeled_images.extend((f"policy overlay {index}", path) for index, path in enumerate(overlay_paths[:10]))
    images = [_labeled_image(label, path) for label, path in labeled_images]
    if not images:
        Image.new("RGB", (1, 1), "white").save(output_path)
        return
    width, height = images[0].size
    columns = 2
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (width * columns, height * rows), "white")
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns) * width, (index // columns) * height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _labeled_image(label: str, image_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB").resize((320, 240))
    canvas = Image.new("RGB", (320, 272), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, 320, 32], fill=(245, 245, 245))
    draw.text((8, 9), label, fill=(0, 0, 0))
    canvas.paste(image, (0, 32))
    return canvas


def draw_selected_point_overlay(rgb: np.ndarray, selected_point: Sequence[int]) -> np.ndarray:
    overlay = _rgb3(rgb).copy()
    image_list = overlay.tolist()
    _draw_cross(image_list, selected_point, (220, 38, 38), 16)
    return np.asarray(image_list, dtype=np.uint8)


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _action_outcome_message(
    *,
    summary: PixelNavRolloutSummary,
    success: bool,
    collision: bool,
    moved_distance_m: float,
    min_success_moved_distance: float,
) -> str:
    action_counts: dict[str, int] = {}
    for step in summary.steps:
        name = step.sim_action or f"pixelnav_action_{step.pixelnav_action}"
        action_counts[name] = action_counts.get(name, 0) + 1
    action_summary = ", ".join(f"{name} x{count}" for name, count in sorted(action_counts.items()))
    if not action_summary:
        return "Invalid PixelNav rollout: empty step log."
    if summary.unsupported_action is not None:
        return f"PixelNav rollout failed: unsupported action {summary.unsupported_action} after {action_summary}."
    if collision:
        return f"PixelNav rollout failed: collision detected after {action_summary}."
    if moved_distance_m < min_success_moved_distance:
        return (
            "PixelNav rollout failed: moved "
            f"{moved_distance_m:.3f} m below {min_success_moved_distance:.3f} m threshold."
        )
    status = "completed" if success else "failed"
    return (
        f"PixelNav rollout {status}: {action_summary}, moved {moved_distance_m:.3f} m, "
        f"collision={summary.collision_count}."
    )


def _rollout_summary_from_result_payload(payload: dict[str, Any]) -> PixelNavRolloutSummary:
    rollout = payload.get("rollout")
    if not isinstance(rollout, dict):
        raise ValueError("rollout result must contain a rollout object")
    step_payloads = rollout.get("steps", [])
    if not isinstance(step_payloads, list):
        raise ValueError("rollout.steps must be a list")
    steps = [_pixelnav_step_from_json(step) for step in step_payloads]
    total_moved = rollout.get("total_moved_distance_m")
    if total_moved is None:
        total_moved = sum(step.moved_distance_m for step in steps)
    unsupported_action = rollout.get("unsupported_action")
    return PixelNavRolloutSummary(
        steps=steps,
        total_moved_distance_m=round(float(total_moved), 6),
        stopped_by_policy=bool(rollout.get("stopped_by_policy", False)),
        unsupported_action=None if unsupported_action is None else int(unsupported_action),
        collision_count=int(rollout.get("collision_count", sum(1 for step in steps if step.collision_before))),
    )


def _pixelnav_step_from_json(payload: dict[str, Any]) -> PixelNavStep:
    if not isinstance(payload, dict):
        raise ValueError("rollout step must be an object")
    before = _float_list(payload.get("position_before_xyz", [0.0, 0.0, 0.0]))
    after = _float_list(payload.get("position_after_xyz", before))
    moved_distance = payload.get("moved_distance_m")
    if moved_distance is None:
        moved_distance = _distance(before, after)
    sim_action = payload.get("sim_action")
    return PixelNavStep(
        step_index=int(payload.get("step_index", 0)),
        pixelnav_action=int(payload.get("pixelnav_action", 0)),
        sim_action=str(sim_action) if sim_action is not None else None,
        collision_before=bool(payload.get("collision_before", False)),
        position_before_xyz=before,
        position_after_xyz=after,
        moved_distance_m=float(moved_distance),
        overlay_path=payload.get("overlay_path"),
    )


def _import_memory_action_types() -> tuple[Any, Any]:
    try:
        from nav_memory_qwen.robot_backend import ActionOutcome
        from nav_memory_qwen.schema import RelativePose2D
    except ModuleNotFoundError:
        qwen_root = Path(__file__).resolve().parents[1] / "qwen_nav_memory_framework_v5"
        if str(qwen_root) not in sys.path:
            sys.path.insert(0, str(qwen_root))
        from nav_memory_qwen.robot_backend import ActionOutcome
        from nav_memory_qwen.schema import RelativePose2D
    return ActionOutcome, RelativePose2D


def _is_two_element_sequence(value: Any) -> bool:
    return not isinstance(value, (str, bytes, bytearray)) and hasattr(value, "__len__") and len(value) == 2


def _float_list(values: Any) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]


def _rgb3(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("RGB observation must have shape (H, W, 3+)")
    return np.ascontiguousarray(array[:, :, :3].astype(np.uint8))
