from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from goal_adapter.habitat_pixelnav_backend import HabitatPixelNavBackend, HabitatPixelNavBackendConfig
from goal_adapter.habitat_pixelnav_execute import action_outcome_to_json


@dataclass(frozen=True)
class HabitatPixelNavMemorySmokeConfig:
    scene_path: str | Path
    output_dir: str | Path
    start_position_xyz: list[float]
    goal_map_xy: list[float]
    start_heading_rad: float = 0.0
    selected_view_type: str = "front"
    selected_image_point: tuple[int, int] = (320, 300)
    initial_yaw_offsets_deg: tuple[float, ...] = (0.0, -90.0, 90.0, 180.0)
    max_agent_steps: int = 1
    max_pixelnav_steps: int = 12
    force_front_view_waypoint: bool = False
    image_width: int = 640
    image_height: int = 480
    sensor_height: float = 0.88
    image_hfov: float = 79.0
    mask_radius: int = 5
    pixelnav_root: str | Path = "/home/icra/Pixel-Navigator"
    checkpoint_path: str | Path = "checkpoints/navigator.pth"


class FixedGoVLMClient:
    def __init__(self, *, selected_view_type: str, selected_image_point: Sequence[int]):
        self.selected_view_type = str(selected_view_type)
        self.selected_image_point = (int(selected_image_point[0]), int(selected_image_point[1]))

    def decide(self, vlm_input: dict[str, Any]) -> dict[str, Any]:
        make_go_output = _import_schema_type("make_go_output")
        observation = vlm_input["observation"]
        views = list(observation.get("views", []))
        view = next((item for item in views if item.get("view_type") == self.selected_view_type), views[0])
        width = int(observation.get("image_width", 640))
        height = int(observation.get("image_height", 480))
        u = max(0, min(width - 1, int(self.selected_image_point[0])))
        v = max(0, min(height - 1, int(self.selected_image_point[1])))
        return make_go_output(
            view_id=int(view.get("view_id", 0)),
            view_type=str(view.get("view_type", "front")),
            point_px=(u, v),
            width=width,
            height=height,
            decision_reason="G02_VISIBLE_FLOOR_TOWARD_GOAL",
            goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
            short_text=f"fixed smoke waypoint in {view.get('view_type', 'front')} view",
            confidence="high",
        )


def load_memory_smoke_config_from_audit(
    audit_result_path: str | Path,
    *,
    output_dir: str | Path,
    max_agent_steps: int = 1,
    force_front_view_waypoint: bool = False,
) -> HabitatPixelNavMemorySmokeConfig:
    payload = json.loads(Path(audit_result_path).read_text(encoding="utf-8"))
    selected_point = payload.get("selected_image_point")
    if not isinstance(selected_point, list) or len(selected_point) != 2:
        raise ValueError("audit result must contain selected_image_point [u, v]")
    coarse_goal_xy = payload.get("coarse_goal_xy")
    if not isinstance(coarse_goal_xy, list) or len(coarse_goal_xy) != 2:
        raise ValueError("audit result must contain coarse_goal_xy [x, y]")
    return HabitatPixelNavMemorySmokeConfig(
        scene_path=str(payload["scene_path"]),
        output_dir=output_dir,
        start_position_xyz=[float(value) for value in payload["robot_position_xyz"]],
        start_heading_rad=float(payload.get("heading", 0.0)),
        goal_map_xy=[float(coarse_goal_xy[0]), float(coarse_goal_xy[1])],
        selected_view_type=str(payload.get("selected_view", "front")),
        selected_image_point=(int(selected_point[0]), int(selected_point[1])),
        max_agent_steps=max_agent_steps,
        force_front_view_waypoint=force_front_view_waypoint,
    )


def run_habitat_pixelnav_memory_smoke(
    config: HabitatPixelNavMemorySmokeConfig,
    *,
    runner_factory: Any | None = None,
    executor_factory: Any | None = None,
    backend: Any | None = None,
    vlm_client: Any | None = None,
) -> dict[str, Any]:
    NavMemoryAgent = _import_agent_type("NavMemoryAgent")
    NavAgentConfig = _import_agent_type("NavAgentConfig")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    robot = backend or HabitatPixelNavBackend(
        HabitatPixelNavBackendConfig(
            scene_path=config.scene_path,
            output_dir=output_dir / "backend",
            start_position_xyz=list(config.start_position_xyz),
            start_heading_rad=config.start_heading_rad,
            image_width=config.image_width,
            image_height=config.image_height,
            sensor_height=config.sensor_height,
            image_hfov=config.image_hfov,
            max_steps=config.max_pixelnav_steps,
            mask_radius=config.mask_radius,
            pixelnav_root=config.pixelnav_root,
            checkpoint_path=config.checkpoint_path,
        ),
        runner_factory=runner_factory,
        executor_factory=executor_factory,
    )
    client = vlm_client or FixedGoVLMClient(
        selected_view_type=config.selected_view_type,
        selected_image_point=config.selected_image_point,
    )
    agent = NavMemoryAgent(
        robot=robot,
        vlm_client=client,
        config=NavAgentConfig(
            max_steps=config.max_agent_steps,
            force_front_view_waypoint=config.force_front_view_waypoint,
            pose_noise_enabled=False,
        ),
    )
    if config.initial_yaw_offsets_deg:
        agent.pending_observation = robot.capture_views(
            config.initial_yaw_offsets_deg,
            mode="directed_sweep",
        )
    try:
        episode = agent.run_until_done(
            goal_map_xy=(float(config.goal_map_xy[0]), float(config.goal_map_xy[1])),
            max_steps=config.max_agent_steps,
        )
        steps = [_step_result_to_json(step) for step in episode.step_results]
        timeline = _timeline_to_json(steps)
        run_dir = output_dir / "agent_run"
        agent.save_run(run_dir)
        summary = {
            "config": {
                "scene_path": str(config.scene_path),
                "start_position_xyz": list(config.start_position_xyz),
                "start_heading_rad": config.start_heading_rad,
                "goal_map_xy": list(config.goal_map_xy),
                "selected_view_type": config.selected_view_type,
                "selected_image_point": list(config.selected_image_point),
                "max_agent_steps": config.max_agent_steps,
                "max_pixelnav_steps": config.max_pixelnav_steps,
                "force_front_view_waypoint": config.force_front_view_waypoint,
            },
            "policy_checks": _policy_checks_to_json(config, steps),
            "outcome_checks": _outcome_checks_to_json(steps, episode.graph),
            "memory_context_checks": _memory_context_checks_to_json(steps),
            "episode": episode.summary(),
            "steps": steps,
            "timeline": timeline,
            "timeline_summary": _timeline_summary_to_json(timeline),
            "memory": {
                "schema_version": episode.graph.to_dict().get("schema_version"),
                "num_nodes": len(episode.graph.nodes),
                "num_edges": len(episode.graph.edges),
                "current_node_id": episode.graph.current_node_id,
                "live_pose_relation": episode.graph.current_pose_relation_to_latest_node(),
                "temporal_edges": _temporal_edges_to_json(episode.graph),
                "deadlock_state": _deadlock_state_to_json(episode.graph),
                "negative_nodes": _negative_nodes_to_json(episode.graph),
            },
            "artifacts": {
                "output_dir": str(output_dir),
                "backend_output_dir": str(output_dir / "backend"),
                "agent_run_dir": str(run_dir),
                "memory_graph_json": str(run_dir / "memory_graph.json"),
                "steps_json": str(run_dir / "steps.json"),
            },
        }
        summary_path = output_dir / "closed_loop_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary["summary_json"] = str(summary_path)
        return summary
    finally:
        close = getattr(robot, "close", None)
        if callable(close):
            close()


def _timeline_to_json(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline = []
    consecutive_no_progress = 0
    for step in steps:
        outcome = step.get("outcome") or {}
        raw = outcome.get("raw") or {}
        vlm_output = step.get("vlm_output") or {}
        memory_context = step.get("memory_context") or {}
        action = str(step.get("action"))
        no_progress = action == "go" and bool(outcome.get("no_progress"))
        collision = bool(outcome.get("collision"))
        if no_progress:
            consecutive_no_progress += 1
        elif outcome and bool(outcome.get("success")) and action == "go":
            consecutive_no_progress = 0
        memory_signal = "normal"
        if (
            memory_context.get("deadlock_status") == "confirmed"
            or memory_context.get("compressed_negative_memory_count", 0)
            or consecutive_no_progress >= 2
        ):
            memory_signal = "confirmed_negative_memory"
        elif no_progress or memory_context.get("deadlock_status") == "suspected":
            memory_signal = "suspected_no_progress"
        if collision:
            memory_signal = "collision"
        timeline.append(
            {
                "step_index": int(step.get("step_index", -1)),
                "action": action,
                "selected_view": raw.get("selected_view") or vlm_output.get("selected_view_type"),
                "outcome_success": None if not outcome else bool(outcome.get("success")),
                "no_progress": no_progress,
                "collision": collision,
                "moved_distance_m": float(outcome.get("moved_distance_m", 0.0) or 0.0),
                "rotated_deg": float(outcome.get("rotated_deg", 0.0) or 0.0),
                "memory_signal": memory_signal,
                "current_node_id": step.get("current_node_id"),
                "deadlock_status": memory_context.get("deadlock_status"),
                "failed_waypoints_count": int(memory_context.get("failed_waypoints_count", 0) or 0),
                "compressed_negative_memory_count": int(
                    memory_context.get("compressed_negative_memory_count", 0) or 0
                ),
                "message": outcome.get("message"),
            }
        )
    return timeline


def _timeline_summary_to_json(timeline: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_steps": len(timeline),
        "num_go_steps": sum(1 for item in timeline if item.get("action") == "go"),
        "num_rotate_steps": sum(1 for item in timeline if item.get("action") == "rotate"),
        "has_successful_motion": any(
            bool(item.get("outcome_success")) and float(item.get("moved_distance_m", 0.0) or 0.0) > 0.0
            for item in timeline
        ),
        "has_no_progress": any(bool(item.get("no_progress")) for item in timeline),
        "has_confirmed_negative_memory": any(
            item.get("memory_signal") == "confirmed_negative_memory" for item in timeline
        ),
        "total_moved_distance_m": round(
            sum(float(item.get("moved_distance_m", 0.0) or 0.0) for item in timeline),
            6,
        ),
    }


def _outcome_checks_to_json(steps: list[dict[str, Any]], graph: Any) -> dict[str, Any]:
    no_progress_steps = [
        step
        for step in steps
        if (step.get("outcome") or {}).get("no_progress")
    ]
    failed_outcome_steps = [
        step
        for step in steps
        if step.get("outcome") is not None and not bool((step.get("outcome") or {}).get("success"))
    ]
    collision_steps = [
        step
        for step in steps
        if bool((step.get("outcome") or {}).get("collision"))
    ]
    confirmed_negative_nodes = [
        node
        for node in graph.nodes.values()
        if str(node.navigation_state.get("deadlock_status", "none")) == "confirmed"
        and node.negative_memory
    ]
    return {
        "no_progress_detected": bool(no_progress_steps),
        "no_progress_count": len(no_progress_steps),
        "first_no_progress_step_index": (
            None if not no_progress_steps else int(no_progress_steps[0].get("step_index", -1))
        ),
        "failed_outcome_count": len(failed_outcome_steps),
        "collision_count": len(collision_steps),
        "confirmed_negative_memory_detected": bool(confirmed_negative_nodes),
        "confirmed_negative_node_count": len(confirmed_negative_nodes),
    }


def _memory_context_checks_to_json(steps: list[dict[str, Any]]) -> dict[str, Any]:
    contexts = [step.get("memory_context") or {} for step in steps]
    failed_counts = [int(ctx.get("failed_waypoints_count", 0) or 0) for ctx in contexts]
    negative_counts = [int(ctx.get("compressed_negative_memory_count", 0) or 0) for ctx in contexts]
    guard_counts = [int(ctx.get("avoid_candidate_exit_count", 0) or 0) for ctx in contexts]
    merge_policy = next((ctx.get("merge_policy") for ctx in contexts if ctx.get("merge_policy")), None)
    return {
        "failed_waypoints_detected": any(count > 0 for count in failed_counts),
        "compressed_negative_memory_detected": any(count > 0 for count in negative_counts),
        "avoid_candidate_exit_detected": any(count > 0 for count in guard_counts),
        "max_failed_waypoints_count": max(failed_counts, default=0),
        "max_compressed_negative_memory_count": max(negative_counts, default=0),
        "max_avoid_candidate_exit_count": max(guard_counts, default=0),
        "merge_policy": merge_policy,
    }


def _policy_checks_to_json(config: HabitatPixelNavMemorySmokeConfig, steps: list[dict[str, Any]]) -> dict[str, Any]:
    view_type_to_heading_deg = _import_schema_type("view_type_to_heading_deg")
    rotate_step = next((step for step in steps if step.get("action") == "rotate"), None)
    first_go_after_rotate = None
    if rotate_step is not None:
        rotate_index = int(rotate_step.get("step_index", -1))
        first_go_after_rotate = next(
            (
                step
                for step in steps
                if int(step.get("step_index", -1)) > rotate_index and step.get("action") == "go"
            ),
            None,
        )
    deferred_go = (rotate_step or {}).get("vlm_output", {}).get("deferred_go", {})
    rotate_outcome = (rotate_step or {}).get("outcome") or {}
    go_outcome_raw = ((first_go_after_rotate or {}).get("outcome") or {}).get("raw", {})
    first_go_view = go_outcome_raw.get("selected_view")
    requested_heading = float(view_type_to_heading_deg(config.selected_view_type))
    return {
        "force_front_view_waypoint": bool(config.force_front_view_waypoint),
        "requested_view_type": config.selected_view_type,
        "requested_view_heading_deg": requested_heading,
        "rotate_yaw_deg": rotate_outcome.get("rotated_deg"),
        "rotate_first_handoff_verified": bool(
            config.force_front_view_waypoint
            and config.selected_view_type != "front"
            and rotate_step is not None
            and deferred_go.get("selected_view_type") == config.selected_view_type
            and first_go_view == "front"
        ),
        "rotate_step_index": None if rotate_step is None else int(rotate_step.get("step_index", -1)),
        "first_go_after_rotate_step_index": (
            None if first_go_after_rotate is None else int(first_go_after_rotate.get("step_index", -1))
        ),
        "first_go_after_rotate_view_type": first_go_view,
    }


def _deadlock_state_to_json(graph: Any) -> dict[str, Any]:
    current_node_id = graph.current_node_id
    node = graph.nodes.get(current_node_id) if current_node_id else None
    return {
        "current_node_id": current_node_id,
        "status": "none" if node is None else str(node.navigation_state.get("deadlock_status", "none")),
        "risk_level": None if node is None else node.navigation_state.get("risk_level"),
        "memory_importance": None if node is None else node.navigation_state.get("memory_importance"),
        "incoming_edge_id": graph.latest_incoming_edge_id(current_node_id) if current_node_id else None,
        "negative_memory": None if node is None else node.negative_memory,
    }


def _negative_nodes_to_json(graph: Any) -> list[dict[str, Any]]:
    nodes = []
    for node_id, node in graph.nodes.items():
        status = str(node.navigation_state.get("deadlock_status", "none"))
        if not node.negative_memory and status not in {"suspected", "confirmed", "escaping", "confirmed_escaped"}:
            continue
        nodes.append(
            {
                "node_id": node_id,
                "deadlock_status": status,
                "risk_level": node.navigation_state.get("risk_level"),
                "memory_importance": node.navigation_state.get("memory_importance"),
                "negative_memory": node.negative_memory,
            }
        )
    return nodes


def _step_result_to_json(step: Any) -> dict[str, Any]:
    return {
        "step_index": int(step.step_index),
        "action": step.action,
        "done": bool(step.done),
        "success": bool(step.success),
        "warnings": list(step.warnings),
        "current_node_id": step.current_node_id,
        "goal_distance_m": round(float(step.goal_distance_m), 6),
        "memory_context": _vlm_memory_context_to_json(step.vlm_input),
        "vlm_output": step.vlm_output,
        "outcome": None if step.outcome is None else action_outcome_to_json(step.outcome),
        "error": step.error,
    }


def _vlm_memory_context_to_json(vlm_input: dict[str, Any] | None) -> dict[str, Any]:
    memory = (vlm_input or {}).get("memory") or {}
    failed_waypoints = list(memory.get("failed_waypoints") or [])
    compressed_negative = list(memory.get("compressed_negative_memories") or [])
    deadlock_state = memory.get("deadlock_state") or {}
    candidate_exits = list((memory.get("local_topology") or {}).get("candidate_exits") or [])
    avoid_candidates = [candidate for candidate in candidate_exits if candidate.get("avoid")]
    return {
        "schema_version": memory.get("schema_version"),
        "merge_policy": (memory.get("graph_summary") or {}).get("merge_policy"),
        "control_merge_policy": (memory.get("control_policy") or {}).get("merge_policy"),
        "deadlock_status": deadlock_state.get("status"),
        "deadlock_incoming_edge_id": deadlock_state.get("incoming_edge_id"),
        "failed_waypoints_count": len(failed_waypoints),
        "failed_waypoint_ids": [
            item.get("edge_id")
            for item in failed_waypoints
            if item.get("edge_id") is not None
        ],
        "failed_waypoint_statuses": [
            item.get("status") or (item.get("traversal") or {}).get("status")
            for item in failed_waypoints
        ],
        "compressed_negative_memory_count": len(compressed_negative),
        "compressed_negative_node_ids": [
            item.get("node_id")
            for item in compressed_negative
            if item.get("node_id") is not None
        ],
        "candidate_exit_count": len(candidate_exits),
        "avoid_candidate_exit_count": len(avoid_candidates),
        "avoid_candidate_edge_ids": [
            item.get("edge_id")
            for item in avoid_candidates
            if item.get("edge_id") is not None
        ],
    }


def _temporal_edges_to_json(graph: Any) -> list[dict[str, Any]]:
    edges = []
    for edge_id, edge in graph.edges.items():
        if edge.edge_type != "temporal_transition":
            continue
        edges.append(
            {
                "edge_id": edge_id,
                "src_node_id": edge.src_node_id,
                "dst_node_id": edge.dst_node_id,
                "edge_type": edge.edge_type,
                "relation_type": edge.relation_type,
                "status": edge.traversal.get("status"),
                "relative_pose_src_to_dst": edge.relative_pose_src_to_dst.to_dict(),
            }
        )
    return edges


def _import_schema_type(name: str) -> Any:
    try:
        from nav_memory_qwen import schema
    except ModuleNotFoundError:
        qwen_root = Path(__file__).resolve().parents[1] / "qwen_nav_memory_framework_v5"
        if str(qwen_root) not in sys.path:
            sys.path.insert(0, str(qwen_root))
        from nav_memory_qwen import schema
    return getattr(schema, name)


def _import_agent_type(name: str) -> Any:
    try:
        from nav_memory_qwen import agent
    except ModuleNotFoundError:
        qwen_root = Path(__file__).resolve().parents[1] / "qwen_nav_memory_framework_v5"
        if str(qwen_root) not in sys.path:
            sys.path.insert(0, str(qwen_root))
        from nav_memory_qwen import agent
    return getattr(agent, name)
