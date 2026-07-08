from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from goal_adapter.habitat_pixelnav_closed_loop import (
    HabitatPixelNavMemorySmokeConfig,
    load_memory_smoke_config_from_audit,
)
from goal_adapter.habitat_pixelnav_real_vlm_gated_execution import (
    run_habitat_pixelnav_real_vlm_gated_execution,
    run_habitat_pixelnav_real_vlm_gated_execution_from_env,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update v5 memory from one gated real-VLM PixelNav ActionOutcome.")
    parser.add_argument("--audit-result", required=True, help="Step B farthest goal audit result JSON")
    parser.add_argument("--out", required=True, help="Output directory for real_vlm_memory_update.json")
    parser.add_argument("--force-front-view-waypoint", action="store_true")
    args = parser.parse_args(argv)

    config = load_memory_smoke_config_from_audit(
        args.audit_result,
        output_dir=args.out,
        max_agent_steps=1,
        force_front_view_waypoint=bool(args.force_front_view_waypoint),
    )
    summary = run_habitat_pixelnav_real_vlm_memory_update_from_env(config)
    print(json.dumps({"status": summary.get("status"), "summary_json": summary.get("summary_json")}, ensure_ascii=False))
    return 0


def run_habitat_pixelnav_real_vlm_memory_update_from_env(
    config: HabitatPixelNavMemorySmokeConfig,
    *,
    runner_factory: Any | None = None,
    executor_factory: Any | None = None,
) -> dict[str, Any]:
    gated = run_habitat_pixelnav_real_vlm_gated_execution_from_env(
        config,
        runner_factory=runner_factory,
        executor_factory=executor_factory,
    )
    return update_memory_after_gated_execution(config, gated)


def run_habitat_pixelnav_real_vlm_memory_update(
    config: HabitatPixelNavMemorySmokeConfig,
    *,
    runner_factory: Any | None = None,
    executor_factory: Any | None = None,
    vlm_client: Any,
) -> dict[str, Any]:
    gated = run_habitat_pixelnav_real_vlm_gated_execution(
        config,
        runner_factory=runner_factory,
        executor_factory=executor_factory,
        vlm_client=vlm_client,
    )
    return update_memory_after_gated_execution(config, gated)


def update_memory_after_gated_execution(
    config: HabitatPixelNavMemorySmokeConfig,
    gated: dict[str, Any],
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not bool((gated.get("execution") or {}).get("executed")):
        return _write_memory_update_summary(
            output_dir,
            _blocked_summary(config, gated, reason="gated_execution_not_executed"),
        )

    outcome_json = (gated.get("execution") or {}).get("action_outcome")
    if not isinstance(outcome_json, dict):
        return _write_memory_update_summary(
            output_dir,
            _blocked_summary(config, gated, reason="missing_action_outcome"),
        )

    outcome = _action_outcome_from_json(outcome_json)
    image_path = _memory_replay_image_path(output_dir, gated)
    robot = _ReplayMemoryBackend(
        image_path=image_path,
        image_width=int(config.image_width),
        image_height=int(config.image_height),
        start_position_xyz=list(config.start_position_xyz),
        start_heading_rad=float(config.start_heading_rad),
    )
    NavMemoryAgent, NavAgentConfig = _import_agent_types()
    agent = NavMemoryAgent(
        robot=robot,
        vlm_client=_NoopVLMClient(),
        config=NavAgentConfig(
            max_steps=1,
            pose_noise_enabled=False,
            force_front_view_waypoint=config.force_front_view_waypoint,
            force_new_node_translation_m=0.2,
        ),
    )

    state = robot.get_robot_state()
    goal = agent._build_goal(goal_map_xy=(float(config.goal_map_xy[0]), float(config.goal_map_xy[1])), target_object=None)
    observation = agent._get_observation()
    embedding = agent._embed_front_or_first(observation)
    start_node_id = agent._update_memory_before_decision(observation, embedding, state)

    robot.apply_outcome(outcome)
    agent._update_memory_after_outcome(outcome, observation.frame_index)

    commit_observation = agent._get_observation()
    commit_embedding = agent._embed_front_or_first(commit_observation)
    committed_node_id = agent._update_memory_before_decision(
        commit_observation,
        commit_embedding,
        robot.get_robot_state(),
    )

    run_dir = output_dir / "memory_update_run"
    agent.save_run(run_dir)
    summary = _base_summary(config, gated)
    summary["status"] = "memory_updated"
    summary["memory_update"] = {
        "updated": True,
        "reason": "action_outcome_applied_to_v5_memory",
        "start_node_id": start_node_id,
        "committed_node_id": committed_node_id,
        "action_outcome": outcome_json,
    }
    graph = agent.memory
    summary["memory"] = {
        "schema_version": graph.to_dict().get("schema_version"),
        "num_nodes": len(graph.nodes),
        "num_edges": len(graph.edges),
        "current_node_id": graph.current_node_id,
        "live_pose_relation": graph.current_pose_relation_to_latest_node(),
        "temporal_edges": _temporal_edges_to_json(graph),
        "deadlock_state": _deadlock_state_to_json(graph),
    }
    summary["artifacts"]["memory_update_run_dir"] = str(run_dir)
    summary["artifacts"]["memory_graph_json"] = str(run_dir / "memory_graph.json")
    summary["artifacts"]["steps_json"] = str(run_dir / "steps.json")
    return _write_memory_update_summary(output_dir, summary)


class _NoopVLMClient:
    def decide(self, vlm_input: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("memory replay should not call VLM")


class _ReplayMemoryBackend:
    def __init__(
        self,
        *,
        image_path: str,
        image_width: int,
        image_height: int,
        start_position_xyz: list[float],
        start_heading_rad: float,
    ):
        self.image_path = str(image_path)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.position_xyz = [float(v) for v in start_position_xyz]
        self.heading_rad = float(start_heading_rad)
        self.frame_index = 0

    def get_robot_state(self) -> Any:
        RobotState = _import_schema_type("RobotState")
        return RobotState(
            map_xy=(float(self.position_xyz[0]), float(self.position_xyz[2])),
            heading_rad=float(self.heading_rad),
            position_xyz=tuple(self.position_xyz),
        )

    def get_observation(self) -> Any:
        return self.capture_views([0.0], mode="current_only")

    def capture_views(self, yaw_offsets_deg: list[float] | tuple[float, ...], mode: str = "directed_sweep") -> Any:
        Observation = _import_schema_type("Observation")
        ObservationView = _import_schema_type("ObservationView")
        nearest_view_type = _import_schema_type("nearest_view_type")
        self.frame_index += 1
        ts = int(time.time() * 1000)
        views = []
        used_view_types = set()
        for index, yaw in enumerate(yaw_offsets_deg):
            view_type = nearest_view_type(float(yaw))
            if view_type in used_view_types:
                continue
            used_view_types.add(view_type)
            views.append(
                ObservationView(
                    view_id=index,
                    view_type=view_type,
                    relative_heading_deg=float(yaw),
                    image=self.image_path,
                    timestamp_ms=ts,
                )
            )
        return Observation(
            mode=mode,
            sequence_id="real_vlm_memory_replay",
            frame_index=self.frame_index,
            image_width=self.image_width,
            image_height=self.image_height,
            views=views,
            timestamp_ms=ts,
        )

    def apply_outcome(self, outcome: Any) -> None:
        dx = float(outcome.odom_delta.dx_m)
        dy = float(outcome.odom_delta.dy_m)
        heading = float(self.heading_rad)
        self.position_xyz[0] += dx * math.cos(heading) - dy * math.sin(heading)
        self.position_xyz[2] += dx * math.sin(heading) + dy * math.cos(heading)
        self.heading_rad += math.radians(float(outcome.odom_delta.dyaw_deg))


def _blocked_summary(config: HabitatPixelNavMemorySmokeConfig, gated: dict[str, Any], *, reason: str) -> dict[str, Any]:
    summary = _base_summary(config, gated)
    summary["status"] = "skipped" if gated.get("status") == "skipped" else "blocked"
    summary["memory_update"] = {
        "updated": False,
        "reason": reason,
        "action_outcome": None,
    }
    summary["memory"] = None
    return summary


def _base_summary(config: HabitatPixelNavMemorySmokeConfig, gated: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": {
            "scene_path": str(config.scene_path),
            "start_position_xyz": list(config.start_position_xyz),
            "start_heading_rad": config.start_heading_rad,
            "goal_map_xy": list(config.goal_map_xy),
            "max_pixelnav_steps": config.max_pixelnav_steps,
            "force_front_view_waypoint": config.force_front_view_waypoint,
        },
        "gate": gated.get("gate"),
        "execution": gated.get("execution"),
        "vlm_output": gated.get("vlm_output"),
        "artifacts": {
            "output_dir": str(config.output_dir),
            "gated_execution_json": gated.get("summary_json"),
        },
    }


def _write_memory_update_summary(output_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    gated_path = summary.get("artifacts", {}).get("gated_execution_json")
    if gated_path and Path(gated_path).exists():
        target = output_dir / "real_vlm_gated_execution.json"
        if Path(gated_path).resolve() != target.resolve():
            shutil.copyfile(gated_path, target)
    summary_path = output_dir / "real_vlm_memory_update.json"
    summary["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def _action_outcome_from_json(payload: dict[str, Any]) -> Any:
    ActionOutcome = _import_robot_backend_type("ActionOutcome")
    RelativePose2D = _import_schema_type("RelativePose2D")
    odom = payload.get("odom_delta") or {}
    return ActionOutcome(
        action=str(payload.get("action", "go")),
        success=bool(payload.get("success")),
        collision=bool(payload.get("collision")),
        moved_distance_m=float(payload.get("moved_distance_m", 0.0) or 0.0),
        rotated_deg=float(payload.get("rotated_deg", 0.0) or 0.0),
        odom_delta=RelativePose2D(
            dx_m=float(odom.get("dx_m", 0.0) or 0.0),
            dy_m=float(odom.get("dy_m", 0.0) or 0.0),
            dyaw_deg=float(odom.get("dyaw_deg", 0.0) or 0.0),
        ),
        message=str(payload.get("message", "")),
        raw=payload.get("raw") if isinstance(payload.get("raw"), dict) else {},
    )


def _memory_replay_image_path(output_dir: Path, gated: dict[str, Any]) -> str:
    endpoint_path = ((gated.get("artifacts") or {}).get("endpoint_smoke_json"))
    if endpoint_path and Path(endpoint_path).exists():
        endpoint = json.loads(Path(endpoint_path).read_text(encoding="utf-8"))
        for view in ((endpoint.get("observation") or {}).get("views") or []):
            image = view.get("image")
            if image and Path(image).exists():
                return str(image)
    StaticImageBackend = _import_robot_backend_type("StaticImageBackend")
    return StaticImageBackend.create_demo_image(output_dir / "memory_replay_front.png", width=640, height=480)


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
                "status": edge.traversal.get("status"),
                "relative_pose_src_to_dst": edge.relative_pose_src_to_dst.to_dict(),
            }
        )
    return edges


def _deadlock_state_to_json(graph: Any) -> dict[str, Any]:
    current_node_id = graph.current_node_id
    node = graph.nodes.get(current_node_id) if current_node_id else None
    return {
        "current_node_id": current_node_id,
        "status": "none" if node is None else str(node.navigation_state.get("deadlock_status", "none")),
        "incoming_edge_id": graph.latest_incoming_edge_id(current_node_id) if current_node_id else None,
    }


def _import_agent_types() -> tuple[Any, Any]:
    qwen_root = Path(__file__).resolve().parents[1] / "qwen_nav_memory_framework_v5"
    if str(qwen_root) not in sys.path:
        sys.path.insert(0, str(qwen_root))
    from nav_memory_qwen import NavMemoryAgent, NavAgentConfig

    return NavMemoryAgent, NavAgentConfig


def _import_schema_type(name: str) -> Any:
    qwen_root = Path(__file__).resolve().parents[1] / "qwen_nav_memory_framework_v5"
    if str(qwen_root) not in sys.path:
        sys.path.insert(0, str(qwen_root))
    from nav_memory_qwen import schema

    return getattr(schema, name)


def _import_robot_backend_type(name: str) -> Any:
    qwen_root = Path(__file__).resolve().parents[1] / "qwen_nav_memory_framework_v5"
    if str(qwen_root) not in sys.path:
        sys.path.insert(0, str(qwen_root))
    from nav_memory_qwen import robot_backend

    return getattr(robot_backend, name)


if __name__ == "__main__":
    raise SystemExit(main())
