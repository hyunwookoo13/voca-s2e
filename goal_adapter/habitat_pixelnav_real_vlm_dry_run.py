from __future__ import annotations

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Any

from goal_adapter.habitat_pixelnav_backend import HabitatPixelNavBackend, HabitatPixelNavBackendConfig
from goal_adapter.habitat_pixelnav_closed_loop import (
    HabitatPixelNavMemorySmokeConfig,
    _import_agent_type,
    _memory_context_checks_to_json,
    load_memory_smoke_config_from_audit,
)


REQUIRED_QWEN_ENV = ("QWEN_BASE_URL", "QWEN_API_KEY")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Habitat/PixelNav real-VLM dry-run decision.")
    parser.add_argument("--audit-result", required=True, help="Step B farthest goal audit result JSON")
    parser.add_argument("--out", required=True, help="Output directory for real_vlm_dry_run.json")
    parser.add_argument("--force-front-view-waypoint", action="store_true")
    args = parser.parse_args(argv)

    config = load_memory_smoke_config_from_audit(
        args.audit_result,
        output_dir=args.out,
        max_agent_steps=1,
        force_front_view_waypoint=bool(args.force_front_view_waypoint),
    )
    summary = run_habitat_pixelnav_real_vlm_dry_run_from_env(config)
    print(json.dumps({"status": summary.get("status"), "summary_json": summary.get("summary_json")}, ensure_ascii=False))
    return 0


def run_habitat_pixelnav_real_vlm_dry_run_from_env(
    config: HabitatPixelNavMemorySmokeConfig,
    *,
    runner_factory: Any | None = None,
    backend: Any | None = None,
) -> dict[str, Any]:
    """Run one real-VLM decision if endpoint env is available, otherwise skip."""
    missing = [name for name in REQUIRED_QWEN_ENV if not os.getenv(name)]
    if missing:
        return _write_skipped_summary(config, missing)
    client = _openai_compatible_client_from_env()
    return run_habitat_pixelnav_real_vlm_dry_run(
        config,
        runner_factory=runner_factory,
        backend=backend,
        vlm_client=client,
    )


def run_habitat_pixelnav_real_vlm_dry_run(
    config: HabitatPixelNavMemorySmokeConfig,
    *,
    runner_factory: Any | None = None,
    backend: Any | None = None,
    vlm_client: Any,
) -> dict[str, Any]:
    """Call the VLM once with real Habitat/PixelNav memory context, without acting.

    This is a contract check between the v5 memory prompt/input and a real
    OpenAI-compatible Qwen endpoint. It intentionally stops after
    ``vlm_client.decide`` and schema sanitization, so PixelNav does not execute.
    """
    NavMemoryAgent = _import_agent_type("NavMemoryAgent")
    NavAgentConfig = _import_agent_type("NavAgentConfig")
    build_vlm_input_v1 = _import_schema_type("build_vlm_input_v1")
    sanitize_vlm_output = _import_safety_type("sanitize_vlm_output")
    strip_large_image_values = _import_utils_type("strip_large_image_values")

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
    )
    try:
        agent = NavMemoryAgent(
            robot=robot,
            vlm_client=vlm_client,
            config=NavAgentConfig(
                max_steps=1,
                force_front_view_waypoint=config.force_front_view_waypoint,
                pose_noise_enabled=False,
                log_full_vlm_input=False,
            ),
        )
        if config.initial_yaw_offsets_deg:
            agent.pending_observation = robot.capture_views(
                config.initial_yaw_offsets_deg,
                mode="directed_sweep",
            )

        state = robot.get_robot_state()
        goal = agent._build_goal(goal_map_xy=(float(config.goal_map_xy[0]), float(config.goal_map_xy[1])), target_object=None)
        observation = agent._get_observation()
        embedding = agent._embed_front_or_first(observation)
        current_node_id = agent._update_memory_before_decision(observation, embedding, state)
        memory_context = agent.memory.build_vlm_memory_context(
            goal_bearing_deg=goal.relative_bearing_deg,
            goal_distance_m=goal.distance_m,
            task_mode=goal.task_mode,
            target_object=goal.target_object,
            max_memory_images=agent.config.max_memory_images,
            force_front_view_waypoint=agent.config.force_front_view_waypoint,
        )
        memory_context["loop_warning"] = {"is_looping": False, "repeated_branch_count": 0}
        memory_context["runtime_state"] = {
            "no_progress_count": 0,
            "observation_request_count": 0,
            "force_front_view_waypoint": agent.config.force_front_view_waypoint,
            "last_action_outcome": None,
        }

        vlm_input = build_vlm_input_v1(
            task=goal,
            robot_state=state,
            observation=observation,
            memory=memory_context,
            pose_noise_enabled=False,
        )
        raw_output = vlm_client.decide(vlm_input)
        vlm_output, warnings = sanitize_vlm_output(raw_output, vlm_input)
        summary = {
            "status": "completed",
            "skipped": False,
            "dry_run": {
                "called_vlm": True,
                "executed_backend_action": False,
                "applied_memory_ops": False,
            },
            "config": {
                "scene_path": str(config.scene_path),
                "start_position_xyz": list(config.start_position_xyz),
                "start_heading_rad": config.start_heading_rad,
                "goal_map_xy": list(config.goal_map_xy),
                "force_front_view_waypoint": config.force_front_view_waypoint,
                "initial_yaw_offsets_deg": list(config.initial_yaw_offsets_deg),
            },
            "observation": {
                "mode": observation.mode,
                "frame_index": observation.frame_index,
                "image_width": observation.image_width,
                "image_height": observation.image_height,
                "views": [
                    {
                        "view_id": view.view_id,
                        "view_type": view.view_type,
                        "relative_heading_deg": view.relative_heading_deg,
                        "image": view.image,
                    }
                    for view in observation.views
                ],
            },
            "current_node_id": current_node_id,
            "goal": {
                "task_mode": goal.task_mode,
                "relative_bearing_deg": goal.relative_bearing_deg,
                "distance_m": goal.distance_m,
                "target_object": goal.target_object,
            },
            "memory": {
                "schema_version": agent.memory.to_dict().get("schema_version"),
                "num_nodes": len(agent.memory.nodes),
                "num_edges": len(agent.memory.edges),
                "current_node_id": agent.memory.current_node_id,
                "live_pose_relation": agent.memory.current_pose_relation_to_latest_node(),
            },
            "memory_context_checks": _memory_context_checks_to_json([{"memory_context": _vlm_memory_context_to_json(vlm_input)}]),
            "vlm_input": strip_large_image_values(vlm_input),
            "raw_vlm_output": raw_output,
            "vlm_output": vlm_output,
            "warnings": warnings,
            "artifacts": {
                "output_dir": str(output_dir),
                "backend_output_dir": str(output_dir / "backend"),
            },
        }
        return _write_completed_summary(output_dir, summary)
    finally:
        close = getattr(robot, "close", None)
        if callable(close):
            close()


def _write_skipped_summary(config: HabitatPixelNavMemorySmokeConfig, missing_env: list[str]) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "skipped",
        "skipped": True,
        "reason": "missing_qwen_endpoint_env",
        "missing_env": list(missing_env),
        "required_env": list(REQUIRED_QWEN_ENV),
        "artifacts": {
            "output_dir": str(output_dir),
        },
    }
    return _write_completed_summary(output_dir, summary)


def _write_completed_summary(output_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    summary_path = output_dir / "real_vlm_dry_run.json"
    summary["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def _vlm_memory_context_to_json(vlm_input: dict[str, Any] | None) -> dict[str, Any]:
    memory = (vlm_input or {}).get("memory") or {}
    return {
        "schema_version": memory.get("schema_version"),
        "merge_policy": (memory.get("graph_summary") or {}).get("merge_policy"),
        "deadlock_status": (memory.get("deadlock_state") or {}).get("status"),
        "failed_waypoints_count": len(list(memory.get("failed_waypoints") or [])),
        "compressed_negative_memory_count": len(list(memory.get("compressed_negative_memories") or [])),
        "candidate_exit_count": len(list((memory.get("local_topology") or {}).get("candidate_exits") or [])),
        "avoid_candidate_exit_count": len(
            [
                item
                for item in list((memory.get("local_topology") or {}).get("candidate_exits") or [])
                if item.get("avoid")
            ]
        ),
    }


def _openai_compatible_client_from_env() -> Any:
    qwen_root = Path(__file__).resolve().parents[1] / "qwen_nav_memory_framework_v5"
    if str(qwen_root) not in sys.path:
        sys.path.insert(0, str(qwen_root))
    from nav_memory_qwen import OpenAICompatibleVLMClient

    return OpenAICompatibleVLMClient.from_env()


def _import_schema_type(name: str) -> Any:
    qwen_root = Path(__file__).resolve().parents[1] / "qwen_nav_memory_framework_v5"
    if str(qwen_root) not in sys.path:
        sys.path.insert(0, str(qwen_root))
    from nav_memory_qwen import schema

    return getattr(schema, name)


def _import_safety_type(name: str) -> Any:
    qwen_root = Path(__file__).resolve().parents[1] / "qwen_nav_memory_framework_v5"
    if str(qwen_root) not in sys.path:
        sys.path.insert(0, str(qwen_root))
    from nav_memory_qwen import safety

    return getattr(safety, name)


def _import_utils_type(name: str) -> Any:
    qwen_root = Path(__file__).resolve().parents[1] / "qwen_nav_memory_framework_v5"
    if str(qwen_root) not in sys.path:
        sys.path.insert(0, str(qwen_root))
    from nav_memory_qwen import utils

    return getattr(utils, name)


if __name__ == "__main__":
    raise SystemExit(main())
