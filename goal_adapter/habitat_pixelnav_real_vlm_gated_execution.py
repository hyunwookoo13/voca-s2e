from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from goal_adapter.habitat_pixelnav_backend import HabitatPixelNavBackend, HabitatPixelNavBackendConfig
from goal_adapter.habitat_pixelnav_closed_loop import (
    HabitatPixelNavMemorySmokeConfig,
    load_memory_smoke_config_from_audit,
)
from goal_adapter.habitat_pixelnav_execute import action_outcome_to_json
from goal_adapter.habitat_pixelnav_real_vlm_endpoint_smoke import (
    run_habitat_pixelnav_real_vlm_endpoint_smoke,
    run_habitat_pixelnav_real_vlm_endpoint_smoke_from_env,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PixelNav once only after real VLM endpoint smoke passes.")
    parser.add_argument("--audit-result", required=True, help="Step B farthest goal audit result JSON")
    parser.add_argument("--out", required=True, help="Output directory for real_vlm_gated_execution.json")
    parser.add_argument("--force-front-view-waypoint", action="store_true")
    args = parser.parse_args(argv)

    config = load_memory_smoke_config_from_audit(
        args.audit_result,
        output_dir=args.out,
        max_agent_steps=1,
        force_front_view_waypoint=bool(args.force_front_view_waypoint),
    )
    summary = run_habitat_pixelnav_real_vlm_gated_execution_from_env(config)
    print(json.dumps({"status": summary.get("status"), "summary_json": summary.get("summary_json")}, ensure_ascii=False))
    return 0


def run_habitat_pixelnav_real_vlm_gated_execution_from_env(
    config: HabitatPixelNavMemorySmokeConfig,
    *,
    runner_factory: Any | None = None,
    executor_factory: Any | None = None,
    backend: Any | None = None,
) -> dict[str, Any]:
    endpoint_smoke = run_habitat_pixelnav_real_vlm_endpoint_smoke_from_env(
        config,
        runner_factory=runner_factory,
    )
    return execute_pixelnav_after_endpoint_smoke_gate(
        config,
        endpoint_smoke,
        runner_factory=runner_factory,
        executor_factory=executor_factory,
        backend=backend,
    )


def run_habitat_pixelnav_real_vlm_gated_execution(
    config: HabitatPixelNavMemorySmokeConfig,
    *,
    runner_factory: Any | None = None,
    executor_factory: Any | None = None,
    backend: Any | None = None,
    vlm_client: Any,
) -> dict[str, Any]:
    endpoint_smoke = run_habitat_pixelnav_real_vlm_endpoint_smoke(
        config,
        runner_factory=runner_factory,
        vlm_client=vlm_client,
    )
    return execute_pixelnav_after_endpoint_smoke_gate(
        config,
        endpoint_smoke,
        runner_factory=runner_factory,
        executor_factory=executor_factory,
        backend=backend,
    )


def execute_pixelnav_after_endpoint_smoke_gate(
    config: HabitatPixelNavMemorySmokeConfig,
    endpoint_smoke: dict[str, Any],
    *,
    runner_factory: Any | None = None,
    executor_factory: Any | None = None,
    backend: Any | None = None,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gate_passed = bool((endpoint_smoke.get("endpoint_smoke") or {}).get("passed"))
    vlm_output = endpoint_smoke.get("vlm_output") or {}
    if not gate_passed:
        return _write_gated_execution_summary(
            output_dir,
            _blocked_summary(
                config,
                endpoint_smoke,
                status="skipped" if endpoint_smoke.get("skipped") else "blocked",
                reason="endpoint_smoke_not_passed",
            ),
        )
    if vlm_output.get("action") != "go":
        return _write_gated_execution_summary(
            output_dir,
            _blocked_summary(
                config,
                endpoint_smoke,
                status="blocked",
                reason=f"non_go_action:{vlm_output.get('action')}",
            ),
        )

    robot = backend or HabitatPixelNavBackend(
        HabitatPixelNavBackendConfig(
            scene_path=config.scene_path,
            output_dir=output_dir / "gated_backend",
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
    try:
        point = vlm_output["selected_image_point"]
        outcome = robot.execute_waypoint(
            view_type=str(vlm_output.get("selected_view_type", "front")),
            view_id=int(vlm_output.get("selected_view_id", 0)),
            point_px=(int(point[0]), int(point[1])),
            ttl_ms=int((vlm_output.get("control") or {}).get("ttl_ms", 1000) or 1000),
        )
        summary = _base_summary(config, endpoint_smoke)
        summary["status"] = "executed"
        summary["gate"] = {
            "passed": True,
            "reason": "endpoint_smoke_passed",
        }
        summary["execution"] = {
            "executed": True,
            "reason": "go_output_executed",
            "request": {
                "selected_view_type": str(vlm_output.get("selected_view_type", "front")),
                "selected_view_id": int(vlm_output.get("selected_view_id", 0)),
                "selected_image_point": [int(point[0]), int(point[1])],
                "ttl_ms": int((vlm_output.get("control") or {}).get("ttl_ms", 1000) or 1000),
            },
            "action_outcome": action_outcome_to_json(outcome),
        }
        summary["artifacts"]["gated_backend_output_dir"] = str(output_dir / "gated_backend")
        return _write_gated_execution_summary(output_dir, summary)
    finally:
        close = getattr(robot, "close", None)
        if callable(close):
            close()


def _blocked_summary(
    config: HabitatPixelNavMemorySmokeConfig,
    endpoint_smoke: dict[str, Any],
    *,
    status: str,
    reason: str,
) -> dict[str, Any]:
    summary = _base_summary(config, endpoint_smoke)
    summary["status"] = status
    summary["gate"] = {
        "passed": False,
        "reason": reason,
        "endpoint_smoke_status": endpoint_smoke.get("status"),
    }
    summary["execution"] = {
        "executed": False,
        "reason": reason,
        "action_outcome": None,
    }
    return summary


def _base_summary(config: HabitatPixelNavMemorySmokeConfig, endpoint_smoke: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": {
            "scene_path": str(config.scene_path),
            "start_position_xyz": list(config.start_position_xyz),
            "start_heading_rad": config.start_heading_rad,
            "goal_map_xy": list(config.goal_map_xy),
            "max_pixelnav_steps": config.max_pixelnav_steps,
            "force_front_view_waypoint": config.force_front_view_waypoint,
        },
        "endpoint_smoke": endpoint_smoke.get("endpoint_smoke"),
        "vlm_output": endpoint_smoke.get("vlm_output"),
        "artifacts": {
            "output_dir": str(config.output_dir),
            "endpoint_smoke_json": endpoint_smoke.get("summary_json"),
        },
    }


def _write_gated_execution_summary(output_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    endpoint_path = summary.get("artifacts", {}).get("endpoint_smoke_json")
    if endpoint_path and Path(endpoint_path).exists():
        target = output_dir / "real_vlm_endpoint_smoke.json"
        if Path(endpoint_path).resolve() != target.resolve():
            shutil.copyfile(endpoint_path, target)
    summary_path = output_dir / "real_vlm_gated_execution.json"
    summary["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
