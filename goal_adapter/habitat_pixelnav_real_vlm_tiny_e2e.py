from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from goal_adapter.habitat_pixelnav_closed_loop import (
    HabitatPixelNavMemorySmokeConfig,
    load_memory_smoke_config_from_audit,
)
from goal_adapter.habitat_pixelnav_real_vlm_memory_update import (
    run_habitat_pixelnav_real_vlm_memory_update,
    run_habitat_pixelnav_real_vlm_memory_update_from_env,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a tiny real-VLM Habitat/PixelNav/memory end-to-end smoke.")
    parser.add_argument("--audit-result", required=True, help="Step B farthest goal audit result JSON")
    parser.add_argument("--out", required=True, help="Output directory for real_vlm_tiny_e2e.json")
    parser.add_argument("--force-front-view-waypoint", action="store_true")
    args = parser.parse_args(argv)

    config = load_memory_smoke_config_from_audit(
        args.audit_result,
        output_dir=args.out,
        max_agent_steps=1,
        force_front_view_waypoint=bool(args.force_front_view_waypoint),
    )
    summary = run_habitat_pixelnav_real_vlm_tiny_e2e_from_env(config)
    print(json.dumps({"status": summary.get("status"), "summary_json": summary.get("summary_json")}, ensure_ascii=False))
    return 0


def run_habitat_pixelnav_real_vlm_tiny_e2e_from_env(
    config: HabitatPixelNavMemorySmokeConfig,
    *,
    runner_factory: Any | None = None,
    executor_factory: Any | None = None,
) -> dict[str, Any]:
    memory_update = run_habitat_pixelnav_real_vlm_memory_update_from_env(
        config,
        runner_factory=runner_factory,
        executor_factory=executor_factory,
    )
    return build_tiny_e2e_summary(config, memory_update)


def run_habitat_pixelnav_real_vlm_tiny_e2e(
    config: HabitatPixelNavMemorySmokeConfig,
    *,
    runner_factory: Any | None = None,
    executor_factory: Any | None = None,
    vlm_client: Any,
) -> dict[str, Any]:
    memory_update = run_habitat_pixelnav_real_vlm_memory_update(
        config,
        runner_factory=runner_factory,
        executor_factory=executor_factory,
        vlm_client=vlm_client,
    )
    return build_tiny_e2e_summary(config, memory_update)


def build_tiny_e2e_summary(
    config: HabitatPixelNavMemorySmokeConfig,
    memory_update: dict[str, Any],
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint_smoke = _load_json_if_exists(output_dir / "real_vlm_endpoint_smoke.json")
    gated_execution = _load_json_if_exists(output_dir / "real_vlm_gated_execution.json")

    endpoint_smoke_passed = bool((endpoint_smoke.get("endpoint_smoke") or {}).get("passed"))
    gated_execution_executed = bool((memory_update.get("execution") or {}).get("executed"))
    memory_updated = bool((memory_update.get("memory_update") or {}).get("updated"))
    passed = endpoint_smoke_passed and gated_execution_executed and memory_updated

    status = "passed" if passed else ("skipped" if memory_update.get("status") == "skipped" else "failed")
    summary = {
        "status": status,
        "tiny_e2e": {
            "passed": passed,
            "endpoint_smoke_passed": endpoint_smoke_passed,
            "gated_execution_executed": gated_execution_executed,
            "memory_updated": memory_updated,
            "failure_stage": _failure_stage(
                endpoint_smoke_passed=endpoint_smoke_passed,
                gated_execution_executed=gated_execution_executed,
                memory_updated=memory_updated,
            ),
        },
        "config": {
            "scene_path": str(config.scene_path),
            "start_position_xyz": list(config.start_position_xyz),
            "start_heading_rad": config.start_heading_rad,
            "goal_map_xy": list(config.goal_map_xy),
            "max_pixelnav_steps": config.max_pixelnav_steps,
            "force_front_view_waypoint": config.force_front_view_waypoint,
        },
        "endpoint_smoke": endpoint_smoke.get("endpoint_smoke"),
        "gate": memory_update.get("gate"),
        "execution": memory_update.get("execution"),
        "memory_update": memory_update.get("memory_update"),
        "memory": memory_update.get("memory"),
        "artifacts": {
            "output_dir": str(output_dir),
            "endpoint_smoke_json": str(output_dir / "real_vlm_endpoint_smoke.json"),
            "gated_execution_json": str(output_dir / "real_vlm_gated_execution.json"),
            "memory_update_json": memory_update.get("summary_json"),
        },
    }
    return _write_tiny_e2e_summary(output_dir, summary)


def _failure_stage(
    *,
    endpoint_smoke_passed: bool,
    gated_execution_executed: bool,
    memory_updated: bool,
) -> str | None:
    if not endpoint_smoke_passed:
        return "endpoint_smoke"
    if not gated_execution_executed:
        return "gated_execution"
    if not memory_updated:
        return "memory_update"
    return None


def _write_tiny_e2e_summary(output_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    memory_update_path = summary.get("artifacts", {}).get("memory_update_json")
    if memory_update_path and Path(memory_update_path).exists():
        target = output_dir / "real_vlm_memory_update.json"
        if Path(memory_update_path).resolve() != target.resolve():
            shutil.copyfile(memory_update_path, target)
    summary_path = output_dir / "real_vlm_tiny_e2e.json"
    summary["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def _load_json_if_exists(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
