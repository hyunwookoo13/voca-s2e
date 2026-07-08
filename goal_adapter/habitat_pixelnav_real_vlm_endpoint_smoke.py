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
from goal_adapter.habitat_pixelnav_real_vlm_dry_run import (
    REQUIRED_QWEN_ENV,
    run_habitat_pixelnav_real_vlm_dry_run,
    run_habitat_pixelnav_real_vlm_dry_run_from_env,
)


ALLOWED_ACTIONS = {"go", "rotate", "stop", "request_observation"}
ALLOWED_VIEWS = {"front", "left", "right", "back"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one real VLM endpoint smoke against Habitat/PixelNav context.")
    parser.add_argument("--audit-result", required=True, help="Step B farthest goal audit result JSON")
    parser.add_argument("--out", required=True, help="Output directory for real_vlm_endpoint_smoke.json")
    parser.add_argument("--force-front-view-waypoint", action="store_true")
    args = parser.parse_args(argv)

    config = load_memory_smoke_config_from_audit(
        args.audit_result,
        output_dir=args.out,
        max_agent_steps=1,
        force_front_view_waypoint=bool(args.force_front_view_waypoint),
    )
    summary = run_habitat_pixelnav_real_vlm_endpoint_smoke_from_env(config)
    print(json.dumps({"status": summary.get("status"), "summary_json": summary.get("summary_json")}, ensure_ascii=False))
    return 0


def run_habitat_pixelnav_real_vlm_endpoint_smoke_from_env(
    config: HabitatPixelNavMemorySmokeConfig,
    *,
    runner_factory: Any | None = None,
    backend: Any | None = None,
) -> dict[str, Any]:
    dry_run = run_habitat_pixelnav_real_vlm_dry_run_from_env(
        config,
        runner_factory=runner_factory,
        backend=backend,
    )
    return _build_endpoint_smoke_summary(config.output_dir, dry_run)


def run_habitat_pixelnav_real_vlm_endpoint_smoke(
    config: HabitatPixelNavMemorySmokeConfig,
    *,
    runner_factory: Any | None = None,
    backend: Any | None = None,
    vlm_client: Any,
) -> dict[str, Any]:
    dry_run = run_habitat_pixelnav_real_vlm_dry_run(
        config,
        runner_factory=runner_factory,
        backend=backend,
        vlm_client=vlm_client,
    )
    return _build_endpoint_smoke_summary(config.output_dir, dry_run)


def validate_vlm_endpoint_smoke(dry_run: dict[str, Any]) -> dict[str, Any]:
    if dry_run.get("skipped"):
        return {
            "passed": False,
            "checks": {},
            "failures": [],
            "skipped": True,
        }

    output = dry_run.get("vlm_output") or {}
    observation = dry_run.get("observation") or {}
    width = int(observation.get("image_width", 0) or 0)
    height = int(observation.get("image_height", 0) or 0)
    action = str(output.get("action") or "")
    point = output.get("selected_image_point")
    view_type = output.get("selected_view_type")
    memory_ops = output.get("memory_ops", [])

    checks = {
        "schema_version_valid": output.get("schema_version") == "nav_vlm_waypoint_v1",
        "action_valid": action in ALLOWED_ACTIONS,
        "selected_view_type_valid": action != "go" or view_type in ALLOWED_VIEWS,
        "selected_image_point_valid": action != "go" or _point_is_valid(point, width, height),
        "memory_ops_valid": memory_ops is None or isinstance(memory_ops, list),
        "backend_action_not_executed": not bool((dry_run.get("dry_run") or {}).get("executed_backend_action")),
    }
    failures = [name for name, ok in checks.items() if not ok]
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "skipped": False,
    }


def _build_endpoint_smoke_summary(output_dir: str | Path, dry_run: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dry_run.get("skipped"):
        summary = dict(dry_run)
        summary["endpoint_smoke"] = {
            "passed": False,
            "skipped": True,
            "checks": {},
            "failures": [],
            "required_env": list(REQUIRED_QWEN_ENV),
        }
        summary["status"] = "skipped"
    else:
        endpoint_smoke = validate_vlm_endpoint_smoke(dry_run)
        summary = dict(dry_run)
        summary["endpoint_smoke"] = endpoint_smoke
        summary["status"] = "passed" if endpoint_smoke["passed"] else "failed"

    dry_run_path = dry_run.get("summary_json")
    if dry_run_path and Path(dry_run_path).exists():
        target = output_dir / "real_vlm_dry_run.json"
        if Path(dry_run_path).resolve() != target.resolve():
            shutil.copyfile(dry_run_path, target)

    summary_path = output_dir / "real_vlm_endpoint_smoke.json"
    summary["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def _point_is_valid(point: Any, width: int, height: int) -> bool:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return False
    try:
        u = int(point[0])
        v = int(point[1])
    except Exception:
        return False
    return width > 0 and height > 0 and 0 <= u < width and 0 <= v < height


if __name__ == "__main__":
    raise SystemExit(main())
