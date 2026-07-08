from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from goal_adapter.habitat_pixelnav_closed_loop import (
    HabitatPixelNavMemorySmokeConfig,
    load_memory_smoke_config_from_audit,
    run_habitat_pixelnav_memory_smoke,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run multi-step real-Qwen Habitat/PixelNav/v5-memory closed loop.")
    parser.add_argument("--audit-result", required=True, help="Step B audit result JSON")
    parser.add_argument("--out", required=True, help="Output directory for closed_loop_summary.json")
    parser.add_argument("--max-agent-steps", type=int, default=3)
    parser.add_argument("--max-pixelnav-steps", type=int, default=12)
    parser.add_argument("--force-front-view-waypoint", action="store_true")
    args = parser.parse_args(argv)

    config = load_memory_smoke_config_from_audit(
        args.audit_result,
        output_dir=args.out,
        max_agent_steps=max(1, int(args.max_agent_steps)),
        force_front_view_waypoint=bool(args.force_front_view_waypoint),
    )
    config = replace(config, max_pixelnav_steps=max(1, int(args.max_pixelnav_steps)))
    summary = run_habitat_pixelnav_real_vlm_closed_loop_from_env(config)
    print(json.dumps({"status": summary.get("status"), "summary_json": summary.get("summary_json")}, ensure_ascii=False))
    return 0


def run_habitat_pixelnav_real_vlm_closed_loop_from_env(
    config: HabitatPixelNavMemorySmokeConfig,
    *,
    runner_factory: Any | None = None,
    executor_factory: Any | None = None,
) -> dict[str, Any]:
    OpenAICompatibleVLMClient = _import_vlm_client_type("OpenAICompatibleVLMClient")
    primary = OpenAICompatibleVLMClient.from_env()
    if os.getenv("QWEN_DISABLE_HEURISTIC_FALLBACK", "").strip() in {"1", "true", "True"}:
        client = primary
    else:
        client = FallbackOnVLMErrorClient(primary)
    return run_habitat_pixelnav_real_vlm_closed_loop(
        config,
        vlm_client=client,
        runner_factory=runner_factory,
        executor_factory=executor_factory,
    )


def run_habitat_pixelnav_real_vlm_closed_loop(
    config: HabitatPixelNavMemorySmokeConfig,
    *,
    vlm_client: Any,
    runner_factory: Any | None = None,
    executor_factory: Any | None = None,
) -> dict[str, Any]:
    summary = run_habitat_pixelnav_memory_smoke(
        config,
        runner_factory=runner_factory,
        executor_factory=executor_factory,
        vlm_client=vlm_client,
    )
    summary["status"] = "completed" if not summary.get("steps", [{}])[-1].get("error") else "failed"
    summary["runtime"] = {
        "vlm_client": _client_runtime_name(vlm_client),
        "max_agent_steps": int(config.max_agent_steps),
        "max_pixelnav_steps": int(config.max_pixelnav_steps),
        "force_front_view_waypoint": bool(config.force_front_view_waypoint),
        "fallback_count": int(getattr(vlm_client, "fallback_count", 0) or 0),
        "fallback_errors": list(getattr(vlm_client, "fallback_errors", []) or []),
    }
    summary_path = Path(summary["summary_json"])
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


class FallbackOnVLMErrorClient:
    def __init__(self, primary: Any, fallback: Any | None = None):
        self.primary = primary
        self.fallback = fallback or _make_heuristic_client()
        self.fallback_count = 0
        self.fallback_errors: list[str] = []

    def decide(self, vlm_input: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.primary.decide(vlm_input)
        except Exception as exc:
            self.fallback_count += 1
            self.fallback_errors.append(f"{type(exc).__name__}: {exc}")
            output = self.fallback.decide(vlm_input)
            reasoning = dict(output.get("reasoning") or {})
            reasoning["short_text"] = "heuristic fallback after VLM parse/call failure"
            output["reasoning"] = reasoning
            output["memory_ops"] = list(output.get("memory_ops") or [])
            output["runtime_fallback"] = {
                "used": True,
                "reason": self.fallback_errors[-1],
            }
            return output


def _make_heuristic_client() -> Any:
    HeuristicVLMClient = _import_vlm_client_type("HeuristicVLMClient")
    return HeuristicVLMClient()


def _client_runtime_name(client: Any) -> str:
    if isinstance(client, FallbackOnVLMErrorClient):
        return "openai_compatible_with_heuristic_fallback"
    return "injected"


def _import_vlm_client_type(name: str) -> Any:
    try:
        from nav_memory_qwen import vlm_client
    except ModuleNotFoundError:
        qwen_root = Path(__file__).resolve().parents[1] / "qwen_nav_memory_framework_v5"
        if str(qwen_root) not in sys.path:
            sys.path.insert(0, str(qwen_root))
        from nav_memory_qwen import vlm_client
    return getattr(vlm_client, name)


if __name__ == "__main__":
    raise SystemExit(main())
