from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Docmost-ready final summary for real VLM Habitat/PixelNav smoke.")
    parser.add_argument("--tiny-e2e", required=True, help="Path to real_vlm_tiny_e2e.json")
    parser.add_argument("--out", required=True, help="Output directory for final summary artifacts")
    args = parser.parse_args(argv)
    summary = build_real_vlm_final_summary(args.tiny_e2e, output_dir=args.out)
    print(json.dumps({"status": summary["status"], "docmost_markdown": summary["docmost_markdown"]}, ensure_ascii=False))
    return 0


def build_real_vlm_final_summary(tiny_e2e_json: str | Path, *, output_dir: str | Path) -> dict[str, Any]:
    tiny_path = Path(tiny_e2e_json)
    tiny = json.loads(tiny_path.read_text(encoding="utf-8"))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tiny_e2e = tiny.get("tiny_e2e") or {}
    endpoint_smoke = tiny.get("endpoint_smoke") or {}
    execution = tiny.get("execution") or {}
    memory_update = tiny.get("memory_update") or {}
    memory = tiny.get("memory") or {}

    status = str(tiny.get("status") or "unknown")
    ready = bool(tiny_e2e.get("passed"))
    failure_stage = tiny_e2e.get("failure_stage")
    next_action = _next_action(status=status, failure_stage=failure_stage)
    outcome = execution.get("action_outcome") if isinstance(execution.get("action_outcome"), dict) else {}
    summary = {
        "status": status,
        "ready_for_docmost": ready,
        "next_action": next_action,
        "tiny_e2e": {
            "passed": bool(tiny_e2e.get("passed")),
            "endpoint_smoke_passed": bool(tiny_e2e.get("endpoint_smoke_passed")),
            "gated_execution_executed": bool(tiny_e2e.get("gated_execution_executed")),
            "memory_updated": bool(tiny_e2e.get("memory_updated")),
            "failure_stage": failure_stage,
        },
        "execution": {
            "success": outcome.get("success"),
            "moved_distance_m": outcome.get("moved_distance_m"),
            "collision": outcome.get("collision"),
        },
        "memory": {
            "schema_version": memory.get("schema_version"),
            "num_nodes": memory.get("num_nodes"),
            "num_edges": memory.get("num_edges"),
        },
        "required_env": list(endpoint_smoke.get("required_env") or []),
        "source": {
            "tiny_e2e_json": str(tiny_path),
            "artifact_root": str((tiny.get("artifacts") or {}).get("output_dir") or tiny_path.parent),
        },
    }

    json_path = out / "real_vlm_final_summary.json"
    markdown_path = out / "docmost_real_vlm_final_summary.md"
    summary["summary_json"] = str(json_path)
    summary["docmost_markdown"] = str(markdown_path)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_docmost_markdown(summary), encoding="utf-8")
    return summary


def _next_action(*, status: str, failure_stage: Any) -> str:
    if status == "passed":
        return "share_docmost_summary_and_run_larger_batch"
    if failure_stage == "endpoint_smoke" or status == "skipped":
        return "set_qwen_endpoint_env_and_rerun_step_t"
    if failure_stage == "gated_execution":
        return "inspect_vlm_output_contract_and_pixelnav_gate"
    if failure_stage == "memory_update":
        return "inspect_action_outcome_memory_update"
    return "inspect_tiny_e2e_artifacts"


def _render_docmost_markdown(summary: dict[str, Any]) -> str:
    tiny = summary["tiny_e2e"]
    execution = summary["execution"]
    memory = summary["memory"]
    required_env = summary.get("required_env") or ["QWEN_BASE_URL", "QWEN_API_KEY"]
    lines = [
        "# Real VLM Tiny E2E 최종 요약",
        "",
        "## 1. 결론",
        "",
        f"- 상태: `{summary['status']}`",
        f"- Docmost 공유 준비: `{str(summary['ready_for_docmost']).lower()}`",
        f"- 다음 액션: `{summary['next_action']}`",
        "",
        "## 2. 단계별 확인",
        "",
        "| 단계 | 결과 |",
        "|---|---|",
        f"| Endpoint smoke | `{str(tiny.get('endpoint_smoke_passed')).lower()}` |",
        f"| PixelNav gated execution | `{str(tiny.get('gated_execution_executed')).lower()}` |",
        f"| Memory update | `{str(tiny.get('memory_updated')).lower()}` |",
        f"| failure_stage | `{tiny.get('failure_stage')}` |",
        "",
        "## 3. 실행 결과",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| ActionOutcome success | `{_display_value(execution.get('success'))}` |",
        f"| moved distance | `{_display_value(execution.get('moved_distance_m'))}` |",
        f"| collision | `{_display_value(execution.get('collision'))}` |",
        "",
        "## 4. Memory 결과",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| schema | `{_display_value(memory.get('schema_version'))}` |",
        f"| nodes | `{_display_value(memory.get('num_nodes'))}` |",
        f"| edges | `{_display_value(memory.get('num_edges'))}` |",
        "",
        "## 5. 현재 필요한 설정",
        "",
        "실제 endpoint smoke를 통과하려면 아래 환경변수가 필요하다.",
        "",
        "```bash",
    ]
    for name in required_env:
        lines.append(f'export {name}="..."')
    lines.extend(
        [
            'export PIXELNAV_DEVICE="cpu"',
            'export QWEN_MODEL="qwen3-vl-32b-thinking"',
            'export QWEN_TIMEOUT_S="600"',
            'export QWEN_MAX_TOKENS="8192"',
            'export QWEN_MAX_JSON_RETRIES="1"',
            'export QWEN_EXTRA_PAYLOAD_JSON=\'{"response_format":{"type":"json_object"}}\'',
            "```",
            "",
            "## 6. 재실행 명령",
            "",
            "```bash",
            "PIXELNAV_DEVICE=cpu \\",
            "QWEN_TIMEOUT_S=600 \\",
            "QWEN_MAX_TOKENS=8192 \\",
            "QWEN_EXTRA_PAYLOAD_JSON='{\"response_format\":{\"type\":\"json_object\"}}' \\",
            "PYTHONPATH=/home/icra/voca-s2e \\",
            "/home/icra/micromamba/bin/micromamba run -r /home/icra/micromamba-root -n habitat \\",
            "python -m goal_adapter.habitat_pixelnav_real_vlm_tiny_e2e \\",
            "  --audit-result /home/icra/voca-s2e/reports/habitat_pixelnav_integration/step_b_farthest_policy/farthest_goal_audit_result.json \\",
            "  --out /home/icra/voca-s2e/reports/habitat_pixelnav_integration/step_t_real_vlm_tiny_e2e \\",
            "  --force-front-view-waypoint",
            "```",
            "",
            "## 7. Source Artifacts",
            "",
            f"- tiny E2E JSON: `{summary['source']['tiny_e2e_json']}`",
            f"- artifact root: `{summary['source']['artifact_root']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _display_value(value: Any) -> Any:
    return "not_run" if value is None else value


if __name__ == "__main__":
    raise SystemExit(main())
