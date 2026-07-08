from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


REQUIRED_ENV = ("QWEN_BASE_URL", "QWEN_API_KEY")
OPTIONAL_ENV_DEFAULTS = {
    "PIXELNAV_DEVICE": "cpu",
    "QWEN_MODEL": "qwen3-vl-32b-thinking",
    "QWEN_TIMEOUT_S": "600",
    "QWEN_MAX_TOKENS": "8192",
    "QWEN_MAX_JSON_RETRIES": "1",
    "QWEN_EXTRA_PAYLOAD_JSON": '{"response_format":{"type":"json_object"}}',
    "QWEN_IMAGE_MAX_SIDE": "1024",
    "QWEN_JPEG_QUALITY": "85",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check readiness for real Qwen endpoint Step T execution.")
    parser.add_argument("--audit-result", required=True, help="Step B farthest goal audit result JSON")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--checkpoint", default="/home/icra/Pixel-Navigator/checkpoints/navigator.pth")
    args = parser.parse_args(argv)
    summary = build_real_vlm_endpoint_readiness(
        args.audit_result,
        output_dir=args.out,
        checkpoint_path=args.checkpoint,
    )
    print(json.dumps({"status": summary["status"], "summary_json": summary["summary_json"]}, ensure_ascii=False))
    return 0


def build_real_vlm_endpoint_readiness(
    audit_result: str | Path,
    *,
    output_dir: str | Path,
    checkpoint_path: str | Path = "/home/icra/Pixel-Navigator/checkpoints/navigator.pth",
) -> dict[str, Any]:
    audit_path = Path(audit_result)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    audit_payload: dict[str, Any] = {}
    if audit_path.exists():
        audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    scene_path = Path(str(audit_payload.get("scene_path", ""))) if audit_payload.get("scene_path") else None
    checkpoint = Path(checkpoint_path)

    missing_env = [name for name in REQUIRED_ENV if not os.getenv(name)]
    checks = {
        "required_env_present": not missing_env,
        "audit_result_exists": audit_path.exists(),
        "audit_has_scene_path": bool(audit_payload.get("scene_path")),
        "scene_exists": bool(scene_path and scene_path.exists()),
        "checkpoint_exists": checkpoint.exists(),
        "selected_point_present": _is_two_element_sequence(audit_payload.get("selected_image_point")),
        "coarse_goal_present": _is_two_element_sequence(audit_payload.get("coarse_goal_xy")),
    }
    blockers = [name for name, ok in checks.items() if not ok]
    ready = not blockers
    summary = {
        "status": "ready" if ready else "blocked",
        "ready": ready,
        "next_action": _next_action(ready=ready, blockers=blockers),
        "missing_env": missing_env,
        "checks": checks,
        "blockers": blockers,
        "env": {
            "required": list(REQUIRED_ENV),
            "optional_defaults": dict(OPTIONAL_ENV_DEFAULTS),
        },
        "paths": {
            "audit_result": str(audit_path),
            "scene_path": None if scene_path is None else str(scene_path),
            "checkpoint_path": str(checkpoint),
        },
        "commands": {
            "export_env_template": _export_env_template(),
            "step_t": _step_t_command(audit_path),
            "step_u": _step_u_command(),
        },
    }
    json_path = out / "real_vlm_endpoint_readiness.json"
    markdown_path = out / "docmost_real_vlm_endpoint_readiness.md"
    summary["summary_json"] = str(json_path)
    summary["docmost_markdown"] = str(markdown_path)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_docmost(summary), encoding="utf-8")
    return summary


def _render_docmost(summary: dict[str, Any]) -> str:
    checks = summary["checks"]
    lines = [
        "# Real VLM Endpoint Readiness",
        "",
        "## 1. 결론",
        "",
        f"- 상태: `{summary['status']}`",
        f"- 준비 완료: `{str(summary['ready']).lower()}`",
        f"- 다음 액션: `{summary['next_action']}`",
        "",
        "## 2. Readiness Checks",
        "",
        "| 항목 | 결과 |",
        "|---|---|",
    ]
    for name, ok in checks.items():
        lines.append(f"| {name} | `{str(ok).lower()}` |")
    lines.extend(
        [
            "",
            "## 3. Blockers",
            "",
        ]
    )
    if summary["blockers"]:
        for blocker in summary["blockers"]:
            lines.append(f"- `{blocker}`")
    else:
        lines.append("- 없음")
    lines.extend(
        [
            "",
            "## 4. 필요한 환경변수",
            "",
            "```bash",
            *summary["commands"]["export_env_template"],
            "```",
            "",
            "## 5. Step T 재실행",
            "",
            "```bash",
            *summary["commands"]["step_t"],
            "```",
            "",
            "## 6. Step U 최종 요약 재생성",
            "",
            "```bash",
            *summary["commands"]["step_u"],
            "```",
            "",
            "## 7. Paths",
            "",
            f"- audit: `{summary['paths']['audit_result']}`",
            f"- scene: `{summary['paths']['scene_path']}`",
            f"- checkpoint: `{summary['paths']['checkpoint_path']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _next_action(*, ready: bool, blockers: list[str]) -> str:
    if ready:
        return "run_step_t_real_vlm_tiny_e2e"
    if blockers == ["required_env_present"]:
        return "set_qwen_endpoint_env_and_rerun_step_t"
    return "resolve_readiness_blockers"


def _export_env_template() -> list[str]:
    lines = [f'export {name}="..."' for name in REQUIRED_ENV]
    for name, default in OPTIONAL_ENV_DEFAULTS.items():
        lines.append(f'export {name}="{default}"')
    return lines


def _step_t_command(audit_path: Path) -> list[str]:
    return [
        "PYTHONPATH=/home/icra/voca-s2e \\",
        "/home/icra/micromamba/bin/micromamba run -r /home/icra/micromamba-root -n habitat \\",
        "python -m goal_adapter.habitat_pixelnav_real_vlm_tiny_e2e \\",
        f"  --audit-result {audit_path} \\",
        "  --out /home/icra/voca-s2e/reports/habitat_pixelnav_integration/step_t_real_vlm_tiny_e2e \\",
        "  --force-front-view-waypoint",
    ]


def _step_u_command() -> list[str]:
    return [
        "PYTHONPATH=/home/icra/voca-s2e \\",
        "/home/icra/micromamba/bin/micromamba run -r /home/icra/micromamba-root -n habitat \\",
        "python -m goal_adapter.habitat_pixelnav_real_vlm_final_report \\",
        "  --tiny-e2e /home/icra/voca-s2e/reports/habitat_pixelnav_integration/step_t_real_vlm_tiny_e2e/real_vlm_tiny_e2e.json \\",
        "  --out /home/icra/voca-s2e/reports/habitat_pixelnav_integration/step_u_real_vlm_final_summary",
    ]


def _is_two_element_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2


if __name__ == "__main__":
    raise SystemExit(main())
