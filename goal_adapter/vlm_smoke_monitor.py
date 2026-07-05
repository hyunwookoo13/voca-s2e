from __future__ import annotations

import argparse
import html
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from goal_adapter.ollama_provider import OllamaVLMConfig, OllamaVLMProvider
from goal_adapter.decision_parser import parse_vlm_decision
from goal_adapter.schema import GoalAdapterInput, GoalAdapterOutput


class DecisionProvider(Protocol):
    def decide(self, adapter_input: GoalAdapterInput) -> GoalAdapterOutput:
        ...


@dataclass(frozen=True)
class VLMSmokeReport:
    output_dir: Path
    image_path: Path
    input_path: Path
    decision_path: Path
    raw_response_path: Path
    summary_path: Path
    html_path: Path
    status: str


def run_vlm_smoke_monitor(
    input_path: str | Path,
    output_dir: str | Path = Path("reports") / "vlm_smoke" / "latest",
    decision_provider: DecisionProvider | None = None,
    model_name: str = "gemma4:26b",
    timeout_seconds: float = 420.0,
    raise_on_error: bool = False,
) -> VLMSmokeReport:
    started_at = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_input = _read_json_object(input_path)
    adapter_input = GoalAdapterInput.from_json(raw_input)
    image_path = _copy_current_rgb(adapter_input, output_dir)
    monitored_input = adapter_input.to_json()
    monitored_input["current_rgb"] = str(image_path.resolve())
    adapter_input = GoalAdapterInput.from_json(monitored_input)

    input_output_path = output_dir / "goal_adapter_input.json"
    decision_path = output_dir / "vlm_decision.json"
    raw_response_path = output_dir / "vlm_raw_response.txt"
    summary_path = output_dir / "summary.json"
    html_path = output_dir / "index.html"
    _write_json(input_output_path, adapter_input.to_json())

    provider = decision_provider or OllamaVLMProvider(
        OllamaVLMConfig(model=model_name, timeout_seconds=timeout_seconds)
    )

    try:
        raw_response = _generate_raw_or_decide(provider, adapter_input)
        output = parse_vlm_decision(raw_response)
        decision_payload: dict[str, Any] = output.to_json()
        status = "success"
        summary = _success_summary(
            model_name=model_name,
            adapter_input=adapter_input,
            output=output,
            elapsed_seconds=time.time() - started_at,
        )
    except Exception as exc:
        if raise_on_error:
            raise
        status = "error"
        decision_payload = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        summary = _error_summary(
            model_name=model_name,
            adapter_input=adapter_input,
            exc=exc,
            elapsed_seconds=time.time() - started_at,
        )
        raw_response = locals().get("raw_response", "")

    raw_response_path.write_text(str(raw_response), encoding="utf-8")
    _write_json(decision_path, decision_payload)
    _write_json(summary_path, summary)
    html_path.write_text(
        _render_html(
            image_path=image_path,
            adapter_input=adapter_input.to_json(),
            decision=decision_payload,
            raw_response=str(raw_response),
            summary=summary,
        ),
        encoding="utf-8",
    )

    return VLMSmokeReport(
        output_dir=output_dir,
        image_path=image_path,
        input_path=input_output_path,
        decision_path=decision_path,
        raw_response_path=raw_response_path,
        summary_path=summary_path,
        html_path=html_path,
        status=status,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one VLM smoke decision and write a static HTML monitor report.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a GoalAdapter input JSON with current_rgb set.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("reports") / "vlm_smoke" / "latest"),
        help="Report directory to write artifacts into.",
    )
    parser.add_argument("--model", default="gemma4:26b", help="Ollama model name.")
    parser.add_argument("--timeout-seconds", type=float, default=420.0, help="Ollama request timeout.")
    parser.add_argument(
        "--raise-on-error",
        action="store_true",
        help="Raise provider errors instead of writing an error report.",
    )
    args = parser.parse_args(argv)

    report = run_vlm_smoke_monitor(
        input_path=args.input,
        output_dir=args.output_dir,
        model_name=args.model,
        timeout_seconds=args.timeout_seconds,
        raise_on_error=args.raise_on_error,
    )
    print(f"Wrote VLM smoke monitor to {report.output_dir}")
    print(f"status={report.status}")
    print(f"html={report.html_path}")
    return 0 if report.status == "success" else 1


def _copy_current_rgb(adapter_input: GoalAdapterInput, output_dir: Path) -> Path:
    if not isinstance(adapter_input.current_rgb, str) or not adapter_input.current_rgb:
        raise ValueError("GoalAdapter input must include current_rgb as an image path")
    source = Path(adapter_input.current_rgb)
    if not source.exists():
        raise ValueError(f"current_rgb image does not exist: {source}")

    suffix = source.suffix if source.suffix else ".png"
    destination = output_dir / f"current_rgb{suffix}"
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    return destination


def _generate_raw_or_decide(provider: DecisionProvider, adapter_input: GoalAdapterInput) -> str:
    generate_raw = getattr(provider, "generate_raw_decision", None)
    if callable(generate_raw):
        return str(generate_raw(adapter_input))
    output = provider.decide(adapter_input)
    return json.dumps(output.to_json(), ensure_ascii=False)


def _success_summary(
    model_name: str,
    adapter_input: GoalAdapterInput,
    output: GoalAdapterOutput,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "status": "success",
        "model": model_name,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "target_type": adapter_input.target_type.value,
        "high_level_target": adapter_input.high_level_target,
        "progress_state": adapter_input.progress_state.value,
        "s2e_status": adapter_input.s2e_status.value,
        "action_type": output.action_type.value,
        "confidence": output.confidence.value,
        "refined_goal_xy": output.refined_goal_xy,
        "controller_action": output.controller_action.value if output.controller_action else None,
        "reasoning": output.reasoning,
    }


def _error_summary(
    model_name: str,
    adapter_input: GoalAdapterInput,
    exc: Exception,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "status": "error",
        "model": model_name,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "target_type": adapter_input.target_type.value,
        "high_level_target": adapter_input.high_level_target,
        "progress_state": adapter_input.progress_state.value,
        "s2e_status": adapter_input.s2e_status.value,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def _render_html(
    image_path: Path,
    adapter_input: Mapping[str, Any],
    decision: Mapping[str, Any],
    raw_response: str,
    summary: Mapping[str, Any],
) -> str:
    status = str(summary.get("status", "unknown"))
    action_type = str(summary.get("action_type", "ERROR" if status == "error" else "unknown"))
    confidence = str(summary.get("confidence", "n/a"))
    reasoning = str(summary.get("reasoning") or summary.get("error_message") or "")
    elapsed = str(summary.get("elapsed_seconds", "n/a"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VLM Smoke Monitor</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f2933;
      --muted: #657488;
      --line: #d8dee8;
      --panel: #f7f9fc;
      --accent: #0f766e;
      --warn: #b45309;
      --error: #b91c1c;
    }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    header {{
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }}
    h1 {{
      font-size: 20px;
      margin: 0;
      letter-spacing: 0;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(280px, 1.15fr) minmax(320px, 0.85fr);
      gap: 20px;
      padding: 20px 24px 28px;
      max-width: 1280px;
      margin: 0 auto;
    }}
    .image-pane img {{
      width: 100%;
      max-height: 76vh;
      object-fit: contain;
      border: 1px solid var(--line);
      background: #eef2f7;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .metric {{
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 10px;
      min-height: 56px;
    }}
    .label {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 5px;
    }}
    .value {{
      font-size: 16px;
      overflow-wrap: anywhere;
    }}
    .status-success {{ color: var(--accent); }}
    .status-error {{ color: var(--error); }}
    section {{
      margin-bottom: 16px;
    }}
    h2 {{
      font-size: 15px;
      margin: 0 0 8px;
      letter-spacing: 0;
    }}
    pre {{
      margin: 0;
      padding: 12px;
      border: 1px solid var(--line);
      background: #f9fafb;
      overflow: auto;
      max-height: 36vh;
      font-size: 12px;
      line-height: 1.45;
    }}
    @media (max-width: 860px) {{
      main {{
        grid-template-columns: 1fr;
        padding: 16px;
      }}
      .summary {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>VLM Smoke Monitor</h1>
    <div class="status-{html.escape(status)}">{html.escape(status.upper())}</div>
  </header>
  <main>
    <div class="image-pane">
      <img src="{html.escape(image_path.name)}" alt="Current RGB observation">
    </div>
    <div>
      <div class="summary">
        <div class="metric">
          <div class="label">Action</div>
          <div class="value">{html.escape(action_type)}</div>
        </div>
        <div class="metric">
          <div class="label">Confidence</div>
          <div class="value">{html.escape(confidence)}</div>
        </div>
        <div class="metric">
          <div class="label">Elapsed Seconds</div>
          <div class="value">{html.escape(elapsed)}</div>
        </div>
        <div class="metric">
          <div class="label">Target</div>
          <div class="value">{html.escape(str(summary.get("high_level_target", "")))}</div>
        </div>
      </div>
      <section>
        <h2>Reasoning</h2>
        <pre>{html.escape(reasoning)}</pre>
      </section>
      <section>
        <h2>Decision JSON</h2>
        <pre>{html.escape(json.dumps(decision, indent=2, ensure_ascii=False))}</pre>
      </section>
      <section>
        <h2>Raw VLM Response</h2>
        <pre>{html.escape(raw_response)}</pre>
      </section>
      <section>
        <h2>Input JSON</h2>
        <pre>{html.escape(json.dumps(adapter_input, indent=2, ensure_ascii=False))}</pre>
      </section>
    </div>
  </main>
</body>
</html>
"""


def _read_json_object(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("input JSON must be an object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
