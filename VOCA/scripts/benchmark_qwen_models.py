#!/usr/bin/env python
import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pixel_candidate_gate import validate_pixel_candidate_selection
from qwen_vlm_planner import (
    CompactQwenNavVLMClient,
    _configure_direct_json_payload,
    enforce_memory_verdict_contract,
    selected_point_visual_risk,
)
from voca_s2e_bridge import import_nav_memory_qwen


def normalize_api_base(base_url: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("endpoint base URL is empty")
    return base if base.endswith("/v1") else base + "/v1"


def parse_endpoint_spec(spec: str) -> Dict[str, str]:
    if "=" not in str(spec):
        raise ValueError("endpoint must be LABEL=BASE_URL")
    label, base_url = str(spec).split("=", 1)
    label = label.strip()
    base_url = base_url.strip().rstrip("/")
    if not label or not base_url:
        raise ValueError("endpoint must contain a label and URL")
    return {"label": label, "base_url": base_url}


def parse_case_spec(spec: str) -> Dict[str, Any]:
    if "=" not in str(spec):
        raise ValueError("case must be NAME=QWEN_CALLS_JSONL:INDEX")
    name, source = str(spec).split("=", 1)
    path, index_text = source.rsplit(":", 1)
    return {"name": name.strip(), "path": path.strip(), "index": int(index_text)}


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(percentile)
    lower = int(rank)
    upper = min(len(ordered) - 1, lower + 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _rate(records: Sequence[Dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(1 for record in records if bool(record.get(key))) / float(len(records))


def _mean(records: Sequence[Dict[str, Any]], key: str) -> float:
    values = [float(record.get(key, 0.0) or 0.0) for record in records]
    return float(statistics.mean(values)) if values else 0.0


def summarize_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("warmup"):
            continue
        grouped[(record.get("endpoint"), record.get("model"))].append(record)

    rows: List[Dict[str, Any]] = []
    for (endpoint, model), items in sorted(grouped.items()):
        latencies = [float(item.get("latency_sec", 0.0) or 0.0) for item in items]
        rows.append(
            {
                "endpoint": endpoint,
                "model": model,
                "measured_calls": len(items),
                "case_count": len({item.get("case") for item in items}),
                "latency_mean_sec": round(_mean(items, "latency_sec"), 6),
                "latency_median_sec": round(float(statistics.median(latencies)), 6),
                "latency_p95_sec": round(_percentile(latencies, 0.95), 6),
                "json_success_rate": round(_rate(items, "json_ok"), 6),
                "schema_success_rate": round(_rate(items, "schema_ok"), 6),
                "candidate_gate_pass_rate": round(_rate(items, "candidate_gate_passed"), 6),
                "pipeline_accept_rate": round(_rate(items, "pipeline_accepted"), 6),
                "risky_point_rate": round(_rate(items, "risky_selected_point"), 6),
                "completion_tokens_mean": round(_mean(items, "completion_tokens"), 3),
                "prompt_tokens_mean": round(_mean(items, "prompt_tokens"), 3),
                "timeout_or_error_rate": round(_rate(items, "request_error"), 6),
            }
        )
    return rows


def _load_case(case_spec: Dict[str, Any]) -> Dict[str, Any]:
    path = Path(case_spec["path"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    index = int(case_spec["index"])
    if index < 0 or index >= len(rows):
        raise IndexError("case index {} outside {}".format(index, path))
    row = rows[index]
    vlm_input = row.get("vlm_input")
    if not isinstance(vlm_input, dict):
        raise ValueError("case {} has no vlm_input".format(case_spec["name"]))
    return {"name": case_spec["name"], "vlm_input": vlm_input}


def discover_model(api_base: str, timeout_s: float) -> Dict[str, str]:
    response = requests.get(api_base.rstrip("/") + "/models", timeout=timeout_s)
    response.raise_for_status()
    data = response.json().get("data") or []
    if len(data) != 1:
        raise RuntimeError("expected exactly one served model, found {}".format(len(data)))
    return {
        "model": str(data[0].get("id") or ""),
        "root": str(data[0].get("root") or ""),
    }


def _parse_model_json(compact_client: CompactQwenNavVLMClient, data: Dict[str, Any]) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for text in compact_client._response_text_candidates(data):
        try:
            return compact_client._nav_modules.utils.extract_json_object(text)
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError("response contained no parseable assistant text")


def _selected_point_risk(
    safe_output: Dict[str, Any],
    validation: Dict[str, Any],
    vlm_input: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if str(safe_output.get("action") or "").lower() != "go" or not validation.get("passed"):
        return None
    candidate = validation.get("candidate") if isinstance(validation.get("candidate"), dict) else {}
    raw_paths = vlm_input.get("metadata", {}).get("raw_observation_images", []) or []
    try:
        view_id = int(candidate.get("view_id"))
        point = candidate.get("point_px")
        bgr = cv2.imread(str(raw_paths[view_id]))
        if bgr is None:
            return {"requires_verification": True, "risk_reasons": ["image_unavailable"]}
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return selected_point_visual_risk(rgb, point)
    except Exception as exc:
        return {
            "requires_verification": True,
            "risk_reasons": ["risk_evaluation_error"],
            "error": "{}: {}".format(type(exc).__name__, str(exc))[:240],
        }


def run_one_request(
    *,
    compact_client: CompactQwenNavVLMClient,
    endpoint_label: str,
    model: str,
    model_root: str,
    case: Dict[str, Any],
    repeat: int,
    warmup: bool,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "endpoint": endpoint_label,
        "model": model,
        "model_root": model_root,
        "case": case["name"],
        "repeat": int(repeat),
        "warmup": bool(warmup),
        "json_ok": False,
        "schema_ok": False,
        "candidate_gate_passed": False,
        "pipeline_accepted": False,
        "risky_selected_point": False,
        "request_error": False,
    }
    started = time.perf_counter()
    try:
        payload = compact_client._build_payload(case["vlm_input"])
        data = compact_client._post_chat_completion(payload)
        record["latency_sec"] = round(time.perf_counter() - started, 6)
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        record["finish_reason"] = choice.get("finish_reason") if isinstance(choice, dict) else None
        record["prompt_tokens"] = int(usage.get("prompt_tokens", 0) or 0)
        record["completion_tokens"] = int(usage.get("completion_tokens", 0) or 0)
        record["total_tokens"] = int(usage.get("total_tokens", 0) or 0)

        raw_output = _parse_model_json(compact_client, data)
        record["json_ok"] = True
        record["raw_vlm_output"] = raw_output
        record["raw_action"] = raw_output.get("action")
        safe_output, warnings = compact_client._nav_modules.safety.sanitize_vlm_output(
            raw_output,
            case["vlm_input"],
        )
        safe_output, memory_contract, memory_warnings = enforce_memory_verdict_contract(
            safe_output,
            case["vlm_input"],
        )
        record["schema_ok"] = True
        record["action"] = safe_output.get("action")
        record["warnings"] = list(warnings) + list(memory_warnings)
        record["memory_ops"] = list(safe_output.get("memory_ops") or [])
        record["memory_contract_validation"] = memory_contract
        validation = validate_pixel_candidate_selection(safe_output, case["vlm_input"])
        record["candidate_gate_passed"] = bool(validation.get("passed"))
        record["candidate_gate_reason"] = validation.get("reason")
        candidate = validation.get("candidate") if isinstance(validation.get("candidate"), dict) else {}
        record["selected_candidate_ref"] = candidate.get("candidate_ref") or safe_output.get(
            "selected_candidate_ref"
        )
        risk = _selected_point_risk(safe_output, validation, case["vlm_input"])
        record["selected_point_visual_risk"] = risk
        record["risky_selected_point"] = bool(
            isinstance(risk, dict) and risk.get("requires_verification")
        )
        if str(safe_output.get("action") or "").lower() == "go":
            record["pipeline_accepted"] = bool(
                validation.get("passed") and not record["risky_selected_point"]
            )
            if not validation.get("passed"):
                record["executable_action"] = "request_observation"
                record["execution_guard_reason"] = "pixel_candidate_gate_rejected_go"
            elif record["risky_selected_point"]:
                record["executable_action"] = "go_requires_point_verification"
                record["execution_guard_reason"] = "selected_point_visual_risk"
            else:
                record["executable_action"] = "go"
                record["execution_guard_reason"] = ""
        else:
            record["pipeline_accepted"] = True
            record["executable_action"] = safe_output.get("action")
            record["execution_guard_reason"] = ""
    except Exception as exc:
        record["latency_sec"] = round(time.perf_counter() - started, 6)
        record["request_error"] = True
        record["error"] = "{}: {}".format(type(exc).__name__, str(exc))[:500]
        response = getattr(exc, "response", None)
        if response is not None:
            record["error_response"] = str(getattr(response, "text", ""))[:2000]
    return record


def run_endpoint(
    endpoint: Dict[str, str],
    cases: Sequence[Dict[str, Any]],
    *,
    repeats: int,
    warmups: int,
    api_key: str,
    timeout_s: float,
    max_tokens: int,
    image_max_side: int,
) -> List[Dict[str, Any]]:
    nav_modules = import_nav_memory_qwen()
    api_base = normalize_api_base(endpoint["base_url"])
    model_info = discover_model(api_base, timeout_s)
    base_client = nav_modules.vlm_client.OpenAICompatibleVLMClient(
        base_url=api_base,
        api_key=api_key,
        model=model_info["model"],
        timeout_s=float(timeout_s),
        temperature=0.0,
        max_tokens=int(max_tokens),
        image_max_side=int(image_max_side),
        jpeg_quality=70,
        extra_payload={},
    )
    base_client.max_json_retries = 0
    _configure_direct_json_payload(base_client)
    compact_client = CompactQwenNavVLMClient(base_client, nav_modules)
    records: List[Dict[str, Any]] = []

    for warmup_index in range(max(0, int(warmups))):
        record = run_one_request(
            compact_client=compact_client,
            endpoint_label=endpoint["label"],
            model=model_info["model"],
            model_root=model_info["root"],
            case=cases[0],
            repeat=warmup_index,
            warmup=True,
        )
        records.append(record)
        print(
            "[{}] warmup {} {:.3f}s json={}".format(
                endpoint["label"], warmup_index + 1, record["latency_sec"], record["json_ok"]
            ),
            flush=True,
        )

    for case in cases:
        for repeat in range(max(1, int(repeats))):
            record = run_one_request(
                compact_client=compact_client,
                endpoint_label=endpoint["label"],
                model=model_info["model"],
                model_root=model_info["root"],
                case=case,
                repeat=repeat,
                warmup=False,
            )
            records.append(record)
            print(
                "[{}] {} {}/{} {:.3f}s action={} json={} gate={} risky={}".format(
                    endpoint["label"],
                    case["name"],
                    repeat + 1,
                    repeats,
                    record["latency_sec"],
                    record.get("action"),
                    record["json_ok"],
                    record["candidate_gate_passed"],
                    record["risky_selected_point"],
                ),
                flush=True,
            )
    return records


def _write_outputs(output_dir: Path, records: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "qwen_model_benchmark_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = summarize_records(records)
    (output_dir / "qwen_model_benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if summary:
        with (output_dir / "qwen_model_benchmark_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark served Qwen3-VL models on identical saved VOCA navigation inputs."
    )
    parser.add_argument("--endpoint", action="append", required=True, help="LABEL=BASE_URL")
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="NAME=QWEN_CALLS_JSONL:ZERO_BASED_INDEX",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--image-max-side", type=int, default=512)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    endpoints = [parse_endpoint_spec(spec) for spec in args.endpoint]
    cases = [_load_case(parse_case_spec(spec)) for spec in args.case]
    all_records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
        futures = {
            executor.submit(
                run_endpoint,
                endpoint,
                cases,
                repeats=args.repeats,
                warmups=args.warmups,
                api_key=args.api_key,
                timeout_s=args.timeout,
                max_tokens=args.max_tokens,
                image_max_side=args.image_max_side,
            ): endpoint
            for endpoint in endpoints
        }
        for future in as_completed(futures):
            all_records.extend(future.result())

    _write_outputs(Path(args.output_dir), all_records)
    print("wrote benchmark outputs to {}".format(args.output_dir), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
