import base64
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests
from PIL import ImageDraw

from nav_audit import build_memory_free_vlm_input
from navigation_supervisor import (
    apply_policy_safe_navigation_context,
    assert_policy_input_safe,
    build_localization_contract,
)
from pixel_candidate_gate import (
    PIXEL_CANDIDATE_SCHEMA_VERSION,
    build_pixel_candidates,
    build_saturation_recovery_candidates,
    partition_pixel_candidates,
    render_pixel_candidates,
    validate_pixel_candidate_selection,
)
from qwen_point_planner import (
    QwenPointPlanner,
    QwenPointPlannerConfig,
    _lower_center_point,
    draw_selected_point,
)
from qwen_memory_prompt import (
    build_compact_memory_projection,
    compact_memory_json,
)
from voca_memory_sidecar import VOCAMemorySidecar
from voca_s2e_bridge import import_nav_memory_qwen


def _normalize_angle_deg(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


_NAV_VLM_OUTPUT_GUIDED_JSON: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "schema_version": {"const": "nav_vlm_waypoint_v1"},
        "action": {"enum": ["go", "rotate", "stop", "request_observation"]},
        "selected_view_id": {"type": ["integer", "null"]},
        "selected_view_type": {"type": ["string", "null"]},
        "selected_image_point": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                {"type": "null"},
            ]
        },
        "selected_candidate_ref": {"type": ["string", "null"]},
        "fine_goal": {"type": "object"},
        "observation_request": {"type": ["object", "null"]},
        "strategic_state": {
            "type": "object",
            "properties": {
                "current_room": {
                    "enum": [
                        "living_room",
                        "bedroom",
                        "kitchen",
                        "dining_room",
                        "bathroom",
                        "hallway",
                        "office",
                        "utility_room",
                        "unknown",
                    ]
                },
                "target_evidence": {
                    "enum": ["confirmed", "possible", "context_only", "none"]
                },
                "navigation_intent": {
                    "enum": [
                        "approach_target",
                        "verify_target",
                        "search_current_room",
                        "leave_current_room",
                        "traverse_gateway",
                        "revisit_promising_place",
                        "escape_deadlock",
                    ]
                },
                "room_search_status": {
                    "enum": [
                        "unsearched",
                        "partial",
                        "exhausted",
                        "target_likely",
                        "target_found",
                    ]
                },
                "selected_exit_ref": {"type": ["string", "null"]},
                "reason_code": {"type": "string"},
            },
            "required": [
                "current_room",
                "target_evidence",
                "navigation_intent",
                "room_search_status",
                "selected_exit_ref",
                "reason_code",
            ],
        },
        "reasoning": {"type": "object"},
        "control": {"type": "object"},
        "confidence": {"enum": ["high", "medium", "low"]},
        "memory_ops": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {
                        "enum": [
                            "confirm_revisit_node",
                            "confirm_same_place",
                            "reject_revisit_candidate",
                            "defer_revisit_candidate",
                            "request_merge_nodes",
                            "update_object_belief",
                            "mark_object_seen",
                        ]
                    },
                    "candidate_ref": {"type": ["string", "null"]},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reason": {"type": "string"},
                },
                "required": ["op", "candidate_ref", "confidence", "reason"],
            },
        },
    },
    "required": [
        "schema_version",
        "action",
        "selected_view_id",
        "selected_view_type",
        "selected_image_point",
        "selected_candidate_ref",
        "fine_goal",
        "observation_request",
        "strategic_state",
        "reasoning",
        "control",
        "confidence",
        "memory_ops",
    ],
}

_STRATEGIC_ROOMS = {
    "living_room",
    "bedroom",
    "kitchen",
    "dining_room",
    "bathroom",
    "hallway",
    "office",
    "utility_room",
    "unknown",
}
_STRATEGIC_TARGET_EVIDENCE = {"confirmed", "possible", "context_only", "none"}
_STRATEGIC_INTENTS = {
    "approach_target",
    "verify_target",
    "search_current_room",
    "leave_current_room",
    "traverse_gateway",
    "revisit_promising_place",
    "escape_deadlock",
}
_STRATEGIC_ROOM_SEARCH_STATUS = {
    "unsearched",
    "partial",
    "exhausted",
    "target_likely",
    "target_found",
}
_EXIT_INTENTS = {"leave_current_room", "traverse_gateway"}
_FORCE_LEAVE_COMPATIBLE_INTENTS = _EXIT_INTENTS | {
    "revisit_promising_place",
    "escape_deadlock",
}


def normalize_strategic_state(
    value: Any,
    *,
    action: str = "",
    reasoning: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize the model's room-level plan without granting it backend authority."""
    raw = value if isinstance(value, dict) else {}
    normalized_action = str(action or "").strip().lower()
    reasoning = reasoning if isinstance(reasoning, dict) else {}
    reason_text = " ".join(
        str(reasoning.get(key) or "").lower()
        for key in ("decision_reason", "goal_reason", "failure_mode", "short_text")
    )

    room_aliases = {
        "livingroom": "living_room",
        "lounge": "living_room",
        "diningroom": "dining_room",
        "corridor": "hallway",
        "laundry": "utility_room",
        "utility": "utility_room",
    }
    room = str(raw.get("current_room") or "unknown").strip().lower().replace(" ", "_")
    room = room_aliases.get(room, room)
    if room not in _STRATEGIC_ROOMS:
        room = "unknown"

    evidence = str(raw.get("target_evidence") or "").strip().lower()
    if evidence not in _STRATEGIC_TARGET_EVIDENCE:
        if normalized_action == "stop":
            evidence = "confirmed"
        elif "target" in reason_text and any(
            token in reason_text for token in ("visible", "approach", "center", "verify")
        ):
            evidence = "possible"
        else:
            evidence = "none"

    intent = str(raw.get("navigation_intent") or "").strip().lower()
    if intent not in _STRATEGIC_INTENTS:
        if normalized_action == "stop":
            intent = "verify_target"
        elif "escape" in reason_text or "deadlock" in reason_text:
            intent = "escape_deadlock"
        elif evidence in {"confirmed", "possible"}:
            intent = "approach_target" if normalized_action == "go" else "verify_target"
        else:
            intent = "search_current_room"

    search_status = str(raw.get("room_search_status") or "").strip().lower()
    if search_status not in _STRATEGIC_ROOM_SEARCH_STATUS:
        if evidence == "confirmed":
            search_status = "target_found"
        elif evidence == "possible":
            search_status = "target_likely"
        else:
            search_status = "partial"

    selected_exit = raw.get("selected_exit_ref")
    if selected_exit is not None:
        selected_exit = str(selected_exit).strip()[:64] or None
    reason_code = str(raw.get("reason_code") or "STRATEGY_MISSING_DEFAULT").strip()
    reason_code = reason_code[:96] or "STRATEGY_MISSING_DEFAULT"
    return {
        "current_room": room,
        "target_evidence": evidence,
        "navigation_intent": intent,
        "room_search_status": search_status,
        "selected_exit_ref": selected_exit,
        "reason_code": reason_code,
    }

_COMPACT_JSON_ONLY_SYSTEM = (
    "Return exactly one valid JSON object in assistant content. "
    "Do not include reasoning, analysis, explanations, markdown, or text outside JSON."
)

_COMPACT_JSON_RETRY_INSTRUCTION = (
    "Previous response was prose or invalid JSON. Return exactly one JSON object matching "
    "nav_vlm_waypoint_v1. Do not include reasoning. Do not include analysis, explanations, "
    "markdown, or any text before or after the JSON object."
)

_COMPACT_FINAL_JSON_REMINDER = (
    "FINAL ANSWER JSON ONLY. Your first character must be { and your last character must be }. "
    "Do not include reasoning. Do not describe the views. Do not write analysis, explanation, "
    "markdown, or prose. If uncertain, return a valid rotate JSON object."
)


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def selected_point_visual_risk(
    rgb: np.ndarray,
    point_px: Sequence[int],
    *,
    patch_radius_px: int = 64,
    min_luma_std: float = 6.0,
    min_edge_density: float = 0.01,
    extreme_exposure_fraction: float = 0.85,
) -> Dict[str, Any]:
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        return {
            "requires_verification": True,
            "risk_reasons": ["invalid_rgb_shape"],
            "patch_shape": [],
        }
    height, width = image.shape[:2]
    try:
        u = max(0, min(width - 1, int(round(float(point_px[0])))))
        v = max(0, min(height - 1, int(round(float(point_px[1])))))
    except Exception:
        return {
            "requires_verification": True,
            "risk_reasons": ["invalid_point"],
            "patch_shape": [],
        }
    radius = max(8, int(patch_radius_px))
    patch = image[
        max(0, v - radius) : min(height, v + max(4, radius // 2)),
        max(0, u - radius) : min(width, u + radius),
    ]
    if patch.size == 0:
        return {
            "requires_verification": True,
            "risk_reasons": ["empty_patch"],
            "patch_shape": list(patch.shape),
        }
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    luma_mean = float(np.mean(gray))
    luma_std = float(np.std(gray))
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    edge_density = float(np.mean(np.abs(laplacian) > 8.0))
    overexposed_fraction = float(np.mean(gray >= 245))
    underexposed_fraction = float(np.mean(gray <= 10))
    reasons: List[str] = []
    if luma_std < float(min_luma_std) and edge_density < float(min_edge_density):
        reasons.append("low_texture")
    if overexposed_fraction >= float(extreme_exposure_fraction):
        reasons.append("overexposed")
    if underexposed_fraction >= float(extreme_exposure_fraction):
        reasons.append("underexposed")
    return {
        "requires_verification": bool(reasons),
        "risk_reasons": reasons,
        "point_px": [u, v],
        "patch_shape": list(patch.shape),
        "luma_mean": round(luma_mean, 4),
        "luma_std": round(luma_std, 4),
        "edge_density": round(edge_density, 6),
        "overexposed_fraction": round(overexposed_fraction, 6),
        "underexposed_fraction": round(underexposed_fraction, 6),
    }


def _format_target_context(context: Dict[str, Any], max_items_per_key: int = 8) -> str:
    if not context:
        return "none"
    preferred_keys = ["Supports", "StrongCooccurs", "Gateways", "Lookalikes"]
    remaining_keys = sorted(key for key in context.keys() if key not in preferred_keys)
    parts: List[str] = []
    for key in preferred_keys + remaining_keys:
        values = context.get(key)
        if values is None or values == "":
            continue
        if isinstance(values, dict):
            continue
        if isinstance(values, (list, tuple, set)):
            clean_values: List[str] = []
            for value in values:
                text = str(value).strip()
                if text and text not in clean_values:
                    clean_values.append(text)
            if not clean_values:
                continue
            shown = clean_values[:max_items_per_key]
            suffix = ""
            if len(clean_values) > max_items_per_key:
                suffix = ", +{} more".format(len(clean_values) - max_items_per_key)
            parts.append("{}={}{}".format(key, ", ".join(shown), suffix))
        else:
            parts.append("{}={}".format(key, str(values).strip()))
    return "; ".join(parts) or "none"


def _view_lines(observation: Dict[str, Any]) -> List[str]:
    views = observation.get("views", []) if isinstance(observation.get("views"), list) else []
    lines: List[str] = []
    for view in views:
        lines.append(
            "view_id={view_id} view_type={view_type} angle={angle}".format(
                view_id=view.get("view_id"),
                view_type=view.get("view_type"),
                angle=view.get("relative_heading_deg"),
            )
        )
    return lines


def _format_coarse_goal(task: Dict[str, Any]) -> str:
    coarse = task.get("coarse_goal") if isinstance(task.get("coarse_goal"), dict) else {}
    if not coarse.get("goal_geometry_available"):
        return "semantic object category only; spatial geometry unavailable"
    parts = [
        "type={}".format(coarse.get("type")),
        "source={}".format(coarse.get("source")),
    ]
    for key in ("relative_bearing_deg", "distance_m", "distance_range_m", "map_xy", "uncertainty"):
        if coarse.get(key) is not None:
            parts.append("{}={}".format(key, coarse.get(key)))
    parts.append("temporary_detour_allowed={}".format(bool(coarse.get("temporary_detour_allowed", True))))
    return "; ".join(parts)


def _target_prompt_label(target: Any) -> str:
    normalized = str(target or "target object").strip().lower()
    aliases = {
        "houseplant": "houseplant / indoor potted plant / decorative indoor tree",
        "plant": "houseplant / indoor potted plant / decorative indoor tree",
        "tv_monitor": "television / TV monitor / display screen",
        "tv": "television / TV monitor / display screen",
        "sofa": "sofa / couch",
    }
    return aliases.get(normalized, str(target or "target object"))


def _target_identity_criteria(target: Any) -> str:
    normalized = str(target or "").strip().lower().replace("_", " ")
    criteria = {
        "chair": (
            "Require a conventional single-person sitting chair with a distinct horizontal seat "
            "and an upright backrest plus legs or pedestal. Reject incline workout benches, gym "
            "equipment, folded furniture, stools without backs, sofas, and beds."
        ),
        "sofa": (
            "Require a multi-person upholstered couch with a long seat and backrest. Reject a "
            "single chair, bed, workout bench, ottoman, and piled cushions."
        ),
        "bed": (
            "Require a sleeping mattress on a bed frame or base. Reject sofas, benches, tables, "
            "and a loose blanket without a mattress."
        ),
        "toilet": (
            "Require a ceramic toilet bowl with seat and tank or flush body. Reject sinks, bidets, "
            "buckets, cabinets, and washing machines."
        ),
        "tv monitor": (
            "Require an electronic display screen with a visible screen boundary or bezel. Reject "
            "windows, mirrors, framed pictures, dark cabinet panels, and microwave doors."
        ),
        "plant": (
            "Require coherent visible stems or leaves of a real or decorative indoor plant. A pot is "
            "helpful but not mandatory when connected plant structure is unambiguous. Reject "
            "leaf-pattern artwork, curtains, outdoor vegetation through windows, and an empty pot."
        ),
        "houseplant": (
            "Require coherent visible stems or leaves of a real or decorative indoor plant. A pot is "
            "helpful but not mandatory when connected plant structure is unambiguous. Reject "
            "leaf-pattern artwork, curtains, outdoor vegetation through windows, and an empty pot."
        ),
    }
    return criteria.get(
        normalized,
        "Require unmistakable physical features of the exact category and reject scene-context guesses.",
    )


def _pixel_candidate_lines(vlm_input: Dict[str, Any]) -> List[str]:
    block = vlm_input.get("pixel_candidates")
    candidates = block.get("candidates", []) if isinstance(block, dict) else []
    lines: List[str] = []
    for candidate in candidates[:30] if isinstance(candidates, list) else []:
        if not isinstance(candidate, dict):
            continue
        point = candidate.get("point_px")
        point_text = "null"
        if isinstance(point, (list, tuple)) and len(point) == 2:
            point_text = "[{},{}]".format(int(point[0]), int(point[1]))
        lines.append(
            "{marker} ref={ref} view={view} view_id={view_id} angle={angle} "
            "point={point} verify={verify} score={score} pixelnav={pixelnav} "
            "policy_action={policy_action} status={status} topo={topo} relation={relation}".format(
                marker=candidate.get("marker_id"),
                ref=candidate.get("candidate_ref"),
                view=candidate.get("view_type_hint"),
                view_id=candidate.get("view_id"),
                angle=candidate.get("relative_heading_deg"),
                point=point_text,
                verify=int(bool(candidate.get("requires_rgb_verification"))),
                score=candidate.get("score"),
                pixelnav=(
                    candidate.get("pixelnav_feasibility", {}).get(
                        "policy_feasibility_score", "-"
                    )
                    if isinstance(candidate.get("pixelnav_feasibility"), dict)
                    else "-"
                ),
                policy_action=(
                    candidate.get("pixelnav_feasibility", {}).get(
                        "predicted_action", "-"
                    )
                    if isinstance(candidate.get("pixelnav_feasibility"), dict)
                    else "-"
                ),
                status=candidate.get("status"),
                topo=candidate.get("topological_candidate_ref") or "-",
                relation=candidate.get("topological_relation_type") or "frontier",
            )
        )
    if not lines:
        return ["none; action=go is unavailable, so rotate or request_observation"]
    return lines


def _revisit_candidate_refs(vlm_input: Dict[str, Any]) -> List[str]:
    memory = vlm_input.get("memory") if isinstance(vlm_input.get("memory"), dict) else {}
    candidate_refs = memory.get("candidate_refs") if isinstance(memory.get("candidate_refs"), dict) else {}
    revisits = candidate_refs.get("revisits") if isinstance(candidate_refs.get("revisits"), list) else []
    if not revisits:
        place_recognition = (
            memory.get("place_recognition")
            if isinstance(memory.get("place_recognition"), dict)
            else {}
        )
        revisits = (
            place_recognition.get("revisit_candidates")
            if isinstance(place_recognition.get("revisit_candidates"), list)
            else []
        )
    refs: List[str] = []
    for candidate in revisits:
        if not isinstance(candidate, dict):
            continue
        candidate_ref = str(candidate.get("candidate_ref") or "").strip()
        if candidate_ref and candidate_ref not in refs:
            refs.append(candidate_ref)
    return refs


def enforce_memory_verdict_contract(
    vlm_output: Dict[str, Any],
    vlm_input: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """Make unresolved revisit proposals explicit without mutating topology."""
    output = dict(vlm_output) if isinstance(vlm_output, dict) else {}
    candidate_refs = _revisit_candidate_refs(vlm_input)
    raw_ops = output.get("memory_ops")
    ops = [dict(op) for op in raw_ops if isinstance(op, dict)] if isinstance(raw_ops, list) else []
    output["memory_ops"] = ops
    relevant_ops = [
        op
        for op in ops
        if str(op.get("candidate_ref") or "") in candidate_refs
        and str(op.get("op") or "")
        in {
            "confirm_revisit_node",
            "confirm_same_place",
            "reject_revisit_candidate",
            "defer_revisit_candidate",
            "request_merge_nodes",
        }
    ]
    validation = {
        "schema_version": "memory_verdict_contract_v1",
        "required": bool(candidate_refs),
        "candidate_refs": candidate_refs,
        "passed": not candidate_refs or bool(relevant_ops),
        "reason": "no_revisit_candidate" if not candidate_refs else "explicit_vlm_verdict",
        "backend_inserted_defer": False,
    }
    warnings: List[str] = []
    if candidate_refs and not relevant_ops:
        output["memory_ops"].append(
            {
                "op": "defer_revisit_candidate",
                "candidate_ref": candidate_refs[0],
                "confidence": 0.0,
                "reason": (
                    "VLM omitted the mandatory revisit verdict; backend deferred topology mutation"
                ),
                "source": "backend_memory_contract_guard",
            }
        )
        validation.update(
            reason="vlm_verdict_missing_backend_deferred",
            backend_inserted_defer=True,
        )
        warnings.append("memory_verdict_missing_backend_deferred")
    return output, validation, warnings


def _configure_direct_json_payload(client: Any) -> None:
    if not hasattr(client, "extra_payload"):
        return
    extra = dict(getattr(client, "extra_payload") or {})
    try:
        qwen_seed = int(
            os.environ.get(
                "VOCA_QWEN_SEED",
                os.environ.get("VOCA_BENCHMARK_SEED", "20260710"),
            )
        )
    except ValueError:
        qwen_seed = 20260710
    extra.setdefault("seed", qwen_seed)
    model_name = str(getattr(client, "model", "") or "").lower()
    is_thinking_model = "thinking" in model_name
    if is_thinking_model:
        extra.setdefault("chat_template_kwargs", {"enable_thinking": False})
    else:
        extra.pop("chat_template_kwargs", None)
    if _env_bool("VOCA_QWEN_VLM_GUIDED_JSON", True):
        extra["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "nav_vlm_waypoint",
                "description": "One executable semantic navigation supervisor decision.",
                "schema": _NAV_VLM_OUTPUT_GUIDED_JSON,
                "strict": True,
            },
        }
        # vLLM 0.24 removed the legacy guided_json request field.
        extra.pop("guided_json", None)
    else:
        extra.setdefault("response_format", {"type": "json_object"})
    try:
        thinking_token_budget = int(
            os.environ.get("VOCA_QWEN_VLM_THINKING_TOKEN_BUDGET", "1024")
        )
    except ValueError:
        thinking_token_budget = 1024
    if is_thinking_model and thinking_token_budget > 0:
        extra.setdefault("thinking_token_budget", thinking_token_budget)
    else:
        extra.pop("thinking_token_budget", None)
    client.extra_payload = extra


def _configure_compact_image_options(client: Any) -> None:
    model_name = str(getattr(client, "model", "") or "").lower()
    default_image_max_side = "640" if "instruct" in model_name else "512"
    try:
        client.image_max_side = int(
            os.environ.get("VOCA_QWEN_VLM_IMAGE_MAX_SIDE", default_image_max_side)
        )
    except ValueError:
        client.image_max_side = 512
    try:
        client.jpeg_quality = int(os.environ.get("VOCA_QWEN_VLM_JPEG_QUALITY", "70"))
    except ValueError:
        client.jpeg_quality = 70
    explicit_vlm_tokens = os.environ.get("VOCA_QWEN_VLM_MAX_TOKENS")
    try:
        configured_tokens = int(
            explicit_vlm_tokens
            if explicit_vlm_tokens is not None
            else os.environ.get("QWEN_MAX_TOKENS", "2048")
        )
    except ValueError:
        configured_tokens = 2048
    if "thinking" in model_name and explicit_vlm_tokens is None:
        configured_tokens = max(4096, configured_tokens)
    client.max_tokens = configured_tokens
    try:
        client.max_json_retries = int(os.environ.get("VOCA_QWEN_VLM_MAX_JSON_RETRIES", "1"))
    except ValueError:
        client.max_json_retries = 1


def _compact_nav_prompt(vlm_input: Dict[str, Any]) -> str:
    task = vlm_input.get("task", {}) if isinstance(vlm_input.get("task"), dict) else {}
    observation = vlm_input.get("observation", {}) if isinstance(vlm_input.get("observation"), dict) else {}
    target = _target_prompt_label(
        task.get("target_object") or task.get("instruction") or "target object"
    )
    context = task.get("target_context") if isinstance(task.get("target_context"), dict) else {}
    context_text = _format_target_context(context)
    coarse_goal_text = _format_coarse_goal(task)
    view_lines = _view_lines(observation)
    candidate_lines = _pixel_candidate_lines(vlm_input)
    memory_json = compact_memory_json(vlm_input)
    memory_policy = (
        "Treat memory as backend-verified evidence. Never repeat a listed directional failure "
        "or select an exit with avoid=true. Candidate refs are mandatory backend capabilities; "
        "choose a visible safe pixel in the matching view. Temporary motion opposite the target "
        "context is allowed in escape_deadlock/backtrack. In that mode, prefer a "
        "visible non-avoided known backtrack or escape edge over an unknown frontier, even when it "
        "temporarily points opposite the target context. In verify_target, request evidence instead "
        "of stopping when uncertain. Objects in target_rejections were rejected by the identity "
        "critic; do not stop for or repeatedly approach the same lookalike."
    )
    schema_example = (
        '{"schema_version":"nav_vlm_waypoint_v1","action":"go","selected_candidate_ref":"px_v00_r1_c1",'
        '"selected_view_id":0,"selected_view_type":"front","selected_image_point":[320,360],'
        '"fine_goal":{"valid":true,"view_id":0,"view_type":"front","point_px":[320,360],'
        '"point_norm":[0.5,0.75],"projected_map_xy":null,"navigability":"likely_free"},'
        '"observation_request":{"valid":false,"mode":null,"center_yaw_deg":null,"step_deg":null,'
        '"num_views":null,"yaw_offsets_deg":null,"reason":null},'
        '"strategic_state":{"current_room":"living_room","target_evidence":"context_only",'
        '"navigation_intent":"search_current_room","room_search_status":"partial",'
        '"selected_exit_ref":null,"reason_code":"SAFE_INTERIOR_PROGRESS"},'
        '"reasoning":{"decision_reason":"G02_VISIBLE_FLOOR_TOWARD_GOAL",'
        '"goal_reason":"F02_VISIBLE_FLOOR_TOWARD_GOAL","failure_mode":null,'
        '"short_text":"visible floor leads toward target context"},'
        '"control":{"vlm_control_mode":"resume_async_navigation","rotate_yaw_deg":0,"ttl_ms":1000},'
        '"confidence":"medium","memory_ops":[]}'
    )
    rotate_example = (
        '{"schema_version":"nav_vlm_waypoint_v1","action":"rotate","selected_candidate_ref":null,'
        '"selected_view_id":null,"selected_view_type":null,"selected_image_point":null,'
        '"fine_goal":{"valid":false,"view_id":null,"view_type":null,"point_px":null,'
        '"point_norm":null,"projected_map_xy":null,"navigability":"unknown"},'
        '"observation_request":null,'
        '"strategic_state":{"current_room":"unknown","target_evidence":"none",'
        '"navigation_intent":"search_current_room","room_search_status":"partial",'
        '"selected_exit_ref":null,"reason_code":"NO_VISIBLE_SAFE_ROUTE"},'
        '"reasoning":{"decision_reason":"R02_NO_VISIBLE_NAVIGABLE_FLOOR",'
        '"goal_reason":"F08_NONE_ROTATE_OR_STOP","failure_mode":"no_visible_navigable_floor",'
        '"short_text":"rotate to find navigable floor or exit"},'
        '"control":{"vlm_control_mode":"pause_until_rotation_done","rotate_yaw_deg":45,"ttl_ms":1000},'
        '"confidence":"low","memory_ops":[]}'
    )
    observation_example = (
        '{"schema_version":"nav_vlm_waypoint_v1","action":"request_observation","selected_candidate_ref":null,'
        '"selected_view_id":null,"selected_view_type":null,"selected_image_point":null,'
        '"fine_goal":{"valid":false,"view_id":null,"view_type":null,"point_px":null,'
        '"point_norm":null,"projected_map_xy":null,"navigability":"unknown"},'
        '"observation_request":{"valid":true,"mode":"directed_sweep","center_yaw_deg":0,'
        '"step_deg":30,"num_views":5,"yaw_offsets_deg":[-60,-30,0,30,60],'
        '"reason":"need a wider local sweep before choosing a waypoint"},'
        '"strategic_state":{"current_room":"unknown","target_evidence":"none",'
        '"navigation_intent":"search_current_room","room_search_status":"unsearched",'
        '"selected_exit_ref":null,"reason_code":"INSUFFICIENT_ROOM_CONTEXT"},'
        '"reasoning":{"decision_reason":"R03_NEED_MORE_OBSERVATION",'
        '"goal_reason":"F08_NONE_ROTATE_OR_STOP","failure_mode":"insufficient_view_context",'
        '"short_text":"need wider sweep before selecting a safe waypoint"},'
        '"control":{"vlm_control_mode":"pause_until_observation_done","rotate_yaw_deg":0,"ttl_ms":1000},'
        '"confidence":"low","memory_ops":[]}'
    )
    stop_example = (
        '{"schema_version":"nav_vlm_waypoint_v1","action":"stop","selected_candidate_ref":null,'
        '"selected_view_id":null,"selected_view_type":null,"selected_image_point":null,'
        '"fine_goal":{"valid":false,"view_id":null,"view_type":null,"point_px":null,'
        '"point_norm":null,"projected_map_xy":null,"navigability":"unknown"},'
        '"observation_request":null,'
        '"strategic_state":{"current_room":"living_room","target_evidence":"confirmed",'
        '"navigation_intent":"verify_target","room_search_status":"target_found",'
        '"selected_exit_ref":null,"reason_code":"TARGET_CLOSE_CONFIRMED"},'
        '"reasoning":{"decision_reason":"S02_TARGET_REACHED_OR_TASK_DONE",'
        '"goal_reason":"F01_TARGET_OBJECT_VISIBLE","failure_mode":null,'
        '"short_text":"target object is clearly visible and close"},'
        '"control":{"vlm_control_mode":"hard_stop","rotate_yaw_deg":0,"ttl_ms":2000},'
        '"confidence":"high","memory_ops":[]}'
    )
    return (
        "Target object: {target}\n"
        "Target context cues: {context}\n"
        "Coarse spatial goal: {coarse_goal}\n"
        "Image size: width={width}, height={height}\n"
        "Views:\n{views}\n\n"
        "Pixel waypoint candidates:\n{candidate_lines}\n"
        "For action=go, copy one non-avoided ref into selected_candidate_ref and copy its exact "
        "view_id, view type, and point. Never invent a point or modify candidate coordinates. "
        "If no marked candidate is visibly navigable floor, return request_observation or rotate.\n\n"
        "Verified navigation memory: {memory_json}\n"
        "Memory policy: {memory_policy}\n"
        "Strategic-state policy: always classify current_room, target_evidence, navigation_intent, "
        "and room_search_status. Target context cues are hypotheses, never proof that the target is "
        "visible. For ordinary interior-floor progress use navigation_intent=search_current_room and "
        "selected_exit_ref=null. Use leave_current_room or traverse_gateway only when the selected pixel "
        "is visibly on a doorway/corridor threshold leading out of the room. If "
        "memory.strategy.force_leave_room=true and target_evidence is none or context_only, "
        "do not choose another generic interior-floor go. Select a visible doorway/corridor exit and set "
        "navigation_intent to leave_current_room or traverse_gateway. For those intents, "
        "selected_exit_ref must exactly equal selected_candidate_ref. If an exit cannot be identified "
        "from one view, request a directed sweep; from multiple views, rotate only when every exit is blocked.\n"
        "Mandatory memory_ops policy: always return memory_ops. When revisit_candidates is empty, "
        "return memory_ops=[]. When it is non-empty, return exactly one verdict for its candidate_ref: "
        "confirm_revisit_node only when spatially_plausible=true and the attached current/memory images "
        "show the same layout and openings; reject_revisit_candidate when clearly different; "
        "defer_revisit_candidate when images are missing or uncertain. Never silently omit the verdict. "
        "The backend independently verifies every operation and may reject it.\n\n"
        "Task: choose one visible navigable floor point leading toward the target/context and any "
        "declared coarse spatial goal. The coarse goal is a soft objective: when a direct route is "
        "blocked, temporarily move away from it to exit a room, escape a deadlock, or backtrack. "
        "The point must be inside the selected view image, not the panorama canvas. "
        "Avoid walls, furniture, shelves, object centers, mirrors, windows, blocked areas, "
        "and the outer 10% left/right image edges.\n\n"
        "Navigation policy: first check completion. Return action stop only when the target object is clearly visible, "
        "near enough to inspect, and not a lookalike. Prefer action go when you can see "
        "safe floor, a visible doorway, an open exit, or corridor floor that plausibly leads out "
        "toward the target context. Rotate only when no visible navigable floor, doorway, exit, "
        "or corridor candidate is available. Avoid repeating rotate when the current view is ambiguous; "
        "use request_observation with directed_sweep to inspect nearby headings before selecting a point. "
        "When multiple views are provided, do not request_observation again; choose the best go waypoint "
        "from those views or rotate if every view is blocked. For multi-view inputs, inspect all views "
        "internally, then answer with the final action only. Do not describe the views.\n\n"
        "Return ONLY final JSON in assistant content. Return one JSON object only. "
        "No markdown. No prose outside JSON. "
        "If the target object is clearly visible and close, return this stop shape:\n{stop_example}\n"
        "Use this exact schema shape for go:\n{schema_example}\n"
        "If no safe navigable point is visible, return this rotate shape:\n{rotate_example}\n"
        "If one view is insufficient or repeated rotate would be needed, return this request_observation shape:\n{observation_example}"
    ).format(
        target=target,
        context=context_text,
        coarse_goal=coarse_goal_text,
        width=observation.get("image_width", 640),
        height=observation.get("image_height", 480),
        views="\n".join(view_lines) or "view_id=0 view_type=front angle=0",
        candidate_lines="\n".join(candidate_lines),
        memory_json=memory_json,
        memory_policy=memory_policy,
        schema_example=schema_example,
        rotate_example=rotate_example,
        observation_example=observation_example,
        stop_example=stop_example,
    )


def _compact_nav_retry_prompt(vlm_input: Dict[str, Any]) -> str:
    task = vlm_input.get("task", {}) if isinstance(vlm_input.get("task"), dict) else {}
    observation = vlm_input.get("observation", {}) if isinstance(vlm_input.get("observation"), dict) else {}
    target = _target_prompt_label(
        task.get("target_object") or task.get("instruction") or "target object"
    )
    context = task.get("target_context") if isinstance(task.get("target_context"), dict) else {}
    coarse_goal_text = _format_coarse_goal(task)
    width = int(observation.get("image_width", 640) or 640)
    height = int(observation.get("image_height", 480) or 480)
    candidate_lines = _pixel_candidate_lines(vlm_input)
    memory_json = compact_memory_json(vlm_input)
    memory_policy = (
        "Never repeat a listed directional failure or use avoid=true. Temporary motion opposite "
        "the target context is allowed in escape_deadlock/backtrack; verify before stop."
    )
    retry_go_shape = (
        '{"schema_version":"nav_vlm_waypoint_v1","action":"go","selected_candidate_ref":"px_v00_r1_c1",'
        '"selected_view_id":0,"selected_view_type":"front","selected_image_point":[320,360],'
        '"fine_goal":{"valid":true,"view_id":0,"view_type":"front","point_px":[320,360],'
        '"point_norm":[0.5,0.75],"projected_map_xy":null,"navigability":"likely_free"},'
        '"observation_request":{"valid":false,"mode":null,"center_yaw_deg":null,"step_deg":null,'
        '"num_views":null,"yaw_offsets_deg":null,"reason":null},'
        '"strategic_state":{"current_room":"living_room","target_evidence":"context_only",'
        '"navigation_intent":"search_current_room","room_search_status":"partial",'
        '"selected_exit_ref":null,"reason_code":"SAFE_INTERIOR_PROGRESS"},'
        '"reasoning":{"decision_reason":"G02_VISIBLE_FLOOR_TOWARD_GOAL",'
        '"goal_reason":"F02_VISIBLE_FLOOR_TOWARD_GOAL","failure_mode":null,'
        '"short_text":"safe visible floor waypoint"},'
        '"control":{"vlm_control_mode":"resume_async_navigation","rotate_yaw_deg":0,"ttl_ms":1000},'
        '"confidence":"medium","memory_ops":[]}'
    )
    retry_rotate_shape = (
        '{"schema_version":"nav_vlm_waypoint_v1","action":"rotate","selected_candidate_ref":null,'
        '"selected_view_id":null,"selected_view_type":null,"selected_image_point":null,'
        '"fine_goal":{"valid":false,"view_id":null,"view_type":null,"point_px":null,'
        '"point_norm":null,"projected_map_xy":null,"navigability":"unknown"},'
        '"observation_request":null,'
        '"strategic_state":{"current_room":"unknown","target_evidence":"none",'
        '"navigation_intent":"search_current_room","room_search_status":"partial",'
        '"selected_exit_ref":null,"reason_code":"NO_VISIBLE_SAFE_ROUTE"},'
        '"reasoning":{"decision_reason":"R02_NO_VISIBLE_NAVIGABLE_FLOOR",'
        '"goal_reason":"F08_NONE_ROTATE_OR_STOP","failure_mode":"no_visible_navigable_floor",'
        '"short_text":"no safe visible navigable floor"},'
        '"control":{"vlm_control_mode":"pause_until_rotation_done","rotate_yaw_deg":45,"ttl_ms":1000},'
        '"confidence":"low","memory_ops":[]}'
    )
    retry_stop_shape = (
        '{"schema_version":"nav_vlm_waypoint_v1","action":"stop","selected_candidate_ref":null,'
        '"selected_view_id":null,"selected_view_type":null,"selected_image_point":null,'
        '"fine_goal":{"valid":false,"view_id":null,"view_type":null,"point_px":null,'
        '"point_norm":null,"projected_map_xy":null,"navigability":"unknown"},'
        '"observation_request":null,'
        '"strategic_state":{"current_room":"living_room","target_evidence":"confirmed",'
        '"navigation_intent":"verify_target","room_search_status":"target_found",'
        '"selected_exit_ref":null,"reason_code":"TARGET_CLOSE_CONFIRMED"},'
        '"reasoning":{"decision_reason":"S02_TARGET_REACHED_OR_TASK_DONE",'
        '"goal_reason":"F01_TARGET_OBJECT_VISIBLE","failure_mode":null,'
        '"short_text":"target object is clearly visible and close"},'
        '"control":{"vlm_control_mode":"hard_stop","rotate_yaw_deg":0,"ttl_ms":2000},'
        '"confidence":"high","memory_ops":[]}'
    )
    return (
        "JSON function call: choose the next ObjNav action from the attached view image(s).\n"
        "Target object: {target}\n"
        "Target context cues: {context}\n"
        "Coarse spatial goal: {coarse_goal}\n"
        "Image size: width={width}, height={height}\n"
        "Available views:\n{views}\n\n"
        "Pixel waypoint candidates:\n{candidate_lines}\n"
        "For go, copy one non-avoided selected_candidate_ref and its exact view/point. "
        "Never invent a point. Otherwise rotate or request observation.\n\n"
        "Verified navigation memory: {memory_json}\n"
        "Memory policy: {memory_policy}\n\n"
        "Always return strategic_state. Context cues are not target evidence. Ordinary room-floor go "
        "uses search_current_room with selected_exit_ref=null; exit intents are only for pixels visibly "
        "on a doorway/corridor threshold. If "
        "memory.strategy.force_leave_room=true without possible/confirmed target evidence, choose a "
        "doorway/corridor candidate, use leave_current_room or traverse_gateway, and copy "
        "selected_candidate_ref into selected_exit_ref. Do not continue generic interior-floor go.\n\n"
        "Treat a declared coarse spatial goal as a soft objective. A safe room exit, deadlock escape, "
        "or verified backtrack may temporarily point away from its bearing.\n\n"
        "Always return memory_ops. Use [] when revisit_candidates is empty. Otherwise return exactly "
        "one confirm_revisit_node, reject_revisit_candidate, or defer_revisit_candidate operation "
        "using the listed candidate_ref.\n\n"
        "Do not solve in natural language. Do not describe the views. Return exactly one JSON object. "
        "If the target object is clearly visible, close, and not a lookalike, return action stop. "
        "If a visible safe floor, doorway, exit, or corridor waypoint exists, return action go and set "
        "selected_view_id to one of the listed view_id values. The selected_image_point must be on "
        "navigable floor in that selected view and avoid the outer 10% left/right image edges. "
        "If every view is blocked, return rotate.\n\n"
        "STOP JSON shape:\n{retry_stop_shape}\n"
        "GO JSON shape:\n{retry_go_shape}\n"
        "ROTATE JSON shape:\n{retry_rotate_shape}"
    ).format(
        target=target,
        context=_format_target_context(context),
        coarse_goal=coarse_goal_text,
        width=width,
        height=height,
        views="\n".join(_view_lines(observation)) or "view_id=0 view_type=front angle=0",
        candidate_lines="\n".join(candidate_lines),
        memory_json=memory_json,
        memory_policy=memory_policy,
        retry_go_shape=retry_go_shape,
        retry_rotate_shape=retry_rotate_shape,
        retry_stop_shape=retry_stop_shape,
    )


class CompactQwenNavVLMClient:
    """VOCA memory-free compact prompt wrapper for Qwen OpenAI-compatible clients."""

    def __init__(self, base_client: Any, nav_modules: Any):
        self.base_client = base_client
        self._nav_modules = nav_modules
        _configure_direct_json_payload(self.base_client)

    def _build_user_content(
        self,
        vlm_input: Dict[str, Any],
        retry_instruction: Optional[str] = None,
        retry_compact: bool = False,
    ) -> List[Dict[str, Any]]:
        prompt = _compact_nav_retry_prompt(vlm_input) if retry_compact else _compact_nav_prompt(vlm_input)
        if not prompt.lstrip().startswith("/no_think"):
            prompt = "/no_think\n{}".format(prompt)
        if retry_instruction:
            prompt = "{}\n\n{}".format(prompt, retry_instruction)
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for label, path in self.base_client._collect_image_refs(vlm_input):
            content.append({"type": "text", "text": "Attached image: {}".format(label)})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._nav_modules.utils.image_to_data_url(
                            path,
                            max_side=self.base_client.image_max_side,
                            jpeg_quality=self.base_client.jpeg_quality,
                        )
                    },
                }
            )
        final_instruction = _COMPACT_FINAL_JSON_REMINDER
        if retry_instruction:
            final_instruction = "{}\n{}".format(retry_instruction, final_instruction)
        content.append({"type": "text", "text": final_instruction})
        return content

    def _build_payload(
        self,
        vlm_input: Dict[str, Any],
        retry_instruction: Optional[str] = None,
        retry_compact: bool = False,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.base_client.model,
            "messages": [
                {"role": "system", "content": _COMPACT_JSON_ONLY_SYSTEM},
                {
                    "role": "user",
                    "content": self._build_user_content(
                        vlm_input,
                        retry_instruction=retry_instruction,
                        retry_compact=retry_compact,
                    ),
                },
            ],
            "temperature": self.base_client.temperature,
            "max_tokens": self.base_client.max_tokens,
        }
        payload.update(getattr(self.base_client, "extra_payload", {}) or {})
        if retry_compact and "thinking_token_budget" in payload:
            try:
                retry_budget = int(
                    os.environ.get("VOCA_QWEN_VLM_RETRY_THINKING_TOKEN_BUDGET", "256")
                )
            except ValueError:
                retry_budget = 256
            if retry_budget > 0:
                try:
                    payload["thinking_token_budget"] = min(
                        int(payload["thinking_token_budget"]), retry_budget
                    )
                except (TypeError, ValueError):
                    payload["thinking_token_budget"] = retry_budget
        return payload

    @staticmethod
    def _stringify_text_value(value: Any) -> Optional[str]:
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list):
            chunks: List[str] = []
            for item in value:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text)
                elif isinstance(item, str) and item.strip():
                    chunks.append(item)
            return "\n".join(chunks) if chunks else None
        return None

    def _response_text_candidates(self, data: Dict[str, Any]) -> List[str]:
        base_candidates = getattr(self.base_client, "_response_text_candidates", None)
        if callable(base_candidates):
            return list(base_candidates(data))

        candidates: List[str] = []
        choices = data.get("choices") or []
        if choices:
            first = choices[0]
            message = first.get("message") if isinstance(first, dict) else {}
            if isinstance(message, dict):
                for key in ("content", "reasoning_content", "reasoning"):
                    text = self._stringify_text_value(message.get(key))
                    if text:
                        candidates.append(text)
            text = self._stringify_text_value(first.get("text")) if isinstance(first, dict) else None
            if text:
                candidates.append(text)
        top_level_text = self._stringify_text_value(data.get("output_text"))
        if top_level_text:
            candidates.append(top_level_text)
        return candidates

    def _post_chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        base_post = getattr(self.base_client, "_post_chat_completion", None)
        if callable(base_post):
            return base_post(payload)

        url = "{}/chat/completions".format(str(self.base_client.base_url).rstrip("/"))
        headers = {
            "Authorization": "Bearer {}".format(self.base_client.api_key),
            "Content-Type": "application/json",
        }
        response = requests.post(url, headers=headers, json=payload, timeout=self.base_client.timeout_s)
        response.raise_for_status()
        return response.json()

    def _repair_fragment_output(self, output: Dict[str, Any], vlm_input: Dict[str, Any]) -> Dict[str, Any]:
        if output.get("schema_version") == "nav_vlm_waypoint_v1" and output.get("action"):
            return output
        if not output.get("valid") or "point_px" not in output:
            return output
        obs = vlm_input.get("observation", {}) if isinstance(vlm_input.get("observation"), dict) else {}
        width = int(obs.get("image_width", 640) or 640)
        height = int(obs.get("image_height", 480) or 480)
        view_id = output.get("view_id")
        view_type = output.get("view_type")
        point = output.get("point_px")
        try:
            view_id_i = int(view_id)
            point_px = (int(point[0]), int(point[1]))
        except Exception:
            return output
        if not view_type:
            for view in obs.get("views", []) or []:
                if view.get("view_id") == view_id_i:
                    view_type = view.get("view_type")
                    break
        return self._nav_modules.schema.make_go_output(
            view_id=view_id_i,
            view_type=str(view_type or "front"),
            point_px=point_px,
            width=width,
            height=height,
            decision_reason="G02_VISIBLE_FLOOR_TOWARD_GOAL",
            goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
            short_text="repaired fine_goal fragment from compact Qwen output",
            confidence="medium",
        )

    def decide(self, vlm_input: Dict[str, Any]) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        last_text = ""
        attempts = max(0, int(getattr(self.base_client, "max_json_retries", 0))) + 1
        for attempt in range(attempts):
            retry_instruction = None
            if attempt > 0:
                retry_instruction = _COMPACT_JSON_RETRY_INSTRUCTION
            data = self._post_chat_completion(
                self._build_payload(
                    vlm_input,
                    retry_instruction=retry_instruction,
                    retry_compact=attempt > 0,
                )
            )
            for text in self._response_text_candidates(data):
                last_text = text
                try:
                    return self._repair_fragment_output(
                        self._nav_modules.utils.extract_json_object(text),
                        vlm_input,
                    )
                except Exception as exc:
                    last_error = exc
        if last_error is not None:
            raise ValueError(
                "failed to extract compact VLM JSON after {} attempt(s): {}; last_text={!r}".format(
                    attempts, last_error, last_text[:240]
                )
            )
        raise ValueError("failed to extract compact VLM JSON after {} attempt(s): response had no text candidates".format(attempts))


class QwenSelectedPointVerifier:
    """Focused, fail-closed RGB safety check for visually ambiguous waypoints."""

    schema_version = "qwen_rgb_point_verification_v1"

    def __init__(self, compact_client: CompactQwenNavVLMClient, nav_modules: Any):
        self.compact_client = compact_client
        self.base_client = compact_client.base_client
        self._nav_modules = nav_modules

    def verify(
        self,
        *,
        image_path: str,
        point_px: Sequence[int],
        candidate_ref: Optional[str],
        visual_risk: Dict[str, Any],
        required_surface: Optional[str] = None,
        current_room: Optional[str] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            max_tokens = int(os.environ.get("VOCA_QWEN_POINT_VERIFY_MAX_TOKENS", "1024"))
        except ValueError:
            max_tokens = 1024
        prompt = (
            "Strict RGB navigation safety check. The red square marks the selected waypoint at "
            "pixel [{u},{v}]. Decide whether that exact point lies on visibly traversable floor "
            "or doorway/corridor floor. Reject walls, doors, furniture, object surfaces, vertical "
            "planes, overexposed or ambiguous regions, and any point without clear floor support. "
            "When uncertain, valid must be false. Return only JSON with keys valid (boolean), "
            "surface (floor|doorway_floor|wall|door|furniture|object|unknown), confidence "
            "(high|medium|low), leads_out_of_current_room (boolean), and reason (at most 20 words)."
        ).format(u=int(point_px[0]), v=int(point_px[1]))
        if required_surface == "doorway_floor":
            current_room_hint = str(current_room or "unknown").strip() or "unknown"
            prompt += (
                " This waypoint declares a room-exit intent. Set leads_out_of_current_room=true "
                "only when a continuous traversable route through a visible doorway or corridor "
                "leads out of the current room. The marked point may be on the threshold or on safe "
                "floor immediately beyond that opening. Define the current room as the camera-side "
                "near foreground. A point visibly beyond a doorframe/opening relative to the camera "
                "is an exit-route point even when its surface is labeled floor. Do not reject such "
                "a point merely because it is not exactly on the threshold. Ordinary floor that "
                "does not cross a visible opening must have valid=false and "
                "leads_out_of_current_room=false. Current strategic room estimate: {room}. "
                "Treat this as a semantic hint, not ground truth."
            ).format(room=current_room_hint)
        payload = {
            "model": self.base_client.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a conservative robot waypoint safety verifier. "
                        "Never infer unseen floor. Return one JSON object only."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "/no_think\n{}".format(prompt)},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._nav_modules.utils.image_to_data_url(
                                    image_path,
                                    max_side=self.base_client.image_max_side,
                                    jpeg_quality=self.base_client.jpeg_quality,
                                )
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "FINAL JSON ONLY. If floor support is not visually clear, "
                                "set valid=false and surface=unknown."
                            ),
                        },
                    ],
                },
            ],
            "temperature": self.base_client.temperature,
            "max_tokens": max(64, max_tokens),
        }
        payload.update(getattr(self.base_client, "extra_payload", {}) or {})
        payload["response_format"] = {"type": "json_object"}
        payload.pop("guided_json", None)
        if "thinking_token_budget" in payload:
            try:
                payload["thinking_token_budget"] = min(
                    int(payload["thinking_token_budget"]),
                    max(64, min(256, max_tokens)),
                )
            except (TypeError, ValueError):
                payload["thinking_token_budget"] = max(64, min(256, max_tokens))

        result: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "triggered": True,
            "passed": False,
            "valid": False,
            "surface": "unknown",
            "confidence": "low",
            "reason": "verifier did not return a valid floor confirmation",
            "candidate_ref": candidate_ref,
            "point_px": [int(point_px[0]), int(point_px[1])],
            "image_ref": str(image_path),
            "model": str(self.base_client.model),
            "visual_risk": dict(visual_risk),
            "required_surface": required_surface,
            "current_room_context": str(current_room or "") or None,
            "backend_call_count": 0,
            "backend_error_count": 0,
            "backend_call_durations_sec": [],
            "consistency_retry_triggered": False,
        }
        backend_call_durations: List[float] = []
        try:
            retry_enabled = _env_bool(
                "VOCA_QWEN_POINT_VERIFY_CONSISTENCY_RETRY",
                True,
            )
            current_payload = payload
            for attempt in range(2 if retry_enabled else 1):
                call_started = time.perf_counter()
                try:
                    data = self.compact_client._post_chat_completion(
                        current_payload
                    )
                finally:
                    backend_call_durations.append(
                        float(time.perf_counter() - call_started)
                    )
                choices = data.get("choices") if isinstance(data, dict) else []
                first = choices[0] if isinstance(choices, list) and choices else {}
                message = first.get("message") if isinstance(first, dict) else {}
                content = self.compact_client._stringify_text_value(
                    message.get("content") if isinstance(message, dict) else None
                )
                parsed = self._nav_modules.utils.extract_json_object(content or "")
                surface = str(parsed.get("surface") or "unknown").strip().lower()
                confidence = str(parsed.get("confidence") or "low").strip().lower()
                valid = parsed.get("valid") is True
                leads_out_of_current_room = (
                    parsed.get("leads_out_of_current_room") is True
                )
                allowed_surfaces = {"floor", "doorway_floor"}
                contradiction = bool(
                    required_surface == "doorway_floor"
                    and not valid
                    and surface in allowed_surfaces
                    and confidence in {"high", "medium"}
                    and leads_out_of_current_room
                )
                passed = bool(
                    valid
                    and surface in allowed_surfaces
                    and confidence in {"high", "medium"}
                    and (
                        required_surface != "doorway_floor"
                        or leads_out_of_current_room
                    )
                )
                result.update(
                    {
                        "passed": passed,
                        "valid": valid,
                        "surface": surface,
                        "confidence": confidence,
                        "leads_out_of_current_room": leads_out_of_current_room,
                        "reason": str(
                            parsed.get("reason")
                            or "verifier returned no reason"
                        )[:240],
                        "finish_reason": (
                            first.get("finish_reason")
                            if isinstance(first, dict)
                            else None
                        ),
                        "response_consistency_contradiction": contradiction,
                    }
                )
                if not contradiction or attempt > 0:
                    break
                result["consistency_retry_triggered"] = True
                result["consistency_first_attempt"] = {
                    "valid": valid,
                    "surface": surface,
                    "confidence": confidence,
                    "leads_out_of_current_room": leads_out_of_current_room,
                    "reason": result["reason"],
                }
                retry_payload = dict(payload)
                retry_payload["messages"] = list(payload["messages"]) + [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "/no_think\nYour JSON is internally contradictory: "
                            "valid=false but the surface, confidence, and exit-route "
                            "fields indicate safe doorway floor. Reinspect the same red "
                            "point. Keep valid=false if it is unsafe; otherwise set "
                            "valid=true. Return one corrected JSON object only."
                        ),
                    },
                ]
                current_payload = retry_payload
        except Exception as exc:
            result["error"] = "{}: {}".format(type(exc).__name__, str(exc))[:240]
            result["reason"] = "verifier_error_fail_closed"
            result["backend_error_count"] = 1
        result["backend_call_count"] = len(backend_call_durations)
        result["backend_call_durations_sec"] = [
            round(float(duration), 6) for duration in backend_call_durations
        ]
        result["latency_sec"] = round(float(time.perf_counter() - started), 6)
        return result


class QwenTargetStopVerifier:
    """Fail-closed target completion check before Habitat receives stop."""

    schema_version = "qwen_target_stop_verification_v1"

    def __init__(self, compact_client: CompactQwenNavVLMClient, nav_modules: Any):
        self.compact_client = compact_client
        self.base_client = compact_client.base_client
        self._nav_modules = nav_modules

    @staticmethod
    def _valid_image_paths(image_paths: Sequence[str]) -> List[str]:
        return [
            str(path)
            for path in list(image_paths)[:6]
            if path and Path(path).exists()
        ]

    def _verify_identity_critic(
        self,
        *,
        target: str,
        image_path: str,
        target_bbox_norm: Sequence[float],
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            max_tokens = max(
                128,
                int(os.environ.get("VOCA_QWEN_STOP_CRITIC_MAX_TOKENS", "1024")),
            )
        except ValueError:
            max_tokens = 1024
        bbox = [round(float(value), 4) for value in target_bbox_norm]
        image_data_urls = [self._nav_modules.utils.image_to_data_url(
            image_path,
            max_side=self.base_client.image_max_side,
            jpeg_quality=self.base_client.jpeg_quality,
        )]
        crop_metadata: Dict[str, Any] = {
            "used": False,
            "source_image": str(image_path),
            "input_mode": "full_scene_fallback",
            "image_count": 1,
        }
        try:
            image = self._nav_modules.utils.Image.open(image_path).convert("RGB")
            width, height = image.size
            x1 = max(0, min(width - 1, int(math.floor(bbox[0] * width))))
            y1 = max(0, min(height - 1, int(math.floor(bbox[1] * height))))
            x2 = max(x1 + 1, min(width, int(math.ceil(bbox[2] * width))))
            y2 = max(y1 + 1, min(height, int(math.ceil(bbox[3] * height))))
            try:
                padding_ratio = float(
                    os.environ.get("VOCA_QWEN_STOP_CRITIC_CROP_PADDING", "0.45")
                )
            except ValueError:
                padding_ratio = 0.45
            padding_ratio = max(0.20, min(0.75, padding_ratio))
            pad_x = max(8, int(round((x2 - x1) * padding_ratio)))
            pad_y = max(8, int(round((y2 - y1) * padding_ratio)))
            crop_box = (
                max(0, x1 - pad_x),
                max(0, y1 - pad_y),
                min(width, x2 + pad_x),
                min(height, y2 + pad_y),
            )
            crop = image.crop(crop_box)
            marked = image.copy()
            ImageDraw.Draw(marked).rectangle(
                (x1, y1, x2, y2),
                outline=(255, 0, 0),
                width=max(3, int(round(min(width, height) / 100.0))),
            )

            def _image_data_url(source: Any) -> str:
                buffer = io.BytesIO()
                source.save(
                    buffer,
                    format="JPEG",
                    quality=max(40, min(95, int(self.base_client.jpeg_quality))),
                )
                return "data:image/jpeg;base64,{}".format(
                    base64.b64encode(buffer.getvalue()).decode("ascii")
                )

            image_data_urls = [_image_data_url(marked), _image_data_url(crop)]
            crop_metadata = {
                "used": True,
                "source_image": str(image_path),
                "source_size": [int(width), int(height)],
                "target_box_px": [int(x1), int(y1), int(x2), int(y2)],
                "crop_box_px": [int(value) for value in crop_box],
                "crop_size": [int(crop.size[0]), int(crop.size[1])],
                "padding_ratio": round(float(padding_ratio), 3),
                "input_mode": "marked_full_scene_plus_expanded_crop",
                "image_count": 2,
            }
        except Exception as exc:
            crop_metadata["error"] = "{}: {}".format(
                type(exc).__name__,
                str(exc),
            )[:200]
        result: Dict[str, Any] = {
            "schema_version": "qwen_target_identity_critic_v1",
            "triggered": True,
            "passed": False,
            "exact_target": False,
            "lookalike_detected": True,
            "lookalike_type": "uncertain",
            "confidence": "low",
            "reason": "identity critic did not verify the exact target",
            "target_bbox_norm": bbox,
            "identity_crop": crop_metadata,
            "backend_call_count": 1,
            "backend_error_count": 0,
        }
        prompt = (
            "/no_think\nObjectNav identity audit for target {target}. Image 1 is the full scene; the "
            "red box is an approximate detector box. Image 2 is a larger crop around it. Detector "
            "boxes may cut off the base or part of the object. Inspect the entire physical object that "
            "overlaps or immediately continues outside the red box, and do not classify surrounding "
            "wallpaper, curtains, artwork, or furniture as the object. Exact-category criteria: "
            "{criteria} Return JSON only with exact_target, lookalike_detected, lookalike_type, "
            "confidence, reason."
        ).format(
            bbox=bbox,
            target=_target_prompt_label(target).upper(),
            criteria=_target_identity_criteria(target),
        )
        payload = {
            "model": self.base_client.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise visual identity auditor. Ground the approximate red box "
                        "in the full scene and crop."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *[
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_url},
                            }
                            for image_data_url in image_data_urls
                        ],
                        {
                            "type": "text",
                            "text": "FINAL JSON ONLY.",
                        },
                    ],
                },
            ],
            "temperature": self.base_client.temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "target_identity_critic",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "exact_target": {"type": "boolean"},
                            "lookalike_detected": {"type": "boolean"},
                            "lookalike_type": {"type": "string"},
                            "confidence": {"enum": ["high", "medium", "low"]},
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "exact_target",
                            "lookalike_detected",
                            "lookalike_type",
                            "confidence",
                            "reason",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if "thinking" in str(self.base_client.model or "").lower():
            try:
                thinking_token_budget = int(
                    os.environ.get(
                        "VOCA_QWEN_STOP_CRITIC_THINKING_TOKEN_BUDGET",
                        "256",
                    )
                )
            except ValueError:
                thinking_token_budget = 256
            if thinking_token_budget > 0:
                payload["thinking_token_budget"] = int(thinking_token_budget)
        payload.pop("guided_json", None)
        try:
            data = self.compact_client._post_chat_completion(payload)
            choices = data.get("choices") if isinstance(data, dict) else []
            first = choices[0] if isinstance(choices, list) and choices else {}
            finish_reason = first.get("finish_reason") if isinstance(first, dict) else None
            parsed = None
            for text in self.compact_client._response_text_candidates(data):
                strict_text = str(text or "").strip()
                if strict_text.startswith("{") and strict_text.endswith("}"):
                    parsed = self._nav_modules.utils.extract_json_object(strict_text)
                    break
            if not isinstance(parsed, dict):
                raise ValueError(
                    "identity critic returned no strict JSON object; finish_reason={}".format(
                        finish_reason
                    )
                )
            exact_target = parsed.get("exact_target") is True
            lookalike = parsed.get("lookalike_detected") is True
            confidence = str(parsed.get("confidence") or "low").strip().lower()
            result.update(
                exact_target=exact_target,
                lookalike_detected=lookalike,
                lookalike_type=str(parsed.get("lookalike_type") or "")[:120],
                confidence=confidence,
                reason=str(parsed.get("reason") or "")[:240],
                passed=bool(
                    exact_target
                    and not lookalike
                    and confidence in {"high", "medium"}
                ),
                finish_reason=finish_reason,
            )
        except Exception as exc:
            result["error"] = "{}: {}".format(type(exc).__name__, str(exc))[:240]
            result["backend_error_count"] = 1
            result["reason"] = "identity_critic_error_fail_closed"
        result["latency_sec"] = round(float(time.perf_counter() - started), 6)
        return result

    def _verify_multiple_views(
        self,
        *,
        target: str,
        image_paths: Sequence[str],
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        view_results: List[Dict[str, Any]] = []
        selected: Optional[Dict[str, Any]] = None
        confidence_rank = {"low": 0, "medium": 1, "high": 2}
        for view_index, path in enumerate(image_paths):
            item = dict(self.verify(target=target, image_paths=[path]))
            item["source_view_index"] = int(view_index)
            view_results.append(item)
            if item.get("passed"):
                selected = item
                break
            if selected is None or (
                not bool(item.get("error")),
                bool(item.get("target_match")),
                bool(item.get("target_visible")),
                bool(item.get("close_enough")),
                confidence_rank.get(str(item.get("confidence") or "low"), 0),
            ) > (
                not bool(selected.get("error")),
                bool(selected.get("target_match")),
                bool(selected.get("target_visible")),
                bool(selected.get("close_enough")),
                confidence_rank.get(str(selected.get("confidence") or "low"), 0),
            ):
                selected = item

        selected = dict(selected or {})
        selected["schema_version"] = self.schema_version
        selected["triggered"] = True
        selected["target"] = str(target)
        selected["verification_input_mode"] = "sequential_single_view"
        selected["verification_view_count"] = len(image_paths)
        selected["backend_call_count"] = sum(
            max(1, int(item.get("backend_call_count", 1) or 1)) for item in view_results
        )
        selected["backend_error_count"] = sum(bool(item.get("error")) for item in view_results)
        if selected.get("target_visible") or selected.get("target_match"):
            selected["target_view_index"] = int(selected.get("source_view_index", 0))
        else:
            selected["target_view_index"] = None
        selected.pop("source_view_index", None)
        selected["per_view_results"] = [
            {
                "view_index": int(item.get("source_view_index", 0)),
                "passed": bool(item.get("passed")),
                "target_visible": bool(item.get("target_visible")),
                "target_match": bool(item.get("target_match")),
                "close_enough": bool(item.get("close_enough")),
                "confidence": str(item.get("confidence") or "low"),
                "reason": str(item.get("reason") or "")[:120],
                "error": str(item.get("error") or "")[:120],
            }
            for item in view_results
        ]
        if selected.get("backend_error_count") and selected.get("backend_error_count") == len(view_results):
            selected["error"] = "all_single_view_stop_verifiers_failed"
            selected["passed"] = False
        elif selected.get("error"):
            selected.pop("error", None)
        selected["latency_sec"] = round(float(time.perf_counter() - started), 6)
        return selected

    def verify(self, *, target: str, image_paths: Sequence[str]) -> Dict[str, Any]:
        started = time.perf_counter()
        valid_image_paths = self._valid_image_paths(image_paths)
        if len(valid_image_paths) > 1:
            return self._verify_multiple_views(
                target=target,
                image_paths=valid_image_paths,
            )
        result: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "triggered": True,
            "passed": False,
            "target": str(target),
            "target_visible": False,
            "target_match": False,
            "close_enough": False,
            "confidence": "low",
            "reason": "target completion was not verified",
            "verification_input_mode": "single_view",
            "verification_view_count": len(valid_image_paths),
            "backend_call_count": 1,
            "backend_error_count": 0,
        }
        try:
            max_tokens = int(os.environ.get("VOCA_QWEN_STOP_VERIFY_MAX_TOKENS", "3072"))
        except ValueError:
            max_tokens = 3072
        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "/no_think\nConservative ObjNav completion check. Target category: {target}. "
                    "Approve stop only if the exact target category is clearly visible, is not a "
                    "lookalike, and appears within roughly one meter for navigation completion. "
                    "A target that is small, distant, or only at the image edge is not close enough. "
                    "Partial, tiny, reflected, ambiguous, or distant evidence must fail. Return JSON "
                    "only with target_visible, target_match, close_enough (booleans), confidence "
                    "(high|medium|low), target_view_index (zero-based integer or null), "
                    "target_center_x (0.0 left to 1.0 right, or null), target_bbox_norm "
                    "([x1,y1,x2,y2] normalized to 0..1 for the exact target, or null), and reason "
                    "(at most 20 words). Judge image area relative to the physical category: a "
                    "nearby houseplant can remain visually small. The target center must lie in "
                    "the central half of the view. Exact-category criteria: {criteria}"
                ).format(
                    target=_target_prompt_label(target),
                    criteria=_target_identity_criteria(target),
                ),
            }
        ]
        if valid_image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._nav_modules.utils.image_to_data_url(
                            valid_image_paths[0],
                            max_side=self.base_client.image_max_side,
                            jpeg_quality=self.base_client.jpeg_quality,
                        )
                    },
                }
            )
        if not valid_image_paths:
            result["reason"] = "stop_verifier_no_images"
            result["latency_sec"] = round(float(time.perf_counter() - started), 6)
            return result
        content.append(
            {
                "type": "text",
                "text": (
                    "FINAL ANSWER JSON ONLY. Do not describe or analyze the image. Your first "
                    "character must be { and your last character must be }. Include exactly the "
                    "eight required schema fields with evidence-based values."
                ),
            }
        )
        payload = {
            "model": self.base_client.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You verify robot task completion conservatively. Return one JSON object only.",
                },
                {"role": "user", "content": content},
            ],
            "temperature": self.base_client.temperature,
            "max_tokens": max(64, max_tokens),
            "response_format": {"type": "json_object"},
        }
        payload.update(getattr(self.base_client, "extra_payload", {}) or {})
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "target_stop_verdict",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "target_visible": {"type": "boolean"},
                        "target_match": {"type": "boolean"},
                        "close_enough": {"type": "boolean"},
                        "confidence": {"enum": ["high", "medium", "low"]},
                        "target_view_index": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}]
                        },
                        "target_center_x": {
                            "anyOf": [{"type": "number"}, {"type": "null"}]
                        },
                        "target_bbox_norm": {
                            "anyOf": [
                                {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 4,
                                    "maxItems": 4,
                                },
                                {"type": "null"},
                            ]
                        },
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "target_visible",
                        "target_match",
                        "close_enough",
                        "confidence",
                        "target_view_index",
                        "target_center_x",
                        "target_bbox_norm",
                        "reason",
                    ],
                    "additionalProperties": False,
                },
            },
        }
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload.pop("guided_json", None)
        payload.pop("thinking_token_budget", None)
        try:
            data = self.compact_client._post_chat_completion(payload)
            choices = data.get("choices") if isinstance(data, dict) else []
            first = choices[0] if isinstance(choices, list) and choices else {}
            finish_reason = first.get("finish_reason") if isinstance(first, dict) else None
            if finish_reason != "stop":
                raise ValueError("stop verifier response did not finish: {}".format(finish_reason))
            parsed = None
            parse_error = None
            for text in self.compact_client._response_text_candidates(data):
                strict_text = str(text or "").strip()
                if not (strict_text.startswith("{") and strict_text.endswith("}")):
                    continue
                try:
                    parsed = self._nav_modules.utils.extract_json_object(strict_text)
                    break
                except Exception as exc:
                    parse_error = exc
            if parsed is None:
                if parse_error is not None:
                    raise parse_error
                raise ValueError("no JSON object found in stop verifier response")
            confidence = str(parsed.get("confidence") or "low").strip().lower()
            visible = parsed.get("target_visible") is True
            matched = parsed.get("target_match") is True
            close = parsed.get("close_enough") is True
            try:
                target_view_index = int(parsed.get("target_view_index"))
            except (TypeError, ValueError):
                target_view_index = None
            try:
                target_center_x = max(0.0, min(1.0, float(parsed.get("target_center_x"))))
            except (TypeError, ValueError):
                target_center_x = None
            target_bbox_norm = None
            raw_bbox = parsed.get("target_bbox_norm")
            if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
                try:
                    bbox = [max(0.0, min(1.0, float(value))) for value in raw_bbox]
                    if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                        target_bbox_norm = bbox
                except (TypeError, ValueError):
                    target_bbox_norm = None
            try:
                center_min = max(
                    0.0,
                    min(0.49, float(os.environ.get("VOCA_STOP_TARGET_CENTER_MIN", "0.25"))),
                )
            except ValueError:
                center_min = 0.25
            try:
                center_max = min(
                    1.0,
                    max(0.51, float(os.environ.get("VOCA_STOP_TARGET_CENTER_MAX", "0.75"))),
                )
            except ValueError:
                center_max = 0.75
            try:
                min_bbox_height = max(
                    0.0,
                    min(1.0, float(os.environ.get("VOCA_STOP_MIN_TARGET_HEIGHT_FRAC", "0.18"))),
                )
            except ValueError:
                min_bbox_height = 0.18
            try:
                min_bbox_area = max(
                    0.0,
                    min(1.0, float(os.environ.get("VOCA_STOP_MIN_TARGET_AREA_FRAC", "0.01"))),
                )
            except ValueError:
                min_bbox_area = 0.01
            normalized_target = str(target or "").strip().lower()
            threshold_profile = "default"
            if normalized_target in {"plant", "houseplant"}:
                threshold_profile = "small_physical_target"
                try:
                    min_bbox_height = max(
                        0.0,
                        min(
                            min_bbox_height,
                            float(
                                os.environ.get(
                                    "VOCA_STOP_SMALL_TARGET_MIN_HEIGHT_FRAC",
                                    "0.04",
                                )
                            ),
                        ),
                    )
                except ValueError:
                    min_bbox_height = min(min_bbox_height, 0.04)
                try:
                    min_bbox_area = max(
                        0.0,
                        min(
                            min_bbox_area,
                            float(
                                os.environ.get(
                                    "VOCA_STOP_SMALL_TARGET_MIN_AREA_FRAC",
                                    "0.001",
                                )
                            ),
                        ),
                    )
                except ValueError:
                    min_bbox_area = min(min_bbox_area, 0.001)
            bbox_width = (
                float(target_bbox_norm[2] - target_bbox_norm[0])
                if target_bbox_norm is not None
                else 0.0
            )
            bbox_height = (
                float(target_bbox_norm[3] - target_bbox_norm[1])
                if target_bbox_norm is not None
                else 0.0
            )
            bbox_area = bbox_width * bbox_height
            center_satisfied = bool(
                target_center_x is not None and center_min <= target_center_x <= center_max
            )
            bbox_size_satisfied = bool(
                target_bbox_norm is not None
                and bbox_height + 1e-6 >= min_bbox_height
                and bbox_area + 1e-6 >= min_bbox_area
            )
            target_evidence_passed = bool(
                visible and matched and confidence in {"high", "medium"}
            )
            visual_semantics_passed = bool(target_evidence_passed and close)
            passed = bool(visual_semantics_passed and center_satisfied and bbox_size_satisfied)
            model_reason = str(parsed.get("reason") or "stop verifier returned no reason")[:240]
            guard_reason = model_reason
            if visual_semantics_passed and not center_satisfied:
                guard_reason = "target evidence outside central stop band"
            elif visual_semantics_passed and target_bbox_norm is None:
                guard_reason = "target grounding bbox missing or invalid"
            elif visual_semantics_passed and not bbox_size_satisfied:
                guard_reason = "target grounding bbox too small for conservative stop"
            identity_critic: Dict[str, Any] = {
                "schema_version": "qwen_target_identity_critic_v1",
                "triggered": False,
                "passed": True,
                "reason": "identity_critic_not_required",
                "backend_call_count": 0,
                "backend_error_count": 0,
            }
            if (
                target_evidence_passed
                and target_bbox_norm is not None
                and _env_bool("VOCA_QWEN_STOP_IDENTITY_CRITIC", True)
            ):
                identity_critic = self._verify_identity_critic(
                    target=target,
                    image_path=valid_image_paths[0],
                    target_bbox_norm=target_bbox_norm,
                )
                if not identity_critic.get("passed"):
                    passed = False
                    guard_reason = "identity critic rejected target: {}".format(
                        identity_critic.get("reason")
                        or identity_critic.get("lookalike_type")
                        or "ambiguous exact category"
                    )[:240]
            verified_target_evidence_passed = bool(
                target_evidence_passed
                and (
                    not identity_critic.get("triggered")
                    or identity_critic.get("passed")
                )
            )
            result.update(
                target_visible=visible,
                target_match=matched,
                close_enough=close,
                confidence=confidence,
                target_view_index=target_view_index,
                target_center_x=target_center_x,
                target_bbox_norm=target_bbox_norm,
                target_bbox_width=round(bbox_width, 4),
                target_bbox_height=round(bbox_height, 4),
                target_bbox_area=round(bbox_area, 4),
                target_center_guard_satisfied=center_satisfied,
                target_bbox_guard_satisfied=bbox_size_satisfied,
                target_center_range=[center_min, center_max],
                target_min_bbox_height=min_bbox_height,
                target_min_bbox_area=min_bbox_area,
                target_bbox_threshold_profile=threshold_profile,
                target_evidence_passed=verified_target_evidence_passed,
                identity_critic=identity_critic,
                identity_critic_passed=bool(identity_critic.get("passed")),
                backend_call_count=(
                    1 + int(identity_critic.get("backend_call_count", 0) or 0)
                ),
                backend_error_count=int(
                    identity_critic.get("backend_error_count", 0) or 0
                ),
                passed=passed,
                reason=guard_reason,
                model_reason=model_reason,
                finish_reason=first.get("finish_reason") if isinstance(first, dict) else None,
            )
        except Exception as exc:
            result["reason"] = "stop_verifier_error_fail_closed"
            result["error"] = "{}: {}".format(type(exc).__name__, str(exc))[:240]
            result["backend_error_count"] = 1
        result["latency_sec"] = round(float(time.perf_counter() - started), 6)
        return result


class QwenRevisitVerifier:
    """Resolve a backend-proposed revisit with a dedicated visual comparison call."""

    schema_version = "qwen_revisit_verification_v1"

    def __init__(self, compact_client: CompactQwenNavVLMClient, nav_modules: Any):
        self.compact_client = compact_client
        self.base_client = compact_client.base_client
        self._nav_modules = nav_modules

    @staticmethod
    def _candidate(vlm_input: Dict[str, Any], candidate_ref: str) -> Dict[str, Any]:
        memory = vlm_input.get("memory") if isinstance(vlm_input.get("memory"), dict) else {}
        recognition = (
            memory.get("place_recognition")
            if isinstance(memory.get("place_recognition"), dict)
            else {}
        )
        for item in recognition.get("revisit_candidates", []) or []:
            if isinstance(item, dict) and str(item.get("candidate_ref") or "") == candidate_ref:
                return item
        return {}

    def verify(self, *, vlm_input: Dict[str, Any], candidate_ref: str) -> Dict[str, Any]:
        started = time.perf_counter()
        candidate = self._candidate(vlm_input, candidate_ref)
        spatial = (
            candidate.get("spatial_plausibility")
            if isinstance(candidate.get("spatial_plausibility"), dict)
            else {}
        )
        result: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "triggered": True,
            "valid": False,
            "passed": False,
            "candidate_ref": str(candidate_ref),
            "verdict": "uncertain",
            "confidence": "low",
            "reason": "revisit verifier did not return a valid verdict",
        }
        operation = {
            "op": "defer_revisit_candidate",
            "candidate_ref": str(candidate_ref),
            "confidence": 0.0,
            "reason": "dedicated revisit verifier unavailable or uncertain",
            "source": "dedicated_revisit_verifier",
        }
        metadata = vlm_input.get("metadata") if isinstance(vlm_input.get("metadata"), dict) else {}
        current_paths = [
            str(path)
            for path in metadata.get("raw_observation_images", []) or []
            if path and Path(path).exists()
        ]
        candidate_path = str(candidate.get("candidate_image_ref") or "")
        if not candidate or not current_paths or not candidate_path or not Path(candidate_path).exists():
            result["reason"] = "revisit_verifier_missing_current_or_memory_image"
            result["memory_op"] = operation
            result["latency_sec"] = round(float(time.perf_counter() - started), 6)
            return result

        candidate_digest = hashlib.sha256(Path(candidate_path).read_bytes()).hexdigest()
        exact_digest_match = any(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() == candidate_digest
            for path in current_paths
        )
        if exact_digest_match and bool(spatial.get("accepted")):
            operation = {
                "op": "confirm_revisit_node",
                "candidate_ref": str(candidate_ref),
                "confidence": 1.0,
                "reason": "byte-identical visual observation and accepted spatial gate",
                "source": "exact_image_digest_guard",
            }
            result.update(
                valid=True,
                passed=True,
                verdict="same_place",
                confidence="high",
                reason=operation["reason"],
                spatial_plausibility_accepted=True,
                exact_image_digest_match=True,
                memory_op=operation,
                latency_sec=round(float(time.perf_counter() - started), 6),
            )
            return result

        prompt = (
            "/no_think\nCompare CURRENT PLACE with MEMORY PLACE for robot place recognition. "
            "Judge physical place identity from room layout, wall/opening geometry, doors, fixed "
            "structures, and their relative arrangement. Do not confirm merely because furniture, "
            "texture, or room category looks similar. Prefer uncertain over a false merge. "
            "Spatial gate accepted={accepted}; reason={spatial_reason}; visual retrieval score={score}. "
            "Return JSON only: verdict (same_place|different_place|uncertain), confidence "
            "(high|medium|low), reason (at most 24 words)."
        ).format(
            accepted=bool(spatial.get("accepted")),
            spatial_reason=str(spatial.get("reason") or "unavailable")[:120],
            score=candidate.get("visual_retrieval_score"),
        )
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in current_paths[:3]:
            content.append({"type": "text", "text": "CURRENT PLACE image"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._nav_modules.utils.image_to_data_url(
                            path,
                            max_side=self.base_client.image_max_side,
                            jpeg_quality=self.base_client.jpeg_quality,
                        )
                    },
                }
            )
        content.append({"type": "text", "text": "MEMORY PLACE image"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": self._nav_modules.utils.image_to_data_url(
                        candidate_path,
                        max_side=self.base_client.image_max_side,
                        jpeg_quality=self.base_client.jpeg_quality,
                    )
                },
            }
        )
        try:
            max_tokens = max(64, int(os.environ.get("VOCA_QWEN_REVISIT_VERIFY_MAX_TOKENS", "4096")))
        except ValueError:
            max_tokens = 4096
        payload = {
            "model": self.base_client.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You conservatively verify robot place identity. Return one JSON object only.",
                },
                {"role": "user", "content": content},
            ],
            "temperature": self.base_client.temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        payload.update(getattr(self.base_client, "extra_payload", {}) or {})
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "revisit_verdict",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "verdict": {
                            "enum": ["same_place", "different_place", "uncertain"]
                        },
                        "confidence": {"enum": ["high", "medium", "low"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["verdict", "confidence", "reason"],
                },
            },
        }
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload.pop("guided_json", None)
        payload.pop("thinking_token_budget", None)
        try:
            data = self.compact_client._post_chat_completion(payload)
            choices = data.get("choices") if isinstance(data, dict) else []
            first = choices[0] if isinstance(choices, list) and choices else {}
            finish_reason = first.get("finish_reason") if isinstance(first, dict) else None
            if finish_reason == "length":
                operation = {
                    "op": "defer_revisit_candidate",
                    "candidate_ref": str(candidate_ref),
                    "confidence": 0.0,
                    "reason": "revisit verifier output truncated; conservative defer",
                    "source": "dedicated_revisit_verifier",
                }
                result.update(
                    valid=True,
                    passed=True,
                    verdict="uncertain",
                    confidence="low",
                    reason="revisit verifier output truncated; conservative defer",
                    spatial_plausibility_accepted=bool(spatial.get("accepted")),
                    finish_reason="length",
                    degraded=True,
                    warning="revisit_response_truncated_deferred",
                )
            elif finish_reason != "stop":
                raise ValueError(
                    "revisit verifier response did not finish: {}".format(
                        finish_reason
                    )
                )
            else:
                parsed = None
                parse_error = None
                for text in self.compact_client._response_text_candidates(data):
                    try:
                        parsed = self._nav_modules.utils.extract_json_object(text or "")
                        break
                    except Exception as exc:
                        parse_error = exc
                if not isinstance(parsed, dict):
                    raise ValueError(
                        "no JSON revisit verdict in model response{}".format(
                            ": {}".format(parse_error) if parse_error else ""
                        )
                    )
                verdict = str(parsed.get("verdict") or "uncertain").strip().lower()
                confidence = str(parsed.get("confidence") or "low").strip().lower()
                reason = str(parsed.get("reason") or "revisit verifier returned no reason")[:240]
                valid = verdict in {"same_place", "different_place", "uncertain"} and confidence in {
                    "high",
                    "medium",
                    "low",
                }
                accepted = bool(spatial.get("accepted"))
                numeric_confidence = {"high": 0.9, "medium": 0.7, "low": 0.4}.get(confidence, 0.0)
                if valid and verdict == "same_place" and accepted and confidence in {"high", "medium"}:
                    op = "confirm_revisit_node"
                elif valid and verdict == "different_place":
                    op = "reject_revisit_candidate"
                else:
                    op = "defer_revisit_candidate"
                operation = {
                    "op": op,
                    "candidate_ref": str(candidate_ref),
                    "confidence": numeric_confidence,
                    "reason": reason,
                    "source": "dedicated_revisit_verifier",
                }
                result.update(
                    valid=valid,
                    passed=valid,
                    verdict=verdict,
                    confidence=confidence,
                    reason=reason,
                    spatial_plausibility_accepted=accepted,
                    finish_reason=finish_reason,
                )
        except Exception as exc:
            result["reason"] = "revisit_verifier_error_fail_closed"
            result["error"] = "{}: {}".format(type(exc).__name__, str(exc))[:240]
        result["memory_op"] = operation
        result["latency_sec"] = round(float(time.perf_counter() - started), 6)
        return result


class QwenVLMPlanner(QwenPointPlanner):
    """Qwen planner that calls the voca-s2e nav_vlm_waypoint_v1 client directly."""

    planner_name = "qwen_vlm"

    def __init__(
        self,
        *,
        vlm_client: Optional[Any] = None,
        point_verifier: Optional[Any] = None,
        stop_verifier: Optional[Any] = None,
        revisit_verifier: Optional[Any] = None,
        stop_confirmations_required: Optional[int] = None,
        text_fn: Optional[Any] = None,
        config: Optional[QwenPointPlannerConfig] = None,
    ):
        if config is None:
            try:
                retries = int(os.environ.get("VOCA_QWEN_VLM_RETRIES", "1"))
            except ValueError:
                retries = 1
            config = QwenPointPlannerConfig(retries=max(1, retries))
        super().__init__(text_fn=text_fn, config=config)
        self._nav_modules = import_nav_memory_qwen()
        auto_point_verifier = vlm_client is None
        if auto_point_verifier:
            base_client = self._nav_modules.vlm_client.OpenAICompatibleVLMClient.from_env()
            _configure_compact_image_options(base_client)
            self.vlm_client = CompactQwenNavVLMClient(base_client, self._nav_modules)
        else:
            self.vlm_client = vlm_client
            _configure_direct_json_payload(self.vlm_client)
        self.point_verifier = point_verifier
        if (
            self.point_verifier is None
            and auto_point_verifier
            and _env_bool("VOCA_QWEN_POINT_VERIFY", True)
        ):
            self.point_verifier = QwenSelectedPointVerifier(
                self.vlm_client,
                self._nav_modules,
            )
        self.stop_verifier = stop_verifier
        if (
            self.stop_verifier is None
            and auto_point_verifier
            and _env_bool("VOCA_QWEN_STOP_VERIFY", True)
        ):
            self.stop_verifier = QwenTargetStopVerifier(
                self.vlm_client,
                self._nav_modules,
            )
        self.revisit_verifier = revisit_verifier
        if (
            self.revisit_verifier is None
            and auto_point_verifier
            and _env_bool("VOCA_QWEN_REVISIT_VERIFY", True)
        ):
            self.revisit_verifier = QwenRevisitVerifier(
                self.vlm_client,
                self._nav_modules,
            )
        if stop_confirmations_required is None:
            try:
                stop_confirmations_required = int(
                    os.environ.get("VOCA_STOP_CONFIRMATIONS_REQUIRED", "2")
                )
            except ValueError:
                stop_confirmations_required = 2
        self.stop_confirmations_required = max(1, int(stop_confirmations_required))
        self.consecutive_stop_confirmations = 0
        self._last_stop_confirmation_position: List[float] = []
        self._last_stop_confirmation_frame_index: Optional[int] = None
        self._last_stop_confirmation_bbox_height = 0.0
        self._last_stop_confirmation_bbox_area = 0.0
        self._near_goal_visual_latch: Dict[str, Any] = {}
        self._last_target_approach_execution: Dict[str, Any] = {}
        self._target_approach_failure_streak = 0
        self._target_direct_approach_cooldown = 0
        self._target_terminal_refinement_count = 0
        self._proactive_stop_pending = False
        self._last_proactive_stop_probe: Optional[Dict[str, Any]] = None
        self._target_lock_remaining_cycles = 0
        self._target_lock_reacquisition_attempts = 0
        self._target_anchor_reacquisition_count = 0
        self._target_lock_session_cycles = 0
        self._target_lock_exhaustion_count = 0
        self._target_lock_cooldown_cycles = 0
        self._identity_rejection_cooldown_cycles = 0
        self._identity_rejection_cooldown_last_frame: Optional[int] = None
        self._identity_rejection_cooldown_skip_count = 0
        self._last_identity_rejection: Dict[str, Any] = {}
        self._image_cache_dir = Path(os.environ.get("VOCA_QWEN_VLM_IMAGE_DIR", "/tmp/voca_qwen_vlm_images"))
        self._last_vlm_input: Optional[Dict[str, Any]] = None
        self._runtime_feedback: List[Dict[str, Any]] = []
        self._strategic_state_history: List[Dict[str, Any]] = []
        self._last_strategic_state: Dict[str, Any] = {}
        self._same_room_cycles = 0
        self._last_strategic_position_xyz: List[float] = []
        self._generic_floor_go_streak = 0
        self._recent_selected_exit_refs: List[str] = []
        self._strategic_force_leave_override_count = 0
        self._forced_exit_failure_streak = 0
        self._forced_exit_failure_counts: Dict[str, int] = {}
        self._forced_exit_full_sweep_count = 0
        self._forced_exit_last_full_sweep_failure_streak = 0
        self._forced_exit_memory_backtrack_count = 0
        self._forced_exit_novel_sector_count = 0
        self._normal_goal_seek_backtrack_rejection_count = 0
        self._strategic_exit_alias_repair_count = 0
        self._strategic_visual_exit_ref_repair_count = 0
        self._coarse_direction_guard_count = 0
        self._coarse_direction_rejection_count = 0
        self._coarse_direction_detour_exemption_count = 0
        self._strategic_intent_counts: Dict[str, int] = {}
        self._runtime_position_xyz: List[float] = []
        self._runtime_heading_rad = 0.0
        self._runtime_frame_index: Optional[int] = None
        self._runtime_audit_metrics: Dict[str, Any] = {}
        self._runtime_spatial_coarse_goal: Optional[Dict[str, Any]] = None
        self._pixel_candidate_scorer: Optional[Any] = None
        self.localization_contract = build_localization_contract(
            source=os.environ.get("VOCA_LOCALIZATION_SOURCE", "none"),
            allow_sim_pose=_env_bool("VOCA_ALLOW_SIM_POSE_POLICY", False),
        )
        self.memory_sidecar: Optional[VOCAMemorySidecar] = None
        if _env_bool("VOCA_QWEN_MEMORY_SIDECAR", True):
            self.memory_sidecar = VOCAMemorySidecar()
        self.vlm_json_fallback_count = 0
        self.vlm_json_last_error = ""
        self.point_verification_calls = 0
        self.point_verification_pass_count = 0
        self.point_verification_rejection_count = 0
        self.point_verification_error_count = 0
        self.point_verification_durations: List[float] = []
        self.stop_verification_calls = 0
        self.stop_verification_pass_count = 0
        self.stop_verification_rejection_count = 0
        self.stop_verification_error_count = 0
        self.stop_verification_durations: List[float] = []
        self.revisit_verification_calls = 0
        self.revisit_verification_resolved_count = 0
        self.revisit_verification_error_count = 0
        self.revisit_verification_durations: List[float] = []

    def reset(self, object_goal):
        super().reset(object_goal)
        self.clear_navigation_feedback()
        self._runtime_position_xyz = []
        self._runtime_heading_rad = 0.0
        self._runtime_frame_index = None
        self._runtime_audit_metrics = {}
        self._runtime_spatial_coarse_goal = None
        self._strategic_state_history = []
        self._last_strategic_state = {}
        self._same_room_cycles = 0
        self._last_strategic_position_xyz = []
        self._generic_floor_go_streak = 0
        self._recent_selected_exit_refs = []
        self._strategic_force_leave_override_count = 0
        self._forced_exit_failure_streak = 0
        self._forced_exit_failure_counts = {}
        self._forced_exit_full_sweep_count = 0
        self._forced_exit_last_full_sweep_failure_streak = 0
        self._forced_exit_memory_backtrack_count = 0
        self._forced_exit_novel_sector_count = 0
        self._normal_goal_seek_backtrack_rejection_count = 0
        self._strategic_exit_alias_repair_count = 0
        self._strategic_visual_exit_ref_repair_count = 0
        self._coarse_direction_guard_count = 0
        self._coarse_direction_rejection_count = 0
        self._coarse_direction_detour_exemption_count = 0
        self._strategic_intent_counts = {}
        if self.memory_sidecar is not None:
            self.memory_sidecar.reset(self.object_goal)
        self.vlm_json_fallback_count = 0
        self.vlm_json_last_error = ""
        self.point_verification_calls = 0
        self.point_verification_pass_count = 0
        self.point_verification_rejection_count = 0
        self.point_verification_error_count = 0
        self.point_verification_durations = []
        self.stop_verification_calls = 0
        self.stop_verification_pass_count = 0
        self.stop_verification_rejection_count = 0
        self.stop_verification_error_count = 0
        self.stop_verification_durations = []
        self.revisit_verification_calls = 0
        self.revisit_verification_resolved_count = 0
        self.revisit_verification_error_count = 0
        self.revisit_verification_durations = []
        self.consecutive_stop_confirmations = 0
        self._last_stop_confirmation_position = []
        self._last_stop_confirmation_frame_index = None
        self._last_stop_confirmation_bbox_height = 0.0
        self._last_stop_confirmation_bbox_area = 0.0
        self._near_goal_visual_latch = {}
        self._last_target_approach_execution = {}
        self._target_approach_failure_streak = 0
        self._target_direct_approach_cooldown = 0
        self._target_terminal_refinement_count = 0
        self._proactive_stop_pending = False
        self._last_proactive_stop_probe = None
        self._target_lock_remaining_cycles = 0
        self._target_lock_reacquisition_attempts = 0
        self._target_anchor_reacquisition_count = 0
        self._target_lock_session_cycles = 0
        self._target_lock_exhaustion_count = 0
        self._target_lock_cooldown_cycles = 0
        self._identity_rejection_cooldown_cycles = 0
        self._identity_rejection_cooldown_last_frame = None
        self._identity_rejection_cooldown_skip_count = 0
        self._last_identity_rejection = {}

    def record_navigation_feedback(self, feedback: Dict[str, Any]) -> None:
        if not isinstance(feedback, dict):
            return
        try:
            max_items = int(os.environ.get("VOCA_QWEN_RUNTIME_FEEDBACK_MAX", "16"))
        except ValueError:
            max_items = 16
        self._runtime_feedback.append(dict(feedback))
        self._runtime_feedback = self._runtime_feedback[-max(1, max_items):]

    def clear_navigation_feedback(self) -> None:
        self._runtime_feedback = []

    def _recent_verified_target_memory(self) -> Dict[str, Any]:
        sidecar = self.memory_sidecar
        evidence = (
            getattr(sidecar, "last_verified_target_evidence", None)
            if sidecar is not None
            else None
        )
        if not isinstance(evidence, dict) or not evidence.get("updated"):
            return {}
        try:
            evidence_frame = int(evidence.get("frame_index"))
            current_frame = int(self._runtime_frame_index)
        except (TypeError, ValueError):
            return {}
        try:
            ttl_frames = max(
                1,
                int(
                    os.environ.get(
                        "VOCA_VERIFIED_TARGET_MEMORY_TTL_FRAMES",
                        "80",
                    )
                ),
            )
        except ValueError:
            ttl_frames = 80
        frame_gap = current_frame - evidence_frame
        if frame_gap < 0 or frame_gap > ttl_frames:
            return {}

        displacement_m = None
        evidence_position = evidence.get("position_xyz")
        if (
            isinstance(evidence_position, (list, tuple))
            and len(evidence_position) >= 3
            and len(self._runtime_position_xyz) >= 3
        ):
            try:
                displacement_m = float(
                    np.linalg.norm(
                        np.asarray(self._runtime_position_xyz[:3], dtype=np.float64)
                        - np.asarray(evidence_position[:3], dtype=np.float64)
                    )
                )
            except Exception:
                displacement_m = None
        try:
            radius_m = max(
                0.25,
                float(
                    os.environ.get(
                        "VOCA_VERIFIED_TARGET_MEMORY_RADIUS_M",
                        "1.5",
                    )
                ),
            )
        except ValueError:
            radius_m = 1.5
        if displacement_m is not None and displacement_m > radius_m:
            return {}
        return {
            **dict(evidence),
            "frame_gap": int(frame_gap),
            "displacement_m": (
                round(float(displacement_m), 6)
                if displacement_m is not None
                else None
            ),
            "ttl_frames": int(ttl_frames),
            "radius_m": float(radius_m),
        }

    def _cached_identity_rejection_guard(self) -> Optional[Dict[str, Any]]:
        if self._identity_rejection_cooldown_cycles <= 0:
            return None
        rejection_position = self._last_identity_rejection.get("position_xyz")
        if (
            isinstance(rejection_position, (list, tuple))
            and len(rejection_position) >= 3
            and len(self._runtime_position_xyz) >= 3
        ):
            try:
                distance_m = float(
                    np.linalg.norm(
                        np.asarray(self._runtime_position_xyz[:3], dtype=np.float64)
                        - np.asarray(rejection_position[:3], dtype=np.float64)
                    )
                )
            except Exception:
                distance_m = 0.0
            try:
                clear_distance_m = max(
                    0.25,
                    float(
                        os.environ.get(
                            "VOCA_TARGET_IDENTITY_REJECTION_CLEAR_M",
                            "1.5",
                        )
                    ),
                )
            except ValueError:
                clear_distance_m = 1.5
            if distance_m >= clear_distance_m:
                self._identity_rejection_cooldown_cycles = 0
                self._identity_rejection_cooldown_last_frame = None
                return None

        frame_index = (
            int(self._runtime_frame_index)
            if self._runtime_frame_index is not None
            else -1
        )
        if self._identity_rejection_cooldown_last_frame != frame_index:
            self._identity_rejection_cooldown_cycles = max(
                0,
                self._identity_rejection_cooldown_cycles - 1,
            )
            self._identity_rejection_cooldown_last_frame = frame_index
            self._identity_rejection_cooldown_skip_count += 1
        return {
            "schema_version": "qwen_target_stop_verification_v1",
            "enabled": self.stop_verifier is not None,
            "triggered": True,
            "passed": False,
            "single_frame_passed": False,
            "target_evidence_passed": False,
            "cached_identity_rejection": True,
            "identity_rejection_cooldown_remaining": int(
                self._identity_rejection_cooldown_cycles
            ),
            "identity_rejection": dict(self._last_identity_rejection),
            "identity_critic": {
                "schema_version": "qwen_target_identity_critic_v1",
                "triggered": False,
                "passed": False,
                "cached_rejection": True,
                "reason": "known lookalike cooldown; backend call skipped",
                "backend_call_count": 0,
                "backend_error_count": 0,
            },
            "backend_call_count": 0,
            "backend_error_count": 0,
            "latency_sec": 0.0,
            "reason": "known target lookalike nearby; continue exploration before rechecking stop",
        }

    @staticmethod
    def _positive_int_env(name: str, default: int) -> int:
        try:
            return max(1, int(os.environ.get(name, str(default))))
        except ValueError:
            return max(1, int(default))

    def _strategic_runtime_snapshot(self) -> Dict[str, Any]:
        last = dict(self._last_strategic_state)
        evidence = str(last.get("target_evidence") or "none")
        status = str(last.get("room_search_status") or "unsearched")
        supervisor_mode = (
            self.memory_sidecar.supervisor_mode
            if self.memory_sidecar is not None
            else "goal_seek"
        )
        generic_limit = self._positive_int_env(
            "VOCA_FORCE_LEAVE_GENERIC_GO_LIMIT",
            3,
        )
        same_room_limit = self._positive_int_env(
            "VOCA_FORCE_LEAVE_SAME_ROOM_CYCLES",
            5,
        )
        force_reasons: List[str] = []
        target_unconfirmed = evidence in {"none", "context_only", ""}
        if target_unconfirmed and supervisor_mode != "verify_target":
            if status == "exhausted":
                force_reasons.append("room_search_exhausted")
            if self._generic_floor_go_streak >= generic_limit:
                force_reasons.append("generic_floor_go_streak")
            if self._same_room_cycles >= same_room_limit:
                force_reasons.append("same_room_cycle_limit")
        return {
            "schema_version": "strategic_runtime_v1",
            "last_strategic_state": last,
            "same_room_cycles": int(self._same_room_cycles),
            "generic_floor_go_streak": int(self._generic_floor_go_streak),
            "force_leave_room": bool(force_reasons),
            "force_reason": ",".join(force_reasons),
            "recent_selected_exit_refs": list(self._recent_selected_exit_refs[-3:]),
            "force_leave_override_count": int(
                self._strategic_force_leave_override_count
            ),
            "forced_exit_failure_streak": int(self._forced_exit_failure_streak),
            "forced_exit_failure_counts": dict(self._forced_exit_failure_counts),
            "forced_exit_full_sweep_count": int(
                self._forced_exit_full_sweep_count
            ),
            "forced_exit_memory_backtrack_count": int(
                self._forced_exit_memory_backtrack_count
            ),
            "forced_exit_novel_sector_count": int(
                self._forced_exit_novel_sector_count
            ),
            "normal_goal_seek_backtrack_rejection_count": int(
                self._normal_goal_seek_backtrack_rejection_count
            ),
            "strategic_exit_alias_repair_count": int(
                self._strategic_exit_alias_repair_count
            ),
            "strategic_visual_exit_ref_repair_count": int(
                self._strategic_visual_exit_ref_repair_count
            ),
            "identity_rejection_cooldown_remaining": int(
                self._identity_rejection_cooldown_cycles
            ),
            "identity_rejection_cooldown_skips": int(
                self._identity_rejection_cooldown_skip_count
            ),
            "target_anchor_reacquisition_count": int(
                self._target_anchor_reacquisition_count
            ),
            "target_lock_session_cycles": int(
                self._target_lock_session_cycles
            ),
            "target_lock_exhaustion_count": int(
                self._target_lock_exhaustion_count
            ),
            "target_lock_cooldown_remaining": int(
                self._target_lock_cooldown_cycles
            ),
            "intent_counts": dict(self._strategic_intent_counts),
        }

    def _record_forced_exit_failure(self, reason: str) -> int:
        reason_key = str(reason or "unknown").strip() or "unknown"
        self._forced_exit_failure_streak += 1
        self._forced_exit_failure_counts[reason_key] = int(
            self._forced_exit_failure_counts.get(reason_key, 0)
        ) + 1
        return int(self._forced_exit_failure_streak)

    def _record_forced_exit_success(self) -> None:
        self._forced_exit_failure_streak = 0
        self._forced_exit_last_full_sweep_failure_streak = 0
        self._same_room_cycles = 0
        self._generic_floor_go_streak = 0
        self._last_strategic_position_xyz = []

    @staticmethod
    def _preserve_backend_recovery_metadata(
        canonical_output: Dict[str, Any],
        recovery_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = dict(canonical_output)
        for key in (
            "backend_memory_backtrack",
            "backend_novel_sector_escape",
            "strategic_state",
            "memory_ops",
        ):
            if key in recovery_output:
                merged[key] = recovery_output[key]
        return merged

    def record_strategic_go_execution(
        self,
        decision: Dict[str, Any],
        go_progress: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Commit forced-exit state only after the selected go action executes."""
        decision = decision if isinstance(decision, dict) else {}
        progress = go_progress if isinstance(go_progress, dict) else {}
        validation = (
            decision.get("strategic_validation")
            if isinstance(decision.get("strategic_validation"), dict)
            else {}
        )
        vlm_output = (
            decision.get("vlm_output")
            if isinstance(decision.get("vlm_output"), dict)
            else {}
        )
        strategic_state = (
            decision.get("strategic_state")
            if isinstance(decision.get("strategic_state"), dict)
            else {}
        )
        intent = str(strategic_state.get("navigation_intent") or "")
        recovery_kind = ""
        if isinstance(vlm_output.get("backend_memory_backtrack"), dict):
            recovery_kind = "memory_backtrack"
        elif isinstance(vlm_output.get("backend_novel_sector_escape"), dict):
            recovery_kind = "novel_sector_escape"
        elif validation.get("force_leave_room") and intent in _FORCE_LEAVE_COMPATIBLE_INTENTS:
            recovery_kind = "vlm_exit_intent"
        if not recovery_kind:
            return {
                "triggered": False,
                "reason": "go_not_part_of_forced_exit_recovery",
            }

        strategic_progress = bool(progress.get("strategic_progress"))
        if strategic_progress:
            previous_streak = int(self._forced_exit_failure_streak)
            self._record_forced_exit_success()
            return {
                "triggered": True,
                "passed": True,
                "recovery_kind": recovery_kind,
                "previous_failure_streak": previous_streak,
                "failure_streak": 0,
                "reason": "forced_exit_execution_made_strategic_progress",
            }
        failure_streak = self._record_forced_exit_failure(
            "execution_no_strategic_progress:{}".format(recovery_kind)
        )
        return {
            "triggered": True,
            "passed": False,
            "recovery_kind": recovery_kind,
            "failure_streak": int(failure_streak),
            "reason": "forced_exit_candidate_verified_but_execution_made_no_progress",
        }

    @staticmethod
    def _input_is_full_sweep(vlm_input: Dict[str, Any]) -> bool:
        observation = (
            vlm_input.get("observation")
            if isinstance(vlm_input.get("observation"), dict)
            else {}
        )
        views = observation.get("views") if isinstance(observation, dict) else []
        if not isinstance(views, list) or len(views) < 8:
            return False
        headings = set()
        for view in views:
            if not isinstance(view, dict):
                continue
            try:
                heading = int(round(float(view.get("relative_heading_deg", 0.0))))
            except (TypeError, ValueError):
                continue
            headings.add(((heading + 180) % 360) - 180)
        return len(headings) >= 8

    @staticmethod
    def _selected_pixel_candidate(
        vlm_input: Dict[str, Any],
        candidate_ref: str,
    ) -> Dict[str, Any]:
        pixel_context = (
            vlm_input.get("pixel_candidates")
            if isinstance(vlm_input.get("pixel_candidates"), dict)
            else {}
        )
        for candidate in pixel_context.get("candidates", []) or []:
            if (
                isinstance(candidate, dict)
                and str(candidate.get("candidate_ref") or "") == candidate_ref
            ):
                return dict(candidate)
        return {}

    def _forced_exit_recovery_output(
        self,
        vlm_input: Dict[str, Any],
        *,
        reason: str,
    ) -> Tuple[Dict[str, Any], str]:
        views = (
            vlm_input.get("observation", {}).get("views", [])
            if isinstance(vlm_input.get("observation"), dict)
            else []
        )
        threshold = self._positive_int_env(
            "VOCA_FORCE_LEAVE_FULL_SWEEP_AFTER",
            2,
        )
        failures_since_sweep = (
            self._forced_exit_failure_streak
            - self._forced_exit_last_full_sweep_failure_streak
        )
        request_full_sweep = bool(
            self._forced_exit_failure_streak >= threshold
            and failures_since_sweep >= threshold
            and not self._input_is_full_sweep(vlm_input)
        )
        if request_full_sweep:
            self._forced_exit_full_sweep_count += 1
            self._forced_exit_last_full_sweep_failure_streak = int(
                self._forced_exit_failure_streak
            )
            output = self._nav_modules.schema.make_observation_request_output(
                mode="full_sweep",
                center_yaw_deg=0.0,
                step_deg=45,
                num_views=8,
                yaw_offsets_deg=[-180, -135, -90, -45, 0, 45, 90, 135],
                reason=(
                    "forced room exit unresolved after repeated checks; inspect all "
                    "headings for a real doorway or corridor: {}"
                ).format(reason),
                confidence="low",
            )
            return output, "request_observation.full_sweep"

        if len(views) <= 1:
            output = self._nav_modules.schema.make_observation_request_output(
                mode="directed_sweep",
                center_yaw_deg=0.0,
                step_deg=30,
                num_views=5,
                yaw_offsets_deg=[-60, -30, 0, 30, 60],
                reason="room exit unresolved; inspect nearby doorway and corridor headings: {}".format(
                    reason
                ),
                confidence="low",
            )
            return output, "request_observation.directed_sweep"

        output = self._nav_modules.schema.make_rotate_output(
            45,
            reason="R04_FORCED_ROOM_EXIT_REQUIRES_VISIBLE_GATEWAY",
            confidence="low",
        )
        return output, "rotate.search_visible_gateway"

    def _forced_exit_memory_backtrack_output(
        self,
        vlm_input: Dict[str, Any],
        strategic_state: Dict[str, Any],
        *,
        excluded_candidate_refs: Optional[Sequence[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._input_is_full_sweep(vlm_input):
            return None
        failure_limit = self._positive_int_env(
            "VOCA_FORCE_LEAVE_BACKTRACK_AFTER",
            4,
        )
        if self._forced_exit_failure_streak < failure_limit:
            return None
        pixel_context = (
            vlm_input.get("pixel_candidates")
            if isinstance(vlm_input.get("pixel_candidates"), dict)
            else {}
        )
        excluded_refs = {
            str(candidate_ref)
            for candidate_ref in (excluded_candidate_refs or [])
            if str(candidate_ref or "").strip()
        }
        candidates = [
            dict(candidate)
            for candidate in pixel_context.get("candidates", []) or []
            if isinstance(candidate, dict)
            and str(candidate.get("candidate_ref") or "") not in excluded_refs
            and not bool(candidate.get("avoid"))
            and str(candidate.get("topological_relation_type") or "").strip().lower()
            == "backtrack"
            and isinstance(candidate.get("point_px"), (list, tuple))
            and len(candidate.get("point_px")) == 2
        ]
        if not candidates:
            return None

        def priority(candidate: Dict[str, Any]) -> Tuple[float, float, str]:
            evidence = (
                candidate.get("pixelnav_feasibility")
                if isinstance(candidate.get("pixelnav_feasibility"), dict)
                else {}
            )
            try:
                feasibility = float(evidence.get("policy_feasibility_score", 0.0))
            except (TypeError, ValueError):
                feasibility = 0.0
            try:
                score = float(candidate.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            return feasibility, score, str(candidate.get("candidate_ref") or "")

        candidate = max(candidates, key=priority)
        observation = (
            vlm_input.get("observation")
            if isinstance(vlm_input.get("observation"), dict)
            else {}
        )
        width = int(observation.get("image_width", 640) or 640)
        height = int(observation.get("image_height", 480) or 480)
        point = candidate.get("point_px") or [width // 2, int(height * 0.84)]
        output = self._nav_modules.schema.make_go_output(
            view_id=int(candidate.get("view_id", 0) or 0),
            view_type=str(candidate.get("view_type_hint") or "front"),
            point_px=(int(point[0]), int(point[1])),
            width=width,
            height=height,
            decision_reason="G07_VERIFIED_MEMORY_BACKTRACK",
            goal_reason="F06_ESCAPE_DEADLOCK_BACKTRACK",
            short_text="repeated exit search failed; follow verified memory backtrack edge",
            confidence="medium",
            selected_candidate_ref=str(candidate.get("candidate_ref") or ""),
        )
        output["strategic_state"] = {
            **dict(strategic_state),
            "target_evidence": "none",
            "navigation_intent": "escape_deadlock",
            "room_search_status": "exhausted",
            "selected_exit_ref": None,
            "reason_code": "BACKEND_VERIFIED_MEMORY_BACKTRACK",
        }
        output["backend_memory_backtrack"] = {
            "triggered": True,
            "selected_candidate_ref": candidate.get("candidate_ref"),
            "topological_candidate_ref": candidate.get(
                "topological_candidate_ref"
            ),
            "topological_edge_id": candidate.get("topological_edge_id"),
            "topological_relation_type": "backtrack",
            "failure_streak": int(self._forced_exit_failure_streak),
        }
        self._forced_exit_memory_backtrack_count += 1
        if self.memory_sidecar is not None:
            self.memory_sidecar.update_supervisor_mode(
                "escape_deadlock",
                reason="forced_exit_full_sweep_selected_verified_backtrack",
            )
        return output

    def _forced_exit_novel_sector_output(
        self,
        vlm_input: Dict[str, Any],
        strategic_state: Dict[str, Any],
        *,
        excluded_candidate_refs: Optional[Sequence[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Select one safe unblocked sweep sector when no backtrack edge exists."""
        if not self._input_is_full_sweep(vlm_input):
            return None
        failure_limit = self._positive_int_env(
            "VOCA_FORCE_LEAVE_NOVEL_SECTOR_AFTER",
            4,
        )
        if self._forced_exit_failure_streak < failure_limit:
            return None
        pixel_context = (
            vlm_input.get("pixel_candidates")
            if isinstance(vlm_input.get("pixel_candidates"), dict)
            else {}
        )
        excluded_refs = {
            str(candidate_ref)
            for candidate_ref in (excluded_candidate_refs or [])
            if str(candidate_ref)
        }
        candidates: List[Dict[str, Any]] = []
        for raw_candidate in pixel_context.get("candidates", []) or []:
            if not isinstance(raw_candidate, dict):
                continue
            candidate = dict(raw_candidate)
            if str(candidate.get("candidate_ref") or "") in excluded_refs:
                continue
            visual = (
                candidate.get("visual_evidence")
                if isinstance(candidate.get("visual_evidence"), dict)
                else {}
            )
            relation = str(
                candidate.get("topological_relation_type") or ""
            ).strip().lower()
            try:
                visual_support = float(visual.get("support_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                visual_support = 0.0
            if (
                candidate.get("avoid")
                or (
                    candidate.get("requires_rgb_verification")
                    and not candidate.get(
                        "negative_memory_saturation_recovery"
                    )
                )
                or visual.get("hard_reject")
                or relation in {"backtrack", "directional_failure"}
                or visual_support < 0.55
                or not isinstance(candidate.get("point_px"), (list, tuple))
                or len(candidate.get("point_px")) != 2
            ):
                continue
            candidate["_novel_visual_support"] = visual_support
            candidates.append(candidate)
        if not candidates:
            return None

        def priority(candidate: Dict[str, Any]) -> Tuple[Any, ...]:
            evidence = (
                candidate.get("pixelnav_feasibility")
                if isinstance(candidate.get("pixelnav_feasibility"), dict)
                else {}
            )
            try:
                feasibility = float(evidence.get("policy_feasibility_score", 0.0))
            except (TypeError, ValueError):
                feasibility = 0.0
            try:
                score = float(candidate.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            point_norm = candidate.get("point_norm") or [0.5, 0.84]
            try:
                center_alignment = -abs(float(point_norm[0]) - 0.5)
            except (TypeError, ValueError, IndexError):
                center_alignment = -1.0
            return (
                float(candidate.get("_novel_visual_support", 0.0)),
                feasibility,
                score,
                center_alignment,
                str(candidate.get("candidate_ref") or ""),
            )

        candidate = max(candidates, key=priority)
        observation = (
            vlm_input.get("observation")
            if isinstance(vlm_input.get("observation"), dict)
            else {}
        )
        width = int(observation.get("image_width", 640) or 640)
        height = int(observation.get("image_height", 480) or 480)
        point = candidate.get("point_px") or [width // 2, int(height * 0.84)]
        output = self._nav_modules.schema.make_go_output(
            view_id=int(candidate.get("view_id", 0) or 0),
            view_type=str(candidate.get("view_type_hint") or "front"),
            point_px=(int(point[0]), int(point[1])),
            width=width,
            height=height,
            decision_reason="G08_NOVEL_SAFE_SECTOR_ESCAPE",
            goal_reason="F06_ESCAPE_DEADLOCK_NOVEL_SECTOR",
            short_text=(
                "no verified backtrack is available; traverse the safest unblocked sweep sector"
            ),
            confidence="medium",
            selected_candidate_ref=str(candidate.get("candidate_ref") or ""),
        )
        output["strategic_state"] = {
            **dict(strategic_state),
            "target_evidence": "none",
            "navigation_intent": "escape_deadlock",
            "room_search_status": "exhausted",
            "selected_exit_ref": None,
            "reason_code": "BACKEND_NOVEL_SAFE_SECTOR_ESCAPE",
        }
        output["backend_novel_sector_escape"] = {
            "triggered": True,
            "selected_candidate_ref": candidate.get("candidate_ref"),
            "selected_heading_deg": candidate.get("relative_heading_deg"),
            "visual_support_score": round(
                float(candidate.get("_novel_visual_support", 0.0)),
                4,
            ),
            "failure_streak": int(self._forced_exit_failure_streak),
            "authority": "full_sweep_unblocked_candidate_after_repeated_exit_failure",
        }
        self._forced_exit_novel_sector_count += 1
        if self.memory_sidecar is not None:
            self.memory_sidecar.update_supervisor_mode(
                "escape_deadlock",
                reason="forced_exit_full_sweep_selected_novel_safe_sector",
            )
        return output

    def _record_strategic_state(
        self,
        output: Dict[str, Any],
        *,
        frame_index: int,
    ) -> Dict[str, Any]:
        state = normalize_strategic_state(
            output.get("strategic_state"),
            action=str(output.get("action") or ""),
            reasoning=(
                output.get("reasoning")
                if isinstance(output.get("reasoning"), dict)
                else {}
            ),
        )
        previous = dict(self._last_strategic_state)
        previous_room = str(previous.get("current_room") or "unknown")
        current_room = str(state.get("current_room") or "unknown")
        current_position = list(self._runtime_position_xyz)
        previous_position = list(self._last_strategic_position_xyz)
        movement_m = 0.0
        if len(current_position) >= 3 and len(previous_position) >= 3:
            movement_m = float(
                np.linalg.norm(
                    np.asarray(current_position[:3], dtype=np.float64)
                    - np.asarray(previous_position[:3], dtype=np.float64)
                )
            )
        try:
            progress_reset_m = max(
                0.0,
                float(os.environ.get("VOCA_SAME_ROOM_PROGRESS_RESET_M", "0.20")),
            )
        except ValueError:
            progress_reset_m = 0.20
        spatial_progress = bool(
            len(current_position) >= 3
            and (len(previous_position) < 3 or movement_m >= progress_reset_m)
        )
        room_changed = bool(
            previous_room != "unknown"
            and current_room != "unknown"
            and previous_room != current_room
        )
        if room_changed or spatial_progress:
            self._same_room_cycles = 1
        elif current_room == previous_room or current_room == "unknown":
            self._same_room_cycles += 1
        else:
            self._same_room_cycles = 1

        action = str(output.get("action") or "").strip().lower()
        intent = str(state.get("navigation_intent") or "search_current_room")
        evidence = str(state.get("target_evidence") or "none")
        generic_search_go = bool(
            action == "go"
            and intent == "search_current_room"
            and evidence in {"none", "context_only"}
        )
        if generic_search_go:
            self._generic_floor_go_streak += 1
        elif evidence in {"possible", "confirmed"} or (
            action == "go" and intent in _FORCE_LEAVE_COMPATIBLE_INTENTS
        ):
            self._generic_floor_go_streak = 0

        selected_exit = state.get("selected_exit_ref")
        if selected_exit:
            self._recent_selected_exit_refs.append(str(selected_exit))
            self._recent_selected_exit_refs = self._recent_selected_exit_refs[-8:]
        self._strategic_intent_counts[intent] = (
            int(self._strategic_intent_counts.get(intent, 0)) + 1
        )
        self._last_strategic_state = dict(state)
        self._last_strategic_position_xyz = current_position
        self._strategic_state_history.append(
            {
                "frame_index": int(frame_index),
                "action": action,
                "room_changed": room_changed,
                "spatial_progress": spatial_progress,
                "movement_since_last_cycle_m": round(movement_m, 4),
                **dict(state),
            }
        )
        self._strategic_state_history = self._strategic_state_history[-32:]
        return state

    def _apply_strategic_exit_guard(
        self,
        output: Dict[str, Any],
        vlm_input: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
        safe_output = dict(output)
        state = normalize_strategic_state(
            safe_output.get("strategic_state"),
            action=str(safe_output.get("action") or ""),
            reasoning=(
                safe_output.get("reasoning")
                if isinstance(safe_output.get("reasoning"), dict)
                else {}
            ),
        )
        safe_output["strategic_state"] = state
        runtime = (
            vlm_input.get("runtime", {}).get("strategic_runtime", {})
            if isinstance(vlm_input.get("runtime"), dict)
            else {}
        )
        force_leave = bool(runtime.get("force_leave_room"))
        action = str(safe_output.get("action") or "").strip().lower()
        intent = str(state.get("navigation_intent") or "")
        target_evidence = str(state.get("target_evidence") or "none")
        selected_candidate = str(
            safe_output.get("selected_candidate_ref") or ""
        ).strip()
        selected_exit = str(state.get("selected_exit_ref") or "").strip()
        validation = {
            "schema_version": "strategic_exit_validation_v1",
            "triggered": False,
            "passed": True,
            "reason": "force_leave_not_active",
            "force_leave_room": force_leave,
            "force_reason": str(runtime.get("force_reason") or ""),
            "navigation_intent": intent,
            "selected_candidate_ref": selected_candidate or None,
            "selected_exit_ref": selected_exit or None,
            "checkpoint": {
                "rule_id": "STRATEGY_EXIT_001",
                "name": "forced_room_exit_requires_explicit_exit_intent",
                "stage": "strategic_action_validation",
                "passed": True,
                "message": "force_leave_not_active",
                "on_fail": None,
            },
        }
        selected_pixel_candidate = self._selected_pixel_candidate(
            vlm_input,
            selected_candidate,
        )
        topological_candidate_ref = str(
            selected_pixel_candidate.get("topological_candidate_ref") or ""
        ).strip()
        exit_alias_repaired = bool(
            action == "go"
            and intent in _EXIT_INTENTS
            and selected_candidate
            and selected_exit
            and selected_exit != selected_candidate
            and topological_candidate_ref
            and selected_exit == topological_candidate_ref
        )
        if exit_alias_repaired:
            selected_exit = selected_candidate
            state = {**dict(state), "selected_exit_ref": selected_candidate}
            safe_output["strategic_state"] = dict(state)
            validation["selected_exit_ref"] = selected_candidate
            validation["selected_exit_ref_alias_repair"] = {
                "triggered": True,
                "from": topological_candidate_ref,
                "to": selected_candidate,
                "authority": "validated_pixel_candidate_topological_alias",
            }
            self._strategic_exit_alias_repair_count += 1
        visual_exit_ref_repaired = bool(
            _env_bool("VOCA_FORCE_EXIT_VISUAL_REF_REPAIR", True)
            and force_leave
            and action == "go"
            and intent in _EXIT_INTENTS
            and selected_candidate
            and not selected_exit
            and selected_pixel_candidate
        )
        if visual_exit_ref_repaired:
            selected_exit = selected_candidate
            state = {**dict(state), "selected_exit_ref": selected_candidate}
            safe_output["strategic_state"] = dict(state)
            validation["selected_exit_ref"] = selected_candidate
            validation["selected_exit_ref_visual_repair"] = {
                "triggered": True,
                "from": None,
                "to": selected_candidate,
                "authority": (
                    "validated_pixel_candidate_with_mandatory_doorway_floor_verifier"
                ),
            }
            self._strategic_visual_exit_ref_repair_count += 1
        selected_relation = str(
            selected_pixel_candidate.get("topological_relation_type") or ""
        ).strip().lower()
        memory_mode = str(
            self.memory_sidecar.supervisor_mode
            if self.memory_sidecar is not None
            else "goal_seek"
        ).strip().lower()
        backtrack_authorized = bool(
            force_leave
            or target_evidence in {"possible", "confirmed"}
            or intent in _FORCE_LEAVE_COMPATIBLE_INTENTS
            or memory_mode in {"backtrack", "escape_deadlock"}
            or safe_output.get("backend_memory_backtrack")
        )
        if (
            action == "go"
            and selected_relation == "backtrack"
            and not backtrack_authorized
        ):
            replacement = self._nav_modules.schema.make_observation_request_output(
                mode="directed_sweep",
                center_yaw_deg=180.0,
                step_deg=30,
                num_views=5,
                yaw_offsets_deg=[120, 150, 180, -150, -120],
                reason=(
                    "normal goal-seek cannot immediately reverse a verified transition; "
                    "inspect directions away from the incoming edge"
                ),
                confidence="low",
            )
            replacement["memory_ops"] = list(safe_output.get("memory_ops") or [])
            replacement["strategic_state"] = dict(state)
            self._normal_goal_seek_backtrack_rejection_count += 1
            validation.update(
                triggered=True,
                passed=False,
                reason="normal_goal_seek_backtrack_requires_recovery_intent",
                selected_topological_edge_id=selected_pixel_candidate.get(
                    "topological_edge_id"
                ),
                selected_topological_relation_type=selected_relation,
                recovery_action="request_observation.opposite_incoming_edge",
            )
            validation["checkpoint"] = {
                "rule_id": "STRATEGY_BACKTRACK_001",
                "name": "normal_goal_seek_rejects_unjustified_backtrack",
                "stage": "strategic_action_validation",
                "passed": False,
                "message": validation["reason"],
                "on_fail": validation["recovery_action"],
            }
            return replacement, validation, [
                "strategic_backtrack_guard:normal_goal_seek_backtrack_rejected"
            ]
        if not force_leave or target_evidence in {"possible", "confirmed"}:
            if force_leave:
                validation["reason"] = "target_evidence_exempts_room_exit"
                validation["checkpoint"]["message"] = validation["reason"]
            return safe_output, validation, []

        if action != "go":
            backtrack_output = (
                self._forced_exit_memory_backtrack_output(vlm_input, state)
                if action in {"rotate", "request_observation"}
                else None
            )
            novel_sector_output = (
                self._forced_exit_novel_sector_output(vlm_input, state)
                if backtrack_output is None
                and action in {"rotate", "request_observation"}
                else None
            )
            recovery_output = backtrack_output or novel_sector_output
            if recovery_output is not None:
                recovery_output["memory_ops"] = list(
                    safe_output.get("memory_ops") or []
                )
                self._strategic_force_leave_override_count += 1
                backtrack_selected = backtrack_output is not None
                recovery_audit = dict(
                    recovery_output.get(
                        "backend_memory_backtrack"
                        if backtrack_selected
                        else "backend_novel_sector_escape"
                    )
                    or {}
                )
                validation.update(
                    triggered=True,
                    passed=True,
                    reason=(
                        "forced_exit_memory_backtrack_selected"
                        if backtrack_selected
                        else "forced_exit_novel_sector_selected"
                    ),
                    recovery_action=(
                        "go.verified_memory_backtrack"
                        if backtrack_selected
                        else "go.novel_safe_sector"
                    ),
                )
                validation[
                    "backend_memory_backtrack"
                    if backtrack_selected
                    else "backend_novel_sector_escape"
                ] = recovery_audit
                validation["checkpoint"].update(
                    passed=True,
                    message=validation["reason"],
                    on_fail=None,
                )
                return recovery_output, validation, [
                    (
                        "strategic_exit_guard:verified_memory_backtrack_override"
                        if backtrack_selected
                        else "strategic_exit_guard:novel_safe_sector_override"
                    )
                ]
            validation["reason"] = "non_go_action_allowed_during_forced_exit"
            validation["checkpoint"]["message"] = validation["reason"]
            return safe_output, validation, []

        validation["triggered"] = True
        compatible = intent in _FORCE_LEAVE_COMPATIBLE_INTENTS
        selected_exit_aliases = {
            value
            for value in (
                selected_candidate,
                str(
                    selected_pixel_candidate.get("topological_candidate_ref")
                    or ""
                ).strip(),
                str(
                    selected_pixel_candidate.get("topological_edge_id") or ""
                ).strip(),
            )
            if value
        }
        validation["selected_exit_aliases"] = sorted(selected_exit_aliases)
        exit_ref_matches = bool(
            intent not in _EXIT_INTENTS
            or (
                selected_candidate
                and selected_exit
                and selected_exit in selected_exit_aliases
            )
        )
        if compatible and exit_ref_matches:
            validation["reason"] = "explicit_exit_strategy_valid"
            validation["checkpoint"]["message"] = validation["reason"]
            return safe_output, validation, []

        reason = (
            "forced_exit_candidate_ref_mismatch"
            if compatible
            else "forced_exit_rejected_generic_go"
        )
        memory_ops = list(safe_output.get("memory_ops") or [])
        failure_streak = self._record_forced_exit_failure(reason)
        replacement = self._forced_exit_memory_backtrack_output(vlm_input, state)
        if replacement is None:
            replacement = self._forced_exit_novel_sector_output(vlm_input, state)
        if replacement is not None:
            recovery_action = (
                "go.verified_memory_backtrack"
                if replacement.get("backend_memory_backtrack")
                else "go.novel_safe_sector"
            )
        else:
            replacement, recovery_action = self._forced_exit_recovery_output(
                vlm_input,
                reason=reason,
            )
        replacement["memory_ops"] = memory_ops
        if not replacement.get("strategic_state"):
            replacement["strategic_state"] = state
        self._strategic_force_leave_override_count += 1
        backend_recovery = bool(
            replacement.get("backend_memory_backtrack")
            or replacement.get("backend_novel_sector_escape")
        )
        validation.update(
            passed=backend_recovery,
            reason=(
                "forced_exit_memory_backtrack_selected"
                if replacement.get("backend_memory_backtrack")
                else (
                    "forced_exit_novel_sector_selected"
                    if replacement.get("backend_novel_sector_escape")
                    else reason
                )
            ),
            forced_exit_failure_streak=failure_streak,
            recovery_action=recovery_action,
        )
        if replacement.get("backend_memory_backtrack"):
            validation["backend_memory_backtrack"] = dict(
                replacement["backend_memory_backtrack"]
            )
        if replacement.get("backend_novel_sector_escape"):
            validation["backend_novel_sector_escape"] = dict(
                replacement["backend_novel_sector_escape"]
            )
        validation["checkpoint"].update(
            passed=backend_recovery,
            message=validation["reason"],
            on_fail=(
                None
                if backend_recovery
                else recovery_action
            ),
        )
        warning = (
            "strategic_exit_guard:verified_memory_backtrack_override"
            if replacement.get("backend_memory_backtrack")
            else (
                "strategic_exit_guard:novel_safe_sector_override"
                if replacement.get("backend_novel_sector_escape")
                else "strategic_exit_guard:{}".format(reason)
            )
        )
        return replacement, validation, [warning]

    def _apply_coarse_goal_direction_guard(
        self,
        output: Dict[str, Any],
        vlm_input: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
        """Reject unsupported generic motion away from a declared coarse goal."""
        safe_output = dict(output)
        task = vlm_input.get("task") if isinstance(vlm_input.get("task"), dict) else {}
        coarse_goal = (
            task.get("coarse_goal")
            if isinstance(task.get("coarse_goal"), dict)
            else {}
        )
        action = str(safe_output.get("action") or "").strip().lower()
        uncertainty = str(coarse_goal.get("uncertainty") or "medium").strip().lower()
        validation = {
            "schema_version": "coarse_goal_direction_guard_v1",
            "enabled": bool(coarse_goal.get("goal_geometry_available")),
            "triggered": False,
            "passed": True,
            "classification": "not_applicable",
            "coarse_goal_source": coarse_goal.get("source"),
            "coarse_goal_uncertainty": uncertainty,
            "coarse_goal_distance_m": coarse_goal.get("distance_m"),
            "coarse_goal_bearing_deg": coarse_goal.get("relative_bearing_deg"),
        }
        if not validation["enabled"] or action != "go":
            return safe_output, validation, []

        try:
            coarse_bearing = float(coarse_goal.get("relative_bearing_deg"))
        except (TypeError, ValueError):
            validation["classification"] = "coarse_bearing_unavailable"
            return safe_output, validation, []
        if not math.isfinite(coarse_bearing):
            validation["classification"] = "coarse_bearing_unavailable"
            return safe_output, validation, []
        coarse_bearing = _normalize_angle_deg(coarse_bearing)

        local_search_defaults = {"low": 1.0, "medium": 2.5, "high": 4.0}
        try:
            local_search_radius_m = max(
                0.1,
                float(
                    os.environ.get(
                        "VOCA_COARSE_GOAL_STOP_RADIUS_{}_M".format(
                            uncertainty.upper()
                        ),
                        str(local_search_defaults.get(uncertainty, 2.5)),
                    )
                ),
            )
        except ValueError:
            local_search_radius_m = local_search_defaults.get(uncertainty, 2.5)
        validation["local_search_radius_m"] = round(
            float(local_search_radius_m),
            4,
        )

        try:
            coarse_distance_m = float(coarse_goal.get("distance_m"))
        except (TypeError, ValueError):
            coarse_distance_m = None
        if coarse_distance_m is not None and not math.isfinite(coarse_distance_m):
            coarse_distance_m = None
        if (
            coarse_distance_m is not None
            and coarse_distance_m <= local_search_radius_m
        ):
            validation["classification"] = "inside_local_search_region"
            return safe_output, validation, []

        state = normalize_strategic_state(
            safe_output.get("strategic_state"),
            action=action,
            reasoning=(
                safe_output.get("reasoning")
                if isinstance(safe_output.get("reasoning"), dict)
                else {}
            ),
        )
        intent = str(state.get("navigation_intent") or "")
        target_evidence = str(state.get("target_evidence") or "none")
        runtime = (
            vlm_input.get("runtime", {}).get("strategic_runtime", {})
            if isinstance(vlm_input.get("runtime"), dict)
            else {}
        )
        memory_mode = str(
            self.memory_sidecar.supervisor_mode
            if self.memory_sidecar is not None
            else "goal_seek"
        ).strip().lower()
        detour_authorized = bool(
            intent in _FORCE_LEAVE_COMPATIBLE_INTENTS
            or bool(runtime.get("force_leave_room"))
            or memory_mode in {"backtrack", "escape_deadlock"}
            or safe_output.get("backend_memory_backtrack")
            or safe_output.get("backend_novel_sector_escape")
        )
        if detour_authorized:
            validation.update(
                triggered=True,
                classification="explicit_detour_authorized",
                navigation_intent=intent,
                target_evidence=target_evidence,
                supervisor_mode=memory_mode,
            )
            safe_output["backend_coarse_goal_direction_guard"] = dict(validation)
            self._coarse_direction_guard_count += 1
            self._coarse_direction_detour_exemption_count += 1
            return safe_output, validation, []

        selected_ref = str(safe_output.get("selected_candidate_ref") or "").strip()
        candidate = self._selected_pixel_candidate(vlm_input, selected_ref)
        try:
            candidate_bearing = float(
                candidate.get(
                    "bearing_deg_robot",
                    candidate.get("relative_heading_deg", 0.0),
                )
            )
        except (TypeError, ValueError):
            candidate_bearing = 0.0
        point_norm = candidate.get("point_norm")
        if isinstance(point_norm, (list, tuple)) and point_norm:
            try:
                horizontal_fov_deg = float(
                    os.environ.get("VOCA_CAMERA_HORIZONTAL_FOV_DEG", "90.0")
                )
                candidate_bearing += (float(point_norm[0]) - 0.5) * horizontal_fov_deg
            except (TypeError, ValueError):
                pass
        candidate_bearing = _normalize_angle_deg(candidate_bearing)
        angular_error_deg = abs(
            _normalize_angle_deg(candidate_bearing - coarse_bearing)
        )
        error_defaults = {"low": 75.0, "medium": 90.0, "high": 120.0}
        try:
            max_error_deg = max(
                15.0,
                min(
                    180.0,
                    float(
                        os.environ.get(
                            "VOCA_COARSE_GOAL_MAX_GENERIC_GO_ERROR_{}_DEG".format(
                                uncertainty.upper()
                            ),
                            str(error_defaults.get(uncertainty, 90.0)),
                        )
                    ),
                ),
            )
        except ValueError:
            max_error_deg = error_defaults.get(uncertainty, 90.0)
        validation.update(
            triggered=True,
            selected_candidate_ref=selected_ref or None,
            selected_candidate_bearing_deg=round(float(candidate_bearing), 4),
            angular_error_deg=round(float(angular_error_deg), 4),
            max_generic_go_error_deg=round(float(max_error_deg), 4),
            navigation_intent=intent,
            target_evidence=target_evidence,
            supervisor_mode=memory_mode,
        )
        self._coarse_direction_guard_count += 1
        if angular_error_deg <= max_error_deg:
            validation["classification"] = "generic_go_direction_consistent"
            safe_output["backend_coarse_goal_direction_guard"] = dict(validation)
            return safe_output, validation, []

        offsets = [
            int(round(_normalize_angle_deg(coarse_bearing + delta)))
            for delta in (-60.0, -30.0, 0.0, 30.0, 60.0)
        ]
        replacement = self._nav_modules.schema.make_observation_request_output(
            mode="directed_sweep",
            center_yaw_deg=float(coarse_bearing),
            step_deg=30,
            num_views=5,
            yaw_offsets_deg=offsets,
            reason=(
                "generic floor waypoint points away from the supplied coarse goal; "
                "inspect goal-aligned headings or declare an explicit doorway/deadlock detour"
            ),
            confidence="low",
        )
        replacement["memory_ops"] = list(safe_output.get("memory_ops") or [])
        replacement["strategic_state"] = {
            **dict(state),
            "target_evidence": "none",
            "navigation_intent": "escape_deadlock",
            "room_search_status": "exhausted",
            "selected_exit_ref": None,
            "reason_code": "BACKEND_COARSE_DIRECTION_REALIGN",
        }
        generic_limit = self._positive_int_env(
            "VOCA_FORCE_LEAVE_GENERIC_GO_LIMIT",
            3,
        )
        self._generic_floor_go_streak = max(
            int(self._generic_floor_go_streak),
            int(generic_limit),
        )
        validation.update(
            passed=False,
            classification="unsupported_generic_go_away_from_coarse_goal",
            recovery_action="request_observation.coarse_goal_directed_sweep",
            requested_yaw_offsets_deg=offsets,
            force_leave_armed_next_cycle=True,
        )
        replacement["backend_coarse_goal_direction_guard"] = dict(validation)
        self._coarse_direction_rejection_count += 1
        if self.memory_sidecar is not None:
            self.memory_sidecar.update_supervisor_mode(
                "escape_deadlock",
                reason="coarse_goal_direction_blocked_requires_explicit_detour",
            )
        self.record_navigation_feedback(
            {
                "event": "coarse_goal_direction_rejected",
                "frame_index": self._runtime_frame_index,
                "coarse_goal_bearing_deg": round(float(coarse_bearing), 4),
                "selected_candidate_bearing_deg": round(float(candidate_bearing), 4),
                "angular_error_deg": round(float(angular_error_deg), 4),
                "selected_candidate_ref": selected_ref or None,
            }
        )
        return replacement, validation, [
            "coarse_goal_direction_guard:unsupported_generic_go"
        ]

    def record_target_approach_execution(
        self,
        go_execution: Dict[str, Any],
        *,
        frame_index: int,
    ) -> Dict[str, Any]:
        execution = go_execution if isinstance(go_execution, dict) else {}
        progress = (
            execution.get("go_progress")
            if isinstance(execution.get("go_progress"), dict)
            else {}
        )
        self._last_target_approach_execution = {
            "frame_index": int(frame_index),
            "translation_m": float(progress.get("translation_m", 0.0) or 0.0),
            "collision_count": int(progress.get("collision_count", 0) or 0),
            "controller_reached_waypoint": bool(
                progress.get("controller_reached_waypoint")
            ),
            "stop_action_seen": bool(execution.get("stop_action_seen")),
            "no_progress": bool(progress.get("no_progress")),
            "execution_success": bool(progress.get("execution_success")),
            "selected_candidate_ref": execution.get("selected_candidate_ref"),
        }
        approach_progress_satisfied = bool(
            progress.get("execution_success")
            and int(progress.get("collision_count", 0) or 0) == 0
        )
        self._last_target_approach_execution["progress_satisfied"] = bool(
            approach_progress_satisfied
        )
        near_latch = dict(self._near_goal_visual_latch)
        try:
            near_latch_frame = int(near_latch.get("frame_index"))
        except (TypeError, ValueError):
            near_latch_frame = -1
        latch_translation = float(progress.get("translation_m", 0.0) or 0.0)
        try:
            min_latch_translation = max(
                0.0,
                float(
                    os.environ.get(
                        "VOCA_STOP_TARGET_APPROACH_CONFIRM_TRANSLATION_M",
                        "0.10",
                    )
                ),
            )
        except ValueError:
            min_latch_translation = 0.10
        if (
            near_latch_frame >= 0
            and int(frame_index) > near_latch_frame
            and latch_translation + 1e-6 >= min_latch_translation
            and approach_progress_satisfied
            and int(frame_index)
            > int(near_latch.get("last_approach_frame_index", -1) or -1)
        ):
            near_latch["approach_count"] = int(
                near_latch.get("approach_count", 0) or 0
            ) + 1
            near_latch["last_approach_frame_index"] = int(frame_index)
            near_latch["last_approach_translation_m"] = latch_translation
            self._near_goal_visual_latch = near_latch
        terminal = bool(
            execution.get("stop_action_seen")
            or progress.get("controller_reached_waypoint")
        )
        outcome = {
            "detour_triggered": False,
            "terminal": terminal,
            "progress_satisfied": bool(approach_progress_satisfied),
            "failure_streak": int(self._target_approach_failure_streak),
        }
        if terminal or approach_progress_satisfied:
            self._target_approach_failure_streak = 0
            self._target_direct_approach_cooldown = 0
            outcome["failure_streak"] = 0
            return outcome
        self._target_approach_failure_streak += 1
        outcome["failure_streak"] = int(self._target_approach_failure_streak)
        try:
            failure_limit = max(
                1,
                int(os.environ.get("VOCA_TARGET_APPROACH_FAILURE_LIMIT", "3")),
            )
        except ValueError:
            failure_limit = 3
        if self._target_approach_failure_streak < failure_limit:
            return outcome
        try:
            cooldown_cycles = max(
                1,
                int(os.environ.get("VOCA_TARGET_APPROACH_COOLDOWN_CYCLES", "4")),
            )
        except ValueError:
            cooldown_cycles = 4
        self._target_direct_approach_cooldown = max(
            self._target_direct_approach_cooldown,
            cooldown_cycles,
        )
        feedback = {
            "event": "target_approach_unresolved",
            "selected_candidate_ref": execution.get("selected_candidate_ref"),
            "selected_view_id": execution.get("selected_view_id"),
            "angle_deg": execution.get("selected_angle_deg"),
            "point_px": list(execution.get("selected_point_px") or []),
            "failure_streak": int(self._target_approach_failure_streak),
            "failure_class": "target_approach_unresolved",
            "reason": (
                "repeated direct target approach did not reach its PixelNav waypoint; "
                "seek a doorway, gateway, or detour before approaching the target again"
            ),
        }
        self.record_navigation_feedback(feedback)
        if self.memory_sidecar is not None:
            self.memory_sidecar.update_supervisor_mode(
                "escape_deadlock",
                reason="repeated_target_approach_unresolved",
            )
        outcome.update(
            detour_triggered=True,
            cooldown_cycles=int(self._target_direct_approach_cooldown),
            feedback=dict(feedback),
        )
        return outcome

    def update_runtime_context(
        self,
        *,
        position_xyz: Optional[Sequence[float]] = None,
        heading_rad: Optional[float] = None,
        frame_index: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        spatial_coarse_goal: Optional[Dict[str, Any]] = None,
    ) -> None:
        if position_xyz is not None:
            self._runtime_position_xyz = [float(v) for v in position_xyz]
        if heading_rad is not None:
            self._runtime_heading_rad = float(heading_rad)
        if frame_index is not None:
            self._runtime_frame_index = int(frame_index)
        if metrics is not None:
            self._runtime_audit_metrics = dict(metrics)
        if spatial_coarse_goal is not None:
            self._runtime_spatial_coarse_goal = dict(spatial_coarse_goal)

    def set_qwen_call_log_path(self, path: str) -> None:
        super().set_qwen_call_log_path(path)
        self._image_cache_dir = Path(path).parent / "qwen_vlm_images"
        self._image_cache_dir.mkdir(parents=True, exist_ok=True)

    def set_pixel_candidate_scorer(self, scorer: Optional[Any]) -> None:
        self._pixel_candidate_scorer = scorer if callable(scorer) else None

    def _apply_pixelnav_candidate_scores(
        self,
        rgb_images: Sequence[np.ndarray],
        candidates: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        output = [dict(candidate) for candidate in candidates if isinstance(candidate, dict)]
        audit = {
            "schema_version": "pixelnav_candidate_conditioning_v1",
            "status": "unavailable",
            "scored_count": 0,
            "candidate_count": len(output),
            "duration_sec": 0.0,
        }
        if self._pixel_candidate_scorer is None or not output:
            audit["reason"] = "scorer_not_attached" if output else "no_executable_candidates"
            return output, audit

        started = time.perf_counter()
        try:
            records = self._pixel_candidate_scorer(rgb_images, output)
            score_by_ref = {
                str(record.get("candidate_ref") or ""): dict(record)
                for record in records
                if isinstance(record, dict) and record.get("candidate_ref")
            }
            scored_count = 0
            for candidate in output:
                evidence = score_by_ref.get(str(candidate.get("candidate_ref") or ""))
                if not evidence:
                    continue
                feasibility = float(evidence.get("policy_feasibility_score", 0.0))
                prior_score = float(candidate.get("score", 0.5))
                candidate["pixelnav_feasibility"] = evidence
                candidate["score_before_pixelnav"] = round(prior_score, 4)
                candidate["score"] = round(0.4 * prior_score + 0.6 * feasibility, 4)
                candidate["score_source"] = "rgb_memory_pixelnav_blend_v1"
                action_probabilities = (
                    evidence.get("action_probabilities")
                    if isinstance(evidence.get("action_probabilities"), dict)
                    else {}
                )
                try:
                    stop_reject_probability = float(
                        os.environ.get("VOCA_PIXELNAV_STOP_REJECT_PROB", "0.50")
                    )
                except ValueError:
                    stop_reject_probability = 0.50
                stop_probability = float(
                    evidence.get("stop_probability", action_probabilities.get("stop", 0.0))
                )
                locomotion_probability = float(
                    evidence.get(
                        "locomotion_probability",
                        sum(
                            float(action_probabilities.get(name, 0.0))
                            for name in ("forward", "turn_left", "turn_right")
                        ),
                    )
                )
                if (
                    str(evidence.get("predicted_action") or "") == "stop"
                    and stop_probability >= stop_reject_probability
                    and locomotion_probability <= 0.20
                ):
                    candidate.update(
                        avoid=True,
                        executable=False,
                        status="pixelnav_immediate_stop_risk",
                        reason=(
                            "PixelNav predicts immediate stop without locomotion "
                            "(p_stop={:.3f}, p_move={:.3f})"
                        ).format(stop_probability, locomotion_probability),
                        exclusion_reason="pixelnav_immediate_stop_risk",
                    )
                scored_count += 1
            output.sort(
                key=lambda item: (
                    -float(item.get("score", 0.0)),
                    str(item.get("candidate_ref") or ""),
                )
            )
            audit.update(
                status="available" if scored_count == len(output) else "partial",
                scored_count=scored_count,
                reason="pixelnav_one_step_batch",
            )
        except Exception as exc:
            audit.update(
                status="error",
                reason="{}: {}".format(type(exc).__name__, str(exc)),
            )
        audit["duration_sec"] = round(time.perf_counter() - started, 6)
        return output, audit

    def _write_view_images(
        self,
        pano_images: Sequence[np.ndarray],
        *,
        variant: str,
    ) -> List[str]:
        self._image_cache_dir.mkdir(parents=True, exist_ok=True)
        call_idx = len(self.qwen_call_log)
        paths: List[str] = []
        for idx, image in enumerate(pano_images):
            arr = np.asarray(image, dtype=np.uint8)
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr_to_write = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                arr_to_write = arr
            path = self._image_cache_dir / "call_{:04d}_{}_view_{:02d}.jpg".format(
                call_idx,
                str(variant or "image"),
                idx,
            )
            cv2.imwrite(str(path), arr_to_write)
            paths.append(str(path))
        return paths

    def _build_vlm_input(
        self,
        pano_images: Sequence[np.ndarray],
        angles: Sequence[int],
        call_type: str,
    ) -> Dict[str, Any]:
        selected_shape = pano_images[0].shape if pano_images else (480, 640, 3)
        observation_frame_index = (
            int(self._runtime_frame_index)
            if self._runtime_frame_index is not None
            else len(self.qwen_call_log)
        )
        policy_position_xyz = (
            self._runtime_position_xyz
            if self.localization_contract.get("pose_available_to_policy")
            else []
        )
        vlm_input = build_memory_free_vlm_input(
            target_object=self.object_goal,
            image_shape=selected_shape,
            frame_index=observation_frame_index,
            priors=self.latest_priors,
            angles=angles,
            call_type=call_type,
            planner_name=self.planner_name,
            position_xyz=policy_position_xyz,
            heading_rad=self._runtime_heading_rad,
        )
        supervisor_mode = getattr(self.memory_sidecar, "supervisor_mode", "goal_seek")
        apply_policy_safe_navigation_context(
            vlm_input,
            target_object=self.object_goal,
            supervisor_mode=supervisor_mode,
            spatial_goal=self._runtime_spatial_coarse_goal,
        )
        strategic_runtime = self._strategic_runtime_snapshot()
        vlm_input.setdefault("runtime", {})["strategic_runtime"] = dict(
            strategic_runtime
        )
        raw_image_paths = self._write_view_images(pano_images, variant="raw")
        vlm_input.setdefault("metadata", {})["localization_contract"] = dict(
            self.localization_contract
        )
        if self.memory_sidecar is not None:
            try:
                memory_context = self.memory_sidecar.build_context(
                    target_object=self.object_goal,
                    priors=self.latest_priors,
                    image_ref=raw_image_paths[0] if raw_image_paths else None,
                    image_refs=raw_image_paths,
                    frame_index=observation_frame_index,
                    image_shape=selected_shape,
                    position_xyz=policy_position_xyz,
                    heading_rad=self._runtime_heading_rad,
                    supervisor_mode=supervisor_mode,
                    spatial_goal=self._runtime_spatial_coarse_goal,
                )
                vlm_input["memory"] = memory_context
            except Exception as exc:
                vlm_input.setdefault("memory", {}).setdefault("runtime_state", {})[
                    "memory_sidecar_error"
                ] = "{}: {}".format(type(exc).__name__, str(exc))

        observation = vlm_input.get("observation", {})
        views = observation.get("views", []) if isinstance(observation, dict) else []
        memory_context = vlm_input.setdefault("memory", {})
        memory_context.setdefault("voca_sidecar", {})["strategic_runtime"] = dict(
            strategic_runtime
        )
        if self._runtime_feedback:
            memory_context["runtime_navigation_feedback"] = [
                dict(item) for item in self._runtime_feedback
            ]
        topological_refs = memory_context.get("candidate_refs")
        topological_refs = topological_refs if isinstance(topological_refs, dict) else {}
        generated_pixel_candidates = build_pixel_candidates(
            views,
            selected_shape,
            memory_context,
            max_candidates=32,
            rgb_images=pano_images,
        )
        pixel_candidates, excluded_pixel_candidates = partition_pixel_candidates(
            generated_pixel_candidates
        )
        negative_exclusion_count = sum(
            1
            for candidate in excluded_pixel_candidates
            if isinstance(candidate, dict)
            and (
                candidate.get("avoid")
                or str(candidate.get("status") or "").strip().lower()
                in {
                    "blocked",
                    "collision",
                    "deadlock_entry",
                    "deadlock_entry_candidate",
                    "failed",
                    "verified_failed_pixel_neighborhood",
                }
            )
        )
        saturation_recovery_candidates: List[Dict[str, Any]] = []
        saturation_recovery_enabled = _env_bool(
            "VOCA_NEGATIVE_SATURATION_RECOVERY",
            True,
        )
        if (
            saturation_recovery_enabled
            and not pixel_candidates
            and len(views) >= 5
            and negative_exclusion_count > 0
            and negative_exclusion_count == len(excluded_pixel_candidates)
        ):
            saturation_recovery_candidates = build_saturation_recovery_candidates(
                views,
                selected_shape,
                rgb_images=pano_images,
                max_candidates=min(24, max(3, len(views) * 3)),
            )
            pixel_candidates = saturation_recovery_candidates
            generated_pixel_candidates.extend(saturation_recovery_candidates)
        pixel_candidates, policy_conditioning = self._apply_pixelnav_candidate_scores(
            pano_images,
            pixel_candidates,
        )
        pixel_candidates, policy_excluded_candidates = partition_pixel_candidates(
            pixel_candidates
        )
        excluded_pixel_candidates.extend(policy_excluded_candidates)
        policy_conditioning["policy_excluded_count"] = len(policy_excluded_candidates)
        policy_conditioning["negative_saturation_recovery"] = {
            "enabled": bool(saturation_recovery_enabled),
            "triggered": bool(saturation_recovery_candidates),
            "negative_exclusion_count": int(negative_exclusion_count),
            "offered_count": len(saturation_recovery_candidates),
            "executable_after_pixelnav_count": sum(
                bool(candidate.get("negative_memory_saturation_recovery"))
                for candidate in pixel_candidates
            ),
            "policy": (
                "far pixels are temporary verified probes; negative edges remain stored"
            ),
        }
        memory_context["topological_candidate_refs"] = {
            "exits": [
                dict(item)
                for item in topological_refs.get("exits", [])
                if isinstance(item, dict)
            ],
            "revisits": [
                dict(item)
                for item in topological_refs.get("revisits", [])
                if isinstance(item, dict)
            ],
        }
        memory_context["candidate_refs"] = {
            "exits": pixel_candidates,
            "revisits": memory_context["topological_candidate_refs"]["revisits"],
            "policy": (
                "selected_candidate_ref must resolve to one executable RGB pixel candidate; "
                "known failed candidates are omitted and backend rejects missing or unknown refs"
            ),
        }
        harness = memory_context.setdefault("policy_harness_state", {})
        harness.setdefault("requirements", {})["go_requires_selected_candidate_ref"] = True
        memory_context.setdefault("control_policy", {})["candidate_ref_policy"] = (
            "strict RGB pixel candidate gate; raw VLM coordinates are not executable"
        )
        vlm_input["pixel_candidates"] = {
            "schema_version": PIXEL_CANDIDATE_SCHEMA_VERSION,
            "required_for_go": True,
            "max_candidates": 32,
            "candidates": pixel_candidates,
            "excluded_candidates": excluded_pixel_candidates,
            "generated_count": len(generated_pixel_candidates),
            "executable_count": len(pixel_candidates),
            "excluded_count": len(excluded_pixel_candidates),
            "exclusion_policy": "audit_only_not_rendered_or_prompted",
            "policy_conditioning": policy_conditioning,
        }

        marked_images = []
        for index, image in enumerate(pano_images):
            view_id = int(views[index].get("view_id", index)) if index < len(views) else index
            marked_images.append(
                render_pixel_candidates(image, pixel_candidates, view_id=view_id)
            )
        marked_image_paths = self._write_view_images(marked_images, variant="marked")
        for view, image_path in zip(views, marked_image_paths):
            view["image"] = image_path
        if self._runtime_feedback:
            vlm_input.setdefault("runtime", {})["recent_navigation_feedback"] = [
                dict(item) for item in self._runtime_feedback
            ]
        vlm_input.setdefault("metadata", {})["image_cache_dir"] = str(self._image_cache_dir)
        vlm_input["metadata"]["raw_observation_images"] = raw_image_paths
        vlm_input["metadata"]["marked_observation_images"] = marked_image_paths
        compact_projection = build_compact_memory_projection(vlm_input)
        compact_json = compact_memory_json(vlm_input)
        vlm_input["metadata"]["compact_memory_projection"] = compact_projection
        vlm_input["metadata"]["compact_memory_sha256"] = hashlib.sha256(
            compact_json.encode("utf-8")
        ).hexdigest()
        vlm_input["metadata"]["compact_memory_used"] = bool(
            compact_projection.get("enabled")
        )
        assert_policy_input_safe(vlm_input)
        self._last_vlm_input = vlm_input
        return vlm_input

    @staticmethod
    def _point_from_output(output: Dict[str, Any], image_shape: Sequence[int]) -> Optional[List[int]]:
        point = output.get("selected_image_point")
        fine_goal = output.get("fine_goal") if isinstance(output.get("fine_goal"), dict) else {}
        if point is None:
            point = fine_goal.get("point_px") or fine_goal.get("selected_image_point")
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        try:
            u = int(round(float(point[0])))
            v = int(round(float(point[1])))
        except Exception:
            return None
        h, w = int(image_shape[0]), int(image_shape[1])
        if 0 <= u < w and 0 <= v < h:
            return [u, v]
        return None

    @staticmethod
    def _reason_from_output(output: Dict[str, Any]) -> str:
        reasoning = output.get("reasoning") if isinstance(output.get("reasoning"), dict) else {}
        return str(
            reasoning.get("short_text")
            or reasoning.get("decision_reason")
            or reasoning.get("reason")
            or output.get("action")
            or "qwen_vlm decision"
        )

    @staticmethod
    def _confidence_from_output(output: Dict[str, Any], fallback: bool) -> str:
        if fallback:
            return "fallback"
        value = str(output.get("confidence") or "medium").strip().lower()
        return value if value in {"high", "medium", "low"} else "medium"

    @staticmethod
    def _go_point_guard_reason(point: Optional[Sequence[int]], image_shape: Sequence[int]) -> str:
        if point is None:
            return "go_point_missing"
        h, w = int(image_shape[0]), int(image_shape[1])
        if h <= 0 or w <= 0:
            return "go_point_invalid_image_shape"
        u, v = int(point[0]), int(point[1])
        try:
            min_v_norm = float(os.environ.get("VOCA_QWEN_GO_MIN_V_NORM", "0.45"))
        except ValueError:
            min_v_norm = 0.45
        try:
            max_v_norm = float(os.environ.get("VOCA_QWEN_GO_MAX_V_NORM", "0.98"))
        except ValueError:
            max_v_norm = 0.98
        try:
            edge_margin_norm = float(os.environ.get("VOCA_QWEN_GO_EDGE_MARGIN_NORM", "0.10"))
        except ValueError:
            edge_margin_norm = 0.10

        u_norm = (float(u) + 0.5) / float(w)
        v_norm = (float(v) + 0.5) / float(h)
        if v_norm < min_v_norm or v_norm > max_v_norm:
            return "go_point_outside_floor_band"
        if u_norm < edge_margin_norm or u_norm > 1.0 - edge_margin_norm:
            return "go_point_near_image_edge"
        return ""

    @staticmethod
    def _recent_no_progress_feedback(vlm_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        runtime = vlm_input.get("runtime") if isinstance(vlm_input.get("runtime"), dict) else {}
        feedback = runtime.get("recent_navigation_feedback") if isinstance(runtime.get("recent_navigation_feedback"), list) else []
        return [dict(item) for item in feedback if isinstance(item, dict) and item.get("event") == "go_no_progress"]

    @staticmethod
    def _is_repeated_no_progress_point(
        point: Optional[Sequence[int]],
        *,
        angle: int,
        selected_view_id: int,
        vlm_input: Dict[str, Any],
    ) -> bool:
        if point is None:
            return False
        try:
            radius_px = float(os.environ.get("VOCA_QWEN_REPEAT_POINT_RADIUS_PX", "56"))
        except ValueError:
            radius_px = 56.0
        u, v = float(point[0]), float(point[1])
        for item in reversed(QwenVLMPlanner._recent_no_progress_feedback(vlm_input)):
            failed_point = item.get("point_px")
            if not isinstance(failed_point, (list, tuple)) or len(failed_point) != 2:
                continue
            try:
                fu, fv = float(failed_point[0]), float(failed_point[1])
            except Exception:
                continue
            point_distance = float(np.hypot(u - fu, v - fv))
            if point_distance > radius_px:
                continue
            same_view = False
            same_angle = False
            try:
                same_view = int(item.get("selected_view_id")) == int(selected_view_id)
            except Exception:
                pass
            try:
                same_angle = abs(float(item.get("angle_deg")) - float(angle)) < 1e-6
            except Exception:
                pass
            if same_view or same_angle:
                return True
        return False

    @staticmethod
    def _alternative_point_after_no_progress(point: Sequence[int], image_shape: Sequence[int]) -> List[int]:
        h, w = int(image_shape[0]), int(image_shape[1])
        u = max(0, min(w - 1, int(point[0])))
        v = max(0, min(h - 1, int(point[1])))
        offset = max(4, int(round(w * 0.25)))
        candidates = []
        if u <= w // 2:
            candidates.extend([u + offset, u - offset])
        else:
            candidates.extend([u - offset, u + offset])
        candidates.extend([int(round(w * 0.25)), int(round(w * 0.75)), w // 2])
        for candidate_u in candidates:
            candidate_u = max(0, min(w - 1, int(candidate_u)))
            if abs(candidate_u - u) >= 2:
                return [candidate_u, v]
        return [max(0, min(w - 1, u + 1)), v]

    @staticmethod
    def _direction_from_output(output: Dict[str, Any], vlm_input: Dict[str, Any], num_views: int) -> int:
        views = vlm_input.get("observation", {}).get("views", []) or []
        view_id = output.get("selected_view_id")
        try:
            idx = int(view_id)
            if 0 <= idx < num_views:
                return idx
        except Exception:
            pass

        view_type = output.get("selected_view_type")
        for view in views:
            if view.get("view_type") == view_type:
                try:
                    idx = int(view.get("view_id"))
                    if 0 <= idx < num_views:
                        return idx
                except Exception:
                    continue
        return 0

    def _verify_selected_point_if_needed(
        self,
        *,
        safe_output: Dict[str, Any],
        pixel_candidate_validation: Dict[str, Any],
        pano_images: Sequence[np.ndarray],
    ) -> Dict[str, Any]:
        base_result = {
            "schema_version": "qwen_rgb_point_verification_v1",
            "enabled": self.point_verifier is not None,
            "triggered": False,
            "passed": True,
            "reason": "verification_not_required",
        }
        if str(safe_output.get("action") or "").strip().lower() != "go":
            return base_result
        if not pixel_candidate_validation.get("passed"):
            base_result["reason"] = "pixel_candidate_gate_failed_first"
            return base_result
        candidate = pixel_candidate_validation.get("candidate")
        if not isinstance(candidate, dict):
            base_result.update(
                {
                    "triggered": True,
                    "passed": False,
                    "reason": "verified_candidate_missing",
                }
            )
            return base_result
        try:
            view_id = int(candidate.get("view_id"))
            point = [int(candidate["point_px"][0]), int(candidate["point_px"][1])]
            image = np.asarray(pano_images[view_id], dtype=np.uint8)
        except Exception:
            base_result.update(
                {
                    "triggered": True,
                    "passed": False,
                    "reason": "selected_view_or_point_unavailable",
                }
            )
            return base_result

        visual_risk = selected_point_visual_risk(image, point)
        strategic_state = (
            safe_output.get("strategic_state")
            if isinstance(safe_output.get("strategic_state"), dict)
            else {}
        )
        exit_surface_required = bool(
            str(strategic_state.get("navigation_intent") or "") in _EXIT_INTENTS
        )
        backtrack_floor_verification_required = bool(
            safe_output.get("backend_memory_backtrack")
        )
        saturation_recovery_verification_required = bool(
            candidate.get("negative_memory_saturation_recovery")
        )
        base_result.update(
            {
                "candidate_ref": candidate.get("candidate_ref"),
                "point_px": point,
                "view_id": view_id,
                "visual_risk": visual_risk,
                "required_surface": (
                    "doorway_floor" if exit_surface_required else None
                ),
                "memory_backtrack_floor_verification_required": bool(
                    backtrack_floor_verification_required
                ),
                "negative_saturation_recovery_verification_required": bool(
                    saturation_recovery_verification_required
                ),
            }
        )
        if (
            not visual_risk.get("requires_verification")
            and not exit_surface_required
            and not backtrack_floor_verification_required
            and not saturation_recovery_verification_required
        ):
            base_result["reason"] = "rgb_patch_has_sufficient_visual_structure"
            return base_result
        if self.point_verifier is None:
            if (
                exit_surface_required
                or backtrack_floor_verification_required
                or saturation_recovery_verification_required
            ):
                base_result.update(
                    triggered=True,
                    passed=False,
                    reason=(
                        "doorway_floor_verifier_unavailable_fail_closed"
                        if exit_surface_required
                        else (
                            "memory_backtrack_floor_verifier_unavailable_fail_closed"
                            if backtrack_floor_verification_required
                            else "negative_saturation_recovery_verifier_unavailable_fail_closed"
                        )
                    ),
                )
            else:
                base_result["reason"] = "verifier_unavailable_observe_only"
            return base_result

        verification_image = draw_selected_point(image, point, radius=12)
        variant = "verify_selected_v{:02d}".format(view_id)
        image_path = self._write_view_images([verification_image], variant=variant)[0]
        result = self.point_verifier.verify(
            image_path=image_path,
            point_px=point,
            candidate_ref=candidate.get("candidate_ref"),
            visual_risk=visual_risk,
            required_surface=(
                "doorway_floor" if exit_surface_required else None
            ),
            current_room=(
                str(strategic_state.get("current_room") or "") or None
            ),
        )
        result = dict(result) if isinstance(result, dict) else {}
        result.setdefault("schema_version", "qwen_rgb_point_verification_v1")
        result.setdefault("enabled", True)
        result.setdefault("triggered", True)
        result.setdefault("passed", False)
        result.setdefault("candidate_ref", candidate.get("candidate_ref"))
        result.setdefault("point_px", point)
        result.setdefault("view_id", view_id)
        result.setdefault("visual_risk", visual_risk)

        intent = str(strategic_state.get("navigation_intent") or "")
        selected_exit_ref = str(
            strategic_state.get("selected_exit_ref") or ""
        ).strip()
        candidate_ref = str(candidate.get("candidate_ref") or "").strip()
        topological_ref = str(
            candidate.get("topological_candidate_ref") or ""
        ).strip()
        exit_aliases = {
            value for value in (candidate_ref, topological_ref) if value
        }
        visual_evidence = (
            candidate.get("visual_evidence")
            if isinstance(candidate.get("visual_evidence"), dict)
            else {}
        )
        try:
            visual_support = float(
                visual_evidence.get("support_score", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            visual_support = 0.0
        semantic_exit_relaxation = bool(
            not result.get("passed")
            and intent == "leave_current_room"
            and selected_exit_ref in exit_aliases
            and str(result.get("surface") or "").lower() == "floor"
            and str(result.get("confidence") or "").lower()
            in {"high", "medium"}
            and not visual_evidence.get("hard_reject")
            and visual_support >= 0.55
            and (
                bool(topological_ref)
                or self._forced_exit_failure_streak >= 2
            )
        )
        if semantic_exit_relaxation:
            result["strict_exit_verifier_passed"] = False
            result["passed"] = True
            result["semantic_exit_floor_relaxation"] = {
                "triggered": True,
                "policy": (
                    "verified_floor_plus_exit_reference_under_recovery_pressure"
                ),
                "selected_exit_ref": selected_exit_ref,
                "candidate_ref": candidate_ref,
                "topological_candidate_ref": topological_ref or None,
                "visual_support_score": round(float(visual_support), 4),
                "forced_exit_failure_streak": int(
                    self._forced_exit_failure_streak
                ),
                "original_reason": str(result.get("reason") or "")[:240],
            }
            result["reason"] = (
                "verified floor accepted as bounded exit progress after strict "
                "doorway semantics remained unresolved"
            )

        self.point_verification_calls += 1
        duration = float(result.get("latency_sec", 0.0) or 0.0)
        self.point_verification_durations.append(duration)
        backend_call_count = max(
            1,
            int(result.get("backend_call_count", 1) or 1),
        )
        backend_error_count = max(
            0,
            min(
                backend_call_count,
                int(result.get("backend_error_count", 0) or 0),
            ),
        )
        backend_durations = result.get("backend_call_durations_sec")
        backend_durations = (
            [float(value) for value in backend_durations]
            if isinstance(backend_durations, list) and backend_durations
            else [duration / float(backend_call_count)] * backend_call_count
        )
        self.llm_call_count += backend_call_count
        self.llm_durations.extend(backend_durations)
        if result.get("error"):
            self.point_verification_error_count += 1
        self.llm_error_count += backend_error_count
        self.llm_success_count += backend_call_count - backend_error_count
        if result.get("passed"):
            self.point_verification_pass_count += 1
        else:
            self.point_verification_rejection_count += 1
        return result

    def _verify_stop_if_needed(
        self,
        *,
        safe_output: Dict[str, Any],
        vlm_input: Dict[str, Any],
        current_view_only: bool = False,
    ) -> Dict[str, Any]:
        if str(safe_output.get("action") or "").lower() != "stop":
            if not self._proactive_stop_pending:
                self.consecutive_stop_confirmations = 0
                self._last_stop_confirmation_position = []
                self._last_stop_confirmation_frame_index = None
                self._last_stop_confirmation_bbox_height = 0.0
                self._last_stop_confirmation_bbox_area = 0.0
            return {"enabled": self.stop_verifier is not None, "triggered": False, "passed": False}
        if self.stop_verifier is None:
            return {
                "enabled": False,
                "triggered": False,
                "passed": False,
                "reason": "stop_verifier_unavailable",
            }
        cached_rejection = self._cached_identity_rejection_guard()
        if cached_rejection is not None:
            return cached_rejection
        metadata = vlm_input.get("metadata") if isinstance(vlm_input.get("metadata"), dict) else {}
        image_paths = metadata.get("raw_observation_images") or []
        if current_view_only and image_paths:
            image_paths = list(image_paths)[:1]
        result = self.stop_verifier.verify(target=self.object_goal, image_paths=image_paths)
        result = dict(result) if isinstance(result, dict) else {}
        result.setdefault("enabled", True)
        result.setdefault("triggered", True)
        result.setdefault("passed", False)
        single_frame_passed = bool(result.get("passed"))
        identity_critic = (
            result.get("identity_critic")
            if isinstance(result.get("identity_critic"), dict)
            else {}
        )
        identity_rejected = bool(
            identity_critic.get("triggered")
            and not identity_critic.get("passed")
        )
        if identity_rejected:
            verified_target_memory = self._recent_verified_target_memory()
            preserve_near_goal_latch = bool(
                verified_target_memory and self._near_goal_visual_latch
            )
            if not preserve_near_goal_latch:
                self._near_goal_visual_latch = {}
            self._proactive_stop_pending = False
            self.consecutive_stop_confirmations = 0
            self._last_stop_confirmation_position = []
            self._last_stop_confirmation_frame_index = None
            self._last_stop_confirmation_bbox_height = 0.0
            self._last_stop_confirmation_bbox_area = 0.0
            critic_error = bool(identity_critic.get("error"))
            rejection = {
                "target": str(self.object_goal),
                "frame_index": self._runtime_frame_index,
                "position_xyz": list(self._runtime_position_xyz[:3]),
                "target_view_index": result.get("target_view_index"),
                "target_bbox_norm": list(result.get("target_bbox_norm") or []),
                "lookalike_type": str(
                    identity_critic.get("lookalike_type") or "uncertain"
                )[:120],
                "reason": str(identity_critic.get("reason") or "")[:240],
                "critic_error": critic_error,
            }
            if verified_target_memory:
                target_lock_cycles = self._positive_int_env(
                    "VOCA_TARGET_LOCK_REACQUIRE_CYCLES",
                    6,
                )
                self._target_lock_remaining_cycles = max(
                    self._target_lock_remaining_cycles,
                    target_lock_cycles,
                )
                self._identity_rejection_cooldown_cycles = 0
                self._identity_rejection_cooldown_last_frame = None
                result[
                    "identity_rejection_preserved_verified_target_lock"
                ] = True
                result["identity_rejection_preserved_near_goal_latch"] = bool(
                    preserve_near_goal_latch
                )
                result[
                    "identity_rejection_preserved_reacquisition_attempts"
                ] = int(self._target_lock_reacquisition_attempts)
                result["verified_target_memory"] = dict(verified_target_memory)
                self.record_navigation_feedback(
                    {
                        **rejection,
                        "event": (
                            "target_lookalike_rejected_preserve_verified_lock"
                        ),
                        "verified_target_frame_index": verified_target_memory.get(
                            "frame_index"
                        ),
                    }
                )
            else:
                self._target_lock_remaining_cycles = 0
                result["identity_rejection_cleared_target_state"] = True
                cooldown_env = (
                    "VOCA_TARGET_IDENTITY_REJECTION_ERROR_COOLDOWN_CYCLES"
                    if critic_error
                    else "VOCA_TARGET_IDENTITY_REJECTION_COOLDOWN_CYCLES"
                )
                cooldown_default = 2 if critic_error else 4
                try:
                    cooldown_cycles = max(
                        1,
                        int(os.environ.get(cooldown_env, str(cooldown_default))),
                    )
                except ValueError:
                    cooldown_cycles = cooldown_default
                self._identity_rejection_cooldown_cycles = max(
                    self._identity_rejection_cooldown_cycles,
                    cooldown_cycles,
                )
                self._identity_rejection_cooldown_last_frame = None
                self._last_identity_rejection = dict(rejection)
                self.record_navigation_feedback(
                    {**rejection, "event": "target_lookalike_rejected"}
                )
        coarse_goal = (
            self._runtime_spatial_coarse_goal
            if isinstance(self._runtime_spatial_coarse_goal, dict)
            else {}
        )
        coarse_guard = {
            "schema_version": "coarse_goal_target_consistency_v1",
            "enabled": bool(coarse_goal),
            "triggered": False,
            "passed": True,
            "source": coarse_goal.get("source"),
            "uncertainty": coarse_goal.get("uncertainty"),
            "distance_m": coarse_goal.get("distance_m"),
        }
        exact_identity = bool(
            identity_critic.get("triggered")
            and identity_critic.get("passed")
            and identity_critic.get("exact_target")
        )
        target_bearing_robot_deg = None
        if exact_identity and result.get("target_evidence_passed"):
            observation = (
                vlm_input.get("observation")
                if isinstance(vlm_input.get("observation"), dict)
                else {}
            )
            views = observation.get("views") if isinstance(observation, dict) else []
            try:
                target_view_index = int(result.get("target_view_index", 0) or 0)
                target_view = (
                    views[target_view_index]
                    if isinstance(views, list)
                    and 0 <= target_view_index < len(views)
                    and isinstance(views[target_view_index], dict)
                    else {}
                )
                target_view_heading_deg = float(
                    target_view.get("relative_heading_deg", 0.0) or 0.0
                )
                target_center_x = float(result.get("target_center_x", 0.5) or 0.5)
                horizontal_fov_deg = float(
                    os.environ.get("VOCA_CAMERA_HORIZONTAL_FOV_DEG", "90.0")
                )
                target_bearing_robot_deg = _normalize_angle_deg(
                    target_view_heading_deg
                    + (target_center_x - 0.5) * horizontal_fov_deg
                )
            except (TypeError, ValueError, IndexError):
                target_bearing_robot_deg = None
        if coarse_goal and exact_identity and result.get("target_evidence_passed"):
            uncertainty = str(coarse_goal.get("uncertainty") or "medium").lower()
            stop_defaults = {"low": 0.5, "medium": 1.0, "high": 2.0}
            acquire_defaults = {"low": 4.0, "medium": 6.0, "high": 10.0}
            try:
                stop_radius_m = max(
                    0.1,
                    float(
                        os.environ.get(
                            "VOCA_COARSE_GOAL_STOP_RADIUS_{}_M".format(
                                uncertainty.upper()
                            ),
                            str(stop_defaults.get(uncertainty, 2.5)),
                        )
                    ),
                )
            except ValueError:
                stop_radius_m = stop_defaults.get(uncertainty, 2.5)
            try:
                acquire_radius_m = max(
                    stop_radius_m,
                    float(
                        os.environ.get(
                            "VOCA_COARSE_GOAL_ACQUIRE_RADIUS_{}_M".format(
                                uncertainty.upper()
                            ),
                            str(acquire_defaults.get(uncertainty, 6.0)),
                        )
                    ),
                )
            except ValueError:
                acquire_radius_m = acquire_defaults.get(uncertainty, 6.0)
            alignment_defaults = {"low": 45.0, "medium": 75.0, "high": 110.0}
            try:
                target_alignment_max_deg = max(
                    15.0,
                    min(
                        180.0,
                        float(
                            os.environ.get(
                                "VOCA_COARSE_GOAL_TARGET_ALIGNMENT_{}_DEG".format(
                                    uncertainty.upper()
                                ),
                                str(alignment_defaults.get(uncertainty, 75.0)),
                            )
                        ),
                    ),
                )
            except ValueError:
                target_alignment_max_deg = alignment_defaults.get(
                    uncertainty,
                    75.0,
                )
            try:
                coarse_distance_m = float(coarse_goal.get("distance_m"))
            except (TypeError, ValueError):
                coarse_distance_m = None
            if coarse_distance_m is not None:
                try:
                    coarse_bearing_deg = float(
                        coarse_goal.get("relative_bearing_deg")
                    )
                except (TypeError, ValueError):
                    coarse_bearing_deg = None
                target_alignment_error_deg = (
                    abs(
                        _normalize_angle_deg(
                            float(target_bearing_robot_deg)
                            - float(coarse_bearing_deg)
                        )
                    )
                    if target_bearing_robot_deg is not None
                    and coarse_bearing_deg is not None
                    else None
                )
                target_direction_consistent = bool(
                    target_alignment_error_deg is None
                    or target_alignment_error_deg
                    <= target_alignment_max_deg + 1e-6
                )
                coarse_guard.update(
                    triggered=True,
                    stop_radius_m=round(float(stop_radius_m), 4),
                    acquire_radius_m=round(float(acquire_radius_m), 4),
                    target_bearing_robot_deg=(
                        round(float(target_bearing_robot_deg), 4)
                        if target_bearing_robot_deg is not None
                        else None
                    ),
                    target_alignment_error_deg=(
                        round(float(target_alignment_error_deg), 4)
                        if target_alignment_error_deg is not None
                        else None
                    ),
                    target_alignment_max_deg=round(
                        float(target_alignment_max_deg),
                        4,
                    ),
                    target_direction_consistent=target_direction_consistent,
                    stop_consistent=bool(
                        coarse_distance_m <= stop_radius_m + 1e-6
                    ),
                    acquisition_consistent=bool(
                        coarse_distance_m <= acquire_radius_m + 1e-6
                        and (
                            coarse_distance_m <= stop_radius_m + 1e-6
                            or target_direction_consistent
                        )
                    ),
                )
                distance_inconsistent = bool(
                    coarse_distance_m > acquire_radius_m + 1e-6
                )
                direction_inconsistent = bool(
                    coarse_distance_m > stop_radius_m + 1e-6
                    and not target_direction_consistent
                )
                if distance_inconsistent or direction_inconsistent:
                    coarse_guard["passed"] = False
                    coarse_guard["classification"] = (
                        "same_category_instance_outside_coarse_goal_region"
                        if distance_inconsistent
                        else "same_category_instance_directionally_inconsistent"
                    )
                    result["visual_target_evidence_passed_before_coarse_guard"] = bool(
                        result.get("target_evidence_passed")
                    )
                    result["target_evidence_passed"] = False
                    result["passed"] = False
                    single_frame_passed = False
                    self._near_goal_visual_latch = {}
                    self._proactive_stop_pending = False
                    self.consecutive_stop_confirmations = 0
                    self._last_stop_confirmation_position = []
                    self._last_stop_confirmation_frame_index = None
                    self._target_lock_remaining_cycles = 0
                    result["reason"] = (
                        "exact category object is outside the supplied coarse-goal region"
                        if distance_inconsistent
                        else "exact category object points away from the supplied coarse goal"
                    )
                    self.record_navigation_feedback(
                        {
                            "event": (
                                "coarse_inconsistent_target_instance"
                                if distance_inconsistent
                                else "coarse_direction_inconsistent_target_instance"
                            ),
                            "target": str(self.object_goal),
                            "frame_index": self._runtime_frame_index,
                            "coarse_goal_distance_m": round(
                                float(coarse_distance_m),
                                4,
                            ),
                            "acquire_radius_m": round(
                                float(acquire_radius_m),
                                4,
                            ),
                            "target_bearing_robot_deg": (
                                round(float(target_bearing_robot_deg), 4)
                                if target_bearing_robot_deg is not None
                                else None
                            ),
                            "target_alignment_error_deg": (
                                round(float(target_alignment_error_deg), 4)
                                if target_alignment_error_deg is not None
                                else None
                            ),
                        }
                    )
                elif coarse_distance_m > stop_radius_m + 1e-6:
                    coarse_guard["passed"] = False
                    coarse_guard["classification"] = (
                        "target_acquisition_allowed_stop_deferred"
                    )
                    result["passed"] = False
                    single_frame_passed = False
                    result["reason"] = (
                        "target matches coarse direction but coarse region remains distant"
                    )
                else:
                    coarse_guard["classification"] = "inside_coarse_goal_stop_region"
        result["coarse_goal_consistency_guard"] = coarse_guard
        try:
            min_confirmation_translation_m = max(
                0.0,
                float(os.environ.get("VOCA_STOP_CONFIRM_TRANSLATION_M", "0.25")),
            )
        except ValueError:
            min_confirmation_translation_m = 0.25
        try:
            min_confirmation_frame_gap = max(
                1,
                int(os.environ.get("VOCA_STOP_CONFIRM_MIN_FRAME_GAP", "1")),
            )
        except ValueError:
            min_confirmation_frame_gap = 1
        try:
            strong_target_height = max(
                0.0,
                min(
                    1.0,
                    float(
                        os.environ.get(
                            "VOCA_STOP_STRONG_TARGET_HEIGHT_FRAC",
                            "0.20",
                        )
                    ),
                ),
            )
        except ValueError:
            strong_target_height = 0.20
        try:
            strong_target_area = max(
                0.0,
                min(
                    1.0,
                    float(
                        os.environ.get(
                            "VOCA_STOP_STRONG_TARGET_AREA_FRAC",
                            "0.03",
                        )
                    ),
                ),
            )
        except ValueError:
            strong_target_area = 0.03
        strong_visual_proximity = bool(
            float(result.get("target_bbox_height", 0.0) or 0.0) + 1e-6
            >= strong_target_height
            and float(result.get("target_bbox_area", 0.0) or 0.0) + 1e-6
            >= strong_target_area
        )
        try:
            max_confirmation_frame_gap = max(
                1,
                int(os.environ.get("VOCA_STOP_CONFIRM_MAX_FRAME_GAP", "40")),
            )
        except ValueError:
            max_confirmation_frame_gap = 40
        confirmation_restarted_reason = None
        if (
            self.consecutive_stop_confirmations > 0
            and self._last_stop_confirmation_frame_index is not None
            and self._runtime_frame_index is not None
            and int(self._runtime_frame_index)
            - int(self._last_stop_confirmation_frame_index)
            > max_confirmation_frame_gap
        ):
            self.consecutive_stop_confirmations = 0
            self._last_stop_confirmation_position = []
            self._last_stop_confirmation_frame_index = None
            self._last_stop_confirmation_bbox_height = 0.0
            self._last_stop_confirmation_bbox_area = 0.0
            self._proactive_stop_pending = False
            confirmation_restarted_reason = "confirmation_ttl_expired"
        try:
            min_target_approach_translation_m = max(
                0.0,
                float(
                    os.environ.get(
                        "VOCA_STOP_TARGET_APPROACH_CONFIRM_TRANSLATION_M",
                        "0.10",
                    )
                ),
            )
        except ValueError:
            min_target_approach_translation_m = 0.10
        target_approach = dict(self._last_target_approach_execution)
        try:
            target_approach_frame = int(target_approach.get("frame_index"))
        except (TypeError, ValueError):
            target_approach_frame = -1
        target_approach_after_confirmation = bool(
            self.consecutive_stop_confirmations > 0
            and self._last_stop_confirmation_frame_index is not None
            and target_approach_frame > int(self._last_stop_confirmation_frame_index)
        )
        target_approach_translation = float(
            target_approach.get("translation_m", 0.0) or 0.0
        )
        target_approach_terminal = bool(
            target_approach.get("stop_action_seen")
            or
            target_approach.get("controller_reached_waypoint")
            or (
                target_approach.get("no_progress")
                and int(target_approach.get("collision_count", 0) or 0) > 0
            )
        )
        target_approach_progress_satisfied = bool(
            target_approach.get("execution_success")
            and int(target_approach.get("collision_count", 0) or 0) == 0
            and target_approach_translation + 1e-6
            >= min_target_approach_translation_m
        )
        translation_only_allowed = _env_bool(
            "VOCA_STOP_ALLOW_TRANSLATION_ONLY_CONFIRM",
            False,
        )
        previous_bbox_height = float(self._last_stop_confirmation_bbox_height)
        previous_bbox_area = float(self._last_stop_confirmation_bbox_area)
        current_bbox_height = float(result.get("target_bbox_height", 0.0) or 0.0)
        current_bbox_area = float(result.get("target_bbox_area", 0.0) or 0.0)
        try:
            near_latch_height = max(
                0.0,
                min(
                    1.0,
                    float(
                        os.environ.get(
                            "VOCA_STOP_NEAR_LATCH_MIN_BBOX_HEIGHT",
                            "0.40",
                        )
                    ),
                ),
            )
        except ValueError:
            near_latch_height = 0.40
        try:
            near_latch_area = max(
                0.0,
                min(
                    1.0,
                    float(
                        os.environ.get(
                            "VOCA_STOP_NEAR_LATCH_MIN_BBOX_AREA",
                            "0.08",
                        )
                    ),
                ),
            )
        except ValueError:
            near_latch_area = 0.08
        try:
            near_latch_ttl_frames = max(
                1,
                int(os.environ.get("VOCA_STOP_NEAR_LATCH_TTL_FRAMES", "80")),
            )
        except ValueError:
            near_latch_ttl_frames = 80
        try:
            near_latch_radius_m = max(
                0.1,
                float(os.environ.get("VOCA_STOP_NEAR_LATCH_RADIUS_M", "1.0")),
            )
        except ValueError:
            near_latch_radius_m = 1.0

        near_latch = dict(self._near_goal_visual_latch)
        try:
            near_latch_frame = int(near_latch.get("frame_index"))
        except (TypeError, ValueError):
            near_latch_frame = -1
        try:
            near_latch_last_evidence_frame = int(
                near_latch.get("last_evidence_frame_index", near_latch_frame)
            )
        except (TypeError, ValueError):
            near_latch_last_evidence_frame = near_latch_frame
        near_latch_frame_gap = None
        near_latch_origin_frame_gap = None
        if self._runtime_frame_index is not None and near_latch_last_evidence_frame >= 0:
            near_latch_frame_gap = (
                int(self._runtime_frame_index) - near_latch_last_evidence_frame
            )
        if self._runtime_frame_index is not None and near_latch_frame >= 0:
            near_latch_origin_frame_gap = int(self._runtime_frame_index) - near_latch_frame
        near_latch_displacement_m = None
        near_latch_position = near_latch.get("position_xyz") or []
        if len(near_latch_position) >= 3 and len(self._runtime_position_xyz) >= 3:
            near_latch_displacement_m = float(
                np.linalg.norm(
                    np.asarray(self._runtime_position_xyz[:3], dtype=np.float64)
                    - np.asarray(near_latch_position[:3], dtype=np.float64)
                )
            )
        near_latch_active = bool(
            near_latch_frame >= 0
            and near_latch_frame_gap is not None
            and 0 <= near_latch_frame_gap <= near_latch_ttl_frames
            and near_latch_displacement_m is not None
            and near_latch_displacement_m <= near_latch_radius_m + 1e-6
        )
        if near_latch and not near_latch_active:
            self._near_goal_visual_latch = {}
            near_latch = {}
        near_latch_evidence = bool(
            single_frame_passed
            and str(result.get("confidence") or "").lower() == "high"
            and (
                current_bbox_height + 1e-6 >= near_latch_height
                or current_bbox_area + 1e-6 >= near_latch_area
            )
        )
        try:
            near_latch_single_height = max(
                near_latch_height,
                float(
                    os.environ.get(
                        "VOCA_STOP_NEAR_LATCH_SINGLE_APPROACH_HEIGHT",
                        "0.45",
                    )
                ),
            )
        except ValueError:
            near_latch_single_height = max(near_latch_height, 0.45)
        try:
            near_latch_single_area = max(
                near_latch_area,
                float(
                    os.environ.get(
                        "VOCA_STOP_NEAR_LATCH_SINGLE_APPROACH_AREA",
                        "0.10",
                    )
                ),
            )
        except ValueError:
            near_latch_single_area = max(near_latch_area, 0.10)
        near_latch_best_height = max(
            float(near_latch.get("max_bbox_height", 0.0) or 0.0),
            current_bbox_height if near_latch_evidence else 0.0,
        )
        near_latch_best_area = max(
            float(near_latch.get("max_bbox_area", 0.0) or 0.0),
            current_bbox_area if near_latch_evidence else 0.0,
        )
        near_latch_single_approach_evidence = bool(
            near_latch_best_height + 1e-6 >= near_latch_single_height
            and near_latch_best_area + 1e-6 >= near_latch_single_area
        )
        near_latch_approaches_required = (
            1 if near_latch_single_approach_evidence else 2
        )
        near_latch_approach_count = int(
            near_latch.get("approach_count", 0) or 0
        )
        try:
            near_latch_last_approach_frame = int(
                near_latch.get("last_approach_frame_index")
            )
        except (TypeError, ValueError):
            near_latch_last_approach_frame = -1
        near_latch_approach_after_evidence = bool(
            near_latch_active
            and near_latch_approach_count > 0
            and near_latch_last_approach_frame > near_latch_frame
        )
        near_latch_post_approach_observation = bool(
            self._runtime_frame_index is not None
            and near_latch_last_approach_frame >= 0
            and int(self._runtime_frame_index) >= near_latch_last_approach_frame
        )
        try:
            min_scale_height_ratio = max(
                0.0,
                min(
                    1.0,
                    float(
                        os.environ.get(
                            "VOCA_STOP_SCALE_MIN_HEIGHT_RATIO",
                            "0.85",
                        )
                    ),
                ),
            )
        except ValueError:
            min_scale_height_ratio = 0.85
        try:
            min_scale_area_ratio = max(
                0.0,
                min(
                    1.0,
                    float(
                        os.environ.get(
                            "VOCA_STOP_SCALE_MIN_AREA_RATIO",
                            "0.75",
                        )
                    ),
                ),
            )
        except ValueError:
            min_scale_area_ratio = 0.75
        scale_consistent = bool(
            self.consecutive_stop_confirmations == 0
            or (
                current_bbox_height + 1e-6
                >= previous_bbox_height * min_scale_height_ratio
                and current_bbox_area + 1e-6
                >= previous_bbox_area * min_scale_area_ratio
            )
        )
        base_latch_approaches_required = int(near_latch_approaches_required)
        target_name = str(self.object_goal or "").strip().lower().replace("_", " ")
        category_min_approaches = 0
        if target_name in {"plant", "houseplant"}:
            try:
                category_min_approaches = max(
                    1,
                    int(
                        os.environ.get(
                            "VOCA_STOP_NEAR_LATCH_PLANT_MIN_APPROACHES",
                            "4",
                        )
                    ),
                )
            except ValueError:
                category_min_approaches = 4
        if not scale_consistent:
            base_latch_approaches_required += 1
        near_latch_approaches_required = max(
            base_latch_approaches_required,
            category_min_approaches,
        )
        terminal_bypass_allowed = bool(category_min_approaches <= 0)
        near_latch_approach_requirement_satisfied = bool(
            near_latch_approach_count >= near_latch_approaches_required
            or (target_approach_terminal and terminal_bypass_allowed)
        )
        scale_drop_bypass_allowed = bool(
            target_approach_terminal and category_min_approaches <= 0
        )
        near_latch_scale_guard_satisfied = bool(
            scale_consistent or scale_drop_bypass_allowed
        )
        near_latch_confirmation_satisfied = bool(
            single_frame_passed
            and near_latch_approach_after_evidence
            and near_latch_approach_requirement_satisfied
            and near_latch_post_approach_observation
            and near_latch_scale_guard_satisfied
        )
        post_confirmation_compact_target_evidence = bool(
            current_bbox_height + 1e-6 >= near_latch_height
            and current_bbox_area + 1e-6 >= near_latch_area
        )
        target_approach_execution_satisfied = bool(
            near_latch_confirmation_satisfied
            or (
                target_approach_after_confirmation
                and strong_visual_proximity
                and scale_consistent
                and (
                    target_approach_terminal
                    or (
                        target_approach_progress_satisfied
                        and post_confirmation_compact_target_evidence
                    )
                    or (
                        translation_only_allowed
                        and target_approach_translation + 1e-6
                        >= min_target_approach_translation_m
                    )
                )
            )
        )
        confirmation_translation_m = None
        confirmation_frame_gap = None
        motion_satisfied = self.consecutive_stop_confirmations == 0
        frame_change_satisfied = self.consecutive_stop_confirmations == 0
        if self.consecutive_stop_confirmations > 0:
            if len(self._last_stop_confirmation_position) >= 3 and len(self._runtime_position_xyz) >= 3:
                confirmation_translation_m = float(
                    np.linalg.norm(
                        np.asarray(self._runtime_position_xyz[:3], dtype=np.float64)
                        - np.asarray(self._last_stop_confirmation_position[:3], dtype=np.float64)
                    )
                )
                motion_satisfied = confirmation_translation_m >= min_confirmation_translation_m
            else:
                motion_satisfied = False
            if (
                self._last_stop_confirmation_frame_index is not None
                and self._runtime_frame_index is not None
            ):
                confirmation_frame_gap = int(self._runtime_frame_index) - int(
                    self._last_stop_confirmation_frame_index
                )
                frame_change_satisfied = (
                    confirmation_frame_gap >= min_confirmation_frame_gap
                )
            else:
                frame_change_satisfied = False
        independent_observation_satisfied = bool(
            self.consecutive_stop_confirmations == 0
            or (
                frame_change_satisfied
                and target_approach_execution_satisfied
            )
        )
        if single_frame_passed:
            if independent_observation_satisfied:
                self.consecutive_stop_confirmations += 1
                self._last_stop_confirmation_position = list(self._runtime_position_xyz[:3])
                self._last_stop_confirmation_frame_index = self._runtime_frame_index
                self._last_stop_confirmation_bbox_height = current_bbox_height
                self._last_stop_confirmation_bbox_area = current_bbox_area
        else:
            confirmation_grace = bool(
                self.consecutive_stop_confirmations > 0
                and self._target_lock_remaining_cycles > 0
            )
            result["confirmation_grace_preserved"] = confirmation_grace
            if not confirmation_grace:
                self.consecutive_stop_confirmations = 0
                self._last_stop_confirmation_position = []
                self._last_stop_confirmation_frame_index = None
                self._last_stop_confirmation_bbox_height = 0.0
                self._last_stop_confirmation_bbox_area = 0.0
                self._proactive_stop_pending = False
        near_latch_updated = False
        if near_latch_evidence:
            if near_latch_active:
                refreshed_latch = dict(near_latch)
                refreshed_latch["max_bbox_height"] = near_latch_best_height
                refreshed_latch["max_bbox_area"] = near_latch_best_area
                refreshed_latch["last_evidence_frame_index"] = self._runtime_frame_index
                refreshed_latch["position_xyz"] = list(
                    self._runtime_position_xyz[:3]
                )
                self._near_goal_visual_latch = refreshed_latch
            else:
                self._near_goal_visual_latch = {
                    "frame_index": self._runtime_frame_index,
                    "last_evidence_frame_index": self._runtime_frame_index,
                    "position_xyz": list(self._runtime_position_xyz[:3]),
                    "max_bbox_height": current_bbox_height,
                    "max_bbox_area": current_bbox_area,
                    "confidence": str(result.get("confidence") or ""),
                    "approach_count": 0,
                }
            near_latch_updated = True
        consecutive_confirmation_passed = bool(
            single_frame_passed
            and self.consecutive_stop_confirmations
            >= self.stop_confirmations_required
        )
        temporal_passed = bool(
            consecutive_confirmation_passed
            or near_latch_confirmation_satisfied
        )
        result["single_frame_passed"] = single_frame_passed
        result["consecutive_confirmations"] = int(self.consecutive_stop_confirmations)
        result["confirmations_required"] = int(self.stop_confirmations_required)
        result["consecutive_confirmation_passed"] = bool(
            consecutive_confirmation_passed
        )
        result["temporal_confirmation_source"] = (
            "consecutive_verified_observations"
            if consecutive_confirmation_passed
            else (
                "near_goal_visual_latch"
                if near_latch_confirmation_satisfied
                else "insufficient"
            )
        )
        result["effective_confirmations"] = int(
            self.stop_confirmations_required
            if temporal_passed
            else self.consecutive_stop_confirmations
        )
        result["confirmation_motion_satisfied"] = bool(motion_satisfied)
        result["confirmation_frame_change_satisfied"] = bool(
            frame_change_satisfied
        )
        result["confirmation_independence_satisfied"] = bool(
            independent_observation_satisfied
        )
        result["strong_visual_proximity_satisfied"] = bool(
            strong_visual_proximity
        )
        result["strong_visual_proximity_threshold"] = {
            "min_bbox_height": strong_target_height,
            "min_bbox_area": strong_target_area,
        }
        result["target_approach_after_confirmation"] = bool(
            target_approach_after_confirmation
        )
        result["target_approach_execution_satisfied"] = bool(
            target_approach_execution_satisfied
        )
        result["target_approach_terminal_satisfied"] = bool(
            target_approach_terminal
        )
        result["target_approach_progress_satisfied"] = bool(
            target_approach_progress_satisfied
        )
        result["post_confirmation_compact_target_evidence"] = bool(
            post_confirmation_compact_target_evidence
        )
        result["target_scale_consistent"] = bool(scale_consistent)
        result["target_scale_consistency_threshold"] = {
            "min_height_ratio": float(min_scale_height_ratio),
            "min_area_ratio": float(min_scale_area_ratio),
        }
        result["near_goal_visual_latch_evidence"] = bool(near_latch_evidence)
        result["near_goal_visual_latch_active"] = bool(
            near_latch_active or near_latch_updated
        )
        result["near_goal_visual_latch_updated"] = bool(near_latch_updated)
        result["near_goal_visual_latch_confirmation_satisfied"] = bool(
            near_latch_confirmation_satisfied
        )
        result["near_goal_visual_latch_scale_guard_satisfied"] = bool(
            near_latch_scale_guard_satisfied
        )
        result["near_goal_visual_latch_scale_drop_bypass_allowed"] = bool(
            scale_drop_bypass_allowed
        )
        result["near_goal_visual_latch_base_approaches_required"] = int(
            1 if near_latch_single_approach_evidence else 2
        )
        result["near_goal_visual_latch_category_min_approaches"] = int(
            category_min_approaches
        )
        result["near_goal_visual_latch_scale_drop_extra_approach"] = int(
            not scale_consistent
        )
        result["near_goal_visual_latch_threshold"] = {
            "min_bbox_height": float(near_latch_height),
            "min_bbox_area": float(near_latch_area),
            "combination": "height_or_area",
            "single_approach_min_bbox_height": float(
                near_latch_single_height
            ),
            "single_approach_min_bbox_area": float(near_latch_single_area),
            "plant_min_approaches": int(category_min_approaches),
            "ttl_frames": int(near_latch_ttl_frames),
            "radius_m": float(near_latch_radius_m),
        }
        result["near_goal_visual_latch_frame_gap"] = near_latch_frame_gap
        result["near_goal_visual_latch_origin_frame_gap"] = (
            near_latch_origin_frame_gap
        )
        result["near_goal_visual_latch_displacement_m"] = (
            round(float(near_latch_displacement_m), 6)
            if near_latch_displacement_m is not None
            else None
        )
        result["near_goal_visual_latch_approach_after_evidence"] = bool(
            near_latch_approach_after_evidence
        )
        result["near_goal_visual_latch_approach_count"] = int(
            near_latch_approach_count
        )
        result["near_goal_visual_latch_approaches_required"] = int(
            near_latch_approaches_required
        )
        result["near_goal_visual_latch_terminal_bypass_allowed"] = bool(
            terminal_bypass_allowed
        )
        result["near_goal_visual_latch_single_approach_evidence"] = bool(
            near_latch_single_approach_evidence
        )
        result["previous_confirmation_bbox_height"] = previous_bbox_height
        result["previous_confirmation_bbox_area"] = previous_bbox_area
        result["translation_only_confirmation_allowed"] = bool(
            translation_only_allowed
        )
        result["target_approach_execution"] = target_approach
        result["min_target_approach_translation_m"] = float(
            min_target_approach_translation_m
        )
        result["confirmation_frame_gap"] = confirmation_frame_gap
        result["min_confirmation_frame_gap"] = int(min_confirmation_frame_gap)
        result["max_confirmation_frame_gap"] = int(max_confirmation_frame_gap)
        result["confirmation_restarted_reason"] = confirmation_restarted_reason
        result["confirmation_translation_m"] = (
            round(float(confirmation_translation_m), 6)
            if confirmation_translation_m is not None
            else None
        )
        result["min_confirmation_translation_m"] = float(
            min_confirmation_translation_m
        )
        result["passed"] = temporal_passed
        if (
            single_frame_passed
            and (near_latch_active or near_latch_evidence)
            and not near_latch_scale_guard_satisfied
        ):
            result["reason"] = (
                "target scale decreased after approach; additional approach or "
                "re-observation required"
            )
        elif single_frame_passed and not independent_observation_satisfied:
            result["reason"] = (
                "target evidence repeated without post-confirmation target-approach execution"
            )
        elif single_frame_passed and not temporal_passed:
            result["reason"] = "target evidence verified once; temporal confirmation required"
        backend_call_count = max(1, int(result.get("backend_call_count", 1) or 1))
        backend_error_count = max(0, int(result.get("backend_error_count", 0) or 0))
        self.stop_verification_calls += backend_call_count
        duration = float(result.get("latency_sec", 0.0) or 0.0)
        self.stop_verification_durations.append(duration)
        self.llm_call_count += backend_call_count
        self.llm_durations.append(duration)
        if result.get("error"):
            self.stop_verification_error_count += max(1, backend_error_count)
            self.llm_error_count += max(1, backend_error_count)
        self.llm_success_count += max(0, backend_call_count - backend_error_count)
        if result.get("passed"):
            self.stop_verification_pass_count += 1
        else:
            self.stop_verification_rejection_count += 1
        return result

    @staticmethod
    def _target_lock_cycle_limit() -> int:
        try:
            return max(
                1,
                int(os.environ.get("VOCA_TARGET_LOCK_MAX_CYCLES", "12")),
            )
        except ValueError:
            return 12

    @staticmethod
    def _target_lock_exhaustion_cooldown() -> int:
        try:
            return max(
                1,
                int(
                    os.environ.get(
                        "VOCA_TARGET_LOCK_EXHAUSTION_COOLDOWN_CYCLES",
                        "4",
                    )
                ),
            )
        except ValueError:
            return 4

    def _reset_target_lock_session(self, *, clear_cooldown: bool = False) -> None:
        self._target_lock_remaining_cycles = 0
        self._target_lock_reacquisition_attempts = 0
        self._target_anchor_reacquisition_count = 0
        self._target_lock_session_cycles = 0
        if clear_cooldown:
            self._target_lock_cooldown_cycles = 0

    def _exhaust_target_lock_session(self, *, reason: str) -> None:
        consumed_cycles = int(self._target_lock_session_cycles)
        cycle_limit = int(self._target_lock_cycle_limit())
        self._target_lock_exhaustion_count += 1
        self._target_lock_cooldown_cycles = max(
            self._target_lock_cooldown_cycles,
            self._target_lock_exhaustion_cooldown(),
        )
        self._reset_target_lock_session(clear_cooldown=False)
        self._near_goal_visual_latch = {}
        self._proactive_stop_pending = False
        self.consecutive_stop_confirmations = 0
        self._last_stop_confirmation_position = []
        self._last_stop_confirmation_frame_index = None
        self._last_stop_confirmation_bbox_height = 0.0
        self._last_stop_confirmation_bbox_area = 0.0
        feedback = {
            "event": "target_lock_budget_exhausted",
            "frame_index": self._runtime_frame_index,
            "reason": str(reason or "target_lock_cycle_limit"),
            "consumed_cycles": consumed_cycles,
            "cycle_limit": cycle_limit,
            "cooldown_cycles": int(self._target_lock_cooldown_cycles),
        }
        self.record_navigation_feedback(feedback)
        if self.memory_sidecar is not None:
            self.memory_sidecar.update_supervisor_mode(
                "escape_deadlock",
                reason="target_lock_budget_exhausted",
            )

    def _verified_target_anchor_reacquisition_output(
        self,
    ) -> Optional[Dict[str, Any]]:
        evidence = self._recent_verified_target_memory()
        if not evidence or len(self._runtime_position_xyz) < 3:
            return None
        try:
            max_rotations = max(
                0,
                int(
                    os.environ.get(
                        "VOCA_TARGET_ANCHOR_REACQUIRE_MAX_ROTATIONS",
                        "2",
                    )
                ),
            )
        except ValueError:
            max_rotations = 2
        if self._target_anchor_reacquisition_count >= max_rotations:
            return None

        evidence_position = evidence.get("position_xyz") or []
        try:
            displacement_m = float(evidence.get("displacement_m") or 0.0)
        except (TypeError, ValueError):
            displacement_m = 0.0
        try:
            min_pose_bearing_displacement_m = max(
                0.0,
                float(
                    os.environ.get(
                        "VOCA_TARGET_ANCHOR_POSE_BEARING_MIN_DISPLACEMENT_M",
                        "0.10",
                    )
                ),
            )
        except ValueError:
            min_pose_bearing_displacement_m = 0.10

        bearing_source = ""
        bearing_deg: Optional[float] = None
        try:
            target_bearing_world_rad = float(
                evidence.get("target_bearing_world_rad")
            )
            bearing_deg = _normalize_angle_deg(
                math.degrees(
                    target_bearing_world_rad
                    - float(self._runtime_heading_rad)
                )
            )
            bearing_source = "verified_target_visual_bearing"
        except (TypeError, ValueError):
            if (
                len(evidence_position) >= 3
                and displacement_m + 1e-6 >= min_pose_bearing_displacement_m
            ):
                dx_world = float(evidence_position[0]) - float(
                    self._runtime_position_xyz[0]
                )
                dz_world = float(evidence_position[2]) - float(
                    self._runtime_position_xyz[2]
                )
                heading = float(self._runtime_heading_rad)
                c = math.cos(heading)
                s = math.sin(heading)
                dx_robot = c * dx_world + s * dz_world
                dy_robot = -s * dx_world + c * dz_world
                if math.hypot(dx_robot, dy_robot) > 1e-6:
                    bearing_deg = _normalize_angle_deg(
                        math.degrees(math.atan2(dy_robot, dx_robot))
                    )
                    bearing_source = "verified_target_observation_pose_fallback"
        if bearing_deg is None:
            return None

        try:
            min_turn_deg = max(
                1.0,
                float(
                    os.environ.get(
                        "VOCA_TARGET_ANCHOR_REACQUIRE_MIN_TURN_DEG",
                        "15.0",
                    )
                ),
            )
        except ValueError:
            min_turn_deg = 15.0
        if abs(float(bearing_deg)) < min_turn_deg:
            return None
        turn_steps = max(1, min(6, int(round(abs(float(bearing_deg)) / 30.0))))
        commanded_yaw_deg = float(turn_steps * 30) * (
            1.0 if float(bearing_deg) > 0.0 else -1.0
        )
        output = self._nav_modules.schema.make_rotate_output(
            commanded_yaw_deg,
            reason="R07_VERIFIED_TARGET_ANCHOR_REACQUIRE",
            confidence="medium",
        )
        output["backend_verified_target_anchor_reacquisition"] = {
            "schema_version": "verified_target_anchor_reacquisition_v1",
            "node_id": evidence.get("node_id"),
            "target_object": evidence.get("target_object"),
            "evidence_frame_index": evidence.get("frame_index"),
            "frame_gap": evidence.get("frame_gap"),
            "displacement_m": round(float(displacement_m), 6),
            "bearing_source": bearing_source,
            "bearing_deg_robot": round(float(bearing_deg), 3),
            "commanded_yaw_deg": commanded_yaw_deg,
            "rotation_index": int(self._target_anchor_reacquisition_count + 1),
            "max_rotations": int(max_rotations),
        }
        reasoning = (
            output.get("reasoning")
            if isinstance(output.get("reasoning"), dict)
            else {}
        )
        reasoning["short_text"] = (
            "rotate toward the last independently verified target anchor"
        )
        output["reasoning"] = reasoning
        return output

    def _proactive_stop_decision(
        self,
        *,
        vlm_input: Dict[str, Any],
        image_shape: Sequence[int],
        angles: Sequence[int],
        call_type: str,
    ) -> Optional[Dict[str, Any]]:
        if not _env_bool("VOCA_QWEN_PROACTIVE_STOP_VERIFY", True):
            return None
        if self.stop_verifier is None:
            return None
        if self._runtime_frame_index is None or int(self._runtime_frame_index) <= 0:
            return None
        if self._target_lock_cooldown_cycles > 0:
            self._target_lock_cooldown_cycles -= 1
            return None

        had_target_lock = bool(
            self._target_lock_remaining_cycles > 0
            or self._proactive_stop_pending
            or self._near_goal_visual_latch
            or self._target_lock_session_cycles > 0
        )

        verification = self._verify_stop_if_needed(
            safe_output={"action": "stop"},
            vlm_input=vlm_input,
        )
        self._last_proactive_stop_probe = dict(verification)
        target_evidence_passed = bool(
            verification.get("single_frame_passed")
            or verification.get("target_evidence_passed")
        )
        if had_target_lock or target_evidence_passed:
            self._target_lock_session_cycles += 1
        verification["target_lock_session"] = {
            "active": bool(had_target_lock or target_evidence_passed),
            "cycle": int(self._target_lock_session_cycles),
            "cycle_limit": int(self._target_lock_cycle_limit()),
            "reacquisition_attempts": int(
                self._target_lock_reacquisition_attempts
            ),
            "exhaustion_count": int(self._target_lock_exhaustion_count),
        }
        if (
            self._target_lock_session_cycles > self._target_lock_cycle_limit()
            and not verification.get("passed")
        ):
            exhausted = dict(verification["target_lock_session"])
            exhausted["exhausted"] = True
            self._last_proactive_stop_probe = dict(verification)
            self._exhaust_target_lock_session(
                reason="target verification/approach did not terminate within budget",
            )
            output = self._nav_modules.schema.make_rotate_output(
                90.0 if self._target_lock_exhaustion_count % 2 else -90.0,
                reason="R08_TARGET_LOCK_BUDGET_EXHAUSTED",
                confidence="low",
            )
            reasoning = (
                output.get("reasoning")
                if isinstance(output.get("reasoning"), dict)
                else {}
            )
            reasoning["short_text"] = (
                "target reacquisition budget exhausted; reposition laterally"
            )
            output["reasoning"] = reasoning
            output["memory_ops"] = []
            decision = self._vlm_output_to_decision(
                raw_output={
                    **dict(output),
                    "source": "target_lock_budget_exhausted",
                },
                safe_output=output,
                warnings=["target_lock_budget_exhausted"],
                vlm_input=vlm_input,
                image_shape=image_shape,
                angles=angles,
                call_type=call_type,
                stop_verification=verification,
            )
            decision["proactive_stop_probe"] = True
            decision["target_lock_budget_exhausted"] = exhausted
            decision["proactive_stop_probe_result"] = dict(verification)
            return decision
        if target_evidence_passed:
            if not had_target_lock:
                self._target_lock_reacquisition_attempts = 0
                self._target_anchor_reacquisition_count = 0
            if self.memory_sidecar is not None:
                image_ref = None
                observation = (
                    vlm_input.get("observation")
                    if isinstance(vlm_input.get("observation"), dict)
                    else {}
                )
                views = [
                    item
                    for item in observation.get("views", []) or []
                    if isinstance(item, dict)
                ]
                target_view_relative_heading_deg = 0.0
                try:
                    target_view_id = int(verification.get("target_view_index"))
                except (TypeError, ValueError):
                    target_view_id = None
                if target_view_id is not None:
                    for view in views:
                        try:
                            if int(view.get("view_id", -1)) == target_view_id:
                                image_ref = view.get("image")
                                target_view_relative_heading_deg = float(
                                    view.get("relative_heading_deg", 0.0) or 0.0
                                )
                                break
                        except (TypeError, ValueError):
                            continue
                    if image_ref is None and 0 <= target_view_id < len(views):
                        image_ref = views[target_view_id].get("image")
                        target_view_relative_heading_deg = float(
                            views[target_view_id].get(
                                "relative_heading_deg",
                                0.0,
                            )
                            or 0.0
                        )
                try:
                    target_center_x = max(
                        0.0,
                        min(1.0, float(verification.get("target_center_x"))),
                    )
                except (TypeError, ValueError):
                    target_center_x = 0.5
                try:
                    camera_hfov_deg = max(
                        1.0,
                        float(
                            os.environ.get(
                                "VOCA_CAMERA_HORIZONTAL_FOV_DEG",
                                "90.0",
                            )
                        ),
                    )
                except ValueError:
                    camera_hfov_deg = 90.0
                target_bearing_robot_deg = _normalize_angle_deg(
                    target_view_relative_heading_deg
                    + (target_center_x - 0.5) * camera_hfov_deg
                )
                target_bearing_world_rad = float(
                    self._runtime_heading_rad
                    + math.radians(target_bearing_robot_deg)
                )
                verification["target_view_relative_heading_deg"] = round(
                    float(target_view_relative_heading_deg),
                    3,
                )
                verification["target_bearing_robot_deg"] = round(
                    float(target_bearing_robot_deg),
                    3,
                )
                verification["target_bearing_world_rad"] = round(
                    float(target_bearing_world_rad),
                    6,
                )
                memory_update = self.memory_sidecar.record_verified_target_evidence(
                    verification,
                    frame_index=int(self._runtime_frame_index or 0),
                    image_ref=str(image_ref or "") or None,
                )
                verification["verified_target_memory_update"] = dict(memory_update)
        if not target_evidence_passed:
            self._proactive_stop_pending = False
            if self._target_lock_remaining_cycles > 0:
                self._target_lock_remaining_cycles -= 1
                self._target_lock_reacquisition_attempts += 1
                try:
                    full_sweep_after = max(
                        1,
                        int(
                            os.environ.get(
                                "VOCA_TARGET_LOCK_FULL_SWEEP_AFTER",
                                "2",
                            )
                        ),
                    )
                except ValueError:
                    full_sweep_after = 2
                use_full_sweep = bool(
                    self._target_lock_reacquisition_attempts >= full_sweep_after
                )
                has_persistent_target_memory = bool(
                    self.memory_sidecar is not None
                    and getattr(
                        self.memory_sidecar,
                        "last_verified_target_evidence",
                        None,
                    )
                )
                if use_full_sweep and has_persistent_target_memory:
                    anchor_output = (
                        self._verified_target_anchor_reacquisition_output()
                    )
                    if anchor_output is not None:
                        anchor_output["memory_ops"] = []
                        anchor_audit = dict(
                            anchor_output.get(
                                "backend_verified_target_anchor_reacquisition"
                            )
                            or {}
                        )
                        self._target_anchor_reacquisition_count += 1
                        if self.memory_sidecar is not None:
                            self.memory_sidecar.update_supervisor_mode(
                                "verify_target",
                                reason="verified_target_anchor_reacquisition",
                            )
                        decision = self._vlm_output_to_decision(
                            raw_output={
                                **dict(anchor_output),
                                "source": "verified_target_anchor_reacquisition",
                            },
                            safe_output=anchor_output,
                            warnings=["verified_target_anchor_reacquisition"],
                            vlm_input=vlm_input,
                            image_shape=image_shape,
                            angles=angles,
                            call_type=call_type,
                            stop_verification=verification,
                        )
                        decision["proactive_stop_probe"] = True
                        decision["target_lock_reacquisition"] = True
                        decision["verified_target_anchor_reacquisition"] = (
                            anchor_audit
                        )
                        decision["proactive_stop_probe_result"] = dict(
                            verification
                        )
                        return decision
                if use_full_sweep and has_persistent_target_memory:
                    reacquisition_reason = (
                        "reacquire identity-verified target from persistent memory "
                        "with full sweep"
                    )
                elif use_full_sweep:
                    reacquisition_reason = (
                        "reacquire recently detected target with full sweep"
                    )
                else:
                    reacquisition_reason = (
                        "reacquire recently visible target before resuming exploration"
                    )
                output = self._nav_modules.schema.make_observation_request_output(
                    mode="full_sweep" if use_full_sweep else "directed_sweep",
                    center_yaw_deg=0.0,
                    step_deg=45 if use_full_sweep else 30,
                    num_views=8 if use_full_sweep else 5,
                    yaw_offsets_deg=(
                        [-180, -135, -90, -45, 0, 45, 90, 135]
                        if use_full_sweep
                        else [-60, -30, 0, 30, 60]
                    ),
                    reason=reacquisition_reason,
                    confidence="medium",
                )
                output["memory_ops"] = []
                if self.memory_sidecar is not None:
                    self.memory_sidecar.update_supervisor_mode(
                        "verify_target",
                        reason="target_lock_reacquisition_sweep",
                    )
                decision = self._vlm_output_to_decision(
                    raw_output={**dict(output), "source": "target_lock_reacquisition"},
                    safe_output=output,
                    warnings=["target_lock_reacquisition"],
                    vlm_input=vlm_input,
                    image_shape=image_shape,
                    angles=angles,
                    call_type=call_type,
                    stop_verification=verification,
                )
                decision["proactive_stop_probe"] = True
                decision["target_lock_reacquisition"] = True
                decision["proactive_stop_probe_result"] = dict(verification)
                return decision
            if self._target_lock_session_cycles > 0:
                self._exhaust_target_lock_session(
                    reason="target lock expired without reacquiring the target",
                )
            return None

        try:
            target_lock_cycles = max(
                1,
                int(os.environ.get("VOCA_TARGET_LOCK_REACQUIRE_CYCLES", "6")),
            )
        except ValueError:
            target_lock_cycles = 6
        self._target_lock_remaining_cycles = target_lock_cycles
        self._proactive_stop_pending = bool(
            verification.get("single_frame_passed")
            and not verification.get("passed")
        )

        if not verification.get("passed"):
            near_goal_latched = bool(
                verification.get("near_goal_visual_latch_evidence")
                or verification.get("near_goal_visual_latch_active")
            )
            if self._target_direct_approach_cooldown > 0 and not near_goal_latched:
                self._target_direct_approach_cooldown -= 1
                self._proactive_stop_pending = False
                self._target_lock_remaining_cycles = 0
                verification["direct_target_approach_suppressed"] = True
                verification["direct_target_approach_cooldown_remaining"] = int(
                    self._target_direct_approach_cooldown
                )
                verification["target_approach_failure_streak"] = int(
                    self._target_approach_failure_streak
                )
                self._reset_target_lock_session(clear_cooldown=False)
                return None
            if self._target_direct_approach_cooldown > 0 and near_goal_latched:
                verification["direct_target_approach_cooldown_overridden"] = True
                verification["direct_target_approach_cooldown_before_override"] = int(
                    self._target_direct_approach_cooldown
                )
                self._target_direct_approach_cooldown = 0
            prefer_approach = self._prefer_target_approach_before_centering(
                verification
            )
            verification["approach_preferred_before_centering"] = bool(
                prefer_approach
            )
            approach_output = (
                self._approach_output_after_stop_guard(vlm_input, verification)
                if prefer_approach
                else None
            )
            centering_output = (
                None
                if approach_output is not None
                else self._target_centering_output_after_stop_guard(
                    vlm_input,
                    verification,
                )
            )
            if centering_output is not None:
                centering_output["memory_ops"] = []
                if self.memory_sidecar is not None:
                    self.memory_sidecar.update_supervisor_mode(
                        "verify_target",
                        reason="proactive_target_centering_required",
                    )
                decision = self._vlm_output_to_decision(
                    raw_output={
                        **dict(centering_output),
                        "source": "proactive_target_centering",
                    },
                    safe_output=centering_output,
                    warnings=["proactive_target_centering"],
                    vlm_input=vlm_input,
                    image_shape=image_shape,
                    angles=angles,
                    call_type=call_type,
                    stop_verification=verification,
                )
                decision["proactive_stop_probe"] = True
                decision["proactive_target_centering"] = True
                decision["proactive_stop_probe_result"] = dict(verification)
                return decision
            if approach_output is None:
                approach_output = self._approach_output_after_stop_guard(
                    vlm_input,
                    verification,
                )
            if approach_output is None:
                output = self._nav_modules.schema.make_observation_request_output(
                    mode="directed_sweep",
                    center_yaw_deg=0.0,
                    step_deg=30,
                    num_views=5,
                    yaw_offsets_deg=[-60, -30, 0, 30, 60],
                    reason="target visible but a safe approach floor candidate needs reacquisition",
                    confidence="medium",
                )
                output["memory_ops"] = []
                decision = self._vlm_output_to_decision(
                    raw_output={
                        **dict(output),
                        "source": "target_approach_floor_reacquisition",
                    },
                    safe_output=output,
                    warnings=["target_approach_floor_reacquisition"],
                    vlm_input=vlm_input,
                    image_shape=image_shape,
                    angles=angles,
                    call_type=call_type,
                    stop_verification=verification,
                )
                decision["proactive_stop_probe"] = True
                decision["target_approach_floor_reacquisition"] = True
                decision["proactive_stop_probe_result"] = dict(verification)
                return decision
            approach_output["memory_ops"] = []
            validation = validate_pixel_candidate_selection(approach_output, vlm_input)
            canonical_output = validation.get("canonical_output")
            if not validation.get("passed") or not isinstance(canonical_output, dict):
                return None
            selected = validation.get("candidate") or {}
            verification["forced_approach"] = {
                "triggered": True,
                "selected_candidate_ref": selected.get("candidate_ref"),
                "selected_view_id": selected.get("view_id"),
                "point_px": list(selected.get("point_px") or []),
                "reason": "proactive target evidence requires approach translation",
            }
            if self.memory_sidecar is not None:
                self.memory_sidecar.update_supervisor_mode(
                    "goal_seek",
                    reason="proactive_target_visible_approach_motion_required",
                )
            decision = self._vlm_output_to_decision(
                raw_output={
                    **dict(canonical_output),
                    "source": "proactive_target_approach",
                },
                safe_output=canonical_output,
                warnings=["proactive_target_approach"],
                vlm_input=vlm_input,
                image_shape=image_shape,
                angles=angles,
                call_type=call_type,
                pixel_candidate_validation=validation,
                stop_verification=verification,
            )
            decision["proactive_stop_probe"] = True
            decision["proactive_target_approach"] = True
            decision["proactive_stop_probe_result"] = dict(verification)
            return decision

        terminal_refinement_output = self._terminal_refinement_output_after_stop_guard(
            vlm_input,
            verification,
        )
        if terminal_refinement_output is not None:
            terminal_refinement_output["memory_ops"] = []
            validation = validate_pixel_candidate_selection(
                terminal_refinement_output,
                vlm_input,
            )
            canonical_output = validation.get("canonical_output")
            if validation.get("passed") and isinstance(canonical_output, dict):
                refinement = dict(
                    terminal_refinement_output.get("target_terminal_refinement")
                    or {}
                )
                canonical_output["target_terminal_refinement"] = refinement
                self._target_terminal_refinement_count += 1
                verification["terminal_refinement_required"] = True
                verification["terminal_refinement"] = dict(refinement)
                if self.memory_sidecar is not None:
                    self.memory_sidecar.update_supervisor_mode(
                        "verify_target",
                        reason="verified_stop_terminal_refinement",
                    )
                decision = self._vlm_output_to_decision(
                    raw_output={
                        **dict(canonical_output),
                        "source": "verified_stop_terminal_refinement",
                    },
                    safe_output=canonical_output,
                    warnings=["verified_stop_terminal_refinement"],
                    vlm_input=vlm_input,
                    image_shape=image_shape,
                    angles=angles,
                    call_type=call_type,
                    pixel_candidate_validation=validation,
                    stop_verification=verification,
                )
                decision["proactive_stop_probe"] = True
                decision["proactive_target_approach"] = True
                decision["verified_stop_terminal_refinement"] = True
                decision["proactive_stop_probe_result"] = dict(verification)
                return decision

        if self.memory_sidecar is not None:
            self.memory_sidecar.update_supervisor_mode(
                "verify_target",
                reason=(
                    "proactive_target_stop_verified"
                    if verification.get("passed")
                    else "proactive_target_stop_requires_confirmation"
                ),
            )
        output = self._nav_modules.schema.make_stop_output()
        warnings: List[str] = []
        output["memory_ops"] = []
        decision = self._vlm_output_to_decision(
            raw_output={
                **dict(output),
                "source": "proactive_target_stop_probe",
            },
            safe_output=output,
            warnings=warnings,
            vlm_input=vlm_input,
            image_shape=image_shape,
            angles=angles,
            call_type=call_type,
            stop_verification=verification,
        )
        decision["proactive_stop_probe"] = True
        decision["proactive_stop_probe_result"] = dict(verification)
        self._reset_target_lock_session(clear_cooldown=True)
        return decision

    def _terminal_refinement_output_after_stop_guard(
        self,
        vlm_input: Dict[str, Any],
        stop_verification: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        verification = (
            stop_verification if isinstance(stop_verification, dict) else {}
        )
        try:
            max_refinements = max(
                0,
                int(os.environ.get("VOCA_TARGET_STOP_REFINEMENT_MAX", "1")),
            )
        except ValueError:
            max_refinements = 1
        if self._target_terminal_refinement_count >= max_refinements:
            return None

        identity = (
            verification.get("identity_critic")
            if isinstance(verification.get("identity_critic"), dict)
            else {}
        )
        execution = (
            verification.get("target_approach_execution")
            if isinstance(verification.get("target_approach_execution"), dict)
            else {}
        )
        coarse_guard = (
            verification.get("coarse_goal_consistency_guard")
            if isinstance(
                verification.get("coarse_goal_consistency_guard"),
                dict,
            )
            else {}
        )
        try:
            coarse_terminal_stop_radius_m = max(
                0.1,
                float(
                    os.environ.get(
                        "VOCA_COARSE_GOAL_TERMINAL_STOP_RADIUS_M",
                        "0.75",
                    )
                ),
            )
        except ValueError:
            coarse_terminal_stop_radius_m = 0.75
        try:
            coarse_distance_m = float(coarse_guard.get("distance_m"))
        except (TypeError, ValueError):
            coarse_distance_m = None
        coarse_terminal_stop_authorized = bool(
            coarse_guard.get("enabled")
            and coarse_guard.get("triggered")
            and coarse_guard.get("passed")
            and coarse_guard.get("stop_consistent")
            and coarse_distance_m is not None
            and coarse_distance_m
            <= coarse_terminal_stop_radius_m + 1e-6
        )
        verification["coarse_terminal_stop_authorized"] = bool(
            coarse_terminal_stop_authorized
        )
        verification["coarse_terminal_stop_radius_m"] = float(
            coarse_terminal_stop_radius_m
        )
        if coarse_terminal_stop_authorized:
            return None
        try:
            max_bbox_height = max(
                0.0,
                min(
                    1.0,
                    float(
                        os.environ.get(
                            "VOCA_TARGET_STOP_REFINEMENT_MAX_BBOX_HEIGHT",
                            "0.90",
                        )
                    ),
                ),
            )
        except ValueError:
            max_bbox_height = 0.90
        try:
            max_bbox_area = max(
                0.0,
                min(
                    1.0,
                    float(
                        os.environ.get(
                            "VOCA_TARGET_STOP_REFINEMENT_MAX_BBOX_AREA",
                            "0.45",
                        )
                    ),
                ),
            )
        except ValueError:
            max_bbox_area = 0.45
        try:
            bbox_height = float(
                verification.get("target_bbox_height", 0.0) or 0.0
            )
            bbox_area = float(
                verification.get("target_bbox_area", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            return None

        refinement_required = bool(
            verification.get("passed")
            and verification.get("single_frame_passed")
            and str(verification.get("confidence") or "").lower() == "high"
            and verification.get("target_center_guard_satisfied")
            and verification.get("target_scale_consistent")
            and verification.get("post_confirmation_compact_target_evidence")
            and verification.get("target_approach_progress_satisfied")
            and identity.get("triggered")
            and identity.get("passed")
            and identity.get("exact_target")
            and not execution.get("controller_reached_waypoint")
            and not execution.get("stop_action_seen")
            and bbox_height < max_bbox_height
            and bbox_area < max_bbox_area
        )
        if not refinement_required:
            return None

        output = self._approach_output_after_stop_guard(
            vlm_input,
            verification,
        )
        if output is None:
            return None
        refinement = {
            "schema_version": "verified_stop_terminal_refinement_v1",
            "execution_index": int(self._target_terminal_refinement_count + 1),
            "max_executions": int(max_refinements),
            "source_candidate_ref": output.get("selected_candidate_ref"),
            "prior_approach_frame_index": execution.get("frame_index"),
            "prior_approach_translation_m": execution.get("translation_m"),
            "reason": (
                "visual stop is verified, but PixelNav has not completed its final "
                "target-directed waypoint"
            ),
        }
        output["target_terminal_refinement"] = refinement
        reasoning = (
            output.get("reasoning")
            if isinstance(output.get("reasoning"), dict)
            else {}
        )
        reasoning["decision_reason"] = "G07_VERIFIED_STOP_REFINEMENT"
        reasoning["short_text"] = (
            "verified target requires one bounded final PixelNav approach"
        )
        output["reasoning"] = reasoning
        return output

    @staticmethod
    def _prefer_target_approach_before_centering(
        stop_verification: Optional[Dict[str, Any]],
    ) -> bool:
        verification = (
            stop_verification if isinstance(stop_verification, dict) else {}
        )
        near_goal_latched = bool(
            verification.get("near_goal_visual_latch_active")
            or verification.get("near_goal_visual_latch_evidence")
        )
        return bool(
            near_goal_latched
            and (
                not verification.get("target_scale_consistent", True)
                or not verification.get("target_approach_execution_satisfied", False)
            )
        )

    def _target_centering_output_after_stop_guard(
        self,
        vlm_input: Dict[str, Any],
        stop_verification: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        verification = (
            stop_verification if isinstance(stop_verification, dict) else {}
        )
        if not verification.get("target_evidence_passed"):
            return None
        observation = (
            vlm_input.get("observation")
            if isinstance(vlm_input.get("observation"), dict)
            else {}
        )
        try:
            target_view_id = int(verification.get("target_view_index"))
        except (TypeError, ValueError):
            target_view_id = 0
        view_heading = 0.0
        for view in observation.get("views", []) or []:
            if not isinstance(view, dict):
                continue
            try:
                if int(view.get("view_id", -1)) == target_view_id:
                    view_heading = float(view.get("relative_heading_deg", 0.0) or 0.0)
                    break
            except (TypeError, ValueError):
                continue
        try:
            center_x = max(0.0, min(1.0, float(verification.get("target_center_x"))))
        except (TypeError, ValueError):
            center_x = 0.5
        needs_centering = bool(
            abs(view_heading) >= 15.0
            or not verification.get("target_center_guard_satisfied", True)
        )
        if not needs_centering:
            return None
        desired_yaw = view_heading + (center_x - 0.5) * 90.0
        if abs(desired_yaw) < 15.0:
            desired_yaw = -30.0 if center_x < 0.5 else 30.0
        turn_steps = max(1, min(6, int(round(abs(desired_yaw) / 30.0))))
        commanded_yaw = float(turn_steps * 30) * (1.0 if desired_yaw > 0 else -1.0)
        output = self._nav_modules.schema.make_rotate_output(
            commanded_yaw,
            reason="R04_CENTER_VISIBLE_TARGET",
            confidence="medium",
        )
        reasoning = output.get("reasoning") if isinstance(output.get("reasoning"), dict) else {}
        reasoning["failure_mode"] = "target_outside_stop_center_band"
        reasoning["short_text"] = "rotate to center visible target before approach or stop"
        output["reasoning"] = reasoning
        verification["forced_centering"] = {
            "triggered": True,
            "target_view_id": target_view_id,
            "target_center_x": center_x,
            "view_heading_deg": round(view_heading, 3),
            "commanded_yaw_deg": commanded_yaw,
            "reason": "visible target must be centered before conservative stop",
        }
        return output

    def _approach_output_after_stop_guard(
        self,
        vlm_input: Dict[str, Any],
        stop_verification: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        block = (
            vlm_input.get("pixel_candidates")
            if isinstance(vlm_input.get("pixel_candidates"), dict)
            else {}
        )
        candidates = [
            item
            for item in block.get("candidates", []) or []
            if isinstance(item, dict)
            and not item.get("avoid")
            and not item.get("requires_rgb_verification")
        ]

        verification = (
            stop_verification if isinstance(stop_verification, dict) else {}
        )
        try:
            target_view_id = int(verification.get("target_view_index"))
        except (TypeError, ValueError):
            target_view_id = None
        try:
            target_center_x = max(
                0.0,
                min(1.0, float(verification.get("target_center_x"))),
            )
        except (TypeError, ValueError):
            target_center_x = 0.5

        target_memory_override: Optional[Dict[str, Any]] = None
        if target_view_id is not None:
            matching_view = [
                candidate
                for candidate in candidates
                if int(candidate.get("view_id", -1)) == target_view_id
            ]
            if matching_view:
                candidates = matching_view
            else:
                # Never approach an unrelated heading merely because its RGB score is
                # higher. A current exact-target observation may restore one visually
                # safe target-view point that stale directional memory excluded.
                candidates = []
                identity_critic = (
                    verification.get("identity_critic")
                    if isinstance(verification.get("identity_critic"), dict)
                    else {}
                )
                exact_target_verified = bool(
                    verification.get("target_evidence_passed")
                    and verification.get("confidence") == "high"
                    and identity_critic.get("triggered")
                    and identity_critic.get("passed")
                    and identity_critic.get("exact_target")
                )
                excluded = [
                    item
                    for item in block.get("excluded_candidates", []) or []
                    if isinstance(item, dict)
                    and int(item.get("view_id", -1)) == target_view_id
                ]

                def _eligible_target_override(item: Dict[str, Any]) -> bool:
                    visual = (
                        item.get("visual_evidence")
                        if isinstance(item.get("visual_evidence"), dict)
                        else {}
                    )
                    try:
                        support = float(visual.get("support_score", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        support = 0.0
                    return bool(
                        exact_target_verified
                        and item.get("status")
                        in {
                            "deadlock_entry_candidate",
                            "verified_failed_pixel_neighborhood",
                        }
                        and visual.get("available")
                        and not visual.get("hard_reject")
                        and not item.get("requires_rgb_verification")
                        and support >= 0.70
                    )

                eligible = [item for item in excluded if _eligible_target_override(item)]
                if eligible:
                    def _override_priority(item: Dict[str, Any]) -> Tuple[Any, ...]:
                        point_norm = item.get("point_norm") or [0.5, 0.84]
                        try:
                            alignment = -abs(float(point_norm[0]) - target_center_x)
                            forward_reach = -float(point_norm[1])
                        except (TypeError, ValueError, IndexError):
                            alignment = -1.0
                            forward_reach = -1.0
                        status_priority = int(
                            item.get("status") == "deadlock_entry_candidate"
                        )
                        return (
                            status_priority,
                            alignment,
                            forward_reach,
                            str(item.get("candidate_ref") or ""),
                        )

                    original = max(eligible, key=_override_priority)
                    restored = dict(original)
                    restored.update(
                        avoid=False,
                        executable=True,
                        status="verified_target_evidence_override",
                        reason=(
                            "current identity-verified target evidence overrides one stale "
                            "directional failure for a bounded approach"
                        ),
                    )
                    restored.pop("exclusion_reason", None)
                    target_memory_override = {
                        "schema_version": "verified_target_memory_override_v1",
                        "candidate_ref": restored.get("candidate_ref"),
                        "target_view_id": int(target_view_id),
                        "original_status": original.get("status"),
                        "original_reason": original.get("reason"),
                        "original_edge_id": original.get("topological_edge_id"),
                        "authority": "current_identity_verified_target_evidence",
                    }
                    restored["target_evidence_override"] = dict(
                        target_memory_override
                    )
                    block.setdefault("candidates", []).append(restored)
                    block["executable_count"] = len(block.get("candidates", []))
                    block["target_evidence_override"] = dict(
                        target_memory_override
                    )
                    candidates = [restored]
        if not candidates:
            return None

        target_latched = bool(
            verification.get("near_goal_visual_latch_active")
            or verification.get("near_goal_visual_latch_evidence")
        )

        def priority(candidate: Dict[str, Any]) -> Tuple[Any, ...]:
            evidence = (
                candidate.get("pixelnav_feasibility")
                if isinstance(candidate.get("pixelnav_feasibility"), dict)
                else {}
            )
            action = str(evidence.get("predicted_action") or "")
            action_priority = {
                "forward": 3,
                "turn_left": 2,
                "turn_right": 2,
                "look_down": 1,
                "look_up": 1,
            }.get(action, 0)
            point_norm = candidate.get("point_norm") or [0.5, 0.84]
            try:
                horizontal_alignment = -abs(float(point_norm[0]) - target_center_x)
            except (TypeError, ValueError, IndexError):
                horizontal_alignment = -1.0
            try:
                forward_reach = -float(point_norm[1])
            except (TypeError, ValueError, IndexError):
                forward_reach = -1.0
            if target_latched:
                return (
                    horizontal_alignment,
                    forward_reach,
                    action_priority,
                    float(candidate.get("score", 0.0)),
                    str(candidate.get("candidate_ref") or ""),
                )
            return (
                horizontal_alignment,
                action_priority,
                float(candidate.get("score", 0.0)),
                str(candidate.get("candidate_ref") or ""),
            )

        candidate = max(candidates, key=priority)
        observation = (
            vlm_input.get("observation")
            if isinstance(vlm_input.get("observation"), dict)
            else {}
        )
        width = int(observation.get("image_width", 640) or 640)
        height = int(observation.get("image_height", 480) or 480)
        point = candidate.get("point_px") or [width // 2, int(height * 0.84)]
        output = self._nav_modules.schema.make_go_output(
            view_id=int(candidate.get("view_id", 0) or 0),
            view_type=str(candidate.get("view_type_hint") or "front"),
            point_px=(int(point[0]), int(point[1])),
            width=width,
            height=height,
            decision_reason="G06_TARGET_VISIBLE_APPROACH_REQUIRED",
            goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
            short_text="target visible but stop requires additional approach motion",
            confidence="medium",
            selected_candidate_ref=str(candidate.get("candidate_ref") or ""),
        )
        if target_memory_override is not None:
            output["target_approach_memory_override"] = dict(
                target_memory_override
            )
        return output

    def _verify_revisit_if_needed(
        self,
        *,
        safe_output: Dict[str, Any],
        vlm_input: Dict[str, Any],
        contract_validation: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not contract_validation.get("backend_inserted_defer"):
            return {
                "enabled": self.revisit_verifier is not None,
                "triggered": False,
                "passed": bool(contract_validation.get("passed")),
            }
        refs = contract_validation.get("candidate_refs") or []
        candidate_ref = str(refs[0]) if refs else ""
        if self.revisit_verifier is None or not candidate_ref:
            return {
                "enabled": self.revisit_verifier is not None,
                "triggered": False,
                "passed": False,
                "candidate_ref": candidate_ref or None,
                "reason": "dedicated_revisit_verifier_unavailable",
            }

        result = self.revisit_verifier.verify(
            vlm_input=vlm_input,
            candidate_ref=candidate_ref,
        )
        result = dict(result) if isinstance(result, dict) else {}
        result.setdefault("schema_version", "qwen_revisit_verification_v1")
        result.setdefault("enabled", True)
        result.setdefault("triggered", True)
        result.setdefault("passed", False)
        operation = result.get("memory_op") if isinstance(result.get("memory_op"), dict) else {}
        if operation:
            retained = [
                op
                for op in safe_output.get("memory_ops", []) or []
                if not (
                    isinstance(op, dict)
                    and op.get("source") == "backend_memory_contract_guard"
                    and str(op.get("candidate_ref") or "") == candidate_ref
                )
            ]
            retained.append(dict(operation))
            safe_output["memory_ops"] = retained
            contract_validation.update(
                passed=bool(result.get("passed")),
                reason="dedicated_revisit_verifier_resolved",
                backend_inserted_defer=False,
                main_output_missing=True,
                dedicated_verifier_triggered=True,
                dedicated_verifier_operation=operation.get("op"),
            )

        self.revisit_verification_calls += 1
        duration = float(result.get("latency_sec", 0.0) or 0.0)
        self.revisit_verification_durations.append(duration)
        self.llm_call_count += 1
        self.llm_durations.append(duration)
        if result.get("error"):
            self.revisit_verification_error_count += 1
            self.llm_error_count += 1
        else:
            self.llm_success_count += 1
        if result.get("passed"):
            self.revisit_verification_resolved_count += 1
        return result

    def _vlm_output_to_decision(
        self,
        *,
        raw_output: Dict[str, Any],
        safe_output: Dict[str, Any],
        warnings: Sequence[str],
        vlm_input: Dict[str, Any],
        image_shape: Sequence[int],
        angles: Sequence[int],
        call_type: str,
        pixel_candidate_validation: Optional[Dict[str, Any]] = None,
        selected_point_verification: Optional[Dict[str, Any]] = None,
        stop_verification: Optional[Dict[str, Any]] = None,
        strategic_validation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        safe_output = dict(safe_output)
        frame_index = int(
            vlm_input.get("observation", {}).get("frame_index", 0) or 0
        )
        strategic_state = self._record_strategic_state(
            safe_output,
            frame_index=frame_index,
        )
        safe_output["strategic_state"] = dict(strategic_state)
        warnings = list(warnings)
        semantic_memory_update: Dict[str, Any] = {}
        if self.memory_sidecar is not None:
            try:
                semantic_memory_update = self.memory_sidecar.record_semantic_state(
                    strategic_state,
                    frame_index=frame_index,
                )
            except Exception as exc:
                warnings.append(
                    "strategic_memory_update:{}:{}".format(
                        type(exc).__name__,
                        str(exc)[:120],
                    )
                )
        action = str(safe_output.get("action") or "")
        direction = self._direction_from_output(safe_output, vlm_input, len(angles))
        angle = int(angles[direction]) if angles else 0
        if action == "rotate":
            control = (
                safe_output.get("control")
                if isinstance(safe_output.get("control"), dict)
                else {}
            )
            try:
                angle = int(round(float(control.get("rotate_yaw_deg", angle))))
            except (TypeError, ValueError):
                pass
        point = self._point_from_output(safe_output, image_shape)
        original_point = list(point) if point is not None else None
        fallback_reasons = list(warnings)
        if point is None:
            point = _lower_center_point(image_shape)
            fallback_reasons.append("action_{}_uses_lower_center_point".format(action or "unknown"))
        point_guard_reason = ""
        if action == "go":
            point_guard_reason = self._go_point_guard_reason(point, image_shape)
            if point_guard_reason:
                point = _lower_center_point(image_shape)
                fallback_reasons.append(point_guard_reason)
            elif not (
                isinstance(pixel_candidate_validation, dict)
                and pixel_candidate_validation.get("passed")
            ) and self._is_repeated_no_progress_point(
                point,
                angle=angle,
                selected_view_id=direction,
                vlm_input=vlm_input,
            ):
                point_guard_reason = "go_point_repeated_no_progress"
                point = self._alternative_point_after_no_progress(point, image_shape)
                fallback_reasons.append(point_guard_reason)
        if action != "go":
            fallback_reasons.append("action_{}_no_pixelnav_waypoint".format(action or "unknown"))
        fallback = bool(fallback_reasons)
        validated_candidate = (
            pixel_candidate_validation.get("candidate")
            if isinstance(pixel_candidate_validation, dict)
            and isinstance(pixel_candidate_validation.get("candidate"), dict)
            else {}
        )
        decision = {
            "schema_version": "nav_vlm_waypoint_v1",
            "action": action,
            "Reason": self._reason_from_output(safe_output),
            "Angle": angle,
            "Point": point,
            "Flag": action == "stop",
            "Confidence": self._confidence_from_output(safe_output, fallback),
            "fallback": fallback,
            "fallback_reason": ",".join(fallback_reasons),
            "target": self.object_goal,
            "call_type": call_type,
            "priors": self.latest_priors,
            "angles": list(map(int, angles)),
            "selected_view_id": direction,
            "selected_candidate_ref": safe_output.get("selected_candidate_ref"),
            "selected_topological_candidate_ref": validated_candidate.get(
                "topological_candidate_ref"
            ),
            "selected_topological_edge_id": validated_candidate.get("topological_edge_id"),
            "selected_topological_relation_type": validated_candidate.get(
                "topological_relation_type"
            ),
            "supervisor_mode": (
                self.memory_sidecar.supervisor_mode
                if self.memory_sidecar is not None
                else "goal_seek"
            ),
            "strategic_state": dict(strategic_state),
            "strategic_runtime": self._strategic_runtime_snapshot(),
            "vlm_input": vlm_input,
            "vlm_output": safe_output,
            "raw_vlm_output": raw_output,
            "warnings": list(warnings),
        }
        if isinstance(pixel_candidate_validation, dict):
            decision["pixel_candidate_validation"] = {
                key: value
                for key, value in pixel_candidate_validation.items()
                if key != "canonical_output"
            }
        if isinstance(selected_point_verification, dict):
            decision["selected_point_verification"] = dict(selected_point_verification)
        if isinstance(stop_verification, dict):
            decision["stop_verification"] = dict(stop_verification)
        if isinstance(strategic_validation, dict):
            decision["strategic_validation"] = dict(strategic_validation)
        if isinstance(safe_output.get("backend_memory_backtrack"), dict):
            decision["backend_memory_backtrack"] = dict(
                safe_output["backend_memory_backtrack"]
            )
        if isinstance(safe_output.get("backend_novel_sector_escape"), dict):
            decision["backend_novel_sector_escape"] = dict(
                safe_output["backend_novel_sector_escape"]
            )
        if isinstance(
            safe_output.get("backend_coarse_goal_direction_guard"),
            dict,
        ):
            decision["coarse_goal_direction_guard"] = dict(
                safe_output["backend_coarse_goal_direction_guard"]
            )
        if semantic_memory_update:
            decision["semantic_memory_update"] = dict(semantic_memory_update)
        if point_guard_reason:
            decision["point_guard"] = {
                "reason": point_guard_reason,
                "original_point": original_point,
                "replacement_point": list(point),
            }
        if self.memory_sidecar is not None:
            decision["memory_visualizer"] = self.memory_sidecar.visualizer_state()
        return decision

    def _fallback_decision(
        self,
        *,
        vlm_input: Dict[str, Any],
        image_shape: Sequence[int],
        angles: Sequence[int],
        call_type: str,
        reason: str,
    ) -> Dict[str, Any]:
        output = self._nav_modules.schema.make_rotate_output(
            45,
            reason="R02_NO_VISIBLE_NAVIGABLE_FLOOR",
            confidence="low",
        )
        return self._vlm_output_to_decision(
            raw_output=output,
            safe_output=output,
            warnings=["qwen_vlm_failed:{}".format(reason)],
            vlm_input=vlm_input,
            image_shape=image_shape,
            angles=angles,
            call_type=call_type,
        )

    def _invalid_json_fallback_decision(
        self,
        *,
        vlm_input: Dict[str, Any],
        image_shape: Sequence[int],
        angles: Sequence[int],
        call_type: str,
    ) -> Dict[str, Any]:
        output = self._nav_modules.schema.make_observation_request_output(
            mode="directed_sweep",
            center_yaw_deg=0.0,
            step_deg=30,
            num_views=5,
            yaw_offsets_deg=[-60, -30, 0, 30, 60],
            reason="invalid Qwen JSON; request observation instead of inventing a waypoint",
            confidence="low",
        )
        validation = {
            "schema_version": "pixel_candidate_validation_v1",
            "passed": False,
            "reason": "invalid_vlm_json_no_candidate_selection",
            "requested_ref": None,
            "candidate": None,
            "coordinate_corrected": False,
            "checkpoint": {
                "rule_id": "PIXEL_GO_001",
                "name": "go_requires_valid_rgb_pixel_candidate",
                "stage": "action_validation",
                "passed": False,
                "message": "invalid_vlm_json_no_candidate_selection",
                "on_fail": "request_observation.directed_sweep",
            },
        }
        return self._vlm_output_to_decision(
            raw_output=output,
            safe_output=output,
            warnings=["qwen_vlm_invalid_json"],
            vlm_input=vlm_input,
            image_shape=image_shape,
            angles=angles,
            call_type=call_type,
            pixel_candidate_validation=validation,
        )

    @staticmethod
    def _is_invalid_json_response_error(exc: Exception) -> bool:
        text = str(exc or "").lower()
        return isinstance(exc, ValueError) and (
            "failed to extract compact vlm json" in text
            or "no json object found" in text
            or "invalid json" in text
        )

    def _query_point_decision(self, pano_images: Sequence[np.ndarray], angles: Sequence[int], call_type: str) -> Dict[str, Any]:
        selected_shape = pano_images[0].shape if pano_images else (480, 640, 3)
        vlm_input = self._build_vlm_input(pano_images, angles, call_type)
        decision: Optional[Dict[str, Any]] = None
        self._last_proactive_stop_probe = None

        for try_idx in range(self.config.retries):
            if try_idx == 0:
                decision = self._proactive_stop_decision(
                    vlm_input=vlm_input,
                    image_shape=selected_shape,
                    angles=angles,
                    call_type=call_type,
                )
                if decision is not None:
                    break
            print("[LLM/QWEN_VLM] try {}/{}".format(try_idx + 1, self.config.retries))
            t0 = time.perf_counter()
            model_call_elapsed: Optional[float] = None
            strategic_validation: Dict[str, Any] = {}
            try:
                raw_output = self.vlm_client.decide(vlm_input)
                model_call_elapsed = time.perf_counter() - t0
                requested_strategic_state = normalize_strategic_state(
                    raw_output.get("strategic_state"),
                    action=str(raw_output.get("action") or ""),
                    reasoning=(
                        raw_output.get("reasoning")
                        if isinstance(raw_output.get("reasoning"), dict)
                        else {}
                    ),
                )
                strategic_runtime = (
                    vlm_input.get("runtime", {}).get("strategic_runtime", {})
                    if isinstance(vlm_input.get("runtime"), dict)
                    else {}
                )
                force_leave_active = bool(
                    isinstance(strategic_runtime, dict)
                    and strategic_runtime.get("force_leave_room")
                )
                requested_exit_intent = bool(
                    requested_strategic_state.get("navigation_intent")
                    in _EXIT_INTENTS
                )
                safe_output, warnings = self._nav_modules.safety.sanitize_vlm_output(raw_output, vlm_input)
                safe_output["strategic_state"] = dict(requested_strategic_state)
                raw_memory_ops = list(safe_output.get("memory_ops") or [])
                pixel_validation = validate_pixel_candidate_selection(
                    safe_output,
                    vlm_input,
                )
                canonical_output = pixel_validation.get("canonical_output")
                if pixel_validation.get("passed") and isinstance(canonical_output, dict):
                    safe_output = canonical_output
                elif str(safe_output.get("action") or "").lower() == "go":
                    validation_reason = str(pixel_validation.get("reason") or "candidate_validation_failed")
                    if force_leave_active:
                        self._record_forced_exit_failure(
                            "pixel_candidate_gate:{}".format(validation_reason)
                        )
                        safe_output, _ = self._forced_exit_recovery_output(
                            vlm_input,
                            reason="pixel candidate gate rejected forced-exit go: {}".format(
                                validation_reason
                            ),
                        )
                    else:
                        safe_output = self._nav_modules.schema.make_observation_request_output(
                            mode="directed_sweep",
                            center_yaw_deg=0.0,
                            step_deg=30,
                            num_views=5,
                            yaw_offsets_deg=[-60, -30, 0, 30, 60],
                            reason="pixel candidate gate rejected go: {}".format(validation_reason),
                            confidence="low",
                        )
                    safe_output["memory_ops"] = raw_memory_ops
                    warnings = list(warnings) + [
                        "pixel_candidate_gate:{}".format(validation_reason)
                    ]
                safe_output, _, _, gap_warnings = self._nav_modules.policy.apply_gap_lite_validation(
                    safe_output,
                    vlm_input.get("memory", {}),
                    goal_bearing_deg=0.0,
                    force_front_view_waypoint=False,
                    require_candidate_ref_for_go=True,
                    require_candidate_ref_for_revisit_ops=True,
                )
                warnings = list(warnings) + list(gap_warnings)
                safe_output["strategic_state"] = dict(requested_strategic_state)
                safe_output, strategic_validation, strategic_warnings = (
                    self._apply_strategic_exit_guard(safe_output, vlm_input)
                )
                warnings = list(warnings) + list(strategic_warnings)
                (
                    safe_output,
                    coarse_direction_validation,
                    coarse_direction_warnings,
                ) = self._apply_coarse_goal_direction_guard(
                    safe_output,
                    vlm_input,
                )
                warnings = list(warnings) + list(coarse_direction_warnings)
                if (
                    str(safe_output.get("action") or "").lower() == "go"
                    and (
                        safe_output.get("backend_memory_backtrack")
                        or safe_output.get("backend_novel_sector_escape")
                    )
                ):
                    strategic_pixel_validation = validate_pixel_candidate_selection(
                        safe_output,
                        vlm_input,
                    )
                    strategic_canonical_output = strategic_pixel_validation.get(
                        "canonical_output"
                    )
                    if strategic_pixel_validation.get("passed") and isinstance(
                        strategic_canonical_output,
                        dict,
                    ):
                        safe_output = self._preserve_backend_recovery_metadata(
                            strategic_canonical_output,
                            safe_output,
                        )
                        pixel_validation = strategic_pixel_validation
                safe_output, memory_contract_validation, memory_contract_warnings = (
                    enforce_memory_verdict_contract(safe_output, vlm_input)
                )
                warnings = list(warnings) + list(memory_contract_warnings)
                revisit_verification = self._verify_revisit_if_needed(
                    safe_output=safe_output,
                    vlm_input=vlm_input,
                    contract_validation=memory_contract_validation,
                )
                validated_memory_ops = list(safe_output.get("memory_ops") or [])
                point_verification = self._verify_selected_point_if_needed(
                    safe_output=safe_output,
                    pixel_candidate_validation=pixel_validation,
                    pano_images=pano_images,
                )
                backend_memory_backtrack = bool(
                    safe_output.get("backend_memory_backtrack")
                )
                backend_novel_sector_escape = bool(
                    safe_output.get("backend_novel_sector_escape")
                )
                if (
                    point_verification.get("triggered")
                    and not point_verification.get("passed")
                    and str(safe_output.get("action") or "").lower() == "go"
                ):
                    rejected_candidate = (
                        pixel_validation.get("candidate")
                        if isinstance(pixel_validation.get("candidate"), dict)
                        else {}
                    )
                    self.record_navigation_feedback(
                        {
                            "event": "point_verifier_rejected",
                            "selected_candidate_ref": rejected_candidate.get(
                                "candidate_ref"
                            ),
                            "selected_view_id": rejected_candidate.get("view_id"),
                            "angle_deg": rejected_candidate.get(
                                "relative_heading_deg", 0.0
                            ),
                            "world_bearing_deg": _normalize_angle_deg(
                                math.degrees(self._runtime_heading_rad)
                                + float(
                                    rejected_candidate.get(
                                        "relative_heading_deg", 0.0
                                    )
                                    or 0.0
                                )
                            ),
                            "point_px": list(rejected_candidate.get("point_px") or []),
                            "surface": point_verification.get("surface") or "unknown",
                            "reason": point_verification.get("reason"),
                        }
                    )
                    verification_reason = str(
                        point_verification.get("reason") or "selected point not verified"
                    )
                    verification_surface = str(
                        point_verification.get("surface") or "unknown"
                    )
                    retry_selected = False
                    backtrack_retry_attempts: List[Dict[str, Any]] = []
                    novel_retry_attempts: List[Dict[str, Any]] = []
                    if (
                        requested_exit_intent
                        or backend_memory_backtrack
                        or backend_novel_sector_escape
                    ):
                        self._record_forced_exit_failure(
                            "point_verifier:{}".format(verification_surface)
                        )
                        try:
                            retry_limit = max(
                                1,
                                int(
                                    os.environ.get(
                                        "VOCA_FORCE_LEAVE_BACKTRACK_RETRIES",
                                        "3",
                                    )
                                ),
                            )
                        except ValueError:
                            retry_limit = 3
                        excluded_refs = {
                            str(rejected_candidate.get("candidate_ref") or "")
                        }
                        strategic_state = (
                            safe_output.get("strategic_state")
                            if isinstance(safe_output.get("strategic_state"), dict)
                            else self._last_strategic_state
                        )
                        for retry_index in range(retry_limit):
                            replacement = self._forced_exit_memory_backtrack_output(
                                vlm_input,
                                dict(strategic_state),
                                excluded_candidate_refs=sorted(excluded_refs),
                            )
                            if replacement is None:
                                break
                            replacement["memory_ops"] = validated_memory_ops
                            replacement_validation = validate_pixel_candidate_selection(
                                replacement,
                                vlm_input,
                            )
                            replacement_output = replacement_validation.get(
                                "canonical_output"
                            )
                            if not isinstance(replacement_output, dict):
                                replacement_output = replacement
                            replacement_output = (
                                self._preserve_backend_recovery_metadata(
                                    replacement_output,
                                    replacement,
                                )
                            )
                            replacement_candidate = (
                                replacement_validation.get("candidate")
                                if isinstance(
                                    replacement_validation.get("candidate"),
                                    dict,
                                )
                                else {}
                            )
                            replacement_ref = str(
                                replacement_candidate.get("candidate_ref")
                                or replacement_output.get("selected_candidate_ref")
                                or ""
                            )
                            excluded_refs.add(replacement_ref)
                            replacement_verification = (
                                self._verify_selected_point_if_needed(
                                    safe_output=replacement_output,
                                    pixel_candidate_validation=replacement_validation,
                                    pano_images=pano_images,
                                )
                            )
                            replacement_passed = bool(
                                replacement_validation.get("passed")
                                and replacement_verification.get("passed")
                            )
                            backtrack_retry_attempts.append(
                                {
                                    "retry_index": int(retry_index),
                                    "candidate_ref": replacement_ref or None,
                                    "view_id": replacement_candidate.get("view_id"),
                                    "point_px": list(
                                        replacement_candidate.get("point_px") or []
                                    ),
                                    "topological_edge_id": replacement_candidate.get(
                                        "topological_edge_id"
                                    ),
                                    "passed": replacement_passed,
                                    "surface": replacement_verification.get("surface"),
                                    "reason": replacement_verification.get("reason"),
                                }
                            )
                            if replacement_passed:
                                replacement_verification = dict(
                                    replacement_verification
                                )
                                replacement_verification[
                                    "backtrack_retry_attempts"
                                ] = list(backtrack_retry_attempts)
                                safe_output = replacement_output
                                pixel_validation = replacement_validation
                                point_verification = replacement_verification
                                backend_memory_backtrack = True
                                retry_selected = True
                                warnings = list(warnings) + [
                                    "point_verifier_memory_backtrack_retry_selected"
                                ]
                                break
                            self.record_navigation_feedback(
                                {
                                    "event": "point_verifier_rejected",
                                    "selected_candidate_ref": replacement_ref or None,
                                    "selected_view_id": replacement_candidate.get(
                                        "view_id"
                                    ),
                                    "angle_deg": replacement_candidate.get(
                                        "relative_heading_deg", 0.0
                                    ),
                                    "world_bearing_deg": _normalize_angle_deg(
                                        math.degrees(self._runtime_heading_rad)
                                        + float(
                                            replacement_candidate.get(
                                                "relative_heading_deg", 0.0
                                            )
                                            or 0.0
                                        )
                                    ),
                                    "point_px": list(
                                        replacement_candidate.get("point_px") or []
                                    ),
                                    "surface": replacement_verification.get(
                                        "surface"
                                    )
                                    or "unknown",
                                    "reason": replacement_verification.get("reason"),
                                }
                            )
                            self._record_forced_exit_failure(
                                "backtrack_point_verifier:{}".format(
                                    replacement_verification.get("surface")
                                    or "unknown"
                                )
                            )
                        if not retry_selected:
                            try:
                                novel_retry_limit = max(
                                    1,
                                    int(
                                        os.environ.get(
                                            "VOCA_FORCE_LEAVE_NOVEL_SECTOR_RETRIES",
                                            "2",
                                        )
                                    ),
                                )
                            except ValueError:
                                novel_retry_limit = 2
                            for retry_index in range(novel_retry_limit):
                                replacement = self._forced_exit_novel_sector_output(
                                    vlm_input,
                                    dict(strategic_state),
                                    excluded_candidate_refs=sorted(excluded_refs),
                                )
                                if replacement is None:
                                    break
                                replacement["memory_ops"] = validated_memory_ops
                                replacement_validation = (
                                    validate_pixel_candidate_selection(
                                        replacement,
                                        vlm_input,
                                    )
                                )
                                replacement_output = replacement_validation.get(
                                    "canonical_output"
                                )
                                if not isinstance(replacement_output, dict):
                                    replacement_output = replacement
                                replacement_output = (
                                    self._preserve_backend_recovery_metadata(
                                        replacement_output,
                                        replacement,
                                    )
                                )
                                replacement_candidate = (
                                    replacement_validation.get("candidate")
                                    if isinstance(
                                        replacement_validation.get("candidate"),
                                        dict,
                                    )
                                    else {}
                                )
                                replacement_ref = str(
                                    replacement_candidate.get("candidate_ref")
                                    or replacement_output.get(
                                        "selected_candidate_ref"
                                    )
                                    or ""
                                )
                                excluded_refs.add(replacement_ref)
                                replacement_verification = (
                                    self._verify_selected_point_if_needed(
                                        safe_output=replacement_output,
                                        pixel_candidate_validation=(
                                            replacement_validation
                                        ),
                                        pano_images=pano_images,
                                    )
                                )
                                replacement_passed = bool(
                                    replacement_validation.get("passed")
                                    and replacement_verification.get("passed")
                                )
                                novel_retry_attempts.append(
                                    {
                                        "retry_index": int(retry_index),
                                        "recovery_kind": "novel_sector_escape",
                                        "candidate_ref": replacement_ref or None,
                                        "view_id": replacement_candidate.get(
                                            "view_id"
                                        ),
                                        "point_px": list(
                                            replacement_candidate.get("point_px")
                                            or []
                                        ),
                                        "passed": replacement_passed,
                                        "surface": replacement_verification.get(
                                            "surface"
                                        ),
                                        "reason": replacement_verification.get(
                                            "reason"
                                        ),
                                    }
                                )
                                if replacement_passed:
                                    replacement_verification = dict(
                                        replacement_verification
                                    )
                                    replacement_verification[
                                        "backtrack_retry_attempts"
                                    ] = list(backtrack_retry_attempts)
                                    replacement_verification[
                                        "novel_sector_retry_attempts"
                                    ] = list(novel_retry_attempts)
                                    safe_output = replacement_output
                                    pixel_validation = replacement_validation
                                    point_verification = replacement_verification
                                    backend_memory_backtrack = False
                                    backend_novel_sector_escape = True
                                    retry_selected = True
                                    strategic_validation.update(
                                        passed=True,
                                        reason=(
                                            "forced_exit_novel_sector_selected_"
                                            "after_backtrack_rejection"
                                        ),
                                        recovery_action="go.novel_safe_sector",
                                    )
                                    strategic_validation.pop(
                                        "backend_memory_backtrack",
                                        None,
                                    )
                                    strategic_validation[
                                        "backend_novel_sector_escape"
                                    ] = dict(
                                        replacement.get(
                                            "backend_novel_sector_escape"
                                        )
                                        or {}
                                    )
                                    strategic_validation["checkpoint"].update(
                                        passed=True,
                                        message=strategic_validation["reason"],
                                        on_fail=None,
                                    )
                                    warnings = list(warnings) + [
                                        "point_verifier_novel_sector_retry_selected"
                                    ]
                                    break
                                self.record_navigation_feedback(
                                    {
                                        "event": "point_verifier_rejected",
                                        "selected_candidate_ref": (
                                            replacement_ref or None
                                        ),
                                        "selected_view_id": (
                                            replacement_candidate.get("view_id")
                                        ),
                                        "angle_deg": replacement_candidate.get(
                                            "relative_heading_deg",
                                            0.0,
                                        ),
                                        "world_bearing_deg": _normalize_angle_deg(
                                            math.degrees(
                                                self._runtime_heading_rad
                                            )
                                            + float(
                                                replacement_candidate.get(
                                                    "relative_heading_deg",
                                                    0.0,
                                                )
                                                or 0.0
                                            )
                                        ),
                                        "point_px": list(
                                            replacement_candidate.get("point_px")
                                            or []
                                        ),
                                        "surface": (
                                            replacement_verification.get("surface")
                                            or "unknown"
                                        ),
                                        "reason": replacement_verification.get(
                                            "reason"
                                        ),
                                    }
                                )
                                self._record_forced_exit_failure(
                                    "novel_sector_point_verifier:{}".format(
                                        replacement_verification.get("surface")
                                        or "unknown"
                                    )
                                )
                        if not retry_selected:
                            point_verification = dict(point_verification)
                            point_verification[
                                "backtrack_retry_attempts"
                            ] = list(backtrack_retry_attempts)
                            point_verification[
                                "novel_sector_retry_attempts"
                            ] = list(novel_retry_attempts)
                            safe_output, _ = self._forced_exit_recovery_output(
                                vlm_input,
                                reason=(
                                    "RGB doorway point verifier rejected forced-exit go: {} ({})"
                                ).format(verification_surface, verification_reason),
                            )
                    else:
                        safe_output = self._nav_modules.schema.make_observation_request_output(
                            mode="directed_sweep",
                            center_yaw_deg=0.0,
                            step_deg=30,
                            num_views=5,
                            yaw_offsets_deg=[-60, -30, 0, 30, 60],
                            reason=(
                                "RGB point verifier rejected go: {} ({})"
                            ).format(verification_surface, verification_reason),
                            confidence="low",
                        )
                    safe_output["memory_ops"] = validated_memory_ops
                    warnings = list(warnings) + [
                        "point_verifier:{}:{}".format(
                            verification_surface,
                            verification_reason,
                        )
                    ]
                stop_verification = self._verify_stop_if_needed(
                    safe_output=safe_output,
                    vlm_input=vlm_input,
                )
                if (
                    stop_verification.get("triggered")
                    and not stop_verification.get("passed")
                    and str(safe_output.get("action") or "").lower() == "stop"
                ):
                    stop_reason = str(
                        stop_verification.get("reason") or "target completion not verified"
                    )
                    prefer_approach = self._prefer_target_approach_before_centering(
                        stop_verification
                    )
                    stop_verification["approach_preferred_before_centering"] = bool(
                        prefer_approach
                    )
                    approach_output = (
                        self._approach_output_after_stop_guard(
                            vlm_input,
                            stop_verification,
                        )
                        if prefer_approach
                        else None
                    )
                    centering_output = (
                        None
                        if approach_output is not None
                        else self._target_centering_output_after_stop_guard(
                            vlm_input,
                            stop_verification,
                        )
                    )
                    if centering_output is not None:
                        safe_output = centering_output
                        safe_output["memory_ops"] = validated_memory_ops
                        warnings = list(warnings) + ["stop_guard_forced_target_centering"]
                    elif approach_output is None and (
                        (
                            stop_verification.get("single_frame_passed")
                            or stop_verification.get("target_evidence_passed")
                        )
                        and not stop_verification.get("passed")
                    ):
                        approach_output = self._approach_output_after_stop_guard(
                            vlm_input,
                            stop_verification,
                        )
                    if approach_output is not None:
                        approach_output["memory_ops"] = validated_memory_ops
                        approach_validation = validate_pixel_candidate_selection(
                            approach_output,
                            vlm_input,
                        )
                        canonical_approach = approach_validation.get("canonical_output")
                        if approach_validation.get("passed") and isinstance(
                            canonical_approach, dict
                        ):
                            safe_output = canonical_approach
                            pixel_validation = approach_validation
                            selected = approach_validation.get("candidate") or {}
                            stop_verification["forced_approach"] = {
                                "triggered": True,
                                "selected_candidate_ref": selected.get("candidate_ref"),
                                "selected_view_id": selected.get("view_id"),
                                "point_px": list(selected.get("point_px") or []),
                                "reason": "verified target requires approach translation before stop",
                            }
                            warnings = list(warnings) + [
                                "stop_motion_guard_forced_approach"
                            ]
                        else:
                            approach_output = None
                    if centering_output is None and approach_output is None:
                        safe_output = self._nav_modules.schema.make_observation_request_output(
                            mode="directed_sweep",
                            center_yaw_deg=0.0,
                            step_deg=30,
                            num_views=5,
                            yaw_offsets_deg=[-60, -30, 0, 30, 60],
                            reason="stop verifier rejected completion: {}".format(stop_reason),
                            confidence="low",
                        )
                    safe_output["memory_ops"] = validated_memory_ops
                    if not (
                        stop_verification.get("forced_approach")
                        or stop_verification.get("forced_centering")
                    ):
                        warnings = list(warnings) + [
                            "stop_verifier_rejected:{}".format(stop_reason)
                        ]
                if self.memory_sidecar is not None:
                    if (
                        stop_verification.get("forced_approach")
                        or stop_verification.get("forced_centering")
                    ):
                        self.memory_sidecar.update_supervisor_mode(
                            "verify_target",
                            reason="target_visible_center_or_approach_required",
                        )
                    elif stop_verification.get("triggered"):
                        self.memory_sidecar.update_supervisor_mode(
                            "verify_target",
                            reason=(
                                "target_stop_verified"
                                if stop_verification.get("passed")
                                else "target_stop_requires_more_evidence"
                            ),
                        )
                    elif (
                        self.memory_sidecar.supervisor_mode == "verify_target"
                        and str(safe_output.get("action") or "").lower() in {"go", "rotate"}
                    ):
                        self.memory_sidecar.update_supervisor_mode(
                            "goal_seek",
                            reason="target_not_confirmed_resume_search",
                        )
                final_strategic_state = dict(requested_strategic_state)
                if stop_verification.get("forced_approach"):
                    final_strategic_state.update(
                        target_evidence=(
                            final_strategic_state.get("target_evidence")
                            if final_strategic_state.get("target_evidence")
                            in {"possible", "confirmed"}
                            else "possible"
                        ),
                        navigation_intent="approach_target",
                        room_search_status="target_likely",
                        selected_exit_ref=None,
                        reason_code="BACKEND_TARGET_APPROACH_REQUIRED",
                    )
                elif stop_verification.get("forced_centering"):
                    final_strategic_state.update(
                        target_evidence=(
                            final_strategic_state.get("target_evidence")
                            if final_strategic_state.get("target_evidence")
                            in {"possible", "confirmed"}
                            else "possible"
                        ),
                        navigation_intent="verify_target",
                        room_search_status="target_likely",
                        selected_exit_ref=None,
                        reason_code="BACKEND_TARGET_CENTERING_REQUIRED",
                    )
                safe_output["strategic_state"] = final_strategic_state
                memory_op_results = []
                if self.memory_sidecar is not None:
                    memory_op_results = self.memory_sidecar.apply_vlm_memory_ops(
                        safe_output,
                        frame_index=int(vlm_input.get("observation", {}).get("frame_index", 0) or 0),
                    )
                decision = self._vlm_output_to_decision(
                    raw_output=raw_output,
                    safe_output=safe_output,
                    warnings=warnings,
                    vlm_input=vlm_input,
                    image_shape=selected_shape,
                    angles=angles,
                    call_type=call_type,
                    pixel_candidate_validation=pixel_validation,
                    selected_point_verification=point_verification,
                    stop_verification=stop_verification,
                    strategic_validation=strategic_validation,
                )
                if memory_op_results:
                    decision["memory_op_results"] = memory_op_results
                if isinstance(self._last_proactive_stop_probe, dict):
                    decision["proactive_stop_probe_result"] = dict(
                        self._last_proactive_stop_probe
                    )
                    probe = self._last_proactive_stop_probe
                    selected_view = decision.get("selected_view_id")
                    target_view = probe.get("target_view_index")
                    same_target_view = bool(
                        target_view is None
                        or selected_view is None
                        or int(selected_view) == int(target_view)
                    )
                    if (
                        str(decision.get("action") or "").lower() == "go"
                        and probe.get("target_visible") is True
                        and probe.get("target_match") is True
                        and same_target_view
                    ):
                        decision["target_context_approach"] = True
                if stop_verification.get("forced_approach"):
                    decision["backend_action_override"] = {
                        "from": str(raw_output.get("action") or "stop"),
                        "to": "go",
                        "reason": "stop_confirmation_requires_approach_motion",
                        **dict(stop_verification["forced_approach"]),
                    }
                elif stop_verification.get("forced_centering"):
                    decision["backend_action_override"] = {
                        "from": str(raw_output.get("action") or "stop"),
                        "to": "rotate",
                        "reason": "stop_confirmation_requires_target_centering",
                        **dict(stop_verification["forced_centering"]),
                    }
                decision["revisit_verification"] = revisit_verification
                decision["memory_contract_validation"] = memory_contract_validation
                self.llm_success_count += 1
            except Exception as exc:
                if self._is_invalid_json_response_error(exc):
                    self.vlm_json_fallback_count += 1
                    self.vlm_json_last_error = str(exc)[:240]
                    decision = self._invalid_json_fallback_decision(
                        vlm_input=vlm_input,
                        image_shape=selected_shape,
                        angles=angles,
                        call_type=call_type,
                    )
                    decision["vlm_json_fallback"] = True
                    decision["vlm_json_last_error"] = self.vlm_json_last_error
                    print("[LLM/QWEN_VLM] invalid JSON fallback:", self.vlm_json_last_error)
                else:
                    self._record_llm_exception("QWEN_VLM", try_idx, exc)
            finally:
                self.llm_call_count += 1
                dt = (
                    float(model_call_elapsed)
                    if model_call_elapsed is not None
                    else time.perf_counter() - t0
                )
                self.llm_durations.append(dt)
                self.navigation_vlm_durations.append(dt)
                print("[LLM/QWEN_VLM] elapsed={:.3f}s has_decision={}".format(dt, bool(decision)))
            if decision is not None:
                break

        if decision is None:
            decision = self._fallback_decision(
                vlm_input=vlm_input,
                image_shape=selected_shape,
                angles=angles,
                call_type=call_type,
                reason=self.llm_last_error or "no_valid_output",
            )

        self.last_decision = decision
        self.qwen_call_log.append(decision.copy())
        self._flush_qwen_call(decision)
        print("[QWEN_VLM] action={} angle={} point={} confidence={} fallback={}".format(
            decision.get("action"),
            decision.get("Angle"),
            decision.get("Point"),
            decision.get("Confidence"),
            decision.get("fallback"),
        ))
        if decision.get("Reason"):
            print("[QWEN_VLM] reason:", decision.get("Reason"))
        return decision
