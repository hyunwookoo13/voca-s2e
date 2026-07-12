import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from settings import LLM_BACKEND
from llm_utils.navigation_prompts import PRIOR_CLASS_LIST, PRIORS_PROMPT
from llm_utils.priors_parser import extract_priors, parse_llm_json

if LLM_BACKEND == "gemini":
    from llm_utils.gemini_request import text_response, vision_response
elif LLM_BACKEND == "qwen":
    from llm_utils.qwen_request import text_response, vision_response
elif LLM_BACKEND == "ollama":
    from llm_utils.ollama_request import text_response, vision_response
else:
    raise ValueError("Unsupported VOCA_LLM_BACKEND: {}".format(LLM_BACKEND))


QWEN_POINT_PROMPT = """
You are a Qwen VLM semantic supervisor for indoor ObjectNav.

You receive:
- <Target Object>: the object category to find.
- <Target Context Cues>: likely supports, co-occurring objects, gateways, and lookalikes.
- <Panoramic Image>: a labeled grid of robot RGB views. Each view label is a relative angle.

Task:
Choose the single best view and one visible navigable image point for a fast PixelNav-style controller.

Rules:
1. The Point must be in the selected single view coordinate frame, not the panorama canvas.
2. Point format is [u, v], where u is horizontal pixel coordinate and v is vertical pixel coordinate.
3. Select visible navigable floor, doorway entry, corridor floor, or safe branch leading toward the target context.
4. Do not select the visual center of the object unless that point is also navigable floor.
5. Prefer target-context cues such as bathroom doorways for toilets, media/living areas for TVs, bed/nightstand areas for beds, and plant pots/floor near plants.
6. Avoid walls, furniture surfaces, obstacles, mirrors, windows, and unreachable object centers.
7. Set Flag true only when the target object is unambiguously visible in the selected view.
8. Keep Reason concise, at most 20 words.

Return exactly one single-line JSON object:
{"Reason":"...","Angle":<int>,"Point":[<int>,<int>],"Flag":<true|false>,"Confidence":"high|medium|low"}

No markdown, no prose, no extra keys.
"""


@dataclass
class QwenPointPlannerConfig:
    point_radius: int = 8
    retries: int = 10
    panel_width: int = 430


PRIOR_KEYS = ("Supports", "StrongCooccurs", "Gateways", "Lookalikes")
PRIORS_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "target_context_priors",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                key: {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 10,
                }
                for key in PRIOR_KEYS
            },
            "required": list(PRIOR_KEYS),
            "additionalProperties": False,
        },
    },
}
QWEN_PRIORS_COMPACT_PROMPT = (
    "Return exactly one JSON object with Supports, StrongCooccurs, Gateways, "
    "and Lookalikes arrays. Choose at most four exact lowercase strings per "
    "array from VALID_CLASSES. Supports are physical supports; StrongCooccurs "
    "are nearby objects; Gateways are passages; Lookalikes are visual "
    "confusers. Do not include the target itself. No prose in final content."
)
VALID_PRIOR_CLASSES = frozenset(
    line.strip().lower()
    for line in PRIOR_CLASS_LIST.splitlines()
    if line.strip() and line.strip().upper() != "VALID_CLASSES:"
)

_TARGET_PRIOR_FALLBACKS: Dict[str, Dict[str, List[str]]] = {
    "bed": {
        "Supports": ["nightstand"],
        "StrongCooccurs": ["lamp", "dresser", "wardrobe", "rug"],
        "Gateways": ["gateway"],
        "Lookalikes": ["sofa"],
    },
    "chair": {
        "Supports": [],
        "StrongCooccurs": ["dining table", "desk", "table", "lamp"],
        "Gateways": ["gateway"],
        "Lookalikes": ["sofa"],
    },
    "houseplant": {
        "Supports": ["pot"],
        "StrongCooccurs": ["window", "rug", "table"],
        "Gateways": ["gateway"],
        "Lookalikes": [],
    },
    "plant": {
        "Supports": ["pot"],
        "StrongCooccurs": ["window", "rug", "table"],
        "Gateways": ["gateway"],
        "Lookalikes": [],
    },
    "sofa": {
        "Supports": [],
        "StrongCooccurs": ["coffee table", "tv stand", "media console", "rug"],
        "Gateways": ["gateway"],
        "Lookalikes": ["chair"],
    },
    "toilet": {
        "Supports": ["floor"],
        "StrongCooccurs": ["sink", "bathtub", "mirror", "trash bin"],
        "Gateways": ["gateway"],
        "Lookalikes": ["sink"],
    },
    "tv monitor": {
        "Supports": ["tv stand", "media console"],
        "StrongCooccurs": ["sofa", "speaker", "coffee table"],
        "Gateways": ["gateway"],
        "Lookalikes": ["monitor", "picture frame"],
    },
    "tv_monitor": {
        "Supports": ["tv stand", "media console"],
        "StrongCooccurs": ["sofa", "speaker", "coffee table"],
        "Gateways": ["gateway"],
        "Lookalikes": ["monitor", "picture frame"],
    },
}


def _empty_priors() -> Dict[str, List[str]]:
    return {key: [] for key in PRIOR_KEYS}


def fallback_priors_for_target(target: str) -> Dict[str, List[str]]:
    normalized = str(target or "").strip().lower().replace("_", " ")
    priors = _TARGET_PRIOR_FALLBACKS.get(normalized)
    if priors is None:
        priors = {"Supports": [], "StrongCooccurs": [], "Gateways": ["gateway"], "Lookalikes": []}
    return {key: list(priors.get(key, [])) for key in PRIOR_KEYS}


def qwen_priors_thinking_enabled() -> bool:
    model_identity = " ".join(
        str(os.environ.get(name) or "")
        for name in ("QWEN_MODEL", "VOCA_QWEN_MODEL_ROOT")
    ).lower()
    return "thinking" in model_identity


def prior_candidate_shortlist_for_target(target: str) -> List[str]:
    normalized_target = str(target or "").strip().lower().replace("_", " ")
    fallback = fallback_priors_for_target(normalized_target)
    values: List[str] = []
    seen = set()
    for key in PRIOR_KEYS:
        for raw_value in fallback.get(key, []):
            value = str(raw_value or "").strip().lower()
            if (
                value
                and value != normalized_target
                and value in VALID_PRIOR_CLASSES
                and value not in seen
            ):
                values.append(value)
                seen.add(value)
    for value in ("floor", "gateway"):
        if value != normalized_target and value not in seen:
            values.append(value)
            seen.add(value)
    return values


def merge_target_priors(
    primary: Dict[str, Any],
    supplemental: Dict[str, Any],
) -> Dict[str, List[str]]:
    """Preserve model priors while guaranteeing conservative target-specific cues."""
    merged: Dict[str, List[str]] = {}
    for key in PRIOR_KEYS:
        values: List[str] = []
        seen = set()
        for source in (primary, supplemental):
            raw_values = source.get(key, []) if isinstance(source, dict) else []
            if not isinstance(raw_values, (list, tuple)):
                continue
            for value in raw_values:
                text = str(value or "").strip().lower()
                if text and text not in seen:
                    values.append(text)
                    seen.add(text)
        merged[key] = values
    return merged


def filter_detector_prior_classes(
    priors: Dict[str, Any],
) -> Tuple[Dict[str, List[str]], List[str]]:
    filtered: Dict[str, List[str]] = {key: [] for key in PRIOR_KEYS}
    rejected: List[str] = []
    for key in PRIOR_KEYS:
        values = priors.get(key, []) if isinstance(priors, dict) else []
        if not isinstance(values, (list, tuple)):
            continue
        for value in values:
            text = str(value or "").strip().lower()
            if not text:
                continue
            if text not in VALID_PRIOR_CLASSES:
                rejected.append("{}:{}".format(key, text))
                continue
            if text not in filtered[key]:
                filtered[key].append(text)
    return filtered, rejected


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "t"}
    return False


def _balanced_json_blocks(text: str) -> List[str]:
    blocks: List[str] = []
    depth = 0
    start = -1
    in_string = False
    quote_ch = ""
    escaped = False
    for i, ch in enumerate(text or ""):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote_ch:
                in_string = False
            continue
        if ch in {"'", '"'}:
            in_string = True
            quote_ch = ch
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                blocks.append(text[start : i + 1])
                start = -1
    return blocks


def _json_dict_from_text(raw_text: str) -> Optional[Dict[str, Any]]:
    raw_text = (raw_text or "").strip()
    candidates = [raw_text] + _balanced_json_blocks(raw_text)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            if isinstance(obj.get("Answer"), dict):
                return dict(obj["Answer"])
            return dict(obj)
    return None


def _pick_first(data: Dict[str, Any], names: Sequence[str], default=None):
    for name in names:
        if name in data:
            return data[name]
        for key, value in data.items():
            if str(key).lower() == name.lower():
                return value
    return default


def _trim_words(text: Any, max_words: int = 30) -> str:
    words = str(text or "").strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _lower_center_point(image_shape: Sequence[int]) -> List[int]:
    h, w = int(image_shape[0]), int(image_shape[1])
    return [max(0, w // 2), max(0, min(h - 1, int(round(h * 0.8))))]


def _normalize_confidence(value: Any, fallback: bool = False) -> str:
    if fallback:
        return "fallback"
    text = str(value or "medium").strip().lower()
    return text if text in {"high", "medium", "low"} else "medium"


def _normalize_point(value: Any, image_shape: Sequence[int]) -> Optional[List[int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        u = int(round(float(value[0])))
        v = int(round(float(value[1])))
    except Exception:
        return None
    h, w = int(image_shape[0]), int(image_shape[1])
    if 0 <= u < w and 0 <= v < h:
        return [u, v]
    return None


def _normalize_angle(value: Any, valid_angles: Iterable[int]) -> Optional[int]:
    valid = {int(angle) for angle in valid_angles}
    try:
        angle = int(value)
    except Exception:
        return None
    return angle if angle in valid else None


def parse_qwen_point_decision(
    raw_text: str,
    *,
    image_shape: Sequence[int],
    valid_angles: Iterable[int],
) -> Dict[str, Any]:
    data = _json_dict_from_text(raw_text) or {}
    fallback_reasons: List[str] = []

    angle = _normalize_angle(_pick_first(data, ["Angle", "angle"]), valid_angles)
    if angle is None:
        angle = int(next(iter(valid_angles), 0))
        fallback_reasons.append("angle_invalid")

    point_raw = _pick_first(
        data,
        ["Point", "point", "SelectedPoint", "selected_point", "selected_image_point"],
    )
    point = _normalize_point(point_raw, image_shape)
    if point is None:
        point = _lower_center_point(image_shape)
        fallback_reasons.append("point_invalid")

    fallback = bool(fallback_reasons)
    return {
        "Reason": _trim_words(_pick_first(data, ["Reason", "reason", "Rationale", "rationale"], "")),
        "Angle": int(angle),
        "Point": point,
        "Flag": _to_bool(_pick_first(data, ["Flag", "flag", "Found", "found"], False)),
        "Confidence": _normalize_confidence(
            _pick_first(data, ["Confidence", "confidence"], "medium"),
            fallback=fallback,
        ),
        "fallback": fallback,
        "fallback_reason": ",".join(fallback_reasons),
        "raw": raw_text or "",
    }


def make_point_goal_mask(image_shape: Sequence[int], point: Sequence[int], radius: int = 8) -> np.ndarray:
    h, w = int(image_shape[0]), int(image_shape[1])
    u = max(0, min(w - 1, int(point[0])))
    v = max(0, min(h - 1, int(point[1])))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(
        mask,
        (max(0, u - radius), max(0, v - radius)),
        (min(w - 1, u + radius), min(h - 1, v + radius)),
        255,
        -1,
    )
    return mask


def draw_selected_point(rgb: np.ndarray, point: Sequence[int], radius: int = 8) -> np.ndarray:
    out = np.array(rgb, copy=True)
    h, w = out.shape[:2]
    u = max(0, min(w - 1, int(point[0])))
    v = max(0, min(h - 1, int(point[1])))
    cv2.rectangle(
        out,
        (max(0, u - radius), max(0, v - radius)),
        (min(w - 1, u + radius), min(h - 1, v + radius)),
        (255, 0, 0),
        2,
    )
    cv2.circle(out, (u, v), 4, (255, 0, 0), -1)
    return out


def _wrap_text(text: str, max_chars: int = 34) -> List[str]:
    words = str(text or "").split()
    lines: List[str] = []
    current: List[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) > max_chars and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def fallback_display_reason(reason: Any, max_chars: int = 42) -> str:
    text = str(reason or "yes").strip()
    if text.startswith("point_verifier:"):
        parts = text.split(":", 2)
        text = ":".join(parts[:2])
    limit = max(8, int(max_chars))
    if len(text) > limit:
        text = "{}...".format(text[: limit - 3])
    return text


def _put_panel_line(panel: np.ndarray, text: str, y: int, scale: float = 0.55, color=(230, 230, 230)) -> int:
    cv2.putText(panel, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return y + int(25 * scale / 0.55)


def decision_debug_point(decision: Optional[Dict[str, Any]]) -> Optional[List[int]]:
    decision = decision if isinstance(decision, dict) else {}
    verification = (
        decision.get("selected_point_verification")
        if isinstance(decision.get("selected_point_verification"), dict)
        else {}
    )
    point = verification.get("point_px") if verification.get("triggered") else None
    direct_output = decision.get("vlm_output")
    if point is None and isinstance(direct_output, dict) and direct_output.get("action") == "go":
        point = direct_output.get("selected_image_point")
    if point is None and not isinstance(direct_output, dict):
        point = decision.get("Point")
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    try:
        return [int(round(float(point[0]))), int(round(float(point[1])))]
    except Exception:
        return None


def _has_drawable_selected_point(decision: Dict[str, Any]) -> bool:
    return decision_debug_point(decision) is not None


def decision_action_label(decision: Optional[Dict[str, Any]]) -> str:
    decision = decision if isinstance(decision, dict) else {}
    output = decision.get("vlm_output") if isinstance(decision.get("vlm_output"), dict) else {}
    action = str(output.get("action") or decision.get("action") or "").strip().upper()
    return action if action else "UNKNOWN"


def executed_action_label(decision: Optional[Dict[str, Any]]) -> str:
    decision = decision if isinstance(decision, dict) else {}
    action = str(decision.get("runner_action") or "").strip().upper()
    return action if action else decision_action_label(decision)


def candidate_validation_lines(decision: Optional[Dict[str, Any]]) -> List[str]:
    decision = decision if isinstance(decision, dict) else {}
    validation = (
        decision.get("pixel_candidate_validation")
        if isinstance(decision.get("pixel_candidate_validation"), dict)
        else {}
    )
    candidate = validation.get("candidate") if isinstance(validation.get("candidate"), dict) else {}
    requested_ref = (
        decision.get("selected_candidate_ref")
        or validation.get("requested_ref")
        or candidate.get("candidate_ref")
    )
    if not validation and not requested_ref:
        return []
    marker = str(candidate.get("marker_id") or "").strip()
    ref = str(requested_ref or "-").strip()
    candidate_text = "Candidate: {}{}".format(
        "{} ".format(marker) if marker else "",
        ref,
    )
    reason = str(validation.get("reason") or "not_checked")
    gate = "PASS" if bool(validation.get("passed")) else "REJECT"
    lines = [candidate_text[:58], "Gate: {} {}".format(gate, reason)[:58]]
    point_verification = (
        decision.get("selected_point_verification")
        if isinstance(decision.get("selected_point_verification"), dict)
        else {}
    )
    if point_verification:
        if point_verification.get("triggered"):
            visual_gate = "PASS" if bool(point_verification.get("passed")) else "REJECT"
            visual_detail = str(
                point_verification.get("surface")
                or point_verification.get("reason")
                or "unknown"
            )
        else:
            visual_gate = "SKIP"
            visual_detail = "structured_rgb"
        lines.append("Visual: {} {}".format(visual_gate, visual_detail)[:58])
    topological_ref = (
        decision.get("selected_topological_candidate_ref")
        or candidate.get("topological_candidate_ref")
    )
    if topological_ref:
        lines.append("Memory edge: {}".format(topological_ref)[:58])
    return lines


def _action_color(action_label: str):
    if action_label == "GO":
        return (80, 220, 120)
    if action_label == "ROTATE":
        return (120, 170, 255)
    if action_label == "STOP":
        return (255, 100, 100)
    if action_label == "REQUEST_OBSERVATION":
        return (255, 210, 80)
    return (190, 190, 190)


def memory_visualizer_lines(memory_state: Optional[Dict[str, Any]], max_lines: int = 6) -> List[str]:
    state = memory_state if isinstance(memory_state, dict) else {}
    if not state or not state.get("enabled"):
        return ["Memory inactive"]
    lines = ["Memory v6"]
    lines.append("nodes={} edges={}".format(int(state.get("num_nodes", 0) or 0), int(state.get("num_edges", 0) or 0)))
    lines.append("deadlocks={}".format(int(state.get("num_deadlock_edges", 0) or 0)))
    current = state.get("current_node_id") or "-"
    lines.append("current={}".format(current))
    if state.get("supervisor_mode"):
        lines.append(
            "mode={} no_progress={}".format(
                state.get("supervisor_mode"),
                int(state.get("no_progress_count", 0) or 0),
            )
        )
    strategic = (
        state.get("last_strategic_state")
        if isinstance(state.get("last_strategic_state"), dict)
        else {}
    )
    if strategic:
        lines.append(
            "room={} intent={}".format(
                strategic.get("current_room") or "unknown",
                strategic.get("navigation_intent") or "-",
            )
        )
    event = state.get("last_event") if isinstance(state.get("last_event"), dict) else {}
    if event:
        event_type = str(event.get("event_type") or "event")
        node = event.get("node_id") or event.get("src") or event.get("edge_id") or ""
        suffix = " {}".format(node) if node else ""
        lines.append("last={}{}".format(event_type, suffix))
    return lines[:max_lines]


def draw_memory_graph_thumbnail(
    panel: np.ndarray,
    memory_state: Optional[Dict[str, Any]],
    *,
    top: int,
    left: int = 10,
    right_margin: int = 10,
    bottom_margin: int = 12,
) -> None:
    state = memory_state if isinstance(memory_state, dict) else {}
    height, width = panel.shape[:2]
    right = width - int(right_margin)
    bottom = height - int(bottom_margin)
    top = max(0, min(int(top), bottom - 40))
    cv2.rectangle(panel, (left, top), (right, bottom), (18, 18, 22), -1)
    cv2.rectangle(panel, (left, top), (right, bottom), (56, 56, 66), 1)
    cv2.putText(
        panel,
        "Memory Graph v6",
        (left + 10, top + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 225, 255),
        1,
        cv2.LINE_AA,
    )

    mode_labels = {
        "goal_seek": "GOAL SEEK",
        "escape_deadlock": "ESCAPE",
        "backtrack": "BACKTRACK",
    }
    mode = str(state.get("supervisor_mode") or "-")
    mode_label = mode_labels.get(mode, mode.replace("_", " ").upper())[:14]
    mode_size = cv2.getTextSize(mode_label, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)[0]
    cv2.putText(
        panel,
        mode_label,
        (max(left + 150, right - mode_size[0] - 10), top + 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (145, 155, 170),
        1,
        cv2.LINE_AA,
    )

    layout = state.get("graph_layout") if isinstance(state.get("graph_layout"), dict) else {}
    nodes = [node for node in layout.get("nodes", []) if isinstance(node, dict)]
    edges = [edge for edge in layout.get("edges", []) if isinstance(edge, dict)]
    graph_top = top + 32
    graph_bottom = bottom - 27
    graph_left = left + 24
    graph_right = right - 24

    if not nodes:
        cv2.putText(
            panel,
            "No memory nodes",
            (graph_left, graph_top + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (165, 165, 172),
            1,
            cv2.LINE_AA,
        )
        return

    coordinates = np.asarray(
        [
            [
                float(node.get("x_m", 0.0) or 0.0),
                float(node.get("y_m", 0.0) or 0.0),
            ]
            for node in nodes
        ],
        dtype=np.float64,
    )
    centered_coordinates = coordinates - np.mean(coordinates, axis=0, keepdims=True)
    display_coordinates = centered_coordinates
    if len(nodes) > 1 and float(np.max(np.abs(centered_coordinates))) > 1e-6:
        covariance = centered_coordinates.T @ centered_coordinates
        _, eigenvectors = np.linalg.eigh(covariance)
        primary_axis = eigenvectors[:, -1]
        if primary_axis[0] < 0.0 or (abs(primary_axis[0]) < 1e-6 and primary_axis[1] < 0.0):
            primary_axis = -primary_axis
        secondary_axis = np.asarray([-primary_axis[1], primary_axis[0]], dtype=np.float64)
        display_coordinates = centered_coordinates @ np.column_stack((primary_axis, secondary_axis))
    display_positions = {
        str(node.get("node_id")): display_coordinates[index]
        for index, node in enumerate(nodes)
    }
    xs = [float(position[0]) for position in display_positions.values()]
    ys = [float(position[1]) for position in display_positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    raw_span_x = max_x - min_x
    raw_span_y = max_y - min_y
    span_x = max(2.8, raw_span_x)
    span_y = max(1.5, raw_span_y)
    available_width = max(1, graph_right - graph_left)
    available_height = max(1, graph_bottom - graph_top)
    scale = min(available_width / span_x, available_height / span_y)
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    screen_center_x = (graph_left + graph_right) * 0.5
    screen_center_y = (graph_top + graph_bottom) * 0.5

    def project(node: Dict[str, Any]) -> Tuple[int, int]:
        x, y = display_positions[str(node.get("node_id"))]
        px = int(round(screen_center_x + ((x - center_x) * scale)))
        py = int(round(screen_center_y - ((y - center_y) * scale)))
        return px, py

    points = {str(node.get("node_id")): project(node) for node in nodes}
    node_by_id = {str(node.get("node_id")): node for node in nodes}
    negative_statuses = {"blocked", "collision", "deadlock_entry", "deadlock_entry_candidate", "failed"}
    rendered_edge_keys = set()
    for edge in edges:
        src_id = str(edge.get("src_node_id"))
        dst_id = str(edge.get("dst_node_id"))
        edge_type = str(edge.get("edge_type") or "transition")
        if edge_type.endswith("_reverse"):
            continue
        edge_family = "revisit" if edge_type.startswith("revisit_link") else edge_type
        if edge_family == "revisit":
            edge_key = (edge_family, tuple(sorted((src_id, dst_id))))
        else:
            edge_key = (edge_family, src_id, dst_id)
        if edge_key in rendered_edge_keys:
            continue
        rendered_edge_keys.add(edge_key)

        src = points.get(src_id)
        dst = points.get(dst_id)
        if src is None or dst is None:
            continue
        status = str(edge.get("status") or "").lower()
        color = (90, 95, 105)
        if status in negative_statuses:
            color = (90, 90, 245)
        elif status in {"success", "escape_success"}:
            color = (105, 210, 125)
        elif edge_family == "revisit" or status == "same_place_constraint":
            color = (205, 145, 100)

        dx = float(dst[0] - src[0])
        dy = float(dst[1] - src[1])
        distance = float(np.hypot(dx, dy))
        if distance < 2.0:
            continue
        src_pad = 6.0
        dst_node = node_by_id.get(dst_id, {})
        dst_pad = 7.0 if str(dst_node.get("node_type") or "") == "failed_frontier" else 6.0
        if distance > src_pad + dst_pad + 2.0:
            unit_x = dx / distance
            unit_y = dy / distance
            line_src = (
                int(round(src[0] + unit_x * src_pad)),
                int(round(src[1] + unit_y * src_pad)),
            )
            line_dst = (
                int(round(dst[0] - unit_x * dst_pad)),
                int(round(dst[1] - unit_y * dst_pad)),
            )
        else:
            line_src, line_dst = src, dst
        line_length = max(1.0, float(np.hypot(line_dst[0] - line_src[0], line_dst[1] - line_src[1])))
        if edge_family == "revisit" or status == "same_place_constraint":
            cv2.line(panel, line_src, line_dst, color, 1, cv2.LINE_AA)
        elif status in negative_statuses:
            cv2.line(panel, line_src, line_dst, color, 1, cv2.LINE_AA)
        else:
            tip_length = max(0.015, min(0.14, 4.5 / line_length))
            cv2.arrowedLine(panel, line_src, line_dst, color, 1, cv2.LINE_AA, tipLength=tip_length)

    labeled_rectangles: List[Tuple[int, int, int, int]] = []
    crowded = len(nodes) > 7
    first_place_id = next(
        (
            str(node.get("node_id"))
            for node in nodes
            if str(node.get("node_type") or "place") != "failed_frontier"
        ),
        "",
    )
    for node in nodes:
        point = points[str(node.get("node_id"))]
        current = bool(node.get("current"))
        failed_frontier = str(node.get("node_type") or "") == "failed_frontier"
        color = (90, 170, 255) if current else (220, 160, 70)
        if failed_frontier:
            color = (245, 90, 90)
            cv2.rectangle(
                panel,
                (point[0] - 5, point[1] - 5),
                (point[0] + 5, point[1] + 5),
                (10, 10, 12),
                -1,
            )
            cv2.line(panel, (point[0] - 4, point[1] - 4), (point[0] + 4, point[1] + 4), color, 2, cv2.LINE_AA)
            cv2.line(panel, (point[0] - 4, point[1] + 4), (point[0] + 4, point[1] - 4), color, 2, cv2.LINE_AA)
        else:
            cv2.circle(panel, point, 7 if current else 5, (10, 10, 12), -1, cv2.LINE_AA)
            cv2.circle(panel, point, 5 if current else 3, color, -1, cv2.LINE_AA)

        node_id = str(node.get("node_id") or "n")
        if failed_frontier or (crowded and not current and node_id != first_place_id):
            continue
        numeric_id = node_id.rsplit("_", 1)[-1].lstrip("0") or "0"
        label = "N{}".format(numeric_id)
        category = str(node.get("place_category") or "unknown")
        if current and category != "unknown":
            label = "{} {}".format(label, category)
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.31, 1)
        label_candidates = (
            (point[0] + 8, point[1] - 7),
            (point[0] + 8, point[1] + text_size[1] + 7),
            (point[0] - text_size[0] - 8, point[1] - 7),
            (point[0] - text_size[0] - 8, point[1] + text_size[1] + 7),
        )
        label_origin = None
        for candidate_x, candidate_y in label_candidates:
            text_x = max(graph_left, min(graph_right - text_size[0], candidate_x))
            text_y = max(graph_top + text_size[1], min(graph_bottom, candidate_y))
            label_rect = (
                text_x - 2,
                text_y - text_size[1] - 2,
                text_x + text_size[0] + 2,
                text_y + baseline + 2,
            )
            overlaps = any(
                not (
                    label_rect[2] < placed[0]
                    or label_rect[0] > placed[2]
                    or label_rect[3] < placed[1]
                    or label_rect[1] > placed[3]
                )
                for placed in labeled_rectangles
            )
            if not overlaps:
                label_origin = (text_x, text_y)
                labeled_rectangles.append(label_rect)
                break
        if label_origin is not None:
            cv2.putText(
                panel,
                label,
                label_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.31,
                (205, 205, 210),
                1,
                cv2.LINE_AA,
            )

    failed_count = sum(1 for node in nodes if str(node.get("node_type") or "") == "failed_frontier")
    status_text = "{} places  |  {} blocked  |  {} edges".format(
        len(nodes) - failed_count,
        failed_count,
        len(edges),
    )
    cv2.putText(
        panel,
        status_text[:58],
        (left + 10, bottom - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.33,
        (180, 180, 188),
        1,
        cv2.LINE_AA,
    )


def render_qwen_debug_frame(
    rgb: np.ndarray,
    *,
    decision: Optional[Dict[str, Any]],
    target: str,
    planner_name: str,
    draw_point: bool = False,
    panel_width: int = 430,
) -> np.ndarray:
    left = np.array(rgb, copy=True)
    debug_point = decision_debug_point(decision)
    if decision and draw_point and debug_point is not None:
        left = draw_selected_point(left, debug_point)

    h = left.shape[0]
    panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
    panel[:, :] = (24, 24, 28)
    memory_state = decision.get("memory_visualizer") if isinstance(decision, dict) else None
    graph_panel_height = min(max(150, int(round(h * 0.34))), max(40, h - 12))
    graph_panel_top = max(0, h - 12 - graph_panel_height)
    cv2.putText(
        panel,
        "Qwen Decision",
        (16, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    y = 58
    if decision:
        vlm_action_label = decision_action_label(decision)
        action_label = executed_action_label(decision)
        badge_color = _action_color(action_label)
        badge_scale = 0.42
        badge_size = cv2.getTextSize(action_label, cv2.FONT_HERSHEY_SIMPLEX, badge_scale, 1)[0]
        badge_left = max(190, panel_width - badge_size[0] - 34)
        cv2.rectangle(panel, (badge_left, 10), (panel_width - 16, 37), badge_color, -1)
        cv2.putText(
            panel,
            action_label,
            (badge_left + 9, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            badge_scale,
            (20, 20, 24),
            1,
            cv2.LINE_AA,
        )
    y = _put_panel_line(
        panel,
        "Target: {} | {}".format(target or "-", planner_name),
        y,
        scale=0.52,
    )

    if decision:
        if action_label != vlm_action_label:
            y = _put_panel_line(
                panel,
                "VLM requested: {}".format(vlm_action_label),
                y,
                scale=0.42,
                color=(255, 205, 90),
            )
        strategic = (
            decision.get("strategic_state")
            if isinstance(decision.get("strategic_state"), dict)
            else {}
        )
        if strategic:
            y = _put_panel_line(
                panel,
                "Room: {} | {} | {}".format(
                    strategic.get("current_room") or "unknown",
                    strategic.get("target_evidence") or "none",
                    strategic.get("navigation_intent") or "-",
                )[:60],
                y,
                scale=0.40,
                color=(180, 225, 255),
            )
        point = debug_point or decision.get("Point") or ["-", "-"]
        y = _put_panel_line(
            panel,
            "Angle: {} | Point: ({}, {})".format(
                decision.get("Angle", "-"),
                point[0],
                point[1],
            ),
            y,
            scale=0.50,
        )
        y = _put_panel_line(
            panel,
            "Confidence: {}".format(decision.get("Confidence", "-")),
            y,
            scale=0.48,
        )

        optional_lines = []
        for line in candidate_validation_lines(decision):
            if line.startswith("Visual: SKIP"):
                continue
            if line.startswith(("Gate: PASS", "Visual: PASS")):
                color = (120, 235, 150)
            elif line.startswith(("Gate: REJECT", "Visual: REJECT")):
                color = (255, 140, 110)
            else:
                color = (255, 205, 90)
            optional_lines.append((line[:50], color))
        if decision.get("fallback"):
            optional_lines.append(
                (
                    "Fallback: {}".format(
                        fallback_display_reason(decision.get("fallback_reason", "yes"), max_chars=34)
                    ),
                    (255, 200, 80),
                )
            )

        reason_line_height = 20
        reason_lines = _wrap_text(str(decision.get("Reason") or "-"), max_chars=40)
        reason_reserve = 4 + 22 + min(2, len(reason_lines)) * reason_line_height + 7
        max_optional_lines = max(0, (graph_panel_top - y - reason_reserve) // 20)
        while len(optional_lines) > max_optional_lines:
            removable_index = next(
                (
                    index
                    for index, (line, _) in enumerate(optional_lines)
                    if line.startswith("Memory edge:")
                ),
                None,
            )
            if removable_index is None:
                removable_index = next(
                    (
                        index
                        for index, (line, _) in enumerate(optional_lines)
                        if line.startswith("Visual: PASS")
                    ),
                    None,
                )
            if removable_index is None:
                removable_index = len(optional_lines) - 1
            optional_lines.pop(removable_index)
        for line, color in optional_lines:
            y = _put_panel_line(panel, line, y, scale=0.44, color=color)

        y += 4
        if graph_panel_top - y >= 42:
            y = _put_panel_line(panel, "Reason:", y, scale=0.48, color=(255, 255, 255))
            available_reason_lines = max(0, min(2, (graph_panel_top - y - 7) // reason_line_height))
            for line in reason_lines[:available_reason_lines]:
                y = _put_panel_line(panel, line, y, scale=0.44)
    else:
        y += 12
        _put_panel_line(panel, "No active Qwen decision yet", y, color=(180, 180, 180))

    draw_memory_graph_thumbnail(panel, memory_state, top=graph_panel_top)

    return np.concatenate([left, panel], axis=1)


class QwenPointPlanner:
    planner_name = "qwen_point"

    def __init__(
        self,
        *,
        text_fn: Optional[Callable[..., str]] = None,
        vision_fn: Optional[Callable[..., str]] = None,
        config: Optional[QwenPointPlannerConfig] = None,
    ):
        self._uses_default_text_fn = text_fn is None
        self.text_fn = text_fn or text_response
        self.vision_fn = vision_fn or vision_response
        self.config = config or QwenPointPlannerConfig()
        self.object_goal = ""
        self.latest_priors: Dict[str, List[str]] = {}
        self.priors_log: List[Dict[str, Any]] = []
        self.priors_success_count = 0
        self.priors_parse_fail_count = 0
        self.priors_fallback_count = 0
        self.priors_last_error = ""
        self.qwen_call_log: List[Dict[str, Any]] = []
        self.last_decision: Optional[Dict[str, Any]] = None
        self.llm_call_count = 0
        self.llm_success_count = 0
        self.llm_error_count = 0
        self.llm_last_error = ""
        self.llm_durations: List[float] = []
        self.priors_durations: List[float] = []
        self.navigation_vlm_durations: List[float] = []
        self.planner_durations: List[float] = []
        self.yoloe_durations: List[float] = []
        self._last_bboxes: List[Dict[str, Any]] = []
        self._qwen_call_log_path: Optional[str] = None

    def set_qwen_call_log_path(self, path: str) -> None:
        self._qwen_call_log_path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        Path(path).write_text("", encoding="utf-8")

    def _flush_qwen_call(self, item: Dict[str, Any]) -> None:
        if not self._qwen_call_log_path:
            return
        with open(self._qwen_call_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def reset(self, object_goal):
        if object_goal == "tv_monitor":
            self.object_goal = "tv monitor"
        elif object_goal == "plant":
            self.object_goal = "houseplant"
        else:
            self.object_goal = str(object_goal)
        self.latest_priors = {}
        self.priors_log = []
        self.priors_success_count = 0
        self.priors_parse_fail_count = 0
        self.priors_fallback_count = 0
        self.priors_last_error = ""
        self.qwen_call_log = []
        self.last_decision = None
        self.llm_call_count = 0
        self.llm_success_count = 0
        self.llm_error_count = 0
        self.llm_last_error = ""
        self.llm_durations = []
        self.priors_durations = []
        self.navigation_vlm_durations = []
        self.planner_durations = []
        self.yoloe_durations = []
        self._last_bboxes = []

    def _record_llm_exception(self, stage: str, try_idx: int, exc: Exception):
        self.llm_error_count += 1
        self.llm_last_error = "{}[try={}] {}: {}".format(stage, try_idx + 1, type(exc).__name__, str(exc))
        print("[LLM/{}] ERROR try {}/{}: {}".format(stage, try_idx + 1, self.config.retries, self.llm_last_error))

    def concat_panoramic(self, images, angles):
        try:
            height, width = images[0].shape[0], images[0].shape[1]
        except Exception:
            height, width = 480, 640
        cols = min(4, max(1, len(images)))
        rows = int(np.ceil(len(images) / float(cols)))
        background_image = np.zeros((rows * height + (rows + 1) * 10, cols * width + (cols + 1) * 10, 3), np.uint8)
        copy_images = [np.array(image, dtype=np.uint8, copy=True) for image in images]
        for i in range(len(copy_images)):
            copy_images[i] = cv2.putText(
                copy_images[i],
                "Angle {}".format(int(angles[i])),
                (max(10, width // 8), max(32, height // 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.5, min(width, height) / 240.0),
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            row = i // cols
            col = i % cols
            background_image[
                10 * (row + 1) + row * height : 10 * (row + 1) + row * height + height,
                10 * (col + 1) + col * width : 10 * (col + 1) + col * width + width,
                :,
            ] = copy_images[i]
        return background_image

    def _priors_prompt_text(self) -> str:
        pri = self.latest_priors or {}
        return json.dumps(
            {
                "Supports": pri.get("Supports", []),
                "StrongCooccurs": pri.get("StrongCooccurs", []),
                "Gateways": pri.get("Gateways", []),
                "Lookalikes": pri.get("Lookalikes", []),
            },
            ensure_ascii=False,
        )

    def query_priors_text(self):
        priors_thinking_enabled = bool(
            LLM_BACKEND == "qwen" and qwen_priors_thinking_enabled()
        )
        text_content = "<Target Object>:{}\n".format(self.object_goal)
        if not priors_thinking_enabled:
            text_content = "/no_think\n" + text_content
        raw_answer = None
        last_raw_answer = ""
        priors = None
        priors_request_contract: Dict[str, Any] = {
            "backend": str(LLM_BACKEND),
            "response_schema": "target_context_priors_v1",
            "strict_json_schema": bool(LLM_BACKEND == "qwen"),
            "thinking_enabled": bool(priors_thinking_enabled),
        }
        priors_candidate_text = PRIOR_CLASS_LIST
        priors_system_prompt = PRIORS_PROMPT
        if LLM_BACKEND == "qwen" and self._uses_default_text_fn:
            shortlist = prior_candidate_shortlist_for_target(self.object_goal)
            priors_candidate_text = "\nVALID_CLASSES:\n{}\n".format(
                "\n".join(shortlist)
            )
            priors_system_prompt = QWEN_PRIORS_COMPACT_PROMPT
            priors_request_contract["candidate_shortlist"] = list(shortlist)
        for try_idx in range(self.config.retries):
            print("[LLM/PRIORS] try {}/{}".format(try_idx + 1, self.config.retries))
            t0 = time.perf_counter()
            try:
                if LLM_BACKEND == "qwen" and self._uses_default_text_fn:
                    try:
                        priors_max_tokens = max(
                            64,
                            int(
                                os.environ.get(
                                    "VOCA_QWEN_PRIORS_MAX_TOKENS",
                                    "4096" if priors_thinking_enabled else "512",
                                )
                            ),
                        )
                    except ValueError:
                        priors_max_tokens = (
                            4096 if priors_thinking_enabled else 512
                        )
                    priors_request_contract["max_tokens"] = int(
                        priors_max_tokens
                    )
                    raw_answer = self.text_fn(
                        text_content + priors_candidate_text,
                        system_prompt=priors_system_prompt,
                        max_tokens=priors_max_tokens,
                        extra_payload={
                            "response_format": PRIORS_RESPONSE_FORMAT,
                            "chat_template_kwargs": {
                                "enable_thinking": bool(priors_thinking_enabled)
                            },
                        },
                    )
                else:
                    raw_answer = self.text_fn(
                        text_content + PRIOR_CLASS_LIST,
                        system_prompt=PRIORS_PROMPT,
                    )
                if raw_answer:
                    self.llm_success_count += 1
            except Exception as exc:
                raw_answer = None
                self._record_llm_exception("PRIORS", try_idx, exc)
            finally:
                self.llm_call_count += 1
                dt = time.perf_counter() - t0
                self.llm_durations.append(dt)
                self.priors_durations.append(dt)
                print("[LLM/PRIORS] elapsed={:.3f}s has_response={}".format(dt, bool(raw_answer)))
            if not raw_answer:
                continue
            last_raw_answer = raw_answer
            try:
                parsed = parse_llm_json(raw_answer)
            except Exception as exc:
                parsed = None
                self.priors_last_error = "priors_parse_exception: {}".format(type(exc).__name__)
            if (
                isinstance(parsed, dict)
                and priors_thinking_enabled
                and "relaxed_parse_used" in list(parsed.get("violations") or [])
            ):
                parsed = None
                self.priors_last_error = "priors_truncated_reasoning_rejected"
            if not parsed:
                self.priors_parse_fail_count += 1
                if not self.priors_last_error:
                    self.priors_last_error = "priors_parse_failed"
                print("[LLM/PRIORS] parse failed, retrying.")
                continue
            priors = extract_priors(parsed, raw_answer) or {}
            for key in PRIOR_KEYS:
                priors.setdefault(key, [])
            model_priors, allowlist_rejections = filter_detector_prior_classes(
                priors
            )
            priors = merge_target_priors(
                model_priors,
                fallback_priors_for_target(self.object_goal),
            )
            self.priors_success_count += 1
            self.priors_last_error = ""
            self.priors_log.append(
                {
                    "target": self.object_goal,
                    "priors": priors,
                    "model_priors": model_priors,
                    "allowlist_rejections": allowlist_rejections,
                    "target_fallback_augmented": priors != model_priors,
                    "raw": raw_answer,
                    "call_index": try_idx,
                    "call_type": "text_priors",
                    "fallback": False,
                    "fallback_reason": "",
                    "request_contract": dict(priors_request_contract),
                }
            )
            print("[LLM/PRIORS] sizes:", len(priors["Supports"]), len(priors["StrongCooccurs"]), len(priors["Gateways"]), len(priors["Lookalikes"]))
            break

        if not isinstance(priors, dict):
            fallback_reason = self.priors_last_error or "priors_no_response"
            priors = fallback_priors_for_target(self.object_goal)
            self.priors_fallback_count += 1
            self.priors_last_error = fallback_reason
            self.priors_log.append(
                {
                    "target": self.object_goal,
                    "priors": priors,
                    "raw": last_raw_answer,
                    "call_index": max(0, self.config.retries - 1),
                    "call_type": "text_priors",
                    "fallback": True,
                    "fallback_reason": fallback_reason,
                    "parse_fail_count": self.priors_parse_fail_count,
                    "request_contract": dict(priors_request_contract),
                }
            )
            print("[LLM/PRIORS] fallback={} target={}".format(fallback_reason, self.object_goal))
        self.latest_priors = priors
        return priors

    def _build_point_prompt(self, angles: Sequence[int], image_shape: Sequence[int]) -> str:
        h, w = int(image_shape[0]), int(image_shape[1])
        return (
            "<Target Object>:{}\n"
            "<Target Context Cues>:{}\n"
            "<Valid Angles>:{}\n"
            "<Selected View Size>: image_width={} image_height={}\n"
            "<Point Bounds>: 0 <= u < {}, 0 <= v < {}\n"
            "Select the best Angle and Point in that selected single view."
        ).format(self.object_goal, self._priors_prompt_text(), list(map(int, angles)), w, h, w, h)

    def _query_point_decision(self, pano_images: Sequence[np.ndarray], angles: Sequence[int], call_type: str) -> Dict[str, Any]:
        inference_image = cv2.cvtColor(self.concat_panoramic(pano_images, angles), cv2.COLOR_BGR2RGB)
        selected_shape = pano_images[0].shape if pano_images else (480, 640, 3)
        raw_answer = None
        decision = None

        for try_idx in range(self.config.retries):
            print("[LLM/QWEN_POINT] try {}/{}".format(try_idx + 1, self.config.retries))
            t0 = time.perf_counter()
            try:
                raw_answer = self.vision_fn(self._build_point_prompt(angles, selected_shape), inference_image, QWEN_POINT_PROMPT)
                if raw_answer:
                    self.llm_success_count += 1
            except Exception as exc:
                raw_answer = None
                self._record_llm_exception("QWEN_POINT", try_idx, exc)
            finally:
                self.llm_call_count += 1
                dt = time.perf_counter() - t0
                self.llm_durations.append(dt)
                self.navigation_vlm_durations.append(dt)
                print("[LLM/QWEN_POINT] elapsed={:.3f}s has_response={}".format(dt, bool(raw_answer)))

            if not raw_answer:
                continue
            decision = parse_qwen_point_decision(
                raw_answer,
                image_shape=selected_shape,
                valid_angles=angles,
            )
            if decision:
                break

        if not decision:
            decision = parse_qwen_point_decision(
                "{}",
                image_shape=selected_shape,
                valid_angles=angles,
            )

        decision.update(
            {
                "target": self.object_goal,
                "call_type": call_type,
                "priors": self.latest_priors,
                "angles": list(map(int, angles)),
            }
        )
        self.last_decision = decision
        self.qwen_call_log.append(decision.copy())
        self._flush_qwen_call(decision)
        print("[QWEN_POINT] angle={} point={} confidence={} fallback={}".format(
            decision.get("Angle"),
            decision.get("Point"),
            decision.get("Confidence"),
            decision.get("fallback"),
        ))
        if decision.get("Reason"):
            print("[QWEN_POINT] reason:", decision.get("Reason"))
        return decision

    def _decision_to_goal(self, frame: np.ndarray, decision: Dict[str, Any]):
        point = decision["Point"]
        goal_mask = make_point_goal_mask(frame.shape, point, radius=self.config.point_radius)
        vis_rgb = draw_selected_point(frame, point, radius=self.config.point_radius)
        self._last_bboxes = [
            {
                "cls": "qwen_point",
                "cls_id": None,
                "score": None,
                "bbox_xyxy": [
                    max(0, int(point[0]) - self.config.point_radius),
                    max(0, int(point[1]) - self.config.point_radius),
                    min(frame.shape[1], int(point[0]) + self.config.point_radius),
                    min(frame.shape[0], int(point[1]) + self.config.point_radius),
                ],
                "center": [float(point[0]), float(point[1])],
                "norm": {
                    "cx": float(point[0]) / max(1, frame.shape[1]),
                    "cy": float(point[1]) / max(1, frame.shape[0]),
                    "area": 0.0,
                },
            }
        ]
        return frame.copy(), goal_mask, vis_rgb, self._last_bboxes

    def make_plan_from_views(self, pano_images, angles, call_type: str = "make_plan"):
        t0 = time.perf_counter()
        angles = [int(angle) for angle in angles]
        decision = self._query_point_decision(pano_images, angles, call_type=call_type)
        try:
            direction = angles.index(int(decision["Angle"]))
        except ValueError:
            direction = 0
        direction = max(0, min(len(pano_images) - 1, direction))
        frame = pano_images[direction]
        goal_rgb, goal_mask, vis_rgb, _ = self._decision_to_goal(frame, decision)
        self.planner_durations.append(time.perf_counter() - t0)
        flag = bool(decision.get("Flag", False))
        return goal_rgb, goal_mask, vis_rgb.copy(), vis_rgb, int(direction), flag, flag

    def make_plan(self, pano_images):
        angles = [i * 30 for i in range(len(pano_images))]
        return self.make_plan_from_views(pano_images, angles, call_type="make_plan")

    def apply_priors_on_image(
        self,
        image_or_pano,
        conf_threshold: float = 0.05,
        MIN_TARGET_CONF: float = 0.5,
        MIN_TARGET_AREA: float = 0.02,
        iou_threshold: float = 0.50,
        return_boxes: bool = False,
    ):
        if isinstance(image_or_pano, (list, tuple)):
            if len(image_or_pano) == 7:
                angles = [-90, -60, -30, 0, 30, 60, 90]
            else:
                angles = [i * 30 for i in range(len(image_or_pano))]
            decision = self._query_point_decision(image_or_pano, angles, call_type="apply_priors_on_image")
            try:
                best_idx = angles.index(int(decision["Angle"]))
            except ValueError:
                best_idx = 0
            best_idx = max(0, min(len(image_or_pano) - 1, best_idx))
            frame = image_or_pano[best_idx]
            goal_rgb, goal_mask, vis_rgb, boxes = self._decision_to_goal(frame, decision)
            flag = bool(decision.get("Flag", False))
            if return_boxes:
                return goal_rgb, goal_mask, flag, flag, vis_rgb, boxes, int(best_idx)
            return goal_rgb, goal_mask, flag, flag, vis_rgb, int(best_idx)

        decision = {
            "Point": _lower_center_point(image_or_pano.shape),
            "Angle": 0,
            "Flag": False,
            "Confidence": "fallback",
            "Reason": "single-image lower-center fallback",
            "fallback": True,
            "fallback_reason": "single_image_no_vlm",
        }
        self.last_decision = decision
        goal_rgb, goal_mask, vis_rgb, boxes = self._decision_to_goal(image_or_pano, decision)
        if return_boxes:
            return goal_rgb, goal_mask, False, False, vis_rgb, boxes
        return goal_rgb, goal_mask, False, False, vis_rgb

    def are_bboxes_similar(
        self,
        prev_boxes: List[Dict[str, Any]],
        curr_boxes: List[Dict[str, Any]],
        *,
        iou_thresh: float = 0.70,
        center_tol: float = 0.15,
        area_tol: float = 0.20,
        min_match_ratio: float = 0.80,
        class_sensitive: bool = True,
        ignore_classes: Optional[List[str]] = None,
        max_count_delta: int = 0,
        return_detail: bool = False,
    ) -> Union[bool, Tuple[bool, Dict[str, Any]]]:
        if not prev_boxes or not curr_boxes:
            result = False
            detail = {"reason": "missing_boxes"}
            return (result, detail) if return_detail else result
        pa = prev_boxes[0].get("norm", {})
        ca = curr_boxes[0].get("norm", {})
        dist = float(np.hypot(float(pa.get("cx", 0.0)) - float(ca.get("cx", 0.0)), float(pa.get("cy", 0.0)) - float(ca.get("cy", 0.0))))
        result = dist <= center_tol
        detail = {"center_distance": dist}
        return (result, detail) if return_detail else result

    def save_qwen_calls(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in self.qwen_call_log:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
