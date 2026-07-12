import json
from typing import Any, Dict, List, Optional

from navigation_supervisor import assert_policy_input_safe, normalize_supervisor_mode


COMPACT_MEMORY_SCHEMA_VERSION = "compact_nav_memory_v2"
MAX_CANDIDATE_EXITS = 4
MAX_DIRECTIONAL_FAILURES = 4
MAX_NEGATIVE_MEMORIES = 3
MAX_REVISIT_CANDIDATES = 3
MAX_TARGET_REJECTIONS = 3
MAX_REASON_CHARS = 120


def _mapping(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _items(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, dict)]


def _text(value: Any, limit: int = MAX_REASON_CHARS) -> str:
    text = " ".join(str(value or "").split())
    return text[: max(0, int(limit))]


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _optional_float(value: Any) -> Optional[float]:
    try:
        return round(float(value), 4)
    except Exception:
        return None


def _candidate_exits(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = _items(_mapping(memory.get("candidate_refs")).get("exits"))
    candidates.sort(
        key=lambda item: (
            bool(item.get("avoid")),
            -_float(item.get("score")),
            str(item.get("candidate_ref") or ""),
        )
    )
    projected = []
    for item in candidates[:MAX_CANDIDATE_EXITS]:
        point = item.get("point_px")
        point_px = None
        if isinstance(point, (list, tuple)) and len(point) == 2:
            point_px = [_int(point[0]), _int(point[1])]
        view_id = item.get("view_id")
        projected.append(
            {
                "candidate_ref": _text(item.get("candidate_ref"), 48),
                "topological_ref": _text(item.get("topological_candidate_ref"), 48),
                "topological_edge_ref": _text(item.get("topological_edge_id"), 48),
                "topological_relation": _text(item.get("topological_relation_type"), 48),
                "marker_id": _text(item.get("marker_id"), 16),
                "view_id": _int(view_id) if view_id is not None else None,
                "view": _text(item.get("view_type_hint"), 24),
                "bearing_deg": _optional_float(item.get("bearing_deg_robot")),
                "point_px": point_px,
                "status": _text(item.get("status"), 48),
                "avoid": bool(item.get("avoid")),
                "score": round(_float(item.get("score")), 4),
                "reason": _text(item.get("reason")),
            }
        )
    return projected


def _directional_failures(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    failures = _items(_mapping(memory.get("voca_sidecar")).get("directional_failures"))
    failures.sort(key=lambda item: _int(item.get("frame_index")), reverse=True)
    projected = []
    for item in failures[:MAX_DIRECTIONAL_FAILURES]:
        point = item.get("point_px")
        point_px = None
        if isinstance(point, (list, tuple)) and len(point) == 2:
            point_px = [_int(point[0]), _int(point[1])]
        view_id = item.get("selected_view_id")
        projected.append(
            {
                "frame_index": _int(item.get("frame_index")),
                "angle_deg": round(_float(item.get("angle_deg")), 3),
                "view_id": _int(view_id) if view_id is not None else None,
                "point_px": point_px,
                "failure_class": _text(item.get("failure_class"), 64),
                "collision_count": max(0, _int(item.get("collision_count"))),
                "translation_m": round(_float(item.get("translation_m")), 4),
            }
        )
    return projected


def _negative_memories(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    negatives = _items(memory.get("compressed_negative_memories"))
    negatives.sort(key=lambda item: -_float(item.get("severity")))
    projected = []
    for item in negatives[:MAX_NEGATIVE_MEMORIES]:
        negative = _mapping(item.get("negative_memory"))
        projected.append(
            {
                "node_ref": _text(item.get("node_id"), 48),
                "reason": _text(negative.get("reason") or item.get("summary")),
                "avoid_scope": _text(negative.get("avoid_scope"), 48),
                "failed_edge_ref": _text(negative.get("failed_entry_edge_id"), 48),
                "severity": round(_float(item.get("severity")), 4),
            }
        )
    return projected


def _revisit_candidates(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = _items(_mapping(memory.get("candidate_refs")).get("revisits"))
    if not candidates:
        candidates = _items(_mapping(memory.get("place_recognition")).get("revisit_candidates"))
    candidates.sort(
        key=lambda item: -_float(
            item.get("visual_retrieval_score", item.get("score", 0.0))
        )
    )
    projected = []
    for item in candidates[:MAX_REVISIT_CANDIDATES]:
        spatial = _mapping(item.get("spatial_plausibility"))
        projected.append(
            {
                "candidate_ref": _text(item.get("candidate_ref"), 48),
                "node_ref": _text(
                    item.get("candidate_node_id", item.get("node_id")),
                    48,
                ),
                "score": round(
                    _float(item.get("visual_retrieval_score", item.get("score", 0.0))),
                    4,
                ),
                "spatially_plausible": bool(spatial.get("accepted")),
            }
        )
    return projected


def _target_rejections(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    feedback = _items(memory.get("runtime_navigation_feedback"))
    projected = []
    for item in reversed(feedback):
        if str(item.get("event") or "") != "target_lookalike_rejected":
            continue
        bbox = item.get("target_bbox_norm")
        bbox_norm = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bbox_norm = [round(_float(value), 4) for value in bbox]
        projected.append(
            {
                "target": _text(item.get("target"), 48),
                "frame_index": _int(item.get("frame_index")),
                "view_id": (
                    _int(item.get("target_view_index"))
                    if item.get("target_view_index") is not None
                    else None
                ),
                "bbox_norm": bbox_norm,
                "lookalike_type": _text(item.get("lookalike_type"), 80),
                "reason": _text(item.get("reason")),
            }
        )
        if len(projected) >= MAX_TARGET_REJECTIONS:
            break
    return projected


def build_compact_memory_projection(vlm_input: Dict[str, Any]) -> Dict[str, Any]:
    memory = _mapping(vlm_input.get("memory"))
    schema_version = str(memory.get("schema_version") or "")
    enabled = bool(memory) and memory.get("enabled") is not False and not schema_version.startswith("null_")
    goal_context = _mapping(memory.get("goal_context"))
    sidecar = _mapping(memory.get("voca_sidecar"))
    harness = _mapping(memory.get("policy_harness_state"))
    mode = normalize_supervisor_mode(
        goal_context.get("supervisor_mode")
        or sidecar.get("supervisor_mode")
        or harness.get("current_stage")
    )
    graph = _mapping(memory.get("graph_summary"))
    deadlock = _mapping(memory.get("deadlock_state"))
    loop = _mapping(memory.get("loop_warning"))
    runtime = _mapping(vlm_input.get("runtime"))
    strategy = _mapping(runtime.get("strategic_runtime"))
    if not strategy:
        strategy = _mapping(sidecar.get("strategic_runtime"))
    last_strategy = _mapping(
        strategy.get("last_strategic_state")
        or sidecar.get("last_strategic_state")
    )
    recent_exit_refs = strategy.get("recent_selected_exit_refs")
    if not isinstance(recent_exit_refs, (list, tuple)):
        recent_exit_refs = []

    projection = {
        "schema_version": COMPACT_MEMORY_SCHEMA_VERSION,
        "enabled": enabled,
        "supervisor_mode": mode,
        "current_stage": _text(
            harness.get("current_stage")
            or goal_context.get("detour_status")
            or "normal_goal_seek",
            48,
        ),
        "no_progress_count": max(0, _int(sidecar.get("no_progress_count"))),
        "graph": {
            "nodes": max(0, _int(graph.get("num_nodes"))),
            "place_nodes": max(0, _int(graph.get("num_place_nodes", graph.get("num_nodes")))),
            "failed_frontiers": max(0, _int(graph.get("num_failed_frontier_nodes"))),
            "edges": max(0, _int(graph.get("num_edges"))),
            "negative_edges": max(0, _int(graph.get("num_deadlock_edges"))),
        },
        "deadlock": {
            "status": _text(deadlock.get("status") or "none", 48),
            "incoming_edge_ref": _text(deadlock.get("incoming_edge_id"), 48),
        },
        "loop": {
            "is_looping": bool(loop.get("is_looping")),
            "repeated_branch_count": max(0, _int(loop.get("repeated_branch_count"))),
        },
        "strategy": {
            "last_room": _text(last_strategy.get("current_room") or "unknown", 32),
            "last_target_evidence": _text(
                last_strategy.get("target_evidence") or "none",
                24,
            ),
            "last_intent": _text(
                last_strategy.get("navigation_intent") or "search_current_room",
                40,
            ),
            "room_search_status": _text(
                last_strategy.get("room_search_status") or "unsearched",
                32,
            ),
            "same_room_cycles": max(0, _int(strategy.get("same_room_cycles"))),
            "generic_floor_go_streak": max(
                0,
                _int(strategy.get("generic_floor_go_streak")),
            ),
            "force_leave_room": bool(strategy.get("force_leave_room")),
            "force_reason": _text(strategy.get("force_reason"), 64),
            "recent_exit_refs": [
                _text(item, 48)
                for item in recent_exit_refs[-3:]
                if _text(item, 48)
            ],
        },
        "directional_failures": _directional_failures(memory),
        "candidate_exits": _candidate_exits(memory),
        "negative_memories": _negative_memories(memory),
        "target_rejections": _target_rejections(memory),
        "revisit_candidates": _revisit_candidates(memory),
    }
    assert_policy_input_safe(projection)
    return projection


def compact_memory_json(vlm_input: Dict[str, Any]) -> str:
    return json.dumps(
        build_compact_memory_projection(vlm_input),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
