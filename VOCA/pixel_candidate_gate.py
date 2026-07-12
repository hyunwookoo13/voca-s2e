import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np


PIXEL_CANDIDATE_SCHEMA_VERSION = "rgb_pixel_candidates_v1"
_CANDIDATE_COLUMNS = (0.25, 0.50, 0.75)
_CANDIDATE_ROWS = (0.97, 0.84)
_SATURATION_RECOVERY_ROW = 0.72
_NEGATIVE_STATUSES = {
    "blocked",
    "collision",
    "deadlock_entry",
    "deadlock_entry_candidate",
    "failed",
}


def _dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, dict)]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _angular_distance_deg(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def _normalize_angle_deg(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


def _optional_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def _topological_candidate_for_view(
    memory_context: Dict[str, Any],
    views: Sequence[Dict[str, Any]],
    view_index: int,
    *,
    max_alignment_error_deg: float = 40.0,
) -> Optional[Dict[str, Any]]:
    if view_index < 0 or view_index >= len(views):
        return None
    exits = _list_of_dicts(_dict(memory_context.get("candidate_refs")).get("exits"))
    current_view = views[view_index]
    current_type = str(current_view.get("view_type") or "front")
    view_angles = [_float(view.get("relative_heading_deg")) for view in views]
    matches: List[Dict[str, Any]] = []
    for item in exits:
        bearing = _optional_float(item.get("bearing_deg_robot"))
        if bearing is not None:
            alignment_error = _angular_distance_deg(
                view_angles[view_index],
                bearing,
            )
            # Adjacent panoramic views overlap (79-degree camera HFOV with
            # 45-degree full-sweep spacing). Preserve that overlap so a path
            # tangent near a doorway edge can use the neighboring view where
            # its floor is centered, while still preventing coarse-label
            # propagation to unrelated headings.
            if alignment_error > float(max_alignment_error_deg):
                continue
            matched = dict(item)
            matched["view_alignment_error_deg"] = round(alignment_error, 3)
            matches.append(matched)
            continue

        if str(item.get("view_type_hint") or "") != current_type:
            continue
        matching_view_indices = [
            index
            for index, view in enumerate(views)
            if str(view.get("view_type") or "front") == current_type
        ]
        if matching_view_indices and matching_view_indices[0] == view_index:
            matches.append(dict(item))
    if not matches:
        return None
    negative = [
        item
        for item in matches
        if bool(item.get("avoid"))
        or str(item.get("status") or "").strip().lower() in _NEGATIVE_STATUSES
    ]
    pool = negative or matches
    return max(
        pool,
        key=lambda item: (
            _float(item.get("score")),
            str(item.get("candidate_ref") or ""),
        ),
    )


def _matching_failure(
    *,
    failures: Sequence[Dict[str, Any]],
    topological_edge_id: Optional[str],
    point_px: Sequence[int],
    radius_px: float,
) -> Optional[Dict[str, Any]]:
    edge_id = str(topological_edge_id or "").strip()
    if not edge_id:
        return None
    best = None
    best_distance = float("inf")
    for failure in failures:
        if str(failure.get("negative_edge_id") or "").strip() != edge_id:
            continue
        failed_point = failure.get("point_px")
        if not isinstance(failed_point, (list, tuple)) or len(failed_point) != 2:
            continue
        try:
            distance = math.hypot(
                float(point_px[0]) - float(failed_point[0]),
                float(point_px[1]) - float(failed_point[1]),
            )
        except Exception:
            continue
        if distance <= float(radius_px) and distance < best_distance:
            best = failure
            best_distance = distance
    return best


def _matching_runtime_rejection(
    *,
    feedback: Sequence[Dict[str, Any]],
    candidate_ref: str,
    angle_deg: float,
    world_bearing_deg: Optional[float],
    point_px: Sequence[int],
    radius_px: float,
) -> Optional[Dict[str, Any]]:
    for item in reversed(feedback):
        if str(item.get("event") or "") not in {
            "point_verifier_rejected",
            "target_approach_unresolved",
        }:
            continue
        item_world_bearing = _optional_float(item.get("world_bearing_deg"))
        if world_bearing_deg is not None and item_world_bearing is not None:
            angle_delta = _angular_distance_deg(
                world_bearing_deg,
                item_world_bearing,
            )
        else:
            try:
                angle_delta = _angular_distance_deg(
                    float(angle_deg),
                    float(item.get("angle_deg", 0.0)),
                )
            except Exception:
                continue
        failed_point = item.get("point_px")
        if not isinstance(failed_point, (list, tuple)) or len(failed_point) != 2:
            continue
        try:
            point_delta = math.hypot(
                float(point_px[0]) - float(failed_point[0]),
                float(point_px[1]) - float(failed_point[1]),
            )
        except Exception:
            continue
        same_ref = str(item.get("selected_candidate_ref") or "") == str(candidate_ref or "")
        if angle_delta <= 15.0 and point_delta <= float(radius_px) and same_ref:
            return item
    return None


def candidate_visual_evidence(
    rgb: np.ndarray,
    point_px: Sequence[int],
    *,
    patch_radius_px: int = 48,
) -> Dict[str, Any]:
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        return {
            "available": False,
            "requires_verification": True,
            "hard_reject": False,
            "support_score": 0.0,
            "reasons": ["invalid_rgb_shape"],
        }
    height, width = image.shape[:2]
    try:
        u = max(0, min(width - 1, int(point_px[0])))
        v = max(0, min(height - 1, int(point_px[1])))
    except Exception:
        return {
            "available": False,
            "requires_verification": True,
            "hard_reject": True,
            "support_score": 0.0,
            "reasons": ["invalid_point"],
        }
    radius = max(12, int(patch_radius_px))
    patch = image[
        max(0, v - radius) : min(height, v + max(6, radius // 3)),
        max(0, u - radius) : min(width, u + radius),
    ]
    if patch.size == 0:
        return {
            "available": False,
            "requires_verification": True,
            "hard_reject": True,
            "support_score": 0.0,
            "reasons": ["empty_patch"],
        }
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    luma_mean = float(np.mean(gray))
    luma_std = float(np.std(gray))
    edge_density = float(np.mean(cv2.Canny(gray, 40, 120) > 0))
    extreme_fraction = float(np.mean((gray <= 4) | (gray >= 251)))
    reasons: List[str] = []
    if luma_std < 6.0:
        reasons.append("low_texture")
    if edge_density < 0.01:
        reasons.append("low_structure")
    if extreme_fraction > 0.85:
        reasons.append("extreme_exposure")
    support = min(1.0, luma_std / 24.0) * 0.55 + min(1.0, edge_density / 0.08) * 0.45
    return {
        "available": True,
        "requires_verification": bool(reasons),
        # RGB appearance alone cannot safely distinguish a dark floor from a
        # black wall. Keep valid patches selectable but force VLM verification.
        "hard_reject": False,
        "support_score": round(float(support), 4),
        "luma_mean": round(luma_mean, 4),
        "luma_std": round(luma_std, 4),
        "edge_density": round(edge_density, 6),
        "extreme_exposure_fraction": round(extreme_fraction, 6),
        "reasons": reasons,
    }


def build_saturation_recovery_candidates(
    views: Sequence[Dict[str, Any]],
    image_shape: Sequence[int],
    *,
    rgb_images: Optional[Sequence[np.ndarray]] = None,
    max_candidates: int = 24,
    row_norm: float = _SATURATION_RECOVERY_ROW,
) -> List[Dict[str, Any]]:
    """Offer far-floor probes only after negative memory exhausts normal choices."""
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        return []
    budget = max(0, min(int(max_candidates), 24))
    row = max(0.55, min(0.80, float(row_norm)))
    candidates: List[Dict[str, Any]] = []
    for view_index, view in enumerate(views):
        view_id = int(view.get("view_id", view_index) or 0)
        for column_index, column_norm in enumerate(_CANDIDATE_COLUMNS):
            if len(candidates) >= budget:
                return candidates
            u = max(0, min(width - 1, int(round(column_norm * (width - 1)))))
            v = max(0, min(height - 1, int(round(row * (height - 1)))))
            visual = {
                "available": False,
                "requires_verification": True,
                "hard_reject": False,
                "support_score": 0.0,
                "reasons": ["rgb_unavailable"],
            }
            if rgb_images is not None and view_index < len(rgb_images):
                visual = candidate_visual_evidence(rgb_images[view_index], [u, v])
            candidates.append(
                {
                    "candidate_ref": "px_v{:02d}_r2_c{}".format(
                        view_id,
                        column_index,
                    ),
                    "candidate_type": "pixel_waypoint",
                    "marker_id": "S{:02d}".format(len(candidates) + 1),
                    "view_id": view_id,
                    "view_type_hint": str(view.get("view_type") or "front"),
                    "relative_heading_deg": _float(
                        view.get("relative_heading_deg")
                    ),
                    "bearing_deg_robot": _float(
                        view.get("relative_heading_deg")
                    ),
                    "point_px": [u, v],
                    "point_norm": [
                        round(u / float(max(1, width - 1)), 4),
                        round(v / float(max(1, height - 1)), 4),
                    ],
                    "avoid": False,
                    "executable": True,
                    "status": "negative_memory_saturation_recovery",
                    "score": round(
                        0.05 + 0.10 * _float(visual.get("support_score")),
                        4,
                    ),
                    "reason": (
                        "far-floor recovery probe offered only because all normal "
                        "headings were excluded by negative memory"
                    ),
                    "topological_candidate_ref": None,
                    "topological_edge_id": None,
                    "topological_relation_type": None,
                    "source": "negative_memory_saturation_recovery_v1",
                    "visual_evidence": dict(visual),
                    "requires_rgb_verification": True,
                    "negative_memory_saturation_recovery": True,
                    "negative_memory_override_scope": "far_pixel_only_not_edge_clear",
                }
            )
    return candidates


def build_pixel_candidates(
    views: Sequence[Dict[str, Any]],
    image_shape: Sequence[int],
    memory_context: Optional[Dict[str, Any]],
    *,
    max_candidates: int = 32,
    failed_point_radius_px: float = 56.0,
    rgb_images: Optional[Sequence[np.ndarray]] = None,
) -> List[Dict[str, Any]]:
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        return []
    memory = _dict(memory_context)
    failures = _list_of_dicts(_dict(memory.get("voca_sidecar")).get("directional_failures"))
    runtime_feedback = _list_of_dicts(memory.get("runtime_navigation_feedback"))
    robot_heading_rad = _optional_float(
        _dict(memory.get("voca_sidecar")).get("heading_rad")
    )
    candidates: List[Dict[str, Any]] = []
    budget = max(0, min(int(max_candidates), 40))

    # A full 360-degree sweep has more views than the ordinary five-view
    # panorama. Six candidates per view would exhaust the global budget before
    # the rear headings are reached, making those directions impossible for
    # the VLM to select. Use one navigable-floor row for wide sweeps so every
    # view receives three mid-floor choices plus one bottom-center fallback.
    wide_sweep = bool(
        len(views) * len(_CANDIDATE_ROWS) * len(_CANDIDATE_COLUMNS) > budget
    )
    candidate_rows = (
        (_CANDIDATE_ROWS[1], _CANDIDATE_ROWS[0])
        if wide_sweep
        else _CANDIDATE_ROWS
    )

    for view_index, view in enumerate(views):
        view_id = int(view.get("view_id", len(candidates)) or 0)
        view_type = str(view.get("view_type") or "front")
        angle_deg = _float(view.get("relative_heading_deg"))
        topo = _topological_candidate_for_view(memory, views, view_index)
        for row_norm in candidate_rows:
            columns = (
                (_CANDIDATE_COLUMNS[1],)
                if wide_sweep and row_norm == _CANDIDATE_ROWS[0]
                else _CANDIDATE_COLUMNS
            )
            for column_norm in columns:
                source_row_index = _CANDIDATE_ROWS.index(row_norm)
                column_index = _CANDIDATE_COLUMNS.index(column_norm)
                if len(candidates) >= budget:
                    return candidates
                u = max(0, min(width - 1, int(round(column_norm * (width - 1)))))
                v = max(0, min(height - 1, int(round(row_norm * (height - 1)))))
                point = [u, v]
                status = "rgb_visual_candidate"
                avoid = False
                score = 0.5
                reason = "marked RGB waypoint; VLM must verify visible navigable floor"
                topological_ref = None
                topological_edge_id = None
                topological_relation_type = None
                topological_bearing_deg_robot = None
                topological_view_alignment_error_deg = None
                topological_bearing_source = None
                topological_chord_bearing_deg_robot = None
                topological_path_departure_bearing_node_deg = None
                topological_path_guidance = None
                if topo is not None:
                    topological_ref = str(topo.get("candidate_ref") or "") or None
                    topological_edge_id = str(topo.get("edge_id") or "") or None
                    topological_relation_type = str(topo.get("relation_type") or "") or None
                    topological_bearing_deg_robot = _optional_float(
                        topo.get("bearing_deg_robot")
                    )
                    topological_view_alignment_error_deg = _optional_float(
                        topo.get("view_alignment_error_deg")
                    )
                    topological_bearing_source = (
                        str(topo.get("bearing_source") or "") or None
                    )
                    topological_chord_bearing_deg_robot = _optional_float(
                        topo.get("chord_bearing_deg_robot")
                    )
                    topological_path_departure_bearing_node_deg = _optional_float(
                        topo.get("path_departure_bearing_node_deg")
                    )
                    topological_path_guidance = (
                        dict(topo.get("path_guidance"))
                        if isinstance(topo.get("path_guidance"), dict)
                        else None
                    )
                    status = str(topo.get("status") or status)
                    avoid = bool(topo.get("avoid")) or status.strip().lower() in _NEGATIVE_STATUSES
                    score = _float(topo.get("score"), score)
                    reason = str(topo.get("reason") or reason)

                visual_evidence = None
                if rgb_images is not None and view_index < len(rgb_images):
                    visual_evidence = candidate_visual_evidence(rgb_images[view_index], point)
                    if visual_evidence.get("hard_reject"):
                        avoid = True
                        status = "rgb_invalid_candidate"
                        reason = "RGB evidence rejected candidate: {}".format(
                            ",".join(visual_evidence.get("reasons") or ["invalid_patch"])
                        )
                    elif visual_evidence.get("requires_verification") and not avoid:
                        status = "rgb_verification_required"
                        score = min(
                            float(score),
                            0.35 + 0.3 * float(visual_evidence.get("support_score", 0.0)),
                        )
                        reason = "RGB candidate requires verification: {}".format(
                            ",".join(visual_evidence.get("reasons") or ["ambiguous_patch"])
                        )
                    elif not avoid:
                        status = "rgb_visually_supported"
                        score = max(
                            float(score),
                            0.55 + 0.35 * float(visual_evidence.get("support_score", 0.0)),
                        )
                        reason = "RGB patch has sufficient local structure for VLM floor inspection"

                failure = _matching_failure(
                    failures=failures,
                    topological_edge_id=topological_edge_id,
                    point_px=point,
                    radius_px=failed_point_radius_px,
                )
                failure_class = None
                if failure is not None:
                    avoid = True
                    status = "verified_failed_pixel_neighborhood"
                    failure_class = str(failure.get("failure_class") or "no_progress")
                    reason = "verified prior failure near this marked pixel: {}".format(
                        failure_class
                    )

                marker_number = len(candidates) + 1
                candidate = {
                    "candidate_ref": "px_v{:02d}_r{}_c{}".format(
                        view_id,
                        source_row_index,
                        column_index,
                    ),
                    "candidate_type": "pixel_waypoint",
                    "marker_id": "M{:02d}".format(marker_number),
                    "view_id": view_id,
                    "view_type_hint": view_type,
                    "relative_heading_deg": round(angle_deg, 3),
                    "bearing_deg_robot": round(angle_deg, 3),
                    "point_px": point,
                    "point_norm": [
                        round(float(u) / max(1, width - 1), 4),
                        round(float(v) / max(1, height - 1), 4),
                    ],
                    "avoid": bool(avoid),
                    "status": status,
                    "score": round(float(score), 4),
                    "reason": reason,
                    "topological_candidate_ref": topological_ref,
                    "topological_edge_id": topological_edge_id,
                    "topological_relation_type": topological_relation_type,
                    "topological_bearing_deg_robot": topological_bearing_deg_robot,
                    "topological_bearing_source": topological_bearing_source,
                    "topological_chord_bearing_deg_robot": (
                        topological_chord_bearing_deg_robot
                    ),
                    "topological_path_departure_bearing_node_deg": (
                        topological_path_departure_bearing_node_deg
                    ),
                    "topological_path_guidance": topological_path_guidance,
                    "topological_view_alignment_error_deg": (
                        topological_view_alignment_error_deg
                    ),
                    "source": PIXEL_CANDIDATE_SCHEMA_VERSION,
                }
                if visual_evidence is not None:
                    candidate["visual_evidence"] = visual_evidence
                    candidate["requires_rgb_verification"] = bool(
                        visual_evidence.get("requires_verification")
                    )
                if failure_class is not None:
                    candidate["failure_class"] = failure_class
                runtime_rejection = _matching_runtime_rejection(
                    feedback=runtime_feedback,
                    candidate_ref=candidate["candidate_ref"],
                    angle_deg=angle_deg,
                    world_bearing_deg=(
                        _normalize_angle_deg(
                            math.degrees(robot_heading_rad) + angle_deg
                        )
                        if robot_heading_rad is not None
                        else None
                    ),
                    point_px=point,
                    radius_px=failed_point_radius_px,
                )
                if runtime_rejection is not None:
                    if runtime_rejection.get("event") == "target_approach_unresolved":
                        candidate.update(
                            avoid=True,
                            status="target_approach_unresolved_candidate",
                            reason=(
                                "repeated direct target approach did not reach this waypoint; "
                                "select a gateway or detour"
                            ),
                            failure_class="target_approach_unresolved",
                        )
                    else:
                        candidate.update(
                            avoid=True,
                            status="point_verifier_rejected_candidate",
                            reason="point verifier rejected this pixel as {}".format(
                                runtime_rejection.get("surface") or "non_navigable"
                            ),
                            failure_class="point_verifier_{}".format(
                                runtime_rejection.get("surface") or "rejected"
                            ),
                        )
                candidates.append(candidate)
    return candidates


def partition_pixel_candidates(
    candidates: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split candidates into VLM-visible choices and audit-only exclusions."""
    executable: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        candidate = dict(item)
        status = str(candidate.get("status") or "").strip().lower()
        exclusion_reason = None
        if bool(candidate.get("avoid")):
            exclusion_reason = str(candidate.get("exclusion_reason") or "avoid_true")
        elif status in _NEGATIVE_STATUSES:
            exclusion_reason = "negative_status:{}".format(status)
        elif not str(candidate.get("candidate_ref") or "").strip():
            exclusion_reason = "missing_candidate_ref"
        elif not isinstance(candidate.get("point_px"), (list, tuple)) or len(
            candidate.get("point_px")
        ) != 2:
            exclusion_reason = "missing_canonical_point"

        if exclusion_reason is None:
            candidate["avoid"] = False
            candidate["executable"] = True
            executable.append(candidate)
            continue

        candidate["avoid"] = True
        candidate["executable"] = False
        candidate["exclusion_reason"] = exclusion_reason
        excluded.append(candidate)
    return executable, excluded


def render_pixel_candidates(
    rgb: np.ndarray,
    candidates: Sequence[Dict[str, Any]],
    *,
    view_id: int,
) -> np.ndarray:
    output = np.asarray(rgb, dtype=np.uint8).copy()
    height, width = output.shape[:2]
    radius = max(5, min(12, int(round(min(height, width) * 0.025))))
    font_scale = max(0.35, min(0.55, min(height, width) / 360.0))
    for candidate in candidates:
        if _int_or_none(candidate.get("view_id")) != int(view_id):
            continue
        point = candidate.get("point_px")
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        u = max(0, min(width - 1, int(point[0])))
        v = max(0, min(height - 1, int(point[1])))
        avoid = bool(candidate.get("avoid"))
        fill = (255, 80, 80) if avoid else (255, 220, 0)
        cv2.circle(output, (u, v), radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(output, (u, v), radius, fill, -1, cv2.LINE_AA)
        label = str(candidate.get("marker_id") or "M")
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            1,
        )
        text_x = max(0, min(width - text_width, u - text_width // 2))
        text_y = max(text_height, min(height - 1, v + text_height // 2))
        cv2.putText(
            output,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return output


def _validation_result(
    *,
    passed: bool,
    reason: str,
    requested_ref: Optional[str],
    candidate: Optional[Dict[str, Any]],
    canonical_output: Optional[Dict[str, Any]],
    coordinate_corrected: bool = False,
    resolved_ref: Optional[str] = None,
    ref_alias_applied: bool = False,
) -> Dict[str, Any]:
    return {
        "schema_version": "pixel_candidate_validation_v1",
        "passed": bool(passed),
        "reason": str(reason),
        "requested_ref": requested_ref,
        "resolved_ref": resolved_ref,
        "ref_alias_applied": bool(ref_alias_applied),
        "candidate": dict(candidate) if isinstance(candidate, dict) else None,
        "canonical_output": canonical_output,
        "coordinate_corrected": bool(coordinate_corrected),
        "checkpoint": {
            "rule_id": "PIXEL_GO_001",
            "name": "go_requires_valid_rgb_pixel_candidate",
            "stage": "action_validation",
            "passed": bool(passed),
            "message": str(reason),
            "on_fail": None if passed else "request_observation.directed_sweep",
        },
    }


def validate_pixel_candidate_selection(
    vlm_output: Dict[str, Any],
    vlm_input: Dict[str, Any],
) -> Dict[str, Any]:
    output = vlm_output if isinstance(vlm_output, dict) else {}
    if str(output.get("action") or "").strip().lower() != "go":
        return _validation_result(
            passed=True,
            reason="candidate_not_required_for_action",
            requested_ref=None,
            candidate=None,
            canonical_output=deepcopy(output),
        )

    requested = output.get("selected_candidate_ref") or output.get("candidate_ref")
    requested_ref = str(requested).strip() if requested is not None else ""
    if not requested_ref:
        return _validation_result(
            passed=False,
            reason="missing_selected_candidate_ref",
            requested_ref=None,
            candidate=None,
            canonical_output=None,
        )

    block = _dict(vlm_input.get("pixel_candidates"))
    candidates = _list_of_dicts(block.get("candidates"))
    index = {
        str(candidate.get("candidate_ref")): candidate
        for candidate in candidates
        if candidate.get("candidate_ref")
    }
    candidate = index.get(requested_ref)
    resolved_ref = requested_ref
    ref_alias_applied = False
    if candidate is None:
        marker_matches = [
            item
            for item in candidates
            if str(item.get("marker_id") or "").strip().lower()
            == requested_ref.lower()
        ]
        if len(marker_matches) == 1:
            candidate = marker_matches[0]
            resolved_ref = str(candidate.get("candidate_ref") or "").strip()
            ref_alias_applied = bool(resolved_ref)
    if candidate is None:
        excluded = {
            str(item.get("candidate_ref")): item
            for item in _list_of_dicts(block.get("excluded_candidates"))
            if item.get("candidate_ref")
        }
        excluded_candidate = excluded.get(requested_ref)
        if excluded_candidate is not None:
            return _validation_result(
                passed=False,
                reason="selected_candidate_excluded",
                requested_ref=requested_ref,
                candidate=excluded_candidate,
                canonical_output=None,
            )
        return _validation_result(
            passed=False,
            reason="unknown_selected_candidate_ref",
            requested_ref=requested_ref,
            candidate=None,
            canonical_output=None,
        )
    if bool(candidate.get("avoid")):
        return _validation_result(
            passed=False,
            reason="selected_candidate_avoid_true",
            requested_ref=requested_ref,
            candidate=candidate,
            canonical_output=None,
        )

    expected_view_id = _int_or_none(candidate.get("view_id"))
    expected_view_type = str(candidate.get("view_type_hint") or "front")
    output_view_id = _int_or_none(output.get("selected_view_id"))
    output_view_type = output.get("selected_view_type")
    if (
        output_view_id is not None
        and expected_view_id is not None
        and output_view_id != expected_view_id
    ) or (
        output_view_type is not None
        and str(output_view_type) != expected_view_type
    ):
        return _validation_result(
            passed=False,
            reason="selected_candidate_view_mismatch",
            requested_ref=requested_ref,
            candidate=candidate,
            canonical_output=None,
        )

    point = candidate.get("point_px")
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return _validation_result(
            passed=False,
            reason="candidate_missing_canonical_point",
            requested_ref=requested_ref,
            candidate=candidate,
            canonical_output=None,
        )
    canonical_point = [int(point[0]), int(point[1])]
    raw_point = output.get("selected_image_point")
    coordinate_corrected = list(raw_point) != canonical_point if isinstance(raw_point, (list, tuple)) else True
    canonical = deepcopy(output)
    canonical["selected_candidate_ref"] = resolved_ref
    canonical["selected_view_id"] = expected_view_id
    canonical["selected_view_type"] = expected_view_type
    canonical["selected_image_point"] = canonical_point
    fine_goal = canonical.get("fine_goal")
    fine_goal = deepcopy(fine_goal) if isinstance(fine_goal, dict) else {}
    fine_goal.update(
        {
            "valid": True,
            "view_id": expected_view_id,
            "view_type": expected_view_type,
            "point_px": canonical_point,
            "point_norm": list(candidate.get("point_norm") or []),
            "projected_map_xy": None,
            "navigability": "likely_free",
        }
    )
    canonical["fine_goal"] = fine_goal
    return _validation_result(
        passed=True,
        reason=(
            "valid_pixel_candidate_marker_alias"
            if ref_alias_applied
            else "valid_pixel_candidate"
        ),
        requested_ref=requested_ref,
        candidate=candidate,
        canonical_output=canonical,
        coordinate_corrected=coordinate_corrected,
        resolved_ref=resolved_ref,
        ref_alias_applied=ref_alias_applied,
    )
