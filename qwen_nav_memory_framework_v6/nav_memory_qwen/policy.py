"""GaP-lite policy harness helpers for the navigation memory framework.

This module does not implement a full Graph-as-Policy interpreter.  Instead it
adds the smallest useful pieces for the v5 spatial-memory system:

* typed candidate references selected by the VLM instead of free-form node/edge ids,
* navigation skill/rule cards shown to the VLM,
* validation checkpoints with stable rule ids,
* conservative fallback/recovery for invalid candidate references,
* memory-op filtering before backend graph mutation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from .schema import make_observation_request_output, normalize_angle_deg


NAV_SKILL_CARDS: Dict[str, Dict[str, Any]] = {
    "go_front_fine_goal": {
        "type": "action",
        "use_when": [
            "visible navigable floor is available in the selected view",
            "selected_candidate_ref points to an exit with avoid=false",
            "selected_image_point is inside image bounds",
        ],
        "forbidden_when": [
            "selected_candidate_ref is missing or not in memory.candidate_refs.exits",
            "selected_candidate_ref has avoid=true or status is deadlock_entry/blocked/collision",
            "selected_view_type conflicts with the selected candidate's view_type_hint",
            "navigability is blocked or unknown",
        ],
        "fallback": "request_observation.directed_sweep",
        "postconditions": ["progress_positive_or_goal_reached", "no_collision"],
    },
    "rotate_to_selected_view_then_reobserve": {
        "type": "backend_policy",
        "use_when": [
            "force_front_view_waypoint=true",
            "VLM selected a non-front view for a go action",
        ],
        "effect": "backend rotates first and obtains a new front observation before the fast module receives a fine goal",
    },
    "request_directed_sweep": {
        "type": "observation_action",
        "use_when": [
            "goal bearing is outside current view",
            "deadlock or loop is suspected",
            "place recognition candidate needs more visual evidence",
            "candidate_ref validation failed",
        ],
        "postconditions": ["new_observation_views_available"],
    },
    "confirm_revisit_node": {
        "type": "memory_op",
        "use_when": [
            "VLM sees the same place layout in current and candidate memory images",
            "candidate_ref points to memory.candidate_refs.revisits",
            "spatial_plausibility.accepted=true",
        ],
        "forbidden_when": [
            "spatial_plausibility.accepted=false",
            "candidate is only visually similar but spatially implausible",
            "candidate_ref is missing or unknown",
        ],
        "commit_policy": "backend_verified_soft_merge_only",
        "fallback": "keep_duplicate_node_or_request_observation",
    },
    "reject_revisit_candidate": {
        "type": "memory_op",
        "use_when": [
            "layout differs",
            "spatial_plausibility.accepted=false",
            "similar appearance is better explained as a repeated structure",
        ],
        "effect": "audit only; topology is not mutated",
    },
}


DEFAULT_RECOVERY_POLICY: Dict[str, Dict[str, Any]] = {
    "invalid_candidate_ref": {
        "action": "request_observation",
        "mode": "directed_sweep",
        "reason": "candidate_ref_invalid_or_missing",
    },
    "unsafe_negative_exit": {
        "action": "request_observation",
        "mode": "directed_sweep",
        "reason": "selected_candidate_ref_is_avoid_true",
    },
    "invalid_revisit_memory_op": {
        "action": "drop_memory_op",
        "reason": "candidate_ref_or_spatial_gate_failed",
    },
}


def make_checkpoint(
    *,
    rule_id: str,
    name: str,
    stage: str,
    passed: bool,
    message: str,
    severity: str = "error",
    details: Optional[Dict[str, Any]] = None,
    on_fail: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a JSON-safe validation checkpoint."""
    return {
        "rule_id": rule_id,
        "name": name,
        "stage": stage,
        "validate": True,
        "passed": bool(passed),
        "severity": severity,
        "message": message,
        "details": details or {},
        "on_fail": on_fail,
    }


def summarize_checkpoints(checkpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a compact validation feedback block."""
    failed = [c for c in checkpoints if c.get("validate") and not c.get("passed") and c.get("severity") == "error"]
    warning = [c for c in checkpoints if c.get("validate") and not c.get("passed") and c.get("severity") != "error"]
    return {
        "valid": len(failed) == 0,
        "failed_rule_ids": [str(c.get("rule_id")) for c in failed],
        "warning_rule_ids": [str(c.get("rule_id")) for c in warning],
        "num_checkpoints": len(checkpoints),
        "num_failed": len(failed),
        "safe_recovery": failed[0].get("on_fail") if failed else None,
    }


def candidate_ref_indexes(memory_context: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Index exit and revisit candidates by backend-provided candidate_ref."""
    refs = memory_context.get("candidate_refs") or {}
    exit_index: Dict[str, Dict[str, Any]] = {}
    revisit_index: Dict[str, Dict[str, Any]] = {}
    for cand in refs.get("exits", []) or []:
        ref = cand.get("candidate_ref")
        if ref:
            exit_index[str(ref)] = cand
    for cand in refs.get("revisits", []) or []:
        ref = cand.get("candidate_ref")
        if ref:
            revisit_index[str(ref)] = cand

    # Backward/defensive fallback if the context was built by an older caller.
    for cand in memory_context.get("local_topology", {}).get("candidate_exits", []) or []:
        ref = cand.get("candidate_ref")
        if ref and str(ref) not in exit_index:
            exit_index[str(ref)] = cand
    for cand in memory_context.get("place_recognition", {}).get("revisit_candidates", []) or []:
        ref = cand.get("candidate_ref")
        if ref and str(ref) not in revisit_index:
            revisit_index[str(ref)] = cand
    return exit_index, revisit_index


def _fallback_scan(reason: str, goal_bearing_deg: float = 0.0, confidence: str = "medium") -> Dict[str, Any]:
    center = normalize_angle_deg(goal_bearing_deg)
    return make_observation_request_output(
        mode="directed_sweep",
        center_yaw_deg=center,
        step_deg=45,
        num_views=3,
        yaw_offsets_deg=[normalize_angle_deg(center - 45), center, normalize_angle_deg(center + 45)],
        reason=reason,
        confidence=confidence,
    )


def _resolve_and_filter_memory_ops(
    vlm_output: Dict[str, Any],
    memory_context: Dict[str, Any],
    revisit_index: Dict[str, Dict[str, Any]],
    *,
    require_candidate_ref_for_revisit_ops: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Resolve memory op candidate refs and drop unsafe mutation requests."""
    checkpoints: List[Dict[str, Any]] = []
    warnings: List[str] = []
    filtered_ops: List[Dict[str, Any]] = []
    current_node_id = memory_context.get("current_localization", {}).get("current_node_id")

    ops = vlm_output.get("memory_ops") or []
    if not isinstance(ops, list):
        checkpoints.append(make_checkpoint(
            rule_id="NAV_MEM_000",
            name="memory_ops_must_be_list",
            stage="memory_op_validation",
            passed=False,
            message="memory_ops is not a list; dropping all memory ops",
            on_fail="drop_memory_ops",
        ))
        return [], checkpoints, ["memory_ops is not a list; dropped"]

    revisit_ops = {"confirm_revisit_node", "confirm_same_place", "reject_revisit_candidate", "request_merge_nodes", "merge_nodes"}
    for op in ops:
        if not isinstance(op, dict):
            checkpoints.append(make_checkpoint(
                rule_id="NAV_MEM_001",
                name="memory_op_must_be_object",
                stage="memory_op_validation",
                passed=False,
                message="one memory op was not a JSON object and was dropped",
                on_fail="drop_memory_op",
            ))
            continue
        op_name = str(op.get("op"))
        if op_name not in revisit_ops:
            filtered_ops.append(dict(op))
            continue

        ref = op.get("candidate_ref")
        if not ref:
            passed = not require_candidate_ref_for_revisit_ops
            checkpoints.append(make_checkpoint(
                rule_id="NAV_MEM_002",
                name="revisit_memory_op_requires_candidate_ref",
                stage="memory_op_validation",
                passed=passed,
                message="revisit/merge memory ops should use backend-provided candidate_ref",
                details={"op": op_name},
                on_fail="drop_memory_op",
            ))
            if not passed:
                warnings.append(f"dropped {op_name}: missing candidate_ref")
                continue
            filtered_ops.append(dict(op))
            continue

        cand = revisit_index.get(str(ref))
        if not cand:
            checkpoints.append(make_checkpoint(
                rule_id="NAV_MEM_003",
                name="revisit_candidate_ref_must_exist",
                stage="memory_op_validation",
                passed=False,
                message="memory op referenced an unknown revisit candidate_ref",
                details={"op": op_name, "candidate_ref": ref},
                on_fail="drop_memory_op",
            ))
            warnings.append(f"dropped {op_name}: unknown candidate_ref={ref}")
            continue

        spatial = cand.get("spatial_plausibility") or {}
        if op_name in {"confirm_revisit_node", "confirm_same_place", "request_merge_nodes", "merge_nodes"} and not bool(spatial.get("accepted", False)):
            checkpoints.append(make_checkpoint(
                rule_id="NAV_MEM_004",
                name="same_place_commit_requires_spatial_gate",
                stage="memory_op_validation",
                passed=False,
                message="same-place memory op was dropped because spatial_plausibility.accepted is false",
                details={"op": op_name, "candidate_ref": ref, "spatial_plausibility": spatial},
                on_fail="keep_duplicate_node",
            ))
            warnings.append(f"dropped {op_name}: spatial_plausibility rejected candidate_ref={ref}")
            continue

        resolved = dict(op)
        node_id = cand.get("node_id") or cand.get("candidate_node_id")
        if node_id:
            resolved.setdefault("node_id", node_id)
            resolved.setdefault("candidate_node_id", node_id)
        if op_name in {"request_merge_nodes", "merge_nodes"}:
            resolved.setdefault("keep_node_id", node_id)
            resolved.setdefault("remove_node_id", current_node_id)
        checkpoints.append(make_checkpoint(
            rule_id="NAV_MEM_005",
            name="revisit_memory_op_candidate_ref_resolved",
            stage="memory_op_validation",
            passed=True,
            severity="info",
            message="memory op candidate_ref resolved to a backend revisit candidate",
            details={"op": op_name, "candidate_ref": ref, "node_id": node_id},
        ))
        filtered_ops.append(resolved)
    return filtered_ops, checkpoints, warnings


def apply_gap_lite_validation(
    vlm_output: Dict[str, Any],
    memory_context: Dict[str, Any],
    *,
    goal_bearing_deg: float = 0.0,
    force_front_view_waypoint: bool = True,
    require_candidate_ref_for_go: bool = True,
    require_candidate_ref_for_revisit_ops: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], List[str]]:
    """Apply GaP-lite validation and conservative recovery.

    Returns:
        safe_output, checkpoints, validation_feedback, warnings
    """
    output = deepcopy(vlm_output)
    warnings: List[str] = []
    checkpoints: List[Dict[str, Any]] = []
    exit_index, revisit_index = candidate_ref_indexes(memory_context)

    filtered_ops, mem_checkpoints, mem_warnings = _resolve_and_filter_memory_ops(
        output,
        memory_context,
        revisit_index,
        require_candidate_ref_for_revisit_ops=require_candidate_ref_for_revisit_ops,
    )
    checkpoints.extend(mem_checkpoints)
    warnings.extend(mem_warnings)
    if filtered_ops:
        output["memory_ops"] = filtered_ops
    elif "memory_ops" in output:
        output["memory_ops"] = []

    if output.get("action") == "go":
        ref = output.get("selected_candidate_ref") or output.get("candidate_ref")
        if not ref:
            passed = not require_candidate_ref_for_go
            checkpoints.append(make_checkpoint(
                rule_id="NAV_GO_001",
                name="go_requires_selected_candidate_ref",
                stage="action_validation",
                passed=passed,
                message="go action should select one backend-provided exit candidate_ref",
                on_fail="request_observation.directed_sweep",
            ))
            if not passed:
                warnings.append("blocked go: missing selected_candidate_ref")
                safe = _fallback_scan("gap_lite_missing_selected_candidate_ref", goal_bearing_deg, confidence="medium")
                safe["memory_ops"] = filtered_ops
                output = safe
        if output.get("action") == "go" and ref:
            exit_cand = exit_index.get(str(ref))
            if not exit_cand:
                checkpoints.append(make_checkpoint(
                    rule_id="NAV_GO_002",
                    name="selected_exit_candidate_ref_must_exist",
                    stage="action_validation",
                    passed=False,
                    message="selected_candidate_ref was not present in memory.candidate_refs.exits",
                    details={"selected_candidate_ref": ref},
                    on_fail="request_observation.directed_sweep",
                ))
                warnings.append(f"blocked go: unknown selected_candidate_ref={ref}")
                safe = _fallback_scan("gap_lite_unknown_selected_candidate_ref", goal_bearing_deg, confidence="medium")
                safe["memory_ops"] = filtered_ops
                output = safe
            else:
                avoid = bool(exit_cand.get("avoid", False))
                checkpoints.append(make_checkpoint(
                    rule_id="NAV_GO_003",
                    name="selected_exit_candidate_must_not_be_avoid",
                    stage="action_validation",
                    passed=not avoid,
                    message="selected exit candidate must not be a known deadlock/blocked branch",
                    details={"selected_candidate_ref": ref, "candidate": exit_cand},
                    on_fail="request_observation.directed_sweep",
                ))
                if avoid:
                    warnings.append(f"blocked go: candidate_ref={ref} has avoid=true")
                    safe = _fallback_scan("gap_lite_selected_candidate_ref_avoid_true", goal_bearing_deg, confidence="medium")
                    safe["memory_ops"] = filtered_ops
                    output = safe
                else:
                    output["selected_candidate_ref"] = str(ref)
                    vt = str(output.get("selected_view_type") or "front")
                    hint = str(exit_cand.get("view_type_hint") or vt)
                    view_match = vt == hint
                    # Non-front hints are allowed only when the output itself uses that
                    # non-front view; the backend will rotate-to-front before execution.
                    checkpoints.append(make_checkpoint(
                        rule_id="NAV_GO_004",
                        name="selected_view_matches_candidate_ref_hint",
                        stage="action_validation",
                        passed=view_match,
                        message="selected_view_type should match the selected candidate's view_type_hint",
                        details={"selected_view_type": vt, "candidate_view_type_hint": hint, "force_front_view_waypoint": force_front_view_waypoint},
                        on_fail="request_observation.directed_sweep",
                    ))
                    if not view_match:
                        warnings.append(f"blocked go: view_type {vt} conflicts with candidate_ref={ref} hint {hint}")
                        safe = _fallback_scan("gap_lite_candidate_ref_view_mismatch", goal_bearing_deg, confidence="medium")
                        safe["memory_ops"] = filtered_ops
                        output = safe
                    else:
                        nonfront = vt != "front"
                        checkpoints.append(make_checkpoint(
                            rule_id="NAV_GO_005",
                            name="front_only_fast_module_policy",
                            stage="action_validation",
                            passed=True,
                            severity="info",
                            message="front fine goal will be executed directly; non-front go will be rotated-to-front first",
                            details={"selected_view_type": vt, "will_rotate_to_front": bool(nonfront and force_front_view_waypoint)},
                        ))

    checkpoints.append(make_checkpoint(
        rule_id="NAV_POLICY_001",
        name="gap_lite_validation_completed",
        stage="policy_harness",
        passed=True,
        severity="info",
        message="GaP-lite validation sidecar executed",
        details={"num_exit_refs": len(exit_index), "num_revisit_refs": len(revisit_index)},
    ))
    feedback = summarize_checkpoints(checkpoints)
    output["validation_checkpoints"] = checkpoints
    output["validation_feedback"] = feedback
    return output, checkpoints, feedback, warnings


def build_policy_harness_state(
    *,
    current_stage: str,
    last_validation_feedback: Optional[Dict[str, Any]] = None,
    require_candidate_ref_for_go: bool = True,
    require_candidate_ref_for_revisit_ops: bool = True,
) -> Dict[str, Any]:
    """Return a compact policy-harness state block for the VLM memory field."""
    return {
        "schema_version": "nav_policy_harness_lite_v1",
        "policy_graph_mode": "GaP-lite_static_sidecar_not_full_interpreter",
        "current_stage": current_stage,
        "enabled_features": [
            "candidate_ref_selection",
            "validation_checkpoints",
            "memory_op_candidate_ref_resolution",
            "validation_feedback_logging",
            "nav_skill_cards",
        ],
        "requirements": {
            "go_requires_selected_candidate_ref": bool(require_candidate_ref_for_go),
            "revisit_memory_ops_require_candidate_ref": bool(require_candidate_ref_for_revisit_ops),
            "false_positive_merge_policy": "keep_duplicate_node_when_uncertain",
        },
        "last_validation_feedback": last_validation_feedback,
    }
