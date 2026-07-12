import math
from typing import Any, Dict, Optional, Sequence


SUPERVISOR_MODES = (
    "goal_seek",
    "escape_deadlock",
    "backtrack",
    "verify_target",
)

_MODE_ALIASES = {
    "normal_goal_seek": "goal_seek",
    "escaping_deadlock": "escape_deadlock",
    "verify_revisit": "goal_seek",
}

_FORBIDDEN_POLICY_KEYS = {
    "distance_to_goal",
    "success",
    "spl",
    "soft_spl",
    "top_down_map",
    "shortest_path",
    "shortest_path_length",
}

_ALLOWED_SPATIAL_GOAL_SOURCES = {
    "upstream_global_planner",
    "topological_memory",
    "exploration_frontier",
    "user_instruction",
    "s2e_backbone",
}

_FORBIDDEN_SPATIAL_GOAL_SOURCES = {
    "habitat_ground_truth",
    "oracle",
    "evaluation_metric",
    "shortest_path",
    "shortest_path_follower",
}

_LOCALIZATION_SOURCES = {
    "none",
    "habitat_sim_pose_declared",
    "gps_compass_sensor",
    "controller_odometry",
    "estimated_egomotion",
}


def normalize_supervisor_mode(mode: Any) -> str:
    normalized = str(mode or "").strip().lower()
    normalized = _MODE_ALIASES.get(normalized, normalized)
    return normalized if normalized in SUPERVISOR_MODES else "goal_seek"


def build_localization_contract(
    *,
    source: Any,
    allow_sim_pose: bool = False,
) -> Dict[str, Any]:
    normalized = str(source or "none").strip().lower()
    if normalized not in _LOCALIZATION_SOURCES:
        raise ValueError("unsupported localization source: {}".format(normalized))
    uses_sim_ground_truth = normalized == "habitat_sim_pose_declared"
    if uses_sim_ground_truth and not bool(allow_sim_pose):
        raise ValueError(
            "habitat simulator pose requires VOCA_ALLOW_SIM_POSE_POLICY=1 and explicit reporting"
        )
    return {
        "schema_version": "voca_localization_contract_v1",
        "source": normalized,
        "pose_available_to_policy": normalized != "none",
        "uses_sim_ground_truth": uses_sim_ground_truth,
        "sim_pose_explicitly_allowed": bool(allow_sim_pose) if uses_sim_ground_truth else False,
        "reporting_requirement": (
            "report RGB+sim-pose setting; do not describe as RGB-only"
            if uses_sim_ground_truth
            else "report the declared localization sensor/backend"
        ),
    }


def _optional_finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("spatial coarse goal values must be finite")
    return parsed


def build_policy_coarse_goal(spatial_goal: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    spatial = spatial_goal if isinstance(spatial_goal, dict) else {}
    if not spatial:
        return {
            "type": "object_category",
            "source": "objectnav_category",
            "map_xy": None,
            "relative_bearing_deg": None,
            "distance_m": None,
            "distance_range_m": None,
            "uncertainty": None,
            "goal_geometry_available": False,
        }

    source = str(spatial.get("source") or "").strip().lower()
    if source in _FORBIDDEN_SPATIAL_GOAL_SOURCES or source not in _ALLOWED_SPATIAL_GOAL_SOURCES:
        raise ValueError("unsupported or oracle spatial coarse-goal source: {}".format(source or "missing"))
    goal_type = str(spatial.get("type") or "relative_waypoint").strip().lower()
    if goal_type not in {"relative_waypoint", "map_waypoint", "topological_target"}:
        raise ValueError("unsupported spatial coarse-goal type: {}".format(goal_type))
    bearing = _optional_finite_float(spatial.get("relative_bearing_deg"))
    if bearing is not None:
        bearing = ((bearing + 180.0) % 360.0) - 180.0
    distance = _optional_finite_float(spatial.get("distance_m"))
    if distance is not None and distance < 0.0:
        raise ValueError("spatial coarse-goal distance_m must be non-negative")
    raw_range = spatial.get("distance_range_m")
    distance_range = None
    if raw_range is not None:
        if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
            raise ValueError("distance_range_m must contain [min_m, max_m]")
        distance_range = [
            _optional_finite_float(raw_range[0]),
            _optional_finite_float(raw_range[1]),
        ]
        if distance_range[0] is None or distance_range[1] is None:
            raise ValueError("distance_range_m values cannot be null")
        if distance_range[0] < 0.0 or distance_range[1] < distance_range[0]:
            raise ValueError("distance_range_m must satisfy 0 <= min <= max")
    raw_map_xy = spatial.get("map_xy")
    map_xy = None
    if raw_map_xy is not None:
        if not isinstance(raw_map_xy, (list, tuple)) or len(raw_map_xy) != 2:
            raise ValueError("map_xy must contain two finite coordinates")
        map_xy = [
            _optional_finite_float(raw_map_xy[0]),
            _optional_finite_float(raw_map_xy[1]),
        ]
        if map_xy[0] is None or map_xy[1] is None:
            raise ValueError("map_xy values cannot be null")
    if bearing is None and distance is None and distance_range is None and map_xy is None:
        raise ValueError("spatial coarse goal must provide bearing, distance, range, or map_xy")
    uncertainty = str(spatial.get("uncertainty") or "medium").strip().lower()
    if uncertainty not in {"low", "medium", "high"}:
        raise ValueError("spatial coarse-goal uncertainty must be low, medium, or high")
    return {
        "type": goal_type,
        "source": source,
        "map_xy": map_xy,
        "relative_bearing_deg": bearing,
        "distance_m": distance,
        "distance_range_m": distance_range,
        "uncertainty": uncertainty,
        "goal_geometry_available": True,
        "temporary_detour_allowed": True,
    }


def build_policy_goal_context(
    *,
    task_mode: str,
    target_object: str,
    supervisor_mode: str,
    spatial_goal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    mode = normalize_supervisor_mode(supervisor_mode)
    coarse_goal = build_policy_coarse_goal(spatial_goal)
    effective_task_mode = "CoarseGoalNav" if coarse_goal["goal_geometry_available"] else str(task_mode or "ObjNav")
    detour_status = {
        "goal_seek": "normal_goal_seek",
        "escape_deadlock": "escaping_deadlock",
        "backtrack": "backtracking",
        "verify_target": "verifying_target",
    }[mode]
    return {
        "task_mode": effective_task_mode,
        "target_object": str(target_object or ""),
        "goal_geometry_available": bool(coarse_goal["goal_geometry_available"]),
        "goal_bearing_from_current_deg": coarse_goal["relative_bearing_deg"],
        "goal_distance_m": coarse_goal["distance_m"],
        "coarse_goal": coarse_goal,
        "supervisor_mode": mode,
        "detour_status": detour_status,
        "goal_resume_hint": (
            "Treat the spatial coarse goal as a soft objective; temporary detours are allowed."
            if coarse_goal["goal_geometry_available"]
            else "Use RGB target-context evidence and verified directional memory; "
            "ObjectNav target geometry is unavailable."
        ),
    }


def apply_policy_safe_objectnav_context(
    vlm_input: Dict[str, Any],
    *,
    target_object: str,
    supervisor_mode: str,
) -> Dict[str, Any]:
    task = vlm_input.setdefault("task", {})
    task["task_mode"] = "ObjNav"
    task["target_object"] = str(target_object or task.get("target_object") or "")
    task["coarse_goal"] = build_policy_coarse_goal()
    return vlm_input


def apply_policy_safe_navigation_context(
    vlm_input: Dict[str, Any],
    *,
    target_object: str,
    supervisor_mode: str,
    spatial_goal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not spatial_goal:
        return apply_policy_safe_objectnav_context(
            vlm_input,
            target_object=target_object,
            supervisor_mode=supervisor_mode,
        )
    task = vlm_input.setdefault("task", {})
    task["task_mode"] = "CoarseGoalNav"
    task["target_object"] = str(target_object or task.get("target_object") or "")
    task["coarse_goal"] = build_policy_coarse_goal(spatial_goal)
    task["coarse_goal_policy"] = (
        "soft objective; temporary opposite-direction detours are allowed for room exit, "
        "deadlock escape, and verified backtracking"
    )
    task["supervisor_mode"] = normalize_supervisor_mode(supervisor_mode)
    return vlm_input


def _assert_unknown_objectnav_geometry(context: Dict[str, Any], path: Sequence[str]) -> None:
    if bool(context.get("goal_geometry_available")):
        raise ValueError("ObjectNav policy input exposes goal geometry at {}".format(".".join(path)))
    for key in ("goal_bearing_from_current_deg", "goal_distance_m"):
        if key in context and context.get(key) is not None:
            raise ValueError("ObjectNav policy input exposes {} at {}".format(key, ".".join(path)))


def assert_policy_input_safe(value: Any) -> None:
    def walk(item: Any, path: Sequence[str]) -> None:
        if isinstance(item, dict):
            for raw_key in item:
                key = str(raw_key).strip().lower()
                if key in _FORBIDDEN_POLICY_KEYS:
                    raise ValueError(
                        "evaluation-only key {} found in policy input at {}".format(
                            raw_key,
                            ".".join(list(path) + [str(raw_key)]),
                        )
                    )

            if str(item.get("task_mode") or "").strip().lower() == "objnav":
                _assert_unknown_objectnav_geometry(item, path)
                coarse_goal = item.get("coarse_goal")
                if isinstance(coarse_goal, dict):
                    if bool(coarse_goal.get("goal_geometry_available")):
                        raise ValueError("ObjectNav coarse goal geometry is marked available")
                    for key in ("map_xy", "relative_bearing_deg", "distance_m"):
                        if coarse_goal.get(key) is not None:
                            raise ValueError("ObjectNav coarse goal exposes {}".format(key))
            if str(item.get("task_mode") or "").strip().lower() == "coarsegoalnav":
                coarse_goal = item.get("coarse_goal")
                if not isinstance(coarse_goal, dict) or not coarse_goal.get("goal_geometry_available"):
                    raise ValueError("CoarseGoalNav requires an explicit spatial coarse goal")
                source = str(coarse_goal.get("source") or "").strip().lower()
                if source not in _ALLOWED_SPATIAL_GOAL_SOURCES:
                    raise ValueError("CoarseGoalNav exposes an unsupported source: {}".format(source))

            for raw_key, child in item.items():
                walk(child, list(path) + [str(raw_key)])
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, list(path) + [str(index)])

    walk(value, ["vlm_input"])


def _translation_m(start: Sequence[float], final: Sequence[float]) -> float:
    try:
        dimensions = min(len(start), len(final), 3)
        if dimensions <= 0:
            return 0.0
        return float(
            math.sqrt(
                sum(
                    (float(final[index]) - float(start[index])) ** 2
                    for index in range(dimensions)
                )
            )
        )
    except Exception:
        return 0.0


def evaluate_execution_progress(
    *,
    start_position_xyz: Sequence[float],
    final_position_xyz: Sequence[float],
    collision_count: int,
    controller_reached_waypoint: bool,
    supervisor_mode: str,
    min_translation_m: float,
    audit_goal_distance_delta_m: Optional[float] = None,
    strategic_evidence: Optional[Dict[str, Any]] = None,
    collision_tolerant_min_translation_m: Optional[float] = None,
    collision_tolerant_max_collisions: int = 0,
) -> Dict[str, Any]:
    mode = normalize_supervisor_mode(supervisor_mode)
    translation = _translation_m(start_position_xyz, final_position_xyz)
    threshold = max(0.0, float(min_translation_m))
    collisions = max(0, int(collision_count or 0))
    translated = translation >= threshold
    reached = bool(controller_reached_waypoint)
    reached_with_motion = bool(reached and translation > 1e-3)
    collision_free = collisions == 0
    collision_tolerant_progress = bool(
        collision_tolerant_min_translation_m is not None
        and collisions > 0
        and collisions <= max(0, int(collision_tolerant_max_collisions))
        and translation >= max(0.0, float(collision_tolerant_min_translation_m))
    )
    collision_policy_passed = bool(collision_free or collision_tolerant_progress)
    execution_success = collision_policy_passed and (translated or reached_with_motion)
    supplied = dict(strategic_evidence) if isinstance(strategic_evidence, dict) else {}
    sector_departure = bool(supplied.get("sector_departure_selected"))
    visual_novelty = bool(supplied.get("visual_novelty_passed"))
    backtrack_selected = bool(
        supplied.get("backtrack_edge_selected")
        or supplied.get("backtrack_edge_completed")
    )
    target_verified = bool(supplied.get("target_verification_stable"))

    if mode == "escape_deadlock":
        strategic_progress = bool(
            collision_policy_passed
            and translated
            and (sector_departure or visual_novelty or backtrack_selected)
        )
        strategic_rule_id = "PROGRESS_ESCAPE_001"
        required_evidence = "translation plus sector departure, visual novelty, or backtrack edge"
    elif mode == "backtrack":
        strategic_progress = bool(
            collision_policy_passed and translated and backtrack_selected
        )
        strategic_rule_id = "PROGRESS_BACKTRACK_001"
        required_evidence = "translation along a verified backtrack edge"
    elif mode == "verify_target":
        strategic_progress = bool(target_verified)
        strategic_rule_id = "PROGRESS_VERIFY_001"
        required_evidence = "temporally stable target verification"
    else:
        strategic_progress = bool(execution_success)
        strategic_rule_id = "PROGRESS_GOAL_SEEK_001"
        required_evidence = "collision-free translation or controller waypoint completion"
    result: Dict[str, Any] = {
        "execution_success": bool(execution_success),
        "strategic_progress": bool(strategic_progress),
        "no_progress": not bool(strategic_progress),
        "translation_m": round(float(translation), 6),
        "collision_count": collisions,
        "controller_reached_waypoint": reached,
        "supervisor_mode": mode,
        "min_translation_m": threshold,
        "strategic_rule_id": strategic_rule_id,
        "required_evidence": required_evidence,
        "evidence": {
            "translation_threshold_met": bool(translated),
            "collision_free": bool(collision_free),
            "collision_tolerant_progress": bool(collision_tolerant_progress),
            "collision_tolerant_min_translation_m": (
                float(collision_tolerant_min_translation_m)
                if collision_tolerant_min_translation_m is not None
                else None
            ),
            "collision_tolerant_max_collisions": int(
                max(0, int(collision_tolerant_max_collisions))
            ),
            "controller_reached_waypoint": reached,
            "controller_reached_with_motion": reached_with_motion,
            "sector_departure_selected": sector_departure,
            "visual_novelty_passed": visual_novelty,
            "backtrack_edge_selected": backtrack_selected,
            "target_verification_stable": target_verified,
            **supplied,
        },
    }
    result["collision_tolerant_progress"] = bool(collision_tolerant_progress)
    if audit_goal_distance_delta_m is not None:
        result["audit_goal_distance_delta_m"] = round(
            float(audit_goal_distance_delta_m),
            6,
        )
    return result
