import os
import json
import hashlib
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import imageio
import numpy as np

from nav_audit import NavigationAuditLogger
from navigation_supervisor import assert_policy_input_safe, evaluate_execution_progress
from qwen_point_planner import render_qwen_debug_frame


STOP_ACTION = 0
FORWARD_ACTION = 1
LEFT_ACTION = 2
RIGHT_ACTION = 3
LOOK_UP_ACTION = 4
LOOK_DOWN_ACTION = 5
DEFAULT_TURN_DEG = 30

_SUPPORTED_QWEN_ACTIONS = {"go", "rotate", "stop", "request_observation"}
_ROTATE_STALL_OBSERVATION_OFFSETS = [-60, -30, 0, 30, 60]
_GO_NO_PROGRESS_OBSERVATION_OFFSETS = [-60, -30, 0, 30, 60]
_LEGACY_FULL_SWEEP_OBSERVATION_OFFSETS = [
    -180,
    -135,
    -90,
    -45,
    0,
    45,
    90,
    135,
]
_FULL_SWEEP_OBSERVATION_OFFSETS = [0, 60, 120, -180, -120, -60]
_FULL_SWEEP_POLICY_ORDER = [-180, -120, -60, 0, 60, 120]
MULTI_VIEW_OBSERVATION_OVERRIDE_REASON = "multi_view_request_observation_suppressed"
_FILE_FINGERPRINT_CACHE: Dict[str, Dict[str, Any]] = {}
_GIT_STATE_CACHE: Optional[Dict[str, Any]] = None


_REPRO_ENV_KEYS = (
    "VOCA_PLANNER",
    "QWEN_BASE_URL",
    "QWEN_MODEL",
    "VOCA_QWEN_MODEL_ROOT",
    "VOCA_QWEN_SEED",
    "QWEN_MAX_TOKENS",
    "VOCA_QWEN_PRIORS_MAX_TOKENS",
    "VOCA_QWEN_MEMORY_SIDECAR",
    "VOCA_QWEN_POINT_VERIFY",
    "VOCA_QWEN_POINT_VERIFY_CONSISTENCY_RETRY",
    "VOCA_QWEN_STOP_VERIFY",
    "VOCA_QWEN_PROACTIVE_STOP_VERIFY",
    "VOCA_QWEN_STOP_VERIFY_MAX_TOKENS",
    "VOCA_QWEN_STOP_IDENTITY_CRITIC",
    "VOCA_QWEN_STOP_CRITIC_MAX_TOKENS",
    "VOCA_QWEN_STOP_CRITIC_THINKING_TOKEN_BUDGET",
    "VOCA_QWEN_STOP_CRITIC_CROP_PADDING",
    "VOCA_STOP_CONFIRMATIONS_REQUIRED",
    "VOCA_STOP_CONFIRM_TRANSLATION_M",
    "VOCA_STOP_CONFIRM_MIN_FRAME_GAP",
    "VOCA_STOP_CONFIRM_MAX_FRAME_GAP",
    "VOCA_STOP_STRONG_TARGET_HEIGHT_FRAC",
    "VOCA_STOP_STRONG_TARGET_AREA_FRAC",
    "VOCA_STOP_SCALE_MIN_HEIGHT_RATIO",
    "VOCA_STOP_SCALE_MIN_AREA_RATIO",
    "VOCA_STOP_NEAR_LATCH_MIN_BBOX_HEIGHT",
    "VOCA_STOP_NEAR_LATCH_MIN_BBOX_AREA",
    "VOCA_STOP_NEAR_LATCH_SINGLE_APPROACH_HEIGHT",
    "VOCA_STOP_NEAR_LATCH_SINGLE_APPROACH_AREA",
    "VOCA_STOP_NEAR_LATCH_PLANT_MIN_APPROACHES",
    "VOCA_STOP_NEAR_LATCH_TTL_FRAMES",
    "VOCA_STOP_NEAR_LATCH_RADIUS_M",
    "VOCA_STOP_TARGET_CENTER_MIN",
    "VOCA_STOP_TARGET_CENTER_MAX",
    "VOCA_STOP_MIN_TARGET_HEIGHT_FRAC",
    "VOCA_STOP_MIN_TARGET_AREA_FRAC",
    "VOCA_STOP_SMALL_TARGET_MIN_HEIGHT_FRAC",
    "VOCA_STOP_SMALL_TARGET_MIN_AREA_FRAC",
    "VOCA_QWEN_REVISIT_VERIFY",
    "VOCA_QWEN_REVISIT_VERIFY_MAX_TOKENS",
    "VOCA_PLACE_EMBEDDER",
    "VOCA_DINOV2_MODEL",
    "VOCA_PLACE_REVISIT_THRESHOLD",
    "VOCA_LOCALIZATION_SOURCE",
    "VOCA_ALLOW_SIM_POSE_POLICY",
    "VOCA_BENCHMARK_SEED",
    "VOCA_BENCHMARK_PROTOCOL",
    "VOCA_COARSE_GOAL_FILE",
    "VOCA_COARSE_GOAL_EXPECTED_SHA256",
    "VOCA_COARSE_GOAL_STOP_RADIUS_LOW_M",
    "VOCA_COARSE_GOAL_STOP_RADIUS_MEDIUM_M",
    "VOCA_COARSE_GOAL_STOP_RADIUS_HIGH_M",
    "VOCA_COARSE_GOAL_TERMINAL_STOP_RADIUS_M",
    "VOCA_COARSE_GOAL_ACQUIRE_RADIUS_LOW_M",
    "VOCA_COARSE_GOAL_ACQUIRE_RADIUS_MEDIUM_M",
    "VOCA_COARSE_GOAL_ACQUIRE_RADIUS_HIGH_M",
    "VOCA_COARSE_GOAL_MAX_GENERIC_GO_ERROR_LOW_DEG",
    "VOCA_COARSE_GOAL_MAX_GENERIC_GO_ERROR_MEDIUM_DEG",
    "VOCA_COARSE_GOAL_MAX_GENERIC_GO_ERROR_HIGH_DEG",
    "VOCA_COARSE_GOAL_TARGET_ALIGNMENT_LOW_DEG",
    "VOCA_COARSE_GOAL_TARGET_ALIGNMENT_MEDIUM_DEG",
    "VOCA_COARSE_GOAL_TARGET_ALIGNMENT_HIGH_DEG",
    "VOCA_HM3D_SUCCESS_DISTANCE_M",
    "VOCA_HM3D_ALLOW_SLIDING",
    "VOCA_EPISODE_START_INDEX",
    "VOCA_EPISODE_POOL_SIZE",
    "VOCA_PIXELNAV_STOP_REJECT_PROB",
    "VOCA_PIXELNAV_CANDIDATE_SCORING",
    "VOCA_NEGATIVE_SATURATION_RECOVERY",
    "VOCA_COLLISION_PROGRESS_MIN_TRANSLATION_M",
    "VOCA_COLLISION_PROGRESS_MAX_COLLISIONS",
    "VOCA_MAX_PLANNING_CYCLES",
    "VOCA_MAX_EPISODE_STEPS",
    "VOCA_TARGET_APPROACH_GO_STEPS",
    "VOCA_TARGET_APPROACH_MAX_TRANSLATION_M",
    "VOCA_TARGET_STOP_REFINEMENT_MAX",
    "VOCA_TARGET_STOP_REFINEMENT_MAX_TRANSLATION_M",
    "VOCA_TARGET_STOP_REFINEMENT_MAX_BBOX_HEIGHT",
    "VOCA_TARGET_STOP_REFINEMENT_MAX_BBOX_AREA",
    "VOCA_TARGET_TERMINAL_MICRO_APPROACH_MAX",
    "VOCA_TARGET_MEMORY_OVERRIDE_MAX_TRANSLATION_M",
    "VOCA_STOP_TARGET_APPROACH_CONFIRM_TRANSLATION_M",
    "VOCA_STOP_ALLOW_TRANSLATION_ONLY_CONFIRM",
    "VOCA_TARGET_LOCK_REACQUIRE_CYCLES",
    "VOCA_TARGET_LOCK_FULL_SWEEP_AFTER",
    "VOCA_TARGET_LOCK_MAX_CYCLES",
    "VOCA_TARGET_LOCK_EXHAUSTION_COOLDOWN_CYCLES",
    "VOCA_TARGET_ANCHOR_REACQUIRE_MAX_ROTATIONS",
    "VOCA_TARGET_ANCHOR_POSE_BEARING_MIN_DISPLACEMENT_M",
    "VOCA_TARGET_ANCHOR_REACQUIRE_MIN_TURN_DEG",
    "VOCA_CAMERA_HORIZONTAL_FOV_DEG",
    "VOCA_VERIFIED_TARGET_MEMORY_TTL_FRAMES",
    "VOCA_VERIFIED_TARGET_MEMORY_RADIUS_M",
    "VOCA_TARGET_APPROACH_FAILURE_LIMIT",
    "VOCA_TARGET_APPROACH_COOLDOWN_CYCLES",
    "VOCA_FORCE_LEAVE_GENERIC_GO_LIMIT",
    "VOCA_FORCE_LEAVE_SAME_ROOM_CYCLES",
    "VOCA_FORCE_LEAVE_FULL_SWEEP_AFTER",
    "VOCA_FORCE_LEAVE_BACKTRACK_AFTER",
    "VOCA_FORCE_LEAVE_NOVEL_SECTOR_AFTER",
    "VOCA_FORCE_LEAVE_NOVEL_SECTOR_RETRIES",
    "VOCA_FORCE_EXIT_VISUAL_REF_REPAIR",
    "VOCA_EFFICIENT_CIRCULAR_SWEEP",
    "VOCA_SAME_ROOM_PROGRESS_RESET_M",
    "VOCA_VLM_CONSECUTIVE_FAILURE_LIMIT",
)


def _file_fingerprint(path_value: Any) -> Dict[str, Any]:
    path = Path(str(path_value or "")).expanduser()
    cache_key = str(path.resolve()) if path.exists() else str(path)
    if cache_key in _FILE_FINGERPRINT_CACHE:
        return dict(_FILE_FINGERPRINT_CACHE[cache_key])
    result: Dict[str, Any] = {
        "path": cache_key,
        "exists": path.is_file(),
        "size_bytes": 0,
        "sha256": "",
    }
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result["size_bytes"] = int(path.stat().st_size)
        result["sha256"] = digest.hexdigest()
    _FILE_FINGERPRINT_CACHE[cache_key] = dict(result)
    return result


def _git_repro_state(repo_root: Path) -> Dict[str, Any]:
    global _GIT_STATE_CACHE
    if _GIT_STATE_CACHE is not None:
        return dict(_GIT_STATE_CACHE)
    state: Dict[str, Any] = {"commit": "", "tracked_worktree_dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        state = {"commit": commit, "tracked_worktree_dirty": bool(status)}
    except (OSError, subprocess.SubprocessError):
        pass
    _GIT_STATE_CACHE = dict(state)
    return state


def _reproducibility_snapshot() -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parent
    config = {key: os.environ.get(key, "") for key in _REPRO_ENV_KEYS}
    source_files = {}
    for name in (
        "qwen_vlm_planner.py",
        "qwen_action_runner.py",
        "voca_memory_sidecar.py",
        "navigation_supervisor.py",
        "pixel_candidate_gate.py",
        "policy_agent.py",
    ):
        source_files[name] = _file_fingerprint(repo_root / name)
    checkpoint = _file_fingerprint(os.environ.get("VOCA_POLICY_CHECKPOINT", ""))
    stable_payload = {
        "config": config,
        "git": _git_repro_state(repo_root),
        "policy_checkpoint": checkpoint,
        "source_files": source_files,
    }
    experiment_fingerprint = hashlib.sha256(
        json.dumps(stable_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        **stable_payload,
        "experiment_fingerprint_sha256": experiment_fingerprint,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": str(np.__version__),
            "imageio": str(getattr(imageio, "__version__", "")),
        },
    }


def _runner_env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off", ""}


def _runner_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _runner_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_runner_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _runner_json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "shape"):
        try:
            return "<{} shape={}>".format(type(value).__name__, list(value.shape))
        except Exception:
            return "<{}>".format(type(value).__name__)
    return "<{}>".format(type(value).__name__)


def _priors_context_item_count(priors: Any) -> int:
    if not isinstance(priors, dict):
        return 0
    total = 0
    for key in ("Supports", "StrongCooccurs", "Gateways", "Lookalikes"):
        value = priors.get(key, [])
        if isinstance(value, (list, tuple, set)):
            total += len(value)
        elif value:
            total += 1
    return int(total)


def turn_actions_for_yaw(yaw_deg: float, turn_deg: int = DEFAULT_TURN_DEG) -> List[int]:
    yaw = float(yaw_deg or 0.0)
    steps = int(round(abs(yaw) / float(turn_deg)))
    if steps <= 0:
        return []
    action = RIGHT_ACTION if yaw > 0 else LEFT_ACTION
    return [action] * steps


def action_from_decision(decision: Optional[Dict[str, Any]]) -> str:
    decision = decision if isinstance(decision, dict) else {}
    vlm_output = decision.get("vlm_output") if isinstance(decision.get("vlm_output"), dict) else {}
    action = str(vlm_output.get("action") or decision.get("action") or "go").strip().lower()
    return action if action in _SUPPORTED_QWEN_ACTIONS else "go"


def vlm_backend_failure_reason(decision: Optional[Dict[str, Any]]) -> str:
    decision = decision if isinstance(decision, dict) else {}
    if decision.get("vlm_json_fallback"):
        return "invalid_vlm_json:{}".format(
            str(decision.get("vlm_json_last_error") or "unknown")[:200]
        )
    warnings = decision.get("warnings")
    warnings = warnings if isinstance(warnings, (list, tuple)) else []
    for warning in warnings:
        text = str(warning or "")
        if text.startswith("qwen_vlm_failed:"):
            return text.split(":", 1)[1][:200] or "qwen_vlm_failed"
    return ""


def is_target_approach_decision(decision: Optional[Dict[str, Any]]) -> bool:
    decision = decision if isinstance(decision, dict) else {}
    override = (
        decision.get("backend_action_override")
        if isinstance(decision.get("backend_action_override"), dict)
        else {}
    )
    return bool(
        decision.get("proactive_target_approach")
        or decision.get("target_context_approach")
        or str(override.get("reason") or "")
        == "stop_confirmation_requires_approach_motion"
    )


def go_step_limit_for_decision(
    decision: Optional[Dict[str, Any]],
    default_max_steps: int,
    *,
    target_approach_steps: Optional[int] = None,
) -> int:
    default_limit = max(1, int(default_max_steps))
    if not is_target_approach_decision(decision):
        return default_limit
    if target_approach_steps is None:
        try:
            target_approach_steps = int(
                os.environ.get("VOCA_TARGET_APPROACH_GO_STEPS", "12")
            )
        except ValueError:
            target_approach_steps = 12
    return max(1, min(default_limit, int(target_approach_steps)))


def target_approach_translation_cap(
    decision: Optional[Dict[str, Any]],
    *,
    default_cap_m: Optional[float] = None,
) -> float:
    if default_cap_m is None:
        try:
            default_cap_m = float(
                os.environ.get("VOCA_TARGET_APPROACH_MAX_TRANSLATION_M", "0.50")
            )
        except ValueError:
            default_cap_m = 0.50
    cap_m = max(0.05, float(default_cap_m))
    decision = decision if isinstance(decision, dict) else {}
    vlm_output = (
        decision.get("vlm_output")
        if isinstance(decision.get("vlm_output"), dict)
        else {}
    )
    memory_override = vlm_output.get("target_approach_memory_override")
    if isinstance(memory_override, dict):
        try:
            micro_cap_m = max(
                0.05,
                float(
                    os.environ.get(
                        "VOCA_TARGET_MEMORY_OVERRIDE_MAX_TRANSLATION_M",
                        "0.25",
                    )
                ),
            )
        except ValueError:
            micro_cap_m = 0.25
        cap_m = min(cap_m, micro_cap_m)
    terminal_refinement = vlm_output.get("target_terminal_refinement")
    if isinstance(terminal_refinement, dict):
        try:
            refinement_cap_m = max(
                0.05,
                float(
                    os.environ.get(
                        "VOCA_TARGET_STOP_REFINEMENT_MAX_TRANSLATION_M",
                        "0.15",
                    )
                ),
            )
        except ValueError:
            refinement_cap_m = 0.15
        cap_m = min(cap_m, refinement_cap_m)
    return float(cap_m)


def should_execute_target_terminal_micro_approach(
    decision: Optional[Dict[str, Any]],
    *,
    executed_count: int = 0,
    max_executions: Optional[int] = None,
) -> bool:
    if not is_target_approach_decision(decision):
        return False
    if max_executions is None:
        try:
            max_executions = max(
                0,
                int(os.environ.get("VOCA_TARGET_TERMINAL_MICRO_APPROACH_MAX", "2")),
            )
        except ValueError:
            max_executions = 2
    if int(executed_count) >= int(max_executions):
        return False

    decision = decision if isinstance(decision, dict) else {}
    verification = (
        decision.get("proactive_stop_probe_result")
        if isinstance(decision.get("proactive_stop_probe_result"), dict)
        else {}
    )
    identity = (
        verification.get("identity_critic")
        if isinstance(verification.get("identity_critic"), dict)
        else {}
    )
    try:
        approach_count = int(
            verification.get("near_goal_visual_latch_approach_count", 0) or 0
        )
        approaches_required = int(
            verification.get("near_goal_visual_latch_approaches_required", 1) or 1
        )
    except (TypeError, ValueError):
        return False
    return bool(
        verification.get("target_evidence_passed")
        and str(verification.get("confidence") or "").lower() == "high"
        and verification.get("target_center_guard_satisfied")
        and identity.get("triggered")
        and identity.get("passed")
        and identity.get("exact_target")
        and approach_count >= approaches_required
        and (
            not verification.get("target_scale_consistent", True)
            or not verification.get("target_approach_execution_satisfied", False)
        )
    )


def rotate_yaw_from_decision(decision: Optional[Dict[str, Any]], default: float = DEFAULT_TURN_DEG) -> float:
    decision = decision if isinstance(decision, dict) else {}
    vlm_output = decision.get("vlm_output") if isinstance(decision.get("vlm_output"), dict) else {}
    control = vlm_output.get("control") if isinstance(vlm_output.get("control"), dict) else {}
    try:
        value = control.get("rotate_yaw_deg", default)
        return float(default if value is None else value)
    except Exception:
        return float(default)


def observation_offsets_from_decision(decision: Optional[Dict[str, Any]]) -> List[int]:
    decision = decision if isinstance(decision, dict) else {}
    vlm_output = decision.get("vlm_output") if isinstance(decision.get("vlm_output"), dict) else {}
    request = vlm_output.get("observation_request") if isinstance(vlm_output.get("observation_request"), dict) else {}

    mode = str(request.get("mode") or "").strip().lower()
    if mode == "full_sweep":
        return full_sweep_observation_offsets()

    offsets = request.get("yaw_offsets_deg")
    if isinstance(offsets, (list, tuple)) and offsets:
        return [int(round(float(offset))) for offset in offsets]

    if mode == "directed_view":
        try:
            return [int(round(float(request.get("center_yaw_deg", 0))))]
        except Exception:
            return [0]

    return [-30, 0, 30]


def observation_mode_from_decision(decision: Optional[Dict[str, Any]]) -> str:
    decision = decision if isinstance(decision, dict) else {}
    vlm_output = (
        decision.get("vlm_output")
        if isinstance(decision.get("vlm_output"), dict)
        else {}
    )
    request = (
        vlm_output.get("observation_request")
        if isinstance(vlm_output.get("observation_request"), dict)
        else {}
    )
    return str(request.get("mode") or "").strip().lower()


def rotate_stall_observation_offsets() -> List[int]:
    return list(_ROTATE_STALL_OBSERVATION_OFFSETS)


def go_no_progress_observation_offsets(supervisor_mode: str = "goal_seek") -> List[int]:
    if str(supervisor_mode or "").strip().lower() in {"escape_deadlock", "backtrack"}:
        return full_sweep_observation_offsets()
    return list(_GO_NO_PROGRESS_OBSERVATION_OFFSETS)


def full_sweep_observation_offsets() -> List[int]:
    if _runner_env_bool("VOCA_EFFICIENT_CIRCULAR_SWEEP", False):
        return list(_FULL_SWEEP_OBSERVATION_OFFSETS)
    return list(_LEGACY_FULL_SWEEP_OBSERVATION_OFFSETS)


def shortest_yaw_delta_deg(target_offset_deg: float, current_offset_deg: float) -> float:
    delta = (
        float(target_offset_deg) - float(current_offset_deg) + 180.0
    ) % 360.0 - 180.0
    if abs(delta + 180.0) < 1e-6:
        return 180.0
    return float(delta)


def order_full_sweep_views_for_policy(
    views: Sequence[np.ndarray],
    angles: Sequence[int],
) -> Tuple[List[np.ndarray], List[int]]:
    if len(views) != len(angles):
        return list(views), [int(angle) for angle in angles]
    by_angle = {
        int(angle): np.asarray(view)
        for view, angle in zip(views, angles)
    }
    if set(by_angle) != set(_FULL_SWEEP_POLICY_ORDER):
        return list(views), [int(angle) for angle in angles]
    return (
        [by_angle[angle] for angle in _FULL_SWEEP_POLICY_ORDER],
        list(_FULL_SWEEP_POLICY_ORDER),
    )


def evaluate_go_progress(
    start_metrics: Dict[str, Any],
    final_metrics: Dict[str, Any],
    *,
    start_position_xyz: Sequence[float],
    final_position_xyz: Sequence[float],
    collision_count: int,
    controller_reached_waypoint: bool,
    supervisor_mode: str,
    min_progress_m: float,
    strategic_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        start_distance = float(start_metrics.get("distance_to_goal"))
        final_distance = float(final_metrics.get("distance_to_goal"))
        distance_delta = start_distance - final_distance
    except Exception:
        start_distance = None
        final_distance = None
        distance_delta = None
    try:
        collision_tolerant_min_translation_m = max(
            0.0,
            float(
                os.environ.get(
                    "VOCA_COLLISION_PROGRESS_MIN_TRANSLATION_M",
                    "0.75",
                )
            ),
        )
    except ValueError:
        collision_tolerant_min_translation_m = 0.75
    try:
        collision_tolerant_max_collisions = max(
            0,
            int(os.environ.get("VOCA_COLLISION_PROGRESS_MAX_COLLISIONS", "1")),
        )
    except ValueError:
        collision_tolerant_max_collisions = 1
    progress = evaluate_execution_progress(
        start_position_xyz=start_position_xyz,
        final_position_xyz=final_position_xyz,
        collision_count=collision_count,
        controller_reached_waypoint=controller_reached_waypoint,
        supervisor_mode=supervisor_mode,
        min_translation_m=min_progress_m,
        audit_goal_distance_delta_m=distance_delta,
        strategic_evidence=strategic_evidence,
        collision_tolerant_min_translation_m=(
            collision_tolerant_min_translation_m
        ),
        collision_tolerant_max_collisions=collision_tolerant_max_collisions,
    )
    progress["success"] = bool(progress["strategic_progress"])
    progress["audit_start_distance_to_goal"] = start_distance
    progress["audit_final_distance_to_goal"] = final_distance
    return progress


def _angular_distance_deg(first: float, second: float) -> float:
    delta = (float(first) - float(second) + 180.0) % 360.0 - 180.0
    return abs(float(delta))


def build_strategic_progress_evidence(
    decision: Dict[str, Any],
    memory_sidecar: Any,
    *,
    sector_departure_threshold_deg: float = 45.0,
) -> Dict[str, Any]:
    relation = str(decision.get("selected_topological_relation_type") or "").strip().lower()
    vlm_output = (
        decision.get("vlm_output")
        if isinstance(decision.get("vlm_output"), dict)
        else {}
    )
    backend_novel_sector = isinstance(
        vlm_output.get("backend_novel_sector_escape"),
        dict,
    )
    selected_angle = _safe_float(decision.get("Angle"), 0.0)
    failures = getattr(memory_sidecar, "directional_failures", []) if memory_sidecar is not None else []
    latest_failure = next(
        (item for item in reversed(failures) if isinstance(item, dict)),
        None,
    )
    direction_change = None
    if latest_failure is not None:
        direction_change = _angular_distance_deg(
            selected_angle,
            _safe_float(latest_failure.get("angle_deg"), 0.0),
        )
    return {
        "schema_version": "strategic_progress_evidence_v1",
        "selected_angle_deg": round(float(selected_angle), 3),
        "last_failed_angle_deg": (
            round(_safe_float(latest_failure.get("angle_deg"), 0.0), 3)
            if latest_failure is not None
            else None
        ),
        "direction_change_from_last_failure_deg": (
            round(float(direction_change), 3) if direction_change is not None else None
        ),
        "sector_departure_selected": bool(
            backend_novel_sector
            or (
                direction_change is not None
                and direction_change >= max(0.0, float(sector_departure_threshold_deg))
            )
        ),
        "backend_novel_sector_selected": bool(backend_novel_sector),
        "backtrack_edge_selected": relation == "backtrack",
        "selected_topological_relation_type": relation or None,
    }


def effective_go_supervisor_mode(
    memory_sidecar: Any,
    decision: Dict[str, Any],
) -> str:
    current = str(getattr(memory_sidecar, "supervisor_mode", "goal_seek") or "goal_seek")
    relation = str(decision.get("selected_topological_relation_type") or "").strip().lower()
    if relation == "backtrack" and current in {"escape_deadlock", "backtrack"}:
        if hasattr(memory_sidecar, "update_supervisor_mode"):
            memory_sidecar.update_supervisor_mode(
                "backtrack",
                reason="selected_verified_backtrack_edge",
            )
        return "backtrack"
    return current


def policy_action_name(action: int) -> str:
    names = {
        STOP_ACTION: "stop",
        FORWARD_ACTION: "move_forward",
        LEFT_ACTION: "turn_left",
        RIGHT_ACTION: "turn_right",
        LOOK_UP_ACTION: "look_up",
        LOOK_DOWN_ACTION: "look_down",
    }
    return names.get(int(action), "action_{}".format(int(action)))


def pitch_offset_from_actions(actions: Sequence[int]) -> int:
    offset = 0
    for action in actions:
        action_i = int(action)
        if action_i == LOOK_UP_ACTION:
            offset += 1
        elif action_i == LOOK_DOWN_ACTION:
            offset -= 1
    return int(offset)


def pitch_reset_actions_for_offset(pitch_offset: int) -> List[int]:
    offset = int(pitch_offset)
    if offset > 0:
        return [LOOK_DOWN_ACTION] * offset
    if offset < 0:
        return [LOOK_UP_ACTION] * abs(offset)
    return []


def build_pitch_reset_audit(
    *,
    policy_actions: Sequence[int],
    pitch_reset_actions: Sequence[int],
    pitch_reset_truncated: bool,
) -> Dict[str, Any]:
    policy_offset = pitch_offset_from_actions(policy_actions)
    reset_offset = pitch_offset_from_actions(pitch_reset_actions)
    reset_actions = [int(action) for action in pitch_reset_actions]
    return {
        "policy_pitch_offset": int(policy_offset),
        "pitch_reset_actions": reset_actions,
        "pitch_reset_action_names": [policy_action_name(action) for action in reset_actions],
        "pitch_reset_steps": int(len(reset_actions)),
        "pitch_reset_truncated": bool(pitch_reset_truncated),
        "residual_pitch_offset": int(policy_offset + reset_offset),
    }


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def classify_go_failure(go_execution: Dict[str, Any]) -> Dict[str, Any]:
    execution = go_execution if isinstance(go_execution, dict) else {}
    progress = execution.get("go_progress") if isinstance(execution.get("go_progress"), dict) else {}
    no_progress = bool(progress.get("no_progress"))
    distance_delta = _safe_float(
        progress.get("audit_goal_distance_delta_m", progress.get("distance_delta_m")),
        0.0,
    )
    translation_m = _safe_float(progress.get("translation_m"), 0.0)
    policy_names = [
        str(name)
        for name in execution.get("policy_action_names", [])
        if str(name)
    ]
    if not policy_names and isinstance(execution.get("policy_actions"), (list, tuple)):
        policy_names = [policy_action_name(action) for action in execution.get("policy_actions", [])]

    policy_steps = _safe_int(execution.get("policy_steps"), len(policy_names))
    max_go_steps = _safe_int(execution.get("max_go_steps"), 0)
    collision_count = _safe_int(execution.get("collision_count"), 0)
    stop_action_seen = bool(execution.get("stop_action_seen"))
    has_forward = "move_forward" in policy_names
    look_only = bool(policy_names) and all(name in {"look_up", "look_down"} for name in policy_names)

    if not no_progress:
        return {
            "primary": "progress",
            "labels": ["progress"],
            "needs_replan": False,
            "message": "local go produced observable execution progress",
        }

    labels: List[str] = ["no_progress"]
    if collision_count > 0:
        labels.append("collision_blocked")
    if distance_delta < 0:
        labels.append("moved_away_from_goal")
    if translation_m <= 1e-6:
        labels.append("stationary_execution")
    if stop_action_seen:
        labels.append("stopped_early")
    if look_only:
        labels.append("look_only_policy")
    if policy_names and not has_forward:
        labels.append("no_translation")
    if max_go_steps > 0 and policy_steps >= max_go_steps and not stop_action_seen:
        labels.append("step_budget_exhausted")
    if bool(execution.get("turn_truncated")):
        labels.append("turn_truncated")

    primary = "no_progress"
    for candidate in (
        "collision_blocked",
        "look_only_policy",
        "stopped_early",
        "no_translation",
        "step_budget_exhausted",
        "turn_truncated",
        "stationary_execution",
    ):
        if candidate in labels:
            primary = candidate
            break

    messages = {
        "collision_blocked": "collision blocked progress",
        "look_only_policy": "policy only changed camera pitch",
        "stopped_early": "policy stopped before useful progress",
        "no_translation": "policy did not move forward",
        "moved_away_from_goal": "audit goal distance increased",
        "step_budget_exhausted": "go step budget was exhausted",
        "turn_truncated": "turn was truncated by episode limit",
        "stationary_execution": "go produced no observable translation",
        "no_progress": "go made less than required progress",
    }
    return {
        "primary": primary,
        "labels": labels,
        "needs_replan": True,
        "message": messages.get(primary, messages["no_progress"]),
    }


def build_go_execution_audit(
    *,
    decision: Dict[str, Any],
    start_metrics: Dict[str, Any],
    final_metrics: Dict[str, Any],
    start_position_xyz: Sequence[float],
    final_position_xyz: Sequence[float],
    policy_actions: Sequence[int],
    collision_count: int,
    stop_action_seen: bool,
    max_go_steps: int,
    turn_yaw_deg: float,
    turn_truncated: bool,
    go_progress: Dict[str, Any],
    pitch_reset: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    actions = [int(action) for action in policy_actions]
    payload = {
        "selected_angle_deg": int(decision.get("Angle", 0) or 0),
        "selected_view_id": decision.get("selected_view_id"),
        "selected_candidate_ref": decision.get("selected_candidate_ref"),
        "selected_topological_candidate_ref": decision.get("selected_topological_candidate_ref"),
        "selected_topological_edge_id": decision.get("selected_topological_edge_id"),
        "selected_topological_relation_type": decision.get("selected_topological_relation_type"),
        "selected_point_px": list(decision.get("Point")) if isinstance(decision.get("Point"), (list, tuple)) else None,
        "target_approach": bool(is_target_approach_decision(decision)),
        "target_context_approach": bool(decision.get("target_context_approach")),
        "target_approach_translation_cap_reached": bool(
            decision.get("runner_target_approach_translation_cap_reached")
        ),
        "target_approach_blocked_probe_action": decision.get(
            "runner_target_approach_blocked_probe_action"
        ),
        "target_terminal_micro_approach": _runner_json_safe(
            dict(decision.get("runner_target_terminal_micro_approach") or {})
        ),
        "policy_actions": actions,
        "policy_action_names": [policy_action_name(action) for action in actions],
        "policy_steps": int(len(actions)),
        "collision_count": int(collision_count),
        "stop_action_seen": bool(stop_action_seen),
        "max_go_steps": int(max_go_steps),
        "turn_yaw_deg": float(turn_yaw_deg),
        "turn_truncated": bool(turn_truncated),
        "start_position_xyz": [float(x) for x in start_position_xyz],
        "final_position_xyz": [float(x) for x in final_position_xyz],
        "start_metrics": _runner_json_safe(dict(start_metrics or {})),
        "final_metrics": _runner_json_safe(dict(final_metrics or {})),
        "go_progress": _runner_json_safe(dict(go_progress or {})),
    }
    if pitch_reset is not None:
        reset_payload = _runner_json_safe(dict(pitch_reset))
        reset_actions = [int(action) for action in reset_payload.get("pitch_reset_actions", []) or []]
        reset_payload["pitch_reset_actions"] = reset_actions
        reset_payload["pitch_reset_action_names"] = [policy_action_name(action) for action in reset_actions]
        payload["pitch_reset"] = reset_payload
    payload["failure_classification"] = classify_go_failure(payload)
    return payload


def attach_go_execution_audit(
    nav_planner: Any,
    audit_logger: NavigationAuditLogger,
    decision: Dict[str, Any],
    go_execution: Dict[str, Any],
) -> None:
    go_execution = _runner_json_safe(dict(go_execution))
    decision["runner_go_execution"] = dict(go_execution)
    if isinstance(go_execution.get("go_progress"), dict):
        decision["runner_go_progress"] = dict(go_execution["go_progress"])
    audit_logger.annotate_pending_runtime("go_execution", go_execution)
    if isinstance(go_execution.get("go_progress"), dict):
        audit_logger.annotate_pending_runtime("go_progress", go_execution["go_progress"])

    qwen_log = getattr(nav_planner, "qwen_call_log", None)
    if isinstance(qwen_log, list) and qwen_log and isinstance(qwen_log[-1], dict):
        qwen_log[-1]["runner_go_execution"] = dict(go_execution)
        if isinstance(go_execution.get("go_progress"), dict):
            qwen_log[-1]["runner_go_progress"] = dict(go_execution["go_progress"])


def attach_runner_action_audit(
    nav_planner: Any,
    *,
    decision: Dict[str, Any],
    vlm_requested_action: str,
    runner_action: str,
    override_reason: str,
    observation_view_count: int,
) -> None:
    fields = {
        "vlm_requested_action": str(vlm_requested_action or "unknown"),
        "runner_action": str(runner_action or "unknown"),
        "runner_override_reason": str(override_reason or ""),
        "runner_observation_view_count": int(observation_view_count),
    }
    decision.update(fields)
    qwen_log = getattr(nav_planner, "qwen_call_log", None)
    if isinstance(qwen_log, list) and qwen_log and isinstance(qwen_log[-1], dict):
        qwen_log[-1].update(fields)


def build_go_no_progress_feedback(decision: Dict[str, Any], go_progress: Dict[str, Any]) -> Dict[str, Any]:
    decision = decision if isinstance(decision, dict) else {}
    vlm_output = decision.get("vlm_output") if isinstance(decision.get("vlm_output"), dict) else {}
    progress = go_progress if isinstance(go_progress, dict) else {}
    policy_progress_keys = (
        "execution_success",
        "strategic_progress",
        "no_progress",
        "translation_m",
        "collision_count",
        "controller_reached_waypoint",
        "supervisor_mode",
        "min_translation_m",
        "evidence",
    )
    feedback = {
        "event": "go_no_progress",
        "selected_view_id": decision.get("selected_view_id"),
        "selected_view_type": vlm_output.get("selected_view_type") or decision.get("selected_view_type"),
        "angle_deg": decision.get("Angle"),
        "point_px": list(decision.get("Point")) if isinstance(decision.get("Point"), (list, tuple)) else None,
        "progress": {
            key: _runner_json_safe(progress[key])
            for key in policy_progress_keys
            if key in progress
        },
    }
    assert_policy_input_safe(feedback)
    return feedback


def sync_go_progress_feedback(nav_planner: Any, decision: Dict[str, Any], go_progress: Dict[str, Any]) -> None:
    go_progress = go_progress if isinstance(go_progress, dict) else {}
    if go_progress.get("no_progress"):
        if hasattr(nav_planner, "record_navigation_feedback"):
            nav_planner.record_navigation_feedback(build_go_no_progress_feedback(decision, go_progress))
        return
    if hasattr(nav_planner, "clear_navigation_feedback"):
        nav_planner.clear_navigation_feedback()


def sync_memory_sidecar_go_execution(
    nav_planner: Any,
    *,
    go_execution: Dict[str, Any],
    frame_index: int,
    start_position_xyz: Sequence[float],
    final_position_xyz: Sequence[float],
    start_heading_rad: float = 0.0,
    final_heading_rad: float = 0.0,
) -> None:
    sidecar = getattr(nav_planner, "memory_sidecar", None)
    if sidecar is None or not hasattr(sidecar, "record_go_execution"):
        return
    localization_contract = getattr(nav_planner, "localization_contract", None)
    policy_pose_available = not isinstance(localization_contract, dict) or bool(
        localization_contract.get("pose_available_to_policy")
    )
    policy_start_position = list(start_position_xyz) if policy_pose_available else []
    policy_final_position = list(final_position_xyz) if policy_pose_available else []
    try:
        sidecar.record_go_execution(
            go_execution=go_execution,
            frame_index=int(frame_index),
            start_position_xyz=[float(x) for x in policy_start_position],
            final_position_xyz=[float(x) for x in policy_final_position],
            start_heading_rad=float(start_heading_rad),
            final_heading_rad=float(final_heading_rad),
        )
    except Exception as exc:
        go_execution["memory_sidecar_error"] = "{}: {}".format(type(exc).__name__, str(exc))


def sync_memory_visualizer_to_decision(nav_planner: Any, decision: Dict[str, Any]) -> None:
    sidecar = getattr(nav_planner, "memory_sidecar", None)
    if sidecar is None or not hasattr(sidecar, "visualizer_state"):
        return
    try:
        memory_visualizer = _runner_json_safe(sidecar.visualizer_state())
    except Exception as exc:
        memory_visualizer = {"enabled": False, "error": "{}: {}".format(type(exc).__name__, str(exc))}
    decision["memory_visualizer"] = memory_visualizer
    qwen_log = getattr(nav_planner, "qwen_call_log", None)
    if isinstance(qwen_log, list) and qwen_log and isinstance(qwen_log[-1], dict):
        qwen_log[-1]["memory_visualizer"] = memory_visualizer


def save_memory_sidecar_artifacts(nav_planner: Any, audit_paths: Dict[str, str], memory_dir: str) -> Dict[str, str]:
    paths = dict(audit_paths or {})
    sidecar = getattr(nav_planner, "memory_sidecar", None)
    if sidecar is None or not hasattr(sidecar, "save_artifacts"):
        return paths
    try:
        sidecar_paths = sidecar.save_artifacts(Path(memory_dir))
    except Exception as exc:
        paths["memory_sidecar_error"] = "{}: {}".format(type(exc).__name__, str(exc))
        return paths
    if isinstance(sidecar_paths, dict):
        paths.update({str(key): str(value) for key, value in sidecar_paths.items()})
    return paths


def save_benchmark_manifest(
    *,
    trajectory_dir: str,
    episode_index: int,
    nav_planner: Any,
    audit_paths: Dict[str, str],
    action_counts: Dict[str, int],
    metrics: Dict[str, Any],
    episode_descriptor: Optional[Dict[str, Any]] = None,
) -> str:
    """Write the reproducibility contract for one Qwen episode.

    Metrics alone do not say which controller, memory version, or audit files
    produced an episode. Keeping this small manifest beside the video makes
    later ablations and paper tables traceable to the exact run.
    """
    sidecar = getattr(nav_planner, "memory_sidecar", None)
    memory_state = {}
    if sidecar is not None and hasattr(sidecar, "visualizer_state"):
        try:
            memory_state = _runner_json_safe(sidecar.visualizer_state())
        except Exception as exc:
            memory_state = {"enabled": False, "error": "{}: {}".format(type(exc).__name__, str(exc))}
    manifest = {
        "schema_version": "voca_qwen_benchmark_manifest_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "episode": int(episode_index),
        "episode_descriptor": _runner_json_safe(episode_descriptor or {}),
        "planner": getattr(nav_planner, "planner_name", "qwen_vlm"),
        "qwen_model": os.environ.get("QWEN_MODEL", ""),
        "qwen_model_root": os.environ.get("VOCA_QWEN_MODEL_ROOT", ""),
        "qwen_base_url": os.environ.get("QWEN_BASE_URL", ""),
        "benchmark_seed": int(os.environ.get("VOCA_BENCHMARK_SEED", "20260710")),
        "localization_contract": _runner_json_safe(
            getattr(nav_planner, "localization_contract", {"source": "unspecified"})
        ),
        "coarse_goal": {
            "provider_record": _runner_json_safe(
                getattr(nav_planner, "_coarse_goal_provider_record", {})
            ),
            "final_runtime_goal": _runner_json_safe(
                getattr(nav_planner, "_runtime_spatial_coarse_goal", {})
            ),
        },
        "memory": memory_state,
        "memory_schema": memory_state.get("schema_version", "null_memory_context_v0"),
        "action_counts": {str(k): int(v) for k, v in (action_counts or {}).items()},
        "metrics_snapshot": _runner_json_safe(dict(metrics or {})),
        "artifacts": {str(k): str(v) for k, v in (audit_paths or {}).items()},
        "videos": {
            "rgb_reasoning": os.path.join(trajectory_dir, "fps.mp4"),
            "topdown_metric": os.path.join(trajectory_dir, "metric.mp4"),
        },
        "reproducibility": _reproducibility_snapshot(),
    }
    path = os.path.join(trajectory_dir, "benchmark_manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def runner_action_after_repeat(
    qwen_action: str,
    consecutive_rotates: int,
    *,
    rotate_to_observe_after: int,
) -> Tuple[str, int, bool]:
    action = str(qwen_action or "go").strip().lower()
    if action != "rotate":
        return action, 0, False

    next_count = int(consecutive_rotates) + 1
    threshold = int(rotate_to_observe_after or 0)
    if threshold > 0 and next_count >= threshold:
        return "request_observation", 0, True
    return "rotate", next_count, False


def runner_action_with_context(
    qwen_action: str,
    consecutive_rotates: int,
    *,
    current_view_count: int,
    rotate_to_observe_after: int,
    observation_mode: str = "",
) -> Tuple[str, int, bool, str]:
    action = str(qwen_action or "go").strip().lower()
    allow_multi_view_full_sweep = str(observation_mode).strip().lower() == "full_sweep"
    if (
        action == "request_observation"
        and int(current_view_count) > 1
        and not allow_multi_view_full_sweep
    ):
        return "rotate", 0, False, MULTI_VIEW_OBSERVATION_OVERRIDE_REASON

    runner_action, next_count, forced_observation = runner_action_after_repeat(
        action,
        consecutive_rotates,
        rotate_to_observe_after=rotate_to_observe_after,
    )
    override_reason = "rotate_stall_forced_observation" if forced_observation else ""
    return runner_action, next_count, forced_observation, override_reason


def rotate_yaw_for_runner_action(decision: Optional[Dict[str, Any]], override_reason: str = "") -> float:
    yaw = rotate_yaw_from_decision(decision)
    if abs(float(yaw)) < 1e-6:
        return float(DEFAULT_TURN_DEG)
    return float(yaw)


def _agent_position_xyz(habitat_env: Any) -> List[float]:
    return [float(x) for x in habitat_env.sim.get_agent_state().position]


def _agent_position_np(habitat_env: Any) -> np.ndarray:
    return np.array(habitat_env.sim.get_agent_state().position, dtype=np.float32)


def _agent_heading_rad(habitat_env: Any) -> float:
    """Return Habitat camera-forward bearing on the world X/Z plane.

    Habitat agent-state rotations transform agent-frame vectors into the
    world frame.  Applying the inverse here mirrors the heading and makes
    world-pose memory point toward the wrong panoramic view.
    """
    rotation = habitat_env.sim.get_agent_state().rotation
    try:
        from habitat_sim.utils.common import quat_rotate_vector

        forward = np.asarray(
            quat_rotate_vector(rotation, np.array([0.0, 0.0, -1.0], dtype=np.float32)),
            dtype=np.float64,
        )
        return float(math.atan2(float(forward[2]), float(forward[0])))
    except Exception:
        return 0.0


def executed_path_tangent_summary(
    position_trace: Sequence[Sequence[float]],
    *,
    min_segment_m: float = 0.02,
) -> Dict[str, Any]:
    """Summarize the first and last translated path segments in world X/Z."""
    points: List[np.ndarray] = []
    for position in position_trace:
        try:
            point = np.asarray(position, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if point.size < 3 or not np.all(np.isfinite(point[:3])):
            continue
        points.append(point[:3])

    translated_segments: List[Tuple[np.ndarray, float]] = []
    path_length_m = 0.0
    threshold = max(0.0, float(min_segment_m))
    for start, final in zip(points, points[1:]):
        delta = final - start
        distance = float(np.linalg.norm(delta))
        path_length_m += distance
        horizontal = np.asarray([delta[0], delta[2]], dtype=np.float64)
        horizontal_distance = float(np.linalg.norm(horizontal))
        if horizontal_distance >= threshold:
            translated_segments.append((delta, horizontal_distance))

    def heading(segment: np.ndarray) -> Tuple[float, float]:
        radians = float(math.atan2(float(segment[2]), float(segment[0])))
        return radians, float(math.degrees(radians))

    summary: Dict[str, Any] = {
        "schema_version": "executed_path_tangents_v1",
        "position_sample_count": len(points),
        "translated_segment_count": len(translated_segments),
        "path_length_m": round(path_length_m, 6),
        "available": bool(translated_segments),
        "source": "executed_position_trace",
    }
    compact_trace: List[np.ndarray] = []
    for point in points:
        if not compact_trace:
            compact_trace.append(point)
            continue
        horizontal_delta = np.asarray(
            [point[0] - compact_trace[-1][0], point[2] - compact_trace[-1][2]],
            dtype=np.float64,
        )
        if float(np.linalg.norm(horizontal_delta)) >= 0.01:
            compact_trace.append(point)
    if points and compact_trace and not np.allclose(
        compact_trace[-1],
        points[-1],
        atol=1e-5,
    ):
        compact_trace.append(points[-1])
    if len(compact_trace) > 32:
        indices = np.linspace(0, len(compact_trace) - 1, num=32, dtype=int)
        compact_trace = [compact_trace[int(index)] for index in indices]
    summary["position_trace_xyz"] = [
        [round(float(value), 6) for value in point[:3]]
        for point in compact_trace
    ]
    summary["compact_position_sample_count"] = len(compact_trace)
    if translated_segments:
        departure_rad, departure_deg = heading(translated_segments[0][0])
        arrival_rad, arrival_deg = heading(translated_segments[-1][0])
        summary.update(
            departure_heading_world_rad=departure_rad,
            departure_heading_world_deg=round(departure_deg, 6),
            arrival_heading_world_rad=arrival_rad,
            arrival_heading_world_deg=round(arrival_deg, 6),
        )
    return summary


def _metrics_snapshot(habitat_env: Any) -> Dict[str, Any]:
    return dict(habitat_env.get_metrics())


def _topdown_frame(
    habitat_env: Any,
    topdown_fn: Optional[Callable[[Dict[str, Any]], np.ndarray]],
    rgb: np.ndarray,
) -> np.ndarray:
    if topdown_fn is None:
        return np.zeros_like(rgb)
    return topdown_fn(_metrics_snapshot(habitat_env))


def _max_steps_reached(habitat_env: Any, max_episode_steps: int) -> bool:
    if int(max_episode_steps or 0) <= 0:
        return False
    return int(_metrics_snapshot(habitat_env).get("num_steps", 0)) >= int(max_episode_steps)


def _step_env(habitat_env: Any, action: int, stats: Dict[str, Any]) -> Dict[str, Any]:
    obs = habitat_env.step(int(action))
    cur = _agent_position_np(habitat_env)
    stats["dist_m"] += float(np.linalg.norm(cur - stats["prev"]))
    stats["prev"] = cur
    return obs


def _append_observation_frame(
    *,
    obs: Dict[str, Any],
    habitat_env: Any,
    episode_images: List[np.ndarray],
    episode_topdowns: List[np.ndarray],
    topdown_fn: Optional[Callable[[Dict[str, Any]], np.ndarray]],
) -> None:
    rgb = obs["rgb"]
    episode_images.append(rgb)
    episode_topdowns.append(_topdown_frame(habitat_env, topdown_fn, rgb))


def _execute_turn(
    *,
    habitat_env: Any,
    obs: Dict[str, Any],
    yaw_deg: float,
    stats: Dict[str, Any],
    episode_images: List[np.ndarray],
    episode_topdowns: List[np.ndarray],
    topdown_fn: Optional[Callable[[Dict[str, Any]], np.ndarray]],
    max_episode_steps: int,
) -> Tuple[Dict[str, Any], bool]:
    truncated = False
    for action in turn_actions_for_yaw(yaw_deg):
        if habitat_env.episode_over:
            break
        if _max_steps_reached(habitat_env, max_episode_steps):
            truncated = True
            break
        obs = _step_env(habitat_env, action, stats)
        _append_observation_frame(
            obs=obs,
            habitat_env=habitat_env,
            episode_images=episode_images,
            episode_topdowns=episode_topdowns,
            topdown_fn=topdown_fn,
        )
    return obs, truncated


def _execute_pitch_reset(
    *,
    habitat_env: Any,
    obs: Dict[str, Any],
    policy_actions: Sequence[int],
    stats: Dict[str, Any],
    episode_images: List[np.ndarray],
    episode_topdowns: List[np.ndarray],
    topdown_fn: Optional[Callable[[Dict[str, Any]], np.ndarray]],
    max_episode_steps: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    planned_actions = pitch_reset_actions_for_offset(pitch_offset_from_actions(policy_actions))
    executed_actions: List[int] = []
    truncated = False
    for action in planned_actions:
        if habitat_env.episode_over:
            break
        if _max_steps_reached(habitat_env, max_episode_steps):
            truncated = True
            break
        obs = _step_env(habitat_env, int(action), stats)
        executed_actions.append(int(action))
        _append_observation_frame(
            obs=obs,
            habitat_env=habitat_env,
            episode_images=episode_images,
            episode_topdowns=episode_topdowns,
            topdown_fn=topdown_fn,
        )
    if len(executed_actions) < len(planned_actions):
        truncated = True
    audit = build_pitch_reset_audit(
        policy_actions=policy_actions,
        pitch_reset_actions=executed_actions,
        pitch_reset_truncated=truncated,
    )
    return obs, audit, truncated


def _collect_observation_sweep(
    *,
    habitat_env: Any,
    obs: Dict[str, Any],
    offsets: Sequence[int],
    stats: Dict[str, Any],
    episode_images: List[np.ndarray],
    episode_topdowns: List[np.ndarray],
    topdown_fn: Optional[Callable[[Dict[str, Any]], np.ndarray]],
    max_episode_steps: int,
) -> Tuple[Dict[str, Any], List[np.ndarray], List[int], bool]:
    views: List[np.ndarray] = []
    angles: List[int] = []
    current_offset = 0
    truncated = False

    for offset in offsets:
        offset_i = int(offset)
        obs, turn_truncated = _execute_turn(
            habitat_env=habitat_env,
            obs=obs,
            yaw_deg=shortest_yaw_delta_deg(offset_i, current_offset),
            stats=stats,
            episode_images=episode_images,
            episode_topdowns=episode_topdowns,
            topdown_fn=topdown_fn,
            max_episode_steps=max_episode_steps,
        )
        truncated = truncated or turn_truncated
        current_offset = offset_i
        if truncated or habitat_env.episode_over:
            break
        views.append(obs["rgb"])
        angles.append(offset_i)

    if current_offset != 0 and (not habitat_env.episode_over) and (not _max_steps_reached(habitat_env, max_episode_steps)):
        obs, turn_truncated = _execute_turn(
            habitat_env=habitat_env,
            obs=obs,
            yaw_deg=shortest_yaw_delta_deg(0, current_offset),
            stats=stats,
            episode_images=episode_images,
            episode_topdowns=episode_topdowns,
            topdown_fn=topdown_fn,
            max_episode_steps=max_episode_steps,
        )
        truncated = truncated or turn_truncated

    if not truncated:
        views, angles = order_full_sweep_views_for_policy(views, angles)

    if not views:
        views = [obs["rgb"]]
        angles = [0]
    return obs, views, angles, truncated


def _selected_view_id(decision: Dict[str, Any], fallback: int) -> int:
    try:
        return int(decision.get("selected_view_id", fallback))
    except Exception:
        return int(fallback)


def _selected_angle(decision: Dict[str, Any], fallback: int = 0) -> int:
    try:
        return int(decision.get("Angle", fallback) or fallback)
    except Exception:
        return int(fallback)


def _append_planning_frame(
    *,
    image: np.ndarray,
    decision: Optional[Dict[str, Any]],
    episode_images: List[np.ndarray],
    decision_by_frame: Dict[int, Dict[str, Any]],
    planning_frame_indices: set,
) -> None:
    idx = len(episode_images)
    if decision is not None:
        decision_by_frame[idx] = decision
    planning_frame_indices.add(idx)
    episode_images.append(image)


def _record_qwen_audit(
    *,
    audit_logger: NavigationAuditLogger,
    habitat_env: Any,
    nav_planner: Any,
    image: np.ndarray,
    frame_index: int,
    call_type: str,
    selected_view_id: Optional[int],
) -> None:
    decision = getattr(nav_planner, "last_decision", None)
    if not decision:
        return
    audit_logger.record_decision(
        decision=decision,
        target_object=habitat_env.current_episode.object_category,
        image_shape=image.shape,
        frame_index=frame_index,
        position_xyz=_agent_position_xyz(habitat_env),
        metrics=_metrics_snapshot(habitat_env),
        priors=getattr(nav_planner, "latest_priors", {}),
        angles=decision.get("angles"),
        call_type=call_type,
        selected_view_id=selected_view_id,
    )


def _qwen_last_point(qwen_last: Dict[str, Any]) -> List[Any]:
    point = qwen_last.get("Point")
    if isinstance(point, list) and len(point) == 2:
        return point
    return [None, None]


def _pixel_candidate_metrics(qwen_calls: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    generated = 0
    offered = 0
    avoided = 0
    verification_required = 0
    visually_supported = 0
    raw_go_count = 0
    gate_passes = 0
    gate_rejections = 0
    selected_verification_required = 0
    saturation_recovery_calls = 0
    saturation_recovery_offered = 0
    saturation_recovery_selected = 0
    saturation_recovery_executed = 0
    saturation_recovery_verifier_passes = 0
    saturation_recovery_verifier_rejections = 0
    marker_alias_repairs = 0
    selected_support_scores: List[float] = []
    for call in qwen_calls:
        if not isinstance(call, dict):
            continue
        vlm_input = call.get("vlm_input") if isinstance(call.get("vlm_input"), dict) else {}
        pixel_block = (
            vlm_input.get("pixel_candidates")
            if isinstance(vlm_input.get("pixel_candidates"), dict)
            else {}
        )
        candidates = [
            item
            for item in pixel_block.get("candidates", [])
            if isinstance(item, dict)
        ]
        excluded = [
            item
            for item in pixel_block.get("excluded_candidates", [])
            if isinstance(item, dict)
        ]
        generated += len(candidates) + len(excluded)
        offered += len(candidates)
        avoided += len(excluded) + sum(1 for item in candidates if item.get("avoid"))
        verification_required += sum(
            1 for item in candidates if item.get("requires_rgb_verification")
        )
        visually_supported += sum(
            1 for item in candidates if item.get("status") == "rgb_visually_supported"
        )
        conditioning = (
            pixel_block.get("policy_conditioning")
            if isinstance(pixel_block.get("policy_conditioning"), dict)
            else {}
        )
        saturation_audit = (
            conditioning.get("negative_saturation_recovery")
            if isinstance(conditioning.get("negative_saturation_recovery"), dict)
            else {}
        )
        if saturation_audit.get("triggered"):
            saturation_recovery_calls += 1
        saturation_recovery_offered += sum(
            bool(item.get("negative_memory_saturation_recovery"))
            for item in candidates
        )

        raw_output = (
            call.get("raw_vlm_output")
            if isinstance(call.get("raw_vlm_output"), dict)
            else {}
        )
        if str(raw_output.get("action") or "").lower() != "go":
            continue
        raw_go_count += 1
        validation = (
            call.get("pixel_candidate_validation")
            if isinstance(call.get("pixel_candidate_validation"), dict)
            else {}
        )
        if validation.get("passed"):
            gate_passes += 1
        else:
            gate_rejections += 1
        marker_alias_repairs += int(bool(validation.get("ref_alias_applied")))
        selected = validation.get("candidate") if isinstance(validation.get("candidate"), dict) else {}
        selected_saturation_recovery = bool(
            selected.get("negative_memory_saturation_recovery")
        )
        if selected_saturation_recovery:
            saturation_recovery_selected += 1
            point_verification = (
                call.get("selected_point_verification")
                if isinstance(call.get("selected_point_verification"), dict)
                else {}
            )
            if point_verification.get("triggered"):
                if point_verification.get("passed"):
                    saturation_recovery_verifier_passes += 1
                else:
                    saturation_recovery_verifier_rejections += 1
            if (
                str(call.get("action") or "").lower() == "go"
                and validation.get("passed")
                and point_verification.get("passed")
            ):
                saturation_recovery_executed += 1
        if selected.get("requires_rgb_verification"):
            selected_verification_required += 1
        evidence = selected.get("visual_evidence") if isinstance(selected.get("visual_evidence"), dict) else {}
        try:
            selected_support_scores.append(float(evidence["support_score"]))
        except (KeyError, TypeError, ValueError):
            pass
    return {
        "generated": generated,
        "offered": offered,
        "avoided": avoided,
        "verification_required": verification_required,
        "visually_supported": visually_supported,
        "raw_go_count": raw_go_count,
        "gate_passes": gate_passes,
        "gate_rejections": gate_rejections,
        "selected_verification_required": selected_verification_required,
        "saturation_recovery_calls": saturation_recovery_calls,
        "saturation_recovery_offered": saturation_recovery_offered,
        "saturation_recovery_selected": saturation_recovery_selected,
        "saturation_recovery_executed": saturation_recovery_executed,
        "saturation_recovery_verifier_passes": saturation_recovery_verifier_passes,
        "saturation_recovery_verifier_rejections": (
            saturation_recovery_verifier_rejections
        ),
        "marker_alias_repairs": marker_alias_repairs,
        "selected_support_avg": (
            float(np.mean(selected_support_scores)) if selected_support_scores else 0.0
        ),
    }


def _strategic_state_metrics(qwen_calls: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    intent_counts: Dict[str, int] = {}
    room_counts: Dict[str, int] = {}
    forced_exit_checks = 0
    forced_exit_passes = 0
    forced_exit_rejections = 0
    memory_backtrack_attempts = 0
    memory_backtrack_executed = 0
    memory_backtrack_point_rejections = 0
    novel_sector_attempts = 0
    novel_sector_executed = 0
    novel_sector_no_progress = 0
    last_state: Dict[str, Any] = {}
    for call in qwen_calls:
        if not isinstance(call, dict):
            continue
        state = (
            call.get("strategic_state")
            if isinstance(call.get("strategic_state"), dict)
            else {}
        )
        if state:
            last_state = dict(state)
            intent = str(state.get("navigation_intent") or "unknown")
            room = str(state.get("current_room") or "unknown")
            intent_counts[intent] = int(intent_counts.get(intent, 0)) + 1
            room_counts[room] = int(room_counts.get(room, 0)) + 1
        validation = (
            call.get("strategic_validation")
            if isinstance(call.get("strategic_validation"), dict)
            else {}
        )
        if validation.get("triggered"):
            forced_exit_checks += 1
            if validation.get("passed"):
                forced_exit_passes += 1
            else:
                forced_exit_rejections += 1
        if isinstance(validation.get("backend_memory_backtrack"), dict):
            memory_backtrack_attempts += 1
            point_verification = (
                call.get("selected_point_verification")
                if isinstance(call.get("selected_point_verification"), dict)
                else {}
            )
            if (
                str(call.get("action") or "").lower() == "go"
                and point_verification.get("passed")
            ):
                memory_backtrack_executed += 1
            elif (
                point_verification.get("triggered")
                and not point_verification.get("passed")
            ):
                memory_backtrack_point_rejections += 1
        if isinstance(validation.get("backend_novel_sector_escape"), dict):
            novel_sector_attempts += 1
            point_verification = (
                call.get("selected_point_verification")
                if isinstance(call.get("selected_point_verification"), dict)
                else {}
            )
            if (
                str(call.get("action") or "").lower() == "go"
                and point_verification.get("passed")
            ):
                novel_sector_executed += 1
            execution = (
                call.get("runner_strategic_go_execution")
                if isinstance(call.get("runner_strategic_go_execution"), dict)
                else {}
            )
            if execution.get("triggered") and not execution.get("passed"):
                novel_sector_no_progress += 1
    return {
        "intent_counts": intent_counts,
        "room_counts": room_counts,
        "forced_exit_checks": forced_exit_checks,
        "forced_exit_passes": forced_exit_passes,
        "forced_exit_rejections": forced_exit_rejections,
        "memory_backtrack_attempts": memory_backtrack_attempts,
        "memory_backtrack_executed": memory_backtrack_executed,
        "memory_backtrack_point_rejections": memory_backtrack_point_rejections,
        "novel_sector_attempts": novel_sector_attempts,
        "novel_sector_executed": novel_sector_executed,
        "novel_sector_no_progress": novel_sector_no_progress,
        "last_state": last_state,
    }


def _memory_verdict_metrics(qwen_calls: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    required_calls = 0
    explicit_calls = 0
    main_output_explicit_calls = 0
    dedicated_verifier_explicit_calls = 0
    dedicated_verifier_unresolved_calls = 0
    missing_calls = 0
    backend_deferred_calls = 0
    operation_counts = {
        "confirm_revisit_node": 0,
        "reject_revisit_candidate": 0,
        "defer_revisit_candidate": 0,
        "request_merge_nodes": 0,
    }
    for call in qwen_calls:
        if not isinstance(call, dict):
            continue
        validation = (
            call.get("memory_contract_validation")
            if isinstance(call.get("memory_contract_validation"), dict)
            else {}
        )
        required = bool(validation.get("required"))
        if not validation:
            backend_generated_proactive = bool(
                call.get("proactive_stop_probe")
                or call.get("target_lock_reacquisition")
                or call.get("proactive_target_centering")
                or call.get("proactive_target_approach")
                or call.get("target_approach_floor_reacquisition")
            )
            if backend_generated_proactive:
                continue
            memory = (
                call.get("vlm_input", {}).get("memory", {})
                if isinstance(call.get("vlm_input"), dict)
                else {}
            )
            refs = memory.get("candidate_refs") if isinstance(memory.get("candidate_refs"), dict) else {}
            required = bool(
                [item for item in refs.get("revisits", []) if isinstance(item, dict)]
            )
        if not required:
            continue
        required_calls += 1
        raw_output = (
            call.get("raw_vlm_output")
            if isinstance(call.get("raw_vlm_output"), dict)
            else {}
        )
        raw_ops = [
            item for item in raw_output.get("memory_ops", []) if isinstance(item, dict)
        ]
        revisit_verification = (
            call.get("revisit_verification")
            if isinstance(call.get("revisit_verification"), dict)
            else {}
        )
        dedicated_operation = (
            revisit_verification.get("memory_op")
            if isinstance(revisit_verification.get("memory_op"), dict)
            else {}
        )
        dedicated_explicit = bool(
            validation.get("dedicated_verifier_triggered")
            and revisit_verification.get("valid")
            and revisit_verification.get("passed")
            and dedicated_operation
        )
        main_explicit = bool(raw_ops) and not bool(
            validation.get("backend_inserted_defer")
        )
        explicit = bool(main_explicit or dedicated_explicit)
        if explicit:
            explicit_calls += 1
        else:
            missing_calls += 1
        if main_explicit:
            main_output_explicit_calls += 1
        elif dedicated_explicit:
            dedicated_verifier_explicit_calls += 1
        elif validation.get("dedicated_verifier_triggered"):
            dedicated_verifier_unresolved_calls += 1
        if validation.get("backend_inserted_defer"):
            backend_deferred_calls += 1
        effective_ops = raw_ops if main_explicit else (
            [dedicated_operation] if dedicated_explicit else []
        )
        for op in effective_ops:
            name = str(op.get("op") or "")
            if name == "confirm_same_place":
                name = "confirm_revisit_node"
            if name in operation_counts:
                operation_counts[name] += 1
    return {
        "required_calls": required_calls,
        "explicit_calls": explicit_calls,
        "main_output_explicit_calls": main_output_explicit_calls,
        "dedicated_verifier_explicit_calls": dedicated_verifier_explicit_calls,
        "dedicated_verifier_unresolved_calls": dedicated_verifier_unresolved_calls,
        "missing_calls": missing_calls,
        "backend_deferred_calls": backend_deferred_calls,
        "explicit_rate": (
            float(explicit_calls) / float(required_calls) if required_calls else 1.0
        ),
        "operation_counts": operation_counts,
    }


def _build_metrics_row(
    *,
    habitat_env: Any,
    nav_planner: Any,
    episode_index: int,
    start_geodesic_m: float,
    episode_time_sec: float,
    truncated_by_max_steps: bool,
    audit_paths: Dict[str, str],
    stats: Dict[str, Any],
    action_counts: Dict[str, int],
    configured_max_episode_steps: int = 0,
    benchmark_invalid_reason: str = "",
) -> Dict[str, Any]:
    qwen_calls = getattr(nav_planner, "qwen_call_log", [])
    qwen_last = qwen_calls[-1] if qwen_calls else {}
    qwen_point = _qwen_last_point(qwen_last if isinstance(qwen_last, dict) else {})
    metrics = _metrics_snapshot(habitat_env)
    collision_metrics = (
        metrics.get("collisions")
        if isinstance(metrics.get("collisions"), dict)
        else {}
    )
    priors = getattr(nav_planner, "latest_priors", {})
    sidecar = getattr(nav_planner, "memory_sidecar", None)
    memory_state = sidecar.visualizer_state() if sidecar is not None else {}
    memory_runtime = (
        memory_state.get("runtime_stats")
        if isinstance(memory_state.get("runtime_stats"), dict)
        else {}
    )
    embedding_state = (
        memory_state.get("place_embedding")
        if isinstance(memory_state.get("place_embedding"), dict)
        else {}
    )
    pixel_metrics = _pixel_candidate_metrics(qwen_calls)
    strategic_metrics = _strategic_state_metrics(qwen_calls)
    strategic_last = strategic_metrics["last_state"]
    point_verifications = [
        call.get("selected_point_verification")
        for call in qwen_calls
        if isinstance(call, dict)
        and isinstance(call.get("selected_point_verification"), dict)
        and call["selected_point_verification"].get("triggered")
    ]
    identity_critics = []
    coarse_goal_guards = []
    for call in qwen_calls:
        if not isinstance(call, dict):
            continue
        seen_in_call = set()
        for field in ("stop_verification", "proactive_stop_probe_result"):
            verification = call.get(field) if isinstance(call.get(field), dict) else {}
            critic = (
                verification.get("identity_critic")
                if isinstance(verification.get("identity_critic"), dict)
                else {}
            )
            if not critic.get("triggered"):
                continue
            signature = json.dumps(critic, sort_keys=True, default=str)
            if signature in seen_in_call:
                continue
            seen_in_call.add(signature)
            identity_critics.append(critic)
        for field in ("stop_verification", "proactive_stop_probe_result"):
            verification = call.get(field) if isinstance(call.get(field), dict) else {}
            guard = (
                verification.get("coarse_goal_consistency_guard")
                if isinstance(
                    verification.get("coarse_goal_consistency_guard"),
                    dict,
                )
                else {}
            )
            if guard.get("triggered"):
                signature = json.dumps(guard, sort_keys=True, default=str)
                if signature not in seen_in_call:
                    seen_in_call.add(signature)
                    coarse_goal_guards.append(guard)
    memory_verdict_metrics = _memory_verdict_metrics(qwen_calls)
    localization_contract = getattr(nav_planner, "localization_contract", {})
    coarse_goal_record = getattr(nav_planner, "_coarse_goal_provider_record", {})
    coarse_goal_record = (
        coarse_goal_record if isinstance(coarse_goal_record, dict) else {}
    )
    runtime_coarse_goal = getattr(nav_planner, "_runtime_spatial_coarse_goal", {})
    runtime_coarse_goal = (
        runtime_coarse_goal if isinstance(runtime_coarse_goal, dict) else {}
    )
    return {
        "episode": episode_index,
        "planner_name": getattr(nav_planner, "planner_name", "qwen_vlm"),
        "object_goal": habitat_env.current_episode.object_category,
        "localization_source": str(
            localization_contract.get("source", "unspecified")
            if isinstance(localization_contract, dict)
            else "unspecified"
        ),
        "uses_sim_ground_truth_pose": int(
            bool(localization_contract.get("uses_sim_ground_truth"))
            if isinstance(localization_contract, dict)
            else False
        ),
        "benchmark_protocol": os.environ.get("VOCA_BENCHMARK_PROTOCOL", "unspecified"),
        "coarse_goal_enabled": int(bool(coarse_goal_record)),
        "coarse_goal_source": str(runtime_coarse_goal.get("source") or ""),
        "coarse_goal_uncertainty": str(
            runtime_coarse_goal.get("uncertainty") or ""
        ),
        "coarse_goal_map_xy": json.dumps(
            coarse_goal_record.get("map_xy"),
            separators=(",", ":"),
        ) if coarse_goal_record else "",
        "coarse_goal_provider_sha256": str(
            coarse_goal_record.get("provider_sha256") or ""
        ),
        "coarse_goal_final_distance_m": (
            float(runtime_coarse_goal.get("distance_m"))
            if runtime_coarse_goal.get("distance_m") is not None
            else ""
        ),
        "coarse_goal_guard_calls": int(len(coarse_goal_guards)),
        "coarse_goal_distractor_rejections": int(
            sum(
                guard.get("classification")
                in {
                    "same_category_instance_outside_coarse_goal_region",
                    "same_category_instance_directionally_inconsistent",
                }
                for guard in coarse_goal_guards
            )
        ),
        "coarse_goal_stop_deferrals": int(
            sum(
                guard.get("classification")
                == "target_acquisition_allowed_stop_deferred"
                for guard in coarse_goal_guards
            )
        ),
        "coarse_goal_direction_guard_calls": int(
            getattr(nav_planner, "_coarse_direction_guard_count", 0)
        ),
        "coarse_goal_direction_rejections": int(
            getattr(nav_planner, "_coarse_direction_rejection_count", 0)
        ),
        "coarse_goal_direction_detour_exemptions": int(
            getattr(
                nav_planner,
                "_coarse_direction_detour_exemption_count",
                0,
            )
        ),
        "success_distance_m": float(
            os.environ.get("VOCA_HM3D_SUCCESS_DISTANCE_M", "0.10")
        ),
        "allow_sliding": int(_runner_env_bool("VOCA_HM3D_ALLOW_SLIDING", False)),
        "success": metrics["success"],
        "spl": metrics["spl"],
        "soft_spl": float(metrics.get("soft_spl", 0.0) or 0.0),
        "collision_count": int(collision_metrics.get("count", 0) or 0),
        "start_distance_to_goal": start_geodesic_m,
        "final_distance_to_goal": metrics["distance_to_goal"],
        "llm_calls": int(getattr(nav_planner, "llm_call_count", 0)),
        "llm_calls_deadlock": 0,
        "llm_calls_verification": int(
            getattr(nav_planner, "point_verification_calls", 0)
            + getattr(nav_planner, "stop_verification_calls", 0)
            + getattr(nav_planner, "revisit_verification_calls", 0)
        ),
        "llm_success_calls": int(getattr(nav_planner, "llm_success_count", 0)),
        "llm_error_calls": int(getattr(nav_planner, "llm_error_count", 0)),
        "llm_avg_time_sec": float(np.mean(nav_planner.llm_durations)) if len(getattr(nav_planner, "llm_durations", [])) > 0 else 0.0,
        "priors_calls": int(len(getattr(nav_planner, "priors_durations", []))),
        "priors_avg_time_sec": (
            float(np.mean(getattr(nav_planner, "priors_durations", [])))
            if getattr(nav_planner, "priors_durations", [])
            else 0.0
        ),
        "qwen_navigation_vlm_calls": int(
            len(getattr(nav_planner, "navigation_vlm_durations", []))
        ),
        "qwen_navigation_vlm_avg_time_sec": (
            float(np.mean(getattr(nav_planner, "navigation_vlm_durations", [])))
            if getattr(nav_planner, "navigation_vlm_durations", [])
            else 0.0
        ),
        "llm_last_error": str(getattr(nav_planner, "llm_last_error", "")) if getattr(nav_planner, "llm_last_error", "") else "",
        "priors_success_count": int(getattr(nav_planner, "priors_success_count", 0)),
        "priors_parse_fail_count": int(getattr(nav_planner, "priors_parse_fail_count", 0)),
        "priors_fallback_count": int(getattr(nav_planner, "priors_fallback_count", 0)),
        "priors_context_item_count": _priors_context_item_count(priors),
        "priors_last_error": str(getattr(nav_planner, "priors_last_error", "")) if getattr(nav_planner, "priors_last_error", "") else "",
        "vlm_json_fallback_count": int(getattr(nav_planner, "vlm_json_fallback_count", 0)),
        "vlm_json_last_error": str(getattr(nav_planner, "vlm_json_last_error", "")) if getattr(nav_planner, "vlm_json_last_error", "") else "",
        "qwen_point_verification_calls": int(
            getattr(nav_planner, "point_verification_calls", 0)
        ),
        "qwen_point_verification_passes": int(
            getattr(nav_planner, "point_verification_pass_count", 0)
        ),
        "qwen_point_verification_rejections": int(
            getattr(nav_planner, "point_verification_rejection_count", 0)
        ),
        "qwen_point_verification_errors": int(
            getattr(nav_planner, "point_verification_error_count", 0)
        ),
        "qwen_point_verification_backend_calls": int(
            sum(
                int(verification.get("backend_call_count", 1) or 1)
                for verification in point_verifications
            )
        ),
        "qwen_point_verification_consistency_retries": int(
            sum(
                bool(verification.get("consistency_retry_triggered"))
                for verification in point_verifications
            )
        ),
        "qwen_point_verification_avg_time_sec": (
            float(
                np.mean(
                    getattr(nav_planner, "point_verification_durations", [])
                )
            )
            if getattr(nav_planner, "point_verification_durations", [])
            else 0.0
        ),
        "qwen_stop_verification_calls": int(
            getattr(nav_planner, "stop_verification_calls", 0)
        ),
        "qwen_stop_verification_passes": int(
            getattr(nav_planner, "stop_verification_pass_count", 0)
        ),
        "qwen_stop_verification_rejections": int(
            getattr(nav_planner, "stop_verification_rejection_count", 0)
        ),
        "qwen_stop_verification_errors": int(
            getattr(nav_planner, "stop_verification_error_count", 0)
        ),
        "qwen_stop_verification_avg_time_sec": (
            float(np.mean(getattr(nav_planner, "stop_verification_durations", [])))
            if getattr(nav_planner, "stop_verification_durations", [])
            else 0.0
        ),
        "qwen_stop_identity_critic_calls": int(len(identity_critics)),
        "qwen_stop_identity_critic_passes": int(
            sum(bool(item.get("passed")) for item in identity_critics)
        ),
        "qwen_stop_identity_critic_rejections": int(
            sum(not bool(item.get("passed")) for item in identity_critics)
        ),
        "qwen_stop_identity_critic_errors": int(
            sum(bool(item.get("error")) for item in identity_critics)
        ),
        "qwen_stop_identity_rejection_cooldown_skips": int(
            getattr(
                nav_planner,
                "_identity_rejection_cooldown_skip_count",
                0,
            )
        ),
        "qwen_revisit_verification_calls": int(
            getattr(nav_planner, "revisit_verification_calls", 0)
        ),
        "qwen_revisit_verification_resolved": int(
            getattr(nav_planner, "revisit_verification_resolved_count", 0)
        ),
        "qwen_revisit_verification_errors": int(
            getattr(nav_planner, "revisit_verification_error_count", 0)
        ),
        "qwen_revisit_verification_avg_time_sec": (
            float(np.mean(getattr(nav_planner, "revisit_verification_durations", [])))
            if getattr(nav_planner, "revisit_verification_durations", [])
            else 0.0
        ),
        "qwen_point_calls": int(len(qwen_calls)),
        "qwen_point_fallback_count": int(sum(1 for call in qwen_calls if isinstance(call, dict) and call.get("fallback"))),
        "qwen_rgb_candidates_generated": int(pixel_metrics["generated"]),
        "qwen_rgb_candidates_offered": int(pixel_metrics["offered"]),
        "qwen_rgb_candidates_avoided": int(pixel_metrics["avoided"]),
        "qwen_rgb_candidates_verification_required": int(
            pixel_metrics["verification_required"]
        ),
        "qwen_rgb_candidates_visually_supported": int(
            pixel_metrics["visually_supported"]
        ),
        "qwen_raw_go_decisions": int(pixel_metrics["raw_go_count"]),
        "qwen_pixel_candidate_gate_passes": int(pixel_metrics["gate_passes"]),
        "qwen_pixel_candidate_gate_rejections": int(pixel_metrics["gate_rejections"]),
        "qwen_selected_candidates_verification_required": int(
            pixel_metrics["selected_verification_required"]
        ),
        "qwen_selected_candidate_visual_support_avg": float(
            pixel_metrics["selected_support_avg"]
        ),
        "qwen_negative_saturation_recovery_calls": int(
            pixel_metrics["saturation_recovery_calls"]
        ),
        "qwen_negative_saturation_recovery_candidates_offered": int(
            pixel_metrics["saturation_recovery_offered"]
        ),
        "qwen_negative_saturation_recovery_candidates_selected": int(
            pixel_metrics["saturation_recovery_selected"]
        ),
        "qwen_negative_saturation_recovery_go_executed": int(
            pixel_metrics["saturation_recovery_executed"]
        ),
        "qwen_negative_saturation_recovery_verifier_passes": int(
            pixel_metrics["saturation_recovery_verifier_passes"]
        ),
        "qwen_negative_saturation_recovery_verifier_rejections": int(
            pixel_metrics["saturation_recovery_verifier_rejections"]
        ),
        "qwen_pixel_candidate_marker_alias_repairs": int(
            pixel_metrics["marker_alias_repairs"]
        ),
        "qwen_strategic_forced_exit_checks": int(
            strategic_metrics["forced_exit_checks"]
        ),
        "qwen_strategic_forced_exit_passes": int(
            strategic_metrics["forced_exit_passes"]
        ),
        "qwen_strategic_forced_exit_rejections": int(
            strategic_metrics["forced_exit_rejections"]
        ),
        "qwen_strategic_force_leave_overrides": int(
            getattr(nav_planner, "_strategic_force_leave_override_count", 0)
        ),
        "qwen_strategic_forced_exit_failures": int(
            sum(
                int(value)
                for value in getattr(
                    nav_planner,
                    "_forced_exit_failure_counts",
                    {},
                ).values()
            )
        ),
        "qwen_strategic_forced_exit_failure_streak": int(
            getattr(nav_planner, "_forced_exit_failure_streak", 0)
        ),
        "qwen_strategic_full_sweep_requests": int(
            getattr(nav_planner, "_forced_exit_full_sweep_count", 0)
        ),
        "qwen_strategic_memory_backtrack_overrides": int(
            getattr(nav_planner, "_forced_exit_memory_backtrack_count", 0)
        ),
        "qwen_strategic_memory_backtrack_attempts": int(
            strategic_metrics["memory_backtrack_attempts"]
        ),
        "qwen_strategic_memory_backtrack_executed": int(
            strategic_metrics["memory_backtrack_executed"]
        ),
        "qwen_strategic_memory_backtrack_point_rejections": int(
            strategic_metrics["memory_backtrack_point_rejections"]
        ),
        "qwen_strategic_novel_sector_overrides": int(
            getattr(nav_planner, "_forced_exit_novel_sector_count", 0)
        ),
        "qwen_strategic_novel_sector_attempts": int(
            strategic_metrics["novel_sector_attempts"]
        ),
        "qwen_strategic_novel_sector_executed": int(
            strategic_metrics["novel_sector_executed"]
        ),
        "qwen_strategic_novel_sector_no_progress": int(
            strategic_metrics["novel_sector_no_progress"]
        ),
        "qwen_strategic_normal_backtrack_rejections": int(
            getattr(
                nav_planner,
                "_normal_goal_seek_backtrack_rejection_count",
                0,
            )
        ),
        "qwen_strategic_exit_alias_repairs": int(
            getattr(nav_planner, "_strategic_exit_alias_repair_count", 0)
        ),
        "qwen_strategic_visual_exit_ref_repairs": int(
            getattr(
                nav_planner,
                "_strategic_visual_exit_ref_repair_count",
                0,
            )
        ),
        "qwen_exit_floor_relaxations": int(
            sum(
                1
                for call in qwen_calls
                if isinstance(call, dict)
                and isinstance(call.get("selected_point_verification"), dict)
                and isinstance(
                    call["selected_point_verification"].get(
                        "semantic_exit_floor_relaxation"
                    ),
                    dict,
                )
                and call["selected_point_verification"][
                    "semantic_exit_floor_relaxation"
                ].get("triggered")
            )
        ),
        "qwen_strategic_search_current_room": int(
            strategic_metrics["intent_counts"].get("search_current_room", 0)
        ),
        "qwen_strategic_leave_current_room": int(
            strategic_metrics["intent_counts"].get("leave_current_room", 0)
        ),
        "qwen_strategic_traverse_gateway": int(
            strategic_metrics["intent_counts"].get("traverse_gateway", 0)
        ),
        "qwen_strategic_approach_target": int(
            strategic_metrics["intent_counts"].get("approach_target", 0)
        ),
        "qwen_strategic_verify_target": int(
            strategic_metrics["intent_counts"].get("verify_target", 0)
        ),
        "qwen_strategic_escape_deadlock": int(
            strategic_metrics["intent_counts"].get("escape_deadlock", 0)
        ),
        "qwen_last_room": str(strategic_last.get("current_room") or ""),
        "qwen_last_navigation_intent": str(
            strategic_last.get("navigation_intent") or ""
        ),
        "qwen_last_room_search_status": str(
            strategic_last.get("room_search_status") or ""
        ),
        "memory_verdict_required_calls": int(memory_verdict_metrics["required_calls"]),
        "memory_verdict_explicit_calls": int(memory_verdict_metrics["explicit_calls"]),
        "memory_verdict_main_output_explicit_calls": int(
            memory_verdict_metrics["main_output_explicit_calls"]
        ),
        "memory_verdict_dedicated_verifier_explicit_calls": int(
            memory_verdict_metrics["dedicated_verifier_explicit_calls"]
        ),
        "memory_verdict_dedicated_verifier_unresolved_calls": int(
            memory_verdict_metrics["dedicated_verifier_unresolved_calls"]
        ),
        "memory_verdict_missing_calls": int(memory_verdict_metrics["missing_calls"]),
        "memory_verdict_backend_deferred_calls": int(
            memory_verdict_metrics["backend_deferred_calls"]
        ),
        "memory_verdict_explicit_rate": float(memory_verdict_metrics["explicit_rate"]),
        "memory_verdict_confirm_calls": int(
            memory_verdict_metrics["operation_counts"]["confirm_revisit_node"]
        ),
        "memory_verdict_reject_calls": int(
            memory_verdict_metrics["operation_counts"]["reject_revisit_candidate"]
        ),
        "memory_verdict_defer_calls": int(
            memory_verdict_metrics["operation_counts"]["defer_revisit_candidate"]
        ),
        "qwen_last_angle": qwen_last.get("Angle", "") if isinstance(qwen_last, dict) else "",
        "qwen_last_u": qwen_point[0],
        "qwen_last_v": qwen_point[1],
        "qwen_last_confidence": qwen_last.get("Confidence", "") if isinstance(qwen_last, dict) else "",
        "qwen_last_reason": qwen_last.get("Reason", "") if isinstance(qwen_last, dict) else "",
        "steps_json": audit_paths.get("steps_json", ""),
        "memory_graph_json": audit_paths.get("memory_graph_json", ""),
        "yoloe_detect_calls": int(len(getattr(nav_planner, "yoloe_durations", []))),
        "yoloe_detect_avg_time_sec": 0.0,
        "yoloe_detect_total_time_sec": 0.0,
        "episode_time_sec": float(episode_time_sec),
        "configured_max_episode_steps": int(configured_max_episode_steps or 0),
        "truncated_by_max_steps": int(truncated_by_max_steps),
        "benchmark_valid": int(not bool(benchmark_invalid_reason)),
        "benchmark_invalid_reason": str(benchmark_invalid_reason or ""),
        "qwen_vlm_circuit_breaker_tripped": int(
            action_counts.get("vlm_circuit_breaker_tripped", 0)
        ),
        "qwen_vlm_max_consecutive_failures": int(
            action_counts.get("vlm_max_consecutive_failures", 0)
        ),
        "planning_cycles": int(action_counts.get("planning_cycles", 0)),
        "planning_cycle_limit_reached": int(
            action_counts.get("planning_cycle_limit_reached", 0)
        ),
        "num_steps": int(metrics.get("num_steps", 0)),
        "total_distance_m": float(stats["dist_m"]),
        "qwen_go_actions": int(action_counts.get("go", 0)),
        "qwen_rotate_actions": int(action_counts.get("rotate", 0)),
        "qwen_zero_yaw_rotate_corrections": int(
            action_counts.get("zero_yaw_rotate_correction", 0)
        ),
        "qwen_observation_requests": int(action_counts.get("request_observation", 0)),
        "qwen_forced_observation_requests": int(action_counts.get("forced_request_observation", 0)),
        "qwen_runner_overrides": int(action_counts.get("runner_override", 0)),
        "qwen_go_no_progress": int(action_counts.get("go_no_progress", 0)),
        "qwen_collision_tolerant_progress_actions": int(
            action_counts.get("collision_tolerant_progress", 0)
        ),
        "qwen_target_approach_limited_actions": int(
            action_counts.get("target_approach_limited", 0)
        ),
        "qwen_target_approach_memory_overrides": int(
            action_counts.get("target_approach_memory_override", 0)
        ),
        "qwen_target_stop_refinement_actions": int(
            action_counts.get("target_stop_refinement", 0)
        ),
        "qwen_target_anchor_reacquisition_actions": int(
            action_counts.get("target_anchor_reacquisition", 0)
        ),
        "qwen_target_lock_exhaustions": int(
            getattr(nav_planner, "_target_lock_exhaustion_count", 0)
        ),
        "qwen_target_terminal_micro_approach_actions": int(
            action_counts.get("target_terminal_micro_approach", 0)
        ),
        "qwen_target_terminal_micro_approach_collisions": int(
            action_counts.get("target_terminal_micro_approach_collision", 0)
        ),
        "qwen_target_approach_detour_triggers": int(
            action_counts.get("target_approach_detour_triggered", 0)
        ),
        "qwen_pitch_reset_actions": int(action_counts.get("pitch_reset_action", 0)),
        "qwen_stop_actions": int(action_counts.get("stop", 0)),
        "qwen_memory_backtrack_go_actions": int(
            sum(
                1
                for call in qwen_calls
                if isinstance(call, dict)
                and action_from_decision(call) == "go"
                and call.get("selected_topological_relation_type") == "backtrack"
            )
        ),
        "qwen_memory_negative_selection_rejections": int(
            sum(
                1
                for call in qwen_calls
                if isinstance(call, dict)
                and isinstance(call.get("pixel_candidate_validation"), dict)
                and call["pixel_candidate_validation"].get("reason")
                == "selected_candidate_avoid_true"
            )
        ),
        "memory_place_nodes": int(memory_state.get("num_place_nodes", 0) or 0),
        "memory_failed_frontier_nodes": int(
            memory_state.get("num_failed_frontier_nodes", 0) or 0
        ),
        "memory_negative_edges": int(memory_state.get("num_deadlock_edges", 0) or 0),
        "memory_same_place_edges": int(memory_state.get("num_same_place_edges", 0) or 0),
        "memory_visual_keyframes": int(memory_state.get("num_visual_keyframes", 0) or 0),
        "memory_embedding_backend": str(embedding_state.get("backend") or ""),
        "memory_revisit_threshold": float(
            embedding_state.get("revisit_threshold", 0.0) or 0.0
        ),
        "memory_embedding_observations": int(
            memory_runtime.get("embedding_observations", 0) or 0
        ),
        "memory_embedding_views": int(memory_runtime.get("embedding_views", 0) or 0),
        "memory_revisit_queries": int(memory_runtime.get("revisit_queries", 0) or 0),
        "memory_revisit_candidate_contexts": int(
            memory_runtime.get("revisit_candidate_contexts", 0) or 0
        ),
        "memory_revisit_candidates_offered": int(
            memory_runtime.get("revisit_candidates_offered", 0) or 0
        ),
        "memory_ops_requested": int(memory_runtime.get("memory_ops_requested", 0) or 0),
        "memory_ops_accepted": int(memory_runtime.get("memory_ops_accepted", 0) or 0),
        "memory_ops_rejected": int(memory_runtime.get("memory_ops_rejected", 0) or 0),
        "memory_revisits_confirmed": int(
            memory_runtime.get("revisits_confirmed", 0) or 0
        ),
        "memory_revisits_rejected": int(
            memory_runtime.get("revisits_rejected", 0) or 0
        ),
        "memory_soft_merges": int(memory_runtime.get("soft_merges", 0) or 0),
        "memory_object_belief_updates": int(
            memory_runtime.get("object_belief_updates", 0) or 0
        ),
        "memory_semantic_state_updates": int(
            memory_runtime.get("semantic_state_updates", 0) or 0
        ),
        "memory_room_category_updates": int(
            memory_runtime.get("room_category_updates", 0) or 0
        ),
        "memory_room_transitions": int(
            memory_runtime.get("room_transitions", 0) or 0
        ),
    }


def run_qwen_vlm_action_episode(
    habitat_env: Any,
    nav_planner: Any,
    nav_executor: Any,
    *,
    episode_index: int,
    output_dir: str,
    max_episode_steps: int = 0,
    topdown_fn: Optional[Callable[[Dict[str, Any]], np.ndarray]] = None,
    coarse_goal_provider: Optional[Any] = None,
) -> Dict[str, Any]:
    obs = habitat_env.reset()
    coarse_goal_record: Optional[Dict[str, Any]] = None
    if coarse_goal_provider is not None:
        coarse_goal_record = coarse_goal_provider.resolve(
            episode_index=int(episode_index),
            episode=habitat_env.current_episode,
        )
    if hasattr(nav_planner, "set_pixel_candidate_scorer"):
        scorer = None
        if _runner_env_bool("VOCA_PIXELNAV_CANDIDATE_SCORING", True) and hasattr(
            nav_executor, "score_pixel_candidates"
        ):
            scorer = nav_executor.score_pixel_candidates
        nav_planner.set_pixel_candidate_scorer(scorer)
    trajectory_dir = os.path.join(output_dir, "trajectory_%d" % episode_index)
    os.makedirs(trajectory_dir, exist_ok=True)
    if hasattr(nav_planner, "set_qwen_call_log_path"):
        nav_planner.set_qwen_call_log_path(os.path.join(trajectory_dir, "qwen_calls.jsonl"))

    stats = {"dist_m": 0.0, "prev": _agent_position_np(habitat_env)}
    audit_logger = NavigationAuditLogger()
    audit_paths = {"steps_json": "", "memory_graph_json": ""}
    fps_writer = imageio.get_writer("%s/fps.mp4" % trajectory_dir, fps=4)
    topdown_writer = imageio.get_writer("%s/metric.mp4" % trajectory_dir, fps=4)
    episode_images = [obs["rgb"]]
    episode_topdowns = [_topdown_frame(habitat_env, topdown_fn, obs["rgb"])]
    decision_by_frame: Dict[int, Dict[str, Any]] = {}
    planning_frame_indices = set()
    current_images = [obs["rgb"]]
    current_angles = [0]
    truncated_by_max_steps = False
    action_counts: Dict[str, int] = {}
    consecutive_rotates = 0
    planning_cycles = 0
    target_terminal_micro_approach_count = 0
    consecutive_vlm_failures = 0
    benchmark_invalid_reason = ""
    try:
        max_planning_cycles = max(
            1,
            int(os.environ.get("VOCA_MAX_PLANNING_CYCLES", "128")),
        )
    except ValueError:
        max_planning_cycles = 128
    try:
        vlm_failure_limit = max(
            1,
            int(os.environ.get("VOCA_VLM_CONSECUTIVE_FAILURE_LIMIT", "3")),
        )
    except ValueError:
        vlm_failure_limit = 3

    start_geodesic_m = float(_metrics_snapshot(habitat_env)["distance_to_goal"])
    episode_t0 = time.perf_counter()

    nav_planner.reset(habitat_env.current_episode.object_category)
    nav_planner._coarse_goal_provider_record = dict(coarse_goal_record or {})
    if coarse_goal_provider is not None and coarse_goal_record is not None:
        initial_position = _agent_position_xyz(habitat_env)
        initial_heading = _agent_heading_rad(habitat_env)
        initial_spatial_goal = coarse_goal_provider.runtime_goal(
            coarse_goal_record,
            position_xyz=initial_position,
            heading_rad=initial_heading,
        )
        nav_planner.update_runtime_context(
            position_xyz=initial_position,
            heading_rad=initial_heading,
            frame_index=len(episode_images) - 1,
            metrics=_metrics_snapshot(habitat_env),
            spatial_coarse_goal=initial_spatial_goal,
        )
    nav_planner.query_priors_text()
    max_go_steps = int(os.environ.get("VOCA_QWEN_GO_STEPS", "16"))
    rotate_to_observe_after = int(os.environ.get("VOCA_QWEN_ROTATE_TO_OBSERVE_AFTER", "2"))
    min_go_progress_m = float(os.environ.get("VOCA_QWEN_GO_MIN_PROGRESS_M", "0.05"))

    while not habitat_env.episode_over:
        if planning_cycles >= max_planning_cycles:
            action_counts["planning_cycle_limit_reached"] = 1
            truncated_by_max_steps = True
            break
        planning_cycles += 1
        action_counts["planning_cycles"] = planning_cycles
        if _max_steps_reached(habitat_env, max_episode_steps):
            truncated_by_max_steps = True
            break

        if hasattr(nav_planner, "update_runtime_context"):
            try:
                runtime_position = _agent_position_xyz(habitat_env)
                runtime_heading = _agent_heading_rad(habitat_env)
                runtime_spatial_goal = None
                if coarse_goal_provider is not None and coarse_goal_record is not None:
                    runtime_spatial_goal = coarse_goal_provider.runtime_goal(
                        coarse_goal_record,
                        position_xyz=runtime_position,
                        heading_rad=runtime_heading,
                    )
                nav_planner.update_runtime_context(
                    position_xyz=runtime_position,
                    heading_rad=runtime_heading,
                    frame_index=len(episode_images) - 1,
                    metrics=_metrics_snapshot(habitat_env),
                    spatial_coarse_goal=runtime_spatial_goal,
                )
            except Exception:
                pass

        (
            goal_image,
            goal_mask,
            _debug_image,
            vis_rgb,
            direction,
            _pri_flag,
            _obj_detected,
        ) = nav_planner.make_plan_from_views(
            current_images,
            current_angles,
            call_type="qwen_action_loop",
        )
        decision = getattr(nav_planner, "last_decision", {}) or {}
        vlm_failure_reason = vlm_backend_failure_reason(decision)
        if vlm_failure_reason:
            consecutive_vlm_failures += 1
        else:
            consecutive_vlm_failures = 0
        action_counts["vlm_max_consecutive_failures"] = max(
            int(action_counts.get("vlm_max_consecutive_failures", 0)),
            int(consecutive_vlm_failures),
        )
        decision["runner_vlm_backend_health"] = {
            "failure": bool(vlm_failure_reason),
            "failure_reason": vlm_failure_reason or None,
            "consecutive_failures": int(consecutive_vlm_failures),
            "failure_limit": int(vlm_failure_limit),
        }
        qwen_log = getattr(nav_planner, "qwen_call_log", [])
        if qwen_log and isinstance(qwen_log[-1], dict):
            qwen_log[-1]["runner_vlm_backend_health"] = dict(
                decision["runner_vlm_backend_health"]
            )
        vlm_circuit_breaker_tripped = bool(
            vlm_failure_reason
            and consecutive_vlm_failures >= vlm_failure_limit
        )
        selected_view_id = _selected_view_id(decision, direction)
        qwen_action = action_from_decision(decision)
        action_counts[qwen_action] = int(action_counts.get(qwen_action, 0)) + 1
        if decision.get("verified_target_anchor_reacquisition"):
            action_counts["target_anchor_reacquisition"] = int(
                action_counts.get("target_anchor_reacquisition", 0)
            ) + 1
        runner_action, consecutive_rotates, forced_observation, override_reason = runner_action_with_context(
            qwen_action,
            consecutive_rotates,
            current_view_count=len(current_images),
            rotate_to_observe_after=rotate_to_observe_after,
            observation_mode=observation_mode_from_decision(decision),
        )
        if forced_observation:
            action_counts["forced_request_observation"] = int(action_counts.get("forced_request_observation", 0)) + 1
        attach_runner_action_audit(
            nav_planner,
            decision=decision,
            vlm_requested_action=qwen_action,
            runner_action=runner_action,
            override_reason=override_reason,
            observation_view_count=len(current_images),
        )
        if override_reason:
            action_counts["runner_override"] = int(action_counts.get("runner_override", 0)) + 1

        _record_qwen_audit(
            audit_logger=audit_logger,
            habitat_env=habitat_env,
            nav_planner=nav_planner,
            image=goal_image,
            frame_index=len(episode_images) - 1,
            call_type="qwen_action_loop",
            selected_view_id=selected_view_id,
        )
        _append_planning_frame(
            image=vis_rgb,
            decision=decision,
            episode_images=episode_images,
            decision_by_frame=decision_by_frame,
            planning_frame_indices=planning_frame_indices,
        )
        if vlm_circuit_breaker_tripped:
            benchmark_invalid_reason = (
                "vlm_backend_unavailable_after_{}_consecutive_failures:{}"
            ).format(vlm_failure_limit, vlm_failure_reason)
            action_counts["vlm_circuit_breaker_tripped"] = 1
            decision["benchmark_invalid_reason"] = benchmark_invalid_reason
            if qwen_log and isinstance(qwen_log[-1], dict):
                qwen_log[-1]["benchmark_invalid_reason"] = (
                    benchmark_invalid_reason
                )
            break

        if runner_action == "go":
            target_approach_mode = is_target_approach_decision(decision)
            go_step_limit = go_step_limit_for_decision(decision, max_go_steps)
            decision["runner_go_step_limit"] = int(go_step_limit)
            decision["runner_target_approach_limited"] = bool(
                target_approach_mode
            )
            target_approach_max_translation_m = target_approach_translation_cap(
                decision
            )
            target_memory_override = bool(
                isinstance(decision.get("vlm_output"), dict)
                and isinstance(
                    decision["vlm_output"].get("target_approach_memory_override"),
                    dict,
                )
            )
            decision["runner_target_approach_memory_override"] = (
                target_memory_override
            )
            if target_memory_override:
                action_counts["target_approach_memory_override"] = int(
                    action_counts.get("target_approach_memory_override", 0)
                ) + 1
            target_stop_refinement = bool(
                isinstance(decision.get("vlm_output"), dict)
                and isinstance(
                    decision["vlm_output"].get("target_terminal_refinement"),
                    dict,
                )
            )
            decision["runner_target_stop_refinement"] = target_stop_refinement
            if target_stop_refinement:
                action_counts["target_stop_refinement"] = int(
                    action_counts.get("target_stop_refinement", 0)
                ) + 1
            decision["runner_target_approach_max_translation_m"] = (
                target_approach_max_translation_m if target_approach_mode else None
            )
            if decision["runner_target_approach_limited"]:
                action_counts["target_approach_limited"] = int(
                    action_counts.get("target_approach_limited", 0)
                ) + 1
            selected_turn_yaw = _selected_angle(decision)
            obs, turn_truncated = _execute_turn(
                habitat_env=habitat_env,
                obs=obs,
                yaw_deg=selected_turn_yaw,
                stats=stats,
                episode_images=episode_images,
                episode_topdowns=episode_topdowns,
                topdown_fn=topdown_fn,
                max_episode_steps=max_episode_steps,
            )
            truncated_by_max_steps = truncated_by_max_steps or turn_truncated
            go_start_metrics = _metrics_snapshot(habitat_env)
            go_start_position = _agent_position_xyz(habitat_env)
            go_start_heading_rad = _agent_heading_rad(habitat_env)
            go_path_positions: List[List[float]] = [list(go_start_position)]
            policy_actions: List[int] = []
            collision_count = 0
            stop_action_seen = False
            target_translation_cap_reached = False
            nav_executor.reset(goal_image, goal_mask)
            for _ in range(go_step_limit):
                if habitat_env.episode_over:
                    break
                if _max_steps_reached(habitat_env, max_episode_steps):
                    truncated_by_max_steps = True
                    break
                policy_action, _skill_image = nav_executor.step(
                    obs["rgb"],
                    bool(getattr(habitat_env.sim, "previous_step_collided", False)),
                )
                if (
                    target_approach_mode
                    and target_translation_cap_reached
                    and int(policy_action) == FORWARD_ACTION
                ):
                    decision["runner_target_approach_blocked_probe_action"] = int(
                        policy_action
                    )
                    break
                policy_actions.append(int(policy_action))
                if int(policy_action) == STOP_ACTION:
                    stop_action_seen = True
                    if (
                        should_execute_target_terminal_micro_approach(
                            decision,
                            executed_count=target_terminal_micro_approach_count,
                        )
                        and not _max_steps_reached(habitat_env, max_episode_steps)
                    ):
                        micro_start = _agent_position_xyz(habitat_env)
                        obs = _step_env(habitat_env, FORWARD_ACTION, stats)
                        policy_actions.append(FORWARD_ACTION)
                        go_path_positions.append(_agent_position_xyz(habitat_env))
                        micro_collision = bool(
                            getattr(habitat_env.sim, "previous_step_collided", False)
                        )
                        if micro_collision:
                            collision_count += 1
                            action_counts["target_terminal_micro_approach_collision"] = int(
                                action_counts.get(
                                    "target_terminal_micro_approach_collision",
                                    0,
                                )
                            ) + 1
                        _append_observation_frame(
                            obs=obs,
                            habitat_env=habitat_env,
                            episode_images=episode_images,
                            episode_topdowns=episode_topdowns,
                            topdown_fn=topdown_fn,
                        )
                        target_terminal_micro_approach_count += 1
                        action_counts["target_terminal_micro_approach"] = int(
                            action_counts.get("target_terminal_micro_approach", 0)
                        ) + 1
                        micro_final = _agent_position_xyz(habitat_env)
                        decision["runner_target_terminal_micro_approach"] = {
                            "triggered": True,
                            "source_action": "pixelnav_stop",
                            "forced_action": "move_forward",
                            "execution_index": int(target_terminal_micro_approach_count),
                            "start_position_xyz": list(micro_start),
                            "final_position_xyz": list(micro_final),
                            "translation_m": round(
                                float(
                                    np.linalg.norm(
                                        np.asarray(micro_final, dtype=np.float64)
                                        - np.asarray(micro_start, dtype=np.float64)
                                    )
                                ),
                                6,
                            ),
                            "collision": micro_collision,
                            "reason": (
                                "identity-verified target approach had enough visual evidence but "
                                "PixelNav stopped without final translation"
                            ),
                        }
                        qwen_log = getattr(nav_planner, "qwen_call_log", None)
                        if (
                            isinstance(qwen_log, list)
                            and qwen_log
                            and isinstance(qwen_log[-1], dict)
                        ):
                            qwen_log[-1]["runner_target_terminal_micro_approach"] = dict(
                                decision["runner_target_terminal_micro_approach"]
                            )
                    break
                obs = _step_env(habitat_env, int(policy_action), stats)
                go_path_positions.append(_agent_position_xyz(habitat_env))
                if bool(getattr(habitat_env.sim, "previous_step_collided", False)):
                    collision_count += 1
                _append_observation_frame(
                    obs=obs,
                    habitat_env=habitat_env,
                    episode_images=episode_images,
                    episode_topdowns=episode_topdowns,
                    topdown_fn=topdown_fn,
                )
                if target_approach_mode:
                    current_position = np.asarray(
                        _agent_position_xyz(habitat_env),
                        dtype=np.float64,
                    )
                    target_translation = float(
                        np.linalg.norm(
                            current_position
                            - np.asarray(go_start_position, dtype=np.float64)
                        )
                    )
                    if target_translation >= target_approach_max_translation_m - 1e-4:
                        decision["runner_target_approach_translation_cap_reached"] = True
                        target_translation_cap_reached = True
            go_final_metrics = _metrics_snapshot(habitat_env)
            go_final_position = _agent_position_xyz(habitat_env)
            memory_sidecar = getattr(nav_planner, "memory_sidecar", None)
            supervisor_mode = effective_go_supervisor_mode(memory_sidecar, decision)
            strategic_evidence = build_strategic_progress_evidence(
                decision,
                memory_sidecar,
            )
            go_progress = evaluate_go_progress(
                go_start_metrics,
                go_final_metrics,
                start_position_xyz=go_start_position,
                final_position_xyz=go_final_position,
                collision_count=collision_count,
                controller_reached_waypoint=stop_action_seen,
                supervisor_mode=supervisor_mode,
                min_progress_m=min_go_progress_m,
                strategic_evidence=strategic_evidence,
            )
            if go_progress.get("collision_tolerant_progress"):
                action_counts["collision_tolerant_progress"] = int(
                    action_counts.get("collision_tolerant_progress", 0)
                ) + 1
            if hasattr(nav_planner, "record_strategic_go_execution"):
                strategic_execution = nav_planner.record_strategic_go_execution(
                    decision,
                    go_progress,
                )
                if isinstance(strategic_execution, dict):
                    decision["runner_strategic_go_execution"] = dict(
                        strategic_execution
                    )
                    qwen_log = getattr(nav_planner, "qwen_call_log", [])
                    if qwen_log and isinstance(qwen_log[-1], dict):
                        qwen_log[-1]["runner_strategic_go_execution"] = dict(
                            strategic_execution
                        )
            obs, pitch_reset, pitch_reset_truncated = _execute_pitch_reset(
                habitat_env=habitat_env,
                obs=obs,
                policy_actions=policy_actions,
                stats=stats,
                episode_images=episode_images,
                episode_topdowns=episode_topdowns,
                topdown_fn=topdown_fn,
                max_episode_steps=max_episode_steps,
            )
            action_counts["pitch_reset_action"] = int(action_counts.get("pitch_reset_action", 0)) + int(
                pitch_reset.get("pitch_reset_steps", 0)
            )
            truncated_by_max_steps = truncated_by_max_steps or pitch_reset_truncated
            go_final_heading_rad = _agent_heading_rad(habitat_env)
            pitch_reset["post_pitch_reset_metrics"] = _runner_json_safe(_metrics_snapshot(habitat_env))
            go_execution = build_go_execution_audit(
                decision=decision,
                start_metrics=go_start_metrics,
                final_metrics=go_final_metrics,
                start_position_xyz=go_start_position,
                final_position_xyz=go_final_position,
                policy_actions=policy_actions,
                collision_count=collision_count,
                stop_action_seen=stop_action_seen,
                max_go_steps=go_step_limit,
                turn_yaw_deg=selected_turn_yaw,
                turn_truncated=turn_truncated,
                go_progress=go_progress,
                pitch_reset=pitch_reset,
            )
            go_execution["executed_path_tangents"] = executed_path_tangent_summary(
                go_path_positions
            )
            target_approach_outcome: Dict[str, Any] = {}
            if target_approach_mode and hasattr(
                nav_planner,
                "record_target_approach_execution",
            ):
                recorded = nav_planner.record_target_approach_execution(
                    go_execution,
                    frame_index=len(episode_images) - 1,
                )
                if isinstance(recorded, dict):
                    target_approach_outcome = dict(recorded)
            if target_approach_outcome.get("detour_triggered"):
                action_counts["target_approach_detour_triggered"] = int(
                    action_counts.get("target_approach_detour_triggered", 0)
                ) + 1
                go_progress.update(
                    execution_success=False,
                    strategic_progress=False,
                    no_progress=True,
                    strategic_rule_id="PROGRESS_TARGET_APPROACH_002",
                    required_evidence=(
                        "target-directed PixelNav waypoint completion; translation alone is insufficient"
                    ),
                )
                go_execution["go_progress"] = _runner_json_safe(dict(go_progress))
                go_execution["target_approach_outcome"] = _runner_json_safe(
                    target_approach_outcome
                )
                go_execution["failure_classification"] = {
                    "primary": "target_approach_unresolved",
                    "labels": [
                        "no_progress",
                        "target_approach_unresolved",
                        "detour_required",
                    ],
                    "needs_replan": True,
                    "message": (
                        "repeated target-directed fine goals did not reach their PixelNav waypoint"
                    ),
                }
            sync_memory_sidecar_go_execution(
                nav_planner,
                go_execution=go_execution,
                frame_index=len(episode_images) - 1,
                start_position_xyz=go_start_position,
                final_position_xyz=go_final_position,
                start_heading_rad=go_start_heading_rad,
                final_heading_rad=go_final_heading_rad,
            )
            sync_memory_visualizer_to_decision(nav_planner, decision)
            attach_go_execution_audit(nav_planner, audit_logger, decision, go_execution)
            sync_go_progress_feedback(nav_planner, decision, go_progress)
            if go_progress["no_progress"] and not habitat_env.episode_over and not _max_steps_reached(habitat_env, max_episode_steps):
                action_counts["go_no_progress"] = int(action_counts.get("go_no_progress", 0)) + 1
                obs, current_images, current_angles, sweep_truncated = _collect_observation_sweep(
                    habitat_env=habitat_env,
                    obs=obs,
                    offsets=go_no_progress_observation_offsets(
                        getattr(memory_sidecar, "supervisor_mode", "goal_seek")
                    ),
                    stats=stats,
                    episode_images=episode_images,
                    episode_topdowns=episode_topdowns,
                    topdown_fn=topdown_fn,
                    max_episode_steps=max_episode_steps,
                )
                truncated_by_max_steps = truncated_by_max_steps or sweep_truncated
            else:
                current_images = [obs["rgb"]]
                current_angles = [0]
            continue

        if runner_action == "rotate":
            requested_rotate_yaw = rotate_yaw_from_decision(decision)
            runner_rotate_yaw = rotate_yaw_for_runner_action(
                decision,
                override_reason,
            )
            if abs(float(requested_rotate_yaw)) < 1e-6:
                action_counts["zero_yaw_rotate_correction"] = int(
                    action_counts.get("zero_yaw_rotate_correction", 0)
                ) + 1
                decision["runner_zero_yaw_rotate_corrected"] = True
                decision["runner_rotate_yaw_deg"] = float(runner_rotate_yaw)
                qwen_log = getattr(nav_planner, "qwen_call_log", [])
                if qwen_log and isinstance(qwen_log[-1], dict):
                    qwen_log[-1]["runner_zero_yaw_rotate_corrected"] = True
                    qwen_log[-1]["runner_rotate_yaw_deg"] = float(
                        runner_rotate_yaw
                    )
            obs, turn_truncated = _execute_turn(
                habitat_env=habitat_env,
                obs=obs,
                yaw_deg=runner_rotate_yaw,
                stats=stats,
                episode_images=episode_images,
                episode_topdowns=episode_topdowns,
                topdown_fn=topdown_fn,
                max_episode_steps=max_episode_steps,
            )
            truncated_by_max_steps = truncated_by_max_steps or turn_truncated
            current_images = [obs["rgb"]]
            current_angles = [0]
            continue

        if runner_action == "request_observation":
            offsets = rotate_stall_observation_offsets() if forced_observation else observation_offsets_from_decision(decision)
            obs, current_images, current_angles, sweep_truncated = _collect_observation_sweep(
                habitat_env=habitat_env,
                obs=obs,
                offsets=offsets,
                stats=stats,
                episode_images=episode_images,
                episode_topdowns=episode_topdowns,
                topdown_fn=topdown_fn,
                max_episode_steps=max_episode_steps,
            )
            truncated_by_max_steps = truncated_by_max_steps or sweep_truncated
            continue

        if runner_action == "stop":
            if not _max_steps_reached(habitat_env, max_episode_steps) and not habitat_env.episode_over:
                obs = _step_env(habitat_env, STOP_ACTION, stats)
                _append_observation_frame(
                    obs=obs,
                    habitat_env=habitat_env,
                    episode_images=episode_images,
                    episode_topdowns=episode_topdowns,
                    topdown_fn=topdown_fn,
                )
            break

    final_loop_metrics = _metrics_snapshot(habitat_env)
    if (
        _max_steps_reached(habitat_env, max_episode_steps)
        and not bool(final_loop_metrics.get("success"))
    ):
        truncated_by_max_steps = True
    episode_time_sec = time.perf_counter() - episode_t0

    active_decision = None
    for idx, image in enumerate(episode_images):
        if idx in decision_by_frame:
            active_decision = decision_by_frame[idx]
        fps_writer.append_data(
            render_qwen_debug_frame(
                image,
                decision=active_decision,
                target=habitat_env.current_episode.object_category,
                planner_name=getattr(nav_planner, "planner_name", "qwen_vlm"),
                draw_point=idx in planning_frame_indices,
            )
        )
    for topdown in episode_topdowns:
        topdown_writer.append_data(topdown)
    fps_writer.close()
    topdown_writer.close()

    if hasattr(nav_planner, "save_qwen_calls"):
        nav_planner.save_qwen_calls(os.path.join(trajectory_dir, "qwen_calls.jsonl"))

    audit_logger.finalize_pending(
        position_xyz=_agent_position_xyz(habitat_env),
        metrics=_metrics_snapshot(habitat_env),
        collision=bool(getattr(habitat_env.sim, "previous_step_collided", False)),
    )
    memory_dir = os.path.join(trajectory_dir, "memory")
    audit_paths = audit_logger.save_run(memory_dir)
    audit_paths = save_memory_sidecar_artifacts(nav_planner, audit_paths, memory_dir)

    metrics_row = _build_metrics_row(
        habitat_env=habitat_env,
        nav_planner=nav_planner,
        episode_index=episode_index,
        start_geodesic_m=start_geodesic_m,
        episode_time_sec=episode_time_sec,
        truncated_by_max_steps=truncated_by_max_steps,
        audit_paths=audit_paths,
        stats=stats,
        action_counts=action_counts,
        configured_max_episode_steps=max_episode_steps,
        benchmark_invalid_reason=benchmark_invalid_reason,
    )
    manifest_path = save_benchmark_manifest(
        trajectory_dir=trajectory_dir,
        episode_index=episode_index,
        nav_planner=nav_planner,
        audit_paths=audit_paths,
        action_counts=action_counts,
        metrics=metrics_row,
        episode_descriptor={
            "episode_id": str(getattr(habitat_env.current_episode, "episode_id", "")),
            "scene_id": str(getattr(habitat_env.current_episode, "scene_id", "")),
            "object_category": str(
                getattr(habitat_env.current_episode, "object_category", "")
            ),
        },
    )
    metrics_row["benchmark_manifest"] = manifest_path
    metrics_row["memory_schema"] = (
        getattr(nav_planner, "memory_sidecar", None).visualizer_state().get("schema_version", "")
        if getattr(nav_planner, "memory_sidecar", None) is not None
        else "null_memory_context_v0"
    )
    return metrics_row
