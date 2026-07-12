import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from voca_s2e_bridge import import_nav_memory_qwen


NAV_SCHEMA_VERSION = "nav_vlm_waypoint_v1"
NULL_MEMORY_CONTEXT_VERSION = "null_memory_context_v0"
NULL_MEMORY_GRAPH_VERSION = "null_memory_graph_v0"

_NAV_MEMORY = import_nav_memory_qwen()
_schema = _NAV_MEMORY.schema
_safety = _NAV_MEMORY.safety
SCHEMA_SOURCE_FILE = str(_schema.__file__)
SAFETY_SOURCE_FILE = str(_safety.__file__)


def _image_hw(image_shape: Sequence[int]) -> Tuple[int, int]:
    h = int(image_shape[0]) if len(image_shape) > 0 else 480
    w = int(image_shape[1]) if len(image_shape) > 1 else 640
    return max(1, h), max(1, w)


def _normalize_confidence(value: Any) -> str:
    text = str(value or "medium").strip().lower()
    if text == "fallback":
        return "low"
    return text if text in {"high", "medium", "low"} else "medium"


def _view_type_from_angle(angle: Any) -> str:
    try:
        angle_i = int(angle)
    except Exception:
        angle_i = 0
    return _schema.nearest_view_type(angle_i)


def _decision_angle(decision: Dict[str, Any]) -> int:
    try:
        return int(decision.get("Angle", 0) or 0)
    except Exception:
        return 0


def _decision_point(decision: Dict[str, Any]) -> Any:
    return decision.get("Point") or decision.get("point") or decision.get("selected_image_point")


def _angles_for_selected_view(angle: int, selected_view_id: Optional[int], angles: Optional[Iterable[int]]) -> List[int]:
    if angles is not None:
        values = [int(a) for a in angles]
        if values:
            return values
    view_id = max(0, int(selected_view_id or 0))
    values = [0 for _ in range(view_id + 1)]
    values[view_id] = int(angle)
    return values


def qwen_point_decision_to_vlm_output(
    decision: Dict[str, Any],
    *,
    image_shape: Sequence[int],
    selected_view_id: Optional[int] = None,
    angles: Optional[Iterable[int]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Map VOCA's legacy Qwen point JSON into nav_vlm_waypoint_v1 output."""
    h, w = _image_hw(image_shape)
    direct_output = decision.get("vlm_output")
    if isinstance(direct_output, dict):
        vlm_input = decision.get("vlm_input")
        if not isinstance(vlm_input, dict):
            angle_for_input = _decision_angle(decision)
            vlm_input = build_memory_free_vlm_input(
                target_object=str(decision.get("target", "")),
                image_shape=image_shape,
                frame_index=0,
                priors=decision.get("priors") if isinstance(decision.get("priors"), dict) else None,
                angles=_angles_for_selected_view(angle_for_input, selected_view_id, angles),
                call_type=str(decision.get("call_type", "qwen_direct_decision")),
                planner_name=str(decision.get("planner_name", "qwen_vlm")),
            )
        safe_output, warnings = _safety.sanitize_vlm_output(direct_output, vlm_input)
        if decision.get("fallback"):
            reason = str(decision.get("fallback_reason") or "qwen_vlm_fallback")
            warnings.append("qwen_vlm_fallback:{}".format(reason))
        return safe_output, warnings

    angle = _decision_angle(decision)
    view_id = int(selected_view_id) if selected_view_id is not None else 0
    view_type = _view_type_from_angle(angle)
    point = _decision_point(decision)
    short_text = str(decision.get("Reason") or "").strip() or "qwen selected navigable point"
    confidence = _normalize_confidence(decision.get("Confidence", "medium"))

    try:
        point_px = (int(point[0]), int(point[1]))
    except Exception:
        point_px = (-1, -1)

    raw_output = _schema.make_go_output(
        view_id=view_id,
        view_type=view_type,
        point_px=point_px,
        width=w,
        height=h,
        decision_reason="G02_VISIBLE_FLOOR_TOWARD_GOAL",
        goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
        short_text=short_text,
        confidence=confidence,
    )
    vlm_input = build_memory_free_vlm_input(
        target_object=str(decision.get("target", "")),
        image_shape=image_shape,
        frame_index=0,
        priors=decision.get("priors") if isinstance(decision.get("priors"), dict) else None,
        angles=_angles_for_selected_view(angle, selected_view_id, angles),
        call_type=str(decision.get("call_type", "qwen_point_decision")),
        planner_name=str(decision.get("planner_name", "qwen_point")),
    )
    safe_output, warnings = _safety.sanitize_vlm_output(raw_output, vlm_input)
    if decision.get("fallback"):
        reason = str(decision.get("fallback_reason") or "qwen_point_fallback")
        warnings.append("qwen_point_fallback:{}".format(reason))
    return safe_output, warnings


def build_memory_free_vlm_input(
    *,
    target_object: str,
    image_shape: Sequence[int],
    frame_index: int,
    priors: Optional[Dict[str, Any]] = None,
    angles: Optional[Iterable[int]] = None,
    call_type: str = "make_plan",
    planner_name: str = "qwen_point",
    position_xyz: Optional[Sequence[float]] = None,
    heading_rad: float = 0.0,
    ) -> Dict[str, Any]:
    h, w = _image_hw(image_shape)
    angle_list = [0] if angles is None else [int(a) for a in angles]
    pos = _float_list(position_xyz or [0.0, 0.0, 0.0])
    robot_state = _schema.RobotState(
        map_xy=(float(pos[0]), float(pos[2]) if len(pos) > 2 else 0.0),
        heading_rad=float(heading_rad),
        position_xyz=(float(pos[0]), float(pos[1]) if len(pos) > 1 else 0.0, float(pos[2]) if len(pos) > 2 else 0.0),
    )
    task = _schema.CoarseGoal.from_object_goal(str(target_object or ""), robot_state)
    views = [
        _schema.ObservationView(
            view_id=idx,
            view_type=_view_type_from_angle(angle),
            relative_heading_deg=float(angle),
            image="<rgb_frame_{}>".format(idx),
        )
        for idx, angle in enumerate(angle_list)
    ]
    observation = _schema.Observation(
        mode="current_or_panoramic",
        sequence_id="voca_objnav_memory_free",
        frame_index=int(frame_index),
        image_width=w,
        image_height=h,
        views=views,
    )
    memory = {
        "schema_version": NULL_MEMORY_CONTEXT_VERSION,
        "enabled": False,
        "graph_summary": {
            "num_nodes": 0,
            "num_edges": 0,
            "num_deadlock_edges": 0,
        },
        "runtime_state": {
            "memory_disabled_reason": "memory integration intentionally deferred",
        },
    }
    vlm_input = _schema.build_vlm_input_v1(
        task=task,
        robot_state=robot_state,
        observation=observation,
        memory=memory,
        pose_noise_enabled=False,
    )
    vlm_input["task"]["target_context"] = dict(priors or {})
    vlm_input["metadata"] = {
        "planner_name": planner_name,
        "call_type": call_type,
        "angles": angle_list,
        "schema_source": SCHEMA_SOURCE_FILE,
        "safety_source": SAFETY_SOURCE_FILE,
    }
    return vlm_input


def _float_list(values: Sequence[float]) -> List[float]:
    return [float(v) for v in values]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "shape"):
        try:
            return "<{} shape={}>".format(type(value).__name__, list(value.shape))
        except Exception:
            return "<{}>".format(type(value).__name__)
    return "<{}>".format(type(value).__name__)


def build_action_outcome_from_positions(
    *,
    action: str = "go",
    start_position_xyz: Sequence[float],
    final_position_xyz: Sequence[float],
    collision: bool = False,
    progress_threshold_m: float = 0.05,
    raw: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    start = _float_list(start_position_xyz)
    final = _float_list(final_position_xyz)
    moved = math.sqrt(sum((final[i] - start[i]) ** 2 for i in range(min(len(start), len(final), 3))))
    no_progress = moved < float(progress_threshold_m)
    action_name = str(action or "go")
    requires_translation = action_name == "go"
    success = (not bool(collision)) and ((not requires_translation) or (not no_progress))
    payload = {
        "action": action_name,
        "success": bool(success),
        "collision": bool(collision),
        "no_progress": bool(no_progress),
        "moved_distance_m": round(float(moved), 6),
        "rotated_deg": 0.0,
        "message": "moved {:.3f}m between VLM decisions".format(float(moved)),
        "raw": dict(raw or {}),
    }
    payload["raw"].setdefault("start_position_xyz", start)
    payload["raw"].setdefault("final_position_xyz", final)
    return payload


class NavigationAuditLogger:
    """Memory-free VOCA audit logger with a future NavMemoryAgent-compatible shape."""

    def __init__(self, *, progress_threshold_m: float = 0.05):
        self.progress_threshold_m = float(progress_threshold_m)
        self.steps: List[Dict[str, Any]] = []
        self._pending_index: Optional[int] = None
        self._pending_position: Optional[List[float]] = None
        self._pending_metrics: Dict[str, Any] = {}
        self._pending_action: str = "go"

    def record_decision(
        self,
        *,
        decision: Dict[str, Any],
        target_object: str,
        image_shape: Sequence[int],
        frame_index: int,
        position_xyz: Sequence[float],
        metrics: Optional[Dict[str, Any]] = None,
        priors: Optional[Dict[str, Any]] = None,
        angles: Optional[Iterable[int]] = None,
        call_type: str = "make_plan",
        selected_view_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        current_position = _float_list(position_xyz)
        self.finalize_pending(position_xyz=current_position, metrics=metrics or {})

        vlm_input = decision.get("vlm_input")
        if not isinstance(vlm_input, dict):
            vlm_input = build_memory_free_vlm_input(
                target_object=target_object,
                image_shape=image_shape,
                frame_index=frame_index,
                priors=priors,
                angles=angles,
                call_type=call_type,
                position_xyz=current_position,
            )
        vlm_output, warnings = qwen_point_decision_to_vlm_output(
            decision,
            image_shape=image_shape,
            selected_view_id=selected_view_id,
            angles=angles,
        )
        step = {
            "step_index": len(self.steps),
            "action": vlm_output.get("action"),
            "done": False,
            "success": False,
            "warnings": warnings,
            "current_node_id": None,
            "goal_distance_m": _metric_float(metrics, "distance_to_goal"),
            "vlm_input": vlm_input,
            "vlm_output": vlm_output,
            "outcome": None,
            "error": None,
            "qwen_point_decision": dict(decision),
            "runtime": {
                "frame_index": int(frame_index),
                "call_type": call_type,
                "start_position_xyz": current_position,
                "start_metrics": _json_safe(dict(metrics or {})),
            },
        }
        self.steps.append(step)
        self._pending_index = step["step_index"]
        self._pending_position = current_position
        self._pending_metrics = dict(metrics or {})
        self._pending_action = str(vlm_output.get("action") or "go")
        return step

    def finalize_pending(
        self,
        *,
        position_xyz: Sequence[float],
        metrics: Optional[Dict[str, Any]] = None,
        collision: bool = False,
    ) -> None:
        if self._pending_index is None or self._pending_position is None:
            return
        final_metrics = dict(metrics or {})
        outcome = build_action_outcome_from_positions(
            action=self._pending_action,
            start_position_xyz=self._pending_position,
            final_position_xyz=position_xyz,
            collision=collision,
            progress_threshold_m=self.progress_threshold_m,
            raw={
                "start_metrics": _json_safe(self._pending_metrics),
                "final_metrics": _json_safe(final_metrics),
            },
        )
        step = self.steps[self._pending_index]
        step["outcome"] = outcome
        step["success"] = bool(outcome.get("success"))
        step["done"] = bool(final_metrics.get("success", 0.0))
        step["runtime"]["final_position_xyz"] = _float_list(position_xyz)
        step["runtime"]["final_metrics"] = _json_safe(final_metrics)
        self._pending_index = None
        self._pending_position = None
        self._pending_metrics = {}
        self._pending_action = "go"

    def annotate_pending_runtime(self, key: str, value: Any) -> None:
        if self._pending_index is None:
            return
        self.steps[self._pending_index].setdefault("runtime", {})[str(key)] = _json_safe(value)

    def save_run(self, out_dir: Any) -> Dict[str, str]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        graph_path = out / "memory_graph.json"
        steps_path = out / "steps.json"
        memory_graph = {
            "schema_version": NULL_MEMORY_GRAPH_VERSION,
            "memory_enabled": False,
            "nodes": [],
            "edges": [],
            "message": "Memory graph is intentionally disabled for this VOCA audit run.",
        }
        graph_path.write_text(json.dumps(memory_graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        steps_path.write_text(json.dumps(_json_safe({"steps": self.steps}), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {
            "memory_graph_json": str(graph_path),
            "steps_json": str(steps_path),
        }


def _metric_float(metrics: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not metrics or key not in metrics:
        return None
    try:
        return float(metrics[key])
    except Exception:
        return None
