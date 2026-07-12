import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from navigation_supervisor import (
    assert_policy_input_safe,
    build_policy_goal_context,
    normalize_supervisor_mode,
)
from place_recognition import build_place_embedder
from voca_s2e_bridge import import_memory_graph_visualizer, import_nav_memory_qwen


_NAV_MEMORY = import_nav_memory_qwen()
_MEMORY_VISUALIZER = import_memory_graph_visualizer()


def _position_delta_m(start: Sequence[float], final: Sequence[float]) -> float:
    try:
        n = min(len(start), len(final), 3)
        return float(math.sqrt(sum((float(final[i]) - float(start[i])) ** 2 for i in range(n))))
    except Exception:
        return 0.0


def _normalize_angle_deg(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


def _finite_float_or_none(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def _valid_position_xyz(position_xyz: Optional[Sequence[float]]) -> bool:
    return position_xyz is not None and len(position_xyz) >= 3


def _relative_pose_from_world_states(
    anchor_position_xyz: Sequence[float],
    anchor_heading_rad: float,
    current_position_xyz: Sequence[float],
    current_heading_rad: float,
) -> Any:
    """Return anchor-to-current SE(2) using Habitat's world X/Z plane."""
    dx_world = float(current_position_xyz[0]) - float(anchor_position_xyz[0])
    dz_world = float(current_position_xyz[2]) - float(anchor_position_xyz[2])
    heading = float(anchor_heading_rad)
    c = math.cos(heading)
    s = math.sin(heading)
    dx_local = c * dx_world + s * dz_world
    dy_local = -s * dx_world + c * dz_world
    dyaw_deg = math.degrees(float(current_heading_rad) - heading)
    dyaw_deg = ((dyaw_deg + 180.0) % 360.0) - 180.0
    return _NAV_MEMORY.schema.RelativePose2D(
        dx_m=dx_local,
        dy_m=dy_local,
        dyaw_deg=dyaw_deg,
    )


def _relative_polyline_xy(
    position_trace_xyz: Sequence[Sequence[float]],
    *,
    anchor_position_xyz: Sequence[float],
    anchor_heading_rad: float,
    max_points: int = 24,
) -> List[List[float]]:
    points: List[List[float]] = []
    for position in position_trace_xyz:
        if not _valid_position_xyz(position):
            continue
        pose = _relative_pose_from_world_states(
            anchor_position_xyz,
            anchor_heading_rad,
            position,
            anchor_heading_rad,
        )
        point = [float(pose.dx_m), float(pose.dy_m)]
        if points and math.hypot(
            point[0] - points[-1][0],
            point[1] - points[-1][1],
        ) < 0.01:
            continue
        points.append(point)
    limit = max(2, int(max_points))
    if len(points) > limit:
        indices = np.linspace(0, len(points) - 1, num=limit, dtype=int)
        points = [points[int(index)] for index in indices]
    return [
        [round(float(point[0]), 4), round(float(point[1]), 4)]
        for point in points
    ]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
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


class VOCAMemorySidecar:
    """v6 MemoryGraph adapter that keeps VOCA's existing runner loop intact."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        force_new_node_translation_m: float = 0.75,
        progress_threshold_m: float = 0.05,
        max_memory_images: int = 4,
        failed_frontier_distance_m: float = 0.75,
        embedder: Optional[Any] = None,
    ):
        self.enabled = bool(enabled)
        self.force_new_node_translation_m = float(force_new_node_translation_m)
        self.progress_threshold_m = float(progress_threshold_m)
        self.max_memory_images = int(max_memory_images)
        self.failed_frontier_distance_m = float(failed_frontier_distance_m)
        self.place_revisit_threshold = float(
            os.environ.get("VOCA_PLACE_REVISIT_THRESHOLD", "0.84")
        )
        self.embedder = embedder if embedder is not None else build_place_embedder(_NAV_MEMORY)
        embedding_dim = int(getattr(self.embedder, "dim", 256) or 256)
        self.memory = _NAV_MEMORY.memory_graph.MemoryGraph(embedding_dim=embedding_dim)
        self.target_object = ""
        self.no_progress_count = 0
        self.directional_failures: List[Dict[str, Any]] = []
        self.supervisor_mode = "goal_seek"
        self.last_mode_transition: Dict[str, Any] = {
            "from": None,
            "to": "goal_seek",
            "reason": "initialized",
        }
        self.last_context: Dict[str, Any] = {}
        self.last_image_ref: Optional[str] = None
        self.last_frame_index = 0
        self.current_node_anchor_position_xyz: Optional[List[float]] = None
        self.current_node_anchor_heading_rad: Optional[float] = None
        self.last_embedded_frame_index: Optional[int] = None
        self.last_memory_op_results: List[Dict[str, Any]] = []
        self.last_strategic_state: Dict[str, Any] = {}
        self.last_semantic_update: Dict[str, Any] = {}
        self.last_verified_target_evidence: Dict[str, Any] = {}
        self.runtime_stats: Dict[str, int] = self._new_runtime_stats()

    @staticmethod
    def _new_runtime_stats() -> Dict[str, int]:
        return {
            "embedding_observations": 0,
            "embedding_views": 0,
            "revisit_queries": 0,
            "revisit_candidate_contexts": 0,
            "revisit_candidates_offered": 0,
            "memory_ops_requested": 0,
            "memory_ops_accepted": 0,
            "memory_ops_rejected": 0,
            "revisits_confirmed": 0,
            "revisits_rejected": 0,
            "soft_merges": 0,
            "object_belief_updates": 0,
            "semantic_state_updates": 0,
            "room_category_updates": 0,
            "room_transitions": 0,
        }

    def reset(self, target_object: str = "") -> None:
        embedding_dim = int(getattr(self.embedder, "dim", 256) or 256)
        self.memory = _NAV_MEMORY.memory_graph.MemoryGraph(embedding_dim=embedding_dim)
        self.target_object = str(target_object or "")
        self.no_progress_count = 0
        self.directional_failures = []
        self.supervisor_mode = "goal_seek"
        self.last_mode_transition = {
            "from": None,
            "to": "goal_seek",
            "reason": "episode_reset",
        }
        self.last_context = {}
        self.last_image_ref = None
        self.last_frame_index = 0
        self.current_node_anchor_position_xyz = None
        self.current_node_anchor_heading_rad = None
        self.last_embedded_frame_index = None
        self.last_memory_op_results = []
        self.last_strategic_state = {}
        self.last_semantic_update = {}
        self.last_verified_target_evidence = {}
        self.runtime_stats = self._new_runtime_stats()

    def _embedding_status(self) -> Dict[str, Any]:
        status_fn = getattr(self.embedder, "status", None)
        if callable(status_fn):
            status = dict(status_fn())
            status["revisit_threshold"] = round(self.place_revisit_threshold, 6)
            return status
        return {
            "backend": type(self.embedder).__name__ if self.embedder is not None else "disabled",
            "dimension": int(getattr(self.embedder, "dim", 0) or 0),
            "load_error": None,
            "revisit_threshold": round(self.place_revisit_threshold, 6),
        }

    def _embed_image(self, image_ref: Optional[str]) -> Any:
        if self.embedder is None or not image_ref or not Path(image_ref).exists():
            return None
        try:
            return self.embedder.embed_image(image_ref)
        except Exception as exc:
            log_fn = getattr(self.memory, "_log", None)
            if callable(log_fn):
                log_fn(
                    "place_embedding_error",
                    image_ref=str(image_ref),
                    error="{}: {}".format(type(exc).__name__, str(exc))[:240],
                )
            return None

    def _embed_image_descriptors(self, image_refs: Sequence[str]) -> List[Any]:
        return [
            descriptor
            for descriptor in (self._embed_image(path) for path in image_refs)
            if descriptor is not None
        ]

    @staticmethod
    def _aggregate_descriptors(descriptors: Sequence[Any]) -> Any:
        if not descriptors:
            return None
        try:
            descriptor = np.mean(np.stack(descriptors, axis=0), axis=0).astype(np.float32)
            norm = float(np.linalg.norm(descriptor))
            return descriptor if norm <= 1e-8 else descriptor / norm
        except Exception:
            return descriptors[0]

    def _update_multiview_localization(self, descriptors: Sequence[Any]) -> None:
        if not descriptors or len(self.memory.visual_index) == 0:
            return
        self.runtime_stats["revisit_queries"] += 1
        current_node_id = self.memory.current_node_id
        candidate_scores: Dict[str, float] = {}
        candidate_keyframes: Dict[str, List[str]] = {}
        for descriptor in descriptors:
            for hit in self.memory.visual_index.search(descriptor, top_k=8, min_score=-1.0):
                node_id = str(hit.metadata.get("node_id") or "")
                if not node_id or node_id == current_node_id or node_id not in self.memory.nodes:
                    continue
                score = float(hit.score)
                candidate_scores[node_id] = max(score, candidate_scores.get(node_id, -1.0))
                candidate_keyframes.setdefault(node_id, []).append(str(hit.key))
        ordered = sorted(candidate_scores, key=lambda node_id: candidate_scores[node_id], reverse=True)
        ordered = [
            node_id
            for node_id in ordered
            if candidate_scores[node_id] >= self.place_revisit_threshold
        ]
        matched_keyframes = [
            keyframe
            for node_id in ordered
            for keyframe in candidate_keyframes.get(node_id, [])
        ]
        best_score = candidate_scores.get(ordered[0], 0.0) if ordered else 0.0
        self.memory.last_localization = _NAV_MEMORY.memory_graph.LocalizationResult(
            match_status="uncertain" if ordered else "new_place",
            current_node_id=None,
            match_confidence=max(0.0, float(best_score)),
            candidate_node_ids=ordered,
            matched_keyframe_ids=matched_keyframes,
            candidate_scores={node_id: candidate_scores[node_id] for node_id in ordered},
            candidate_keyframe_ids={node_id: candidate_keyframes[node_id] for node_id in ordered},
            backend_commit_allowed=False,
        )

    def update_supervisor_mode(self, mode: str, reason: str = "external_update") -> str:
        normalized = normalize_supervisor_mode(mode)
        previous = self.supervisor_mode
        self.supervisor_mode = normalized
        if previous != normalized:
            self.last_mode_transition = {
                "from": previous,
                "to": normalized,
                "reason": str(reason or "unspecified"),
            }
            log_fn = getattr(self.memory, "_log", None)
            if callable(log_fn):
                log_fn(
                    "supervisor_mode_transition",
                    previous_mode=previous,
                    next_mode=normalized,
                    reason=str(reason or "unspecified"),
                )
        return self.supervisor_mode

    def _ensure_current_node(
        self,
        *,
        image_ref: Optional[str],
        frame_index: int,
        position_xyz: Optional[Sequence[float]] = None,
        heading_rad: Optional[float] = None,
        embedding: Any = None,
    ) -> str:
        if self.memory.current_node_id:
            if self.current_node_anchor_position_xyz is None and _valid_position_xyz(position_xyz):
                self.current_node_anchor_position_xyz = [float(value) for value in position_xyz]
                self.current_node_anchor_heading_rad = float(heading_rad or 0.0)
            return str(self.memory.current_node_id)
        node_id = self.memory.add_node(
            frame_index=int(frame_index),
            image_ref=image_ref,
            embedding=embedding,
            view_type="front",
            relative_heading_deg=0.0,
            place_category="unknown",
            semantic_summary="VOCA observed place at frame {}".format(int(frame_index)),
        )
        if _valid_position_xyz(position_xyz):
            self.current_node_anchor_position_xyz = [float(value) for value in position_xyz]
            self.current_node_anchor_heading_rad = float(heading_rad or 0.0)
        return node_id

    def _live_pose(
        self,
        position_xyz: Optional[Sequence[float]],
        heading_rad: Optional[float],
    ) -> Any:
        if not _valid_position_xyz(position_xyz) or not _valid_position_xyz(self.current_node_anchor_position_xyz):
            return _NAV_MEMORY.schema.RelativePose2D()
        return _relative_pose_from_world_states(
            self.current_node_anchor_position_xyz,
            float(self.current_node_anchor_heading_rad or 0.0),
            position_xyz,
            float(heading_rad or 0.0),
        )

    def _policy_harness_state(self, supervisor_mode: str = "goal_seek") -> Dict[str, Any]:
        mode = normalize_supervisor_mode(supervisor_mode)
        stage = {
            "goal_seek": "normal_goal_seek",
            "escape_deadlock": "escape_deadlock",
            "backtrack": "escape_deadlock",
            "verify_target": "verify_target",
        }[mode]
        state = _NAV_MEMORY.policy.build_policy_harness_state(
            current_stage=stage,
            last_validation_feedback=None,
            require_candidate_ref_for_go=True,
            require_candidate_ref_for_revisit_ops=True,
        )
        state.setdefault("requirements", {})
        state["requirements"]["go_requires_selected_candidate_ref"] = True
        state["requirements"]["revisit_memory_ops_require_candidate_ref"] = True
        state["voca_sidecar_policy"] = "strict_verified_candidate_refs"
        return state

    def build_context(
        self,
        *,
        target_object: str,
        priors: Optional[Dict[str, Any]],
        image_ref: Optional[str],
        image_refs: Optional[Sequence[str]] = None,
        frame_index: int,
        image_shape: Sequence[int],
        position_xyz: Optional[Sequence[float]] = None,
        heading_rad: Optional[float] = None,
        supervisor_mode: Optional[str] = None,
        spatial_goal: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "schema_version": "null_memory_context_v0",
                "enabled": False,
                "runtime_state": {"memory_disabled_reason": "VOCA memory sidecar disabled"},
            }
        self.target_object = str(target_object or self.target_object)
        self.last_image_ref = image_ref
        self.last_frame_index = int(frame_index)
        current_image_refs = [str(path) for path in (image_refs or []) if path]
        if not current_image_refs and image_ref:
            current_image_refs = [str(image_ref)]
        descriptors = self._embed_image_descriptors(current_image_refs)
        if descriptors:
            self.runtime_stats["embedding_observations"] += 1
            self.runtime_stats["embedding_views"] += len(descriptors)
        embedding = self._aggregate_descriptors(descriptors)
        node_already_existed = bool(self.memory.current_node_id)
        if node_already_existed:
            self._update_multiview_localization(descriptors)
        current_node_id = self._ensure_current_node(
            image_ref=image_ref,
            frame_index=frame_index,
            position_xyz=position_xyz,
            heading_rad=heading_rad,
            embedding=embedding,
        )
        if (
            embedding is not None
            and self.last_embedded_frame_index != int(frame_index)
            and self.memory.current_node_id
        ):
            node = self.memory.nodes.get(str(self.memory.current_node_id))
            already_indexed = self.last_embedded_frame_index == int(frame_index)
            if not already_indexed and node:
                descriptors_to_add = (
                    descriptors
                    if node_already_existed or len(descriptors) > 1
                    else []
                )
                for index, descriptor in enumerate(descriptors_to_add):
                    path = current_image_refs[index] if index < len(current_image_refs) else image_ref
                    self.memory.add_keyframe(
                        node_id=str(self.memory.current_node_id),
                        frame_index=int(frame_index),
                        image_ref=path,
                        embedding=descriptor,
                        view_type="front",
                        relative_heading_deg=0.0,
                        caption="VOCA multi-view place descriptor at frame {} view {}".format(
                            int(frame_index), index
                        ),
                    )
                if not node_already_existed:
                    descriptor = None
                elif not descriptors:
                    descriptor = embedding
                elif len(descriptors) == 1:
                    descriptor = None
                else:
                    descriptor = embedding
                if descriptor is not None:
                    self.memory.add_keyframe(
                        node_id=str(self.memory.current_node_id),
                        frame_index=int(frame_index),
                        image_ref=image_ref,
                        embedding=descriptor,
                        view_type="front",
                        relative_heading_deg=0.0,
                        caption="VOCA aggregate place descriptor at frame {}".format(int(frame_index)),
                    )
            self.last_embedded_frame_index = int(frame_index)
        self.memory.update_current_pose_relation_to_latest_node(
            current_node_id,
            self._live_pose(position_xyz, heading_rad),
            frame_index=int(frame_index),
            source="voca_runtime_pose",
        )
        if supervisor_mode is None:
            mode = self.supervisor_mode
        else:
            mode = normalize_supervisor_mode(supervisor_mode)
            if mode != self.supervisor_mode:
                self.update_supervisor_mode(mode, reason="context_override")
        context = self.memory.build_vlm_memory_context(
            goal_bearing_deg=0.0,
            goal_distance_m=0.0,
            task_mode="ObjNav",
            target_object=self.target_object,
            max_memory_images=self.max_memory_images,
            force_front_view_waypoint=False,
        )
        revisit_candidates = context.get("candidate_refs", {}).get("revisits", [])
        if isinstance(revisit_candidates, list) and revisit_candidates:
            self.runtime_stats["revisit_candidate_contexts"] += 1
            self.runtime_stats["revisit_candidates_offered"] += len(revisit_candidates)
        context["goal_context"] = build_policy_goal_context(
            task_mode="ObjNav",
            target_object=self.target_object,
            supervisor_mode=mode,
            spatial_goal=spatial_goal,
        )
        context["policy_harness_state"] = self._policy_harness_state(mode)
        context.setdefault("control_policy", {})["candidate_ref_policy"] = (
            "pixel and revisit candidate refs are mandatory and backend verified"
        )
        context["voca_sidecar"] = {
            "enabled": True,
            "phase": "closed_loop_v6",
            "target_context_priors": dict(priors or {}),
            "position_xyz": list(position_xyz or []),
            "heading_rad": float(heading_rad or 0.0),
            "image_shape": list(image_shape),
            "no_progress_count": int(self.no_progress_count),
            "supervisor_mode": mode,
            "directional_failures": [
                dict(item) for item in self.directional_failures[-4:]
            ],
            "place_embedding": self._embedding_status(),
            "place_embedding_view_count": len(current_image_refs),
            "last_memory_op_results": [dict(item) for item in self.last_memory_op_results[-4:]],
            "last_strategic_state": dict(self.last_strategic_state),
            "last_semantic_update": dict(self.last_semantic_update),
            "last_verified_target_evidence": dict(
                self.last_verified_target_evidence
            ),
        }
        assert_policy_input_safe(context)
        self.last_context = context
        return context

    def record_semantic_state(
        self,
        strategic_state: Dict[str, Any],
        *,
        frame_index: int,
    ) -> Dict[str, Any]:
        """Attach a VLM room hypothesis to the current place node as unverified semantics."""
        if not self.enabled:
            return {"updated": False, "reason": "memory_disabled"}
        state = strategic_state if isinstance(strategic_state, dict) else {}
        node_id = str(self.memory.current_node_id or "")
        node = self.memory.nodes.get(node_id) if node_id else None
        if node is None:
            result = {"updated": False, "reason": "no_current_place_node"}
            self.last_semantic_update = result
            return result

        room = str(state.get("current_room") or "unknown").strip().lower()
        evidence = str(state.get("target_evidence") or "none").strip().lower()
        intent = str(
            state.get("navigation_intent") or "search_current_room"
        ).strip().lower()
        search_status = str(
            state.get("room_search_status") or "unsearched"
        ).strip().lower()
        previous_room = str(node.place_category or "unknown")
        room_category_updated = bool(room != "unknown" and room != previous_room)
        room_transition = bool(
            room_category_updated and previous_room not in {"", "unknown"}
        )
        if room != "unknown":
            node.place_category = room

        normalized_state = {
            "current_room": room,
            "target_evidence": evidence,
            "navigation_intent": intent,
            "room_search_status": search_status,
            "selected_exit_ref": state.get("selected_exit_ref"),
            "reason_code": str(state.get("reason_code") or "")[:96],
            "frame_index": int(frame_index),
            "source": "qwen_vlm_unverified_semantic_hypothesis",
        }
        node.semantic["strategic_state"] = dict(normalized_state)
        node.semantic["room_search_status"] = search_status
        existing_target = (
            node.semantic.get("target_hypothesis")
            if isinstance(node.semantic.get("target_hypothesis"), dict)
            else {}
        )
        if (
            existing_target.get("verified")
            and existing_target.get("target_object") == self.target_object
        ):
            preserved_target = dict(existing_target)
            preserved_target["latest_unverified_evidence"] = evidence
            preserved_target["latest_unverified_frame_index"] = int(frame_index)
            node.semantic["target_hypothesis"] = preserved_target
        else:
            node.semantic["target_hypothesis"] = {
                "target_object": self.target_object,
                "evidence": evidence,
                "verified": False,
                "frame_index": int(frame_index),
                "policy": (
                    "VLM target evidence is a hypothesis; only the dedicated stop verifier "
                    "may authorize completion"
                ),
            }
        description_room = node.place_category or "unknown"
        node.semantic["short_description"] = "{}: {} ({})".format(
            description_room,
            intent,
            search_status,
        )
        if search_status in {"exhausted", "target_likely", "target_found"}:
            node.navigation_state["memory_importance"] = "high"

        self.runtime_stats["semantic_state_updates"] += 1
        if room_category_updated:
            self.runtime_stats["room_category_updates"] += 1
        if room_transition:
            self.runtime_stats["room_transitions"] += 1
        self.last_strategic_state = dict(normalized_state)
        result = {
            "updated": True,
            "node_id": node_id,
            "place_category": node.place_category,
            "previous_room": previous_room,
            "room_category_updated": room_category_updated,
            "room_transition": room_transition,
            "target_hypothesis_verified": False,
        }
        self.last_semantic_update = dict(result)
        log_fn = getattr(self.memory, "_log", None)
        if callable(log_fn):
            log_fn(
                "voca_semantic_state_update",
                node_id=node_id,
                frame_index=int(frame_index),
                previous_room=previous_room,
                current_room=node.place_category,
                target_evidence=evidence,
                navigation_intent=intent,
                room_search_status=search_status,
                room_transition=room_transition,
            )
        return result

    def record_verified_target_evidence(
        self,
        verification: Dict[str, Any],
        *,
        frame_index: int,
        image_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist dedicated-verifier target evidence as backend-authoritative memory."""
        if not self.enabled:
            return {"updated": False, "reason": "memory_disabled"}
        evidence = verification if isinstance(verification, dict) else {}
        identity = (
            evidence.get("identity_critic")
            if isinstance(evidence.get("identity_critic"), dict)
            else {}
        )
        # Completion confidence can remain medium when the target is correctly
        # identified but still too far away. Object memory should follow the
        # dedicated identity critic, not the separate close-enough judgment.
        if not (
            evidence.get("target_evidence_passed")
            and identity.get("triggered")
            and identity.get("passed")
            and identity.get("exact_target")
            and str(identity.get("confidence") or "").lower() == "high"
        ):
            return {"updated": False, "reason": "target_not_independently_verified"}

        node_id = str(self.memory.current_node_id or "")
        node = self.memory.nodes.get(node_id) if node_id else None
        if node is None:
            return {"updated": False, "reason": "no_current_place_node"}

        bbox = evidence.get("target_bbox_norm")
        sidecar_context = (
            self.last_context.get("voca_sidecar")
            if isinstance(self.last_context.get("voca_sidecar"), dict)
            else {}
        )
        position_xyz = [
            float(value)
            for value in list(sidecar_context.get("position_xyz") or [])[:3]
        ]
        candidate = {
            "target_object": self.target_object,
            "frame_index": int(frame_index),
            "image_ref": str(image_ref or "") or None,
            "target_view_index": evidence.get("target_view_index"),
            "target_center_x": evidence.get("target_center_x"),
            "target_bbox_norm": list(bbox) if isinstance(bbox, (list, tuple)) else None,
            "confidence": str(evidence.get("confidence") or "high"),
            "identity_confidence": str(identity.get("confidence") or "high"),
            "position_xyz": position_xyz,
            "heading_rad": float(sidecar_context.get("heading_rad", 0.0) or 0.0),
            "target_view_relative_heading_deg": evidence.get(
                "target_view_relative_heading_deg"
            ),
            "target_bearing_robot_deg": evidence.get(
                "target_bearing_robot_deg"
            ),
            "target_bearing_world_rad": evidence.get(
                "target_bearing_world_rad"
            ),
            "verification_source": "dedicated_stop_identity_critic",
            "identity_reason": str(identity.get("reason") or "")[:240],
        }
        belief = (
            node.semantic.get("object_belief")
            if isinstance(node.semantic.get("object_belief"), dict)
            else {}
        )
        candidates = [
            dict(item)
            for item in belief.get("candidate_objects", []) or []
            if isinstance(item, dict)
            and not (
                item.get("frame_index") == candidate["frame_index"]
                and item.get("image_ref") == candidate["image_ref"]
            )
        ]
        candidates.append(candidate)
        candidates = candidates[-4:]
        self.memory.update_object_belief(
            node_id,
            target_object=self.target_object,
            seen_target=True,
            candidate_objects=candidates,
            reason="dedicated_stop_identity_critic_verified_target",
        )
        node.semantic["target_hypothesis"] = {
            "target_object": self.target_object,
            "evidence": "confirmed",
            "verified": True,
            "frame_index": int(frame_index),
            "image_ref": candidate["image_ref"],
            "target_bbox_norm": candidate["target_bbox_norm"],
            "verification_source": candidate["verification_source"],
            "policy": (
                "backend-authoritative target memory; later unverified VLM semantics "
                "cannot clear it"
            ),
        }
        node.navigation_state["memory_importance"] = "critical"
        self.runtime_stats["object_belief_updates"] += 1
        result = {
            "updated": True,
            "node_id": node_id,
            "target_object": self.target_object,
            "frame_index": int(frame_index),
            "image_ref": candidate["image_ref"],
            "position_xyz": position_xyz,
            "heading_rad": candidate["heading_rad"],
            "target_view_relative_heading_deg": candidate[
                "target_view_relative_heading_deg"
            ],
            "target_bearing_robot_deg": candidate["target_bearing_robot_deg"],
            "target_bearing_world_rad": candidate["target_bearing_world_rad"],
            "candidate_count": len(candidates),
            "verification_source": candidate["verification_source"],
        }
        self.last_verified_target_evidence = dict(result)
        log_fn = getattr(self.memory, "_log", None)
        if callable(log_fn):
            log_fn("voca_verified_target_evidence", **result)
        return result

    @staticmethod
    def _memory_op_confidence(value: Any) -> float:
        if isinstance(value, str):
            mapped = {"high": 0.9, "medium": 0.7, "low": 0.4}
            if value.strip().lower() in mapped:
                return mapped[value.strip().lower()]
        try:
            return max(0.0, min(1.0, float(value or 0.0)))
        except Exception:
            return 0.0

    def apply_vlm_memory_ops(
        self,
        vlm_output: Dict[str, Any],
        *,
        frame_index: int,
    ) -> List[Dict[str, Any]]:
        """Commit only backend-resolved and independently verified memory ops."""
        results: List[Dict[str, Any]] = []
        ops = vlm_output.get("memory_ops") if isinstance(vlm_output, dict) else None
        if not isinstance(ops, list):
            return results
        self.runtime_stats["memory_ops_requested"] += sum(
            1 for op in ops if isinstance(op, dict)
        )
        revisit_refs = {
            str(item.get("candidate_ref")): item
            for item in self.last_context.get("candidate_refs", {}).get("revisits", [])
            if isinstance(item, dict) and item.get("candidate_ref")
        }
        for op in ops:
            if not isinstance(op, dict):
                continue
            name = str(op.get("op") or "")
            confidence = self._memory_op_confidence(op.get("confidence"))
            result: Dict[str, Any] = {
                "op": name,
                "candidate_ref": op.get("candidate_ref"),
                "confidence": round(confidence, 4),
                "accepted": False,
                "reason": "unsupported_or_unverified_memory_op",
            }
            if name in {
                "confirm_revisit_node",
                "confirm_same_place",
                "reject_revisit_candidate",
                "request_merge_nodes",
                "merge_nodes",
            }:
                candidate = revisit_refs.get(str(op.get("candidate_ref") or ""))
                if candidate is None:
                    result["reason"] = "unknown_or_missing_revisit_candidate_ref"
                    results.append(result)
                    self.runtime_stats["memory_ops_rejected"] += 1
                    continue
                resolved_node = candidate.get("node_id") or candidate.get("candidate_node_id")
                if resolved_node:
                    op = dict(op)
                    op.setdefault("node_id", resolved_node)
                    op.setdefault("candidate_node_id", resolved_node)
            try:
                if name in {"confirm_revisit_node", "confirm_same_place"}:
                    node_id = op.get("node_id") or op.get("candidate_node_id")
                    if node_id:
                        verification = self.memory.commit_revisit(
                            str(node_id),
                            frame_index=int(frame_index),
                            vlm_confidence=confidence,
                            reason=str(op.get("reason") or "voca_vlm_confirmed_revisit"),
                            min_backend_score=self.place_revisit_threshold,
                        )
                        result.update(
                            accepted=bool(verification.accepted),
                            reason=str(verification.reason),
                            node_id=str(node_id),
                            backend_score=round(float(verification.score), 4),
                        )
                elif name in {"request_merge_nodes", "merge_nodes"}:
                    keep = op.get("keep_node_id") or op.get("node_id") or op.get("candidate_node_id")
                    remove = op.get("remove_node_id") or self.memory.current_node_id
                    if keep and remove:
                        verification = self.memory.verify_merge_request(
                            str(keep),
                            str(remove),
                            vlm_confidence=confidence,
                            min_backend_score=self.place_revisit_threshold,
                        )
                        if verification.accepted:
                            soft = self.memory.soft_merge_revisit(
                                str(keep),
                                str(remove),
                                frame_index=int(frame_index),
                                reason=str(op.get("reason") or "voca_vlm_requested_soft_merge"),
                                vlm_confidence=confidence,
                            )
                            result.update(
                                accepted=bool(soft.accepted),
                                reason=str(soft.reason),
                            )
                        else:
                            result.update(reason=str(verification.reason))
                        result.update(keep_node_id=str(keep), remove_node_id=str(remove))
                elif name == "reject_revisit_candidate":
                    node_id = op.get("node_id") or op.get("candidate_node_id")
                    if node_id and confidence >= 0.55:
                        self.memory.reject_revisit_candidate(
                            str(node_id),
                            reason=str(op.get("reason") or "voca_vlm_rejected_revisit"),
                            vlm_confidence=confidence,
                        )
                        result.update(accepted=True, reason="revisit_rejection_audited", node_id=str(node_id))
                elif name == "defer_revisit_candidate":
                    node_id = op.get("node_id") or op.get("candidate_node_id")
                    result.update(
                        accepted=True,
                        reason="revisit_deferred_no_topology_mutation",
                        node_id=str(node_id) if node_id else None,
                    )
                elif name in {"update_object_belief", "mark_object_seen"} and self.memory.current_node_id:
                    if confidence >= 0.55:
                        self.memory.update_object_belief(
                            self.memory.current_node_id,
                            target_object=op.get("target_object") or self.target_object,
                            seen_target=op.get("seen_target"),
                            candidate_objects=op.get("candidate_objects"),
                            reason=str(op.get("reason") or "voca_vlm_object_belief"),
                        )
                        result.update(accepted=True, reason="object_belief_updated")
            except Exception as exc:
                result["reason"] = "{}: {}".format(type(exc).__name__, str(exc))[:240]
            results.append(result)
            if result.get("accepted"):
                self.runtime_stats["memory_ops_accepted"] += 1
                if name in {"confirm_revisit_node", "confirm_same_place"}:
                    self.runtime_stats["revisits_confirmed"] += 1
                    if "soft" in str(result.get("reason") or ""):
                        self.runtime_stats["soft_merges"] += 1
                elif name in {"request_merge_nodes", "merge_nodes"}:
                    self.runtime_stats["soft_merges"] += 1
                elif name == "reject_revisit_candidate":
                    self.runtime_stats["revisits_rejected"] += 1
                elif name in {"update_object_belief", "mark_object_seen"}:
                    self.runtime_stats["object_belief_updates"] += 1
            else:
                self.runtime_stats["memory_ops_rejected"] += 1
            log_fn = getattr(self.memory, "_log", None)
            if callable(log_fn):
                log_fn("voca_memory_op_result", **result)
        self.last_memory_op_results = results[-8:]
        return results

    def record_go_execution(
        self,
        *,
        go_execution: Dict[str, Any],
        frame_index: int,
        start_position_xyz: Sequence[float],
        final_position_xyz: Sequence[float],
        start_heading_rad: float = 0.0,
        final_heading_rad: float = 0.0,
    ) -> None:
        if not self.enabled:
            return
        src = self._ensure_current_node(
            image_ref=self.last_image_ref,
            frame_index=frame_index,
            position_xyz=start_position_xyz,
            heading_rad=start_heading_rad,
        )
        moved = _position_delta_m(start_position_xyz, final_position_xyz)
        start_pose = self._live_pose(start_position_xyz, start_heading_rad)
        cumulative_pose = self._live_pose(final_position_xyz, final_heading_rad)
        self.memory.update_current_pose_relation_to_latest_node(
            src,
            cumulative_pose,
            frame_index=int(frame_index),
            source="voca_post_go_pose",
        )
        progress = go_execution.get("go_progress") if isinstance(go_execution.get("go_progress"), dict) else {}
        collision = int(go_execution.get("collision_count", 0) or 0) > 0
        has_explicit_progress = any(
            key in progress
            for key in ("execution_success", "strategic_progress", "translation_m")
        )
        if has_explicit_progress:
            strategic_progress = bool(
                progress.get(
                    "strategic_progress",
                    progress.get("execution_success", not bool(progress.get("no_progress"))),
                )
            )
            no_progress = not strategic_progress
        else:
            no_progress = bool(progress.get("no_progress")) or moved < self.progress_threshold_m
        no_progress = bool(
            no_progress
            or (
                collision
                and not bool(progress.get("collision_tolerant_progress"))
            )
        )
        mode_before_execution = self.supervisor_mode

        if not no_progress:
            if cumulative_pose.distance_m() >= self.force_new_node_translation_m:
                dst = self.memory.add_node(
                    frame_index=int(frame_index),
                    image_ref=None,
                    embedding=None,
                    view_type="front",
                    relative_heading_deg=0.0,
                    place_category="unknown",
                    semantic_summary="VOCA reached new place at frame {}".format(int(frame_index)),
                )
                if mode_before_execution in {"escape_deadlock", "backtrack"}:
                    forward_edge_id = self.memory.mark_escape_edge(
                        src,
                        dst,
                        cumulative_pose,
                        frame_index=int(frame_index),
                    )
                else:
                    forward_edge_id = self.memory.add_or_update_edge(
                        src,
                        dst,
                        cumulative_pose,
                        status="success",
                        frame_index=int(frame_index),
                    )
                reverse_edge_id = self.memory.add_or_update_edge(
                    dst,
                    src,
                    cumulative_pose.inverse(),
                    edge_type="temporal_transition_reverse",
                    relation_type="backtrack",
                    status="success",
                    frame_index=int(frame_index),
                )
                tangents = (
                    go_execution.get("executed_path_tangents")
                    if isinstance(go_execution.get("executed_path_tangents"), dict)
                    else {}
                )
                departure_world_rad = _finite_float_or_none(
                    tangents.get("departure_heading_world_rad")
                )
                arrival_world_rad = _finite_float_or_none(
                    tangents.get("arrival_heading_world_rad")
                )
                anchor_heading = float(
                    self.current_node_anchor_heading_rad
                    if self.current_node_anchor_heading_rad is not None
                    else start_heading_rad
                )
                forward_departure_deg = (
                    _normalize_angle_deg(
                        math.degrees(departure_world_rad - anchor_heading)
                    )
                    if departure_world_rad is not None
                    else _normalize_angle_deg(start_pose.dyaw_deg)
                )
                reverse_departure_deg = (
                    _normalize_angle_deg(
                        math.degrees(
                            arrival_world_rad + math.pi - float(final_heading_rad)
                        )
                    )
                    if arrival_world_rad is not None
                    else 180.0
                )
                tangent_source = (
                    "executed_position_trace"
                    if departure_world_rad is not None and arrival_world_rad is not None
                    else "agent_heading_fallback"
                )
                raw_position_trace = tangents.get("position_trace_xyz")
                raw_position_trace = (
                    raw_position_trace
                    if isinstance(raw_position_trace, (list, tuple))
                    else []
                )
                forward_polyline = _relative_polyline_xy(
                    raw_position_trace,
                    anchor_position_xyz=(
                        self.current_node_anchor_position_xyz
                        or start_position_xyz
                    ),
                    anchor_heading_rad=anchor_heading,
                )
                reverse_polyline = _relative_polyline_xy(
                    list(reversed(raw_position_trace)),
                    anchor_position_xyz=final_position_xyz,
                    anchor_heading_rad=float(final_heading_rad),
                )
                forward_edge = self.memory.edges.get(forward_edge_id)
                reverse_edge = self.memory.edges.get(reverse_edge_id)
                if forward_edge is not None:
                    forward_edge.evidence.update(
                        {
                            "path_direction_schema": "executed_path_tangent_v1",
                            "departure_bearing_src_deg": round(
                                forward_departure_deg, 3
                            ),
                            "departure_bearing_source": tangent_source,
                            "arrival_backtrack_bearing_dst_deg": round(
                                reverse_departure_deg, 3
                            ),
                            "executed_path_length_m": tangents.get(
                                "path_length_m"
                            ),
                            "path_polyline_src_xy": forward_polyline,
                            "path_guidance_lookahead_m": 0.6,
                        }
                    )
                if reverse_edge is not None:
                    reverse_edge.evidence.update(
                        {
                            "path_direction_schema": "executed_path_tangent_v1",
                            "departure_bearing_src_deg": round(
                                reverse_departure_deg, 3
                            ),
                            "departure_bearing_source": tangent_source,
                            "reverse_of_edge_id": forward_edge_id,
                            "executed_path_length_m": tangents.get(
                                "path_length_m"
                            ),
                            "path_polyline_src_xy": reverse_polyline,
                            "path_guidance_lookahead_m": 0.6,
                        }
                    )
                self.current_node_anchor_position_xyz = [float(value) for value in final_position_xyz]
                self.current_node_anchor_heading_rad = float(final_heading_rad)
            self.no_progress_count = 0
            if mode_before_execution in {"escape_deadlock", "backtrack"}:
                self.update_supervisor_mode("goal_seek", reason="verified_escape_progress")
            return

        classification = (
            go_execution.get("failure_classification")
            if isinstance(go_execution.get("failure_classification"), dict)
            else {}
        )
        point = go_execution.get("selected_point_px")
        normalized_point = None
        if isinstance(point, (list, tuple)) and len(point) == 2:
            try:
                normalized_point = [int(round(float(point[0]))), int(round(float(point[1])))]
            except Exception:
                normalized_point = None
        try:
            selected_view_id = int(go_execution.get("selected_view_id"))
        except Exception:
            selected_view_id = None
        try:
            angle_deg = float(go_execution.get("selected_angle_deg", 0.0) or 0.0)
        except Exception:
            angle_deg = 0.0
        failure_class = str(
            classification.get("primary")
            or ("collision_blocked" if collision else "no_progress")
        )
        next_no_progress_count = self.no_progress_count + 1
        confirmed_failure = bool(collision or next_no_progress_count >= 2)
        attempted_frontier_pose = start_pose.compose(
            _NAV_MEMORY.schema.RelativePose2D(
                dx_m=max(
                    0.25,
                    min(
                        1.5,
                        max(self.failed_frontier_distance_m, float(moved)),
                    ),
                ),
                dy_m=0.0,
                dyaw_deg=0.0,
            )
        )
        negative_edge = self.memory.record_directional_failure(
            src,
            attempted_frontier_pose,
            frame_index=int(frame_index),
            reason=failure_class,
            confirmed=confirmed_failure,
            evidence={
                "selected_angle_deg": angle_deg,
                "selected_view_id": selected_view_id,
                "selected_candidate_ref": go_execution.get("selected_candidate_ref"),
                "selected_topological_candidate_ref": go_execution.get(
                    "selected_topological_candidate_ref"
                ),
                "selected_topological_edge_id": go_execution.get("selected_topological_edge_id"),
                "selected_topological_relation_type": go_execution.get(
                    "selected_topological_relation_type"
                ),
                "selected_point_px": normalized_point,
                "collision_count": int(go_execution.get("collision_count", 0) or 0),
                "translation_m": round(float(moved), 6),
            },
        )
        self.directional_failures.append(
            {
                "frame_index": int(frame_index),
                "angle_deg": angle_deg,
                "selected_view_id": selected_view_id,
                "selected_candidate_ref": (
                    str(go_execution.get("selected_candidate_ref"))
                    if go_execution.get("selected_candidate_ref") is not None
                    else None
                ),
                "selected_topological_candidate_ref": (
                    str(go_execution.get("selected_topological_candidate_ref"))
                    if go_execution.get("selected_topological_candidate_ref") is not None
                    else None
                ),
                "selected_topological_edge_id": (
                    str(go_execution.get("selected_topological_edge_id"))
                    if go_execution.get("selected_topological_edge_id") is not None
                    else None
                ),
                "selected_topological_relation_type": (
                    str(go_execution.get("selected_topological_relation_type"))
                    if go_execution.get("selected_topological_relation_type") is not None
                    else None
                ),
                "point_px": normalized_point,
                "failure_class": failure_class,
                "collision_count": int(go_execution.get("collision_count", 0) or 0),
                "translation_m": round(float(moved), 6),
                "supervisor_mode": mode_before_execution,
                "negative_edge_id": negative_edge.get("edge_id"),
                "failed_frontier_node_id": negative_edge.get("failed_frontier_node_id"),
                "negative_edge_status": negative_edge.get("status"),
            }
        )
        self.directional_failures = self.directional_failures[-8:]

        self.no_progress_count = next_no_progress_count
        if confirmed_failure:
            self.update_supervisor_mode(
                "escape_deadlock",
                reason="collision_confirmed_deadlock" if collision else "repeated_no_progress",
            )

    def visualizer_state(self) -> Dict[str, Any]:
        graph = self.memory.to_dict()
        layout = _MEMORY_VISUALIZER.reconstruct_memory_graph_layout(graph)
        failed_frontier_count = sum(
            1
            for node in self.memory.nodes.values()
            if node.node_type == "failed_frontier"
        )
        events = graph.get("event_log") or []
        last_event = next(
            (
                event
                for event in reversed(events)
                if event.get("event_type") != "update_live_pose_relation"
            ),
            events[-1] if events else {},
        )
        summary = {
            "enabled": self.enabled,
            "schema_version": "nav_memory_context_v6",
            "num_nodes": len(graph.get("nodes", {})),
            "num_place_nodes": len(graph.get("nodes", {})) - failed_frontier_count,
            "num_failed_frontier_nodes": failed_frontier_count,
            "num_edges": len(graph.get("edges", {})),
            "num_deadlock_edges": sum(1 for edge in self.memory.edges.values() if edge.is_negative()),
            "num_same_place_edges": sum(
                1
                for edge in self.memory.edges.values()
                if str(getattr(edge, "relation_type", "")).startswith("same_place")
            ),
            "num_visual_keyframes": len(self.memory.visual_index),
            "current_node_id": graph.get("current_node_id"),
            "no_progress_count": int(self.no_progress_count),
            "supervisor_mode": self.supervisor_mode,
            "last_mode_transition": dict(self.last_mode_transition),
            "directional_failure_count": len(self.directional_failures),
            "place_embedding": self._embedding_status(),
            "runtime_stats": dict(self.runtime_stats),
            "last_memory_op_results": [dict(item) for item in self.last_memory_op_results[-4:]],
            "last_strategic_state": dict(self.last_strategic_state),
            "last_semantic_update": dict(self.last_semantic_update),
            "last_verified_target_evidence": dict(
                self.last_verified_target_evidence
            ),
            "last_directional_failure": (
                dict(self.directional_failures[-1])
                if self.directional_failures
                else None
            ),
            "last_event": last_event,
            "graph_layout": {
                "nodes": [
                    {
                        "node_id": node_id,
                        "x_m": pose.x_m,
                        "y_m": pose.y_m,
                        "yaw_deg": pose.yaw_deg,
                        "current": node_id == layout.current_node_id,
                        "node_type": layout.nodes.get(node_id, {}).get("node_type"),
                        "place_category": layout.nodes.get(node_id, {}).get("place_category"),
                        "negative": bool(layout.nodes.get(node_id, {}).get("negative_memory")),
                    }
                    for node_id, pose in sorted(layout.poses.items())
                ],
                "edges": [
                    {
                        "edge_id": edge_id,
                        "src_node_id": edge.get("src_node_id"),
                        "dst_node_id": edge.get("dst_node_id"),
                        "status": edge.get("status")
                        or (
                            edge.get("traversal", {}).get("status")
                            if isinstance(edge.get("traversal"), dict)
                            else None
                        ),
                        "edge_type": edge.get("edge_type"),
                    }
                    for edge_id, edge in sorted(layout.edges.items())
                ],
            },
        }
        return _json_safe(summary)

    def save_artifacts(self, out_dir: Path) -> Dict[str, str]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        graph_path = out / "memory_graph.json"
        feedback_path = out / "Feedback.md"
        graph_path.write_text(
            json.dumps(_json_safe(self.memory.to_dict()), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        state = self.visualizer_state()
        lines = [
            "# VOCA Memory Sidecar",
            "",
            "schema: nav_memory_context_v6",
            "nodes: {}".format(state.get("num_nodes", 0)),
            "edges: {}".format(state.get("num_edges", 0)),
            "deadlock_edges: {}".format(state.get("num_deadlock_edges", 0)),
            "current_node_id: {}".format(state.get("current_node_id")),
            "supervisor_mode: {}".format(state.get("supervisor_mode")),
            "last_mode_transition: {}".format(
                json.dumps(state.get("last_mode_transition", {}), ensure_ascii=False)
            ),
            "",
            "## Recent Events",
        ]
        for event in self.memory.event_log[-20:]:
            lines.append("- {event_type}: {payload}".format(
                event_type=event.get("event_type"),
                payload=json.dumps(_json_safe({k: v for k, v in event.items() if k != "event_type"}), ensure_ascii=False),
            ))
        feedback_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        snapshot = _MEMORY_VISUALIZER.render_memory_graph_snapshot(
            graph_path,
            output_dir=out,
            title="VOCA Qwen-VLM Memory Graph",
        )
        video = _MEMORY_VISUALIZER.render_memory_graph_video(
            graph_path,
            output_dir=out,
            output_video="memory_graph_reconstruction.mp4",
            title="VOCA Qwen-VLM Memory Graph",
            fps=2,
            hold_frames=4,
        )
        return {
            "memory_graph_json": str(graph_path),
            "feedback_md": str(feedback_path),
            "memory_graph_png": str(snapshot["png_path"]),
            "memory_graph_html": str(snapshot["html_path"]),
            "memory_graph_visualizer_summary_json": str(snapshot["summary_json"]),
            "memory_graph_video": str(video["video_path"]),
            "memory_graph_video_summary_json": str(video["summary_json"]),
        }
