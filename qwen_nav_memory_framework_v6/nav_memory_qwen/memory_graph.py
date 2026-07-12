"""Relative-pose topo-metric episodic memory graph.

The graph stores no global node pose. Spatial structure is represented by
relative SE(2) transforms on directed edges. This keeps the memory topological at
node level while preserving enough local metric information to reconstruct a
subgoal by composing edge transforms.

This module deliberately does **not** implement pose-graph optimization. Loop
closure and merge decisions are represented as graph events and relative-edge
constraints, but global consistency optimization is left as a TODO for a future
backend module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import heapq
import math
import time
import uuid

import numpy as np

from .indexing import NumpyCosineIndex, SearchHit
from .schema import RelativePose2D, normalize_angle_deg, nearest_view_type, view_type_to_heading_deg
from .policy import NAV_SKILL_CARDS


HOT = "hot"
WARM = "warm"
COMPRESSED = "compressed"
ARCHIVED = "archived"

NEGATIVE_EDGE_STATUSES = {"deadlock_entry", "deadlock_entry_candidate", "blocked", "risky"}
DEADLOCK_STATUSES = {"none", "suspected", "confirmed", "escaping", "confirmed_escaped"}

# Constraint-only edges are useful for spatial reconstruction and same-place
# association, but they are not physical navigation exits to be offered as
# VLM fine-goal candidates. They can still be traversed by relative-pose queries.
CONSTRAINT_EDGE_TYPES = {
    "revisit_link",
    "revisit_link_reverse",
    "loop_closure",
    "same_place_constraint",
    "soft_merge_constraint",
    "internal_merge_tombstone",
}

# Category-specific same-place gates.  These are deliberately conservative:
# a duplicate node is safer than a false-positive merge that corrupts graph
# topology, deadlock memory, and future subgoal reconstruction.
PLACE_CATEGORY_SPATIAL_GATE: Dict[str, Dict[str, Any]] = {
    "doorway": {"max_baseline_m": 1.0, "extended_baseline_m": 1.5, "max_hops": 4, "same_region_baseline_m": 3.0, "allow_extended_same_place": False},
    "dead_end": {"max_baseline_m": 1.5, "extended_baseline_m": 2.0, "max_hops": 5, "same_region_baseline_m": 4.0, "allow_extended_same_place": True},
    "room": {"max_baseline_m": 2.0, "extended_baseline_m": 3.0, "max_hops": 6, "same_region_baseline_m": 6.0, "allow_extended_same_place": True},
    "intersection": {"max_baseline_m": 3.0, "extended_baseline_m": 5.0, "max_hops": 8, "same_region_baseline_m": 8.0, "allow_extended_same_place": True},
    "corridor": {"max_baseline_m": 1.5, "extended_baseline_m": 2.0, "max_hops": 5, "same_region_baseline_m": 10.0, "allow_extended_same_place": False, "repetitive_layout": True},
    "unknown": {"max_baseline_m": 1.5, "extended_baseline_m": 2.0, "max_hops": 4, "same_region_baseline_m": 3.0, "allow_extended_same_place": False},
}

CATEGORY_ALIASES = {
    "hall": "corridor",
    "hallway": "corridor",
    "passage": "corridor",
    "passageway": "corridor",
    "deadlock": "dead_end",
    "dead-end": "dead_end",
    "deadend": "dead_end",
    "dead_end_room": "dead_end",
    "junction": "intersection",
    "crossing": "intersection",
}


def _ts_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class Keyframe:
    """Visual evidence attached to a memory node.

    ``image_ref`` is active prompt evidence. When a node is compressed, image refs
    are removed from the VLM-facing context while ``archived_image_ref`` and the
    embedding metadata can remain available to backend storage.
    """

    keyframe_id: str
    node_id: str
    frame_index: int
    timestamp_ms: int
    view_type: str
    relative_heading_deg: float
    image_ref: Optional[str]
    thumbnail_ref: Optional[str] = None
    embedding_id: Optional[str] = None
    active_for_vlm: bool = True
    storage_tier: str = HOT
    quality_score: float = 1.0
    caption: Optional[str] = None
    archived_image_ref: Optional[str] = None

    def to_context(self) -> Dict[str, Any]:
        return {
            "keyframe_id": self.keyframe_id,
            "node_id": self.node_id,
            "view_type": self.view_type,
            "relative_heading_deg": round(float(self.relative_heading_deg), 3),
            "image_ref": self.image_ref if self.active_for_vlm else None,
            "thumbnail_ref": self.thumbnail_ref if self.active_for_vlm else None,
            "caption": self.caption,
            "storage_tier": self.storage_tier,
            "active_for_vlm": bool(self.active_for_vlm),
            "has_archived_image": bool(self.archived_image_ref),
        }


@dataclass
class MemoryNode:
    """Place/situation node in the episodic memory graph."""

    node_id: str
    node_type: str = "place"
    place_category: str = "unknown"
    created_frame_index: int = 0
    created_timestamp_ms: int = field(default_factory=_ts_ms)
    last_seen_frame_index: int = 0
    last_seen_timestamp_ms: int = field(default_factory=_ts_ms)
    visit_count: int = 1
    keyframes: List[Keyframe] = field(default_factory=list)
    semantic: Dict[str, Any] = field(default_factory=dict)
    navigation_state: Dict[str, Any] = field(default_factory=lambda: {
        "deadlock_status": "none",
        "loop_status": "none",
        "risk_level": "low",
        "memory_importance": "medium",
    })
    negative_memory: Optional[Dict[str, Any]] = None
    lifecycle: Dict[str, Any] = field(default_factory=lambda: {
        "storage_tier": HOT,
        "vlm_visible": True,
        "can_retrieve_image_for_vlm": True,
        "compression_level": "none",
    })
    visual_signature: Dict[str, Any] = field(default_factory=dict)

    def is_critical(self) -> bool:
        if self.negative_memory:
            return True
        if self.navigation_state.get("deadlock_status") in {"suspected", "confirmed", "escaping", "confirmed_escaped"}:
            return True
        if self.semantic.get("object_belief", {}).get("seen_target"):
            return True
        return self.navigation_state.get("memory_importance") == "critical"

    def short_description(self) -> str:
        return str(self.semantic.get("short_description") or self.place_category or self.node_id)

    def active_keyframe(self) -> Optional[Keyframe]:
        for kf in self.keyframes:
            if kf.active_for_vlm and kf.image_ref:
                return kf
        return None

    def to_context(self, include_images: bool = False) -> Dict[str, Any]:
        kfs = []
        if include_images:
            kfs = [kf.to_context() for kf in self.keyframes if kf.active_for_vlm and kf.image_ref]
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "place_category": self.place_category,
            "description": self.short_description(),
            "visit_count": self.visit_count,
            "last_seen_frame_index": self.last_seen_frame_index,
            "navigation_state": dict(self.navigation_state),
            "negative_memory": self.negative_memory,
            "object_belief": self.semantic.get("object_belief"),
            "same_place_cluster_id": self.semantic.get("same_place_cluster_id"),
            "canonical_place_node_id": self.semantic.get("canonical_place_node_id"),
            "same_place_node_ids": self.semantic.get("same_place_node_ids"),
            "lifecycle": dict(self.lifecycle),
            "keyframes": kfs,
        }


@dataclass
class MemoryEdge:
    """Directed graph edge with relative pose and traversal outcome."""

    edge_id: str
    src_node_id: str
    dst_node_id: str
    relative_pose_src_to_dst: RelativePose2D
    edge_type: str = "temporal_transition"
    relation_type: str = "transition"
    directed: bool = True
    traversal: Dict[str, Any] = field(default_factory=lambda: {
        "status": "unknown",
        "success_count": 0,
        "failure_count": 0,
        "last_outcome": None,
        "last_attempt_frame": None,
        "escape_edge_id": None,
    })
    planning_cost: Dict[str, float] = field(default_factory=lambda: {
        "base_cost": 1.0,
        "deadlock_penalty": 0.0,
        "loop_penalty": 0.0,
        "uncertainty_penalty": 0.0,
        "final_cost": 1.0,
    })
    evidence: Dict[str, Any] = field(default_factory=dict)

    def is_negative(self) -> bool:
        return str(self.traversal.get("status")) in NEGATIVE_EDGE_STATUSES

    def update_cost(self) -> None:
        base = max(0.05, self.relative_pose_src_to_dst.distance_m())
        status = str(self.traversal.get("status", "unknown"))
        failure_count = int(self.traversal.get("failure_count", 0) or 0)
        success_count = int(self.traversal.get("success_count", 0) or 0)
        deadlock_penalty = 0.0
        if status in {"deadlock_entry", "deadlock_entry_candidate"}:
            deadlock_penalty = 100.0 + 10.0 * failure_count
        elif status == "blocked":
            deadlock_penalty = 80.0 + 10.0 * failure_count
        elif status == "risky":
            deadlock_penalty = 25.0 + 5.0 * failure_count
        elif status == "escape_success":
            deadlock_penalty = -0.5
        elif success_count > 0:
            deadlock_penalty = -0.15 * min(success_count, 5)
        uncertainty = sum(float(x) for x in self.relative_pose_src_to_dst.covariance_diag)
        uncertainty_penalty = 0.1 * uncertainty
        final = max(0.01, base + deadlock_penalty + uncertainty_penalty + float(self.planning_cost.get("loop_penalty", 0.0)))
        self.planning_cost = {
            "base_cost": round(base, 4),
            "deadlock_penalty": round(deadlock_penalty, 4),
            "loop_penalty": round(float(self.planning_cost.get("loop_penalty", 0.0)), 4),
            "uncertainty_penalty": round(uncertainty_penalty, 4),
            "final_cost": round(final, 4),
        }

    def to_context(self, robot_relative_bearing_deg: Optional[float] = None) -> Dict[str, Any]:
        self.update_cost()
        d = {
            "edge_id": self.edge_id,
            "src_node_id": self.src_node_id,
            "dst_node_id": self.dst_node_id,
            "edge_type": self.edge_type,
            "relation_type": self.relation_type,
            "relative_pose_src_to_dst": self.relative_pose_src_to_dst.to_dict(),
            "traversal": dict(self.traversal),
            "planning_cost": dict(self.planning_cost),
            "avoid": self.is_negative(),
        }
        if robot_relative_bearing_deg is not None:
            d["bearing_deg_robot"] = round(float(robot_relative_bearing_deg), 3)
        return d


@dataclass
class LocalizationResult:
    """Backend retrieval result that proposes, but does not finally decide, place identity."""

    match_status: str
    current_node_id: Optional[str]
    match_confidence: float
    candidate_node_ids: List[str]
    matched_keyframe_ids: List[str]
    candidate_scores: Dict[str, float] = field(default_factory=dict)
    candidate_keyframe_ids: Dict[str, List[str]] = field(default_factory=dict)
    backend_commit_allowed: bool = False


@dataclass
class VerificationResult:
    """Backend verification result for VLM-requested memory operations."""

    accepted: bool
    reason: str
    score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpatialPlausibilityResult:
    """Spatial gate for VLM-confirmed revisit / soft-merge requests.

    The VLM may judge two views as semantically similar, but the backend should
    still reject visual aliases when the graph-composed relative baseline is not
    physically plausible for a same-place revisit.  This is not pose-graph
    optimization; it is a local consistency check over the existing relative
    edges.
    """

    accepted: bool
    reason: str
    relative_pose_src_to_dst: Optional[RelativePose2D] = None
    distance_m: float = math.inf
    abs_yaw_deg: float = math.inf
    node_path: List[str] = field(default_factory=list)
    edge_path: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty_level: str = "unknown"
    details: Dict[str, Any] = field(default_factory=dict)
    same_region_allowed: bool = False
    recommended_action: str = "reject_or_request_observation"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "relative_pose_src_to_dst": self.relative_pose_src_to_dst.to_dict() if self.relative_pose_src_to_dst else None,
            "distance_m": None if math.isinf(self.distance_m) else round(float(self.distance_m), 4),
            "abs_yaw_deg": None if math.isinf(self.abs_yaw_deg) else round(float(self.abs_yaw_deg), 3),
            "node_path": list(self.node_path),
            "edge_path": list(self.edge_path),
            "uncertainty_level": self.uncertainty_level,
            "same_region_allowed": bool(self.same_region_allowed),
            "recommended_action": self.recommended_action,
            "details": dict(self.details),
        }


@dataclass
class RelativePoseQueryResult:
    """Result of composing relative transforms through the graph.

    The pose is always expressed as ``T_src_to_dst``.  The search may use
    directed edges in their stored direction and, when allowed, inverse edges.
    This is the chain-rule operation needed for revisit links such as
    ``aTb = aTs ∘ sTd ∘ dTf ∘ fTg ∘ gTb``.
    """

    found: bool
    src_node_id: str
    dst_node_id: str
    relative_pose_src_to_dst: Optional[RelativePose2D] = None
    node_path: List[str] = field(default_factory=list)
    edge_path: List[Dict[str, Any]] = field(default_factory=list)
    cost: float = math.inf
    reason: str = "not_searched"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "found": self.found,
            "src_node_id": self.src_node_id,
            "dst_node_id": self.dst_node_id,
            "relative_pose_src_to_dst": self.relative_pose_src_to_dst.to_dict() if self.relative_pose_src_to_dst else None,
            "node_path": list(self.node_path),
            "edge_path": list(self.edge_path),
            "cost": None if math.isinf(self.cost) else round(float(self.cost), 4),
            "reason": self.reason,
        }


class MemoryGraph:
    """Relative-pose topo-metric episodic memory graph.

    Implemented ideas:
      * SPTM-style graph of places plus observation retrieval.
      * VLM-grounded place recognition: backend retrieval proposes candidates,
        VLM verifies via prompt, backend commits only after consistency checks.
      * Neural-topological-SLAM-style semantic nodes and coarse edge geometry.
      * PRISM-TopoMap-style no global node pose; only locally aligned graph.
      * SLAM-marginalization-inspired compression: active images can be dropped
        while summaries, embeddings, relative constraints, and negative memories stay.
    """

    def __init__(self, embedding_dim: int = 256, graph_id: Optional[str] = None):
        self.graph_id = graph_id or f"episode_{uuid.uuid4().hex[:8]}"
        self.embedding_dim = int(embedding_dim)
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: Dict[str, MemoryEdge] = {}
        self.out_edges: Dict[str, List[str]] = {}
        self.in_edges: Dict[str, List[str]] = {}
        self.keyframes: Dict[str, Keyframe] = {}
        self.visual_index = NumpyCosineIndex(dim=self.embedding_dim)
        self.current_node_id: Optional[str] = None
        self.previous_node_id: Optional[str] = None
        self.last_localization: Optional[LocalizationResult] = None
        self._node_counter = 0
        self._edge_counter = 0
        self._keyframe_counter = 0
        self.negative_edge_index: Dict[str, List[str]] = {}
        self.loop_history: List[str] = []
        self.event_log: List[Dict[str, Any]] = []
        self.pose_graph_optimization_todo = True
        self.same_place_clusters: Dict[str, Dict[str, Any]] = {}

        # Live, non-persistent pose relation between the latest/current node
        # anchor and the physical robot. This is the "floating edge" needed to
        # use graph exits while the robot has moved/rotated away from the node
        # origin but before a new node is committed. It is not a global node pose.
        self.latest_node_id_for_live_pose: Optional[str] = None
        self.latest_node_to_robot_pose: RelativePose2D = RelativePose2D()
        self.latest_node_pose_relation_frame_index: Optional[int] = None
        self.latest_node_pose_relation_source: str = "uninitialized"

    # ------------------------------------------------------------------
    # ID helpers
    # ------------------------------------------------------------------
    def _new_node_id(self) -> str:
        self._node_counter += 1
        return f"n_{self._node_counter:05d}"

    def _new_edge_id(self) -> str:
        self._edge_counter += 1
        return f"e_{self._edge_counter:05d}"

    def _new_keyframe_id(self) -> str:
        self._keyframe_counter += 1
        return f"kf_{self._keyframe_counter:05d}"

    def _log(self, event_type: str, **payload: Any) -> None:
        self.event_log.append({"timestamp_ms": _ts_ms(), "event_type": event_type, **payload})
        if len(self.event_log) > 500:
            self.event_log = self.event_log[-500:]

    # ------------------------------------------------------------------
    # Live current-pose relation to latest/current node
    # ------------------------------------------------------------------
    def update_current_pose_relation_to_latest_node(
        self,
        node_id: str,
        latest_node_to_robot: Optional[RelativePose2D] = None,
        *,
        frame_index: Optional[int] = None,
        source: str = "runtime_odometry",
    ) -> None:
        """Update the live floating transform ``T_latest_node_to_robot``.

        This relation is required for using outgoing graph edges after the robot
        has moved or rotated away from the latest node origin. It is deliberately
        stored as runtime session state, not as a global node pose.
        """
        if node_id not in self.nodes:
            raise KeyError(f"unknown node_id: {node_id}")
        self.latest_node_id_for_live_pose = node_id
        self.latest_node_to_robot_pose = latest_node_to_robot or RelativePose2D()
        self.latest_node_pose_relation_frame_index = frame_index
        self.latest_node_pose_relation_source = str(source)
        self._log(
            "update_live_pose_relation",
            node_id=node_id,
            frame_index=frame_index,
            source=source,
            latest_node_to_robot=self.latest_node_to_robot_pose.to_dict(),
        )

    def current_pose_relation_to_latest_node(self) -> Dict[str, Any]:
        """Return a VLM/backend-friendly summary of the live node↔robot pose."""
        node_id = self.current_node_id
        valid = bool(
            node_id
            and node_id == self.latest_node_id_for_live_pose
            and node_id in self.nodes
        )
        pose = self.latest_node_to_robot_pose if valid else RelativePose2D()
        robot_to_node = pose.inverse()
        return {
            "valid": valid,
            "latest_node_id": node_id,
            "pose_type": "live_relative_SE2_not_global_node_pose",
            "latest_node_to_robot": pose.to_dict(),
            "robot_to_latest_node": robot_to_node.to_dict(),
            "distance_from_latest_node_m": round(float(pose.distance_m()), 4),
            "heading_offset_from_latest_node_deg": round(float(pose.dyaw_deg), 3),
            "updated_frame_index": self.latest_node_pose_relation_frame_index,
            "source": self.latest_node_pose_relation_source,
            "usage": "transform node-frame graph exits into current robot frame before VLM fine-goal selection",
        }

    def live_pose_distance_from_latest_node_m(self) -> float:
        rel = self.current_pose_relation_to_latest_node()
        return float(rel.get("distance_from_latest_node_m", 0.0) or 0.0)

    def robot_frame_pose_for_edge(self, edge: "MemoryEdge") -> RelativePose2D:
        """Transform an outgoing edge target from node frame into current robot frame."""
        if (
            self.current_node_id
            and edge.src_node_id == self.current_node_id
            and self.latest_node_id_for_live_pose == self.current_node_id
        ):
            return self.latest_node_to_robot_pose.inverse().compose(edge.relative_pose_src_to_dst)
        return edge.relative_pose_src_to_dst

    def compose_path_pose_from_current_robot(self, path: Sequence[str]) -> Optional[RelativePose2D]:
        """Compose a node path and express the final target in the current robot frame."""
        node_frame_pose = self.compose_path_pose(path)
        if node_frame_pose is None:
            return None
        if path and path[0] == self.current_node_id and self.latest_node_id_for_live_pose == self.current_node_id:
            return self.latest_node_to_robot_pose.inverse().compose(node_frame_pose)
        return node_frame_pose

    # ------------------------------------------------------------------
    # Node/keyframe/edge mutation
    # ------------------------------------------------------------------
    def add_node(
        self,
        *,
        frame_index: int,
        image_ref: Optional[str],
        embedding: Optional[np.ndarray] = None,
        view_type: str = "front",
        relative_heading_deg: float = 0.0,
        place_category: str = "unknown",
        semantic_summary: Optional[str] = None,
        timestamp_ms: Optional[int] = None,
        provisional: bool = False,
        revisit_candidate_node_ids: Optional[Sequence[str]] = None,
    ) -> str:
        """Create a new place node and optional keyframe.

        ``provisional`` marks a node whose same-place identity should be
        confirmed by the VLM using ``revisit_candidates`` before merge/revisit
        is committed.
        """
        node_id = self._new_node_id()
        timestamp_ms = _ts_ms() if timestamp_ms is None else int(timestamp_ms)
        node = MemoryNode(
            node_id=node_id,
            place_category=place_category,
            created_frame_index=int(frame_index),
            created_timestamp_ms=timestamp_ms,
            last_seen_frame_index=int(frame_index),
            last_seen_timestamp_ms=timestamp_ms,
            semantic={"short_description": semantic_summary or f"place observed at frame {frame_index}"},
        )
        if provisional:
            node.lifecycle.update({
                "provisional": True,
                "vlm_place_recognition_pending": True,
                "compression_level": "none_provisional",
            })
            node.semantic["revisit_candidate_node_ids"] = list(revisit_candidate_node_ids or [])
        self.nodes[node_id] = node
        self.out_edges.setdefault(node_id, [])
        self.in_edges.setdefault(node_id, [])
        if image_ref is not None or embedding is not None:
            self.add_keyframe(
                node_id=node_id,
                frame_index=frame_index,
                image_ref=image_ref,
                embedding=embedding,
                view_type=view_type,
                relative_heading_deg=relative_heading_deg,
                timestamp_ms=timestamp_ms,
                caption=semantic_summary,
            )
        self.previous_node_id = self.current_node_id
        self.current_node_id = node_id
        self.update_current_pose_relation_to_latest_node(
            node_id,
            RelativePose2D(),
            frame_index=frame_index,
            source="node_created_identity_anchor",
        )
        self.loop_history.append(node_id)
        self.loop_history = self.loop_history[-20:]
        self._log("add_node", node_id=node_id, frame_index=frame_index, image_ref=bool(image_ref))
        return node_id

    def add_keyframe(
        self,
        *,
        node_id: str,
        frame_index: int,
        image_ref: Optional[str],
        embedding: Optional[np.ndarray],
        view_type: str = "front",
        relative_heading_deg: float = 0.0,
        timestamp_ms: Optional[int] = None,
        caption: Optional[str] = None,
    ) -> str:
        """Attach a keyframe and insert it into the visual index."""
        if node_id not in self.nodes:
            raise KeyError(f"unknown node_id: {node_id}")
        kf_id = self._new_keyframe_id()
        timestamp_ms = _ts_ms() if timestamp_ms is None else int(timestamp_ms)
        embedding_id = f"emb_{kf_id}" if embedding is not None else None
        kf = Keyframe(
            keyframe_id=kf_id,
            node_id=node_id,
            frame_index=int(frame_index),
            timestamp_ms=timestamp_ms,
            view_type=view_type,
            relative_heading_deg=float(relative_heading_deg),
            image_ref=image_ref,
            embedding_id=embedding_id,
            caption=caption,
        )
        self.keyframes[kf_id] = kf
        self.nodes[node_id].keyframes.append(kf)
        if embedding is not None:
            self.visual_index.add(kf_id, embedding, metadata={"node_id": node_id, "keyframe_id": kf_id})
            self.nodes[node_id].visual_signature.setdefault("embedding_ids", []).append(embedding_id)
            self.nodes[node_id].visual_signature["place_descriptor_id"] = embedding_id
        self._log("add_keyframe", node_id=node_id, keyframe_id=kf_id, frame_index=frame_index)
        return kf_id

    def localize(self, embedding: Optional[np.ndarray], top_k: int = 5, threshold: float = 0.86) -> LocalizationResult:
        """Generate backend retrieval candidates for VLM place recognition.

        The returned ``current_node_id`` is only an auto-localization proposal when
        the score passes ``threshold``. In the full agent, the VLM may still be
        asked to verify the candidate before a merge/revisit is committed.
        """
        if embedding is None or len(self.visual_index) == 0:
            result = LocalizationResult("new_place", None, 0.0, [], [])
            self.last_localization = result
            return result
        hits = self.visual_index.search(embedding, top_k=top_k, min_score=-1.0)
        candidate_node_ids: List[str] = []
        matched_keyframe_ids: List[str] = []
        candidate_scores: Dict[str, float] = {}
        candidate_keyframes: Dict[str, List[str]] = {}
        best_node: Optional[str] = None
        best_score = -1.0
        seen = set()
        for hit in hits:
            node_id = hit.metadata.get("node_id")
            if not node_id or node_id not in self.nodes:
                continue
            if self.nodes[node_id].lifecycle.get("storage_tier") == ARCHIVED:
                continue
            if node_id not in seen:
                candidate_node_ids.append(node_id)
                seen.add(node_id)
            matched_keyframe_ids.append(hit.key)
            candidate_keyframes.setdefault(node_id, []).append(hit.key)
            candidate_scores[node_id] = max(float(hit.score), candidate_scores.get(node_id, -1.0))
            if hit.score > best_score:
                best_score = float(hit.score)
                best_node = node_id
        if not candidate_node_ids:
            result = LocalizationResult("new_place", None, 0.0, [], [])
        else:
            status = "localized" if best_node and best_score >= threshold else "uncertain"
            result = LocalizationResult(
                status,
                best_node if status == "localized" else None,
                max(0.0, float(best_score)),
                candidate_node_ids,
                matched_keyframe_ids,
                candidate_scores=candidate_scores,
                candidate_keyframe_ids=candidate_keyframes,
                backend_commit_allowed=bool(status == "localized"),
            )
        self.last_localization = result
        return result

    def set_current_node(self, node_id: str, frame_index: int) -> None:
        """Set current node after verified localization/revisit."""
        if node_id not in self.nodes:
            raise KeyError(f"unknown node_id: {node_id}")
        if self.nodes[node_id].lifecycle.get("storage_tier") == ARCHIVED:
            raise ValueError(f"cannot set archived node as current: {node_id}")
        if self.current_node_id != node_id:
            self.previous_node_id = self.current_node_id
        self.current_node_id = node_id
        self.update_current_pose_relation_to_latest_node(
            node_id,
            RelativePose2D(),
            frame_index=frame_index,
            source="set_current_node_identity_anchor",
        )
        node = self.nodes[node_id]
        node.visit_count += 1
        node.last_seen_frame_index = int(frame_index)
        node.last_seen_timestamp_ms = _ts_ms()
        node.navigation_state["loop_status"] = "revisited" if node.visit_count > 1 else node.navigation_state.get("loop_status", "none")
        self.loop_history.append(node_id)
        self.loop_history = self.loop_history[-20:]
        self._log("set_current_node", node_id=node_id, frame_index=frame_index)

    def add_or_update_edge(
        self,
        src_node_id: str,
        dst_node_id: str,
        relative_pose: RelativePose2D,
        *,
        edge_type: str = "temporal_transition",
        relation_type: str = "transition",
        status: str = "success",
        frame_index: Optional[int] = None,
    ) -> str:
        """Create or update a directed edge."""
        if src_node_id == dst_node_id:
            return self.find_edge(src_node_id, dst_node_id) or ""
        if src_node_id not in self.nodes or dst_node_id not in self.nodes:
            raise KeyError("src or dst node does not exist")
        existing = self.find_edge(src_node_id, dst_node_id)
        if existing:
            edge = self.edges[existing]
            old_cov = edge.relative_pose_src_to_dst.covariance_diag
            new_cov = tuple(min(float(a), float(b)) for a, b in zip(old_cov, relative_pose.covariance_diag))
            edge.relative_pose_src_to_dst = RelativePose2D(relative_pose.dx_m, relative_pose.dy_m, relative_pose.dyaw_deg, new_cov)  # type: ignore[arg-type]
            edge.edge_type = edge_type or edge.edge_type
            edge.relation_type = relation_type or edge.relation_type
        else:
            edge_id = self._new_edge_id()
            edge = MemoryEdge(
                edge_id=edge_id,
                src_node_id=src_node_id,
                dst_node_id=dst_node_id,
                relative_pose_src_to_dst=relative_pose,
                edge_type=edge_type,
                relation_type=relation_type,
            )
            self.edges[edge_id] = edge
            self.out_edges.setdefault(src_node_id, []).append(edge_id)
            self.in_edges.setdefault(dst_node_id, []).append(edge_id)
        edge.traversal["last_attempt_frame"] = frame_index
        if status == "success":
            edge.traversal["success_count"] = int(edge.traversal.get("success_count", 0) or 0) + 1
            if edge.traversal.get("status") not in NEGATIVE_EDGE_STATUSES:
                edge.traversal["status"] = "success"
            edge.traversal["last_outcome"] = "success"
        elif status in NEGATIVE_EDGE_STATUSES or status in {"unknown", "escape_success"}:
            edge.traversal["status"] = status
            if status in NEGATIVE_EDGE_STATUSES:
                edge.traversal["failure_count"] = int(edge.traversal.get("failure_count", 0) or 0) + 1
            edge.traversal["last_outcome"] = status
        else:
            edge.traversal["status"] = status
            edge.traversal["last_outcome"] = status
        edge.update_cost()
        self.rebuild_negative_edge_index()
        self._log("add_or_update_edge", edge_id=edge.edge_id, src=src_node_id, dst=dst_node_id, status=status, frame_index=frame_index)
        return edge.edge_id

    def add_or_update_constraint_edge(
        self,
        src_node_id: str,
        dst_node_id: str,
        relative_pose: RelativePose2D,
        *,
        edge_type: str,
        relation_type: str,
        status: str = "same_place_constraint",
        frame_index: Optional[int] = None,
    ) -> str:
        """Create/update a non-navigation constraint edge without overwriting temporal edges."""
        if src_node_id == dst_node_id:
            return ""
        if src_node_id not in self.nodes or dst_node_id not in self.nodes:
            raise KeyError("src or dst node does not exist")
        existing = None
        for eid in self.out_edges.get(src_node_id, []):
            edge = self.edges.get(eid)
            if edge and edge.dst_node_id == dst_node_id and edge.edge_type == edge_type and edge.relation_type == relation_type:
                existing = eid
                break
        if existing:
            edge = self.edges[existing]
            old_cov = edge.relative_pose_src_to_dst.covariance_diag
            new_cov = tuple(min(float(a), float(b)) for a, b in zip(old_cov, relative_pose.covariance_diag))
            edge.relative_pose_src_to_dst = RelativePose2D(relative_pose.dx_m, relative_pose.dy_m, relative_pose.dyaw_deg, new_cov)  # type: ignore[arg-type]
        else:
            edge_id = self._new_edge_id()
            edge = MemoryEdge(
                edge_id=edge_id,
                src_node_id=src_node_id,
                dst_node_id=dst_node_id,
                relative_pose_src_to_dst=relative_pose,
                edge_type=edge_type,
                relation_type=relation_type,
            )
            self.edges[edge_id] = edge
            self.out_edges.setdefault(src_node_id, []).append(edge_id)
            self.in_edges.setdefault(dst_node_id, []).append(edge_id)
        edge.traversal["status"] = status
        edge.traversal["last_outcome"] = status
        edge.traversal["last_attempt_frame"] = frame_index
        edge.evidence["non_navigational_constraint"] = True
        edge.update_cost()
        self._log("add_or_update_constraint_edge", edge_id=edge.edge_id, src=src_node_id, dst=dst_node_id, edge_type=edge_type, frame_index=frame_index)
        return edge.edge_id

    def find_edge(self, src_node_id: str, dst_node_id: str) -> Optional[str]:
        for edge_id in self.out_edges.get(src_node_id, []):
            edge = self.edges.get(edge_id)
            if edge and edge.dst_node_id == dst_node_id:
                return edge_id
        return None

    def rebuild_adjacency(self) -> None:
        self.out_edges = {nid: [] for nid in self.nodes}
        self.in_edges = {nid: [] for nid in self.nodes}
        for eid, edge in list(self.edges.items()):
            if edge.src_node_id == edge.dst_node_id:
                self.edges.pop(eid, None)
                continue
            if edge.src_node_id not in self.nodes or edge.dst_node_id not in self.nodes:
                self.edges.pop(eid, None)
                continue
            self.out_edges.setdefault(edge.src_node_id, []).append(eid)
            self.in_edges.setdefault(edge.dst_node_id, []).append(eid)
        self.rebuild_negative_edge_index()

    def rebuild_negative_edge_index(self) -> None:
        self.negative_edge_index = {}
        for eid, edge in self.edges.items():
            if edge.is_negative():
                self.negative_edge_index.setdefault(edge.src_node_id, []).append(eid)

    # ------------------------------------------------------------------
    # Deadlock / escape / object memory
    # ------------------------------------------------------------------
    def record_directional_failure(
        self,
        src_node_id: str,
        relative_pose: RelativePose2D,
        *,
        frame_index: int,
        reason: str,
        confirmed: bool,
        evidence: Optional[Dict[str, Any]] = None,
        bearing_tolerance_deg: float = 35.0,
    ) -> Dict[str, Any]:
        """Record a failed local branch without moving the graph's current node.

        A failed PixelNav waypoint is not a newly visited place.  It is stored as
        a virtual ``failed_frontier`` node connected by a directional negative
        edge, so the attempted branch can be avoided while a successful incoming
        or reverse/backtrack edge remains usable.
        """
        if src_node_id not in self.nodes:
            raise KeyError("source node does not exist")

        requested_bearing = normalize_angle_deg(
            math.degrees(math.atan2(relative_pose.dy_m, relative_pose.dx_m))
        )
        edge: Optional[MemoryEdge] = None
        failed_node: Optional[MemoryNode] = None
        for candidate in self.outgoing_edges(src_node_id):
            if candidate.relation_type != "directional_failure":
                continue
            candidate_bearing = normalize_angle_deg(
                math.degrees(
                    math.atan2(
                        candidate.relative_pose_src_to_dst.dy_m,
                        candidate.relative_pose_src_to_dst.dx_m,
                    )
                )
            )
            if abs(normalize_angle_deg(requested_bearing - candidate_bearing)) <= float(
                bearing_tolerance_deg
            ):
                edge = candidate
                failed_node = self.nodes.get(candidate.dst_node_id)
                break

        status = "blocked" if confirmed else "deadlock_entry_candidate"
        if edge is None:
            node_id = self._new_node_id()
            failed_node = MemoryNode(
                node_id=node_id,
                node_type="failed_frontier",
                place_category="blocked_frontier",
                created_frame_index=int(frame_index),
                last_seen_frame_index=int(frame_index),
                semantic={
                    "short_description": "failed local frontier: {}".format(reason),
                    "virtual_frontier": True,
                },
            )
            failed_node.lifecycle.update(
                {
                    "storage_tier": COMPRESSED,
                    "vlm_visible": True,
                    "can_retrieve_image_for_vlm": False,
                    "compression_level": "virtual_failed_frontier",
                }
            )
            failed_node.navigation_state.update(
                {
                    "deadlock_status": "confirmed" if confirmed else "suspected",
                    "risk_level": "high" if confirmed else "medium",
                    "memory_importance": "critical" if confirmed else "high",
                }
            )
            self.nodes[node_id] = failed_node
            self.out_edges.setdefault(node_id, [])
            self.in_edges.setdefault(node_id, [])
            edge_id = self.add_or_update_edge(
                src_node_id,
                node_id,
                relative_pose,
                edge_type="failed_frontier",
                relation_type="directional_failure",
                status=status,
                frame_index=int(frame_index),
            )
            edge = self.edges[edge_id]
        else:
            edge_id = self.add_or_update_edge(
                src_node_id,
                edge.dst_node_id,
                relative_pose,
                edge_type="failed_frontier",
                relation_type="directional_failure",
                status=status,
                frame_index=int(frame_index),
            )
            edge = self.edges[edge_id]
            if failed_node is not None:
                failed_node.last_seen_frame_index = int(frame_index)
                failed_node.last_seen_timestamp_ms = _ts_ms()

        edge.evidence.update(dict(evidence or {}))
        edge.evidence.update(
            {
                "virtual_frontier": True,
                "failure_reason": str(reason),
                "bearing_deg_node": round(float(requested_bearing), 3),
            }
        )
        edge.update_cost()

        if failed_node is not None:
            failed_node.negative_memory = {
                "is_negative_region": False,
                "reason": str(reason),
                "avoid_scope": "directional_edge_only",
                "avoid_until": "episode_end",
                "severity": 0.95 if confirmed else 0.65,
                "failed_entry_edge_id": edge.edge_id,
            }
            failed_node.navigation_state.update(
                {
                    "deadlock_status": "confirmed" if confirmed else "suspected",
                    "risk_level": "high" if confirmed else "medium",
                    "memory_importance": "critical" if confirmed else "high",
                }
            )

        src = self.nodes[src_node_id]
        src.navigation_state.update(
            {
                "deadlock_status": "confirmed" if confirmed else "suspected",
                "risk_level": "high" if confirmed else "medium",
                "memory_importance": "critical" if confirmed else "high",
            }
        )
        if confirmed:
            src.negative_memory = {
                "is_negative_region": False,
                "reason": str(reason),
                "avoid_scope": "directional_edge_only",
                "avoid_until": "episode_end",
                "severity": 0.9,
                "failed_entry_edge_id": edge.edge_id,
            }

        self.rebuild_negative_edge_index()
        self._log(
            "record_directional_failure",
            node_id=src_node_id,
            failed_frontier_node_id=edge.dst_node_id,
            edge_id=edge.edge_id,
            status=status,
            bearing_deg_node=round(float(requested_bearing), 3),
            frame_index=int(frame_index),
            reason=str(reason),
        )
        return {
            "edge_id": edge.edge_id,
            "failed_frontier_node_id": edge.dst_node_id,
            "status": status,
            "bearing_deg_node": round(float(requested_bearing), 3),
            "reused_existing_edge": bool(edge.traversal.get("failure_count", 0) > 1),
        }

    def mark_deadlock_suspected(self, node_id: str, incoming_edge_id: Optional[str], reason: str = "deadlock_suspected") -> None:
        """Mark a node/entry edge as suspected deadlock without fully blocking it."""
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        old = str(node.navigation_state.get("deadlock_status", "none"))
        if old not in {"confirmed", "confirmed_escaped"}:
            node.navigation_state.update({"deadlock_status": "suspected", "risk_level": "medium", "memory_importance": "high"})
        if incoming_edge_id and incoming_edge_id in self.edges:
            edge = self.edges[incoming_edge_id]
            if edge.traversal.get("status") not in {"deadlock_entry", "blocked"}:
                edge.traversal["status"] = "deadlock_entry_candidate"
                edge.traversal["last_outcome"] = reason
                edge.update_cost()
        self.rebuild_negative_edge_index()
        self._log("mark_deadlock_suspected", node_id=node_id, incoming_edge_id=incoming_edge_id, reason=reason)

    def mark_deadlock(self, node_id: str, incoming_edge_id: Optional[str], reason: str = "deadlock_detected") -> None:
        """Confirm a node and its incoming edge as deadlock-related negative memory."""
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        node.navigation_state.update({"deadlock_status": "confirmed", "risk_level": "high", "memory_importance": "critical"})
        node.negative_memory = {
            "is_negative_region": True,
            "reason": reason,
            "avoid_scope": "incoming_edge_only" if incoming_edge_id else "node_region",
            "avoid_until": "episode_end",
            "severity": 0.9,
            "failed_entry_edge_id": incoming_edge_id,
        }
        if incoming_edge_id and incoming_edge_id in self.edges:
            edge = self.edges[incoming_edge_id]
            edge.traversal["status"] = "deadlock_entry"
            edge.traversal["failure_count"] = int(edge.traversal.get("failure_count", 0) or 0) + 1
            edge.traversal["last_outcome"] = reason
            edge.update_cost()
        self.rebuild_negative_edge_index()
        self._log("mark_deadlock", node_id=node_id, incoming_edge_id=incoming_edge_id, reason=reason)

    def mark_blocked_edge(self, edge_id: Optional[str], reason: str = "blocked") -> None:
        if not edge_id or edge_id not in self.edges:
            return
        edge = self.edges[edge_id]
        edge.traversal["status"] = "blocked"
        edge.traversal["failure_count"] = int(edge.traversal.get("failure_count", 0) or 0) + 1
        edge.traversal["last_outcome"] = reason
        edge.update_cost()
        self.rebuild_negative_edge_index()
        self._log("mark_blocked_edge", edge_id=edge_id, reason=reason)

    def mark_escape_edge(self, src_node_id: str, dst_node_id: str, relative_pose: RelativePose2D, frame_index: Optional[int] = None) -> str:
        """Mark an edge as an escape/backtrack route from a deadlock region."""
        edge_id = self.add_or_update_edge(
            src_node_id,
            dst_node_id,
            relative_pose,
            edge_type="temporal_transition",
            relation_type="backtrack_or_escape",
            status="escape_success",
            frame_index=frame_index,
        )
        edge = self.edges[edge_id]
        edge.traversal["status"] = "escape_success"
        edge.traversal["success_count"] = int(edge.traversal.get("success_count", 0) or 0) + 1
        edge.update_cost()
        src = self.nodes.get(src_node_id)
        if src and src.negative_memory:
            src.negative_memory["escape_edge_id"] = edge_id
            src.navigation_state["deadlock_status"] = "confirmed_escaped"
        rev = self.find_edge(dst_node_id, src_node_id)
        if rev and rev in self.edges:
            self.edges[rev].traversal["escape_edge_id"] = edge_id
        self._log("mark_escape_edge", edge_id=edge_id, src=src_node_id, dst=dst_node_id)
        return edge_id

    def update_object_belief(
        self,
        node_id: str,
        *,
        target_object: Optional[str] = None,
        seen_target: Optional[bool] = None,
        candidate_objects: Optional[List[Dict[str, Any]]] = None,
        room_object_prior: Optional[Dict[str, float]] = None,
        reason: str = "vlm_object_belief_update",
    ) -> None:
        """Update ObjNav semantic belief stored inside a node."""
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        belief = dict(node.semantic.get("object_belief") or {})
        if target_object is not None:
            belief["target_object"] = target_object
        if seen_target is not None:
            belief["seen_target"] = bool(seen_target)
        if candidate_objects is not None:
            belief["candidate_objects"] = list(candidate_objects)
        if room_object_prior is not None:
            belief["room_object_prior"] = {str(k): float(v) for k, v in room_object_prior.items()}
        belief["last_update_reason"] = reason
        node.semantic["object_belief"] = belief
        if belief.get("seen_target"):
            node.navigation_state["memory_importance"] = "critical"
        self._log("update_object_belief", node_id=node_id, target_object=belief.get("target_object"), seen_target=belief.get("seen_target"))

    # ------------------------------------------------------------------
    # VLM-grounded place recognition and merge verification
    # ------------------------------------------------------------------
    def _candidate_visual_score(self, node_id: str) -> float:
        loc = self.last_localization
        if not loc:
            return 0.0
        return float(loc.candidate_scores.get(node_id, 0.0))

    def _is_retrieval_candidate(self, node_id: str) -> bool:
        loc = self.last_localization
        return bool(loc and node_id in loc.candidate_node_ids)

    def _relative_pose_hint_between_current_and(self, node_id: str) -> Optional[Dict[str, Any]]:
        if not self.current_node_id or self.current_node_id not in self.nodes or node_id not in self.nodes:
            return None
        if self.current_node_id == node_id:
            return {"dx_m": 0.0, "dy_m": 0.0, "dyaw_deg": 0.0, "uncertainty": "low", "source": "same_node"}
        direct = self.find_edge(self.current_node_id, node_id)
        if direct:
            p = self.edges[direct].relative_pose_src_to_dst
            d = p.to_dict()
            d.update({"uncertainty": "low", "source": "direct_edge", "edge_id": direct})
            return d
        rev = self.find_edge(node_id, self.current_node_id)
        if rev:
            p = self.edges[rev].relative_pose_src_to_dst.inverse()
            d = p.to_dict()
            d.update({"uncertainty": "medium", "source": "inverse_edge", "edge_id": rev})
            return d
        query = self.relative_pose_between_nodes(
            self.current_node_id,
            node_id,
            allow_reverse_edges=True,
            avoid_negative=False,
            include_constraint_edges=True,
            include_archived=False,
        )
        if query.found and query.relative_pose_src_to_dst is not None:
            d = query.relative_pose_src_to_dst.to_dict()
            d.update({
                "uncertainty": "high" if len(query.edge_path) > 4 else "medium",
                "source": "composed_graph_path",
                "node_path": query.node_path,
                "edge_path": query.edge_path,
            })
            return d
        return {"uncertainty": "unknown", "source": "no_connected_graph_path"}

    @staticmethod
    def _normalize_place_category_for_gate(category: Optional[str]) -> str:
        raw = str(category or "unknown").strip().lower().replace(" ", "_")
        raw = CATEGORY_ALIASES.get(raw, raw)
        if raw in PLACE_CATEGORY_SPATIAL_GATE:
            return raw
        # Composite labels such as "small_room" or "corridor_segment" are
        # normalized by substring to avoid silently falling back to permissive
        # thresholds.
        if "corridor" in raw or "hallway" in raw:
            return "corridor"
        if "door" in raw:
            return "doorway"
        if "intersect" in raw or "junction" in raw:
            return "intersection"
        if "dead" in raw:
            return "dead_end"
        if "room" in raw:
            return "room"
        return "unknown"

    def _spatial_gate_policy_for_pair(
        self,
        src_node_id: str,
        dst_node_id: str,
        *,
        global_max_same_place_baseline_m: float,
        global_extended_same_place_baseline_m: float,
        global_max_same_place_hops: int,
    ) -> Dict[str, Any]:
        src_cat = self._normalize_place_category_for_gate(self.nodes[src_node_id].place_category)
        dst_cat = self._normalize_place_category_for_gate(self.nodes[dst_node_id].place_category)
        src_policy = PLACE_CATEGORY_SPATIAL_GATE[src_cat]
        dst_policy = PLACE_CATEGORY_SPATIAL_GATE[dst_cat]

        # Use the more conservative side of the pair.  If either endpoint is a
        # doorway/corridor/unknown, the pair inherits that stricter baseline.
        max_baseline = min(float(global_max_same_place_baseline_m), float(src_policy["max_baseline_m"]), float(dst_policy["max_baseline_m"]))
        extended = min(float(global_extended_same_place_baseline_m), float(src_policy["extended_baseline_m"]), float(dst_policy["extended_baseline_m"]))
        max_hops = min(int(global_max_same_place_hops), int(src_policy["max_hops"]), int(dst_policy["max_hops"]))
        same_region_baseline = max(float(src_policy["same_region_baseline_m"]), float(dst_policy["same_region_baseline_m"]))
        allow_extended = bool(src_policy.get("allow_extended_same_place", False) and dst_policy.get("allow_extended_same_place", False))
        repetitive = bool(src_policy.get("repetitive_layout", False) or dst_policy.get("repetitive_layout", False))
        return {
            "src_category": src_cat,
            "dst_category": dst_cat,
            "max_same_place_baseline_m": max_baseline,
            "extended_same_place_baseline_m": max(extended, max_baseline),
            "max_same_place_hops": max(1, max_hops),
            "same_region_baseline_m": same_region_baseline,
            "allow_extended_same_place": allow_extended,
            "repetitive_layout": repetitive,
            "false_positive_priority": "reject_same_place_when_spatially_implausible_even_if_visuals_match",
        }

    def spatial_plausibility_between_nodes(
        self,
        src_node_id: str,
        dst_node_id: str,
        *,
        max_same_place_baseline_m: float = 3.0,
        extended_same_place_baseline_m: float = 5.0,
        max_same_place_hops: int = 12,
        visual_score: float = 0.0,
        vlm_confidence: float = 0.0,
        include_constraint_edges_as_fallback: bool = True,
    ) -> SpatialPlausibilityResult:
        """Check whether two nodes are close enough to be a same-place revisit.

        This function guards against visual aliasing.  The VLM may confirm that
        two views look similar, but the backend only accepts the revisit/soft
        merge when the relative-pose graph can also explain the baseline.  The
        test uses a graph-composed transform, not a global node pose.
        """
        if src_node_id not in self.nodes:
            return SpatialPlausibilityResult(False, "src_node_missing")
        if dst_node_id not in self.nodes:
            return SpatialPlausibilityResult(False, "dst_node_missing")
        if src_node_id == dst_node_id:
            return SpatialPlausibilityResult(
                True,
                "same_node_identity",
                RelativePose2D(),
                0.0,
                0.0,
                [src_node_id],
                [],
                "low",
                {"baseline_policy": "same_node"},
            )

        policy = self._spatial_gate_policy_for_pair(
            src_node_id,
            dst_node_id,
            global_max_same_place_baseline_m=max_same_place_baseline_m,
            global_extended_same_place_baseline_m=extended_same_place_baseline_m,
            global_max_same_place_hops=max_same_place_hops,
        )

        query = self.relative_pose_between_nodes(
            src_node_id,
            dst_node_id,
            allow_reverse_edges=True,
            avoid_negative=False,
            include_constraint_edges=False,
            include_archived=False,
            max_hops=int(policy["max_same_place_hops"]),
        )
        used_constraint_fallback = False
        if not query.found and include_constraint_edges_as_fallback:
            query = self.relative_pose_between_nodes(
                src_node_id,
                dst_node_id,
                allow_reverse_edges=True,
                avoid_negative=False,
                include_constraint_edges=True,
                include_archived=False,
                max_hops=int(policy["max_same_place_hops"]),
            )
            used_constraint_fallback = bool(query.found)

        if not query.found or query.relative_pose_src_to_dst is None:
            return SpatialPlausibilityResult(
                False,
                "no_connected_relative_path",
                node_path=query.node_path,
                edge_path=query.edge_path,
                details={"query": query.to_dict(), "max_same_place_hops": int(policy["max_same_place_hops"]), "gate_policy": policy},
                recommended_action="request_observation_or_keep_duplicate_node",
            )

        pose = query.relative_pose_src_to_dst
        distance = float(pose.distance_m())
        abs_yaw = abs(normalize_angle_deg(pose.dyaw_deg))
        hops = len(query.edge_path)
        xy_cov_trace = float(pose.covariance_diag[0] + pose.covariance_diag[1]) if pose.covariance_diag else 0.0
        if hops <= 2 and xy_cov_trace <= 0.5:
            uncertainty = "low"
        elif hops <= 6 and xy_cov_trace <= 1.5:
            uncertainty = "medium"
        else:
            uncertainty = "high"

        high_multimodal_support = bool(visual_score >= 0.82 and vlm_confidence >= 0.85)
        accepted = False
        reason = "spatial_baseline_too_large"
        same_place_baseline = float(policy["max_same_place_baseline_m"])
        extended_baseline = float(policy["extended_same_place_baseline_m"])
        same_region_baseline = float(policy["same_region_baseline_m"])
        allow_extended = bool(policy["allow_extended_same_place"])
        repetitive_layout = bool(policy["repetitive_layout"])

        if distance <= same_place_baseline:
            accepted = True
            reason = "spatially_plausible_same_place_baseline"
        elif allow_extended and distance <= extended_baseline and high_multimodal_support and uncertainty != "high":
            accepted = True
            reason = "spatially_plausible_extended_baseline_with_high_visual_vlm_support"
        elif used_constraint_fallback and allow_extended and distance <= extended_baseline and high_multimodal_support:
            accepted = True
            reason = "spatially_plausible_existing_constraint_fallback"

        same_region_allowed = bool((not accepted) and distance <= same_region_baseline and high_multimodal_support)
        if same_region_allowed:
            recommended_action = "same_region_only_no_same_place_merge_or_request_observation"
        elif accepted:
            recommended_action = "allow_soft_same_place_link"
        else:
            recommended_action = "reject_or_request_observation_keep_duplicate_node"

        details = {
            "input_global_max_same_place_baseline_m": float(max_same_place_baseline_m),
            "input_global_extended_same_place_baseline_m": float(extended_same_place_baseline_m),
            "effective_max_same_place_baseline_m": same_place_baseline,
            "effective_extended_same_place_baseline_m": extended_baseline,
            "effective_max_same_place_hops": int(policy["max_same_place_hops"]),
            "same_region_baseline_m": same_region_baseline,
            "same_region_allowed": same_region_allowed,
            "hops": int(hops),
            "xy_cov_trace": round(xy_cov_trace, 6),
            "visual_score": round(float(visual_score), 4),
            "vlm_confidence": round(float(vlm_confidence), 4),
            "high_multimodal_support": high_multimodal_support,
            "used_constraint_fallback": used_constraint_fallback,
            "query_reason": query.reason,
            "gate_policy": policy,
            "repetitive_layout": repetitive_layout,
            "false_positive_merge_priority": "duplicates_are_preferred_to_false_positive_same_place_merges",
        }
        return SpatialPlausibilityResult(
            accepted,
            reason,
            pose,
            distance,
            abs_yaw,
            query.node_path,
            query.edge_path,
            uncertainty,
            details,
            same_region_allowed=same_region_allowed,
            recommended_action=recommended_action,
        )

    def get_revisit_candidates(self, max_candidates: int = 4) -> List[Dict[str, Any]]:
        """Return top backend retrieval candidates for VLM verification."""
        loc = self.last_localization
        if not loc or not loc.candidate_node_ids:
            return []
        out: List[Dict[str, Any]] = []
        for node_id in loc.candidate_node_ids[:max_candidates]:
            node = self.nodes.get(node_id)
            if not node or node.lifecycle.get("storage_tier") == ARCHIVED:
                continue
            active_kf = node.active_keyframe()
            candidate_keyframes = loc.candidate_keyframe_ids.get(node_id, [])
            visual_score = self._candidate_visual_score(node_id)
            spatial = self.spatial_plausibility_between_nodes(
                node_id,
                self.current_node_id or node_id,
                visual_score=visual_score,
                vlm_confidence=0.0,
            )
            recommended = (
                "same_place_if_layout_and_openings_match"
                if spatial.accepted
                else "reject_or_request_observation_due_to_spatial_implausibility"
            )
            out.append({
                "candidate_node_id": node_id,
                "candidate_keyframe_ids": candidate_keyframes,
                "candidate_image_ref": active_kf.image_ref if active_kf else None,
                "candidate_view_type": active_kf.view_type if active_kf else None,
                "visual_retrieval_score": round(visual_score, 4),
                "topological_consistency": self._topological_consistency_label(node_id),
                "relative_pose_hint_from_current": self._relative_pose_hint_between_current_and(node_id),
                "spatial_plausibility": spatial.to_dict(),
                "semantic_summary": node.short_description(),
                "negative_memory": node.negative_memory,
                "object_belief": node.semantic.get("object_belief"),
                "recommended_vlm_judgement": recommended,
            })
        return out

    def _topological_consistency_label(self, node_id: str) -> str:
        if not self.current_node_id:
            return "unknown_current_node"
        if node_id == self.current_node_id:
            return "same_as_current_node"
        if self.find_edge(self.current_node_id, node_id) or self.find_edge(node_id, self.current_node_id):
            return "adjacent_in_graph"
        if node_id in self.loop_history[-8:]:
            return "recently_seen_in_loop_history"
        return "unverified_but_retrieved"

    def verify_revisit_candidate(
        self,
        candidate_node_id: str,
        *,
        vlm_confidence: float,
        min_vlm_confidence: float = 0.62,
        min_backend_score: float = 0.45,
        max_same_place_baseline_m: float = 3.0,
        extended_same_place_baseline_m: float = 5.0,
        max_same_place_hops: int = 12,
    ) -> VerificationResult:
        """Verify a VLM ``confirm_revisit_node`` request before committing it.

        Verification has three gates: VLM confidence, backend visual retrieval,
        and spatial plausibility from the relative-pose graph.  The last gate is
        what prevents similar-looking but distant aliases from being merged.
        """
        if candidate_node_id not in self.nodes:
            return VerificationResult(False, "candidate_node_missing")
        if self.nodes[candidate_node_id].lifecycle.get("storage_tier") == ARCHIVED:
            return VerificationResult(False, "candidate_node_archived")
        score = self._candidate_visual_score(candidate_node_id)
        is_candidate = self._is_retrieval_candidate(candidate_node_id)
        if vlm_confidence < min_vlm_confidence:
            return VerificationResult(False, "vlm_confidence_too_low", score)
        if not is_candidate and vlm_confidence < 0.85:
            return VerificationResult(False, "not_in_backend_retrieval_candidates", score)
        if score < min_backend_score and vlm_confidence < 0.85:
            return VerificationResult(False, "backend_visual_score_too_low", score)

        current = self.current_node_id
        if current and current != candidate_node_id:
            spatial = self.spatial_plausibility_between_nodes(
                candidate_node_id,
                current,
                max_same_place_baseline_m=max_same_place_baseline_m,
                extended_same_place_baseline_m=extended_same_place_baseline_m,
                max_same_place_hops=max_same_place_hops,
                visual_score=score,
                vlm_confidence=vlm_confidence,
            )
            if not spatial.accepted:
                return VerificationResult(
                    False,
                    f"spatial_plausibility_failed:{spatial.reason}",
                    score,
                    {"is_backend_candidate": is_candidate, "spatial_plausibility": spatial.to_dict()},
                )
            return VerificationResult(
                True,
                "verified_revisit_candidate",
                score,
                {"is_backend_candidate": is_candidate, "spatial_plausibility": spatial.to_dict()},
            )

        return VerificationResult(True, "verified_revisit_candidate", score, {"is_backend_candidate": is_candidate, "spatial_plausibility": {"accepted": True, "reason": "same_or_no_current_node"}})

    def commit_revisit(
        self,
        candidate_node_id: str,
        *,
        frame_index: int,
        vlm_confidence: float,
        reason: str = "vlm_confirmed_revisit",
        min_vlm_confidence: float = 0.62,
        min_backend_score: float = 0.45,
        max_same_place_baseline_m: float = 3.0,
        extended_same_place_baseline_m: float = 5.0,
        max_same_place_hops: int = 12,
    ) -> VerificationResult:
        """Commit a VLM-confirmed revisit using a soft merge by default.

        A revisit observation can have a meaningful baseline/parallax difference
        from the first visit.  Therefore the default behavior preserves the
        current/revisit node, creates a same-place relative constraint edge from
        the canonical node to the revisit node, and re-anchors the live current
        pose to the canonical node with ``T_canonical_to_robot`` set to that
        baseline.  The old hard merge remains available through ``merge_nodes``
        but is not used here.
        """
        verification = self.verify_revisit_candidate(
            candidate_node_id,
            vlm_confidence=vlm_confidence,
            min_vlm_confidence=min_vlm_confidence,
            min_backend_score=min_backend_score,
            max_same_place_baseline_m=max_same_place_baseline_m,
            extended_same_place_baseline_m=extended_same_place_baseline_m,
            max_same_place_hops=max_same_place_hops,
        )
        if not verification.accepted:
            self._log("reject_revisit", candidate_node_id=candidate_node_id, reason=verification.reason, score=verification.score)
            return verification

        current = self.current_node_id
        if current and current != candidate_node_id and current in self.nodes:
            soft = self.soft_merge_revisit(
                candidate_node_id,
                current,
                frame_index=frame_index,
                reason=reason,
                vlm_confidence=vlm_confidence,
            )
            if not soft.accepted:
                self._log(
                    "reject_revisit_soft_link_failed",
                    candidate_node_id=candidate_node_id,
                    revisit_node_id=current,
                    reason=soft.reason,
                    score=verification.score,
                )
                return VerificationResult(False, f"soft_link_failed:{soft.reason}", verification.score, {"revisit_verification": verification.details})
            result = VerificationResult(True, "verified_revisit_soft_linked", verification.score, {**verification.details, **soft.details})
        else:
            self.set_current_node(candidate_node_id, frame_index)
            result = verification

        self.nodes[candidate_node_id].semantic["last_revisit_confirmation"] = {
            "reason": reason,
            "vlm_confidence": round(float(vlm_confidence), 4),
            "backend_visual_score": round(float(verification.score), 4),
            "merge_policy": "soft_preserve_revisit_node",
        }
        self._log("commit_revisit", candidate_node_id=candidate_node_id, frame_index=frame_index, reason=reason, confidence=vlm_confidence, policy="soft_preserve_revisit_node")
        return result

    def reject_revisit_candidate(self, candidate_node_id: str, *, reason: str = "vlm_rejected_revisit", vlm_confidence: float = 0.0) -> None:
        """Audit a VLM same-place rejection without mutating graph topology."""
        self._log(
            "reject_revisit_candidate",
            candidate_node_id=candidate_node_id,
            current_node_id=self.current_node_id,
            reason=reason,
            vlm_confidence=round(float(vlm_confidence), 4),
        )

    def verify_merge_request(
        self,
        keep_node_id: str,
        remove_node_id: str,
        *,
        vlm_confidence: float,
        min_vlm_confidence: float = 0.72,
        min_backend_score: float = 0.50,
        max_same_place_baseline_m: float = 3.0,
        extended_same_place_baseline_m: float = 5.0,
        max_same_place_hops: int = 12,
    ) -> VerificationResult:
        """Verify a VLM-requested node merge.

        This is intentionally conservative because false merges corrupt the graph.
        """
        if keep_node_id == remove_node_id:
            return VerificationResult(False, "same_node")
        if keep_node_id not in self.nodes or remove_node_id not in self.nodes:
            return VerificationResult(False, "missing_node")
        if vlm_confidence < min_vlm_confidence:
            return VerificationResult(False, "vlm_confidence_too_low")
        keep_score = self._candidate_visual_score(keep_node_id)
        remove_score = self._candidate_visual_score(remove_node_id)
        score = max(keep_score, remove_score)
        if score < min_backend_score and vlm_confidence < 0.90:
            return VerificationResult(False, "backend_score_too_low", score)
        spatial = self.spatial_plausibility_between_nodes(
            keep_node_id,
            remove_node_id,
            max_same_place_baseline_m=max_same_place_baseline_m,
            extended_same_place_baseline_m=extended_same_place_baseline_m,
            max_same_place_hops=max_same_place_hops,
            visual_score=score,
            vlm_confidence=vlm_confidence,
        )
        if not spatial.accepted:
            return VerificationResult(False, f"spatial_plausibility_failed:{spatial.reason}", score, {"spatial_plausibility": spatial.to_dict()})
        keep = self.nodes[keep_node_id]
        rem = self.nodes[remove_node_id]
        if keep.is_critical() and rem.is_critical() and keep.negative_memory != rem.negative_memory:
            return VerificationResult(False, "conflicting_critical_negative_memory", score, {"spatial_plausibility": spatial.to_dict()})
        return VerificationResult(True, "verified_merge_request", score, {"spatial_plausibility": spatial.to_dict()})

    def _cluster_id_for(self, canonical_node_id: str) -> str:
        node = self.nodes[canonical_node_id]
        return str(node.semantic.get("same_place_cluster_id") or f"cluster_{canonical_node_id}")

    def _update_shared_place_cluster(
        self,
        canonical_node_id: str,
        revisit_node_id: str,
        *,
        reason: str,
        frame_index: Optional[int],
    ) -> Dict[str, Any]:
        """Share descriptions/keyframe metadata across same-place nodes without deleting nodes."""
        cluster_id = self._cluster_id_for(canonical_node_id)
        existing = self.same_place_clusters.get(cluster_id, {})
        members = set(existing.get("member_node_ids", []))
        members.update({canonical_node_id, revisit_node_id})
        # Include members already recorded on either node.
        for nid in list(members):
            node = self.nodes.get(nid)
            if not node:
                continue
            members.update(node.semantic.get("same_place_node_ids", []) or [])
        members = {nid for nid in members if nid in self.nodes}

        descriptions: Dict[str, str] = {}
        keyframe_ids: List[str] = []
        image_refs: List[Dict[str, Any]] = []
        for nid in sorted(members):
            node = self.nodes[nid]
            descriptions[nid] = node.short_description()
            for kf in node.keyframes:
                keyframe_ids.append(kf.keyframe_id)
                if kf.image_ref or kf.archived_image_ref:
                    image_refs.append({
                        "node_id": nid,
                        "keyframe_id": kf.keyframe_id,
                        "image_ref": kf.image_ref,
                        "archived_image_ref": kf.archived_image_ref,
                        "active_for_vlm": kf.active_for_vlm,
                        "view_type": kf.view_type,
                    })
                if hasattr(self.visual_index, "update_metadata"):
                    self.visual_index.update_metadata(
                        kf.keyframe_id,
                        {
                            "node_id": nid,
                            "keyframe_id": kf.keyframe_id,
                            "cluster_id": cluster_id,
                            "canonical_node_id": canonical_node_id,
                        },
                    )

        cluster = {
            "cluster_id": cluster_id,
            "canonical_node_id": canonical_node_id,
            "member_node_ids": sorted(members),
            "descriptions": descriptions,
            "keyframe_ids": sorted(set(keyframe_ids)),
            "image_refs": image_refs,
            "last_update_frame_index": frame_index,
            "last_update_reason": reason,
            "merge_policy": "soft_preserve_nodes_share_memory",
        }
        self.same_place_clusters[cluster_id] = cluster

        shared_stub = {
            "cluster_id": cluster_id,
            "canonical_node_id": canonical_node_id,
            "member_node_ids": sorted(members),
            "descriptions": descriptions,
            "keyframe_ids": sorted(set(keyframe_ids)),
        }
        for nid in members:
            node = self.nodes[nid]
            node.semantic["same_place_cluster_id"] = cluster_id
            node.semantic["canonical_place_node_id"] = canonical_node_id
            node.semantic["same_place_node_ids"] = sorted(members)
            node.semantic["shared_place_memory"] = shared_stub
            node.lifecycle["soft_merged"] = True
            node.lifecycle["hard_merged"] = False
            if nid != canonical_node_id:
                node.lifecycle["preserved_as_revisit_node"] = True
                node.lifecycle["canonical_node_id"] = canonical_node_id
        return cluster

    def soft_merge_revisit(
        self,
        canonical_node_id: str,
        revisit_node_id: str,
        *,
        relative_pose_canonical_to_revisit: Optional[RelativePose2D] = None,
        frame_index: Optional[int] = None,
        reason: str = "vlm_confirmed_same_place",
        vlm_confidence: float = 1.0,
    ) -> VerificationResult:
        """Preserve both nodes and connect them with a same-place relative edge.

        This is the preferred revisit merge policy. It keeps parallax/baseline
        evidence intact, creates a loop/revisit constraint from canonical to
        revisit, adds the inverse constraint, and shares image/description
        metadata through a common place cluster.
        """
        if canonical_node_id == revisit_node_id:
            return VerificationResult(True, "same_node_no_soft_link_needed", 1.0)
        if canonical_node_id not in self.nodes or revisit_node_id not in self.nodes:
            return VerificationResult(False, "missing_node_for_soft_merge")
        if self.nodes[canonical_node_id].lifecycle.get("storage_tier") == ARCHIVED:
            return VerificationResult(False, "canonical_node_archived")
        if self.nodes[revisit_node_id].lifecycle.get("storage_tier") == ARCHIVED:
            return VerificationResult(False, "revisit_node_archived")

        query_result: Optional[RelativePoseQueryResult] = None
        pose = relative_pose_canonical_to_revisit
        if pose is None:
            query_result = self.relative_pose_between_nodes(
                canonical_node_id,
                revisit_node_id,
                allow_reverse_edges=True,
                avoid_negative=False,
                include_constraint_edges=False,
                include_archived=False,
            )
            if not query_result.found:
                # Fall back to existing constraint links if this is a repeated
                # same-place association, but do not invent an identity edge.
                query_result = self.relative_pose_between_nodes(
                    canonical_node_id,
                    revisit_node_id,
                    allow_reverse_edges=True,
                    avoid_negative=False,
                    include_constraint_edges=True,
                    include_archived=False,
                )
            if not query_result.found or query_result.relative_pose_src_to_dst is None:
                return VerificationResult(False, "no_relative_path_for_soft_merge", 0.0, {"query": query_result.to_dict() if query_result else None})
            pose = query_result.relative_pose_src_to_dst

        fwd = self.add_or_update_constraint_edge(
            canonical_node_id,
            revisit_node_id,
            pose,
            edge_type="revisit_link",
            relation_type="same_place_soft_merge",
            status="same_place_constraint",
            frame_index=frame_index,
        )
        rev = self.add_or_update_constraint_edge(
            revisit_node_id,
            canonical_node_id,
            pose.inverse(),
            edge_type="revisit_link_reverse",
            relation_type="same_place_soft_merge_inverse",
            status="same_place_constraint",
            frame_index=frame_index,
        )
        self.edges[fwd].evidence.update({
            "reason": reason,
            "vlm_confidence": round(float(vlm_confidence), 4),
            "source": "soft_merge_revisit",
            "chain_rule_query": query_result.to_dict() if query_result else None,
            "non_navigational_constraint": True,
        })
        self.edges[rev].evidence.update({
            "reason": reason,
            "vlm_confidence": round(float(vlm_confidence), 4),
            "source": "soft_merge_revisit_inverse",
            "inverse_of_edge_id": fwd,
            "non_navigational_constraint": True,
        })

        cluster = self._update_shared_place_cluster(canonical_node_id, revisit_node_id, reason=reason, frame_index=frame_index)
        can = self.nodes[canonical_node_id]
        rev_node = self.nodes[revisit_node_id]
        can.visit_count += 1
        can.last_seen_frame_index = max(can.last_seen_frame_index, frame_index or can.last_seen_frame_index)
        rev_node.last_seen_frame_index = max(rev_node.last_seen_frame_index, frame_index or rev_node.last_seen_frame_index)
        rev_node.lifecycle["vlm_place_recognition_pending"] = False
        rev_node.lifecycle["provisional"] = False
        rev_node.navigation_state["loop_status"] = "revisit_linked"
        can.navigation_state["loop_status"] = "revisited"

        if self.current_node_id == revisit_node_id:
            # Use the canonical node for memory retrieval/planning, but keep the
            # physical robot's baseline from the canonical node as live pose.
            self.previous_node_id = revisit_node_id
            self.current_node_id = canonical_node_id
            self.update_current_pose_relation_to_latest_node(
                canonical_node_id,
                pose,
                frame_index=frame_index,
                source="soft_merge_revisit_canonical_to_robot",
            )
        self.loop_history.append(canonical_node_id)
        self.loop_history = self.loop_history[-20:]
        self._log(
            "soft_merge_revisit",
            canonical_node_id=canonical_node_id,
            revisit_node_id=revisit_node_id,
            forward_edge_id=fwd,
            reverse_edge_id=rev,
            reason=reason,
            relative_pose=pose.to_dict(),
            cluster_id=cluster["cluster_id"],
        )
        return VerificationResult(
            True,
            "soft_merge_revisit_linked",
            1.0,
            {
                "canonical_node_id": canonical_node_id,
                "revisit_node_id": revisit_node_id,
                "forward_edge_id": fwd,
                "reverse_edge_id": rev,
                "relative_pose_canonical_to_revisit": pose.to_dict(),
                "chain_rule_query": query_result.to_dict() if query_result else None,
                "cluster": cluster,
            },
        )

    def merge_nodes(self, keep_node_id: str, remove_node_id: str, reason: str = "visual_loop_closure") -> None:
        """Merge two nodes after backend verification.

        Relative edge constraints are preserved. Contradictory edge constraints are
        **not** optimized here; pose-graph optimization remains a TODO.
        """
        if keep_node_id == remove_node_id or keep_node_id not in self.nodes or remove_node_id not in self.nodes:
            return
        keep = self.nodes[keep_node_id]
        rem = self.nodes[remove_node_id]
        if rem.lifecycle.get("storage_tier") == ARCHIVED:
            return
        keep.visit_count += rem.visit_count
        keep.last_seen_frame_index = max(keep.last_seen_frame_index, rem.last_seen_frame_index)
        keep.last_seen_timestamp_ms = max(keep.last_seen_timestamp_ms, rem.last_seen_timestamp_ms)
        if rem.negative_memory and not keep.negative_memory:
            keep.negative_memory = dict(rem.negative_memory)
        if rem.is_critical():
            keep.navigation_state["memory_importance"] = "critical"
            keep.navigation_state["risk_level"] = "high"
        if rem.semantic.get("object_belief") and not keep.semantic.get("object_belief"):
            keep.semantic["object_belief"] = rem.semantic["object_belief"]
        for kf in rem.keyframes:
            kf.node_id = keep_node_id
            keep.keyframes.append(kf)
            if hasattr(self.visual_index, "update_metadata"):
                self.visual_index.update_metadata(kf.keyframe_id, {"node_id": keep_node_id, "keyframe_id": kf.keyframe_id})
        # Redirect edges from/to remove node.
        for edge in self.edges.values():
            if edge.src_node_id == remove_node_id:
                edge.src_node_id = keep_node_id
            if edge.dst_node_id == remove_node_id:
                edge.dst_node_id = keep_node_id
        rem.lifecycle.update({
            "storage_tier": ARCHIVED,
            "vlm_visible": False,
            "can_retrieve_image_for_vlm": False,
            "compression_level": "merged_tombstone",
            "merged_into": keep_node_id,
            "merge_reason": reason,
        })
        rem.keyframes = []
        self.rebuild_adjacency()
        if self.current_node_id == remove_node_id:
            self.current_node_id = keep_node_id
        if self.previous_node_id == remove_node_id:
            self.previous_node_id = keep_node_id
        if self.latest_node_id_for_live_pose == remove_node_id:
            self.latest_node_id_for_live_pose = keep_node_id
        self.loop_history = [keep_node_id if x == remove_node_id else x for x in self.loop_history]
        self._log("merge_nodes", keep_node_id=keep_node_id, remove_node_id=remove_node_id, reason=reason)

    # ------------------------------------------------------------------
    # Compression / marginalization
    # ------------------------------------------------------------------
    def compress_node(self, node_id: str, *, reason: str = "manual_compress") -> None:
        node = self.nodes.get(node_id)
        if not node or node.lifecycle.get("storage_tier") == ARCHIVED:
            return
        if node.is_critical():
            node.lifecycle.update({"storage_tier": COMPRESSED, "vlm_visible": True, "can_retrieve_image_for_vlm": False, "compression_level": "summary_embedding_negative"})
        else:
            node.lifecycle.update({"storage_tier": COMPRESSED, "vlm_visible": False, "can_retrieve_image_for_vlm": False, "compression_level": "summary_embedding_only"})
        for kf in node.keyframes:
            if kf.image_ref and not kf.archived_image_ref:
                kf.archived_image_ref = kf.image_ref
            kf.active_for_vlm = False
            kf.storage_tier = COMPRESSED
            kf.image_ref = None
            kf.thumbnail_ref = None
        self._log("compress_node", node_id=node_id, reason=reason)

    def compress_old_nodes(self, *, current_frame_index: int, hot_window_frames: int = 300, max_hot_nodes: int = 12) -> None:
        """Compress non-critical old nodes.

        This mimics marginalization: active images leave the VLM prompt, but
        embeddings, summaries, relative constraints, and negative memories stay.
        """
        active_nodes = [n for n in self.nodes.values() if n.lifecycle.get("storage_tier") not in {ARCHIVED}]
        hot_candidates = sorted(
            [n for n in active_nodes if n.lifecycle.get("storage_tier") == HOT],
            key=lambda n: n.last_seen_frame_index,
            reverse=True,
        )
        keep_hot = {n.node_id for n in hot_candidates[:max_hot_nodes]}
        for node in active_nodes:
            if node.node_id == self.current_node_id or node.node_id in keep_hot:
                continue
            old_enough = current_frame_index - node.last_seen_frame_index > hot_window_frames
            too_many_hot = node.lifecycle.get("storage_tier") == HOT and node.node_id not in keep_hot
            if old_enough or too_many_hot:
                self.compress_node(node.node_id, reason="hot_window_or_capacity")

    # ------------------------------------------------------------------
    # Planning and context
    # ------------------------------------------------------------------
    def outgoing_edges(self, node_id: str, *, include_constraint_edges: bool = False) -> List[MemoryEdge]:
        return [
            self.edges[eid]
            for eid in self.out_edges.get(node_id, [])
            if eid in self.edges
            and self.edges[eid].src_node_id == node_id
            and self.edges[eid].dst_node_id != node_id
            and (include_constraint_edges or self.edges[eid].edge_type not in CONSTRAINT_EDGE_TYPES)
        ]

    def incoming_edges(self, node_id: str, *, include_constraint_edges: bool = False) -> List[MemoryEdge]:
        return [
            self.edges[eid]
            for eid in self.in_edges.get(node_id, [])
            if eid in self.edges
            and self.edges[eid].dst_node_id == node_id
            and self.edges[eid].src_node_id != node_id
            and (include_constraint_edges or self.edges[eid].edge_type not in CONSTRAINT_EDGE_TYPES)
        ]

    def latest_incoming_edge_id(self, node_id: Optional[str] = None) -> Optional[str]:
        node_id = node_id or self.current_node_id
        if not node_id:
            return None
        incoming = self.incoming_edges(node_id)
        if not incoming:
            return None
        incoming.sort(key=lambda e: int(e.traversal.get("last_attempt_frame") or -1), reverse=True)
        return incoming[0].edge_id

    def is_looping(self, window: int = 8, repeat_threshold: int = 3) -> Tuple[bool, int]:
        hist = self.loop_history[-window:]
        if not hist:
            return False, 0
        counts = {nid: hist.count(nid) for nid in set(hist)}
        repeated = max(counts.values()) if counts else 0
        return repeated >= repeat_threshold, repeated

    def dijkstra_path(self, start_node_id: str, goal_node_id: str, avoid_negative: bool = True) -> Optional[List[str]]:
        """Shortest path over the current topological graph."""
        if start_node_id not in self.nodes or goal_node_id not in self.nodes:
            return None
        q: List[Tuple[float, str, List[str]]] = [(0.0, start_node_id, [start_node_id])]
        best: Dict[str, float] = {start_node_id: 0.0}
        while q:
            cost, node_id, path = heapq.heappop(q)
            if node_id == goal_node_id:
                return path
            if cost > best.get(node_id, math.inf):
                continue
            for edge in self.outgoing_edges(node_id):
                if avoid_negative and edge.is_negative():
                    continue
                edge.update_cost()
                nxt = edge.dst_node_id
                if self.nodes.get(nxt) and self.nodes[nxt].lifecycle.get("storage_tier") == ARCHIVED:
                    continue
                new_cost = cost + float(edge.planning_cost.get("final_cost", 1.0))
                if new_cost < best.get(nxt, math.inf):
                    best[nxt] = new_cost
                    heapq.heappush(q, (new_cost, nxt, path + [nxt]))
        return None

    def compose_path_pose(self, path: Sequence[str]) -> Optional[RelativePose2D]:
        """Compose edge poses along a directed node path.

        This is the simple chain rule for a path whose directed edges all exist.
        For arbitrary node pairs where the path may need reverse traversal, use
        :meth:`relative_pose_between_nodes`.
        """
        if len(path) < 2:
            return RelativePose2D()
        acc = RelativePose2D()
        for src, dst in zip(path[:-1], path[1:]):
            edge_id = self.find_edge(src, dst)
            if not edge_id:
                return None
            acc = acc.compose(self.edges[edge_id].relative_pose_src_to_dst)
        return acc

    def _relative_query_neighbors(
        self,
        node_id: str,
        *,
        allow_reverse_edges: bool,
        avoid_negative: bool,
        include_constraint_edges: bool,
        include_archived: bool,
    ) -> Iterable[Tuple[str, MemoryEdge, RelativePose2D, str, float]]:
        """Yield graph neighbors and the transform from ``node_id`` to neighbor."""
        for edge in self.outgoing_edges(node_id, include_constraint_edges=include_constraint_edges):
            if avoid_negative and edge.is_negative():
                continue
            if not include_constraint_edges and edge.edge_type in CONSTRAINT_EDGE_TYPES:
                continue
            if not include_archived and self.nodes.get(edge.dst_node_id, MemoryNode("missing")).lifecycle.get("storage_tier") == ARCHIVED:
                continue
            edge.update_cost()
            cost = float(edge.planning_cost.get("final_cost", 1.0))
            if edge.edge_type in CONSTRAINT_EDGE_TYPES:
                cost = max(0.01, 0.2 * cost)
            yield edge.dst_node_id, edge, edge.relative_pose_src_to_dst, "forward", cost

        if not allow_reverse_edges:
            return
        for edge in self.incoming_edges(node_id, include_constraint_edges=include_constraint_edges):
            if avoid_negative and edge.is_negative():
                continue
            if not include_constraint_edges and edge.edge_type in CONSTRAINT_EDGE_TYPES:
                continue
            if not include_archived and self.nodes.get(edge.src_node_id, MemoryNode("missing")).lifecycle.get("storage_tier") == ARCHIVED:
                continue
            edge.update_cost()
            cost = float(edge.planning_cost.get("final_cost", 1.0)) + 0.05
            if edge.edge_type in CONSTRAINT_EDGE_TYPES:
                cost = max(0.01, 0.2 * cost)
            yield edge.src_node_id, edge, edge.relative_pose_src_to_dst.inverse(), "reverse", cost

    def relative_pose_between_nodes(
        self,
        src_node_id: str,
        dst_node_id: str,
        *,
        allow_reverse_edges: bool = True,
        avoid_negative: bool = False,
        include_constraint_edges: bool = True,
        include_archived: bool = False,
        max_hops: int = 30,
    ) -> RelativePoseQueryResult:
        """Compute ``T_src_to_dst`` by composing relative edge transforms.

        This is not pose-graph optimization. It selects one low-cost graph path
        and applies the SE(2) chain rule along that path. Reverse traversal uses
        the inverse of a stored directed edge.
        """
        if src_node_id not in self.nodes:
            return RelativePoseQueryResult(False, src_node_id, dst_node_id, reason="src_node_missing")
        if dst_node_id not in self.nodes:
            return RelativePoseQueryResult(False, src_node_id, dst_node_id, reason="dst_node_missing")
        if src_node_id == dst_node_id:
            return RelativePoseQueryResult(
                True,
                src_node_id,
                dst_node_id,
                RelativePose2D(),
                [src_node_id],
                [],
                0.0,
                "same_node_identity",
            )

        counter = 0
        start = RelativePose2D()
        queue: List[Tuple[float, int, int, str, RelativePose2D, List[str], List[Dict[str, Any]]]] = [
            (0.0, 0, counter, src_node_id, start, [src_node_id], [])
        ]
        best: Dict[str, Tuple[float, int]] = {src_node_id: (0.0, 0)}

        while queue:
            cost, hops, _, node_id, pose_src_to_node, node_path, edge_path = heapq.heappop(queue)
            if node_id == dst_node_id:
                return RelativePoseQueryResult(
                    True,
                    src_node_id,
                    dst_node_id,
                    pose_src_to_node,
                    node_path,
                    edge_path,
                    cost,
                    "composed_graph_path",
                )
            if hops >= max_hops:
                continue
            best_cost, best_hops = best.get(node_id, (math.inf, 10**9))
            if cost > best_cost + 1e-9 and hops >= best_hops:
                continue
            for nxt, edge, transform_node_to_next, direction, step_cost in self._relative_query_neighbors(
                node_id,
                allow_reverse_edges=allow_reverse_edges,
                avoid_negative=avoid_negative,
                include_constraint_edges=include_constraint_edges,
                include_archived=include_archived,
            ):
                if nxt in node_path:
                    continue
                new_cost = cost + step_cost
                new_hops = hops + 1
                prev_best = best.get(nxt)
                if prev_best is not None and new_cost >= prev_best[0] and new_hops >= prev_best[1]:
                    continue
                new_pose = pose_src_to_node.compose(transform_node_to_next)
                counter += 1
                best[nxt] = (new_cost, new_hops)
                heapq.heappush(
                    queue,
                    (
                        new_cost,
                        new_hops,
                        counter,
                        nxt,
                        new_pose,
                        node_path + [nxt],
                        edge_path + [
                            {
                                "edge_id": edge.edge_id,
                                "src_node_id": edge.src_node_id,
                                "dst_node_id": edge.dst_node_id,
                                "direction_used": direction,
                                "edge_type": edge.edge_type,
                                "relation_type": edge.relation_type,
                            }
                        ],
                    ),
                )

        return RelativePoseQueryResult(False, src_node_id, dst_node_id, reason="no_graph_path")

    def relative_pose_between_nodes_dict(self, src_node_id: str, dst_node_id: str, **kwargs: Any) -> Dict[str, Any]:
        """JSON-friendly wrapper around :meth:`relative_pose_between_nodes`."""
        return self.relative_pose_between_nodes(src_node_id, dst_node_id, **kwargs).to_dict()

    def score_candidate_exit(self, edge: MemoryEdge, goal_bearing_deg: float) -> Dict[str, Any]:
        """Score one outgoing edge using the current robot pose relative to the node.

        Edge poses are stored in the source-node frame. For action selection, the
        VLM needs the bearing in the **current robot frame**. When the source edge
        belongs to the current/latest node, the live floating pose
        ``T_latest_node_to_robot`` is inverted and composed with the edge pose.
        """
        robot_frame_pose = self.robot_frame_pose_for_edge(edge)
        chord_bearing = normalize_angle_deg(
            math.degrees(math.atan2(robot_frame_pose.dy_m, robot_frame_pose.dx_m))
        )
        bearing = chord_bearing
        bearing_source = "relative_pose_chord"
        path_guidance: Optional[Dict[str, Any]] = None
        raw_polyline = edge.evidence.get("path_polyline_src_xy")
        if (
            isinstance(raw_polyline, list)
            and len(raw_polyline) >= 2
            and self.current_node_id
            and edge.src_node_id == self.current_node_id
        ):
            node_points: List[RelativePose2D] = []
            for raw_point in raw_polyline:
                if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 2:
                    continue
                try:
                    node_points.append(
                        RelativePose2D(
                            dx_m=float(raw_point[0]),
                            dy_m=float(raw_point[1]),
                            dyaw_deg=0.0,
                        )
                    )
                except (TypeError, ValueError):
                    continue
            if len(node_points) >= 2:
                robot_points = [
                    (
                        self.latest_node_to_robot_pose.inverse().compose(point)
                        if self.latest_node_id_for_live_pose == self.current_node_id
                        else point
                    )
                    for point in node_points
                ]
                nearest_index = min(
                    range(len(robot_points)),
                    key=lambda index: robot_points[index].distance_m(),
                )
                try:
                    lookahead_m = max(
                        0.2,
                        float(edge.evidence.get("path_guidance_lookahead_m", 0.6)),
                    )
                except (TypeError, ValueError):
                    lookahead_m = 0.6
                selected_index = nearest_index
                traversed_m = 0.0
                for index in range(nearest_index + 1, len(node_points)):
                    previous = node_points[index - 1]
                    current = node_points[index]
                    traversed_m += math.hypot(
                        current.dx_m - previous.dx_m,
                        current.dy_m - previous.dy_m,
                    )
                    selected_index = index
                    if traversed_m >= lookahead_m:
                        break
                selected_pose = robot_points[selected_index]
                if selected_pose.distance_m() >= 0.05:
                    bearing = normalize_angle_deg(
                        math.degrees(
                            math.atan2(selected_pose.dy_m, selected_pose.dx_m)
                        )
                    )
                    bearing_source = "executed_path_polyline"
                    path_guidance = {
                        "nearest_polyline_index": int(nearest_index),
                        "selected_polyline_index": int(selected_index),
                        "lookahead_m": round(float(lookahead_m), 3),
                        "traversed_polyline_m": round(float(traversed_m), 3),
                        "selected_point_robot_xy": [
                            round(float(selected_pose.dx_m), 4),
                            round(float(selected_pose.dy_m), 4),
                        ],
                        "polyline_point_count": len(node_points),
                    }
        path_bearing_node = edge.evidence.get("departure_bearing_src_deg")
        try:
            path_bearing_node = float(path_bearing_node)
        except (TypeError, ValueError):
            path_bearing_node = None
        if (
            path_guidance is None
            and path_bearing_node is not None
            and math.isfinite(path_bearing_node)
            and self.current_node_id
            and edge.src_node_id == self.current_node_id
        ):
            live_heading_offset = 0.0
            if self.latest_node_id_for_live_pose == self.current_node_id:
                live_heading_offset = float(self.latest_node_to_robot_pose.dyaw_deg)
            bearing = normalize_angle_deg(path_bearing_node - live_heading_offset)
            bearing_source = str(
                edge.evidence.get("departure_bearing_source")
                or "executed_path_tangent"
            )
        goal_alignment = 1.0 - min(abs(normalize_angle_deg(bearing - goal_bearing_deg)), 180.0) / 180.0
        deadlock_risk = 1.0 if edge.is_negative() else 0.0
        status = str(edge.traversal.get("status", "unknown"))
        exploration_value = 0.7 if status == "unknown" else 0.2
        if status == "success":
            exploration_value = 0.35
        if status == "escape_success":
            exploration_value = 0.45
        final_score = 0.55 * goal_alignment + 0.25 * exploration_value - 0.9 * deadlock_risk
        return {
            "edge_id": edge.edge_id,
            "dst_node_id": edge.dst_node_id,
            "edge_type": edge.edge_type,
            "relation_type": edge.relation_type,
            "bearing_deg_robot": round(bearing, 3),
            "bearing_source": bearing_source,
            "chord_bearing_deg_robot": round(chord_bearing, 3),
            "path_departure_bearing_node_deg": (
                round(path_bearing_node, 3)
                if path_bearing_node is not None
                else None
            ),
            "path_guidance": path_guidance,
            "view_type_hint": nearest_view_type(bearing),
            "status": status,
            "avoid": edge.is_negative(),
            "goal_alignment": round(goal_alignment, 4),
            "deadlock_risk": round(deadlock_risk, 4),
            "exploration_value": round(exploration_value, 4),
            "score": round(final_score, 4),
            "reason": "known negative branch" if edge.is_negative() else "known graph branch",
            "relative_pose_node_to_dst": edge.relative_pose_src_to_dst.to_dict(),
            "relative_pose_robot_to_dst": robot_frame_pose.to_dict(),
            "pose_relation_used": self.current_pose_relation_to_latest_node(),
        }

    def _object_context(self, target_object: Optional[str]) -> Dict[str, Any]:
        if not target_object:
            return {"enabled": False}
        candidate_nodes = []
        for node in self.nodes.values():
            if node.lifecycle.get("storage_tier") == ARCHIVED:
                continue
            belief = node.semantic.get("object_belief") or {}
            if not belief:
                continue
            score = 0.0
            if belief.get("target_object") == target_object:
                score += 0.4
            if belief.get("seen_target"):
                score += 1.0
            priors = belief.get("room_object_prior") or {}
            if isinstance(priors, dict):
                score += float(priors.get(target_object, 0.0) or 0.0)
            candidate_nodes.append({
                "node_id": node.node_id,
                "description": node.short_description(),
                "score": round(score, 4),
                "belief": belief,
                "has_active_image": bool(node.active_keyframe()),
            })
        candidate_nodes.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return {
            "enabled": True,
            "target_object": target_object,
            "candidate_object_nodes": candidate_nodes[:8],
            "policy": "use object belief as a soft prior; still require visible navigable fine goal before go",
        }

    def build_vlm_memory_context(
        self,
        *,
        goal_bearing_deg: float,
        goal_distance_m: float,
        task_mode: str = "PointNav",
        target_object: Optional[str] = None,
        max_memory_images: int = 4,
        force_front_view_waypoint: bool = True,
    ) -> Dict[str, Any]:
        """Build compact graph-derived memory context for the VLM input."""
        current_node = self.nodes.get(self.current_node_id) if self.current_node_id else None
        current_pose_relation = self.current_pose_relation_to_latest_node()
        is_looping, repeated_count = self.is_looping()
        loc = self.last_localization or LocalizationResult("new_place", None, 0.0, [], [])
        outgoing = self.outgoing_edges(current_node.node_id) if current_node else []
        scored_edges = [self.score_candidate_exit(e, goal_bearing_deg) for e in outgoing]
        scored_edges.sort(key=lambda x: x["score"], reverse=True)

        known_view_types = {c["view_type_hint"] for c in scored_edges}
        frontier_candidates: List[Dict[str, Any]] = []
        for vt in ["front", "left", "right", "back"]:
            if vt in known_view_types:
                continue
            bearing = view_type_to_heading_deg(vt)
            align = 1.0 - min(abs(normalize_angle_deg(bearing - goal_bearing_deg)), 180.0) / 180.0
            frontier_candidates.append({
                "exit_id": f"frontier_{vt}",
                "edge_id": None,
                "view_type_hint": vt,
                "bearing_deg_robot": bearing,
                "status": "unknown_frontier",
                "avoid": False,
                "goal_alignment": round(align, 4),
                "score": round(0.55 * align + 0.25 * 0.75, 4),
                "reason": "unvisited or unmodeled local view",
            })
        candidate_exits = sorted(scored_edges + frontier_candidates, key=lambda x: x["score"], reverse=True)[:8]
        # GaP-lite: give the VLM stable, typed references to select instead of
        # free-form edge/node ids.  These refs are valid only for the current
        # VLM input snapshot and are verified again by the backend before use.
        for i, cand in enumerate(candidate_exits, start=1):
            cand.setdefault("candidate_ref", f"exit_{i:03d}")
            cand.setdefault("candidate_type", "candidate_exit")

        revisit_candidates = self.get_revisit_candidates(max_candidates=4)
        for i, cand in enumerate(revisit_candidates, start=1):
            cand.setdefault("candidate_ref", f"revisit_{i:03d}")
            cand.setdefault("candidate_type", "revisit_candidate")

        candidate_refs = {
            "exits": [
                {
                    "candidate_ref": c.get("candidate_ref"),
                    "candidate_type": "candidate_exit",
                    "edge_id": c.get("edge_id"),
                    "edge_type": c.get("edge_type"),
                    "relation_type": c.get("relation_type"),
                    "exit_id": c.get("exit_id"),
                    "dst_node_id": c.get("dst_node_id"),
                    "view_type_hint": c.get("view_type_hint"),
                    "bearing_deg_robot": c.get("bearing_deg_robot"),
                    "bearing_source": c.get("bearing_source"),
                    "chord_bearing_deg_robot": c.get("chord_bearing_deg_robot"),
                    "path_departure_bearing_node_deg": c.get(
                        "path_departure_bearing_node_deg"
                    ),
                    "path_guidance": c.get("path_guidance"),
                    "status": c.get("status"),
                    "avoid": c.get("avoid"),
                    "score": c.get("score"),
                    "reason": c.get("reason"),
                }
                for c in candidate_exits
            ],
            "revisits": [
                {
                    "candidate_ref": c.get("candidate_ref"),
                    "candidate_type": "revisit_candidate",
                    "node_id": c.get("candidate_node_id"),
                    "candidate_node_id": c.get("candidate_node_id"),
                    "visual_retrieval_score": c.get("visual_retrieval_score"),
                    "topological_consistency": c.get("topological_consistency"),
                    "spatial_plausibility": c.get("spatial_plausibility"),
                    "recommended_vlm_judgement": c.get("recommended_vlm_judgement"),
                }
                for c in revisit_candidates
            ],
            "policy": "VLM must select candidate_ref values provided in this snapshot; backend rejects unknown refs.",
        }

        memory_images: List[Dict[str, Any]] = []
        # Images for VLM place recognition verification first.
        for cand in revisit_candidates:
            if len(memory_images) >= max_memory_images:
                break
            image_ref = cand.get("candidate_image_ref")
            if image_ref:
                memory_images.append({
                    "memory_image_id": f"mi_revisit_{cand['candidate_node_id']}",
                    "node_id": cand["candidate_node_id"],
                    "keyframe_id": (cand.get("candidate_keyframe_ids") or [None])[0],
                    "view_type": cand.get("candidate_view_type"),
                    "image_ref": image_ref,
                    "caption": cand.get("semantic_summary"),
                    "reason_for_inclusion": "revisit_candidate_verification",
                    "storage_tier": HOT,
                })

        image_nodes: List[MemoryNode] = []
        if current_node:
            image_nodes.append(current_node)
            for e in outgoing:
                if e.dst_node_id in self.nodes:
                    image_nodes.append(self.nodes[e.dst_node_id])
        for n in self.nodes.values():
            if n.negative_memory and n not in image_nodes:
                image_nodes.append(n)
        for node in image_nodes:
            if len(memory_images) >= max_memory_images:
                break
            if not node.lifecycle.get("can_retrieve_image_for_vlm", True):
                continue
            for kf in node.keyframes:
                if len(memory_images) >= max_memory_images:
                    break
                if not (kf.active_for_vlm and kf.image_ref):
                    continue
                if any(mi.get("keyframe_id") == kf.keyframe_id for mi in memory_images):
                    continue
                reason = "current_or_neighbor_place"
                if node.negative_memory:
                    reason = "nearby_or_relevant_negative_memory"
                memory_images.append({
                    "memory_image_id": f"mi_{kf.keyframe_id}",
                    "node_id": node.node_id,
                    "keyframe_id": kf.keyframe_id,
                    "view_type": kf.view_type,
                    "image_ref": kf.image_ref,
                    "caption": kf.caption or node.short_description(),
                    "reason_for_inclusion": reason,
                    "storage_tier": kf.storage_tier,
                })

        compressed_negative = []
        for node in self.nodes.values():
            if node.negative_memory:
                compressed_negative.append({
                    "node_id": node.node_id,
                    "summary": node.short_description(),
                    "has_active_image": any(kf.active_for_vlm and kf.image_ref for kf in node.keyframes),
                    "negative_memory": node.negative_memory,
                    "severity": node.negative_memory.get("severity", 0.0),
                })

        deadlock_state = {
            "current_node_id": self.current_node_id,
            "status": current_node.navigation_state.get("deadlock_status", "none") if current_node else "none",
            "incoming_edge_id": self.latest_incoming_edge_id(self.current_node_id) if self.current_node_id else None,
            "policy": "suspected->scan_or_rotate, confirmed->avoid_incoming_edge, escaped->resume_goal",
        }

        return {
            # Original v1-compatible fields remain present.
            "visited_nodes": [n.to_context(include_images=False) for n in list(self.nodes.values())[-8:] if n.lifecycle.get("storage_tier") != ARCHIVED],
            "failed_waypoints": [self.edges[eid].to_context() for ids in self.negative_edge_index.values() for eid in ids if eid in self.edges][-8:],
            "last_selected_view": None,
            "last_selected_point": None,
            "loop_warning": {"is_looping": bool(is_looping), "repeated_branch_count": int(repeated_count)},
            # Extensions used by the framework.
            "schema_version": "nav_memory_context_v6",
            "graph_summary": {
                "num_nodes": len([n for n in self.nodes.values() if n.lifecycle.get("storage_tier") != ARCHIVED]),
                "num_place_nodes": len([
                    n
                    for n in self.nodes.values()
                    if n.lifecycle.get("storage_tier") != ARCHIVED
                    and n.node_type != "failed_frontier"
                ]),
                "num_failed_frontier_nodes": len([
                    n
                    for n in self.nodes.values()
                    if n.lifecycle.get("storage_tier") != ARCHIVED
                    and n.node_type == "failed_frontier"
                ]),
                "num_edges": len(self.edges),
                "num_deadlock_edges": sum(1 for e in self.edges.values() if e.is_negative()),
                "num_compressed_nodes": sum(1 for n in self.nodes.values() if n.lifecycle.get("storage_tier") == COMPRESSED),
                "pose_graph_optimization": "TODO_not_enabled",
                "merge_policy": "soft_preserve_revisit_nodes",
                "gap_lite": "enabled",
            },
            "control_policy": {
                "fine_goal_execution_policy": "front_only_rotate_then_reobserve" if force_front_view_waypoint else "multi_view_waypoint_allowed",
                "memory_writes_are_requests": True,
                "place_recognition_commit_policy": "vlm_confirms_backend_verifies",
                "merge_policy": "soft_preserve_revisit_nodes_with_same_place_relative_edges",
                "candidate_ref_policy": "VLM selects backend-provided candidate_ref; backend validates before execution or memory mutation",
            },
            "policy_harness_state": {
                "schema_version": "nav_policy_harness_lite_v1",
                "policy_graph_mode": "GaP-lite_static_sidecar_not_full_interpreter",
                "current_stage": "escape_deadlock" if current_node and current_node.navigation_state.get("deadlock_status") in {"suspected", "confirmed", "escaping"} else ("verify_revisit" if revisit_candidates else "normal_goal_seek"),
                "enabled_features": ["candidate_ref_selection", "validation_checkpoints", "memory_op_candidate_ref_resolution", "validation_feedback_logging", "nav_skill_cards"],
                "false_positive_merge_policy": "prefer_duplicate_node_over_false_positive_merge",
            },
            "nav_skill_cards": NAV_SKILL_CARDS,
            "pose_graph_optimization": {
                "enabled": False,
                "status": "TODO",
                "note": "No pose graph optimization is executed in this VLM-prompting MVP.",
            },
            "current_pose_relation_to_latest_node": current_pose_relation,
            "spatial_context": {
                "current_pose_relation_to_latest_node": current_pose_relation,
                "policy": "candidate exits are transformed from latest-node frame into current robot frame before scoring",
            },
            "current_localization": {
                "current_node_id": self.current_node_id,
                "match_status": loc.match_status,
                "match_confidence": round(float(loc.match_confidence), 4),
                "candidate_node_ids": loc.candidate_node_ids,
                "matched_keyframe_ids": loc.matched_keyframe_ids,
                "candidate_scores": {k: round(float(v), 4) for k, v in loc.candidate_scores.items()},
                "revisit_likelihood": round(float(loc.match_confidence), 4),
                "final_place_recognition_policy": "VLM_verifies_backend_candidates_backend_commits",
            },
            "place_recognition": {
                "policy": "backend_retrieval_proposes_candidates; VLM compares current image with candidate memory image/summary; backend verifies and commits confirm_revisit_node/request_merge_nodes",
                "vlm_should_output_memory_ops": ["confirm_revisit_node", "reject_revisit_candidate", "request_merge_nodes", "request_observation"],
                "revisit_candidates": revisit_candidates,
                "false_merge_warning": "prefer uncertain/request_observation over false positive merge",
                "merge_policy": "soft merge: preserve first/revisit nodes, add revisit_link relative constraint, share image/description cluster",
            },
            "same_place_clusters": list(self.same_place_clusters.values())[-8:],
            "goal_context": {
                "task_mode": task_mode,
                "target_object": target_object,
                "goal_bearing_from_current_deg": round(float(goal_bearing_deg), 3),
                "goal_distance_m": round(float(goal_distance_m), 3),
                "detour_status": "escaping_deadlock" if current_node and current_node.navigation_state.get("deadlock_status") in {"suspected", "confirmed", "escaping"} else "normal_goal_seek",
                "goal_resume_hint": "avoid known negative edges; after escape re-align toward goal bearing or ObjNav object belief",
            },
            "deadlock_state": deadlock_state,
            "candidate_refs": candidate_refs,
            "local_topology": {
                "current_node": current_node.to_context(include_images=False) if current_node else None,
                "candidate_exits": candidate_exits,
            },
            "retrieved_memory_images": memory_images,
            "revisit_candidates": revisit_candidates,
            "compressed_negative_memories": compressed_negative[-8:],
            "object_context": self._object_context(target_object if task_mode == "ObjNav" or target_object else None),
        }

    # ------------------------------------------------------------------
    # Pose-graph optimization placeholder
    # ------------------------------------------------------------------
    def optimize_pose_graph(self) -> None:
        """TODO: future backend hook for loop-closure/merge pose-graph optimization.

        The current framework relies on relative-edge constraints plus VLM/backend
        verification. No algorithmic pose graph optimizer is executed here.
        """
        raise NotImplementedError("Pose graph optimization is intentionally TODO and not enabled in this VLM-prompting MVP.")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "relative_topometric_memory_graph_v6",
            "graph_id": self.graph_id,
            "pose_policy": {
                "node_global_pose_stored": False,
                "edge_pose_type": "relative_SE2",
                "edge_pose_direction": "src_to_dst",
                "supports_path_composition": True,
                "pose_graph_optimization": "TODO_not_enabled",
            },
            "nodes": {nid: asdict(node) for nid, node in self.nodes.items()},
            "edges": {
                eid: {
                    **{k: v for k, v in asdict(edge).items() if k != "relative_pose_src_to_dst"},
                    "relative_pose_src_to_dst": edge.relative_pose_src_to_dst.to_dict(),
                }
                for eid, edge in self.edges.items()
            },
            "current_node_id": self.current_node_id,
            "previous_node_id": self.previous_node_id,
            "runtime_current_pose_relation_to_latest_node": self.current_pose_relation_to_latest_node(),
            "same_place_clusters": self.same_place_clusters,
            "negative_edge_index": self.negative_edge_index,
            "event_log": self.event_log[-200:],
        }
