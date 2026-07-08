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


HOT = "hot"
WARM = "warm"
COMPRESSED = "compressed"
ARCHIVED = "archived"

NEGATIVE_EDGE_STATUSES = {"deadlock_entry", "deadlock_entry_candidate", "blocked", "risky"}
DEADLOCK_STATUSES = {"none", "suspected", "confirmed", "escaping", "confirmed_escaped"}


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
        return {"uncertainty": "unknown", "source": "no_direct_edge"}

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
            out.append({
                "candidate_node_id": node_id,
                "candidate_keyframe_ids": candidate_keyframes,
                "candidate_image_ref": active_kf.image_ref if active_kf else None,
                "candidate_view_type": active_kf.view_type if active_kf else None,
                "visual_retrieval_score": round(self._candidate_visual_score(node_id), 4),
                "topological_consistency": self._topological_consistency_label(node_id),
                "relative_pose_hint_from_current": self._relative_pose_hint_between_current_and(node_id),
                "semantic_summary": node.short_description(),
                "negative_memory": node.negative_memory,
                "object_belief": node.semantic.get("object_belief"),
                "recommended_vlm_judgement": "same_place_if_layout_and_openings_match_else_uncertain",
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
    ) -> VerificationResult:
        """Verify a VLM ``confirm_revisit_node`` request before committing it."""
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
        return VerificationResult(True, "verified_revisit_candidate", score, {"is_backend_candidate": is_candidate})

    def commit_revisit(
        self,
        candidate_node_id: str,
        *,
        frame_index: int,
        vlm_confidence: float,
        reason: str = "vlm_confirmed_revisit",
        min_vlm_confidence: float = 0.62,
        min_backend_score: float = 0.45,
    ) -> VerificationResult:
        """Commit a VLM-confirmed revisit and optionally merge a provisional node.

        If the current node is a newly created non-critical provisional node, it
        is merged into the verified candidate. This keeps the graph sparse while
        still letting VLM make the final same-place judgement.
        """
        verification = self.verify_revisit_candidate(
            candidate_node_id,
            vlm_confidence=vlm_confidence,
            min_vlm_confidence=min_vlm_confidence,
            min_backend_score=min_backend_score,
        )
        if not verification.accepted:
            self._log("reject_revisit", candidate_node_id=candidate_node_id, reason=verification.reason, score=verification.score)
            return verification
        current = self.current_node_id
        if current and current != candidate_node_id and current in self.nodes:
            cur_node = self.nodes[current]
            provisional = cur_node.visit_count <= 1 and not cur_node.is_critical()
            if provisional:
                self.merge_nodes(candidate_node_id, current, reason=reason)
        self.set_current_node(candidate_node_id, frame_index)
        self.nodes[candidate_node_id].semantic["last_revisit_confirmation"] = {
            "reason": reason,
            "vlm_confidence": round(float(vlm_confidence), 4),
            "backend_visual_score": round(float(verification.score), 4),
        }
        self._log("commit_revisit", candidate_node_id=candidate_node_id, frame_index=frame_index, reason=reason, confidence=vlm_confidence)
        return verification

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
        keep = self.nodes[keep_node_id]
        rem = self.nodes[remove_node_id]
        if keep.is_critical() and rem.is_critical() and keep.negative_memory != rem.negative_memory:
            return VerificationResult(False, "conflicting_critical_negative_memory", score)
        return VerificationResult(True, "verified_merge_request", score)

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
    def outgoing_edges(self, node_id: str) -> List[MemoryEdge]:
        return [
            self.edges[eid]
            for eid in self.out_edges.get(node_id, [])
            if eid in self.edges
            and self.edges[eid].src_node_id == node_id
            and self.edges[eid].dst_node_id != node_id
            and self.edges[eid].edge_type != "internal_merge_tombstone"
        ]

    def incoming_edges(self, node_id: str) -> List[MemoryEdge]:
        return [
            self.edges[eid]
            for eid in self.in_edges.get(node_id, [])
            if eid in self.edges
            and self.edges[eid].dst_node_id == node_id
            and self.edges[eid].src_node_id != node_id
            and self.edges[eid].edge_type != "internal_merge_tombstone"
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
        """Compose edge poses along a node path."""
        if len(path) < 2:
            return RelativePose2D()
        acc = RelativePose2D()
        for src, dst in zip(path[:-1], path[1:]):
            edge_id = self.find_edge(src, dst)
            if not edge_id:
                return None
            acc = acc.compose(self.edges[edge_id].relative_pose_src_to_dst)
        return acc

    def score_candidate_exit(self, edge: MemoryEdge, goal_bearing_deg: float) -> Dict[str, Any]:
        """Score one outgoing edge using the current robot pose relative to the node.

        Edge poses are stored in the source-node frame. For action selection, the
        VLM needs the bearing in the **current robot frame**. When the source edge
        belongs to the current/latest node, the live floating pose
        ``T_latest_node_to_robot`` is inverted and composed with the edge pose.
        """
        robot_frame_pose = self.robot_frame_pose_for_edge(edge)
        bearing = normalize_angle_deg(math.degrees(math.atan2(robot_frame_pose.dy_m, robot_frame_pose.dx_m)))
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
            "bearing_deg_robot": round(bearing, 3),
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

        revisit_candidates = self.get_revisit_candidates(max_candidates=4)

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
            "schema_version": "nav_memory_context_v4",
            "graph_summary": {
                "num_nodes": len([n for n in self.nodes.values() if n.lifecycle.get("storage_tier") != ARCHIVED]),
                "num_edges": len(self.edges),
                "num_deadlock_edges": sum(1 for e in self.edges.values() if e.is_negative()),
                "num_compressed_nodes": sum(1 for n in self.nodes.values() if n.lifecycle.get("storage_tier") == COMPRESSED),
                "pose_graph_optimization": "TODO_not_enabled",
            },
            "control_policy": {
                "fine_goal_execution_policy": "front_only_rotate_then_reobserve" if force_front_view_waypoint else "multi_view_waypoint_allowed",
                "memory_writes_are_requests": True,
                "place_recognition_commit_policy": "vlm_confirms_backend_verifies",
            },
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
            },
            "goal_context": {
                "task_mode": task_mode,
                "target_object": target_object,
                "goal_bearing_from_current_deg": round(float(goal_bearing_deg), 3),
                "goal_distance_m": round(float(goal_distance_m), 3),
                "detour_status": "escaping_deadlock" if current_node and current_node.navigation_state.get("deadlock_status") in {"suspected", "confirmed", "escaping"} else "normal_goal_seek",
                "goal_resume_hint": "avoid known negative edges; after escape re-align toward goal bearing or ObjNav object belief",
            },
            "deadlock_state": deadlock_state,
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
            "schema_version": "relative_topometric_memory_graph_v4",
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
            "negative_edge_index": self.negative_edge_index,
            "event_log": self.event_log[-200:],
        }
