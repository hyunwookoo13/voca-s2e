"""Relative-pose topo-metric episodic memory graph.

The graph stores no global node pose. Spatial structure is represented by
relative SE(2) transforms on directed edges. This makes the memory topological
at the node level while preserving enough local metric information to compose
subgoals from the currently localized node.
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


def _ts_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class Keyframe:
    """Visual evidence attached to a node."""

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
        """Return a VLM-facing summary of this keyframe."""
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
        if self.navigation_state.get("deadlock_status") in {"suspected", "confirmed", "confirmed_escaped"}:
            return True
        return self.navigation_state.get("memory_importance") == "critical"

    def short_description(self) -> str:
        return str(self.semantic.get("short_description") or self.place_category or self.node_id)

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
            "lifecycle": dict(self.lifecycle),
            "keyframes": kfs,
        }


@dataclass
class MemoryEdge:
    """Directed edge with relative pose and traversal outcome."""

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
        elif success_count > 0:
            deadlock_penalty = -0.15 * min(success_count, 5)
        uncertainty = sum(float(x) for x in self.relative_pose_src_to_dst.covariance_diag)
        uncertainty_penalty = 0.1 * uncertainty
        final = max(0.01, base + deadlock_penalty + uncertainty_penalty)
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
    """Current-observation localization result."""

    match_status: str
    current_node_id: Optional[str]
    match_confidence: float
    candidate_node_ids: List[str]
    matched_keyframe_ids: List[str]


class MemoryGraph:
    """Relative-pose topo-metric episodic memory graph.

    Main ideas implemented here:
      * SPTM-style graph of places plus observation retrieval.
      * Neural-topological-SLAM-style semantic nodes and coarse edge geometry.
      * PRISM-TopoMap-style no global node pose; only locally aligned graph.
      * SLAM-marginalization-inspired compression: drop active images, keep
        constraints, summaries, embeddings, and negative memories.
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

    # ------------------------------------------------------------------
    # Graph mutation
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
    ) -> str:
        """Create a new place node and optional keyframe."""
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
        self.nodes[node_id] = node
        self.out_edges[node_id] = []
        self.in_edges[node_id] = []
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
        self.loop_history.append(node_id)
        if len(self.loop_history) > 20:
            self.loop_history = self.loop_history[-20:]
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
        return kf_id

    def localize(self, embedding: Optional[np.ndarray], top_k: int = 5, threshold: float = 0.86) -> LocalizationResult:
        """Localize current observation to a known node using visual retrieval."""
        if embedding is None or len(self.visual_index) == 0:
            result = LocalizationResult("new_place", None, 0.0, [], [])
            self.last_localization = result
            return result
        hits = self.visual_index.search(embedding, top_k=top_k, min_score=-1.0)
        candidate_node_ids: List[str] = []
        matched_keyframe_ids: List[str] = []
        best_node: Optional[str] = None
        best_score = 0.0
        seen = set()
        for hit in hits:
            node_id = hit.metadata.get("node_id")
            if node_id and node_id not in seen:
                candidate_node_ids.append(node_id)
                seen.add(node_id)
            if hit.key:
                matched_keyframe_ids.append(hit.key)
            if node_id and hit.score > best_score:
                best_score = hit.score
                best_node = node_id
        status = "localized" if best_node and best_score >= threshold else ("uncertain" if best_node else "new_place")
        result = LocalizationResult(status, best_node if status == "localized" else None, float(best_score), candidate_node_ids, matched_keyframe_ids)
        self.last_localization = result
        return result

    def set_current_node(self, node_id: str, frame_index: int) -> None:
        """Set current node after successful localization/revisit."""
        if node_id not in self.nodes:
            raise KeyError(f"unknown node_id: {node_id}")
        if self.current_node_id != node_id:
            self.previous_node_id = self.current_node_id
        self.current_node_id = node_id
        node = self.nodes[node_id]
        node.visit_count += 1
        node.last_seen_frame_index = int(frame_index)
        node.last_seen_timestamp_ms = _ts_ms()
        self.loop_history.append(node_id)
        if len(self.loop_history) > 20:
            self.loop_history = self.loop_history[-20:]

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
            # Keep the newest pose but soften covariance by taking component-wise min.
            old_cov = edge.relative_pose_src_to_dst.covariance_diag
            new_cov = tuple(min(float(a), float(b)) for a, b in zip(old_cov, relative_pose.covariance_diag))
            edge.relative_pose_src_to_dst = RelativePose2D(relative_pose.dx_m, relative_pose.dy_m, relative_pose.dyaw_deg, new_cov)  # type: ignore[arg-type]
        else:
            edge_id = self._new_edge_id()
            edge = MemoryEdge(edge_id=edge_id, src_node_id=src_node_id, dst_node_id=dst_node_id, relative_pose_src_to_dst=relative_pose, edge_type=edge_type, relation_type=relation_type)
            self.edges[edge_id] = edge
            self.out_edges.setdefault(src_node_id, []).append(edge_id)
            self.in_edges.setdefault(dst_node_id, []).append(edge_id)
        edge.traversal["last_attempt_frame"] = frame_index
        if status == "success":
            edge.traversal["success_count"] = int(edge.traversal.get("success_count", 0) or 0) + 1
            if edge.traversal.get("status") not in NEGATIVE_EDGE_STATUSES:
                edge.traversal["status"] = "success"
            edge.traversal["last_outcome"] = "success"
        elif status in NEGATIVE_EDGE_STATUSES or status in {"unknown", "escape_success", "risky"}:
            edge.traversal["status"] = status
            if status in NEGATIVE_EDGE_STATUSES:
                edge.traversal["failure_count"] = int(edge.traversal.get("failure_count", 0) or 0) + 1
            edge.traversal["last_outcome"] = status
        edge.update_cost()
        if edge.is_negative():
            self.negative_edge_index.setdefault(src_node_id, [])
            if edge.edge_id not in self.negative_edge_index[src_node_id]:
                self.negative_edge_index[src_node_id].append(edge.edge_id)
        return edge.edge_id

    def find_edge(self, src_node_id: str, dst_node_id: str) -> Optional[str]:
        for edge_id in self.out_edges.get(src_node_id, []):
            edge = self.edges[edge_id]
            if edge.dst_node_id == dst_node_id:
                return edge_id
        return None

    def mark_deadlock(self, node_id: str, incoming_edge_id: Optional[str], reason: str = "deadlock_detected") -> None:
        """Mark a node and its incoming edge as deadlock-related negative memory."""
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
            self.negative_edge_index.setdefault(edge.src_node_id, [])
            if incoming_edge_id not in self.negative_edge_index[edge.src_node_id]:
                self.negative_edge_index[edge.src_node_id].append(incoming_edge_id)

    def mark_escape_edge(self, src_node_id: str, dst_node_id: str, relative_pose: RelativePose2D, frame_index: Optional[int] = None) -> str:
        """Mark an edge as an escape/backtrack route from a deadlock region."""
        edge_id = self.add_or_update_edge(src_node_id, dst_node_id, relative_pose, edge_type="temporal_transition", relation_type="backtrack_or_escape", status="escape_success", frame_index=frame_index)
        edge = self.edges[edge_id]
        edge.traversal["status"] = "escape_success"
        edge.traversal["success_count"] = int(edge.traversal.get("success_count", 0) or 0) + 1
        edge.update_cost()
        src = self.nodes.get(src_node_id)
        if src and src.negative_memory:
            src.negative_memory["escape_edge_id"] = edge_id
            src.navigation_state["deadlock_status"] = "confirmed_escaped"
        # Link reverse deadlock edge to escape edge if present.
        rev = self.find_edge(dst_node_id, src_node_id)
        if rev and rev in self.edges:
            self.edges[rev].traversal["escape_edge_id"] = edge_id
        return edge_id

    def merge_nodes(self, keep_node_id: str, remove_node_id: str, reason: str = "visual_loop_closure") -> None:
        """Merge two nodes, preserving keyframes and redirecting edges.

        The backend should call this only after visual and odometric consistency
        checks. Relative edge constraints are preserved; contradictory edges are
        left as multiple evidence through updated counts/costs rather than solved
        by full pose-graph optimization.
        """
        if keep_node_id == remove_node_id or keep_node_id not in self.nodes or remove_node_id not in self.nodes:
            return
        keep = self.nodes[keep_node_id]
        rem = self.nodes[remove_node_id]
        keep.visit_count += rem.visit_count
        keep.last_seen_frame_index = max(keep.last_seen_frame_index, rem.last_seen_frame_index)
        if rem.negative_memory and not keep.negative_memory:
            keep.negative_memory = rem.negative_memory
        if rem.is_critical():
            keep.navigation_state["memory_importance"] = "critical"
            keep.navigation_state["risk_level"] = max(str(keep.navigation_state.get("risk_level", "low")), "high")
        for kf in rem.keyframes:
            kf.node_id = keep_node_id
            keep.keyframes.append(kf)
            if kf.keyframe_id in self.visual_index.keys():
                # Re-add updates metadata.
                # Vector is not exposed from index; metadata update without vector is unavailable.
                pass
        # Redirect edges.
        for edge_id in list(self.out_edges.get(remove_node_id, [])):
            edge = self.edges[edge_id]
            edge.src_node_id = keep_node_id
            self.out_edges.setdefault(keep_node_id, []).append(edge_id)
        for edge_id in list(self.in_edges.get(remove_node_id, [])):
            edge = self.edges[edge_id]
            edge.dst_node_id = keep_node_id
            self.in_edges.setdefault(keep_node_id, []).append(edge_id)
        self.out_edges.pop(remove_node_id, None)
        self.in_edges.pop(remove_node_id, None)
        rem.lifecycle.update({"storage_tier": ARCHIVED, "vlm_visible": False, "can_retrieve_image_for_vlm": False, "compression_level": "merged_tombstone", "merged_into": keep_node_id, "merge_reason": reason})
        # Keep tombstone in self.nodes for auditability, but hide it from planning.
        if self.current_node_id == remove_node_id:
            self.current_node_id = keep_node_id

    def compress_old_nodes(self, *, current_frame_index: int, hot_window_frames: int = 300, max_hot_nodes: int = 12) -> None:
        """Compress non-critical old nodes.

        This mimics marginalization: active images leave the VLM prompt, but
        embeddings, summaries, relative constraints, and negative memories stay.
        """
        hot_candidates = sorted(
            [n for n in self.nodes.values() if n.lifecycle.get("storage_tier") == HOT],
            key=lambda n: n.last_seen_frame_index,
            reverse=True,
        )
        keep_hot = {n.node_id for n in hot_candidates[:max_hot_nodes]}
        for node in list(self.nodes.values()):
            if node.node_id == self.current_node_id or node.node_id in keep_hot:
                continue
            old_enough = current_frame_index - node.last_seen_frame_index > hot_window_frames
            too_many_hot = node.lifecycle.get("storage_tier") == HOT and node.node_id not in keep_hot
            if not (old_enough or too_many_hot):
                continue
            if node.is_critical():
                # Critical nodes keep semantic/negative info but raw images do not need active prompt space.
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

    # ------------------------------------------------------------------
    # Planning and context
    # ------------------------------------------------------------------
    def outgoing_edges(self, node_id: str) -> List[MemoryEdge]:
        return [self.edges[eid] for eid in self.out_edges.get(node_id, []) if eid in self.edges and self.edges[eid].src_node_id == node_id]

    def incoming_edges(self, node_id: str) -> List[MemoryEdge]:
        return [self.edges[eid] for eid in self.in_edges.get(node_id, []) if eid in self.edges and self.edges[eid].dst_node_id == node_id]

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
        """Score one outgoing edge for VLM context and backend fallback."""
        bearing = normalize_angle_deg(math.degrees(math.atan2(edge.relative_pose_src_to_dst.dy_m, edge.relative_pose_src_to_dst.dx_m)))
        goal_alignment = 1.0 - min(abs(normalize_angle_deg(bearing - goal_bearing_deg)), 180.0) / 180.0
        deadlock_risk = 1.0 if edge.is_negative() else 0.0
        status = str(edge.traversal.get("status", "unknown"))
        exploration_value = 0.7 if status == "unknown" else 0.2
        if status == "success":
            exploration_value = 0.35
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
        }

    def build_vlm_memory_context(
        self,
        *,
        goal_bearing_deg: float,
        goal_distance_m: float,
        task_mode: str = "PointNav",
        max_memory_images: int = 4,
    ) -> Dict[str, Any]:
        """Build compact memory context for the VLM input ``memory`` field."""
        current_node = self.nodes.get(self.current_node_id) if self.current_node_id else None
        is_looping, repeated_count = self.is_looping()
        loc = self.last_localization or LocalizationResult("new_place", None, 0.0, [], [])
        outgoing = self.outgoing_edges(current_node.node_id) if current_node else []
        scored_edges = [self.score_candidate_exit(e, goal_bearing_deg) for e in outgoing]
        scored_edges.sort(key=lambda x: x["score"], reverse=True)

        # Add view-level frontier candidates when the graph has no outgoing info.
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

        # Retrieved images: hot current/neighbor images first, then negative evidence.
        memory_images: List[Dict[str, Any]] = []
        image_nodes = []
        if current_node:
            image_nodes.append(current_node)
            for e in outgoing:
                if e.dst_node_id in self.nodes:
                    image_nodes.append(self.nodes[e.dst_node_id])
        # Critical negative nodes are useful evidence even after escape.
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

        context = {
            # Original v1-compatible fields remain present.
            "visited_nodes": [n.to_context(include_images=False) for n in list(self.nodes.values())[-8:]],
            "failed_waypoints": [self.edges[eid].to_context() for ids in self.negative_edge_index.values() for eid in ids if eid in self.edges][-8:],
            "last_selected_view": None,
            "last_selected_point": None,
            "loop_warning": {"is_looping": bool(is_looping), "repeated_branch_count": int(repeated_count)},
            # Extensions used by the framework.
            "schema_version": "nav_memory_context_v3",
            "graph_summary": {
                "num_nodes": len([n for n in self.nodes.values() if n.lifecycle.get("storage_tier") != ARCHIVED]),
                "num_edges": len(self.edges),
                "num_deadlock_edges": sum(1 for e in self.edges.values() if e.is_negative()),
                "num_compressed_nodes": sum(1 for n in self.nodes.values() if n.lifecycle.get("storage_tier") == COMPRESSED),
            },
            "current_localization": {
                "current_node_id": self.current_node_id,
                "match_status": loc.match_status,
                "match_confidence": round(float(loc.match_confidence), 4),
                "candidate_node_ids": loc.candidate_node_ids,
                "matched_keyframe_ids": loc.matched_keyframe_ids,
                "revisit_likelihood": round(float(loc.match_confidence), 4),
            },
            "goal_context": {
                "task_mode": task_mode,
                "goal_bearing_from_current_deg": round(float(goal_bearing_deg), 3),
                "goal_distance_m": round(float(goal_distance_m), 3),
                "detour_status": "escaping_deadlock" if current_node and current_node.negative_memory else "normal_goal_seek",
                "goal_resume_hint": "avoid known negative edges; after escape re-align toward goal bearing",
            },
            "local_topology": {
                "current_node": current_node.to_context(include_images=False) if current_node else None,
                "candidate_exits": candidate_exits,
            },
            "retrieved_memory_images": memory_images,
            "compressed_negative_memories": compressed_negative[-8:],
        }
        return context

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "relative_topometric_memory_graph_v3",
            "graph_id": self.graph_id,
            "pose_policy": {
                "node_global_pose_stored": False,
                "edge_pose_type": "relative_SE2",
                "edge_pose_direction": "src_to_dst",
                "supports_path_composition": True,
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
            "negative_edge_index": self.negative_edge_index,
        }
