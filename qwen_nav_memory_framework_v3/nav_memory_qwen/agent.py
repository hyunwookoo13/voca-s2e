"""Closed-loop navigation agent with VLM-grounded memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import traceback

import numpy as np

from .embedding import HashImageEmbedder, ImageEmbedder
from .memory_graph import MemoryGraph
from .robot_backend import ActionOutcome, RobotBackend
from .safety import sanitize_vlm_output
from .schema import (
    CoarseGoal,
    Observation,
    RelativePose2D,
    RobotState,
    build_vlm_input_v1,
    make_observation_request_output,
    make_rotate_output,
    normalize_angle_deg,
    relative_pose_between_robot_states,
    view_type_to_heading_deg,
)
from .utils import save_json, strip_large_image_values
from .vlm_client import BaseVLMClient, HeuristicVLMClient


@dataclass
class NavAgentConfig:
    """Configuration for :class:`NavMemoryAgent`."""

    success_distance_m: float = 0.45
    max_steps: int = 100
    localize_threshold: float = 0.88
    force_new_node_translation_m: float = 0.75
    no_progress_deadlock_count: int = 2
    observation_requests_deadlock_suspect_count: int = 2
    max_hot_nodes: int = 12
    hot_window_frames: int = 300
    max_memory_images: int = 4
    default_ttl_ms: int = 1000
    create_node_every_step: bool = False
    log_full_vlm_input: bool = False
    pose_noise_enabled: bool = True

    # Upgraded behavior.
    vlm_confirms_place_recognition: bool = True
    auto_commit_high_confidence_localization: bool = False
    revisit_min_vlm_confidence: float = 0.62
    revisit_min_backend_score: float = 0.45
    merge_min_vlm_confidence: float = 0.72
    merge_min_backend_score: float = 0.50
    force_front_view_waypoint: bool = True
    negative_branch_guard_deg: float = 45.0
    mark_deadlock_suspected_on_scan: bool = True


@dataclass
class StepResult:
    """One closed-loop step result."""

    step_index: int
    action: str
    done: bool
    success: bool
    vlm_output: Dict[str, Any]
    warnings: List[str]
    current_node_id: Optional[str]
    goal_distance_m: float
    outcome: Optional[ActionOutcome] = None
    vlm_input: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class EpisodeResult:
    """Returned by :meth:`NavMemoryAgent.run_until_done`."""

    success: bool
    done: bool
    steps: int
    final_distance_m: float
    step_results: List[StepResult]
    graph: MemoryGraph

    def summary(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "done": self.done,
            "steps": self.steps,
            "final_distance_m": round(float(self.final_distance_m), 4),
            "num_nodes": len([n for n in self.graph.nodes.values() if n.lifecycle.get("storage_tier") != "archived"]),
            "num_edges": len(self.graph.edges),
            "current_node_id": self.graph.current_node_id,
            "pose_graph_optimization": "TODO_not_enabled",
        }


class NavMemoryAgent:
    """Closed-loop semantic navigation supervisor.

    The VLM chooses between a fine waypoint and interrupt actions. Memory is a
    context/constraint source for that decision, not a low-level controller.
    Backend retrieval proposes revisit candidates, VLM verifies them through the
    prompt, and the backend commits only verified memory operations.
    """

    def __init__(
        self,
        *,
        robot: RobotBackend,
        vlm_client: Optional[BaseVLMClient] = None,
        embedder: Optional[ImageEmbedder] = None,
        memory: Optional[MemoryGraph] = None,
        config: Optional[NavAgentConfig] = None,
    ):
        self.robot = robot
        self.vlm_client = vlm_client or HeuristicVLMClient()
        self.embedder = embedder or HashImageEmbedder()
        self.memory = memory or MemoryGraph(embedding_dim=int(self.embedder.dim))
        self.config = config or NavAgentConfig()
        self.no_progress_count = 0
        self.observation_request_count = 0
        self.last_outcome: Optional[ActionOutcome] = None
        self.pending_edge_src_node_id: Optional[str] = None
        self.pending_edge_pose: Optional[RelativePose2D] = None
        self.pending_observation: Optional[Observation] = None
        self.step_logs: List[StepResult] = []

        # Transient anchor used to compute the live floating pose
        # T_latest_node_to_current_robot. This is not persisted as a node global
        # pose; it only converts robot-state/odometry into node-relative runtime
        # context for graph exits and subgoal reconstruction.
        self._live_pose_anchor_node_id: Optional[str] = None
        self._live_pose_anchor_robot_state: Optional[RobotState] = None

    # ------------------------------------------------------------------
    # Observation and memory pre-update
    # ------------------------------------------------------------------
    def _get_observation(self) -> Observation:
        if self.pending_observation is not None:
            obs = self.pending_observation
            self.pending_observation = None
            return obs
        return self.robot.get_observation()

    def _embed_front_or_first(self, observation: Observation) -> Optional[np.ndarray]:
        if not observation.views:
            return None
        view = next((v for v in observation.views if v.view_type == "front"), observation.views[0])
        try:
            return self.embedder.embed_image(view.image)
        except Exception:
            return None

    def _sync_current_pose_relation(
        self,
        node_id: str,
        robot_state: RobotState,
        frame_index: int,
        *,
        reset_anchor: bool = False,
        source: str = "robot_state_delta_from_latest_node_anchor",
    ) -> RelativePose2D:
        """Update the graph's live ``latest_node -> current robot`` relation.

        The node does not store a global pose. The agent keeps a transient robot
        state anchor for the latest/current node and computes a relative SE(2)
        delta to the current robot state at runtime. This lets graph exits be
        expressed in the current robot frame before Qwen chooses a fine goal.
        """
        if reset_anchor or self._live_pose_anchor_node_id != node_id or self._live_pose_anchor_robot_state is None:
            self._live_pose_anchor_node_id = node_id
            self._live_pose_anchor_robot_state = robot_state
            pose = RelativePose2D()
            self.memory.update_current_pose_relation_to_latest_node(
                node_id,
                pose,
                frame_index=frame_index,
                source=f"{source}:anchor_reset",
            )
            return pose

        pose = relative_pose_between_robot_states(self._live_pose_anchor_robot_state, robot_state)
        self.memory.update_current_pose_relation_to_latest_node(
            node_id,
            pose,
            frame_index=frame_index,
            source=source,
        )
        return pose

    def _update_memory_before_decision(self, observation: Observation, embedding: Optional[np.ndarray], robot_state: RobotState) -> str:
        """Localize or create a node before asking the VLM.

        Retrieval is used to propose candidates. When
        ``vlm_confirms_place_recognition`` is enabled, the agent does not commit
        uncertain/high-score revisits solely by embedding score; Qwen can confirm
        through ``memory_ops`` and the backend verifies before merging.
        """
        loc = self.memory.localize(embedding, threshold=self.config.localize_threshold)

        # Keep a live floating pose from the latest/current node to the robot.
        # This makes node-frame graph exits usable even before a new node is
        # committed. The relation is transient and does not store a global node pose.
        if self.memory.current_node_id is not None:
            self._sync_current_pose_relation(
                self.memory.current_node_id,
                robot_state,
                observation.frame_index,
                reset_anchor=False,
                source="pre_decision_robot_state",
            )
        live_distance = self.memory.live_pose_distance_from_latest_node_m()

        force_new = self.config.create_node_every_step or self.memory.current_node_id is None
        if live_distance >= self.config.force_new_node_translation_m:
            force_new = True
        if self.last_outcome and self.last_outcome.moved_distance_m >= self.config.force_new_node_translation_m:
            force_new = True
        if self.last_outcome and self.last_outcome.collision:
            force_new = False

        node_id: Optional[str] = None
        can_auto_commit = (
            self.config.auto_commit_high_confidence_localization
            and loc.match_status == "localized"
            and loc.current_node_id is not None
            and not force_new
        )
        if can_auto_commit:
            node_id = str(loc.current_node_id)
            self.memory.set_current_node(node_id, observation.frame_index)
        elif not force_new and self.memory.current_node_id is not None:
            # Keep the current node active while VLM verifies any revisit candidates.
            node_id = self.memory.current_node_id
        else:
            first_view = observation.views[0]
            node_id = self.memory.add_node(
                frame_index=observation.frame_index,
                image_ref=first_view.image,
                embedding=embedding,
                view_type=first_view.view_type,
                relative_heading_deg=first_view.relative_heading_deg,
                place_category="unknown",
                semantic_summary=f"observed place at frame {observation.frame_index}",
                timestamp_ms=observation.timestamp_ms,
            )

        # The selected/created node becomes the latest node anchor for the current
        # observation. Existing-node revisits reset the anchor; otherwise the
        # relation remains the accumulated live pose from that node.
        self._sync_current_pose_relation(
            str(node_id),
            robot_state,
            observation.frame_index,
            reset_anchor=(self._live_pose_anchor_node_id != str(node_id)),
            source="pre_decision_selected_node",
        )

        # Convert pending odometry into a graph edge once a destination node is known.
        if self.pending_edge_src_node_id and self.pending_edge_pose and self.pending_edge_src_node_id != node_id:
            src_node = self.memory.nodes.get(self.pending_edge_src_node_id)
            escaping_from_negative = bool(
                src_node
                and (src_node.negative_memory or src_node.navigation_state.get("deadlock_status") in {"escaping", "confirmed", "confirmed_escaped"})
            )
            if escaping_from_negative and self.last_outcome and self.last_outcome.success:
                self.memory.mark_escape_edge(
                    self.pending_edge_src_node_id,
                    node_id,
                    self.pending_edge_pose,
                    frame_index=observation.frame_index,
                )
            else:
                status = "success" if self.last_outcome and self.last_outcome.success else "unknown"
                self.memory.add_or_update_edge(
                    self.pending_edge_src_node_id,
                    node_id,
                    self.pending_edge_pose,
                    edge_type="temporal_transition",
                    relation_type="transition",
                    status=status,
                    frame_index=observation.frame_index,
                )
            self.pending_edge_src_node_id = None
            self.pending_edge_pose = None

        return str(node_id)

    # ------------------------------------------------------------------
    # Verified VLM memory operations
    # ------------------------------------------------------------------
    def _apply_vlm_memory_ops(self, vlm_output: Dict[str, Any], frame_index: int) -> None:
        """Apply optional VLM memory operation requests conservatively."""
        ops = vlm_output.get("memory_ops") or []
        if not isinstance(ops, list):
            return
        for op in ops:
            if not isinstance(op, dict):
                continue
            name = str(op.get("op"))
            try:
                conf = float(op.get("confidence", 0.0) or 0.0)
            except Exception:
                conf = 0.0
            if conf < 0.55:
                continue

            if name in {"mark_deadlock", "mark_incoming_edge", "mark_current_as_deadlock_region"} and self.memory.current_node_id:
                incoming = self.memory.latest_incoming_edge_id(self.memory.current_node_id)
                status = op.get("traversal_status") or op.get("deadlock_status")
                if status in {"deadlock_entry", "confirmed"}:
                    self.memory.mark_deadlock(self.memory.current_node_id, incoming, reason=str(op.get("trigger") or op.get("reason") or "vlm_memory_op"))
                elif status in {"deadlock_entry_candidate", "suspected"}:
                    self.memory.mark_deadlock_suspected(self.memory.current_node_id, incoming, reason=str(op.get("trigger") or op.get("reason") or "vlm_memory_op"))

            elif name == "mark_blocked_edge":
                edge_id = op.get("edge_id") or self.memory.latest_incoming_edge_id(self.memory.current_node_id)
                self.memory.mark_blocked_edge(str(edge_id) if edge_id else None, reason=str(op.get("reason") or "vlm_mark_blocked"))

            elif name == "mark_escape_edge" and self.memory.previous_node_id and self.memory.current_node_id:
                self.memory.mark_escape_edge(self.memory.previous_node_id, self.memory.current_node_id, RelativePose2D(), frame_index=frame_index)

            elif name in {"confirm_revisit_node", "confirm_same_place"}:
                candidate = op.get("node_id") or op.get("candidate_node_id") or op.get("target_node_id")
                if candidate:
                    self.memory.commit_revisit(
                        str(candidate),
                        frame_index=frame_index,
                        vlm_confidence=conf,
                        reason=str(op.get("reason") or "vlm_confirmed_revisit"),
                        min_vlm_confidence=self.config.revisit_min_vlm_confidence,
                        min_backend_score=self.config.revisit_min_backend_score,
                    )

            elif name in {"request_merge_nodes", "merge_nodes"}:
                keep = op.get("keep_node_id") or op.get("target_node_id") or op.get("node_id")
                remove = op.get("remove_node_id") or op.get("source_node_id") or op.get("current_node_id") or self.memory.current_node_id
                if keep and remove:
                    verification = self.memory.verify_merge_request(
                        str(keep),
                        str(remove),
                        vlm_confidence=conf,
                        min_vlm_confidence=self.config.merge_min_vlm_confidence,
                        min_backend_score=self.config.merge_min_backend_score,
                    )
                    if verification.accepted:
                        self.memory.merge_nodes(str(keep), str(remove), reason=str(op.get("reason") or "vlm_requested_merge"))

            elif name == "reject_revisit_candidate":
                # Log-only. Rejections are useful for debugging but do not mutate topology.
                self.memory._log("vlm_reject_revisit_candidate", candidate_node_id=op.get("node_id") or op.get("candidate_node_id"), confidence=conf, reason=op.get("reason"))

            elif name == "compress_node":
                node_id = op.get("node_id")
                if node_id in self.memory.nodes and node_id != self.memory.current_node_id:
                    self.memory.compress_node(str(node_id), reason=str(op.get("reason") or "vlm_requested_compress"))

            elif name in {"update_object_belief", "mark_object_seen"} and self.memory.current_node_id:
                node_update = op.get("node_update") if isinstance(op.get("node_update"), dict) else {}
                self.memory.update_object_belief(
                    self.memory.current_node_id,
                    target_object=op.get("target_object") or node_update.get("target_object"),
                    seen_target=op.get("seen_target") if "seen_target" in op else node_update.get("seen_target"),
                    candidate_objects=op.get("candidate_objects") or node_update.get("candidate_objects"),
                    room_object_prior=op.get("room_object_prior") or node_update.get("room_object_prior"),
                    reason=str(op.get("reason") or "vlm_object_belief_update"),
                )

    # ------------------------------------------------------------------
    # Safety guards and state machine
    # ------------------------------------------------------------------
    def _mark_suspected_current_deadlock(self, reason: str) -> None:
        if not self.memory.current_node_id:
            return
        incoming = self.memory.latest_incoming_edge_id(self.memory.current_node_id)
        self.memory.mark_deadlock_suspected(self.memory.current_node_id, incoming, reason=reason)

    def _guard_negative_branch(self, vlm_output: Dict[str, Any], memory_context: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        warnings: List[str] = []
        if vlm_output.get("action") != "go":
            return vlm_output, warnings
        selected_view_type = str(vlm_output.get("selected_view_type") or "front")
        selected_bearing = view_type_to_heading_deg(selected_view_type)
        avoid_edges = [c for c in memory_context.get("local_topology", {}).get("candidate_exits", []) if c.get("avoid")]
        for cand in avoid_edges:
            try:
                cand_bearing = float(cand.get("bearing_deg_robot", 999.0))
            except Exception:
                continue
            if abs(normalize_angle_deg(selected_bearing - cand_bearing)) <= self.config.negative_branch_guard_deg:
                safe = make_observation_request_output(
                    mode="directed_sweep",
                    center_yaw_deg=normalize_angle_deg(selected_bearing + 90.0),
                    step_deg=45,
                    num_views=3,
                    yaw_offsets_deg=[
                        normalize_angle_deg(selected_bearing + 45.0),
                        normalize_angle_deg(selected_bearing + 90.0),
                        normalize_angle_deg(selected_bearing + 135.0),
                    ],
                    reason="backend_guard_avoids_known_negative_branch",
                    confidence="medium",
                )
                safe["memory_ops"] = [{
                    "op": "reject_go_into_negative_edge",
                    "edge_id": cand.get("edge_id"),
                    "confidence": 1.0,
                    "reason": "selected view overlaps known deadlock/blocked branch",
                }]
                warnings.append("backend blocked go into known negative branch")
                return safe, warnings
        return vlm_output, warnings

    def _rotate_to_front_if_needed(self, vlm_output: Dict[str, Any]) -> Tuple[Optional[ActionOutcome], Dict[str, Any], bool]:
        """Convert non-front go into rotate-first policy when configured.

        The lower S2E/PixelNav skill then only receives front-view fine goals on a
        later step after the robot has physically rotated and re-observed.
        """
        if not self.config.force_front_view_waypoint or vlm_output.get("action") != "go":
            return None, vlm_output, False
        selected_view_type = str(vlm_output.get("selected_view_type") or "front")
        if selected_view_type == "front":
            return None, vlm_output, False
        yaw = view_type_to_heading_deg(selected_view_type)
        outcome = self.robot.rotate(yaw)
        rotated = make_rotate_output(yaw, reason="R01_GOAL_OUTSIDE_CURRENT_VIEW", confidence=str(vlm_output.get("confidence", "medium")))
        rotated["deferred_go"] = vlm_output
        rotated["reasoning"]["short_text"] = "rotate to selected view first; next step will obtain a front-view fine goal"
        rotated["control"]["vlm_control_mode"] = "rotate_to_selected_view_then_reobserve"
        self.pending_observation = self.robot.get_observation()
        return outcome, rotated, True

    def _update_memory_after_outcome(self, outcome: ActionOutcome, frame_index: int) -> None:
        current = self.memory.current_node_id
        if not current:
            return

        # Update the live floating pose immediately after the backend action.
        # Prefer the backend's current robot state, and fall back to odometry
        # composition only if a real backend cannot provide state at this point.
        try:
            post_state = self.robot.get_robot_state()
            live_pose = self._sync_current_pose_relation(
                current,
                post_state,
                frame_index,
                reset_anchor=False,
                source=f"post_action_{outcome.action}_robot_state",
            )
        except Exception:
            live_pose = self.memory.latest_node_to_robot_pose.compose(outcome.odom_delta)
            self.memory.update_current_pose_relation_to_latest_node(
                current,
                live_pose,
                frame_index=frame_index,
                source=f"post_action_{outcome.action}_odom_fallback",
            )

        if outcome.action == "go":
            if outcome.no_progress or outcome.collision:
                self.no_progress_count += 1
                incoming = self.memory.latest_incoming_edge_id(current)
                if outcome.collision:
                    self.memory.mark_blocked_edge(incoming, reason="collision_after_go")
                if self.no_progress_count >= self.config.no_progress_deadlock_count:
                    self.memory.mark_deadlock(current, incoming, reason="no_progress_or_collision")
                else:
                    self.memory.mark_deadlock_suspected(current, incoming, reason="first_no_progress_or_collision")
            else:
                self.no_progress_count = 0
                self.observation_request_count = 0
                self.pending_edge_src_node_id = current
                # The edge to the next committed node must be the cumulative
                # latest-node -> current-robot pose, not just the last action delta.
                self.pending_edge_pose = live_pose
                node = self.memory.nodes.get(current)
                if node and node.negative_memory:
                    node.navigation_state["deadlock_status"] = "escaping"
        elif outcome.action == "rotate":
            # Rotation itself is not a failed edge, but repeated rotations after
            # no-progress should keep the current node in suspected state.
            if self.no_progress_count > 0:
                self._mark_suspected_current_deadlock("rotate_after_no_progress")
        self.last_outcome = outcome
        self.memory.compress_old_nodes(
            current_frame_index=frame_index,
            hot_window_frames=self.config.hot_window_frames,
            max_hot_nodes=self.config.max_hot_nodes,
        )

    # ------------------------------------------------------------------
    # Goal construction and main step loop
    # ------------------------------------------------------------------
    def _build_goal(self, *, goal_map_xy: Optional[Tuple[float, float]], target_object: Optional[str]) -> CoarseGoal:
        state = self.robot.get_robot_state()
        if target_object:
            return CoarseGoal.from_object_goal(target_object, state)
        if goal_map_xy is None:
            raise ValueError("goal_map_xy is required for PointNav when target_object is not provided")
        return CoarseGoal.from_map_goal(goal_map_xy, state)

    def step(
        self,
        *,
        goal_map_xy: Optional[Tuple[float, float]] = None,
        target_object: Optional[str] = None,
        step_index: int = 0,
    ) -> StepResult:
        """Run one Observe→Memory→VLM→Act→Memory cycle."""
        try:
            state = self.robot.get_robot_state()
            goal = self._build_goal(goal_map_xy=goal_map_xy, target_object=target_object)
            observation = self._get_observation()
            embedding = self._embed_front_or_first(observation)
            current_node_id = self._update_memory_before_decision(observation, embedding, state)

            memory_context = self.memory.build_vlm_memory_context(
                goal_bearing_deg=goal.relative_bearing_deg,
                goal_distance_m=goal.distance_m,
                task_mode=goal.task_mode,
                target_object=goal.target_object,
                max_memory_images=self.config.max_memory_images,
                force_front_view_waypoint=self.config.force_front_view_waypoint,
            )
            if self.no_progress_count == 0:
                memory_context["loop_warning"] = {"is_looping": False, "repeated_branch_count": 0}
            memory_context["runtime_state"] = {
                "no_progress_count": self.no_progress_count,
                "observation_request_count": self.observation_request_count,
                "force_front_view_waypoint": self.config.force_front_view_waypoint,
                "last_action_outcome": None if self.last_outcome is None else {
                    "action": self.last_outcome.action,
                    "success": self.last_outcome.success,
                    "collision": self.last_outcome.collision,
                    "moved_distance_m": self.last_outcome.moved_distance_m,
                    "message": self.last_outcome.message,
                },
            }

            vlm_input = build_vlm_input_v1(
                task=goal,
                robot_state=state,
                observation=observation,
                memory=memory_context,
                pose_noise_enabled=self.config.pose_noise_enabled,
            )

            raw_output = self.vlm_client.decide(vlm_input)
            vlm_output, warnings = sanitize_vlm_output(raw_output, vlm_input)
            self._apply_vlm_memory_ops(vlm_output, observation.frame_index)

            # Rebuild context after memory_ops so the guard uses the latest node/edge state.
            guard_context = self.memory.build_vlm_memory_context(
                goal_bearing_deg=goal.relative_bearing_deg,
                goal_distance_m=goal.distance_m,
                task_mode=goal.task_mode,
                target_object=goal.target_object,
                max_memory_images=self.config.max_memory_images,
                force_front_view_waypoint=self.config.force_front_view_waypoint,
            )
            vlm_output, guard_warnings = self._guard_negative_branch(vlm_output, guard_context)
            warnings.extend(guard_warnings)

            action = vlm_output.get("action")
            outcome: Optional[ActionOutcome] = None
            done = False
            success = False

            if goal.task_mode == "PointNav" and goal.distance_m <= self.config.success_distance_m:
                vlm_output = {**vlm_output, "action": "stop"}
                done = True
                success = True
            elif action == "stop":
                done = True
                success = bool(goal.task_mode == "PointNav" and goal.distance_m <= max(self.config.success_distance_m, 0.75))
            elif action == "request_observation":
                req = vlm_output.get("observation_request") or {}
                offsets = req.get("yaw_offsets_deg") or [req.get("center_yaw_deg", 0)]
                self.pending_observation = self.robot.capture_views(offsets, mode=str(req.get("mode", "directed_sweep")))
                self.observation_request_count += 1
                failure_mode = str(vlm_output.get("reasoning", {}).get("failure_mode") or "")
                reason_text = str(req.get("reason") or vlm_output.get("reasoning", {}).get("short_text") or "")
                if self.config.mark_deadlock_suspected_on_scan and (
                    self.no_progress_count > 0
                    or self.observation_request_count >= self.config.observation_requests_deadlock_suspect_count
                    or "deadlock" in failure_mode
                    or "deadlock" in reason_text
                    or "no_visible" in failure_mode
                ):
                    self._mark_suspected_current_deadlock(f"scan_requested:{failure_mode or reason_text}")
            elif action == "rotate":
                yaw = float(vlm_output.get("control", {}).get("rotate_yaw_deg", 45.0) or 45.0)
                outcome = self.robot.rotate(yaw)
                self._update_memory_after_outcome(outcome, observation.frame_index)
            elif action == "go":
                rotate_outcome, maybe_rotated_output, rotated_to_front = self._rotate_to_front_if_needed(vlm_output)
                if rotated_to_front:
                    vlm_output = maybe_rotated_output
                    outcome = rotate_outcome
                    if outcome:
                        self._update_memory_after_outcome(outcome, observation.frame_index)
                else:
                    point = vlm_output["selected_image_point"]
                    outcome = self.robot.execute_waypoint(
                        view_type=str(vlm_output.get("selected_view_type", "front")),
                        view_id=int(vlm_output.get("selected_view_id", 0)),
                        point_px=(int(point[0]), int(point[1])),
                        ttl_ms=int(vlm_output.get("control", {}).get("ttl_ms", self.config.default_ttl_ms) or self.config.default_ttl_ms),
                    )
                    self._update_memory_after_outcome(outcome, observation.frame_index)

            result = StepResult(
                step_index=step_index,
                action=str(vlm_output.get("action")),
                done=done,
                success=success,
                vlm_output=vlm_output,
                warnings=warnings,
                current_node_id=self.memory.current_node_id or current_node_id,
                goal_distance_m=goal.distance_m,
                outcome=outcome,
                vlm_input=vlm_input if self.config.log_full_vlm_input else strip_large_image_values(vlm_input),
            )
            self.step_logs.append(result)
            return result
        except Exception as exc:
            err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            result = StepResult(
                step_index=step_index,
                action="error",
                done=True,
                success=False,
                vlm_output={},
                warnings=[],
                current_node_id=self.memory.current_node_id,
                goal_distance_m=float("inf"),
                outcome=None,
                vlm_input=None,
                error=err,
            )
            self.step_logs.append(result)
            return result

    def run_until_done(
        self,
        *,
        goal_map_xy: Optional[Tuple[float, float]] = None,
        target_object: Optional[str] = None,
        max_steps: Optional[int] = None,
    ) -> EpisodeResult:
        """Run the closed loop until success/stop/error/max_steps."""
        max_steps = int(max_steps or self.config.max_steps)
        step_results: List[StepResult] = []
        final_distance = float("inf")
        for i in range(max_steps):
            result = self.step(goal_map_xy=goal_map_xy, target_object=target_object, step_index=i)
            step_results.append(result)
            final_distance = result.goal_distance_m
            if result.done or result.error:
                return EpisodeResult(success=result.success, done=True, steps=i + 1, final_distance_m=final_distance, step_results=step_results, graph=self.memory)
        if goal_map_xy is not None:
            state = self.robot.get_robot_state()
            final_goal = CoarseGoal.from_map_goal(goal_map_xy, state)
            final_distance = final_goal.distance_m
        return EpisodeResult(success=False, done=False, steps=max_steps, final_distance_m=final_distance, step_results=step_results, graph=self.memory)

    def save_run(self, out_dir: str | Path) -> None:
        """Save graph and compact step logs for offline debugging."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_json(out / "memory_graph.json", self.memory.to_dict())
        logs = []
        for s in self.step_logs:
            logs.append({
                "step_index": s.step_index,
                "action": s.action,
                "done": s.done,
                "success": s.success,
                "warnings": s.warnings,
                "current_node_id": s.current_node_id,
                "goal_distance_m": s.goal_distance_m,
                "vlm_output": s.vlm_output,
                "outcome": None if s.outcome is None else {
                    "action": s.outcome.action,
                    "success": s.outcome.success,
                    "collision": s.outcome.collision,
                    "moved_distance_m": s.outcome.moved_distance_m,
                    "rotated_deg": s.outcome.rotated_deg,
                    "message": s.outcome.message,
                },
                "error": s.error,
            })
        save_json(out / "steps.json", {"steps": logs})
