"""Closed-loop navigation agent with relative-pose memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import traceback

import numpy as np

from .embedding import HashImageEmbedder, ImageEmbedder
from .memory_graph import MemoryGraph
from .robot_backend import ActionOutcome, RobotBackend
from .safety import sanitize_vlm_output
from .schema import CoarseGoal, Observation, RelativePose2D, RobotState, build_vlm_input_v1, normalize_angle_deg
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
    max_hot_nodes: int = 12
    hot_window_frames: int = 300
    max_memory_images: int = 4
    default_ttl_ms: int = 1000
    create_node_every_step: bool = False
    log_full_vlm_input: bool = False
    pose_noise_enabled: bool = True


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
            "num_nodes": len(self.graph.nodes),
            "num_edges": len(self.graph.edges),
            "current_node_id": self.graph.current_node_id,
        }


class NavMemoryAgent:
    """Closed-loop navigation supervisor.

    The agent uses a slow VLM client for semantic decisions and a fast
    ``RobotBackend`` for low-level control. Memory updates are verified by the
    backend from odometry/action outcomes rather than blindly trusting the VLM.
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
        self.last_outcome: Optional[ActionOutcome] = None
        self.pending_edge_src_node_id: Optional[str] = None
        self.pending_edge_pose: Optional[RelativePose2D] = None
        self.pending_observation: Optional[Observation] = None
        self.step_logs: List[StepResult] = []

    # ------------------------------------------------------------------
    # Core step loop
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

    def _update_memory_before_decision(self, observation: Observation, embedding: Optional[np.ndarray]) -> str:
        """Localize or create node before asking the VLM."""
        loc = self.memory.localize(embedding, threshold=self.config.localize_threshold)
        force_new = self.config.create_node_every_step or self.memory.current_node_id is None
        if self.last_outcome and self.last_outcome.moved_distance_m >= self.config.force_new_node_translation_m:
            force_new = True
        if self.last_outcome and self.last_outcome.collision:
            force_new = False

        if loc.match_status == "localized" and loc.current_node_id and not force_new:
            node_id = loc.current_node_id
            prev = self.memory.current_node_id
            self.memory.set_current_node(node_id, observation.frame_index)
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

        return node_id

    def _apply_vlm_memory_ops(self, vlm_output: Dict[str, Any], frame_index: int) -> None:
        """Apply optional VLM memory operation requests conservatively."""
        ops = vlm_output.get("memory_ops") or []
        if not isinstance(ops, list):
            return
        for op in ops:
            if not isinstance(op, dict):
                continue
            name = op.get("op")
            conf = float(op.get("confidence", 0.0) or 0.0)
            if conf < 0.55:
                continue
            if name in {"mark_deadlock", "mark_incoming_edge"} and self.memory.current_node_id:
                incoming = self.memory.latest_incoming_edge_id(self.memory.current_node_id)
                status = op.get("traversal_status") or op.get("deadlock_status")
                if status in {"deadlock_entry", "deadlock_entry_candidate", "confirmed", "suspected"}:
                    self.memory.mark_deadlock(self.memory.current_node_id, incoming, reason=str(op.get("trigger", "vlm_memory_op")))
            elif name == "compress_node":
                node_id = op.get("node_id")
                if node_id in self.memory.nodes and node_id != self.memory.current_node_id:
                    node = self.memory.nodes[node_id]
                    for kf in node.keyframes:
                        if kf.image_ref and not kf.archived_image_ref:
                            kf.archived_image_ref = kf.image_ref
                        kf.image_ref = None
                        kf.thumbnail_ref = None
                        kf.active_for_vlm = False
                    node.lifecycle.update({"storage_tier": "compressed", "can_retrieve_image_for_vlm": False, "compression_level": "vlm_requested"})
            # Merge/remove requests are intentionally not applied here without
            # backend verification; add project-specific checks before enabling.

    def _update_memory_after_outcome(self, outcome: ActionOutcome, frame_index: int) -> None:
        current = self.memory.current_node_id
        if not current:
            return
        if outcome.action == "go":
            if outcome.no_progress:
                self.no_progress_count += 1
                incoming = self.memory.latest_incoming_edge_id(current)
                if self.no_progress_count >= self.config.no_progress_deadlock_count:
                    self.memory.mark_deadlock(current, incoming, reason="no_progress_or_collision")
            else:
                # Save odom edge when the next observation creates/localizes dst node.
                self.no_progress_count = 0
                self.pending_edge_src_node_id = current
                self.pending_edge_pose = outcome.odom_delta
                # If current node had negative state and we moved away next, the edge
                # will be linked as an escape after localization. We also mark a
                # self-contained hint now for the VLM.
                node = self.memory.nodes.get(current)
                if node and node.negative_memory:
                    node.navigation_state["deadlock_status"] = "escaping"
        elif outcome.action == "rotate":
            # Repeated rotate in place with no new info may indicate deadlock, but
            # do not mark immediately; observation requests should follow.
            pass
        self.last_outcome = outcome
        self.memory.compress_old_nodes(
            current_frame_index=frame_index,
            hot_window_frames=self.config.hot_window_frames,
            max_hot_nodes=self.config.max_hot_nodes,
        )

    def step(self, *, goal_map_xy: Tuple[float, float], step_index: int = 0) -> StepResult:
        """Run one VLM-memory-control step."""
        try:
            state = self.robot.get_robot_state()
            goal = CoarseGoal.from_map_goal(goal_map_xy, state)
            observation = self._get_observation()
            embedding = self._embed_front_or_first(observation)
            current_node_id = self._update_memory_before_decision(observation, embedding)

            memory_context = self.memory.build_vlm_memory_context(
                goal_bearing_deg=goal.relative_bearing_deg,
                goal_distance_m=goal.distance_m,
                task_mode=goal.task_mode,
                max_memory_images=self.config.max_memory_images,
            )
            # Repeated localization to the same visual node is common with a
            # coarse or static demo image. Treat it as a loop only when paired
            # with recent no-progress/collision evidence.
            if self.no_progress_count == 0:
                memory_context["loop_warning"] = {"is_looping": False, "repeated_branch_count": 0}
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

            action = vlm_output.get("action")
            outcome: Optional[ActionOutcome] = None
            done = False
            success = False

            if goal.distance_m <= self.config.success_distance_m:
                # Do not rely on VLM to stop once geometry says success.
                vlm_output = {**vlm_output, "action": "stop"}
                done = True
                success = True
            elif action == "stop":
                done = True
                success = goal.distance_m <= max(self.config.success_distance_m, 0.75)
            elif action == "request_observation":
                req = vlm_output.get("observation_request") or {}
                offsets = req.get("yaw_offsets_deg") or [req.get("center_yaw_deg", 0)]
                self.pending_observation = self.robot.capture_views(offsets, mode=str(req.get("mode", "directed_sweep")))
            elif action == "rotate":
                yaw = float(vlm_output.get("control", {}).get("rotate_yaw_deg", 45.0) or 45.0)
                outcome = self.robot.rotate(yaw)
                self._update_memory_after_outcome(outcome, observation.frame_index)
            elif action == "go":
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
                current_node_id=current_node_id,
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

    def run_until_done(self, *, goal_map_xy: Tuple[float, float], max_steps: Optional[int] = None) -> EpisodeResult:
        """Run the closed loop until success/stop/error/max_steps."""
        max_steps = int(max_steps or self.config.max_steps)
        step_results: List[StepResult] = []
        final_distance = float("inf")
        for i in range(max_steps):
            result = self.step(goal_map_xy=goal_map_xy, step_index=i)
            step_results.append(result)
            final_distance = result.goal_distance_m
            if result.done or result.error:
                return EpisodeResult(success=result.success, done=True, steps=i + 1, final_distance_m=final_distance, step_results=step_results, graph=self.memory)
        # Max-steps exhausted.
        state = self.robot.get_robot_state()
        final_goal = CoarseGoal.from_map_goal(goal_map_xy, state)
        return EpisodeResult(success=False, done=False, steps=max_steps, final_distance_m=final_goal.distance_m, step_results=step_results, graph=self.memory)

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
