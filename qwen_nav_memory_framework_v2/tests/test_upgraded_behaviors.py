import tempfile
from pathlib import Path
import unittest

import numpy as np

from nav_memory_qwen.agent import NavMemoryAgent, NavAgentConfig
from nav_memory_qwen.memory_graph import MemoryGraph, ARCHIVED
from nav_memory_qwen.robot_backend import StaticImageBackend
from nav_memory_qwen.schema import RelativePose2D, make_go_output
from nav_memory_qwen.vlm_client import BaseVLMClient


class LeftGoVLM(BaseVLMClient):
    def decide(self, vlm_input):
        obs = vlm_input["observation"]
        width = obs["image_width"]
        height = obs["image_height"]
        left = next(v for v in obs["views"] if v["view_type"] == "left")
        return make_go_output(
            view_id=left["view_id"],
            view_type="left",
            point_px=(width // 2, int(height * 0.75)),
            width=width,
            height=height,
            decision_reason="G01_GOAL_ALIGNED_VIEW",
            goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
            short_text="test selects left view",
        )


class UpgradedBehaviorTest(unittest.TestCase):
    def test_vlm_confirmed_revisit_merges_provisional_node(self):
        g = MemoryGraph(embedding_dim=4)
        emb = np.array([1, 0, 0, 0], dtype=np.float32)
        n1 = g.add_node(frame_index=1, image_ref=None, embedding=emb)
        loc = g.localize(emb, threshold=0.90)
        self.assertIn(n1, loc.candidate_node_ids)
        n2 = g.add_node(frame_index=2, image_ref=None, embedding=emb)
        result = g.commit_revisit(n1, frame_index=2, vlm_confidence=0.95)
        self.assertTrue(result.accepted)
        self.assertEqual(g.current_node_id, n1)
        self.assertEqual(g.nodes[n2].lifecycle["storage_tier"], ARCHIVED)

    def test_context_exposes_place_recognition_candidates(self):
        g = MemoryGraph(embedding_dim=4)
        emb = np.array([1, 0, 0, 0], dtype=np.float32)
        n1 = g.add_node(frame_index=1, image_ref="memory.jpg", embedding=emb)
        g.localize(emb, threshold=0.95)
        ctx = g.build_vlm_memory_context(goal_bearing_deg=0.0, goal_distance_m=2.0)
        self.assertEqual(ctx["schema_version"], "nav_memory_context_v4")
        self.assertIn("place_recognition", ctx)
        self.assertGreaterEqual(len(ctx["place_recognition"]["revisit_candidates"]), 1)
        self.assertEqual(ctx["graph_summary"]["pose_graph_optimization"], "TODO_not_enabled")

    def test_rotate_to_front_policy_prevents_non_front_waypoint_execution(self):
        with tempfile.TemporaryDirectory() as td:
            img = StaticImageBackend.create_demo_image(Path(td) / "img.jpg")
            robot = StaticImageBackend(img, step_m=1.0)
            agent = NavMemoryAgent(
                robot=robot,
                vlm_client=LeftGoVLM(),
                config=NavAgentConfig(max_steps=1, force_front_view_waypoint=True),
            )
            agent.pending_observation = robot.capture_views([-90], mode="directed_view")
            result = agent.step(goal_map_xy=(0.0, -2.0), step_index=0)
            self.assertEqual(result.action, "rotate")
            self.assertIsNotNone(result.outcome)
            self.assertEqual(result.outcome.action, "rotate")
            self.assertIn("deferred_go", result.vlm_output)

    def test_deadlock_suspected_on_repeated_scan_request(self):
        class ScanVLM(BaseVLMClient):
            def decide(self, vlm_input):
                from nav_memory_qwen.schema import make_observation_request_output
                return make_observation_request_output(
                    mode="full_sweep",
                    center_yaw_deg=0,
                    step_deg=45,
                    num_views=8,
                    yaw_offsets_deg=[-180, -135, -90, -45, 0, 45, 90, 135],
                    reason="deadlock_suspected_need_escape_exit",
                )

        with tempfile.TemporaryDirectory() as td:
            img = StaticImageBackend.create_demo_image(Path(td) / "img.jpg")
            robot = StaticImageBackend(img, step_m=1.0)
            agent = NavMemoryAgent(robot=robot, vlm_client=ScanVLM(), config=NavAgentConfig(max_steps=2))
            r1 = agent.step(goal_map_xy=(3.0, 0.0), step_index=0)
            r2 = agent.step(goal_map_xy=(3.0, 0.0), step_index=1)
            current = agent.memory.nodes[agent.memory.current_node_id]
            self.assertIn(current.navigation_state["deadlock_status"], {"suspected", "confirmed"})


if __name__ == "__main__":
    unittest.main()
