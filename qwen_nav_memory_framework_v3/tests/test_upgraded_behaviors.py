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




class FrontGoVLM(BaseVLMClient):
    def decide(self, vlm_input):
        obs = vlm_input["observation"]
        width = obs["image_width"]
        height = obs["image_height"]
        front = next(v for v in obs["views"] if v["view_type"] == "front")
        return make_go_output(
            view_id=front["view_id"],
            view_type="front",
            point_px=(width // 2, int(height * 0.75)),
            width=width,
            height=height,
            decision_reason="G02_VISIBLE_FLOOR_TOWARD_GOAL",
            goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
            short_text="test selects front floor",
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

    def test_live_pose_relation_updates_after_go_without_new_node(self):
        with tempfile.TemporaryDirectory() as td:
            img = StaticImageBackend.create_demo_image(Path(td) / "img.jpg")
            robot = StaticImageBackend(img, step_m=0.5)
            agent = NavMemoryAgent(
                robot=robot,
                vlm_client=FrontGoVLM(),
                config=NavAgentConfig(max_steps=1, force_new_node_translation_m=10.0),
            )
            result = agent.step(goal_map_xy=(5.0, 0.0), step_index=0)
            self.assertEqual(result.action, "go")
            rel = agent.memory.current_pose_relation_to_latest_node()
            self.assertTrue(rel["valid"])
            self.assertAlmostEqual(rel["latest_node_to_robot"]["dx_m"], 0.5, places=3)
            self.assertEqual(rel["latest_node_id"], agent.memory.current_node_id)

    def test_new_edge_uses_cumulative_latest_node_to_robot_pose(self):
        with tempfile.TemporaryDirectory() as td:
            img = StaticImageBackend.create_demo_image(Path(td) / "img.jpg")
            robot = StaticImageBackend(img, step_m=0.65)
            agent = NavMemoryAgent(
                robot=robot,
                vlm_client=FrontGoVLM(),
                config=NavAgentConfig(max_steps=3, force_new_node_translation_m=1.0),
            )
            agent.step(goal_map_xy=(10.0, 0.0), step_index=0)
            agent.step(goal_map_xy=(10.0, 0.0), step_index=1)
            agent.step(goal_map_xy=(10.0, 0.0), step_index=2)
            self.assertGreaterEqual(len(agent.memory.nodes), 2)
            temporal_edges = [e for e in agent.memory.edges.values() if e.edge_type == "temporal_transition"]
            self.assertTrue(temporal_edges)
            self.assertTrue(any(e.relative_pose_src_to_dst.dx_m >= 1.2 for e in temporal_edges))

    def test_candidate_exit_bearing_uses_current_robot_frame(self):
        g = MemoryGraph(embedding_dim=4)
        n1 = g.add_node(frame_index=1, image_ref=None, embedding=np.array([1, 0, 0, 0], dtype=np.float32))
        n2 = g.add_node(frame_index=2, image_ref=None, embedding=np.array([0, 1, 0, 0], dtype=np.float32))
        g.set_current_node(n1, frame_index=3)
        eid = g.add_or_update_edge(n1, n2, RelativePose2D(0.0, 2.0, 0.0), status="success")
        g.update_current_pose_relation_to_latest_node(n1, RelativePose2D(1.0, 0.0, 0.0), frame_index=3, source="unit_test")
        scored = g.score_candidate_exit(g.edges[eid], goal_bearing_deg=90.0)
        self.assertAlmostEqual(scored["relative_pose_robot_to_dst"]["dx_m"], -1.0, places=3)
        self.assertAlmostEqual(scored["relative_pose_robot_to_dst"]["dy_m"], 2.0, places=3)
        self.assertGreater(scored["bearing_deg_robot"], 90.0)

    def test_negative_branch_guard_blocks_go_overlapping_deadlock_exit(self):
        with tempfile.TemporaryDirectory() as td:
            img = StaticImageBackend.create_demo_image(Path(td) / "img.jpg")
            robot = StaticImageBackend(img, step_m=1.0)
            agent = NavMemoryAgent(robot=robot, vlm_client=FrontGoVLM(), config=NavAgentConfig())
            go = make_go_output(
                view_id=0,
                view_type="front",
                point_px=(320, 360),
                width=640,
                height=480,
                decision_reason="G02_VISIBLE_FLOOR_TOWARD_GOAL",
                goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
                short_text="test attempts known negative branch",
            )
            guarded, warnings = agent._guard_negative_branch(
                go,
                {
                    "local_topology": {
                        "candidate_exits": [
                            {
                                "edge_id": "e_dead",
                                "avoid": True,
                                "bearing_deg_robot": 0.0,
                                "status": "deadlock_entry",
                            }
                        ]
                    }
                },
            )

        self.assertEqual(guarded["action"], "request_observation")
        self.assertEqual(guarded["memory_ops"][0]["op"], "reject_go_into_negative_edge")
        self.assertEqual(guarded["memory_ops"][0]["edge_id"], "e_dead")
        self.assertIn("backend blocked go into known negative branch", warnings)



if __name__ == "__main__":
    unittest.main()
