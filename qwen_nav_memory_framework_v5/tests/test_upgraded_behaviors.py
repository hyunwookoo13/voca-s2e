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
    def test_vlm_confirmed_revisit_soft_links_and_preserves_revisit_node(self):
        g = MemoryGraph(embedding_dim=4)
        emb = np.array([1, 0, 0, 0], dtype=np.float32)
        n1 = g.add_node(frame_index=1, image_ref=None, embedding=emb)
        # Make a real temporal baseline before the revisit node is created.
        n_mid = g.add_node(frame_index=2, image_ref=None, embedding=np.array([0, 1, 0, 0], dtype=np.float32))
        g.add_or_update_edge(n1, n_mid, RelativePose2D(1.0, 0.0, 0.0), frame_index=2)
        loc = g.localize(emb, threshold=0.90)
        self.assertIn(n1, loc.candidate_node_ids)
        n2 = g.add_node(frame_index=3, image_ref=None, embedding=emb)
        g.add_or_update_edge(n_mid, n2, RelativePose2D(0.5, 0.0, 0.0), frame_index=3)
        result = g.commit_revisit(n1, frame_index=3, vlm_confidence=0.95)
        self.assertTrue(result.accepted)
        self.assertEqual(g.current_node_id, n1)
        self.assertNotEqual(g.nodes[n2].lifecycle["storage_tier"], ARCHIVED)
        self.assertTrue(g.nodes[n2].lifecycle.get("preserved_as_revisit_node"))
        self.assertTrue(any(e.edge_type == "revisit_link" and e.src_node_id == n1 and e.dst_node_id == n2 for e in g.edges.values()))
        rel = g.current_pose_relation_to_latest_node()
        self.assertAlmostEqual(rel["latest_node_to_robot"]["dx_m"], 1.5, places=3)

    def test_context_exposes_place_recognition_candidates(self):
        g = MemoryGraph(embedding_dim=4)
        emb = np.array([1, 0, 0, 0], dtype=np.float32)
        n1 = g.add_node(frame_index=1, image_ref="memory.jpg", embedding=emb)
        g.localize(emb, threshold=0.95)
        ctx = g.build_vlm_memory_context(goal_bearing_deg=0.0, goal_distance_m=2.0)
        self.assertEqual(ctx["schema_version"], "nav_memory_context_v5")
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


    def test_relative_pose_between_nodes_composes_chain_rule(self):
        g = MemoryGraph(embedding_dim=4)
        z = np.array([1, 0, 0, 0], dtype=np.float32)
        a = g.add_node(frame_index=1, image_ref=None, embedding=z)
        s = g.add_node(frame_index=2, image_ref=None, embedding=z)
        d = g.add_node(frame_index=3, image_ref=None, embedding=z)
        f = g.add_node(frame_index=4, image_ref=None, embedding=z)
        b = g.add_node(frame_index=5, image_ref=None, embedding=z)
        g.add_or_update_edge(a, s, RelativePose2D(1.0, 0.0, 0.0), frame_index=1)
        g.add_or_update_edge(s, d, RelativePose2D(0.0, 2.0, 90.0), frame_index=2)
        g.add_or_update_edge(d, f, RelativePose2D(1.0, 0.0, 0.0), frame_index=3)
        g.add_or_update_edge(f, b, RelativePose2D(0.0, 1.0, -90.0), frame_index=4)
        expected = RelativePose2D().compose(RelativePose2D(1.0, 0.0, 0.0)).compose(RelativePose2D(0.0, 2.0, 90.0)).compose(RelativePose2D(1.0, 0.0, 0.0)).compose(RelativePose2D(0.0, 1.0, -90.0))
        result = g.relative_pose_between_nodes(a, b)
        self.assertTrue(result.found)
        self.assertEqual(result.node_path, [a, s, d, f, b])
        self.assertAlmostEqual(result.relative_pose_src_to_dst.dx_m, expected.dx_m, places=3)
        self.assertAlmostEqual(result.relative_pose_src_to_dst.dy_m, expected.dy_m, places=3)
        self.assertAlmostEqual(result.relative_pose_src_to_dst.dyaw_deg, expected.dyaw_deg, places=3)

    def test_relative_pose_between_nodes_uses_inverse_edges(self):
        g = MemoryGraph(embedding_dim=4)
        z = np.array([1, 0, 0, 0], dtype=np.float32)
        a = g.add_node(frame_index=1, image_ref=None, embedding=z)
        b = g.add_node(frame_index=2, image_ref=None, embedding=z)
        g.add_or_update_edge(b, a, RelativePose2D(2.0, 0.0, 0.0), frame_index=2)
        result = g.relative_pose_between_nodes(a, b, allow_reverse_edges=True)
        self.assertTrue(result.found)
        self.assertAlmostEqual(result.relative_pose_src_to_dst.dx_m, -2.0, places=3)
        self.assertEqual(result.edge_path[0]["direction_used"], "reverse")

    def test_soft_merge_link_is_not_candidate_exit(self):
        g = MemoryGraph(embedding_dim=4)
        z = np.array([1, 0, 0, 0], dtype=np.float32)
        a = g.add_node(frame_index=1, image_ref=None, embedding=z)
        b = g.add_node(frame_index=2, image_ref=None, embedding=z)
        c = g.add_node(frame_index=3, image_ref=None, embedding=z)
        g.add_or_update_edge(a, b, RelativePose2D(1.0, 0.0, 0.0), frame_index=2)
        g.add_or_update_edge(b, c, RelativePose2D(1.0, 0.0, 0.0), frame_index=3)
        soft = g.soft_merge_revisit(a, c, frame_index=3, reason="unit_test")
        self.assertTrue(soft.accepted)
        g.set_current_node(a, frame_index=4)
        ctx = g.build_vlm_memory_context(goal_bearing_deg=0.0, goal_distance_m=3.0)
        edge_ids = {e.get("edge_id") for e in ctx["local_topology"]["candidate_exits"] if e.get("edge_id")}
        soft_edge_id = soft.details["forward_edge_id"]
        self.assertNotIn(soft_edge_id, edge_ids)
        q = g.relative_pose_between_nodes(a, c, include_constraint_edges=True)
        self.assertTrue(q.found)



    def test_soft_merge_does_not_overwrite_temporal_edge_between_same_nodes(self):
        g = MemoryGraph(embedding_dim=4)
        z = np.array([1, 0, 0, 0], dtype=np.float32)
        a = g.add_node(frame_index=1, image_ref=None, embedding=z)
        b = g.add_node(frame_index=2, image_ref=None, embedding=z)
        temporal = g.add_or_update_edge(a, b, RelativePose2D(1.0, 0.0, 0.0), edge_type="temporal_transition", frame_index=2)
        soft = g.soft_merge_revisit(a, b, frame_index=2, reason="unit_test")
        self.assertTrue(soft.accepted)
        self.assertEqual(g.edges[temporal].edge_type, "temporal_transition")
        self.assertTrue(any(e.edge_type == "revisit_link" and e.src_node_id == a and e.dst_node_id == b for e in g.edges.values()))
        self.assertGreaterEqual(len([e for e in g.edges.values() if e.src_node_id == a and e.dst_node_id == b]), 2)

    def test_spatial_plausibility_rejects_far_visual_alias_revisit(self):
        g = MemoryGraph(embedding_dim=4)
        emb = np.array([1, 0, 0, 0], dtype=np.float32)
        first = g.add_node(frame_index=1, image_ref=None, embedding=emb, semantic_summary="repeating white corridor")
        revisit = g.add_node(frame_index=2, image_ref=None, embedding=emb, semantic_summary="visually similar but distant corridor")
        g.add_or_update_edge(first, revisit, RelativePose2D(10.0, 0.0, 0.0), frame_index=2)
        g.set_current_node(revisit, frame_index=2)
        g.localize(emb, threshold=0.99)
        result = g.commit_revisit(
            first,
            frame_index=2,
            vlm_confidence=0.97,
            min_backend_score=0.10,
            max_same_place_baseline_m=3.0,
            extended_same_place_baseline_m=5.0,
        )
        self.assertFalse(result.accepted)
        self.assertIn("spatial_plausibility_failed", result.reason)
        self.assertIn("spatial_plausibility", result.details)
        self.assertGreater(result.details["spatial_plausibility"]["distance_m"], 5.0)
        self.assertFalse(g.nodes[revisit].lifecycle.get("preserved_as_revisit_node", False))

    def test_revisit_candidate_context_exposes_spatial_plausibility(self):
        g = MemoryGraph(embedding_dim=4)
        emb = np.array([1, 0, 0, 0], dtype=np.float32)
        first = g.add_node(frame_index=1, image_ref="first.jpg", embedding=emb)
        cur = g.add_node(frame_index=2, image_ref="current.jpg", embedding=emb)
        g.add_or_update_edge(first, cur, RelativePose2D(1.2, 0.1, 20.0), frame_index=2)
        g.set_current_node(cur, frame_index=2)
        g.localize(emb, threshold=0.99)
        ctx = g.build_vlm_memory_context(goal_bearing_deg=0.0, goal_distance_m=3.0)
        candidates = ctx["place_recognition"]["revisit_candidates"]
        self.assertTrue(candidates)
        first_candidate = next(c for c in candidates if c["candidate_node_id"] == first)
        self.assertIn("spatial_plausibility", first_candidate)
        self.assertTrue(first_candidate["spatial_plausibility"]["accepted"])
        self.assertLess(first_candidate["spatial_plausibility"]["distance_m"], 3.0)

    def test_corridor_category_gate_rejects_plausible_looking_but_too_far_same_place(self):
        g = MemoryGraph(embedding_dim=4)
        emb = np.array([1, 0, 0, 0], dtype=np.float32)
        first = g.add_node(frame_index=1, image_ref=None, embedding=emb, semantic_summary="repeating corridor")
        revisit = g.add_node(frame_index=2, image_ref=None, embedding=emb, semantic_summary="similar corridor segment")
        g.nodes[first].place_category = "corridor"
        g.nodes[revisit].place_category = "corridor"
        g.add_or_update_edge(first, revisit, RelativePose2D(2.0, 0.0, 0.0), frame_index=2)
        spatial = g.spatial_plausibility_between_nodes(first, revisit, visual_score=0.95, vlm_confidence=0.95)
        self.assertFalse(spatial.accepted)
        self.assertTrue(spatial.same_region_allowed)
        self.assertEqual(spatial.recommended_action, "same_region_only_no_same_place_merge_or_request_observation")
        self.assertEqual(spatial.details["gate_policy"]["src_category"], "corridor")
        self.assertEqual(spatial.details["effective_max_same_place_baseline_m"], 1.5)



if __name__ == "__main__":
    unittest.main()
