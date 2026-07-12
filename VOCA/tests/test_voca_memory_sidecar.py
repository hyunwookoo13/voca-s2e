import tempfile
import unittest
import math
from pathlib import Path

import numpy as np
from PIL import Image

from voca_memory_sidecar import VOCAMemorySidecar
from pixel_candidate_gate import build_pixel_candidates


class VOCAMemorySidecarTests(unittest.TestCase):
    class _MeanColorEmbedder:
        dim = 4

        def embed_image(self, image):
            rgb = np.asarray(Image.open(image).convert("RGB"), dtype=np.float32)
            means = rgb.reshape(-1, 3).mean(axis=0) / 255.0
            vector = np.asarray([means[0], means[1], means[2], 1.0], dtype=np.float32)
            return vector / np.linalg.norm(vector)

    def test_verified_target_memory_survives_unverified_semantic_updates(self):
        sidecar = VOCAMemorySidecar(embedder=None)
        sidecar.reset("chair")
        sidecar.build_context(
            target_object="chair",
            priors={},
            image_ref="/tmp/chair_view.jpg",
            frame_index=10,
            image_shape=(48, 64, 3),
            position_xyz=[0.0, 0.0, 0.0],
            heading_rad=0.5,
        )
        result = sidecar.record_verified_target_evidence(
            {
                "target_evidence_passed": True,
                "confidence": "medium",
                "target_view_index": 0,
                "target_center_x": 0.45,
                "target_bbox_norm": [0.3, 0.3, 0.6, 0.8],
                "target_view_relative_heading_deg": 30.0,
                "target_bearing_robot_deg": 25.5,
                "target_bearing_world_rad": 0.945059,
                "identity_critic": {
                    "triggered": True,
                    "passed": True,
                    "exact_target": True,
                    "confidence": "high",
                    "reason": "conventional chair geometry",
                },
            },
            frame_index=10,
            image_ref="/tmp/chair_view.jpg",
        )
        self.assertTrue(result["updated"])
        self.assertEqual(result["heading_rad"], 0.5)
        self.assertEqual(result["target_bearing_robot_deg"], 25.5)
        self.assertEqual(result["target_bearing_world_rad"], 0.945059)

        sidecar.record_semantic_state(
            {
                "current_room": "living_room",
                "target_evidence": "none",
                "navigation_intent": "search_current_room",
                "room_search_status": "partial",
            },
            frame_index=11,
        )
        node = sidecar.memory.nodes[sidecar.memory.current_node_id]
        self.assertTrue(node.semantic["target_hypothesis"]["verified"])
        self.assertTrue(node.semantic["object_belief"]["seen_target"])
        self.assertEqual(
            node.semantic["object_belief"]["candidate_objects"][-1]["image_ref"],
            "/tmp/chair_view.jpg",
        )
        self.assertEqual(sidecar.runtime_stats["object_belief_updates"], 1)
        self.assertEqual(
            node.semantic["object_belief"]["candidate_objects"][-1][
                "identity_confidence"
            ],
            "high",
        )
        self.assertTrue(
            sidecar.visualizer_state()["last_verified_target_evidence"][
                "updated"
            ]
        )

    def test_visual_descriptor_proposes_spatially_plausible_revisit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "first.jpg"
            revisit = Path(tmpdir) / "revisit.jpg"
            Image.new("RGB", (64, 48), (40, 90, 130)).save(first)
            Image.new("RGB", (64, 48), (40, 90, 130)).save(revisit)
            sidecar = VOCAMemorySidecar(
                force_new_node_translation_m=0.5,
                embedder=self._MeanColorEmbedder(),
            )
            sidecar.reset("chair")
            sidecar.build_context(
                target_object="chair",
                priors={},
                image_ref=str(first),
                frame_index=0,
                image_shape=(48, 64, 3),
                position_xyz=[0.0, 0.0, 0.0],
            )
            self.assertEqual(len(sidecar.memory.visual_index), 1)
            sidecar.record_go_execution(
                go_execution={
                    "collision_count": 0,
                    "go_progress": {"execution_success": True, "strategic_progress": True},
                },
                frame_index=1,
                start_position_xyz=[0.0, 0.0, 0.0],
                final_position_xyz=[1.0, 0.0, 0.0],
            )
            context = sidecar.build_context(
                target_object="chair",
                priors={},
                image_ref=str(revisit),
                frame_index=2,
                image_shape=(48, 64, 3),
                position_xyz=[1.0, 0.0, 0.0],
            )

            revisits = context["candidate_refs"]["revisits"]
            self.assertTrue(revisits)
            self.assertEqual(revisits[0]["candidate_node_id"], "n_00001")
            self.assertGreaterEqual(revisits[0]["visual_retrieval_score"], 0.99)
            self.assertTrue(context["retrieved_memory_images"])
            op_results = sidecar.apply_vlm_memory_ops(
                {
                    "memory_ops": [
                        {
                            "op": "confirm_revisit_node",
                            "candidate_ref": revisits[0]["candidate_ref"],
                            "confidence": 0.95,
                            "reason": "same color and room layout in test observation",
                        }
                    ]
                },
                frame_index=2,
            )
            self.assertTrue(op_results[0]["accepted"])
            self.assertIn("revisit", op_results[0]["reason"])
            state = sidecar.visualizer_state()
            self.assertEqual(state["runtime_stats"]["embedding_observations"], 2)
            self.assertEqual(state["runtime_stats"]["revisit_queries"], 1)
            self.assertEqual(state["runtime_stats"]["revisit_candidate_contexts"], 1)
            self.assertEqual(state["runtime_stats"]["memory_ops_requested"], 1)
            self.assertEqual(state["runtime_stats"]["memory_ops_accepted"], 1)
            self.assertEqual(state["runtime_stats"]["revisits_confirmed"], 1)
            self.assertEqual(state["runtime_stats"]["soft_merges"], 1)

    def test_memory_op_requires_backend_revisit_ref(self):
        sidecar = VOCAMemorySidecar(embedder=None)
        sidecar.reset("chair")
        sidecar.build_context(
            target_object="chair",
            priors={},
            image_ref="/tmp/start.jpg",
            frame_index=0,
            image_shape=(48, 64, 3),
            position_xyz=[0.0, 0.0, 0.0],
        )
        results = sidecar.apply_vlm_memory_ops(
            {
                "memory_ops": [
                    {
                        "op": "confirm_revisit_node",
                        "candidate_ref": "invented_ref",
                        "confidence": 0.99,
                    }
                ]
            },
            frame_index=1,
        )
        self.assertFalse(results[0]["accepted"])
        self.assertEqual(results[0]["reason"], "unknown_or_missing_revisit_candidate_ref")
        self.assertEqual(sidecar.visualizer_state()["runtime_stats"]["memory_ops_rejected"], 1)

    def test_deferred_revisit_is_audited_without_topology_mutation(self):
        sidecar = VOCAMemorySidecar(embedder=None)
        sidecar.reset("chair")
        sidecar.build_context(
            target_object="chair",
            priors={},
            image_ref="/tmp/start.jpg",
            frame_index=0,
            image_shape=(48, 64, 3),
            position_xyz=[0.0, 0.0, 0.0],
        )
        sidecar.last_context.setdefault("candidate_refs", {})["revisits"] = [
            {
                "candidate_ref": "revisit_001",
                "candidate_node_id": "n_00001",
            }
        ]
        before_nodes = len(sidecar.memory.nodes)
        before_edges = len(sidecar.memory.edges)

        results = sidecar.apply_vlm_memory_ops(
            {
                "memory_ops": [
                    {
                        "op": "defer_revisit_candidate",
                        "candidate_ref": "revisit_001",
                        "confidence": 0.0,
                        "reason": "insufficient visual evidence",
                    }
                ]
            },
            frame_index=1,
        )

        self.assertTrue(results[0]["accepted"])
        self.assertEqual(results[0]["reason"], "revisit_deferred_no_topology_mutation")
        self.assertEqual(len(sidecar.memory.nodes), before_nodes)
        self.assertEqual(len(sidecar.memory.edges), before_edges)

    def test_collision_blocks_failed_view_but_preserves_backtrack_candidates(self):
        sidecar = VOCAMemorySidecar(force_new_node_translation_m=0.5)
        sidecar.reset("chair")
        sidecar.build_context(
            target_object="chair",
            priors={},
            image_ref="/tmp/start.jpg",
            frame_index=0,
            image_shape=(480, 640, 3),
            position_xyz=[0.0, 0.0, 0.0],
            heading_rad=0.0,
        )
        sidecar.record_go_execution(
            go_execution={
                "collision_count": 0,
                "go_progress": {"execution_success": True, "strategic_progress": True},
            },
            frame_index=10,
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[1.0, 0.0, 0.0],
            start_heading_rad=0.0,
            final_heading_rad=0.0,
        )
        sidecar.build_context(
            target_object="chair",
            priors={},
            image_ref="/tmp/current.jpg",
            frame_index=11,
            image_shape=(480, 640, 3),
            position_xyz=[1.0, 0.0, 0.0],
            heading_rad=0.0,
        )
        sidecar.record_go_execution(
            go_execution={
                "selected_view_id": 0,
                "selected_point_px": [320, 420],
                "collision_count": 1,
                "failure_classification": {"primary": "collision_blocked"},
                "go_progress": {
                    "execution_success": False,
                    "strategic_progress": False,
                    "no_progress": True,
                },
            },
            frame_index=12,
            start_position_xyz=[1.0, 0.0, 0.0],
            final_position_xyz=[1.0, 0.0, 0.0],
            start_heading_rad=0.0,
            final_heading_rad=0.0,
        )
        context = sidecar.build_context(
            target_object="chair",
            priors={},
            image_ref="/tmp/failed.jpg",
            frame_index=13,
            image_shape=(480, 640, 3),
            position_xyz=[1.0, 0.0, 0.0],
            heading_rad=0.0,
        )
        candidates = build_pixel_candidates(
            [
                {"view_id": 0, "view_type": "front", "relative_heading_deg": 0},
                {"view_id": 1, "view_type": "back", "relative_heading_deg": 180},
            ],
            (480, 640, 3),
            context,
        )
        front = [candidate for candidate in candidates if candidate["view_id"] == 0]
        back = [candidate for candidate in candidates if candidate["view_id"] == 1]

        self.assertTrue(front)
        self.assertTrue(all(candidate["avoid"] for candidate in front))
        self.assertTrue(back)
        self.assertTrue(all(not candidate["avoid"] for candidate in back))
        self.assertTrue(all(candidate["topological_relation_type"] == "backtrack" for candidate in back))
        self.assertEqual(sidecar.supervisor_mode, "escape_deadlock")

    def test_verified_directional_failure_is_bounded_in_policy_context_and_reset(self):
        sidecar = VOCAMemorySidecar()
        sidecar.reset("chair")
        sidecar.build_context(
            target_object="chair",
            priors={},
            image_ref="/tmp/start.jpg",
            frame_index=0,
            image_shape=(480, 640, 3),
            position_xyz=[0.0, 0.0, 0.0],
        )

        sidecar.record_go_execution(
            go_execution={
                "selected_angle_deg": 30,
                "selected_view_id": 2,
                "selected_point_px": [420, 350],
                "selected_candidate_ref": "px_v02_r1_c2",
                "selected_topological_candidate_ref": "exit_003",
                "collision_count": 1,
                "failure_classification": {"primary": "collision_blocked"},
                "go_progress": {
                    "execution_success": False,
                    "strategic_progress": False,
                    "no_progress": True,
                    "translation_m": 0.0,
                },
            },
            frame_index=7,
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.0, 0.0, 0.0],
        )
        context = sidecar.build_context(
            target_object="chair",
            priors={},
            image_ref="/tmp/failed.jpg",
            frame_index=7,
            image_shape=(480, 640, 3),
            position_xyz=[0.0, 0.0, 0.0],
        )

        self.assertEqual(len(sidecar.directional_failures), 1)
        failure = context["voca_sidecar"]["directional_failures"][0]
        self.assertEqual(failure["frame_index"], 7)
        self.assertEqual(failure["angle_deg"], 30.0)
        self.assertEqual(failure["selected_view_id"], 2)
        self.assertEqual(failure["point_px"], [420, 350])
        self.assertEqual(failure["selected_candidate_ref"], "px_v02_r1_c2")
        self.assertEqual(failure["selected_topological_candidate_ref"], "exit_003")
        self.assertEqual(failure["failure_class"], "collision_blocked")
        self.assertEqual(failure["collision_count"], 1)
        self.assertEqual(failure["negative_edge_status"], "blocked")
        self.assertIn(failure["negative_edge_id"], sidecar.memory.edges)
        self.assertEqual(sidecar.memory.current_node_id, "n_00001")

        sidecar.record_go_execution(
            go_execution={
                "selected_angle_deg": -30,
                "selected_view_id": 0,
                "selected_point_px": [220, 360],
                "collision_count": 0,
                "go_progress": {
                    "execution_success": True,
                    "strategic_progress": True,
                    "no_progress": False,
                    "translation_m": 0.3,
                },
            },
            frame_index=8,
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.3, 0.0, 0.0],
        )

        self.assertEqual(len(sidecar.directional_failures), 1)
        sidecar.reset("chair")
        self.assertEqual(sidecar.directional_failures, [])

    def test_repeated_no_progress_enters_escape_and_verified_motion_resumes_goal_seek(self):
        sidecar = VOCAMemorySidecar(force_new_node_translation_m=0.25)
        sidecar.reset("chair")
        sidecar.build_context(
            target_object="chair",
            priors={},
            image_ref="/tmp/start.jpg",
            frame_index=0,
            image_shape=(480, 640, 3),
            position_xyz=[0.0, 0.0, 0.0],
        )

        failed_execution = {
            "collision_count": 0,
            "go_progress": {
                "execution_success": False,
                "strategic_progress": False,
                "no_progress": True,
                "translation_m": 0.0,
                "supervisor_mode": "goal_seek",
            },
        }
        for frame_index in (1, 2):
            sidecar.record_go_execution(
                go_execution=failed_execution,
                frame_index=frame_index,
                start_position_xyz=[0.0, 0.0, 0.0],
                final_position_xyz=[0.0, 0.0, 0.0],
            )

        self.assertEqual(sidecar.supervisor_mode, "escape_deadlock")
        escape_context = sidecar.build_context(
            target_object="chair",
            priors={},
            image_ref="/tmp/escape.jpg",
            frame_index=2,
            image_shape=(480, 640, 3),
            position_xyz=[0.0, 0.0, 0.0],
        )
        self.assertEqual(escape_context["goal_context"]["supervisor_mode"], "escape_deadlock")
        self.assertEqual(escape_context["policy_harness_state"]["current_stage"], "escape_deadlock")

        sidecar.record_go_execution(
            go_execution={
                "collision_count": 0,
                "go_progress": {
                    "execution_success": True,
                    "strategic_progress": True,
                    "no_progress": False,
                    "translation_m": 0.4,
                    "supervisor_mode": "escape_deadlock",
                },
            },
            frame_index=3,
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.4, 0.0, 0.0],
        )

        state = sidecar.visualizer_state()
        self.assertEqual(sidecar.supervisor_mode, "goal_seek")
        self.assertEqual(state["supervisor_mode"], "goal_seek")
        self.assertEqual(state["last_mode_transition"]["from"], "escape_deadlock")
        self.assertEqual(state["last_mode_transition"]["to"], "goal_seek")
        self.assertEqual(state["last_mode_transition"]["reason"], "verified_escape_progress")

    def test_short_successful_motion_does_not_accumulate_no_progress(self):
        sidecar = VOCAMemorySidecar(force_new_node_translation_m=0.75)
        sidecar.reset("chair")
        sidecar.build_context(
            target_object="chair",
            priors={},
            image_ref="/tmp/start.jpg",
            frame_index=0,
            image_shape=(480, 640, 3),
            position_xyz=[0.0, 0.0, 0.0],
        )

        sidecar.record_go_execution(
            go_execution={
                "collision_count": 0,
                "go_progress": {
                    "execution_success": True,
                    "strategic_progress": True,
                    "no_progress": False,
                    "translation_m": 0.2,
                },
            },
            frame_index=1,
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.2, 0.0, 0.0],
        )

        self.assertEqual(sidecar.no_progress_count, 0)
        self.assertEqual(sidecar.supervisor_mode, "goal_seek")

    def test_successful_short_go_segments_accumulate_from_node_anchor(self):
        sidecar = VOCAMemorySidecar(force_new_node_translation_m=0.5)
        sidecar.reset("chair")
        sidecar.build_context(
            target_object="chair",
            priors={},
            image_ref="/tmp/start.jpg",
            frame_index=0,
            image_shape=(480, 640, 3),
            position_xyz=[0.0, 0.0, 0.0],
            heading_rad=0.0,
        )
        success = {
            "collision_count": 0,
            "go_progress": {
                "execution_success": True,
                "strategic_progress": True,
                "no_progress": False,
            },
        }

        sidecar.record_go_execution(
            go_execution=success,
            frame_index=1,
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.3, 0.0, 0.0],
        )
        self.assertEqual(len(sidecar.memory.nodes), 1)

        sidecar.record_go_execution(
            go_execution=success,
            frame_index=2,
            start_position_xyz=[0.3, 0.0, 0.0],
            final_position_xyz=[0.6, 0.0, 0.0],
        )

        self.assertEqual(len(sidecar.memory.nodes), 2)
        self.assertEqual(len(sidecar.memory.edges), 2)
        edge = next(
            edge
            for edge in sidecar.memory.edges.values()
            if edge.relation_type == "transition"
        )
        self.assertAlmostEqual(edge.relative_pose_src_to_dst.dx_m, 0.6, places=5)
        self.assertAlmostEqual(edge.relative_pose_src_to_dst.dy_m, 0.0, places=5)
        reverse = next(
            edge
            for edge in sidecar.memory.edges.values()
            if edge.relation_type == "backtrack"
        )
        self.assertAlmostEqual(reverse.relative_pose_src_to_dst.dx_m, -0.6, places=5)

    def test_relative_edge_uses_habitat_heading_and_records_yaw(self):
        sidecar = VOCAMemorySidecar(force_new_node_translation_m=0.25)
        sidecar.reset("chair")
        sidecar.build_context(
            target_object="chair",
            priors={},
            image_ref="/tmp/start.jpg",
            frame_index=0,
            image_shape=(480, 640, 3),
            position_xyz=[0.0, 0.0, 0.0],
            heading_rad=math.pi / 2.0,
        )
        sidecar.record_go_execution(
            go_execution={
                "collision_count": 0,
                "go_progress": {"execution_success": True, "strategic_progress": True},
            },
            frame_index=1,
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.0, 0.0, 0.4],
            start_heading_rad=math.pi / 2.0,
            final_heading_rad=math.pi / 2.0 + math.radians(30.0),
        )

        edge = next(
            edge
            for edge in sidecar.memory.edges.values()
            if edge.relation_type == "transition"
        )
        self.assertAlmostEqual(edge.relative_pose_src_to_dst.dx_m, 0.4, places=5)
        self.assertAlmostEqual(edge.relative_pose_src_to_dst.dy_m, 0.0, places=5)
        self.assertAlmostEqual(edge.relative_pose_src_to_dst.dyaw_deg, 30.0, places=5)

    def test_backtrack_candidate_uses_executed_arrival_tangent_not_pose_chord(self):
        sidecar = VOCAMemorySidecar(force_new_node_translation_m=0.25)
        sidecar.reset("chair")
        sidecar.build_context(
            target_object="chair",
            priors={},
            image_ref="/tmp/start.jpg",
            frame_index=0,
            image_shape=(480, 640, 3),
            position_xyz=[0.0, 0.0, 0.0],
            heading_rad=0.0,
        )
        sidecar.record_go_execution(
            go_execution={
                "collision_count": 0,
                "go_progress": {
                    "execution_success": True,
                    "strategic_progress": True,
                },
                "executed_path_tangents": {
                    "departure_heading_world_rad": 0.0,
                    "arrival_heading_world_rad": math.pi / 2.0,
                    "path_length_m": 1.5,
                },
            },
            frame_index=1,
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[1.0, 0.0, 0.0],
            start_heading_rad=0.0,
            final_heading_rad=math.pi / 2.0,
        )

        context = sidecar.build_context(
            target_object="chair",
            priors={},
            image_ref="/tmp/final.jpg",
            frame_index=2,
            image_shape=(480, 640, 3),
            position_xyz=[1.0, 0.0, 0.0],
            heading_rad=math.pi / 2.0,
        )
        backtrack = next(
            candidate
            for candidate in context["candidate_refs"]["exits"]
            if candidate.get("relation_type") == "backtrack"
        )

        self.assertAlmostEqual(backtrack["bearing_deg_robot"], -180.0)
        self.assertEqual(backtrack["bearing_source"], "executed_position_trace")
        self.assertAlmostEqual(backtrack["chord_bearing_deg_robot"], 90.0)

    def test_backtrack_candidate_looks_ahead_on_reversed_executed_polyline(self):
        sidecar = VOCAMemorySidecar(force_new_node_translation_m=0.25)
        sidecar.reset("chair")
        sidecar.build_context(
            target_object="chair",
            priors={},
            image_ref="/tmp/start.jpg",
            frame_index=0,
            image_shape=(480, 640, 3),
            position_xyz=[0.0, 0.0, 0.0],
            heading_rad=0.0,
        )
        sidecar.record_go_execution(
            go_execution={
                "collision_count": 0,
                "go_progress": {
                    "execution_success": True,
                    "strategic_progress": True,
                },
                "executed_path_tangents": {
                    "departure_heading_world_rad": 0.0,
                    "arrival_heading_world_rad": math.pi / 2.0,
                    "path_length_m": 1.5,
                    "position_trace_xyz": [
                        [0.0, 0.0, 0.0],
                        [0.5, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.5],
                    ],
                },
            },
            frame_index=1,
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[1.0, 0.0, 0.5],
            start_heading_rad=0.0,
            final_heading_rad=math.pi / 2.0,
        )
        context = sidecar.build_context(
            target_object="chair",
            priors={},
            image_ref="/tmp/final.jpg",
            frame_index=2,
            image_shape=(480, 640, 3),
            position_xyz=[1.0, 0.0, 0.5],
            heading_rad=math.pi / 2.0,
        )
        backtrack = next(
            candidate
            for candidate in context["candidate_refs"]["exits"]
            if candidate.get("relation_type") == "backtrack"
        )

        self.assertEqual(backtrack["bearing_source"], "executed_path_polyline")
        self.assertAlmostEqual(backtrack["bearing_deg_robot"], 135.0)
        self.assertEqual(
            backtrack["path_guidance"]["selected_polyline_index"],
            2,
        )

    def test_build_context_creates_v6_memory_node_and_visualizer_state(self):
        sidecar = VOCAMemorySidecar()
        sidecar.reset("chair")

        context = sidecar.build_context(
            target_object="chair",
            priors={"Gateways": ["living room doorway"]},
            image_ref="/tmp/current.jpg",
            frame_index=0,
            image_shape=(480, 640, 3),
            position_xyz=[0.0, 0.0, 0.0],
            supervisor_mode="goal_seek",
        )
        visualizer = sidecar.visualizer_state()

        self.assertEqual(context["schema_version"], "nav_memory_context_v6")
        self.assertEqual(context["graph_summary"]["num_nodes"], 1)
        self.assertTrue(
            context["policy_harness_state"]["requirements"]["go_requires_selected_candidate_ref"]
        )
        self.assertTrue(
            context["policy_harness_state"]["requirements"]["revisit_memory_ops_require_candidate_ref"]
        )
        self.assertEqual(context["voca_sidecar"]["target_context_priors"]["Gateways"], ["living room doorway"])
        self.assertFalse(context["goal_context"]["goal_geometry_available"])
        self.assertIsNone(context["goal_context"]["goal_bearing_from_current_deg"])
        self.assertIsNone(context["goal_context"]["goal_distance_m"])
        self.assertEqual(context["goal_context"]["supervisor_mode"], "goal_seek")
        self.assertTrue(visualizer["enabled"])
        self.assertEqual(visualizer["schema_version"], "nav_memory_context_v6")
        self.assertEqual(visualizer["num_nodes"], 1)
        self.assertEqual(visualizer["last_event"]["event_type"], "add_node")

    def test_build_context_propagates_declared_spatial_goal(self):
        sidecar = VOCAMemorySidecar()
        sidecar.reset("chair")

        context = sidecar.build_context(
            target_object="chair",
            priors={},
            image_ref="/tmp/current.jpg",
            frame_index=0,
            image_shape=(480, 640, 3),
            position_xyz=[0.0, 0.0, 0.0],
            supervisor_mode="goal_seek",
            spatial_goal={
                "type": "map_waypoint",
                "source": "s2e_backbone",
                "map_xy": [2.0, -1.0],
                "relative_bearing_deg": -35.0,
                "distance_m": 3.5,
                "distance_range_m": [2.5, 4.5],
                "uncertainty": "medium",
            },
        )

        goal = context["goal_context"]
        self.assertEqual(goal["task_mode"], "CoarseGoalNav")
        self.assertTrue(goal["goal_geometry_available"])
        self.assertEqual(goal["goal_bearing_from_current_deg"], -35.0)
        self.assertEqual(goal["goal_distance_m"], 3.5)
        self.assertEqual(goal["coarse_goal"]["source"], "s2e_backbone")

    def test_record_go_execution_adds_success_edge_and_deadlock_memory(self):
        sidecar = VOCAMemorySidecar(force_new_node_translation_m=0.25)
        sidecar.reset("toilet")
        sidecar.build_context(
            target_object="toilet",
            priors={},
            image_ref="/tmp/start.jpg",
            frame_index=0,
            image_shape=(480, 640, 3),
            position_xyz=[0.0, 0.0, 0.0],
            supervisor_mode="goal_seek",
        )

        sidecar.record_go_execution(
            go_execution={
                "collision_count": 0,
                "go_progress": {"no_progress": False, "distance_delta_m": 0.4},
            },
            frame_index=1,
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.4, 0.0, 0.0],
        )
        success_state = sidecar.visualizer_state()

        self.assertEqual(success_state["num_nodes"], 2)
        self.assertEqual(success_state["num_edges"], 2)

        sidecar.record_go_execution(
            go_execution={
                "collision_count": 1,
                "go_progress": {"no_progress": True, "distance_delta_m": -0.1},
            },
            frame_index=2,
            start_position_xyz=[0.4, 0.0, 0.0],
            final_position_xyz=[0.4, 0.0, 0.0],
        )
        deadlock_state = sidecar.visualizer_state()

        self.assertGreaterEqual(deadlock_state["num_deadlock_edges"], 1)
        self.assertTrue(
            any(
                event.get("event_type") == "record_directional_failure"
                for event in sidecar.memory.event_log
            )
        )
        incoming = sidecar.memory.find_edge("n_00001", "n_00002")
        self.assertEqual(sidecar.memory.edges[incoming].traversal["status"], "success")
        self.assertEqual(deadlock_state["supervisor_mode"], "escape_deadlock")

    def test_terminal_collision_after_long_progress_keeps_transition_memory(self):
        sidecar = VOCAMemorySidecar(force_new_node_translation_m=0.25)
        sidecar.reset("toilet")
        sidecar.build_context(
            target_object="toilet",
            priors={},
            image_ref="/tmp/start.jpg",
            frame_index=0,
            image_shape=(480, 640, 3),
            position_xyz=[0.0, 0.0, 0.0],
            supervisor_mode="goal_seek",
        )

        sidecar.record_go_execution(
            go_execution={
                "collision_count": 1,
                "go_progress": {
                    "execution_success": True,
                    "strategic_progress": True,
                    "collision_tolerant_progress": True,
                    "no_progress": False,
                },
            },
            frame_index=1,
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[1.8, 0.0, 0.0],
        )
        state = sidecar.visualizer_state()

        self.assertEqual(state["num_place_nodes"], 2)
        self.assertEqual(state["num_failed_frontier_nodes"], 0)
        self.assertEqual(state["num_edges"], 2)
        self.assertTrue(
            any(
                edge.relation_type == "transition"
                and edge.traversal.get("status") == "success"
                for edge in sidecar.memory.edges.values()
            )
        )
        self.assertTrue(
            any(
                edge.relation_type == "backtrack"
                for edge in sidecar.memory.edges.values()
            )
        )

    def test_semantic_state_updates_room_without_promoting_target_hypothesis(self):
        sidecar = VOCAMemorySidecar(embedder=None)
        sidecar.reset("chair")
        sidecar.build_context(
            target_object="chair",
            priors={},
            image_ref="/tmp/current.jpg",
            frame_index=0,
            image_shape=(480, 640, 3),
            position_xyz=[0.0, 0.0, 0.0],
        )

        update = sidecar.record_semantic_state(
            {
                "current_room": "living_room",
                "target_evidence": "confirmed",
                "navigation_intent": "verify_target",
                "room_search_status": "target_found",
                "selected_exit_ref": None,
                "reason_code": "TARGET_VISIBLE_MODEL_CLAIM",
            },
            frame_index=4,
        )

        node = sidecar.memory.nodes[sidecar.memory.current_node_id]
        self.assertTrue(update["updated"])
        self.assertEqual(node.place_category, "living_room")
        self.assertFalse(node.semantic["target_hypothesis"]["verified"])
        self.assertNotIn("object_belief", node.semantic)
        state = sidecar.visualizer_state()
        self.assertEqual(state["last_strategic_state"]["current_room"], "living_room")
        self.assertEqual(state["runtime_stats"]["semantic_state_updates"], 1)
        self.assertEqual(state["runtime_stats"]["room_category_updates"], 1)
        self.assertEqual(state["graph_layout"]["nodes"][0]["place_category"], "living_room")

    def test_save_artifacts_writes_v6_memory_graph_and_feedback(self):
        sidecar = VOCAMemorySidecar()
        sidecar.reset("plant")
        sidecar.build_context(
            target_object="plant",
            priors={},
            image_ref="/tmp/plant.jpg",
            frame_index=0,
            image_shape=(480, 640, 3),
            position_xyz=[0.0, 0.0, 0.0],
            supervisor_mode="goal_seek",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = sidecar.save_artifacts(Path(tmpdir))
            graph_text = Path(paths["memory_graph_json"]).read_text(encoding="utf-8")
            feedback_text = Path(paths["feedback_md"]).read_text(encoding="utf-8")
            self.assertTrue(Path(paths["memory_graph_png"]).exists())
            self.assertTrue(Path(paths["memory_graph_html"]).exists())
            self.assertTrue(Path(paths["memory_graph_video"]).exists())

        self.assertIn("relative_topometric_memory_graph_v6", graph_text)
        self.assertIn("VOCA Memory Sidecar", feedback_text)


if __name__ == "__main__":
    unittest.main()
