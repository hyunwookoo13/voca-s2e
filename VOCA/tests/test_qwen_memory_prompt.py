import json
import unittest

from qwen_memory_prompt import (
    build_compact_memory_projection,
    compact_memory_json,
)


def _oversized_vlm_input():
    exits = []
    for index, score in enumerate((0.2, 0.9, 0.6, 0.8, 0.7, 0.95)):
        exits.append(
            {
                "candidate_ref": "exit_{:03d}".format(index),
                "topological_candidate_ref": "topo_{:03d}".format(index),
                "topological_edge_id": "edge_{:03d}".format(index),
                "topological_relation_type": "backtrack" if index == 1 else "directional_failure",
                "marker_id": "M{:02d}".format(index + 1),
                "view_id": index,
                "view_type_hint": ("front", "left", "right", "back")[index % 4],
                "bearing_deg_robot": float(index * 30),
                "point_px": [100 + index, 200 + index],
                "status": "blocked" if index == 5 else "unknown_frontier",
                "avoid": index == 5,
                "score": score,
                "reason": "candidate {} ".format(index) + ("x" * 180),
                "image_ref": "/tmp/private-exit-{}.jpg".format(index),
            }
        )
    failures = [
        {
            "frame_index": index,
            "angle_deg": float(index * 10),
            "selected_view_id": index % 3,
            "point_px": [300 + index, 350],
            "failure_class": "collision_blocked" if index % 2 else "no_progress",
            "collision_count": index % 2,
            "translation_m": 0.0,
            "supervisor_mode": "escape_deadlock",
        }
        for index in range(6)
    ]
    negatives = [
        {
            "node_id": "node_{}".format(index),
            "summary": "negative {}".format(index),
            "image_ref": "/tmp/private-negative-{}.jpg".format(index),
            "severity": index / 10.0,
            "negative_memory": {
                "reason": "failed branch {}".format(index),
                "avoid_scope": "incoming_edge_only",
                "failed_entry_edge_id": "edge_{}".format(index),
            },
        }
        for index in range(5)
    ]
    revisits = [
        {
            "candidate_ref": "revisit_{:03d}".format(index),
            "candidate_node_id": "node_{}".format(index),
            "visual_retrieval_score": index / 10.0,
            "candidate_image_ref": "/tmp/private-revisit-{}.jpg".format(index),
            "spatial_plausibility": {"accepted": index % 2 == 0},
        }
        for index in range(5)
    ]
    return {
        "memory": {
            "schema_version": "nav_memory_context_v6",
            "goal_context": {"supervisor_mode": "escape_deadlock"},
            "policy_harness_state": {"current_stage": "escape_deadlock"},
            "graph_summary": {
                "num_nodes": 12,
                "num_place_nodes": 10,
                "num_failed_frontier_nodes": 2,
                "num_edges": 11,
                "num_deadlock_edges": 3,
                "full_graph_path": "/tmp/private-graph.json",
            },
            "deadlock_state": {
                "status": "confirmed",
                "incoming_edge_id": "edge_failed",
                "policy": "avoid incoming edge",
            },
            "loop_warning": {"is_looping": True, "repeated_branch_count": 4},
            "candidate_refs": {"exits": exits, "revisits": revisits},
            "compressed_negative_memories": negatives,
            "voca_sidecar": {
                "no_progress_count": 2,
                "supervisor_mode": "escape_deadlock",
                "directional_failures": failures,
                "position_xyz": [100.0, 0.0, 200.0],
            },
            "event_log": [{"event_type": "private", "payload": "do not expose"}],
        }
    }


class QwenMemoryPromptTests(unittest.TestCase):
    def test_projection_is_bounded_deterministic_and_allowlisted(self):
        vlm_input = _oversized_vlm_input()

        first = build_compact_memory_projection(vlm_input)
        second = build_compact_memory_projection(vlm_input)
        encoded = compact_memory_json(vlm_input)

        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], "compact_nav_memory_v2")
        self.assertTrue(first["enabled"])
        self.assertEqual(first["supervisor_mode"], "escape_deadlock")
        self.assertEqual(first["current_stage"], "escape_deadlock")
        self.assertEqual(
            first["graph"],
            {
                "nodes": 12,
                "place_nodes": 10,
                "failed_frontiers": 2,
                "edges": 11,
                "negative_edges": 3,
            },
        )
        self.assertEqual(first["deadlock"]["status"], "confirmed")
        self.assertTrue(first["loop"]["is_looping"])
        self.assertEqual(first["no_progress_count"], 2)
        self.assertEqual(first["strategy"]["last_room"], "unknown")
        self.assertFalse(first["strategy"]["force_leave_room"])
        self.assertEqual(len(first["candidate_exits"]), 4)
        self.assertEqual(first["candidate_exits"][0]["candidate_ref"], "exit_001")
        self.assertEqual(first["candidate_exits"][0]["topological_ref"], "topo_001")
        self.assertEqual(first["candidate_exits"][0]["topological_edge_ref"], "edge_001")
        self.assertEqual(first["candidate_exits"][0]["topological_relation"], "backtrack")
        self.assertEqual(first["candidate_exits"][0]["marker_id"], "M02")
        self.assertEqual(first["candidate_exits"][0]["view_id"], 1)
        self.assertEqual(first["candidate_exits"][0]["point_px"], [101, 201])
        self.assertFalse(first["candidate_exits"][0]["avoid"])
        self.assertEqual(len(first["directional_failures"]), 4)
        self.assertEqual(first["directional_failures"][0]["frame_index"], 5)
        self.assertEqual(len(first["negative_memories"]), 3)
        self.assertEqual(len(first["revisit_candidates"]), 3)
        self.assertLessEqual(len(first["candidate_exits"][0]["reason"]), 120)
        self.assertEqual(encoded, json.dumps(first, sort_keys=True, separators=(",", ":")))
        for forbidden in (
            "/tmp/",
            "position_xyz",
            "event_log",
            "full_graph_path",
            "private",
            "distance_to_goal",
            "top_down_map",
        ):
            self.assertNotIn(forbidden, encoded)

    def test_projection_handles_missing_or_disabled_memory(self):
        projection = build_compact_memory_projection(
            {"memory": {"schema_version": "null_memory_context_v0", "enabled": False}}
        )

        self.assertFalse(projection["enabled"])
        self.assertEqual(projection["supervisor_mode"], "goal_seek")
        self.assertEqual(projection["candidate_exits"], [])
        self.assertEqual(projection["directional_failures"], [])

    def test_projection_exposes_only_bounded_strategic_runtime(self):
        projection = build_compact_memory_projection(
            {
                "memory": {"schema_version": "nav_memory_context_v6"},
                "runtime": {
                    "strategic_runtime": {
                        "last_strategic_state": {
                            "current_room": "living_room",
                            "target_evidence": "context_only",
                            "navigation_intent": "leave_current_room",
                            "room_search_status": "exhausted",
                        },
                        "same_room_cycles": 8,
                        "generic_floor_go_streak": 4,
                        "force_leave_room": True,
                        "force_reason": "generic_floor_go_streak",
                        "recent_selected_exit_refs": ["exit_a", "exit_b"],
                        "private_pose": [1.0, 2.0, 3.0],
                    }
                },
            }
        )

        self.assertEqual(projection["strategy"]["last_room"], "living_room")
        self.assertEqual(projection["strategy"]["same_room_cycles"], 8)
        self.assertTrue(projection["strategy"]["force_leave_room"])
        self.assertEqual(projection["strategy"]["recent_exit_refs"], ["exit_a", "exit_b"])
        self.assertNotIn("private_pose", json.dumps(projection, sort_keys=True))

    def test_projection_exposes_bounded_identity_critic_rejections(self):
        projection = build_compact_memory_projection(
            {
                "memory": {
                    "schema_version": "nav_memory_context_v6",
                    "runtime_navigation_feedback": [
                        {
                            "event": "target_lookalike_rejected",
                            "target": "chair",
                            "frame_index": index,
                            "target_view_index": 0,
                            "target_bbox_norm": [0.1, 0.2, 0.8, 0.9],
                            "lookalike_type": "incline exercise bench",
                            "reason": "not a conventional chair",
                            "position_xyz": [1.0, 2.0, 3.0],
                        }
                        for index in range(5)
                    ],
                }
            }
        )

        self.assertEqual(len(projection["target_rejections"]), 3)
        self.assertEqual(projection["target_rejections"][0]["frame_index"], 4)
        self.assertEqual(
            projection["target_rejections"][0]["lookalike_type"],
            "incline exercise bench",
        )
        self.assertNotIn("position_xyz", json.dumps(projection, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
