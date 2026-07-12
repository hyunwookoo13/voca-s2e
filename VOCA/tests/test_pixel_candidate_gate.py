import math
import unittest

import numpy as np

from pixel_candidate_gate import (
    build_pixel_candidates,
    build_saturation_recovery_candidates,
    candidate_visual_evidence,
    partition_pixel_candidates,
    render_pixel_candidates,
    validate_pixel_candidate_selection,
)


class PixelCandidateGateTests(unittest.TestCase):
    def test_saturation_recovery_candidates_are_far_verified_probes(self):
        views = [
            {"view_id": 0, "view_type": "front", "relative_heading_deg": 0},
            {"view_id": 1, "view_type": "left", "relative_heading_deg": -45},
        ]
        rgb = [
            np.full((100, 200, 3), 120, dtype=np.uint8),
            np.full((100, 200, 3), 160, dtype=np.uint8),
        ]

        candidates = build_saturation_recovery_candidates(
            views,
            (100, 200, 3),
            rgb_images=rgb,
        )

        self.assertEqual(len(candidates), 6)
        self.assertEqual({item["view_id"] for item in candidates}, {0, 1})
        self.assertTrue(
            all(item["negative_memory_saturation_recovery"] for item in candidates)
        )
        self.assertTrue(all(item["requires_rgb_verification"] for item in candidates))
        self.assertTrue(all(not item["avoid"] for item in candidates))
        self.assertTrue(all(item["point_px"][1] == 71 for item in candidates))
        self.assertTrue(
            all(item["topological_candidate_ref"] is None for item in candidates)
        )
    def test_low_information_rgb_requires_verification_without_hard_reject(self):
        evidence = candidate_visual_evidence(
            np.zeros((80, 120, 3), dtype=np.uint8),
            [60, 70],
        )
        self.assertTrue(evidence["requires_verification"])
        self.assertFalse(evidence["hard_reject"])

    def _gate_input(self):
        return {
            "pixel_candidates": {
                "schema_version": "rgb_pixel_candidates_v1",
                "required_for_go": True,
                "candidates": [
                    {
                        "candidate_ref": "px_safe",
                        "marker_id": "M01",
                        "view_id": 0,
                        "view_type_hint": "front",
                        "point_px": [100, 80],
                        "point_norm": [0.5, 0.8],
                        "avoid": False,
                    },
                    {
                        "candidate_ref": "px_avoid",
                        "marker_id": "M02",
                        "view_id": 0,
                        "view_type_hint": "front",
                        "point_px": [140, 80],
                        "point_norm": [0.7, 0.8],
                        "avoid": True,
                    },
                ],
            }
        }

    def test_valid_ref_canonicalizes_tampered_pixel(self):
        result = validate_pixel_candidate_selection(
            {
                "action": "go",
                "selected_candidate_ref": "px_safe",
                "selected_view_id": 0,
                "selected_view_type": "front",
                "selected_image_point": [1, 2],
                "fine_goal": {"point_px": [1, 2], "point_norm": [0.01, 0.02]},
            },
            self._gate_input(),
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["reason"], "valid_pixel_candidate")
        self.assertEqual(result["candidate"]["candidate_ref"], "px_safe")
        self.assertTrue(result["coordinate_corrected"])
        self.assertEqual(result["canonical_output"]["selected_image_point"], [100, 80])
        self.assertEqual(result["canonical_output"]["selected_view_id"], 0)
        self.assertEqual(result["canonical_output"]["selected_view_type"], "front")
        self.assertEqual(result["canonical_output"]["fine_goal"]["point_px"], [100, 80])
        self.assertEqual(result["canonical_output"]["fine_goal"]["point_norm"], [0.5, 0.8])
        self.assertTrue(result["checkpoint"]["passed"])

    def test_visible_marker_alias_resolves_to_canonical_candidate_ref(self):
        result = validate_pixel_candidate_selection(
            {
                "action": "go",
                "selected_candidate_ref": "M01",
                "selected_view_id": 0,
                "selected_view_type": "front",
                "selected_image_point": [1, 2],
            },
            self._gate_input(),
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["reason"], "valid_pixel_candidate_marker_alias")
        self.assertEqual(result["requested_ref"], "M01")
        self.assertEqual(result["resolved_ref"], "px_safe")
        self.assertTrue(result["ref_alias_applied"])
        self.assertEqual(
            result["canonical_output"]["selected_candidate_ref"],
            "px_safe",
        )
        self.assertEqual(result["canonical_output"]["selected_image_point"], [100, 80])

    def test_gate_rejects_missing_unknown_avoided_and_view_mismatched_refs(self):
        cases = [
            ({"action": "go", "selected_view_id": 0, "selected_view_type": "front"}, "missing_selected_candidate_ref"),
            (
                {
                    "action": "go",
                    "selected_candidate_ref": "px_unknown",
                    "selected_view_id": 0,
                    "selected_view_type": "front",
                },
                "unknown_selected_candidate_ref",
            ),
            (
                {
                    "action": "go",
                    "selected_candidate_ref": "px_avoid",
                    "selected_view_id": 0,
                    "selected_view_type": "front",
                },
                "selected_candidate_avoid_true",
            ),
            (
                {
                    "action": "go",
                    "selected_candidate_ref": "px_safe",
                    "selected_view_id": 1,
                    "selected_view_type": "left",
                },
                "selected_candidate_view_mismatch",
            ),
        ]

        for output, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                result = validate_pixel_candidate_selection(output, self._gate_input())
                self.assertFalse(result["passed"])
                self.assertEqual(result["reason"], expected_reason)
                self.assertIsNone(result["candidate"] if expected_reason.startswith("missing") or expected_reason.startswith("unknown") else result.get("canonical_output"))
                self.assertFalse(result["checkpoint"]["passed"])

    def test_non_go_action_does_not_require_candidate(self):
        result = validate_pixel_candidate_selection(
            {"action": "request_observation"},
            self._gate_input(),
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["reason"], "candidate_not_required_for_action")

    def test_partition_removes_avoided_candidates_from_executable_choices(self):
        candidates = self._gate_input()["pixel_candidates"]["candidates"]

        executable, excluded = partition_pixel_candidates(candidates)

        self.assertEqual([item["candidate_ref"] for item in executable], ["px_safe"])
        self.assertTrue(executable[0]["executable"])
        self.assertEqual([item["candidate_ref"] for item in excluded], ["px_avoid"])
        self.assertEqual(excluded[0]["exclusion_reason"], "avoid_true")

    def test_gate_identifies_a_selection_from_audit_only_exclusions(self):
        vlm_input = self._gate_input()
        avoided = vlm_input["pixel_candidates"]["candidates"].pop()
        vlm_input["pixel_candidates"]["excluded_candidates"] = [avoided]

        result = validate_pixel_candidate_selection(
            {"action": "go", "selected_candidate_ref": "px_avoid"},
            vlm_input,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "selected_candidate_excluded")
        self.assertEqual(result["candidate"]["candidate_ref"], "px_avoid")

    def test_builds_deterministic_six_point_set_for_one_view(self):
        views = [
            {
                "view_id": 0,
                "view_type": "front",
                "relative_heading_deg": 0,
            }
        ]

        first = build_pixel_candidates(views, (100, 200, 3), {})
        second = build_pixel_candidates(views, (100, 200, 3), {})

        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual(len({item["candidate_ref"] for item in first}), 6)
        self.assertEqual(first[0]["candidate_ref"], "px_v00_r0_c0")
        self.assertEqual(first[0]["marker_id"], "M01")
        self.assertEqual(first[0]["point_px"], [50, 96])
        self.assertEqual(first[-1]["point_px"], [149, 83])
        self.assertTrue(all(item["point_px"][1] == 96 for item in first[:3]))
        self.assertTrue(all(item["point_px"][1] == 83 for item in first[3:]))
        for candidate in first:
            u, v = candidate["point_px"]
            self.assertGreaterEqual(u, 0)
            self.assertLess(u, 200)
            self.assertGreaterEqual(v, 0)
            self.assertLess(v, 100)
            self.assertAlmostEqual(candidate["point_norm"][0], u / 199.0, places=4)
            self.assertAlmostEqual(candidate["point_norm"][1], v / 99.0, places=4)
            self.assertFalse(candidate["avoid"])

    def test_full_sweep_distributes_candidates_across_every_view(self):
        headings = [-180, -135, -90, -45, 0, 45, 90, 135]
        views = [
            {
                "view_id": index,
                "view_type": "sweep_{:02d}".format(index),
                "relative_heading_deg": heading,
            }
            for index, heading in enumerate(headings)
        ]

        candidates = build_pixel_candidates(views, (480, 640, 3), {})

        self.assertEqual(len(candidates), 32)
        self.assertEqual({item["view_id"] for item in candidates}, set(range(8)))
        for view_id in range(8):
            view_candidates = [
                item for item in candidates if item["view_id"] == view_id
            ]
            self.assertEqual(len(view_candidates), 4)
            mid_floor = [
                item for item in view_candidates if "_r1_" in item["candidate_ref"]
            ]
            bottom_center = [
                item for item in view_candidates if "_r0_c1" in item["candidate_ref"]
            ]
            self.assertEqual(len(mid_floor), 3)
            self.assertTrue(all(item["point_px"][1] == 402 for item in mid_floor))
            self.assertEqual(len(bottom_center), 1)
            self.assertEqual(bottom_center[0]["point_px"], [320, 465])

    def test_full_sweep_maps_topological_edge_only_to_covered_views(self):
        headings = [-180, -135, -90, -45, 0, 45, 90, 135]
        views = [
            {
                "view_id": index,
                "view_type": "right" if heading > 45 else "front",
                "relative_heading_deg": heading,
            }
            for index, heading in enumerate(headings)
        ]
        memory = {
            "candidate_refs": {
                "exits": [
                    {
                        "candidate_ref": "topo_backtrack",
                        "edge_id": "edge_backtrack",
                        "relation_type": "backtrack",
                        "view_type_hint": "right",
                        "bearing_deg_robot": 131.25,
                        "avoid": False,
                        "status": "success",
                        "score": 0.2,
                    }
                ]
            }
        }

        candidates = build_pixel_candidates(views, (480, 640, 3), memory)
        mapped = [
            item
            for item in candidates
            if item["topological_candidate_ref"] == "topo_backtrack"
        ]

        self.assertEqual(len(mapped), 4)
        self.assertEqual({item["view_id"] for item in mapped}, {7})
        self.assertTrue(
            all(item["topological_bearing_deg_robot"] == 131.25 for item in mapped)
        )
        self.assertTrue(
            all(item["topological_view_alignment_error_deg"] == 3.75 for item in mapped)
        )

    def test_full_sweep_preserves_two_view_camera_overlap_for_path_tangent(self):
        headings = [-180, -135, -90, -45, 0, 45, 90, 135]
        views = [
            {
                "view_id": index,
                "view_type": "sweep_{:02d}".format(index),
                "relative_heading_deg": heading,
            }
            for index, heading in enumerate(headings)
        ]
        memory = {
            "candidate_refs": {
                "exits": [
                    {
                        "candidate_ref": "topo_backtrack",
                        "edge_id": "edge_backtrack",
                        "relation_type": "backtrack",
                        "view_type_hint": "right",
                        "bearing_deg_robot": 60.0,
                        "bearing_source": "executed_position_trace",
                        "chord_bearing_deg_robot": 85.0,
                        "path_departure_bearing_node_deg": -180.0,
                        "avoid": False,
                        "status": "success",
                        "score": 0.2,
                    }
                ]
            }
        }

        candidates = build_pixel_candidates(views, (480, 640, 3), memory)
        mapped = [
            item
            for item in candidates
            if item["topological_candidate_ref"] == "topo_backtrack"
        ]

        self.assertEqual(len(mapped), 8)
        self.assertEqual({item["view_id"] for item in mapped}, {5, 6})
        self.assertEqual(
            {
                item["view_id"]: item["topological_view_alignment_error_deg"]
                for item in mapped
            },
            {5: 15.0, 6: 30.0},
        )
        self.assertTrue(
            all(
                item["topological_bearing_source"]
                == "executed_position_trace"
                for item in mapped
            )
        )

    def test_inherits_negative_view_and_verified_failed_pixel_neighborhood(self):
        views = [
            {"view_id": 0, "view_type": "front", "relative_heading_deg": 0},
            {"view_id": 1, "view_type": "left", "relative_heading_deg": -90},
        ]
        memory = {
            "candidate_refs": {
                "exits": [
                    {
                        "candidate_ref": "topo_front",
                        "edge_id": "edge_front_failed",
                        "relation_type": "directional_failure",
                        "view_type_hint": "front",
                        "avoid": True,
                        "status": "deadlock_entry_candidate",
                        "score": 0.1,
                        "reason": "known failed front branch",
                    },
                    {
                        "candidate_ref": "topo_left_failed",
                        "edge_id": "edge_left_failed",
                        "relation_type": "directional_failure",
                        "view_type_hint": "left",
                        "avoid": True,
                        "status": "deadlock_entry",
                        "score": 0.1,
                        "reason": "known failed branch",
                    },
                ]
            },
            "voca_sidecar": {
                "directional_failures": [
                    {
                        "frame_index": 9,
                        "selected_view_id": 0,
                        "angle_deg": 0,
                        "point_px": [320, 465],
                        "failure_class": "collision_blocked",
                        "negative_edge_id": "edge_front_failed",
                    }
                ]
            },
        }

        candidates = build_pixel_candidates(views, (480, 640, 3), memory)
        front = [item for item in candidates if item["view_id"] == 0]
        left = [item for item in candidates if item["view_id"] == 1]

        self.assertEqual(len(front), 6)
        self.assertEqual(len(left), 6)
        verified_front = [
            item for item in front if item["status"] == "verified_failed_pixel_neighborhood"
        ]
        self.assertTrue(all(item["avoid"] for item in front))
        self.assertEqual(len(verified_front), 1)
        self.assertEqual(verified_front[0]["point_px"], [320, 465])
        self.assertEqual(verified_front[0]["failure_class"], "collision_blocked")
        self.assertTrue(all(item["avoid"] for item in left))
        self.assertTrue(all(item["topological_candidate_ref"] == "topo_left_failed" for item in left))
        self.assertTrue(all(item["topological_edge_id"] == "edge_left_failed" for item in left))
        self.assertTrue(
            all(item["topological_relation_type"] == "directional_failure" for item in left)
        )

    def test_reused_view_local_ref_does_not_poison_a_different_direction(self):
        views = [{"view_id": 0, "view_type": "left", "relative_heading_deg": -60}]
        memory = {
            "candidate_refs": {
                "exits": [
                    {
                        "candidate_ref": "topo_left_safe",
                        "edge_id": None,
                        "view_type_hint": "left",
                        "avoid": False,
                        "status": "unknown_frontier",
                        "score": 0.5,
                        "reason": "unvisited left frontier",
                    }
                ]
            },
            "voca_sidecar": {
                "directional_failures": [
                    {
                        "frame_index": 9,
                        "selected_view_id": 0,
                        "selected_candidate_ref": "px_v00_r0_c1",
                        "angle_deg": 0,
                        "point_px": [320, 465],
                        "failure_class": "collision_blocked",
                        "negative_edge_id": "edge_front_failed",
                    }
                ]
            },
        }

        candidates = build_pixel_candidates(views, (480, 640, 3), memory)

        reused_ref = next(
            item for item in candidates if item["candidate_ref"] == "px_v00_r0_c1"
        )
        self.assertFalse(reused_ref["avoid"])
        self.assertEqual(reused_ref["status"], "unknown_frontier")

    def test_runtime_point_verifier_rejection_excludes_same_angle_and_pixel(self):
        views = [{"view_id": 0, "view_type": "front", "relative_heading_deg": 0}]
        memory = {
            "runtime_navigation_feedback": [
                {
                    "event": "point_verifier_rejected",
                    "selected_candidate_ref": "px_v00_r1_c1",
                    "angle_deg": 0,
                    "point_px": [320, 402],
                    "surface": "wall",
                }
            ]
        }

        candidates = build_pixel_candidates(views, (480, 640, 3), memory)
        rejected = next(
            item for item in candidates if item["candidate_ref"] == "px_v00_r1_c1"
        )

        self.assertTrue(rejected["avoid"])
        self.assertEqual(rejected["status"], "point_verifier_rejected_candidate")
        self.assertIn("wall", rejected["reason"])

    def test_runtime_rejection_matches_physical_world_bearing_after_rotation(self):
        memory = {
            "voca_sidecar": {"heading_rad": math.pi},
            "runtime_navigation_feedback": [
                {
                    "event": "point_verifier_rejected",
                    "selected_candidate_ref": "px_v00_r1_c1",
                    "angle_deg": 0,
                    "world_bearing_deg": 0,
                    "point_px": [320, 402],
                    "surface": "wall",
                }
            ],
        }

        candidates = build_pixel_candidates(
            [{"view_id": 0, "view_type": "front", "relative_heading_deg": 0}],
            (480, 640, 3),
            memory,
        )
        same_local_ref_opposite_world_direction = next(
            item for item in candidates if item["candidate_ref"] == "px_v00_r1_c1"
        )

        self.assertFalse(same_local_ref_opposite_world_direction["avoid"])

    def test_repeated_unresolved_target_approach_excludes_same_candidate(self):
        views = [{"view_id": 0, "view_type": "front", "relative_heading_deg": 0}]
        memory = {
            "runtime_navigation_feedback": [
                {
                    "event": "target_approach_unresolved",
                    "selected_candidate_ref": "px_v00_r1_c1",
                    "angle_deg": 0,
                    "point_px": [320, 402],
                }
            ]
        }

        candidates = build_pixel_candidates(views, (480, 640, 3), memory)
        rejected = next(
            item for item in candidates if item["candidate_ref"] == "px_v00_r1_c1"
        )

        self.assertTrue(rejected["avoid"])
        self.assertEqual(rejected["status"], "target_approach_unresolved_candidate")
        self.assertEqual(rejected["failure_class"], "target_approach_unresolved")

    def test_render_marks_candidates_without_mutating_source_rgb(self):
        source = np.zeros((100, 200, 3), dtype=np.uint8)
        original = source.copy()
        candidates = build_pixel_candidates(
            [{"view_id": 0, "view_type": "front", "relative_heading_deg": 0}],
            source.shape,
            {},
        )

        marked = render_pixel_candidates(source, candidates, view_id=0)

        self.assertTrue(np.array_equal(source, original))
        self.assertFalse(np.array_equal(marked, source))
        for candidate in candidates:
            u, v = candidate["point_px"]
            patch = marked[max(0, v - 3) : v + 4, max(0, u - 3) : u + 4]
            self.assertGreater(int(np.count_nonzero(patch)), 0)


if __name__ == "__main__":
    unittest.main()
