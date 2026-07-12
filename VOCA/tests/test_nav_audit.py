import json
import tempfile
import unittest
from pathlib import Path

from nav_audit import (
    NavigationAuditLogger,
    build_action_outcome_from_positions,
    build_memory_free_vlm_input,
    qwen_point_decision_to_vlm_output,
)


class NavigationAuditTests(unittest.TestCase):
    def test_nav_audit_uses_canonical_v6_schema_source(self):
        import nav_audit
        from voca_s2e_bridge import NAV_MEMORY_QWEN_ROOT

        self.assertTrue(Path(nav_audit.SCHEMA_SOURCE_FILE).is_file())
        self.assertTrue(Path(nav_audit.SAFETY_SOURCE_FILE).is_file())
        self.assertIn(str(NAV_MEMORY_QWEN_ROOT), nav_audit.SCHEMA_SOURCE_FILE)
        self.assertIn(str(NAV_MEMORY_QWEN_ROOT), nav_audit.SAFETY_SOURCE_FILE)

    def test_qwen_point_decision_maps_to_nav_vlm_output_schema(self):
        output, warnings = qwen_point_decision_to_vlm_output(
            {
                "Reason": "visible floor toward bathroom doorway",
                "Angle": 30,
                "Point": [4, 5],
                "Confidence": "high",
                "fallback": False,
            },
            image_shape=(12, 16, 3),
            selected_view_id=1,
        )

        self.assertEqual(warnings, [])
        self.assertEqual(output["schema_version"], "nav_vlm_waypoint_v1")
        self.assertEqual(output["action"], "go")
        self.assertEqual(output["selected_view_id"], 1)
        self.assertEqual(output["selected_view_type"], "front")
        self.assertEqual(output["selected_image_point"], [4, 5])
        self.assertEqual(output["fine_goal"]["point_px"], [4, 5])
        self.assertEqual(output["fine_goal"]["point_norm"], [0.2667, 0.4545])
        self.assertEqual(output["reasoning"]["short_text"], "visible floor toward bathroom doorway")
        self.assertEqual(output["confidence"], "high")

    def test_invalid_qwen_point_becomes_rotate_output(self):
        output, warnings = qwen_point_decision_to_vlm_output(
            {
                "Reason": "bad point",
                "Angle": 0,
                "Point": [999, -2],
                "Confidence": "high",
            },
            image_shape=(12, 16, 3),
            selected_view_id=0,
        )

        self.assertIn("invalid selected_image_point; fallback rotate", warnings)
        self.assertEqual(output["schema_version"], "nav_vlm_waypoint_v1")
        self.assertEqual(output["action"], "rotate")
        self.assertIsNone(output["selected_image_point"])
        self.assertFalse(output["fine_goal"]["valid"])
        self.assertEqual(output["reasoning"]["failure_mode"], "no_visible_navigable_floor")

    def test_direct_nav_vlm_output_is_sanitized_without_legacy_reconstruction(self):
        direct_output = {
            "schema_version": "nav_vlm_waypoint_v1",
            "action": "rotate",
            "selected_view_id": None,
            "selected_view_type": None,
            "selected_image_point": None,
            "fine_goal": {"valid": False},
            "observation_request": None,
            "reasoning": {
                "decision_reason": "R02_NO_VISIBLE_NAVIGABLE_FLOOR",
                "goal_reason": "F08_NONE_ROTATE_OR_STOP",
                "failure_mode": "no_visible_navigable_floor",
                "short_text": "scan right because no doorway is visible",
            },
            "control": {"rotate_yaw_deg": 30},
            "confidence": "low",
        }

        output, warnings = qwen_point_decision_to_vlm_output(
            {
                "vlm_output": direct_output,
                "vlm_input": build_memory_free_vlm_input(
                    target_object="toilet",
                    image_shape=(12, 16, 3),
                    frame_index=0,
                    angles=[0, 30],
                    call_type="make_plan",
                ),
            },
            image_shape=(12, 16, 3),
            selected_view_id=0,
            angles=[0, 30],
        )

        self.assertEqual(warnings, [])
        self.assertEqual(output["schema_version"], "nav_vlm_waypoint_v1")
        self.assertEqual(output["action"], "rotate")
        self.assertEqual(output["control"]["rotate_yaw_deg"], 30.0)
        self.assertEqual(output["reasoning"]["decision_reason"], "R02_NO_VISIBLE_NAVIGABLE_FLOOR")

    def test_memory_free_vlm_input_has_null_memory_context(self):
        vlm_input = build_memory_free_vlm_input(
            target_object="toilet",
            image_shape=(480, 640, 3),
            frame_index=7,
            priors={"Gateways": ["bathroom doorway"]},
            angles=[0, 30, 60],
            call_type="make_plan",
        )

        self.assertEqual(vlm_input["schema_version"], "nav_vlm_waypoint_v1")
        self.assertEqual(vlm_input["task"]["task_mode"], "ObjNav")
        self.assertEqual(vlm_input["task"]["target_object"], "toilet")
        self.assertEqual(vlm_input["observation"]["image_width"], 640)
        self.assertEqual(vlm_input["observation"]["image_height"], 480)
        self.assertEqual(vlm_input["memory"]["schema_version"], "null_memory_context_v0")
        self.assertFalse(vlm_input["memory"]["enabled"])
        self.assertEqual(vlm_input["metadata"]["call_type"], "make_plan")
        self.assertEqual(vlm_input["coordinate_frame"]["map_frame"], "habitat_world_xy_or_gps_aligned_local_map")

    def test_action_outcome_from_positions_tracks_progress(self):
        outcome = build_action_outcome_from_positions(
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.3, 0.0, 0.4],
            collision=False,
            raw={"policy_steps": 3},
        )

        self.assertEqual(outcome["action"], "go")
        self.assertTrue(outcome["success"])
        self.assertFalse(outcome["collision"])
        self.assertFalse(outcome["no_progress"])
        self.assertAlmostEqual(outcome["moved_distance_m"], 0.5)
        self.assertEqual(outcome["raw"]["policy_steps"], 3)

    def test_action_outcome_keeps_non_go_action_without_requiring_translation(self):
        outcome = build_action_outcome_from_positions(
            action="rotate",
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.0, 0.0, 0.0],
            collision=False,
        )

        self.assertEqual(outcome["action"], "rotate")
        self.assertTrue(outcome["success"])
        self.assertTrue(outcome["no_progress"])

    def test_audit_logger_outcome_uses_direct_vlm_action(self):
        logger = NavigationAuditLogger()
        direct_output = {
            "schema_version": "nav_vlm_waypoint_v1",
            "action": "rotate",
            "selected_view_id": None,
            "selected_view_type": None,
            "selected_image_point": None,
            "fine_goal": {"valid": False},
            "observation_request": None,
            "reasoning": {
                "decision_reason": "R02_NO_VISIBLE_NAVIGABLE_FLOOR",
                "goal_reason": "F08_NONE_ROTATE_OR_STOP",
                "failure_mode": "no_visible_navigable_floor",
                "short_text": "rotate to inspect the next doorway",
            },
            "control": {"rotate_yaw_deg": 30},
            "confidence": "low",
        }

        logger.record_decision(
            decision={
                "vlm_output": direct_output,
                "Reason": "rotate to inspect the next doorway",
                "Angle": 0,
                "Point": [32, 36],
                "Confidence": "low",
            },
            target_object="chair",
            image_shape=(96, 128, 3),
            frame_index=0,
            position_xyz=[0.0, 0.0, 0.0],
            metrics={"distance_to_goal": 4.0},
            angles=[0],
            call_type="qwen_action_loop",
            selected_view_id=0,
        )
        logger.finalize_pending(
            position_xyz=[0.0, 0.0, 0.0],
            metrics={"distance_to_goal": 4.0, "success": 0.0},
        )

        self.assertEqual(logger.steps[0]["action"], "rotate")
        self.assertEqual(logger.steps[0]["outcome"]["action"], "rotate")

    def test_audit_logger_can_annotate_pending_step_runtime(self):
        logger = NavigationAuditLogger()
        logger.record_decision(
            decision={"Reason": "front floor", "Angle": 0, "Point": [32, 36], "Confidence": "medium"},
            target_object="chair",
            image_shape=(96, 128, 3),
            frame_index=0,
            position_xyz=[0.0, 0.0, 0.0],
            metrics={"distance_to_goal": 4.0},
            angles=[0],
            call_type="qwen_action_loop",
            selected_view_id=0,
        )

        logger.annotate_pending_runtime(
            "go_progress",
            {"success": False, "distance_delta_m": -0.25, "no_progress": True},
        )

        self.assertEqual(logger.steps[0]["runtime"]["go_progress"]["distance_delta_m"], -0.25)

    def test_audit_logger_writes_memory_free_steps_json(self):
        logger = NavigationAuditLogger()
        first = {"Reason": "front floor", "Angle": 0, "Point": [32, 36], "Confidence": "medium"}
        second = {"Reason": "left doorway", "Angle": 30, "Point": [20, 30], "Confidence": "high"}

        logger.record_decision(
            decision=first,
            target_object="chair",
            image_shape=(96, 128, 3),
            frame_index=0,
            position_xyz=[0.0, 0.0, 0.0],
            metrics={"distance_to_goal": 4.0, "top_down_map": object()},
            priors={"Supports": ["floor"]},
            angles=[0, 30],
            call_type="make_plan",
            selected_view_id=0,
        )
        logger.record_decision(
            decision=second,
            target_object="chair",
            image_shape=(96, 128, 3),
            frame_index=8,
            position_xyz=[0.25, 0.0, 0.0],
            metrics={"distance_to_goal": 3.8, "top_down_map": object()},
            priors={"Gateways": ["doorway"]},
            angles=[0, 30],
            call_type="replan",
            selected_view_id=1,
        )
        logger.finalize_pending(
            position_xyz=[0.25, 0.0, 0.25],
            metrics={"distance_to_goal": 3.6, "success": 0.0, "top_down_map": object()},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = logger.save_run(Path(tmpdir))
            steps = json.loads(Path(paths["steps_json"]).read_text(encoding="utf-8"))["steps"]
            graph = json.loads(Path(paths["memory_graph_json"]).read_text(encoding="utf-8"))

        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["vlm_input"]["memory"]["schema_version"], "null_memory_context_v0")
        self.assertEqual(steps[0]["vlm_output"]["selected_image_point"], [32, 36])
        self.assertAlmostEqual(steps[0]["outcome"]["moved_distance_m"], 0.25)
        self.assertEqual(steps[1]["vlm_output"]["selected_view_type"], "front")
        self.assertAlmostEqual(steps[1]["outcome"]["moved_distance_m"], 0.25)
        self.assertEqual(steps[1]["runtime"]["final_metrics"]["top_down_map"], "<object>")
        self.assertFalse(graph["memory_enabled"])


if __name__ == "__main__":
    unittest.main()
