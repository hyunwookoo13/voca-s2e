import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


QWEN_ROOT = Path(__file__).resolve().parents[1] / "qwen_nav_memory_framework_v5"
if str(QWEN_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN_ROOT))

from goal_adapter.habitat_pixelnav_closed_loop import (
    HabitatPixelNavMemorySmokeConfig,
    load_memory_smoke_config_from_audit,
    run_habitat_pixelnav_memory_smoke,
)


class FakeRunner:
    def __init__(self):
        self.position = [0.0, 0.0, 0.0]
        self.heading = 0.0
        self.actions = []
        self.collided = False

    def set_pose(self, position_xyz, heading):
        self.position = [float(v) for v in position_xyz]
        self.heading = float(heading)

    def position_xyz(self):
        return list(self.position)

    def rgb(self):
        return np.full((4, 6, 3), 96, dtype=np.uint8)

    def previous_step_collided(self):
        return self.collided

    def supported_action_names(self):
        return {"move_forward", "turn_left", "turn_right"}

    def step(self, action_name):
        self.actions.append(action_name)
        if action_name == "move_forward":
            self.position = [self.position[0] + 0.25, self.position[1], self.position[2]]


class FakeGoal:
    def __init__(self):
        self.goal_mask = np.ones((4, 6), dtype=np.uint8)


class FakeExecutor:
    def __init__(self, actions):
        self.actions = list(actions)

    def reset_from_point(self, goal_rgb, selected_point_uv, radius=5):
        return FakeGoal()

    def step(self, obs_rgb, collide=False):
        action = self.actions.pop(0) if self.actions else 0
        return type(
            "FakeStep",
            (),
            {
                "action": action,
                "overlay_image": np.zeros_like(obs_rgb),
            },
        )()


class HabitatPixelNavClosedLoopSmokeTest(unittest.TestCase):
    def test_load_memory_smoke_config_from_audit_preserves_selected_goal(self):
        payload = {
            "scene_path": "scene.glb",
            "robot_position_xyz": [1.0, 0.0, 2.0],
            "heading": 0.25,
            "coarse_goal_xy": [4.0, 6.0],
            "selected_view": "left",
            "selected_image_point": [3, 2],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit.json"
            audit_path.write_text(json.dumps(payload), encoding="utf-8")

            config = load_memory_smoke_config_from_audit(audit_path, output_dir=Path(temp_dir) / "out")

        self.assertEqual(config.scene_path, "scene.glb")
        self.assertEqual(config.start_position_xyz, [1.0, 0.0, 2.0])
        self.assertAlmostEqual(config.start_heading_rad, 0.25)
        self.assertEqual(config.goal_map_xy, [4.0, 6.0])
        self.assertEqual(config.selected_view_type, "left")
        self.assertEqual(config.selected_image_point, (3, 2))

    def test_run_memory_smoke_executes_one_go_step_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            executor = FakeExecutor([1, 1])
            config = HabitatPixelNavMemorySmokeConfig(
                scene_path="scene.glb",
                output_dir=temp_dir,
                start_position_xyz=[0.0, 0.0, 0.0],
                start_heading_rad=0.0,
                goal_map_xy=[5.0, 0.0],
                selected_view_type="left",
                selected_image_point=(3, 2),
                image_width=6,
                image_height=4,
                max_agent_steps=1,
                max_pixelnav_steps=2,
                force_front_view_waypoint=False,
            )

            result = run_habitat_pixelnav_memory_smoke(
                config,
                runner_factory=lambda _backend_config: runner,
                executor_factory=lambda: executor,
            )
            summary_path = Path(result["summary_json"])
            summary_exists = summary_path.exists()
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertTrue(summary_exists)
        self.assertEqual(summary["episode"]["steps"], 1)
        self.assertFalse(summary["episode"]["done"])
        self.assertEqual(summary["steps"][0]["action"], "go")
        self.assertTrue(summary["steps"][0]["outcome"]["success"])
        self.assertAlmostEqual(summary["steps"][0]["outcome"]["moved_distance_m"], 0.5)
        self.assertEqual(summary["steps"][0]["outcome"]["raw"]["selected_view"], "left")
        self.assertGreaterEqual(summary["memory"]["num_nodes"], 1)
        self.assertEqual(runner.actions, ["move_forward", "move_forward"])

    def test_run_memory_smoke_uses_qwen_memory_framework_v5_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            executor = FakeExecutor([1, 1])
            config = HabitatPixelNavMemorySmokeConfig(
                scene_path="scene.glb",
                output_dir=temp_dir,
                start_position_xyz=[0.0, 0.0, 0.0],
                start_heading_rad=0.0,
                goal_map_xy=[5.0, 0.0],
                selected_view_type="front",
                selected_image_point=(3, 2),
                image_width=6,
                image_height=4,
                max_agent_steps=1,
                max_pixelnav_steps=2,
                force_front_view_waypoint=False,
            )

            result = run_habitat_pixelnav_memory_smoke(
                config,
                runner_factory=lambda _backend_config: runner,
                executor_factory=lambda: executor,
            )
            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))

        self.assertEqual(summary["memory"]["schema_version"], "relative_topometric_memory_graph_v5")
        self.assertEqual(summary["memory_context_checks"]["merge_policy"], "soft_preserve_revisit_nodes")

    def test_run_memory_smoke_records_new_node_edge_and_live_pose_after_two_steps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            executor = FakeExecutor([1, 1, 1, 1, 1, 1, 1, 1])
            config = HabitatPixelNavMemorySmokeConfig(
                scene_path="scene.glb",
                output_dir=temp_dir,
                start_position_xyz=[0.0, 0.0, 0.0],
                start_heading_rad=0.0,
                goal_map_xy=[5.0, 0.0],
                selected_view_type="front",
                selected_image_point=(3, 2),
                image_width=6,
                image_height=4,
                max_agent_steps=2,
                max_pixelnav_steps=4,
                force_front_view_waypoint=False,
            )

            result = run_habitat_pixelnav_memory_smoke(
                config,
                runner_factory=lambda _backend_config: runner,
                executor_factory=lambda: executor,
            )
            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))

        self.assertEqual(summary["episode"]["steps"], 2)
        self.assertGreaterEqual(summary["memory"]["num_nodes"], 2)
        self.assertGreaterEqual(summary["memory"]["num_edges"], 1)
        self.assertTrue(summary["memory"]["live_pose_relation"]["valid"])
        self.assertGreater(summary["memory"]["live_pose_relation"]["distance_from_latest_node_m"], 0.0)
        self.assertTrue(summary["memory"]["temporal_edges"])
        first_edge = summary["memory"]["temporal_edges"][0]
        self.assertEqual(first_edge["edge_type"], "temporal_transition")
        self.assertEqual(first_edge["status"], "success")
        self.assertGreater(first_edge["relative_pose_src_to_dst"]["dx_m"], 0.9)

    def test_run_memory_smoke_rotates_left_goal_before_front_execution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            executor = FakeExecutor([1, 1, 1, 1, 1, 1, 1, 1])
            config = HabitatPixelNavMemorySmokeConfig(
                scene_path="scene.glb",
                output_dir=temp_dir,
                start_position_xyz=[0.0, 0.0, 0.0],
                start_heading_rad=0.0,
                goal_map_xy=[5.0, 0.0],
                selected_view_type="left",
                selected_image_point=(3, 2),
                image_width=6,
                image_height=4,
                max_agent_steps=3,
                max_pixelnav_steps=4,
                force_front_view_waypoint=True,
            )

            result = run_habitat_pixelnav_memory_smoke(
                config,
                runner_factory=lambda _backend_config: runner,
                executor_factory=lambda: executor,
            )
            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))

        self.assertEqual(summary["episode"]["steps"], 3)
        self.assertTrue(summary["policy_checks"]["rotate_first_handoff_verified"])
        self.assertEqual(summary["steps"][0]["action"], "rotate")
        self.assertEqual(summary["steps"][0]["outcome"]["action"], "rotate")
        self.assertEqual(summary["steps"][0]["vlm_output"]["deferred_go"]["selected_view_type"], "left")
        self.assertEqual(summary["steps"][1]["action"], "go")
        self.assertEqual(summary["steps"][1]["outcome"]["raw"]["selected_view"], "front")
        self.assertTrue(summary["steps"][1]["outcome"]["success"])
        self.assertGreaterEqual(summary["memory"]["num_nodes"], 2)
        self.assertGreaterEqual(summary["memory"]["num_edges"], 1)
        self.assertEqual(runner.actions, ["move_forward"] * 8)

    def test_run_memory_smoke_rotate_first_policy_handles_right_and_back_views(self):
        cases = [("right", 90.0), ("back", 180.0)]
        for view_type, expected_yaw in cases:
            with self.subTest(view_type=view_type):
                with tempfile.TemporaryDirectory() as temp_dir:
                    runner = FakeRunner()
                    executor = FakeExecutor([1, 1, 1, 1])
                    config = HabitatPixelNavMemorySmokeConfig(
                        scene_path="scene.glb",
                        output_dir=temp_dir,
                        start_position_xyz=[0.0, 0.0, 0.0],
                        start_heading_rad=0.0,
                        goal_map_xy=[5.0, 0.0],
                        selected_view_type=view_type,
                        selected_image_point=(3, 2),
                        image_width=6,
                        image_height=4,
                        max_agent_steps=2,
                        max_pixelnav_steps=4,
                        force_front_view_waypoint=True,
                    )

                    result = run_habitat_pixelnav_memory_smoke(
                        config,
                        runner_factory=lambda _backend_config: runner,
                        executor_factory=lambda: executor,
                    )
                    summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))

                policy = summary["policy_checks"]
                self.assertTrue(policy["rotate_first_handoff_verified"])
                self.assertEqual(policy["requested_view_type"], view_type)
                self.assertEqual(policy["requested_view_heading_deg"], expected_yaw)
                self.assertEqual(policy["rotate_yaw_deg"], expected_yaw)
                self.assertEqual(policy["first_go_after_rotate_view_type"], "front")
                self.assertEqual(summary["steps"][0]["action"], "rotate")
                self.assertEqual(summary["steps"][0]["outcome"]["rotated_deg"], expected_yaw)
                self.assertEqual(summary["steps"][1]["outcome"]["raw"]["selected_view"], "front")

    def test_run_memory_smoke_records_no_progress_as_deadlock_suspected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            executor = FakeExecutor([0])
            config = HabitatPixelNavMemorySmokeConfig(
                scene_path="scene.glb",
                output_dir=temp_dir,
                start_position_xyz=[0.0, 0.0, 0.0],
                start_heading_rad=0.0,
                goal_map_xy=[5.0, 0.0],
                selected_view_type="front",
                selected_image_point=(3, 2),
                image_width=6,
                image_height=4,
                max_agent_steps=1,
                max_pixelnav_steps=4,
                force_front_view_waypoint=False,
            )

            result = run_habitat_pixelnav_memory_smoke(
                config,
                runner_factory=lambda _backend_config: runner,
                executor_factory=lambda: executor,
            )
            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))

        self.assertEqual(summary["steps"][0]["action"], "go")
        self.assertFalse(summary["steps"][0]["outcome"]["success"])
        self.assertTrue(summary["steps"][0]["outcome"]["no_progress"])
        self.assertEqual(summary["steps"][0]["outcome"]["moved_distance_m"], 0.0)
        self.assertTrue(summary["outcome_checks"]["no_progress_detected"])
        self.assertEqual(summary["outcome_checks"]["no_progress_count"], 1)
        self.assertEqual(summary["outcome_checks"]["first_no_progress_step_index"], 0)
        self.assertEqual(summary["memory"]["deadlock_state"]["status"], "suspected")
        self.assertEqual(summary["memory"]["deadlock_state"]["risk_level"], "medium")

    def test_run_memory_smoke_confirms_negative_memory_after_repeated_no_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            executor = FakeExecutor([1, 1, 1, 1, 0, 0])
            config = HabitatPixelNavMemorySmokeConfig(
                scene_path="scene.glb",
                output_dir=temp_dir,
                start_position_xyz=[0.0, 0.0, 0.0],
                start_heading_rad=0.0,
                goal_map_xy=[5.0, 0.0],
                selected_view_type="front",
                selected_image_point=(3, 2),
                image_width=6,
                image_height=4,
                max_agent_steps=3,
                max_pixelnav_steps=4,
                force_front_view_waypoint=False,
            )

            result = run_habitat_pixelnav_memory_smoke(
                config,
                runner_factory=lambda _backend_config: runner,
                executor_factory=lambda: executor,
            )
            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))

        self.assertEqual(summary["outcome_checks"]["no_progress_count"], 2)
        self.assertTrue(summary["outcome_checks"]["confirmed_negative_memory_detected"])
        self.assertEqual(summary["outcome_checks"]["confirmed_negative_node_count"], 1)
        self.assertEqual(summary["memory"]["deadlock_state"]["status"], "confirmed")
        self.assertEqual(summary["memory"]["deadlock_state"]["risk_level"], "high")
        self.assertIsNotNone(summary["memory"]["deadlock_state"]["negative_memory"])
        failed_edge_id = summary["memory"]["deadlock_state"]["negative_memory"]["failed_entry_edge_id"]
        self.assertIsNotNone(failed_edge_id)
        self.assertTrue(summary["memory"]["negative_nodes"])
        self.assertEqual(summary["memory"]["negative_nodes"][0]["deadlock_status"], "confirmed")
        deadlock_edges = [
            edge
            for edge in summary["memory"]["temporal_edges"]
            if edge["edge_id"] == failed_edge_id
        ]
        self.assertEqual(len(deadlock_edges), 1)
        self.assertEqual(deadlock_edges[0]["status"], "deadlock_entry")

    def test_run_memory_smoke_exposes_confirmed_negative_memory_in_vlm_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            executor = FakeExecutor([1, 1, 1, 1, 0, 0, 0])
            config = HabitatPixelNavMemorySmokeConfig(
                scene_path="scene.glb",
                output_dir=temp_dir,
                start_position_xyz=[0.0, 0.0, 0.0],
                start_heading_rad=0.0,
                goal_map_xy=[5.0, 0.0],
                selected_view_type="front",
                selected_image_point=(3, 2),
                image_width=6,
                image_height=4,
                max_agent_steps=4,
                max_pixelnav_steps=4,
                force_front_view_waypoint=False,
            )

            result = run_habitat_pixelnav_memory_smoke(
                config,
                runner_factory=lambda _backend_config: runner,
                executor_factory=lambda: executor,
            )
            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))

        self.assertTrue(summary["memory_context_checks"]["failed_waypoints_detected"])
        self.assertTrue(summary["memory_context_checks"]["compressed_negative_memory_detected"])
        self.assertEqual(summary["memory_context_checks"]["max_failed_waypoints_count"], 1)
        self.assertEqual(summary["memory_context_checks"]["max_compressed_negative_memory_count"], 1)
        fourth_context = summary["steps"][3]["memory_context"]
        self.assertEqual(fourth_context["deadlock_status"], "confirmed")
        self.assertEqual(fourth_context["failed_waypoints_count"], 1)
        self.assertEqual(fourth_context["compressed_negative_memory_count"], 1)
        self.assertEqual(fourth_context["failed_waypoint_statuses"], ["deadlock_entry"])

    def test_run_memory_smoke_writes_compact_timeline_for_reporting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            executor = FakeExecutor([1, 1, 1, 1, 0, 0])
            config = HabitatPixelNavMemorySmokeConfig(
                scene_path="scene.glb",
                output_dir=temp_dir,
                start_position_xyz=[0.0, 0.0, 0.0],
                start_heading_rad=0.0,
                goal_map_xy=[5.0, 0.0],
                selected_view_type="front",
                selected_image_point=(3, 2),
                image_width=6,
                image_height=4,
                max_agent_steps=3,
                max_pixelnav_steps=4,
                force_front_view_waypoint=False,
            )

            result = run_habitat_pixelnav_memory_smoke(
                config,
                runner_factory=lambda _backend_config: runner,
                executor_factory=lambda: executor,
            )
            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))

        self.assertEqual(len(summary["timeline"]), 3)
        self.assertEqual(summary["timeline"][0]["action"], "go")
        self.assertEqual(summary["timeline"][0]["selected_view"], "front")
        self.assertTrue(summary["timeline"][0]["outcome_success"])
        self.assertEqual(summary["timeline"][0]["memory_signal"], "normal")
        self.assertEqual(summary["timeline"][1]["memory_signal"], "suspected_no_progress")
        self.assertEqual(summary["timeline"][2]["memory_signal"], "confirmed_negative_memory")
        self.assertTrue(summary["timeline_summary"]["has_successful_motion"])
        self.assertTrue(summary["timeline_summary"]["has_no_progress"])
        self.assertTrue(summary["timeline_summary"]["has_confirmed_negative_memory"])
        self.assertAlmostEqual(summary["timeline_summary"]["total_moved_distance_m"], 1.0)


if __name__ == "__main__":
    unittest.main()
