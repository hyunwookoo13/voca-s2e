import unittest
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from goal_adapter.habitat_pixelnav_execute import (
    PixelNavStep,
    draw_selected_point_overlay,
    load_rollout_request_from_audit,
    pixelnav_action_to_sim_action,
    run_pixelnav_rollout_loop,
)


class FakeStepResult:
    def __init__(self, action, overlay_image=None):
        self.action = action
        self.overlay_image = overlay_image if overlay_image is not None else np.zeros((2, 2, 3), dtype=np.uint8)


class FakeExecutor:
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = []

    def step(self, obs_rgb, collide=False):
        self.calls.append((obs_rgb.copy(), collide))
        action = self.actions.pop(0) if self.actions else 0
        return FakeStepResult(action)


class FakeRunner:
    def __init__(self):
        self.positions = [[0.0, 0.0, 0.0]]
        self.actions = []
        self.collided = False

    def rgb(self):
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def previous_step_collided(self):
        return self.collided

    def position_xyz(self):
        return list(self.positions[-1])

    def supported_action_names(self):
        return {"move_forward", "turn_left", "turn_right"}

    def step(self, action_name):
        self.actions.append(action_name)
        last = self.positions[-1]
        if action_name == "move_forward":
            self.positions.append([last[0] + 0.25, last[1], last[2]])
        else:
            self.positions.append(list(last))


class PixelNavActionMappingTest(unittest.TestCase):
    def test_maps_pixelnav_actions_to_habitat_sim_actions(self):
        self.assertIsNone(pixelnav_action_to_sim_action(0, {"move_forward"}))
        self.assertEqual(pixelnav_action_to_sim_action(1, {"move_forward"}), "move_forward")
        self.assertEqual(pixelnav_action_to_sim_action(2, {"turn_left"}), "turn_left")
        self.assertEqual(pixelnav_action_to_sim_action(3, {"turn_right"}), "turn_right")

    def test_unsupported_action_returns_none(self):
        self.assertIsNone(pixelnav_action_to_sim_action(5, {"move_forward"}))


class PixelNavRolloutLoopTest(unittest.TestCase):
    def test_rollout_records_movement_and_stops_on_stop_action(self):
        runner = FakeRunner()
        executor = FakeExecutor([1, 0])

        summary = run_pixelnav_rollout_loop(executor, runner, max_steps=5)

        self.assertEqual([step.pixelnav_action for step in summary.steps], [1, 0])
        self.assertEqual(runner.actions, ["move_forward"])
        self.assertAlmostEqual(summary.total_moved_distance_m, 0.25)
        self.assertTrue(summary.stopped_by_policy)
        self.assertIsNone(summary.unsupported_action)

    def test_rollout_stops_on_unsupported_action(self):
        runner = FakeRunner()
        executor = FakeExecutor([5])

        summary = run_pixelnav_rollout_loop(executor, runner, max_steps=5)

        self.assertEqual(len(summary.steps), 1)
        self.assertEqual(summary.unsupported_action, 5)
        self.assertFalse(summary.stopped_by_policy)
        self.assertEqual(runner.actions, [])

    def test_rollout_forwards_collision_flag_to_executor(self):
        runner = FakeRunner()
        runner.collided = True
        executor = FakeExecutor([0])

        run_pixelnav_rollout_loop(executor, runner, max_steps=1)

        self.assertTrue(executor.calls[0][1])

    def test_step_serializes_to_json(self):
        step = PixelNavStep(
            step_index=0,
            pixelnav_action=1,
            sim_action="move_forward",
            collision_before=False,
            position_before_xyz=[0.0, 0.0, 0.0],
            position_after_xyz=[0.25, 0.0, 0.0],
            moved_distance_m=0.25,
            overlay_path="overlay_000.png",
        )

        self.assertEqual(step.to_json()["sim_action"], "move_forward")
        self.assertEqual(step.to_json()["moved_distance_m"], 0.25)


class RolloutRequestFromAuditTest(unittest.TestCase):
    def test_loads_selected_view_heading_and_goal_image_path(self):
        payload = {
            "scene_path": "scene.glb",
            "robot_position_xyz": [1.0, 0.0, 2.0],
            "heading": 1.0,
            "selected_view": "left",
            "selected_image_point": [320, 300],
            "views": {
                "left": {
                    "rgb": "left_rgb.png",
                    "overlay": "left_overlay.png",
                }
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            request = load_rollout_request_from_audit(path)

        self.assertEqual(request.scene_path, "scene.glb")
        self.assertEqual(request.robot_position_xyz, [1.0, 0.0, 2.0])
        self.assertEqual(request.selected_view, "left")
        self.assertEqual(request.selected_image_point, [320, 300])
        self.assertAlmostEqual(request.rollout_heading, 1.0 - math.pi / 2.0)
        self.assertEqual(request.goal_rgb_path, "left_rgb.png")


class RolloutVisualizationTest(unittest.TestCase):
    def test_draw_selected_point_overlay_marks_point_without_mutating_input(self):
        rgb = np.zeros((20, 20, 3), dtype=np.uint8)

        overlay = draw_selected_point_overlay(rgb, [10, 10])

        self.assertEqual(rgb.sum(), 0)
        self.assertGreater(overlay.sum(), 0)
        self.assertTrue((overlay[10, 10] == np.array([220, 38, 38])).all())


if __name__ == "__main__":
    unittest.main()
