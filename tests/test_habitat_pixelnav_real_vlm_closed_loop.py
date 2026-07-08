import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from goal_adapter.habitat_pixelnav_closed_loop import HabitatPixelNavMemorySmokeConfig
from goal_adapter.habitat_pixelnav_real_vlm_closed_loop import (
    FallbackOnVLMErrorClient,
    run_habitat_pixelnav_real_vlm_closed_loop,
)


class FakeRunner:
    def __init__(self):
        self.position = [0.0, 0.0, 0.0]
        self.heading = 0.0

    def set_pose(self, position_xyz, heading):
        self.position = [float(v) for v in position_xyz]
        self.heading = float(heading)

    def position_xyz(self):
        return list(self.position)

    def rgb(self):
        return np.full((4, 6, 3), 96, dtype=np.uint8)

    def previous_step_collided(self):
        return False

    def supported_action_names(self):
        return {"move_forward", "turn_left", "turn_right"}

    def step(self, action_name):
        if action_name == "move_forward":
            self.position = [self.position[0] + 0.25, self.position[1], self.position[2]]


class FakeGoal:
    goal_mask = np.ones((4, 6), dtype=np.uint8)


class FakeExecutor:
    def __init__(self, actions):
        self.actions = list(actions)

    def reset_from_point(self, goal_rgb, selected_point_uv, radius=5):
        return FakeGoal()

    def step(self, obs_rgb, collide=False):
        action = self.actions.pop(0) if self.actions else 0
        return type("FakeStep", (), {"action": action, "overlay_image": np.zeros_like(obs_rgb)})()


class FakeQwenClient:
    def decide(self, vlm_input):
        obs = vlm_input["observation"]
        view = obs["views"][0]
        return {
            "schema_version": "nav_vlm_waypoint_v1",
            "action": "go",
            "selected_view_id": int(view["view_id"]),
            "selected_view_type": str(view["view_type"]),
            "selected_image_point": [3, 2],
            "reasoning": {
                "reason_code": "G02_VISIBLE_FLOOR_TOWARD_GOAL",
                "short_text": "test waypoint",
                "confidence": "high",
            },
            "control": {"ttl_ms": 1000},
            "memory_ops": [],
        }


class FailingQwenClient:
    def decide(self, vlm_input):
        raise ValueError("not valid json")


class HabitatPixelNavRealVLMClosedLoopTest(unittest.TestCase):
    def test_real_vlm_closed_loop_runner_accepts_client_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            executor = FakeExecutor([1, 1, 1, 1, 1, 1, 1, 1])
            config = HabitatPixelNavMemorySmokeConfig(
                scene_path="scene.glb",
                output_dir=temp_dir,
                start_position_xyz=[0.0, 0.0, 0.0],
                start_heading_rad=0.0,
                goal_map_xy=[1.0, 0.0],
                image_width=6,
                image_height=4,
                max_agent_steps=2,
                max_pixelnav_steps=4,
            )

            result = run_habitat_pixelnav_real_vlm_closed_loop(
                config,
                vlm_client=FakeQwenClient(),
                runner_factory=lambda _backend_config: runner,
                executor_factory=lambda: executor,
            )
            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))

        self.assertEqual(summary["episode"]["steps"], 2)
        self.assertEqual(summary["config"]["max_agent_steps"], 2)
        self.assertTrue(summary["timeline_summary"]["has_successful_motion"])
        self.assertEqual(summary["runtime"]["vlm_client"], "injected")
        self.assertGreaterEqual(summary["memory"]["num_nodes"], 2)

    def test_fallback_client_returns_schema_valid_decision_when_primary_fails(self):
        client = FallbackOnVLMErrorClient(FailingQwenClient())
        output = client.decide(
            {
                "task": {"coarse_goal": {"relative_bearing_deg": 0.0, "distance_m": 2.0}},
                "observation": {
                    "image_width": 6,
                    "image_height": 4,
                    "views": [{"view_id": 0, "view_type": "front", "relative_heading_deg": 0.0}],
                },
                "memory": {},
            }
        )

        self.assertEqual(output["schema_version"], "nav_vlm_waypoint_v1")
        self.assertEqual(output["action"], "go")
        self.assertEqual(client.fallback_count, 1)
        self.assertIn("not valid json", client.fallback_errors[0])


if __name__ == "__main__":
    unittest.main()
