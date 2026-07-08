import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


QWEN_ROOT = Path(__file__).resolve().parents[1] / "qwen_nav_memory_framework_v5"
if str(QWEN_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN_ROOT))

from goal_adapter.habitat_pixelnav_closed_loop import HabitatPixelNavMemorySmokeConfig
from goal_adapter.habitat_pixelnav_real_vlm_gated_execution import (
    main,
    run_habitat_pixelnav_real_vlm_gated_execution,
    run_habitat_pixelnav_real_vlm_gated_execution_from_env,
)


class FakeRunner:
    def __init__(self):
        self.position = [0.0, 0.0, 0.0]
        self.heading = 0.0
        self.actions = []

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
        self.actions.append(action_name)
        if action_name == "move_forward":
            self.position = [self.position[0] + 0.25, self.position[1], self.position[2]]


class FakeGoal:
    def __init__(self):
        self.goal_mask = np.ones((4, 6), dtype=np.uint8)


class FakeExecutor:
    def __init__(self, actions):
        self.actions = list(actions)
        self.reset_args = None

    def reset_from_point(self, goal_rgb, selected_point_uv, radius=5):
        self.reset_args = (goal_rgb.copy(), tuple(selected_point_uv), radius)
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


class GatedExecutionVLMClient:
    def decide(self, vlm_input):
        from nav_memory_qwen.schema import make_go_output

        obs = vlm_input["observation"]
        front = next(v for v in obs["views"] if v["view_type"] == "front")
        output = make_go_output(
            view_id=front["view_id"],
            view_type="front",
            point_px=(obs["image_width"] // 2, int(obs["image_height"] * 0.75)),
            width=obs["image_width"],
            height=obs["image_height"],
            decision_reason="G02_VISIBLE_FLOOR_TOWARD_GOAL",
            goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
            short_text="gated execution test selects front floor",
            confidence="high",
        )
        output["memory_ops"] = []
        return output


class HabitatPixelNavRealVLMGatedExecutionTest(unittest.TestCase):
    def test_gated_execution_runs_pixelnav_once_after_endpoint_smoke_passes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            executor = FakeExecutor([1, 1])
            config = HabitatPixelNavMemorySmokeConfig(
                scene_path="scene.glb",
                output_dir=temp_dir,
                start_position_xyz=[0.0, 0.0, 0.0],
                goal_map_xy=[5.0, 0.0],
                image_width=6,
                image_height=4,
                max_pixelnav_steps=2,
                mask_radius=1,
                force_front_view_waypoint=True,
            )

            result = run_habitat_pixelnav_real_vlm_gated_execution(
                config,
                vlm_client=GatedExecutionVLMClient(),
                runner_factory=lambda _backend_config: runner,
                executor_factory=lambda: executor,
            )
            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))

        self.assertEqual(summary["status"], "executed")
        self.assertTrue(summary["gate"]["passed"])
        self.assertTrue(summary["execution"]["executed"])
        self.assertEqual(summary["execution"]["action_outcome"]["action"], "go")
        self.assertTrue(summary["execution"]["action_outcome"]["success"])
        self.assertEqual(runner.actions, ["move_forward", "move_forward"])
        self.assertEqual(executor.reset_args[1], (3, 3))

    def test_gated_execution_from_env_skips_without_endpoint_env_and_does_not_execute(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = HabitatPixelNavMemorySmokeConfig(
                scene_path="scene.glb",
                output_dir=temp_dir,
                start_position_xyz=[0.0, 0.0, 0.0],
                goal_map_xy=[5.0, 0.0],
            )

            with patch.dict(os.environ, {}, clear=True):
                result = run_habitat_pixelnav_real_vlm_gated_execution_from_env(config)

            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))

        self.assertEqual(summary["status"], "skipped")
        self.assertFalse(summary["gate"]["passed"])
        self.assertFalse(summary["execution"]["executed"])
        self.assertEqual(summary["execution"]["reason"], "endpoint_smoke_not_passed")

    def test_gated_execution_cli_writes_skip_artifact_without_endpoint_env(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit.json"
            out_dir = Path(temp_dir) / "out"
            audit_path.write_text(
                json.dumps(
                    {
                        "scene_path": "scene.glb",
                        "robot_position_xyz": [0.0, 0.0, 0.0],
                        "heading": 0.0,
                        "coarse_goal_xy": [5.0, 0.0],
                        "selected_view": "front",
                        "selected_image_point": [3, 2],
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True), patch("sys.stdout", new_callable=io.StringIO):
                exit_code = main(["--audit-result", str(audit_path), "--out", str(out_dir)])

            summary = json.loads((out_dir / "real_vlm_gated_execution.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["status"], "skipped")
        self.assertFalse(summary["execution"]["executed"])


if __name__ == "__main__":
    unittest.main()
