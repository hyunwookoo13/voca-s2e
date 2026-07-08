import json
import io
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
from goal_adapter.habitat_pixelnav_real_vlm_endpoint_smoke import (
    main,
    run_habitat_pixelnav_real_vlm_endpoint_smoke,
    run_habitat_pixelnav_real_vlm_endpoint_smoke_from_env,
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


class EndpointSmokeVLMClient:
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
            short_text="endpoint smoke test selects front floor",
            confidence="high",
        )
        output["memory_ops"] = []
        return output


class HabitatPixelNavRealVLMEndpointSmokeTest(unittest.TestCase):
    def test_endpoint_smoke_validates_real_vlm_output_contract_without_backend_action(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            config = HabitatPixelNavMemorySmokeConfig(
                scene_path="scene.glb",
                output_dir=temp_dir,
                start_position_xyz=[0.0, 0.0, 0.0],
                goal_map_xy=[5.0, 0.0],
                image_width=6,
                image_height=4,
                force_front_view_waypoint=True,
            )

            result = run_habitat_pixelnav_real_vlm_endpoint_smoke(
                config,
                vlm_client=EndpointSmokeVLMClient(),
                runner_factory=lambda _backend_config: runner,
            )
            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))

        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["endpoint_smoke"]["passed"])
        self.assertFalse(summary["dry_run"]["executed_backend_action"])
        self.assertEqual(runner.actions, [])
        checks = summary["endpoint_smoke"]["checks"]
        self.assertTrue(checks["schema_version_valid"])
        self.assertTrue(checks["action_valid"])
        self.assertTrue(checks["selected_view_type_valid"])
        self.assertTrue(checks["selected_image_point_valid"])
        self.assertTrue(checks["memory_ops_valid"])

    def test_endpoint_smoke_from_env_skips_when_endpoint_env_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = HabitatPixelNavMemorySmokeConfig(
                scene_path="scene.glb",
                output_dir=temp_dir,
                start_position_xyz=[0.0, 0.0, 0.0],
                goal_map_xy=[5.0, 0.0],
            )

            with patch.dict(os.environ, {}, clear=True):
                result = run_habitat_pixelnav_real_vlm_endpoint_smoke_from_env(config)

            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))

        self.assertEqual(summary["status"], "skipped")
        self.assertTrue(summary["skipped"])
        self.assertEqual(summary["reason"], "missing_qwen_endpoint_env")

    def test_endpoint_smoke_cli_writes_skip_artifact_without_endpoint_env(self):
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

            summary = json.loads((out_dir / "real_vlm_endpoint_smoke.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
