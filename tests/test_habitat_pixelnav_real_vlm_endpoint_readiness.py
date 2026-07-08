import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from goal_adapter.habitat_pixelnav_real_vlm_endpoint_readiness import (
    build_real_vlm_endpoint_readiness,
    main,
)


class HabitatPixelNavRealVLMEndpointReadinessTest(unittest.TestCase):
    def test_readiness_blocks_only_missing_endpoint_env_when_local_assets_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            scene = Path(temp_dir) / "scene.glb"
            checkpoint = Path(temp_dir) / "navigator.pth"
            scene.write_bytes(b"scene")
            checkpoint.write_bytes(b"checkpoint")
            audit = Path(temp_dir) / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "scene_path": str(scene),
                        "robot_position_xyz": [0.0, 0.0, 0.0],
                        "coarse_goal_xy": [1.0, 0.0],
                        "selected_view": "front",
                        "selected_image_point": [3, 2],
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                result = build_real_vlm_endpoint_readiness(
                    audit,
                    output_dir=Path(temp_dir) / "out",
                    checkpoint_path=checkpoint,
                )
            markdown = Path(result["docmost_markdown"]).read_text(encoding="utf-8")

        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["ready"])
        self.assertEqual(result["next_action"], "set_qwen_endpoint_env_and_rerun_step_t")
        self.assertEqual(result["missing_env"], ["QWEN_BASE_URL", "QWEN_API_KEY"])
        self.assertTrue(result["checks"]["audit_result_exists"])
        self.assertTrue(result["checks"]["scene_exists"])
        self.assertTrue(result["checks"]["checkpoint_exists"])
        self.assertIn("QWEN_BASE_URL", markdown)
        self.assertIn("Step T 재실행", markdown)

    def test_readiness_is_ready_when_endpoint_env_and_local_assets_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            scene = Path(temp_dir) / "scene.glb"
            checkpoint = Path(temp_dir) / "navigator.pth"
            scene.write_bytes(b"scene")
            checkpoint.write_bytes(b"checkpoint")
            audit = Path(temp_dir) / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "scene_path": str(scene),
                        "robot_position_xyz": [0.0, 0.0, 0.0],
                        "coarse_goal_xy": [1.0, 0.0],
                        "selected_view": "front",
                        "selected_image_point": [3, 2],
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"QWEN_BASE_URL": "http://qwen.test/v1", "QWEN_API_KEY": "secret"}, clear=True):
                result = build_real_vlm_endpoint_readiness(
                    audit,
                    output_dir=Path(temp_dir) / "out",
                    checkpoint_path=checkpoint,
                )

        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["ready"])
        self.assertEqual(result["missing_env"], [])
        self.assertEqual(result["next_action"], "run_step_t_real_vlm_tiny_e2e")

    def test_readiness_cli_writes_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            scene = Path(temp_dir) / "scene.glb"
            checkpoint = Path(temp_dir) / "navigator.pth"
            scene.write_bytes(b"scene")
            checkpoint.write_bytes(b"checkpoint")
            audit = Path(temp_dir) / "audit.json"
            out_dir = Path(temp_dir) / "out"
            audit.write_text(json.dumps({"scene_path": str(scene)}), encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True), patch("sys.stdout", new_callable=io.StringIO):
                exit_code = main([
                    "--audit-result",
                    str(audit),
                    "--out",
                    str(out_dir),
                    "--checkpoint",
                    str(checkpoint),
                ])
            json_exists = (out_dir / "real_vlm_endpoint_readiness.json").exists()
            markdown_exists = (out_dir / "docmost_real_vlm_endpoint_readiness.md").exists()

        self.assertEqual(exit_code, 0)
        self.assertTrue(json_exists)
        self.assertTrue(markdown_exists)


if __name__ == "__main__":
    unittest.main()
