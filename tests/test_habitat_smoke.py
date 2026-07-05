import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from goal_adapter.habitat_smoke import collect_habitat_smoke_sample, main
from goal_adapter.schema import GoalAdapterInput


class HabitatSmokeCollectionTest(unittest.TestCase):
    def test_collects_rgb_png_and_goal_adapter_input_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            sample = collect_habitat_smoke_sample(
                observation={
                    "rgb": [
                        [[255, 0, 0, 255], [0, 255, 0, 255]],
                        [[0, 0, 255, 255], [255, 255, 255, 255]],
                    ]
                },
                context={
                    "target_type": "object_point",
                    "high_level_target": {"label": "chair"},
                    "current_pose": {"x": 1.0, "y": -0.5},
                    "heading": 1.57,
                    "progress_state": "normal",
                    "s2e_status": "unknown",
                    "memory_summary": {
                        "candidate_waypoints": [
                            {
                                "goal_xy": [1.2, -0.4],
                                "kind": "object-front floor point",
                                "navigable": True,
                            }
                        ]
                    },
                },
                output_dir=output_dir,
            )

            self.assertEqual(sample.image_path, output_dir / "current_rgb.png")
            self.assertEqual(sample.input_path, output_dir / "goal_adapter_input.json")
            self.assertTrue(sample.image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

            payload = json.loads(sample.input_path.read_text(encoding="utf-8"))
            parsed = GoalAdapterInput.from_json(payload)
            self.assertEqual(parsed.target_type.value, "object_point")
            self.assertEqual(parsed.current_rgb, str(sample.image_path.resolve()))
            self.assertEqual(parsed.high_level_target, {"label": "chair"})
            self.assertEqual(
                parsed.memory_summary["candidate_waypoints"][0]["kind"],
                "object-front floor point",
            )

    def test_cli_collects_from_observation_and_context_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            observation_path = root / "observation.json"
            context_path = root / "context.json"
            output_dir = root / "smoke"
            observation_path.write_text(
                json.dumps({"rgb": [[[10, 20, 30], [40, 50, 60]]]}),
                encoding="utf-8",
            )
            context_path.write_text(
                json.dumps(
                    {
                        "target_type": "language",
                        "high_level_target": "go to the kitchen",
                        "progress_state": "normal",
                    }
                ),
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "--observation-json",
                        str(observation_path),
                        "--context-json",
                        str(context_path),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "current_rgb.png").exists())
            payload = json.loads((output_dir / "goal_adapter_input.json").read_text())
            self.assertEqual(payload["target_type"], "language")
            self.assertEqual(payload["current_rgb"], str((output_dir / "current_rgb.png").resolve()))


if __name__ == "__main__":
    unittest.main()
