import json
import tempfile
import unittest
from pathlib import Path

from goal_adapter.schema import ActionType, Confidence, GoalAdapterOutput
from goal_adapter.vlm_smoke_monitor import run_vlm_smoke_monitor


class FakeDecisionProvider:
    def __init__(self, output):
        self.output = output
        self.inputs = []

    def decide(self, adapter_input):
        self.inputs.append(adapter_input)
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


class FakeRawProvider:
    def __init__(self, raw_decision):
        self.raw_decision = raw_decision

    def generate_raw_decision(self, adapter_input):
        return self.raw_decision


class VLMSmokeMonitorTest(unittest.TestCase):
    def test_monitor_writes_success_artifacts_and_static_html(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_image = root / "source.png"
            source_image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
            input_path = root / "goal_adapter_input.json"
            input_path.write_text(
                json.dumps(
                    {
                        "target_type": "language",
                        "high_level_target": "inspect the room",
                        "current_rgb": str(source_image),
                        "current_pose": {"x": 1.0, "y": 2.0},
                        "progress_state": "normal",
                    }
                ),
                encoding="utf-8",
            )
            provider = FakeDecisionProvider(
                GoalAdapterOutput(
                    refined_goal_xy=[1.2, -0.4],
                    selected_image_point=[80, 60],
                    action_type=ActionType.NAVIGATE,
                    reasoning="The path ahead is open.",
                    confidence=Confidence.HIGH,
                )
            )

            report = run_vlm_smoke_monitor(
                input_path=input_path,
                output_dir=root / "report",
                decision_provider=provider,
                model_name="fake-vlm",
            )

            self.assertEqual(report.status, "success")
            self.assertEqual(len(provider.inputs), 1)
            self.assertEqual(provider.inputs[0].current_rgb, str(report.image_path.resolve()))
            self.assertTrue(report.image_path.exists())
            self.assertTrue(report.input_path.exists())
            self.assertTrue(report.decision_path.exists())
            self.assertTrue(report.raw_response_path.exists())
            self.assertTrue(report.summary_path.exists())
            self.assertTrue(report.html_path.exists())

            decision = json.loads(report.decision_path.read_text(encoding="utf-8"))
            summary = json.loads(report.summary_path.read_text(encoding="utf-8"))
            html = report.html_path.read_text(encoding="utf-8")

            self.assertEqual(decision["action_type"], "NAVIGATE")
            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["model"], "fake-vlm")
            self.assertEqual(summary["action_type"], "NAVIGATE")
            self.assertEqual(summary["confidence"], "high")
            self.assertIn('"action_type": "NAVIGATE"', report.raw_response_path.read_text(encoding="utf-8"))
            self.assertIn("current_rgb.png", html)
            self.assertIn("NAVIGATE", html)
            self.assertIn("The path ahead is open.", html)

    def test_monitor_records_error_artifacts_without_raising_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_image = root / "source.png"
            source_image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
            input_path = root / "goal_adapter_input.json"
            input_path.write_text(
                json.dumps(
                    {
                        "target_type": "language",
                        "high_level_target": "inspect the room",
                        "current_rgb": str(source_image),
                    }
                ),
                encoding="utf-8",
            )

            report = run_vlm_smoke_monitor(
                input_path=input_path,
                output_dir=root / "report",
                decision_provider=FakeDecisionProvider(RuntimeError("provider failed")),
            )

            self.assertEqual(report.status, "error")
            summary = json.loads(report.summary_path.read_text(encoding="utf-8"))
            decision = json.loads(report.decision_path.read_text(encoding="utf-8"))
            html = report.html_path.read_text(encoding="utf-8")

            self.assertEqual(summary["status"], "error")
            self.assertEqual(summary["error_type"], "RuntimeError")
            self.assertIn("provider failed", summary["error_message"])
            self.assertEqual(decision["status"], "error")
            self.assertIn("provider failed", html)

    def test_monitor_preserves_raw_response_when_schema_parse_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_image = root / "source.png"
            source_image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
            input_path = root / "goal_adapter_input.json"
            input_path.write_text(
                json.dumps(
                    {
                        "target_type": "language",
                        "high_level_target": "inspect the room",
                        "current_rgb": str(source_image),
                    }
                ),
                encoding="utf-8",
            )

            report = run_vlm_smoke_monitor(
                input_path=input_path,
                output_dir=root / "report",
                decision_provider=FakeRawProvider('{"action_type": "STOP"}'),
            )

            self.assertEqual(report.status, "error")
            self.assertEqual(report.raw_response_path.read_text(encoding="utf-8"), '{"action_type": "STOP"}')
            html = report.html_path.read_text(encoding="utf-8")
            self.assertIn("Raw VLM Response", html)
            self.assertIn("STOP", html)


if __name__ == "__main__":
    unittest.main()
