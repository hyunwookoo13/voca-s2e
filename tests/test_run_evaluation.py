import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from goal_adapter.run_evaluation import main, run_evaluation


class GoalAdapterEvaluationRunnerTest(unittest.TestCase):
    def test_run_evaluation_writes_json_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "reports" / "goal_adapter_eval.json"

            report = run_evaluation(output_path)

            self.assertTrue(output_path.exists())
            saved_report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_report, report)
            self.assertEqual(report["total_cases"], 12)
            self.assertEqual(report["metrics"]["goal_xy_validity_rate"], 1.0)
            self.assertEqual(report["metrics"]["failure_repeat_rate"], 0.0)

    def test_main_returns_success_for_default_fixture_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "eval.json"

            with redirect_stdout(StringIO()):
                exit_code = main(["--output", str(output_path)])

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())

    def test_main_can_write_diagnostics_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "eval.json"
            diagnostics_path = Path(temp_dir) / "diagnostics.json"

            with redirect_stdout(StringIO()):
                exit_code = main(
                    [
                        "--output",
                        str(output_path),
                        "--diagnostics-output",
                        str(diagnostics_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            self.assertEqual(diagnostics["total_diagnostics"], 4)
            self.assertEqual(diagnostics["detection_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
