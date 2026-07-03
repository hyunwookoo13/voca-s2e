import unittest

from goal_adapter.evaluation_diagnostics import (
    EVALUATOR_DIAGNOSTIC_CASES,
    run_evaluator_diagnostics,
)


class GoalAdapterEvaluationDiagnosticsTest(unittest.TestCase):
    def test_diagnostics_detect_intentional_evaluation_failures(self):
        report = run_evaluator_diagnostics()

        self.assertEqual(report["total_diagnostics"], 4)
        self.assertEqual(report["detected_failures"], 4)
        self.assertEqual(report["detection_rate"], 1.0)
        self.assertTrue(all(item["detected"] for item in report["diagnostics"]))

    def test_diagnostics_cover_key_failure_modes(self):
        failure_modes = {case["failure_mode"] for case in EVALUATOR_DIAGNOSTIC_CASES}

        self.assertEqual(
            failure_modes,
            {
                "wrong_waypoint",
                "wrong_action_type",
                "repeated_failed_goal",
                "wrong_controller_action",
            },
        )


if __name__ == "__main__":
    unittest.main()
