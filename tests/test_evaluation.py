import unittest

from goal_adapter import GoalAdapter
from goal_adapter.evaluation import evaluate_cases, evaluate_output
from goal_adapter.schema import ActionType, Confidence, GoalAdapterOutput

from goal_adapter_fixtures import REFINEMENT_CASES


class GoalAdapterEvaluationTest(unittest.TestCase):
    def test_evaluates_baseline_outputs_over_fixture_cases(self):
        summary = evaluate_cases(GoalAdapter(), REFINEMENT_CASES)

        self.assertEqual(summary.total_cases, 12)
        self.assertEqual(summary.goal_xy_validity_rate, 1.0)
        self.assertEqual(summary.waypoint_success_rate, 1.0)
        self.assertEqual(summary.action_type_accuracy, 1.0)
        self.assertEqual(summary.reasoning_correctness_rate, 1.0)
        self.assertEqual(summary.recovery_success_rate, 1.0)
        self.assertEqual(summary.failure_repeat_rate, 0.0)
        self.assertTrue(all(result.passed for result in summary.case_results))

        report = summary.to_json()
        self.assertEqual(report["total_cases"], 12)
        self.assertEqual(report["metrics"]["waypoint_success_rate"], 1.0)
        self.assertEqual(report["case_results"][0]["name"], REFINEMENT_CASES[0]["name"])

    def test_detects_repeated_failed_goal_in_recovery_case(self):
        blocked_case = next(
            case
            for case in REFINEMENT_CASES
            if case["name"] == "blocked_deadlock_reselects_unfailed_waypoint"
        )
        repeated_output = GoalAdapterOutput(
            refined_goal_xy=[1.0, 0.0],
            selected_image_point=[320, 220],
            action_type=ActionType.RESELECT_GOAL,
            reasoning="The blocked direction was selected again.",
            confidence=Confidence.LOW,
        )

        result = evaluate_output(blocked_case, repeated_output)

        self.assertFalse(result.passed)
        self.assertTrue(result.goal_xy_valid)
        self.assertFalse(result.waypoint_success)
        self.assertFalse(result.recovery_success)
        self.assertTrue(result.failure_repeated)

    def test_evaluates_direct_controller_recovery_without_goal_xy(self):
        direct_control_case = {
            "name": "s2e_failed_direct_recovery",
            "input": {
                "target_type": "language",
                "progress_state": "blocked",
                "s2e_status": "no_valid_trajectory",
                "memory_summary": {"failed_goal_xy": [[1.0, 0.0]]},
            },
            "expected_action_type": "DIRECT_CONTROL",
            "expected_controller_action": "MOVE_BACK",
            "reasoning_terms": ["s2e", "controller"],
        }
        output = GoalAdapterOutput(
            refined_goal_xy=None,
            selected_image_point=None,
            action_type=ActionType.DIRECT_CONTROL,
            controller_action="MOVE_BACK",
            reasoning="S2E failed, so the adapter requests a controller recovery action.",
            confidence=Confidence.LOW,
        )

        result = evaluate_output(direct_control_case, output)

        self.assertTrue(result.passed)
        self.assertTrue(result.controller_action_correct)
        self.assertTrue(result.recovery_success)


if __name__ == "__main__":
    unittest.main()
