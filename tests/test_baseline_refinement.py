import unittest

from goal_adapter import GoalAdapter
from goal_adapter.schema import ActionType, ControllerAction

from goal_adapter_fixtures import REFINEMENT_CASES


class GoalAdapterBaselineRefinementTest(unittest.TestCase):
    def test_refines_four_first_stage_cases(self):
        adapter = GoalAdapter()

        for case in REFINEMENT_CASES:
            with self.subTest(case=case["name"]):
                output = adapter.refine(case["input"])

                self.assertEqual(output.refined_goal_xy, case["expected_goal_xy"])
                self.assertEqual(output.selected_image_point, case["expected_image_point"])
                self.assertEqual(output.action_type.value, case["expected_action_type"])
                self.assertEqual(output.confidence.value, case["expected_confidence"])
                for term in case["reasoning_terms"]:
                    self.assertIn(term, output.reasoning.lower())

    def test_direct_control_fallback_when_s2e_fails_without_candidate(self):
        adapter = GoalAdapter()

        output = adapter.refine(
            {
                "target_type": "language",
                "high_level_target": "go to the bathroom",
                "current_goal_xy": [1.0, 0.0],
                "progress_state": "blocked",
                "s2e_status": "no_valid_trajectory",
                "memory_summary": {
                    "failed_goal_xy": [[1.0, 0.0]],
                    "candidate_waypoints": [],
                },
            }
        )

        self.assertIsNone(output.refined_goal_xy)
        self.assertEqual(output.action_type, ActionType.DIRECT_CONTROL)
        self.assertEqual(output.controller_action, ControllerAction.MOVE_BACK)
        self.assertIn("s2e", output.reasoning.lower())


if __name__ == "__main__":
    unittest.main()
