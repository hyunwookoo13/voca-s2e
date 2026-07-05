import unittest

from goal_adapter.decision_parser import VLMDecisionParseError, parse_vlm_decision
from goal_adapter.schema import ActionType, ControllerAction


class VLMDecisionParserTest(unittest.TestCase):
    def test_parses_plain_json_navigation_decision(self):
        output = parse_vlm_decision(
            """
            {
              "action_type": "NAVIGATE",
              "refined_goal_xy": [1.2, -0.4],
              "selected_image_point": [320, 220],
              "reasoning": "The doorway is the next navigable waypoint.",
              "confidence": "medium"
            }
            """
        )

        self.assertEqual(output.action_type, ActionType.NAVIGATE)
        self.assertEqual(output.refined_goal_xy, [1.2, -0.4])
        self.assertEqual(output.selected_image_point, [320, 220])
        self.assertIsNone(output.controller_action)

    def test_parses_json_object_inside_markdown_fence(self):
        output = parse_vlm_decision(
            """
            The decision is:

            ```json
            {
              "action_type": "LOOK_AROUND",
              "refined_goal_xy": null,
              "selected_image_point": null,
              "reasoning": "The target point is missing, so more views are needed.",
              "confidence": "low"
            }
            ```
            """
        )

        self.assertEqual(output.action_type, ActionType.LOOK_AROUND)
        self.assertIsNone(output.refined_goal_xy)

    def test_parses_common_vlm_field_aliases(self):
        output = parse_vlm_decision(
            {
                "action_type": "RESELECT_GOAL",
                "refined_goal_calibrated_xy": [11.25, -5.35],
                "selected_image_post_point": [126, 52],
                "controller_action": None,
                "reasoning": "The previous local goal is blocked, so select a new candidate.",
                "confidence": "high",
            }
        )

        self.assertEqual(output.action_type, ActionType.RESELECT_GOAL)
        self.assertEqual(output.refined_goal_xy, [11.25, -5.35])
        self.assertEqual(output.selected_image_point, [126, 52])

    def test_parses_stop_or_direct_controller_decision_without_goal(self):
        output = parse_vlm_decision(
            {
                "action_type": "STOP",
                "refined_goal_xy": None,
                "selected_image_point": None,
                "controller_action": "STOP",
                "reasoning": "The goal condition is satisfied.",
                "confidence": "high",
            }
        )

        self.assertEqual(output.action_type, ActionType.STOP)
        self.assertEqual(output.controller_action, ControllerAction.STOP)

    def test_rejects_text_without_json_object(self):
        with self.assertRaisesRegex(VLMDecisionParseError, "JSON object"):
            parse_vlm_decision("go forward through the doorway")

    def test_rejects_schema_invalid_decision(self):
        with self.assertRaisesRegex(VLMDecisionParseError, "refined_goal_xy"):
            parse_vlm_decision(
                {
                    "action_type": "NAVIGATE",
                    "refined_goal_xy": None,
                    "selected_image_point": None,
                    "reasoning": "This lacks a required navigation goal.",
                    "confidence": "medium",
                }
            )


if __name__ == "__main__":
    unittest.main()
