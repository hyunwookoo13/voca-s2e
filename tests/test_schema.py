import unittest

from goal_adapter import GoalAdapter
from goal_adapter.schema import (
    ActionType,
    Confidence,
    ControllerAction,
    GoalAdapterInput,
    GoalAdapterOutput,
    ProgressState,
    S2EStatus,
    TargetType,
)


class GoalAdapterInputSchemaTest(unittest.TestCase):
    def test_parses_valid_input_json(self):
        payload = {
            "target_type": "gps",
            "high_level_target": {"world_xy": [5.0, 2.0]},
            "current_rgb": "frames/current.png",
            "optional_lookaround_images": ["frames/left.png"],
            "current_pose": {"x": 1.0, "y": 0.5},
            "heading": 1.57,
            "current_goal_xy": [1.2, -0.4],
            "progress_state": "normal",
            "s2e_status": "failed",
            "memory_summary": {"failed": [[-1.0, 0.0]]},
            "system_prompt": "Return a local goal for S2E.",
        }

        parsed = GoalAdapterInput.from_json(payload)

        self.assertEqual(parsed.target_type, TargetType.GPS)
        self.assertEqual(parsed.progress_state, ProgressState.NORMAL)
        self.assertEqual(parsed.s2e_status, S2EStatus.FAILED)
        self.assertEqual(parsed.current_goal_xy, [1.2, -0.4])
        self.assertEqual(parsed.optional_lookaround_images, ["frames/left.png"])

    def test_rejects_unknown_target_type(self):
        with self.assertRaisesRegex(ValueError, "target_type"):
            GoalAdapterInput.from_json({"target_type": "unknown"})

    def test_rejects_malformed_goal_xy(self):
        with self.assertRaisesRegex(ValueError, "current_goal_xy"):
            GoalAdapterInput.from_json(
                {
                    "target_type": "gps",
                    "current_goal_xy": [1.0, 2.0, 3.0],
                }
            )


class GoalAdapterOutputSchemaTest(unittest.TestCase):
    def test_output_serializes_to_json_contract(self):
        output = GoalAdapterOutput(
            refined_goal_xy=[1.2, -0.4],
            selected_image_point=None,
            action_type=ActionType.NAVIGATE,
            reasoning="The current local goal is already valid for S2E.",
            confidence=Confidence.MEDIUM,
        )

        self.assertEqual(
            output.to_json(),
            {
                "refined_goal_xy": [1.2, -0.4],
                "selected_image_point": None,
                "action_type": "NAVIGATE",
                "controller_action": None,
                "reasoning": "The current local goal is already valid for S2E.",
                "confidence": "medium",
            },
        )

    def test_stop_output_allows_missing_goal_xy_with_controller_action(self):
        output = GoalAdapterOutput(
            refined_goal_xy=None,
            selected_image_point=None,
            action_type=ActionType.STOP,
            controller_action=ControllerAction.STOP,
            reasoning="The target condition is satisfied, so the robot should stop.",
            confidence=Confidence.HIGH,
        )

        self.assertEqual(
            output.to_json(),
            {
                "refined_goal_xy": None,
                "selected_image_point": None,
                "action_type": "STOP",
                "controller_action": "STOP",
                "reasoning": "The target condition is satisfied, so the robot should stop.",
                "confidence": "high",
            },
        )

    def test_direct_control_requires_controller_action(self):
        with self.assertRaisesRegex(ValueError, "controller_action"):
            GoalAdapterOutput(
                refined_goal_xy=None,
                selected_image_point=None,
                action_type=ActionType.DIRECT_CONTROL,
                reasoning="S2E failed to produce a valid trajectory.",
                confidence=Confidence.LOW,
            )

    def test_rejects_invalid_selected_image_point(self):
        with self.assertRaisesRegex(ValueError, "selected_image_point"):
            GoalAdapterOutput(
                refined_goal_xy=[1.2, -0.4],
                selected_image_point=[320],
                action_type=ActionType.NAVIGATE,
                reasoning="invalid",
                confidence=Confidence.LOW,
            )


class GoalAdapterInterfaceTest(unittest.TestCase):
    def test_refine_returns_schema_output_with_attribute_access(self):
        adapter = GoalAdapter(config={"default_goal_xy": [0.0, 0.0]})

        output = adapter.refine(
            {
                "target_type": "language",
                "high_level_target": "bathroom",
                "current_goal_xy": [1.2, -0.4],
                "progress_state": "normal",
            }
        )

        self.assertEqual(output.refined_goal_xy, [1.2, -0.4])
        self.assertEqual(output.action_type, ActionType.NAVIGATE)
        self.assertEqual(output.to_json()["confidence"], "low")


if __name__ == "__main__":
    unittest.main()
