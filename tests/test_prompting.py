import unittest

from goal_adapter.prompting import build_vlm_decision_prompt
from goal_adapter.schema import GoalAdapterInput


class VLMDecisionPromptTest(unittest.TestCase):
    def test_prompt_describes_decision_schema_and_navigation_constraints(self):
        adapter_input = GoalAdapterInput.from_json(
            {
                "target_type": "object_point",
                "high_level_target": {
                    "label": "chair",
                    "image_point": [320, 220],
                },
                "current_rgb": "frames/current.png",
                "optional_lookaround_images": ["frames/left.png", "frames/right.png"],
                "current_pose": {"x": 1.0, "y": -0.5},
                "heading": 1.57,
                "current_goal_xy": [0.5, 0.5],
                "progress_state": "blocked",
                "s2e_status": "collision",
                "memory_summary": {
                    "candidate_waypoints": [
                        {
                            "goal_xy": [1.2, -0.4],
                            "kind": "object-front floor point",
                            "navigable": True,
                        }
                    ],
                    "failed_goal_xy": [[0.5, 0.5]],
                },
            }
        )

        prompt = build_vlm_decision_prompt(adapter_input)

        self.assertIn("NAVIGATE", prompt.system)
        self.assertIn("LOOK_AROUND", prompt.system)
        self.assertIn("RESELECT_GOAL", prompt.system)
        self.assertIn("STOP", prompt.system)
        self.assertIn("DIRECT_CONTROL", prompt.system)
        self.assertIn("refined_goal_xy is required", prompt.system)
        self.assertIn("controller_action is required", prompt.system)
        self.assertIn("object-front navigable floor point", prompt.system)
        self.assertIn('"target_type": "object_point"', prompt.user)
        self.assertIn('"progress_state": "blocked"', prompt.user)
        self.assertIn('"s2e_status": "collision"', prompt.user)
        self.assertEqual(
            prompt.image_paths,
            ["frames/current.png", "frames/left.png", "frames/right.png"],
        )


if __name__ == "__main__":
    unittest.main()
