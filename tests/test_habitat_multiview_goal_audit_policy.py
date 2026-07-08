import unittest

from goal_adapter.habitat_multiview_goal_audit import (
    MultiViewGoalAuditConfig,
    _build_chat_completion_payload,
    _build_multiview_prompt,
)


class MultiViewGoalAuditPolicyPromptTest(unittest.TestCase):
    def test_default_policy_keeps_general_best_waypoint_instruction(self):
        config = MultiViewGoalAuditConfig(scene_path="scene.glb", output_dir="out")

        prompt = _build_multiview_prompt(config, [0.0, 0.0], [5.0, 0.0])

        self.assertIn("best next local waypoint", prompt)
        self.assertNotIn("farthest visible navigable floor point", prompt)

    def test_farthest_visible_policy_prioritizes_far_free_space(self):
        config = MultiViewGoalAuditConfig(
            scene_path="scene.glb",
            output_dir="out",
            waypoint_policy="farthest_visible",
        )

        prompt = _build_multiview_prompt(config, [0.0, 0.0], [5.0, 0.0])

        self.assertIn("farthest visible navigable floor point", prompt)
        self.assertIn("Do not choose a nearby floor point if a farther safe floor point exists", prompt)
        self.assertIn("near the floor-wall boundary", prompt)
        self.assertIn("avoid the lower foreground", prompt)
        self.assertIn("S2E/PixelNav", prompt)

    def test_disable_thinking_adds_chat_template_kwargs_to_payload(self):
        config = MultiViewGoalAuditConfig(
            scene_path="scene.glb",
            output_dir="out",
            model="qwen3-vl-32b-thinking",
            disable_thinking=True,
        )

        payload = _build_chat_completion_payload([{"type": "text", "text": "Return JSON."}], config)

        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["model"], "qwen3-vl-32b-thinking")


if __name__ == "__main__":
    unittest.main()
