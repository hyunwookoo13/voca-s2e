import unittest

from goal_adapter import GoalAdapter
from goal_adapter.schema import ActionType, Confidence, GoalAdapterInput, GoalAdapterOutput


class RecordingDecisionProvider:
    def __init__(self, output):
        self.output = output
        self.inputs = []

    def decide(self, adapter_input):
        self.inputs.append(adapter_input)
        return self.output


class GoalAdapterVLMProviderTest(unittest.TestCase):
    def test_refine_uses_decision_provider_when_supplied(self):
        provider_output = GoalAdapterOutput(
            refined_goal_xy=[2.0, 3.0],
            selected_image_point=[300, 200],
            action_type=ActionType.NAVIGATE,
            reasoning="The VLM selected the object-front floor point.",
            confidence=Confidence.MEDIUM,
        )
        provider = RecordingDecisionProvider(provider_output)
        adapter = GoalAdapter(decision_provider=provider)

        output = adapter.refine(
            {
                "target_type": "object_point",
                "high_level_target": {"label": "chair"},
                "current_rgb": "frames/current.png",
            }
        )

        self.assertIs(output, provider_output)
        self.assertEqual(len(provider.inputs), 1)
        self.assertIsInstance(provider.inputs[0], GoalAdapterInput)
        self.assertEqual(provider.inputs[0].target_type.value, "object_point")


if __name__ == "__main__":
    unittest.main()
