import base64
import json
import tempfile
import unittest
from pathlib import Path

from goal_adapter.ollama_provider import OllamaVLMConfig, OllamaVLMProvider
from goal_adapter.schema import ActionType


class RecordingHttpPost:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, endpoint, payload, timeout_seconds):
        self.calls.append(
            {
                "endpoint": endpoint,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        return self.response


class OllamaVLMProviderTest(unittest.TestCase):
    def test_decide_sends_chat_payload_with_base64_images_and_parses_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "current.png"
            image_path.write_bytes(b"fake-image-bytes")
            http_post = RecordingHttpPost(
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "action_type": "NAVIGATE",
                                "refined_goal_xy": [1.2, -0.4],
                                "selected_image_point": [320, 220],
                                "reasoning": "The visible chair front has a navigable floor point.",
                                "confidence": "medium",
                            }
                        )
                    }
                }
            )
            provider = OllamaVLMProvider(
                config=OllamaVLMConfig(
                    model="gemma4:26b",
                    endpoint="http://localhost:11434/api/chat",
                    timeout_seconds=9.0,
                ),
                http_post=http_post,
            )

            output = provider.decide(
                {
                    "target_type": "object_point",
                    "high_level_target": {"label": "chair"},
                    "current_rgb": str(image_path),
                    "progress_state": "normal",
                    "memory_summary": {
                        "candidate_waypoints": [
                            {"goal_xy": [1.2, -0.4], "navigable": True}
                        ]
                    },
                }
            )

        self.assertEqual(output.action_type, ActionType.NAVIGATE)
        self.assertEqual(output.refined_goal_xy, [1.2, -0.4])
        self.assertEqual(len(http_post.calls), 1)
        call = http_post.calls[0]
        self.assertEqual(call["endpoint"], "http://localhost:11434/api/chat")
        self.assertEqual(call["timeout_seconds"], 9.0)
        payload = call["payload"]
        self.assertEqual(payload["model"], "gemma4:26b")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(
            payload["messages"][1]["images"],
            [base64.b64encode(b"fake-image-bytes").decode("ascii")],
        )
        self.assertIn("format", payload)
        self.assertEqual(payload["options"]["temperature"], 1.0)


if __name__ == "__main__":
    unittest.main()
