import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _load_qwen_request(env):
    with patch.dict(os.environ, env, clear=True):
        sys.modules.pop("llm_utils.qwen_request", None)
        return importlib.import_module("llm_utils.qwen_request")


class QwenRequestTests(unittest.TestCase):
    def test_text_response_posts_openai_compatible_payload(self):
        module = _load_qwen_request(
            {
                "QWEN_BASE_URL": "http://qwen.local:8000/v1",
                "QWEN_API_KEY": "secret",
                "QWEN_MODEL": "qwen3-vl-local",
                "QWEN_MAX_TOKENS": "1234",
                "VOCA_QWEN_SEED": "77",
                "QWEN_EXTRA_PAYLOAD_JSON": '{"response_format":{"type":"json_object"}}',
            }
        )

        with patch.object(module.requests, "post") as post:
            post.return_value = _FakeResponse(
                {"choices": [{"message": {"content": '{"Supports":["table"]}'}}]}
            )

            result = module.text_response("goal priors", system_prompt="return JSON")

        self.assertEqual(result, '{"Supports":["table"]}')
        url = post.call_args.args[0]
        kwargs = post.call_args.kwargs
        self.assertEqual(url, "http://qwen.local:8000/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(kwargs["json"]["model"], "qwen3-vl-local")
        self.assertEqual(kwargs["json"]["max_tokens"], 1234)
        self.assertEqual(kwargs["json"]["seed"], 77)
        self.assertEqual(kwargs["json"]["response_format"], {"type": "json_object"})
        self.assertEqual(kwargs["json"]["messages"][0]["role"], "system")
        self.assertEqual(kwargs["json"]["messages"][1]["content"], "goal priors")

    def test_vision_response_attaches_numpy_image_as_data_url(self):
        module = _load_qwen_request(
            {
                "QWEN_BASE_URL": "http://qwen.local:8000/v1/",
                "QWEN_API_KEY": "EMPTY",
                "QWEN_VISION_MODEL": "qwen-vl",
            }
        )

        with patch.object(module.requests, "post") as post:
            post.return_value = _FakeResponse(
                {"choices": [{"message": {"content": '{"Reason":"go","Angle":0,"Flag":false}'}}]}
            )

            image = np.zeros((4, 6, 3), dtype=np.uint8)
            result = module.vision_response("choose direction", image, system_prompt="return JSON")

        self.assertEqual(result, '{"Reason":"go","Angle":0,"Flag":false}')
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "qwen-vl")
        content = payload["messages"][1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "choose direction"})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_vlm_planner_selects_qwen_backend(self):
        source = Path(__file__).resolve().parents[1] / "vlm_planner.py"
        text = source.read_text(encoding="utf-8")

        self.assertIn('LLM_BACKEND == "qwen"', text)
        self.assertIn("llm_utils.qwen_request", text)


if __name__ == "__main__":
    unittest.main()
