import unittest
from unittest.mock import patch

from nav_memory_qwen.vlm_client import OpenAICompatibleVLMClient


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class OpenAICompatibleVLMClientTest(unittest.TestCase):
    def test_decide_extracts_json_from_reasoning_content_when_content_is_empty(self):
        response = FakeResponse({
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning_content": '{"schema_version":"nav_vlm_waypoint_v1","action":"stop"}',
                }
            }]
        })
        client = OpenAICompatibleVLMClient(base_url="http://qwen.test/v1", api_key="test-key")

        with patch("nav_memory_qwen.vlm_client.requests.post", return_value=response):
            output = client.decide({"observation": {"views": []}})

        self.assertEqual(output["schema_version"], "nav_vlm_waypoint_v1")
        self.assertEqual(output["action"], "stop")

    def test_decide_retries_once_when_first_response_has_no_json(self):
        responses = [
            FakeResponse({"choices": [{"message": {"content": "token budget exhausted before final JSON"}}]}),
            FakeResponse({"choices": [{"message": {"content": '{"schema_version":"nav_vlm_waypoint_v1","action":"rotate"}'}}]}),
        ]
        client = OpenAICompatibleVLMClient(
            base_url="http://qwen.test/v1",
            api_key="test-key",
            max_json_retries=1,
        )

        with patch("nav_memory_qwen.vlm_client.requests.post", side_effect=responses) as post:
            output = client.decide({"observation": {"views": []}})

        self.assertEqual(output["action"], "rotate")
        self.assertEqual(post.call_count, 2)
        retry_payload = post.call_args_list[1].kwargs["json"]
        retry_text_parts = [
            part["text"]
            for part in retry_payload["messages"][-1]["content"]
            if part.get("type") == "text"
        ]
        self.assertTrue(any("Return only one valid JSON object" in text for text in retry_text_parts))

    def test_from_env_reads_runtime_timeout_and_image_options(self):
        with patch.dict(
            "os.environ",
            {
                "QWEN_BASE_URL": "http://qwen.test/v1/",
                "QWEN_API_KEY": "test-key",
                "QWEN_TIMEOUT_S": "300",
                "QWEN_IMAGE_MAX_SIDE": "768",
                "QWEN_JPEG_QUALITY": "80",
            },
            clear=True,
        ):
            client = OpenAICompatibleVLMClient.from_env()

        self.assertEqual(client.base_url, "http://qwen.test/v1")
        self.assertEqual(client.timeout_s, 300.0)
        self.assertEqual(client.image_max_side, 768)
        self.assertEqual(client.jpeg_quality, 80)

    def test_from_env_reads_extra_payload_json(self):
        with patch.dict(
            "os.environ",
            {
                "QWEN_BASE_URL": "http://qwen.test/v1",
                "QWEN_API_KEY": "test-key",
                "QWEN_EXTRA_PAYLOAD_JSON": '{"response_format":{"type":"json_object"}}',
            },
            clear=True,
        ):
            client = OpenAICompatibleVLMClient.from_env()

        self.assertEqual(client.extra_payload, {"response_format": {"type": "json_object"}})


if __name__ == "__main__":
    unittest.main()
