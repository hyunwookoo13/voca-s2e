import unittest
from pathlib import Path


class VocaEnvTests(unittest.TestCase):
    def test_default_qwen_payload_requests_json_and_disables_thinking(self):
        source = (Path(__file__).resolve().parents[1] / "scripts" / "voca_env.sh").read_text(encoding="utf-8")

        self.assertIn("QWEN_EXTRA_PAYLOAD_JSON", source)
        self.assertIn("response_format", source)
        self.assertIn("chat_template_kwargs", source)
        self.assertIn("enable_thinking", source)
        self.assertIn("false", source)

    def test_default_model_is_locked_to_32b_thinking_awq(self):
        source = (Path(__file__).resolve().parents[1] / "scripts" / "voca_env.sh").read_text(encoding="utf-8")

        self.assertIn("http://localhost:8000/v1", source)
        self.assertIn("qwen3-vl-32b-thinking-awq", source)
        self.assertIn("QuantTrio/Qwen3-VL-32B-Thinking-AWQ", source)


if __name__ == "__main__":
    unittest.main()
