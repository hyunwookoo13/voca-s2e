import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from goal_adapter.habitat_pixelnav_real_vlm_final_report import (
    build_real_vlm_final_summary,
    main,
)


class HabitatPixelNavRealVLMFinalReportTest(unittest.TestCase):
    def test_final_summary_marks_passed_tiny_e2e_ready_for_docmost(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tiny_path = Path(temp_dir) / "real_vlm_tiny_e2e.json"
            tiny_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "tiny_e2e": {
                            "passed": True,
                            "endpoint_smoke_passed": True,
                            "gated_execution_executed": True,
                            "memory_updated": True,
                            "failure_stage": None,
                        },
                        "execution": {
                            "action_outcome": {
                                "success": True,
                                "moved_distance_m": 0.5,
                                "collision": False,
                            }
                        },
                        "memory": {
                            "schema_version": "relative_topometric_memory_graph_v5",
                            "num_nodes": 2,
                            "num_edges": 1,
                        },
                        "artifacts": {"output_dir": temp_dir},
                    }
                ),
                encoding="utf-8",
            )

            result = build_real_vlm_final_summary(tiny_path, output_dir=Path(temp_dir) / "out")
            markdown = Path(result["docmost_markdown"]).read_text(encoding="utf-8")

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["ready_for_docmost"], True)
        self.assertIn("Real VLM Tiny E2E 최종 요약", markdown)
        self.assertIn("passed", markdown)
        self.assertIn("relative_topometric_memory_graph_v5", markdown)
        self.assertIn("PIXELNAV_DEVICE", markdown)
        self.assertIn("QWEN_TIMEOUT_S", markdown)
        self.assertIn("QWEN_EXTRA_PAYLOAD_JSON", markdown)

    def test_final_summary_marks_skipped_endpoint_env_as_not_ready_but_actionable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tiny_path = Path(temp_dir) / "real_vlm_tiny_e2e.json"
            tiny_path.write_text(
                json.dumps(
                    {
                        "status": "skipped",
                        "tiny_e2e": {
                            "passed": False,
                            "endpoint_smoke_passed": False,
                            "gated_execution_executed": False,
                            "memory_updated": False,
                            "failure_stage": "endpoint_smoke",
                        },
                        "endpoint_smoke": {
                            "required_env": ["QWEN_BASE_URL", "QWEN_API_KEY"]
                        },
                        "memory": None,
                        "artifacts": {"output_dir": temp_dir},
                    }
                ),
                encoding="utf-8",
            )

            result = build_real_vlm_final_summary(tiny_path, output_dir=Path(temp_dir) / "out")
            markdown = Path(result["docmost_markdown"]).read_text(encoding="utf-8")

        self.assertEqual(result["status"], "skipped")
        self.assertFalse(result["ready_for_docmost"])
        self.assertEqual(result["next_action"], "set_qwen_endpoint_env_and_rerun_step_t")
        self.assertIn("QWEN_BASE_URL", markdown)
        self.assertIn("failure_stage", markdown)

    def test_final_report_cli_writes_markdown_and_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tiny_path = Path(temp_dir) / "real_vlm_tiny_e2e.json"
            out_dir = Path(temp_dir) / "out"
            tiny_path.write_text(
                json.dumps(
                    {
                        "status": "skipped",
                        "tiny_e2e": {"passed": False, "failure_stage": "endpoint_smoke"},
                        "endpoint_smoke": {"required_env": ["QWEN_BASE_URL", "QWEN_API_KEY"]},
                    }
                ),
                encoding="utf-8",
            )

            with patch("sys.stdout", new_callable=io.StringIO):
                exit_code = main(["--tiny-e2e", str(tiny_path), "--out", str(out_dir)])
            summary_exists = (out_dir / "real_vlm_final_summary.json").exists()
            markdown_exists = (out_dir / "docmost_real_vlm_final_summary.md").exists()

        self.assertEqual(exit_code, 0)
        self.assertTrue(summary_exists)
        self.assertTrue(markdown_exists)


if __name__ == "__main__":
    unittest.main()
