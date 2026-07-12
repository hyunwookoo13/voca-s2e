import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_qwen_models import (
    normalize_api_base,
    parse_case_spec,
    parse_endpoint_spec,
    summarize_records,
)


class BenchmarkQwenModelsTests(unittest.TestCase):
    def test_endpoint_and_case_specs_are_explicit_and_stable(self):
        self.assertEqual(
            parse_endpoint_spec("local=http://localhost:8000/"),
            {"label": "local", "base_url": "http://localhost:8000"},
        )
        self.assertEqual(normalize_api_base("http://host:8000"), "http://host:8000/v1")
        self.assertEqual(normalize_api_base("http://host:8000/v1/"), "http://host:8000/v1")
        self.assertEqual(
            parse_case_spec("doorway=/tmp/qwen_calls.jsonl:2"),
            {"name": "doorway", "path": "/tmp/qwen_calls.jsonl", "index": 2},
        )

    def test_summary_reports_latency_tokens_and_navigation_quality(self):
        records = [
            {
                "endpoint": "server01",
                "model": "qwen-instruct",
                "case": "doorway",
                "warmup": False,
                "latency_sec": 2.0,
                "json_ok": True,
                "schema_ok": True,
                "candidate_gate_passed": True,
                "pipeline_accepted": True,
                "risky_selected_point": False,
                "completion_tokens": 100,
            },
            {
                "endpoint": "server01",
                "model": "qwen-instruct",
                "case": "doorway",
                "warmup": False,
                "latency_sec": 4.0,
                "json_ok": False,
                "schema_ok": False,
                "candidate_gate_passed": False,
                "pipeline_accepted": False,
                "risky_selected_point": True,
                "completion_tokens": 300,
            },
            {
                "endpoint": "server01",
                "model": "qwen-instruct",
                "case": "doorway",
                "warmup": True,
                "latency_sec": 99.0,
                "json_ok": True,
            },
        ]

        summary = summarize_records(records)

        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row["measured_calls"], 2)
        self.assertEqual(row["latency_mean_sec"], 3.0)
        self.assertEqual(row["latency_median_sec"], 3.0)
        self.assertEqual(row["json_success_rate"], 0.5)
        self.assertEqual(row["pipeline_accept_rate"], 0.5)
        self.assertEqual(row["risky_point_rate"], 0.5)
        self.assertEqual(row["completion_tokens_mean"], 200.0)


if __name__ == "__main__":
    unittest.main()
