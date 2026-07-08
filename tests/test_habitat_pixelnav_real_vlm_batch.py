import json
import tempfile
import unittest
from pathlib import Path

from goal_adapter.habitat_pixelnav_real_vlm_batch import run_real_vlm_tiny_e2e_batch


class HabitatPixelNavRealVLMBatchTest(unittest.TestCase):
    def test_batch_aggregates_pass_fail_skip_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audits = []
            for index in range(3):
                audit = root / f"audit_{index}.json"
                audit.write_text(json.dumps({"scene_path": "scene.glb"}), encoding="utf-8")
                audits.append(audit)

            canned = [
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
                            "moved_distance_m": 2.0,
                            "collision": False,
                            "no_progress": False,
                        }
                    },
                    "memory": {"num_nodes": 2, "num_edges": 1},
                },
                {
                    "status": "failed",
                    "tiny_e2e": {
                        "passed": False,
                        "endpoint_smoke_passed": True,
                        "gated_execution_executed": False,
                        "memory_updated": False,
                        "failure_stage": "gated_execution",
                    },
                    "execution": {"action_outcome": None},
                    "memory": None,
                },
                {
                    "status": "skipped",
                    "tiny_e2e": {
                        "passed": False,
                        "endpoint_smoke_passed": False,
                        "gated_execution_executed": False,
                        "memory_updated": False,
                        "failure_stage": "endpoint_smoke",
                    },
                    "execution": {"action_outcome": None},
                    "memory": None,
                },
            ]

            call_index = {"value": 0}

            def fake_run_one(audit_path, trial_dir, force_front_view_waypoint):
                payload = dict(canned[call_index["value"]])
                call_index["value"] += 1
                Path(trial_dir).mkdir(parents=True, exist_ok=True)
                summary_path = Path(trial_dir) / "real_vlm_tiny_e2e.json"
                payload["summary_json"] = str(summary_path)
                summary_path.write_text(json.dumps(payload), encoding="utf-8")
                return payload

            result = run_real_vlm_tiny_e2e_batch(
                audits,
                output_dir=root / "batch",
                force_front_view_waypoint=True,
                run_one=fake_run_one,
            )
            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
            markdown = Path(result["docmost_markdown"]).read_text(encoding="utf-8")

        self.assertEqual(summary["total_trials"], 3)
        self.assertEqual(summary["status_counts"], {"passed": 1, "failed": 1, "skipped": 1})
        self.assertAlmostEqual(summary["pass_rate"], 1 / 3)
        self.assertEqual(summary["endpoint_smoke_passed"], 2)
        self.assertEqual(summary["gated_execution_executed"], 1)
        self.assertEqual(summary["memory_updated"], 1)
        self.assertEqual(summary["failure_stage_counts"], {"gated_execution": 1, "endpoint_smoke": 1})
        self.assertEqual(summary["mean_moved_distance_m"], 2.0)
        self.assertEqual(summary["collision_count"], 0)
        self.assertEqual(summary["no_progress_count"], 0)
        self.assertIn("Real VLM Tiny E2E Batch", markdown)
        self.assertIn("pass_rate", markdown)


if __name__ == "__main__":
    unittest.main()
