import csv
import json
import tempfile
import unittest
from pathlib import Path

from navigation_ablation import summarize_run, write_summary


class NavigationAblationTests(unittest.TestCase):
    def _write_run(self, root: Path) -> Path:
        run_dir = root / "run_a"
        run_dir.mkdir()
        rows = [
            {
                "episode": 0,
                "success": 1,
                "spl": 0.8,
                "start_distance_to_goal": 5.0,
                "final_distance_to_goal": 0.5,
                "episode_time_sec": 10,
                "num_steps": 20,
                "llm_calls": 2,
                "llm_avg_time_sec": 4.0,
                "priors_calls": 1,
                "priors_avg_time_sec": 10.0,
                "qwen_navigation_vlm_calls": 2,
                "qwen_navigation_vlm_avg_time_sec": 1.5,
                "qwen_go_actions": 4,
                "qwen_go_no_progress": 1,
                "qwen_stop_actions": 1,
                "qwen_raw_go_decisions": 3,
                "qwen_pixel_candidate_gate_rejections": 1,
                "qwen_point_verification_calls": 2,
                "qwen_point_verification_rejections": 1,
                "memory_ops_requested": 2,
                "memory_ops_accepted": 1,
                "memory_place_nodes": 3,
                "memory_embedding_backend": "dinov2",
            },
            {
                "episode": 1,
                "success": 0,
                "spl": 0.0,
                "start_distance_to_goal": 4.0,
                "final_distance_to_goal": 3.0,
                "episode_time_sec": 20,
                "num_steps": 30,
                "llm_calls": 1,
                "llm_avg_time_sec": 10.0,
                "priors_calls": 1,
                "priors_avg_time_sec": 20.0,
                "qwen_navigation_vlm_calls": 1,
                "qwen_navigation_vlm_avg_time_sec": 3.0,
                "qwen_go_actions": 2,
                "qwen_go_no_progress": 0,
                "qwen_stop_actions": 1,
                "qwen_raw_go_decisions": 1,
                "qwen_pixel_candidate_gate_rejections": 0,
                "qwen_point_verification_calls": 0,
                "qwen_point_verification_rejections": 0,
                "memory_ops_requested": 0,
                "memory_ops_accepted": 0,
                "memory_place_nodes": 5,
                "memory_embedding_backend": "dinov2",
            },
        ]
        csv_path = run_dir / "objnav_hm3d_run_a.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        manifest_dir = run_dir / "trajectory_0"
        manifest_dir.mkdir()
        (manifest_dir / "benchmark_manifest.json").write_text(
            json.dumps(
                {
                    "qwen_model": "test-model",
                    "qwen_model_root": "test-org/test-model",
                    "qwen_base_url": "http://server/v1",
                    "memory_schema": "nav_memory_context_v6",
                }
            ),
            encoding="utf-8",
        )
        return run_dir

    def test_summarize_run_computes_navigation_and_memory_rates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._write_run(Path(tmpdir))
            summary = summarize_run("full-memory", run_dir)

        self.assertEqual(summary["episode_count"], 2)
        self.assertAlmostEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["stop_action_episode_count"], 2)
        self.assertEqual(summary["false_positive_stop_episode_count"], 1)
        self.assertAlmostEqual(summary["stop_episode_precision"], 0.5)
        self.assertAlmostEqual(summary["spl_mean"], 0.4)
        self.assertAlmostEqual(summary["distance_reduction_mean_m"], 2.75)
        self.assertAlmostEqual(summary["llm_decision_time_weighted_mean_sec"], 2.0)
        self.assertAlmostEqual(summary["priors_time_weighted_mean_sec"], 15.0)
        self.assertAlmostEqual(summary["go_no_progress_rate"], 1.0 / 6.0)
        self.assertAlmostEqual(summary["pixel_candidate_gate_rejection_rate"], 0.25)
        self.assertAlmostEqual(summary["memory_op_acceptance_rate"], 0.5)
        self.assertAlmostEqual(summary["memory_place_nodes_mean"], 4.0)
        self.assertEqual(summary["qwen_model"], "test-model")
        self.assertEqual(summary["qwen_model_root"], "test-org/test-model")

    def test_write_summary_creates_csv_and_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = self._write_run(root)
            paths = write_summary([("full-memory", run_dir)], root / "summary")
            payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            with Path(paths["csv"]).open(encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(payload["schema_version"], "voca_navigation_ablation_v2")
        self.assertEqual(payload["run_count"], 1)
        self.assertEqual(csv_rows[0]["label"], "full-memory")

    def test_summarize_run_excludes_infrastructure_invalid_episode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._write_run(Path(tmpdir))
            csv_path = run_dir / "objnav_hm3d_run_a.csv"
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["benchmark_valid"] = "1"
            rows[0]["benchmark_invalid_reason"] = ""
            rows[1]["benchmark_valid"] = "0"
            rows[1]["benchmark_invalid_reason"] = "vlm_backend_unavailable"
            fieldnames = list(rows[0].keys())
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            summary = summarize_run("full-memory", run_dir)

        self.assertEqual(summary["episode_count"], 2)
        self.assertEqual(summary["valid_episode_count"], 1)
        self.assertEqual(summary["invalid_episode_count"], 1)
        self.assertEqual(summary["benchmark_complete"], 0)
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertAlmostEqual(summary["spl_mean"], 0.8)
        self.assertIn("vlm_backend_unavailable", summary["invalid_reasons_json"])


if __name__ == "__main__":
    unittest.main()
