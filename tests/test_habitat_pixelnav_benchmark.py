import json
import tempfile
import unittest
from pathlib import Path

from goal_adapter.habitat_pixelnav_benchmark import (
    compute_episode_benchmark_record,
    run_habitat_pixelnav_benchmark_from_audits,
    run_habitat_pixelnav_benchmark_from_summaries,
)


class HabitatPixelNavBenchmarkTest(unittest.TestCase):
    def test_compute_episode_benchmark_record_extracts_sr_spl_and_memory_metrics(self):
        trial = _trial_summary(
            start=[0.0, 0.0, 0.0],
            goal=[2.0, 0.0],
            rollout_positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.1, 0.0, 0.0]],
            moved_distance=2.1,
            outcome_success=True,
            memory_nodes=3,
            memory_edges=2,
        )

        record = compute_episode_benchmark_record(
            trial,
            episode_id="episode_a",
            success_distance_m=0.25,
            min_progress_m=0.1,
        )

        self.assertEqual(record["episode_id"], "episode_a")
        self.assertTrue(record["local_execution_success"])
        self.assertTrue(record["pointnav_success"])
        self.assertTrue(record["progress_success"])
        self.assertAlmostEqual(record["initial_distance_to_goal_m"], 2.0)
        self.assertAlmostEqual(record["final_distance_to_goal_m"], 0.1)
        self.assertAlmostEqual(record["goal_progress_m"], 1.9)
        self.assertAlmostEqual(record["path_length_m"], 2.1)
        self.assertAlmostEqual(record["shortest_path_m"], 2.0)
        self.assertAlmostEqual(record["spl"], 2.0 / 2.1)
        self.assertEqual(record["memory"]["num_nodes"], 3)
        self.assertEqual(record["memory"]["num_edges"], 2)
        self.assertTrue(record["memory"]["updated"])

    def test_compute_episode_benchmark_record_keeps_spl_zero_when_goal_not_reached(self):
        trial = _trial_summary(
            start=[0.0, 0.0, 0.0],
            goal=[10.0, 0.0],
            rollout_positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            moved_distance=2.0,
            outcome_success=True,
            memory_nodes=2,
            memory_edges=1,
        )

        record = compute_episode_benchmark_record(
            trial,
            episode_id="episode_b",
            success_distance_m=0.5,
            min_progress_m=0.1,
        )

        self.assertTrue(record["local_execution_success"])
        self.assertFalse(record["pointnav_success"])
        self.assertTrue(record["progress_success"])
        self.assertEqual(record["spl"], 0.0)
        self.assertAlmostEqual(record["progress_efficiency"], 2.0 / 10.0)

    def test_compute_episode_benchmark_record_uses_all_closed_loop_steps(self):
        trial = {
            "status": "completed",
            "config": {
                "scene_path": "scene.glb",
                "start_position_xyz": [0.0, 0.0, 0.0],
                "goal_map_xy": [3.0, 0.0],
            },
            "steps": [
                {
                    "action": "go",
                    "outcome": {
                        "success": True,
                        "collision": False,
                        "no_progress": False,
                        "moved_distance_m": 1.0,
                        "raw": {
                            "rollout": {
                                "steps": [
                                    {
                                        "position_after_xyz": [1.0, 0.0, 0.0],
                                    }
                                ]
                            }
                        },
                    },
                },
                {
                    "action": "go",
                    "outcome": {
                        "success": True,
                        "collision": False,
                        "no_progress": False,
                        "moved_distance_m": 2.0,
                        "raw": {
                            "rollout": {
                                "steps": [
                                    {
                                        "position_after_xyz": [3.0, 0.0, 0.0],
                                    }
                                ]
                            }
                        },
                    },
                },
            ],
            "memory": {
                "schema_version": "relative_topometric_memory_graph_v5",
                "num_nodes": 3,
                "num_edges": 2,
                "temporal_edges": [{"status": "success"}, {"status": "success"}],
                "deadlock_state": {"status": "none"},
            },
        }

        record = compute_episode_benchmark_record(
            trial,
            episode_id="multi_step",
            success_distance_m=0.25,
            min_progress_m=0.1,
        )

        self.assertTrue(record["pointnav_success"])
        self.assertAlmostEqual(record["path_length_m"], 3.0)
        self.assertAlmostEqual(record["final_distance_to_goal_m"], 0.0)
        self.assertAlmostEqual(record["spl"], 1.0)

    def test_run_benchmark_from_summaries_writes_json_markdown_and_memory_rollup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trial_a = root / "trial_a.json"
            trial_b = root / "trial_b.json"
            trial_a.write_text(
                json.dumps(
                    _trial_summary(
                        start=[0.0, 0.0, 0.0],
                        goal=[2.0, 0.0],
                        rollout_positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                        moved_distance=2.0,
                        outcome_success=True,
                        memory_nodes=2,
                        memory_edges=1,
                    )
                ),
                encoding="utf-8",
            )
            trial_b.write_text(
                json.dumps(
                    _trial_summary(
                        start=[0.0, 0.0, 0.0],
                        goal=[4.0, 0.0],
                        rollout_positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        moved_distance=1.0,
                        outcome_success=True,
                        memory_nodes=2,
                        memory_edges=1,
                    )
                ),
                encoding="utf-8",
            )

            result = run_habitat_pixelnav_benchmark_from_summaries(
                [trial_a, trial_b],
                output_dir=root / "benchmark",
                success_distance_m=0.25,
                min_progress_m=0.1,
            )
            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
            markdown = Path(result["docmost_markdown"]).read_text(encoding="utf-8")

        self.assertEqual(summary["total_episodes"], 2)
        self.assertEqual(summary["metrics"]["local_execution_sr"], 1.0)
        self.assertEqual(summary["metrics"]["pointnav_sr"], 0.5)
        self.assertEqual(summary["metrics"]["progress_sr"], 1.0)
        self.assertAlmostEqual(summary["metrics"]["spl"], 0.5)
        self.assertEqual(summary["metrics"]["memory_update_rate"], 1.0)
        self.assertEqual(summary["metrics"]["mean_memory_nodes"], 2.0)
        self.assertIn("Habitat PixelNav Benchmark", markdown)
        self.assertIn("pointnav_sr", markdown)

    def test_run_benchmark_from_audits_runs_batch_then_computes_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audit = root / "audit.json"
            audit.write_text(json.dumps({"scene_path": "scene.glb"}), encoding="utf-8")

            def fake_run_batch(audit_paths, output_dir, force_front_view_waypoint):
                self.assertEqual([Path(path) for path in audit_paths], [audit])
                self.assertTrue(force_front_view_waypoint)
                batch_dir = Path(output_dir)
                batch_dir.mkdir(parents=True, exist_ok=True)
                trial_path = batch_dir / "trial_000" / "real_vlm_tiny_e2e.json"
                trial_path.parent.mkdir(parents=True, exist_ok=True)
                trial_path.write_text(
                    json.dumps(
                        _trial_summary(
                            start=[0.0, 0.0, 0.0],
                            goal=[1.0, 0.0],
                            rollout_positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                            moved_distance=1.0,
                            outcome_success=True,
                            memory_nodes=2,
                            memory_edges=1,
                        )
                    ),
                    encoding="utf-8",
                )
                summary_path = batch_dir / "real_vlm_tiny_e2e_batch_summary.json"
                summary_path.write_text(
                    json.dumps({"records": [{"summary_json": str(trial_path)}]}),
                    encoding="utf-8",
                )
                return {"summary_json": str(summary_path), "records": [{"summary_json": str(trial_path)}]}

            result = run_habitat_pixelnav_benchmark_from_audits(
                [audit],
                output_dir=root / "benchmark",
                force_front_view_waypoint=True,
                run_batch=fake_run_batch,
                success_distance_m=0.25,
                min_progress_m=0.1,
            )
            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
            batch_summary_exists = Path(summary["source"]["batch_summary_json"]).exists()

        self.assertEqual(summary["source"]["mode"], "audit_real_vlm_batch")
        self.assertEqual(summary["total_episodes"], 1)
        self.assertEqual(summary["metrics"]["pointnav_sr"], 1.0)
        self.assertTrue(batch_summary_exists)


def _trial_summary(
    *,
    start,
    goal,
    rollout_positions,
    moved_distance,
    outcome_success,
    memory_nodes,
    memory_edges,
):
    steps = []
    for index, position in enumerate(rollout_positions[:-1]):
        steps.append(
            {
                "step_index": index,
                "position_before_xyz": position,
                "position_after_xyz": rollout_positions[index + 1],
                "moved_distance_m": 0.0,
            }
        )
    return {
        "status": "passed" if outcome_success else "failed",
        "tiny_e2e": {"passed": bool(outcome_success)},
        "config": {
            "scene_path": "scene.glb",
            "start_position_xyz": start,
            "goal_map_xy": goal,
        },
        "execution": {
            "action_outcome": {
                "success": bool(outcome_success),
                "collision": False,
                "no_progress": False,
                "moved_distance_m": moved_distance,
                "raw": {
                    "rollout": {
                        "steps": steps,
                        "total_moved_distance_m": moved_distance,
                        "collision_count": 0,
                    },
                    "final_position_delta_m": moved_distance,
                },
            }
        },
        "memory_update": {"updated": bool(outcome_success)},
        "memory": {
            "schema_version": "relative_topometric_memory_graph_v5",
            "num_nodes": memory_nodes,
            "num_edges": memory_edges,
            "temporal_edges": [
                {"status": "success"}
                for _ in range(memory_edges)
            ],
            "deadlock_state": {"status": "none"},
        },
    }


if __name__ == "__main__":
    unittest.main()
