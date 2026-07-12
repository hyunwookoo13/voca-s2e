import csv
import json
import tempfile
import unittest
from pathlib import Path

from pre_memory_readiness import summarize_output_dir, write_pre_memory_readiness


class PreMemoryReadinessTests(unittest.TestCase):
    def _write_episode(
        self,
        root: Path,
        *,
        episode: int = 0,
        include_videos: bool = True,
        include_classification: bool = True,
        include_pitch_reset: bool = True,
        residual_pitch_offset: int = 0,
        include_followup_call: bool = False,
    ) -> None:
        traj = root / "trajectory_{}".format(episode)
        (traj / "memory").mkdir(parents=True)
        if include_videos:
            (traj / "fps.mp4").write_bytes(b"video")
            (traj / "metric.mp4").write_bytes(b"video")

        go_execution = {
            "policy_actions": [5, 5, 1, 2, 1],
            "policy_action_names": ["look_down", "look_down", "move_forward", "turn_left", "move_forward"],
            "policy_steps": 3,
            "collision_count": 1,
            "stop_action_seen": False,
            "max_go_steps": 8,
            "go_progress": {"no_progress": True, "distance_delta_m": -0.12},
        }
        if include_classification:
            go_execution["failure_classification"] = {
                "primary": "collision_blocked",
                "labels": ["no_progress", "collision_blocked", "moved_away_from_goal"],
                "needs_replan": True,
                "message": "collision blocked progress",
            }
        if include_pitch_reset:
            go_execution["pitch_reset"] = {
                "policy_pitch_offset": -2,
                "pitch_reset_actions": [4, 4],
                "pitch_reset_action_names": ["look_up", "look_up"],
                "pitch_reset_truncated": False,
                "residual_pitch_offset": residual_pitch_offset,
            }

        call = {
            "action": "go",
            "Angle": 0,
            "Point": [320, 360],
            "runner_go_execution": go_execution,
        }
        calls = [call]
        if include_followup_call:
            calls.append({"action": "rotate", "Angle": 0, "Point": [320, 384]})
        (traj / "qwen_calls.jsonl").write_text(
            "\n".join(json.dumps(item) for item in calls) + "\n",
            encoding="utf-8",
        )
        (traj / "memory" / "steps.json").write_text(
            json.dumps({"steps": [{"runtime": {"go_execution": go_execution}}]}),
            encoding="utf-8",
        )

    def _write_metrics(
        self,
        root: Path,
        *,
        benchmark_valid: int = 1,
        benchmark_invalid_reason: str = "",
    ) -> Path:
        path = root / "objnav_hm3d.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "episode",
                    "success",
                    "final_distance_to_goal",
                    "llm_error_calls",
                    "qwen_point_calls",
                    "qwen_go_actions",
                    "qwen_go_no_progress",
                    "priors_success_count",
                    "priors_parse_fail_count",
                    "priors_fallback_count",
                    "priors_context_item_count",
                    "priors_last_error",
                    "vlm_json_fallback_count",
                    "vlm_json_last_error",
                    "qwen_point_verification_calls",
                    "qwen_point_verification_passes",
                    "qwen_point_verification_rejections",
                    "qwen_point_verification_errors",
                    "benchmark_valid",
                    "benchmark_invalid_reason",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "episode": 0,
                    "success": 0,
                    "final_distance_to_goal": 3.4,
                    "llm_error_calls": 0,
                    "qwen_point_calls": 1,
                    "qwen_go_actions": 1,
                    "qwen_go_no_progress": 1,
                    "priors_success_count": 1,
                    "priors_parse_fail_count": 0,
                    "priors_fallback_count": 0,
                    "priors_context_item_count": 3,
                    "priors_last_error": "",
                    "vlm_json_fallback_count": 0,
                    "vlm_json_last_error": "",
                    "qwen_point_verification_calls": 1,
                    "qwen_point_verification_passes": 0,
                    "qwen_point_verification_rejections": 1,
                    "qwen_point_verification_errors": 0,
                    "benchmark_valid": benchmark_valid,
                    "benchmark_invalid_reason": benchmark_invalid_reason,
                }
            )
        return path

    def _write_metrics_with_priors(
        self,
        root: Path,
        *,
        priors_success_count: int,
        priors_parse_fail_count: int,
        priors_fallback_count: int,
        priors_context_item_count: int,
        priors_last_error: str = "",
        vlm_json_fallback_count: int = 0,
        vlm_json_last_error: str = "",
    ) -> Path:
        path = root / "objnav_hm3d.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "episode",
                    "success",
                    "final_distance_to_goal",
                    "llm_error_calls",
                    "qwen_point_calls",
                    "qwen_go_actions",
                    "qwen_go_no_progress",
                    "priors_success_count",
                    "priors_parse_fail_count",
                    "priors_fallback_count",
                    "priors_context_item_count",
                    "priors_last_error",
                    "vlm_json_fallback_count",
                    "vlm_json_last_error",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "episode": 0,
                    "success": 0,
                    "final_distance_to_goal": 3.4,
                    "llm_error_calls": 0,
                    "qwen_point_calls": 1,
                    "qwen_go_actions": 1,
                    "qwen_go_no_progress": 1,
                    "priors_success_count": priors_success_count,
                    "priors_parse_fail_count": priors_parse_fail_count,
                    "priors_fallback_count": priors_fallback_count,
                    "priors_context_item_count": priors_context_item_count,
                    "priors_last_error": priors_last_error,
                    "vlm_json_fallback_count": vlm_json_fallback_count,
                    "vlm_json_last_error": vlm_json_last_error,
                }
            )
        return path

    def test_summarize_output_dir_marks_instrumented_episode_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_episode(root)
            metrics = self._write_metrics(root)

            summary = summarize_output_dir(root, metrics)

            self.assertEqual(summary["readiness_status"], "ready")
            self.assertEqual(summary["totals"]["episode_count"], 1)
            self.assertEqual(summary["totals"]["go_execution_count"], 1)
            self.assertEqual(summary["totals"]["failure_class_counts"]["collision_blocked"], 1)
            self.assertEqual(summary["totals"]["pitch_reset_count"], 1)
            self.assertEqual(summary["totals"]["unneutralized_pitch_count"], 0)
            self.assertEqual(summary["totals"]["priors_context_item_count"], 3)
            self.assertEqual(summary["totals"]["vlm_json_fallback_count"], 0)
            self.assertEqual(summary["totals"]["point_verification_calls"], 1)
            self.assertEqual(summary["totals"]["point_verification_rejections"], 1)
            self.assertEqual(summary["episodes"][0]["readiness_status"], "ready")
            self.assertEqual(summary["episodes"][0]["steps_go_execution_count"], 1)
            self.assertTrue(summary["episodes"][0]["videos"]["fps_mp4"].endswith("fps.mp4"))

    def test_summarize_output_dir_reports_priors_fallback_without_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_episode(root)
            metrics = self._write_metrics_with_priors(
                root,
                priors_success_count=0,
                priors_parse_fail_count=2,
                priors_fallback_count=1,
                priors_context_item_count=3,
                priors_last_error="priors_parse_failed",
            )

            summary = summarize_output_dir(root, metrics)

            self.assertEqual(summary["readiness_status"], "ready")
            self.assertEqual(summary["totals"]["priors_parse_fail_count"], 2)
            self.assertEqual(summary["totals"]["priors_fallback_count"], 1)
            self.assertEqual(summary["episodes"][0]["priors_last_error"], "priors_parse_failed")

    def test_summarize_output_dir_reports_vlm_json_fallback_without_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_episode(root)
            metrics = self._write_metrics_with_priors(
                root,
                priors_success_count=1,
                priors_parse_fail_count=0,
                priors_fallback_count=0,
                priors_context_item_count=3,
                vlm_json_fallback_count=2,
                vlm_json_last_error="no JSON object found",
            )

            summary = summarize_output_dir(root, metrics)

            self.assertEqual(summary["readiness_status"], "ready")
            self.assertEqual(summary["totals"]["vlm_json_fallback_count"], 2)
            self.assertEqual(summary["episodes"][0]["vlm_json_last_error"], "no JSON object found")

    def test_summarize_output_dir_marks_empty_priors_context_as_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_episode(root)
            metrics = self._write_metrics_with_priors(
                root,
                priors_success_count=0,
                priors_parse_fail_count=2,
                priors_fallback_count=0,
                priors_context_item_count=0,
                priors_last_error="priors_parse_failed",
            )

            summary = summarize_output_dir(root, metrics)

            self.assertEqual(summary["readiness_status"], "needs_attention")
            self.assertIn("empty_target_context_priors", summary["episodes"][0]["readiness_blockers"])

    def test_summarize_output_dir_marks_missing_pitch_reset_as_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_episode(root, include_pitch_reset=False, include_followup_call=True)
            metrics = self._write_metrics(root)

            summary = summarize_output_dir(root, metrics)

            self.assertEqual(summary["readiness_status"], "needs_attention")
            self.assertIn("missing_pitch_reset", summary["episodes"][0]["readiness_blockers"])

    def test_summarize_output_dir_marks_residual_pitch_as_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_episode(root, residual_pitch_offset=-1, include_followup_call=True)
            metrics = self._write_metrics(root)

            summary = summarize_output_dir(root, metrics)

            self.assertEqual(summary["readiness_status"], "needs_attention")
            self.assertIn("unneutralized_pitch_offset", summary["episodes"][0]["readiness_blockers"])

    def test_summarize_output_dir_allows_terminal_residual_pitch_without_followup_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_episode(root, residual_pitch_offset=-1)
            metrics = self._write_metrics(root)

            summary = summarize_output_dir(root, metrics)

            self.assertEqual(summary["readiness_status"], "ready")
            self.assertEqual(summary["totals"]["terminal_unneutralized_pitch_count"], 1)

    def test_summarize_output_dir_marks_missing_video_as_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_episode(root, include_videos=False)
            metrics = self._write_metrics(root)

            summary = summarize_output_dir(root, metrics)

            self.assertEqual(summary["readiness_status"], "needs_attention")
            self.assertIn("missing_fps_mp4", summary["episodes"][0]["readiness_blockers"])

    def test_summarize_output_dir_marks_missing_memory_verdict_as_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_episode(root)
            metrics = self._write_metrics(root)
            calls_path = root / "trajectory_0" / "qwen_calls.jsonl"
            call = json.loads(calls_path.read_text(encoding="utf-8").strip())
            call["vlm_input"] = {
                "memory": {
                    "candidate_refs": {
                        "revisits": [{"candidate_ref": "revisit_001"}]
                    }
                }
            }
            call["raw_vlm_output"] = {"action": "go"}
            call["memory_contract_validation"] = {
                "required": True,
                "passed": False,
                "backend_inserted_defer": True,
            }
            calls_path.write_text(json.dumps(call) + "\n", encoding="utf-8")

            summary = summarize_output_dir(root, metrics)

            episode = summary["episodes"][0]
            self.assertEqual(summary["readiness_status"], "needs_attention")
            self.assertIn("missing_explicit_memory_verdict", episode["readiness_blockers"])
            self.assertEqual(episode["memory_verdict_required_calls"], 1)
            self.assertEqual(episode["memory_verdict_missing_calls"], 1)
            self.assertEqual(episode["memory_verdict_backend_deferred_calls"], 1)
            self.assertEqual(summary["totals"]["memory_verdict_explicit_rate"], 0.0)

    def test_summarize_output_dir_rejects_infrastructure_invalid_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_episode(root)
            metrics = self._write_metrics(
                root,
                benchmark_valid=0,
                benchmark_invalid_reason="vlm_backend_unavailable",
            )

            summary = summarize_output_dir(root, metrics)

            episode = summary["episodes"][0]
            self.assertEqual(summary["readiness_status"], "needs_attention")
            self.assertIn("benchmark_invalid_run", episode["readiness_blockers"])
            self.assertEqual(
                episode["benchmark_invalid_reason"],
                "vlm_backend_unavailable",
            )

    def test_write_pre_memory_readiness_writes_json_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_episode(root)
            metrics = self._write_metrics(root)

            paths = write_pre_memory_readiness(root, metrics)

            self.assertTrue(Path(paths["json"]).is_file())
            self.assertTrue(Path(paths["csv"]).is_file())
            data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(data["readiness_status"], "ready")


if __name__ == "__main__":
    unittest.main()
