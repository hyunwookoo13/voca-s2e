import unittest
from pathlib import Path


class ObjNavQwenPointWiringTests(unittest.TestCase):
    def setUp(self):
        self.source = (Path(__file__).resolve().parents[1] / "objnav_benchmark.py").read_text(encoding="utf-8")

    def test_runner_exposes_planner_mode(self):
        self.assertIn("--planner", self.source)
        self.assertIn("qwen_point", self.source)
        self.assertIn("qwen_vlm", self.source)
        self.assertIn("voca_yoloe", self.source)

    def test_qwen_point_mode_can_skip_yoloe_load(self):
        self.assertIn("QwenPointPlanner", self.source)
        self.assertIn('args.planner == "qwen_point"', self.source)
        self.assertIn("initialize_yoloe_model", self.source)

    def test_qwen_vlm_mode_uses_voca_s2e_direct_client_without_yoloe(self):
        self.assertIn("QwenVLMPlanner", self.source)
        self.assertIn('args.planner == "qwen_vlm"', self.source)
        self.assertIn("QWEN_PLANNERS", self.source)
        self.assertIn("run_qwen_vlm_action_episode", self.source)

    def test_qwen_vlm_mode_uses_dedicated_action_loop_before_legacy_sweep(self):
        branch = (
            'if args.planner == "qwen_vlm":\n'
            "        episode_metrics = run_qwen_vlm_action_episode("
        )
        self.assertIn(branch, self.source)
        self.assertIn("continue", self.source)

    def test_runner_writes_qwen_calls_and_debug_fps_frames(self):
        self.assertIn("qwen_calls.jsonl", self.source)
        self.assertIn("set_qwen_call_log_path", self.source)
        self.assertIn("save_qwen_calls", self.source)
        self.assertIn("render_qwen_debug_frame", self.source)

    def test_qwen_vlm_mode_writes_pre_memory_readiness_summary(self):
        self.assertIn("write_pre_memory_readiness", self.source)
        branch = (
            'if args.planner == "qwen_vlm":\n'
            "        episode_metrics = run_qwen_vlm_action_episode("
        )
        self.assertIn(branch, self.source)
        self.assertIn("write_pre_memory_readiness(args.output_dir, args.metrics_path)", self.source)

    def test_runner_has_optional_max_episode_steps_for_smoke_runs(self):
        self.assertIn("--max_episode_steps", self.source)
        self.assertIn("truncated_by_max_steps", self.source)

    def test_runner_wires_memory_free_nav_audit(self):
        self.assertIn("NavigationAuditLogger", self.source)
        self.assertIn("audit_logger.record_decision", self.source)
        self.assertIn("audit_logger.finalize_pending", self.source)
        self.assertIn("audit_paths", self.source)
        self.assertIn("'steps_json'", self.source)
        self.assertIn("'memory_graph_json'", self.source)

    def test_rotation_replan_loops_append_rgb_for_both_turn_directions(self):
        common_append_after_turn = (
            "else:\n"
            "                        obs = habitat_env.step(2)\n"
            "                    episode_images.append(obs['rgb'])"
        )
        self.assertGreaterEqual(self.source.count(common_append_after_turn), 2)


if __name__ == "__main__":
    unittest.main()
