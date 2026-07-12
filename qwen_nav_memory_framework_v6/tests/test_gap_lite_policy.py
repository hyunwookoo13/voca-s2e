import tempfile
from pathlib import Path
import unittest
import json

from nav_memory_qwen.agent import NavMemoryAgent, NavAgentConfig
from nav_memory_qwen.policy import apply_gap_lite_validation
from nav_memory_qwen.robot_backend import StaticImageBackend
from nav_memory_qwen.schema import make_go_output
from nav_memory_qwen.vlm_client import BaseVLMClient, HeuristicVLMClient


class GapLitePolicyTest(unittest.TestCase):
    def _memory_context(self):
        return {
            "candidate_refs": {
                "exits": [
                    {"candidate_ref": "exit_001", "view_type_hint": "front", "avoid": False, "status": "unknown_frontier", "bearing_deg_robot": 0, "score": 0.7},
                    {"candidate_ref": "exit_bad", "view_type_hint": "left", "avoid": True, "status": "deadlock_entry", "bearing_deg_robot": -90, "score": -0.2},
                ],
                "revisits": [
                    {"candidate_ref": "revisit_ok", "node_id": "n_00001", "candidate_node_id": "n_00001", "spatial_plausibility": {"accepted": True, "reason": "within_baseline"}},
                    {"candidate_ref": "revisit_far", "node_id": "n_00009", "candidate_node_id": "n_00009", "spatial_plausibility": {"accepted": False, "reason": "spatial_baseline_too_large"}},
                ],
            },
            "current_localization": {"current_node_id": "n_current"},
            "local_topology": {"candidate_exits": []},
            "place_recognition": {"revisit_candidates": []},
        }

    def _go(self, ref):
        return make_go_output(
            view_id=0,
            view_type="front",
            point_px=(320, 360),
            width=640,
            height=480,
            decision_reason="G02_VISIBLE_FLOOR_TOWARD_GOAL",
            goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
            short_text="test go",
            selected_candidate_ref=ref,
        )

    def test_unknown_go_candidate_ref_is_recovered_with_observation(self):
        out, checkpoints, feedback, warnings = apply_gap_lite_validation(self._go("exit_missing"), self._memory_context())
        self.assertEqual(out["action"], "request_observation")
        self.assertIn("NAV_GO_002", feedback["failed_rule_ids"])
        self.assertTrue(any("unknown selected_candidate_ref" in w for w in warnings))

    def test_avoid_true_go_candidate_ref_is_blocked(self):
        out, checkpoints, feedback, warnings = apply_gap_lite_validation(self._go("exit_bad"), self._memory_context())
        self.assertEqual(out["action"], "request_observation")
        self.assertIn("NAV_GO_003", feedback["failed_rule_ids"])

    def test_valid_go_candidate_ref_passes(self):
        out, checkpoints, feedback, warnings = apply_gap_lite_validation(self._go("exit_001"), self._memory_context())
        self.assertEqual(out["action"], "go")
        self.assertTrue(feedback["valid"])
        self.assertEqual(out["selected_candidate_ref"], "exit_001")

    def test_spatially_implausible_revisit_memory_op_is_dropped(self):
        output = {"action": "request_observation", "memory_ops": [{"op": "confirm_revisit_node", "candidate_ref": "revisit_far", "confidence": 0.99}]}
        out, checkpoints, feedback, warnings = apply_gap_lite_validation(output, self._memory_context())
        self.assertEqual(out.get("memory_ops"), [])
        self.assertIn("NAV_MEM_004", feedback["failed_rule_ids"])
        self.assertTrue(any("spatial_plausibility" in w for w in warnings))

    def test_revisit_candidate_ref_resolves_to_node_id(self):
        output = {"action": "request_observation", "memory_ops": [{"op": "confirm_revisit_node", "candidate_ref": "revisit_ok", "confidence": 0.95}]}
        out, checkpoints, feedback, warnings = apply_gap_lite_validation(output, self._memory_context())
        self.assertEqual(out["memory_ops"][0]["node_id"], "n_00001")
        self.assertTrue(feedback["valid"])

    def test_agent_context_and_feedback_log_include_gap_lite(self):
        with tempfile.TemporaryDirectory() as td:
            img = StaticImageBackend.create_demo_image(Path(td) / "img.jpg")
            robot = StaticImageBackend(img, step_m=0.5)
            agent = NavMemoryAgent(robot=robot, vlm_client=HeuristicVLMClient(), config=NavAgentConfig(max_steps=1))
            result = agent.step(goal_map_xy=(2.0, 0.0), step_index=0)
            self.assertIsNotNone(result.validation_feedback)
            self.assertIn("candidate_refs", result.vlm_input["memory"])
            self.assertIn("nav_skill_cards", result.vlm_input["memory"])
            self.assertTrue(result.validation_feedback["num_checkpoints"] >= 1)
            out_dir = Path(td) / "run"
            agent.save_run(out_dir)
            self.assertTrue((out_dir / "Feedback.md").exists())
            steps = json.loads((out_dir / "steps.json").read_text(encoding="utf-8"))["steps"]
            self.assertEqual(steps[0]["vlm_input"]["memory"]["schema_version"], "nav_memory_context_v6")
            self.assertIn("candidate_refs", steps[0]["vlm_input"]["memory"])
            self.assertIn("validation_feedback", steps[0])
            text = (out_dir / "Feedback.md").read_text(encoding="utf-8")
            self.assertIn("Validation Feedback", text)


if __name__ == "__main__":
    unittest.main()
