import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qwen_point_planner import (
    QwenPointPlanner,
    QwenPointPlannerConfig,
    candidate_validation_lines,
    decision_action_label,
    decision_debug_point,
    draw_memory_graph_thumbnail,
    fallback_priors_for_target,
    qwen_priors_thinking_enabled,
    prior_candidate_shortlist_for_target,
    filter_detector_prior_classes,
    make_point_goal_mask,
    memory_visualizer_lines,
    parse_qwen_point_decision,
    render_qwen_debug_frame,
    executed_action_label,
    fallback_display_reason,
)


class QwenPointPlannerTests(unittest.TestCase):
    def test_memory_graph_thumbnail_draws_nodes_and_edges(self):
        panel = np.zeros((240, 430, 3), dtype=np.uint8)
        state = {
            "enabled": True,
            "supervisor_mode": "escape_deadlock",
            "graph_layout": {
                "nodes": [
                    {"node_id": "n_00001", "x_m": 0.0, "y_m": 0.0, "current": False},
                    {"node_id": "n_00002", "x_m": 1.0, "y_m": 0.5, "current": True},
                ],
                "edges": [
                    {
                        "edge_id": "e_00001",
                        "src_node_id": "n_00001",
                        "dst_node_id": "n_00002",
                        "status": "success",
                    }
                ],
            },
        }

        draw_memory_graph_thumbnail(panel, state, top=20)

        self.assertTrue(np.any(panel[:, :, 1] > 180))
        self.assertTrue(np.any(panel[:, :, 2] > 200))

    def test_candidate_validation_lines_show_ref_marker_and_gate_result(self):
        lines = candidate_validation_lines(
            {
                "selected_candidate_ref": "px_v00_r0_c0",
                "pixel_candidate_validation": {
                    "passed": True,
                    "reason": "valid_pixel_candidate",
                    "candidate": {
                        "candidate_ref": "px_v00_r0_c0",
                        "marker_id": "M01",
                    },
                },
                "selected_point_verification": {
                    "triggered": True,
                    "passed": False,
                    "surface": "wall",
                    "reason": "selected point is on a vertical wall",
                },
            }
        )

        self.assertEqual(lines[0], "Candidate: M01 px_v00_r0_c0")
        self.assertEqual(lines[1], "Gate: PASS valid_pixel_candidate")
        self.assertEqual(lines[2], "Visual: REJECT wall")

        rejected = candidate_validation_lines(
            {
                "pixel_candidate_validation": {
                    "passed": False,
                    "reason": "selected_candidate_avoid_true",
                    "requested_ref": "px_v00_r1_c1",
                }
            }
        )
        self.assertEqual(rejected[0], "Candidate: px_v00_r1_c1")
        self.assertEqual(rejected[1], "Gate: REJECT selected_candidate_avoid_true")

    def test_parse_qwen_point_decision_reads_angle_point_and_metadata(self):
        parsed = parse_qwen_point_decision(
            '{"Reason":"doorway likely leads to bathroom","Angle":330,'
            '"Point":[320,390],"Flag":false,"Confidence":"medium"}',
            image_shape=(480, 640, 3),
            valid_angles=range(0, 360, 30),
        )

        self.assertEqual(parsed["Angle"], 330)
        self.assertEqual(parsed["Point"], [320, 390])
        self.assertEqual(parsed["Reason"], "doorway likely leads to bathroom")
        self.assertEqual(parsed["Confidence"], "medium")
        self.assertFalse(parsed["Flag"])
        self.assertFalse(parsed["fallback"])

    def test_parse_qwen_point_decision_falls_back_to_lower_center_for_bad_point(self):
        parsed = parse_qwen_point_decision(
            '{"Reason":"open hallway","Angle":30,"Point":[9999,-4],"Flag":false}',
            image_shape=(480, 640, 3),
            valid_angles=range(0, 360, 30),
        )

        self.assertEqual(parsed["Angle"], 30)
        self.assertEqual(parsed["Point"], [320, 384])
        self.assertEqual(parsed["Confidence"], "fallback")
        self.assertTrue(parsed["fallback"])
        self.assertIn("point_invalid", parsed["fallback_reason"])

    def test_make_point_goal_mask_draws_small_binary_square(self):
        mask = make_point_goal_mask((10, 12, 3), [6, 7], radius=1)

        self.assertEqual(mask.shape, (10, 12))
        self.assertEqual(mask[7, 6], 255)
        self.assertEqual(mask[6, 5], 255)
        self.assertEqual(mask[5, 4], 0)

    def test_render_qwen_debug_frame_combines_rgb_and_reasoning_panel(self):
        rgb = np.zeros((40, 60, 3), dtype=np.uint8)
        decision = {
            "Angle": 0,
            "Point": [20, 30],
            "Confidence": "high",
            "Reason": "visible floor toward doorway",
            "fallback": False,
        }

        frame = render_qwen_debug_frame(
            rgb,
            decision=decision,
            target="toilet",
            planner_name="qwen_point",
            draw_point=True,
        )

        self.assertEqual(frame.shape[0], 40)
        self.assertGreater(frame.shape[1], 60)
        self.assertTrue(np.any(frame[:, :60, 0] > 200))

    def test_render_qwen_debug_frame_keeps_memory_graph_below_reasoning_text(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        decision = {
            "action": "go",
            "Angle": 0,
            "Point": [320, 380],
            "Confidence": "medium",
            "Reason": "leave the room through the alternate doorway after the front route failed",
            "selected_candidate_ref": "px_v00_r1_c1",
            "pixel_candidate_validation": {
                "passed": True,
                "reason": "valid_pixel_candidate",
                "candidate": {"candidate_ref": "px_v00_r1_c1", "marker_id": "M05"},
            },
            "memory_visualizer": {"enabled": True},
        }

        with patch("qwen_point_planner.draw_memory_graph_thumbnail") as draw_graph:
            render_qwen_debug_frame(
                rgb,
                decision=decision,
                target="chair",
                planner_name="qwen_vlm",
                draw_point=True,
            )

        self.assertGreaterEqual(draw_graph.call_args.kwargs["top"], 300)
        self.assertLess(draw_graph.call_args.kwargs["top"], 389)

    def test_memory_graph_thumbnail_preserves_metric_aspect_and_hides_reverse_edges(self):
        panel = np.zeros((240, 430, 3), dtype=np.uint8)
        state = {
            "enabled": True,
            "supervisor_mode": "goal_seek",
            "graph_layout": {
                "nodes": [
                    {"node_id": "n_00001", "x_m": 0.0, "y_m": 0.0, "current": False},
                    {"node_id": "n_00002", "x_m": 1.0, "y_m": 0.0, "current": True},
                ],
                "edges": [
                    {
                        "src_node_id": "n_00001",
                        "dst_node_id": "n_00002",
                        "status": "success",
                        "edge_type": "temporal_transition",
                    },
                    {
                        "src_node_id": "n_00002",
                        "dst_node_id": "n_00001",
                        "status": "success",
                        "edge_type": "temporal_transition_reverse",
                    },
                ],
            },
        }
        original_circle = cv2.circle
        original_arrowed_line = cv2.arrowedLine

        with patch("qwen_point_planner.cv2.circle", wraps=original_circle) as circles, patch(
            "qwen_point_planner.cv2.arrowedLine", wraps=original_arrowed_line
        ) as arrows:
            draw_memory_graph_thumbnail(panel, state, top=20)

        centers = [call.args[1] for call in circles.call_args_list[::2]]
        self.assertEqual(len(centers), 2)
        self.assertEqual(centers[0][1], centers[1][1])
        self.assertLess(abs(centers[1][0] - centers[0][0]), 150)
        self.assertEqual(arrows.call_count, 1)

    def test_render_qwen_debug_frame_skips_point_for_non_go_vlm_action(self):
        rgb = np.zeros((40, 60, 3), dtype=np.uint8)
        decision = {
            "action": "rotate",
            "Angle": 0,
            "Point": [20, 30],
            "Confidence": "fallback",
            "Reason": "rotate to inspect a new view",
            "fallback": True,
            "vlm_output": {
                "action": "rotate",
                "selected_image_point": None,
            },
        }

        frame = render_qwen_debug_frame(
            rgb,
            decision=decision,
            target="toilet",
            planner_name="qwen_vlm",
            draw_point=True,
        )

        self.assertFalse(np.any(frame[:, :60, 0] > 200))

    def test_decision_action_label_prefers_vlm_output_action(self):
        self.assertEqual(
            decision_action_label({"action": "go", "vlm_output": {"action": "request_observation"}}),
            "REQUEST_OBSERVATION",
        )
        self.assertEqual(decision_action_label({"action": "stop"}), "STOP")
        self.assertEqual(decision_action_label({}), "UNKNOWN")

    def test_fallback_display_reason_keeps_short_code_without_clipped_prose(self):
        reason = (
            "point_verifier:unknown:Overexposed, no clear floor support visible "
            "in the selected waypoint patch"
        )

        self.assertEqual(fallback_display_reason(reason), "point_verifier:unknown")
        self.assertLessEqual(len(fallback_display_reason("x" * 100)), 42)

    def test_executed_action_label_prefers_runner_action(self):
        decision = {
            "action": "request_observation",
            "vlm_output": {"action": "request_observation"},
            "runner_action": "rotate",
        }

        self.assertEqual(decision_action_label(decision), "REQUEST_OBSERVATION")
        self.assertEqual(executed_action_label(decision), "ROTATE")

    def test_decision_debug_point_keeps_original_verifier_rejected_point(self):
        decision = {
            "action": "request_observation",
            "Point": [30, 24],
            "selected_point_verification": {
                "triggered": True,
                "passed": False,
                "point_px": [12, 18],
                "surface": "wall",
            },
        }

        self.assertEqual(decision_debug_point(decision), [12, 18])

        frame = render_qwen_debug_frame(
            np.zeros((30, 40, 3), dtype=np.uint8),
            decision=decision,
            target="chair",
            planner_name="qwen_vlm",
            draw_point=True,
        )
        self.assertGreater(int(frame[18, 12, 0]), 200)

    def test_memory_visualizer_lines_summarize_sidecar_state(self):
        lines = memory_visualizer_lines(
            {
                "enabled": True,
                "schema_version": "nav_memory_context_v6",
                "num_nodes": 2,
                "num_edges": 1,
                "num_deadlock_edges": 1,
                "current_node_id": "n_00002",
                "last_event": {"event_type": "mark_deadlock", "node_id": "n_00002"},
            }
        )

        self.assertEqual(lines[0], "Memory v6")
        self.assertIn("nodes=2 edges=1", lines[1])
        self.assertIn("deadlocks=1", lines[2])
        self.assertIn("current=n_00002", lines[3])
        self.assertIn("mark_deadlock", lines[4])

    def test_make_plan_returns_pixelnav_goal_from_qwen_point(self):
        calls = []

        def fake_vision(prompt, image, system_prompt=""):
            calls.append((prompt, image, system_prompt))
            return (
                '{"Reason":"doorway likely leads to bathroom","Angle":30,'
                '"Point":[4,5],"Flag":false,"Confidence":"medium"}'
            )

        planner = QwenPointPlanner(vision_fn=fake_vision)
        planner.reset("toilet")
        planner.latest_priors = {
            "Supports": ["sink"],
            "StrongCooccurs": ["mirror"],
            "Gateways": ["gateway"],
            "Lookalikes": [],
        }
        pano = [np.zeros((12, 16, 3), dtype=np.uint8) for _ in range(12)]

        goal_rgb, goal_mask, debug_image, vis_rgb, direction, pri_flag, obj_detected = planner.make_plan(pano)

        self.assertEqual(direction, 1)
        self.assertEqual(goal_rgb.shape, (12, 16, 3))
        self.assertEqual(goal_mask[5, 4], 255)
        self.assertEqual(debug_image.shape, (12, 16, 3))
        self.assertEqual(vis_rgb.shape, (12, 16, 3))
        self.assertFalse(pri_flag)
        self.assertFalse(obj_detected)
        self.assertEqual(len(calls), 1)
        self.assertEqual(planner.last_decision["Point"], [4, 5])

    def test_make_plan_from_views_accepts_custom_angles(self):
        def fake_vision(prompt, image, system_prompt=""):
            return (
                '{"Reason":"left doorway","Angle":-30,'
                '"Point":[4,5],"Flag":false,"Confidence":"medium"}'
            )

        planner = QwenPointPlanner(vision_fn=fake_vision)
        planner.reset("toilet")
        pano = [np.zeros((12, 16, 3), dtype=np.uint8) for _ in range(3)]

        _, goal_mask, _, _, direction, pri_flag, obj_detected = planner.make_plan_from_views(
            pano,
            [-60, -30, 0],
            call_type="directed_sweep",
        )

        self.assertEqual(direction, 1)
        self.assertEqual(goal_mask[5, 4], 255)
        self.assertFalse(pri_flag)
        self.assertFalse(obj_detected)
        self.assertEqual(planner.last_decision["angles"], [-60, -30, 0])
        self.assertEqual(planner.last_decision["call_type"], "directed_sweep")

    def test_point_prompt_includes_selected_view_bounds(self):
        planner = QwenPointPlanner()
        planner.reset("toilet")

        prompt = planner._build_point_prompt([0, 30], (480, 640, 3))

        self.assertIn("image_width=640", prompt)
        self.assertIn("image_height=480", prompt)
        self.assertIn("0 <= u < 640", prompt)
        self.assertIn("0 <= v < 480", prompt)

    def test_planner_flushes_qwen_calls_to_jsonl_when_log_path_is_set(self):
        def fake_vision(prompt, image, system_prompt=""):
            return '{"Reason":"floor","Angle":0,"Point":[3,4],"Flag":false,"Confidence":"high"}'

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "qwen_calls.jsonl"
            planner = QwenPointPlanner(vision_fn=fake_vision)
            planner.set_qwen_call_log_path(str(path))
            planner.reset("chair")
            planner.latest_priors = {"Supports": [], "StrongCooccurs": [], "Gateways": [], "Lookalikes": []}
            planner.make_plan([np.zeros((8, 10, 3), dtype=np.uint8) for _ in range(12)])

            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 1)
        self.assertIn('"Point": [3, 4]', lines[0])

    def test_fallback_priors_for_target_provides_conservative_context(self):
        priors = fallback_priors_for_target("chair")

        self.assertIn("table", priors["StrongCooccurs"])
        self.assertIn("gateway", priors["Gateways"])
        self.assertIn("sofa", priors["Lookalikes"])

    def test_priors_thinking_mode_follows_locked_model_identity(self):
        with patch.dict(
            os.environ,
            {
                "QWEN_MODEL": "qwen3-vl-32b-thinking-awq",
                "VOCA_QWEN_MODEL_ROOT": "QuantTrio/Qwen3-VL-32B-Thinking-AWQ",
            },
            clear=False,
        ):
            self.assertTrue(qwen_priors_thinking_enabled())
        with patch.dict(
            os.environ,
            {
                "QWEN_MODEL": "qwen3-vl-8b-instruct-awq-4bit",
                "VOCA_QWEN_MODEL_ROOT": "Qwen/Qwen3-VL-8B-Instruct",
            },
            clear=False,
        ):
            self.assertFalse(qwen_priors_thinking_enabled())

    def test_prior_candidate_shortlist_is_bounded_and_excludes_target(self):
        shortlist = prior_candidate_shortlist_for_target("toilet")

        self.assertNotIn("toilet", shortlist)
        self.assertIn("sink", shortlist)
        self.assertIn("mirror", shortlist)
        self.assertIn("floor", shortlist)
        self.assertIn("gateway", shortlist)
        self.assertEqual(len(shortlist), len(set(shortlist)))
        self.assertLessEqual(len(shortlist), 10)

    def test_detector_prior_allowlist_rejects_reasoning_fragments(self):
        filtered, rejected = filter_detector_prior_classes(
            {
                "Supports": ["pot", "near", "shelf"],
                "StrongCooccurs": ["window"],
                "Gateways": ["gateway"],
                "Lookalikes": [],
            }
        )

        self.assertEqual(filtered["Supports"], ["pot", "shelf"])
        self.assertEqual(filtered["StrongCooccurs"], ["window"])
        self.assertEqual(rejected, ["Supports:near"])

    def test_query_priors_text_uses_target_fallback_after_parse_failures(self):
        calls = []

        def fake_text(prompt, system_prompt=""):
            calls.append((prompt, system_prompt))
            return "I think a chair is often near a table, but this is not JSON."

        planner = QwenPointPlanner(
            text_fn=fake_text,
            config=QwenPointPlannerConfig(retries=2),
        )
        planner.reset("chair")

        priors = planner.query_priors_text()

        self.assertEqual(len(calls), 2)
        self.assertIn("table", priors["StrongCooccurs"])
        self.assertIn("gateway", priors["Gateways"])
        self.assertIn("sofa", priors["Lookalikes"])
        self.assertEqual(planner.priors_success_count, 0)
        self.assertEqual(planner.priors_parse_fail_count, 2)
        self.assertEqual(planner.priors_fallback_count, 1)
        self.assertEqual(planner.priors_last_error, "priors_parse_failed")
        self.assertTrue(planner.priors_log[-1]["fallback"])
        self.assertEqual(planner.priors_log[-1]["fallback_reason"], "priors_parse_failed")

    def test_query_priors_text_records_parse_success_without_fallback(self):
        def fake_text(prompt, system_prompt=""):
            return (
                '{"Supports":["sink"],"StrongCooccurs":["mirror"],'
                '"Gateways":["bathroom doorway"],"Lookalikes":["cabinet"]}'
            )

        planner = QwenPointPlanner(
            text_fn=fake_text,
            config=QwenPointPlannerConfig(retries=2),
        )
        planner.reset("toilet")

        priors = planner.query_priors_text()

        self.assertIn("sink", priors["Supports"])
        self.assertIn("mirror", priors["StrongCooccurs"])
        self.assertIn("gateway", priors["Gateways"])
        self.assertIn("cabinet", priors["Lookalikes"])
        self.assertEqual(planner.priors_success_count, 1)
        self.assertEqual(planner.priors_parse_fail_count, 0)
        self.assertEqual(planner.priors_fallback_count, 0)
        self.assertFalse(planner.priors_log[-1]["fallback"])
        self.assertTrue(planner.priors_log[-1]["target_fallback_augmented"])
        self.assertEqual(planner.priors_log[-1]["model_priors"]["Supports"], ["sink"])


if __name__ == "__main__":
    unittest.main()
