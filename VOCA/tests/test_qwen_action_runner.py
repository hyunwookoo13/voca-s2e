import json
import os
import unittest
from unittest.mock import patch

import numpy as np

from qwen_action_runner import (
    FORWARD_ACTION,
    LEFT_ACTION,
    LOOK_DOWN_ACTION,
    LOOK_UP_ACTION,
    RIGHT_ACTION,
    attach_runner_action_audit,
    attach_go_execution_audit,
    action_from_decision,
    build_go_execution_audit,
    build_go_no_progress_feedback,
    build_pitch_reset_audit,
    build_strategic_progress_evidence,
    classify_go_failure,
    evaluate_go_progress,
    effective_go_supervisor_mode,
    executed_path_tangent_summary,
    full_sweep_observation_offsets,
    go_step_limit_for_decision,
    go_no_progress_observation_offsets,
    _build_metrics_row,
    _pixel_candidate_metrics,
    _strategic_state_metrics,
    _memory_verdict_metrics,
    _agent_heading_rad,
    _runner_env_bool,
    pitch_offset_from_actions,
    pitch_reset_actions_for_offset,
    policy_action_name,
    rotate_yaw_from_decision,
    rotate_yaw_for_runner_action,
    runner_action_after_repeat,
    runner_action_with_context,
    save_memory_sidecar_artifacts,
    save_benchmark_manifest,
    should_execute_target_terminal_micro_approach,
    sync_memory_sidecar_go_execution,
    sync_go_progress_feedback,
    target_approach_translation_cap,
    vlm_backend_failure_reason,
    rotate_stall_observation_offsets,
    observation_offsets_from_decision,
    order_full_sweep_views_for_policy,
    shortest_yaw_delta_deg,
    turn_actions_for_yaw,
)
from nav_audit import NavigationAuditLogger


class QwenActionRunnerTests(unittest.TestCase):
    def test_vlm_backend_failure_reason_distinguishes_backend_health(self):
        self.assertEqual(vlm_backend_failure_reason({"warnings": []}), "")
        self.assertIn(
            "ConnectionError",
            vlm_backend_failure_reason(
                {
                    "warnings": [
                        "qwen_vlm_failed:QWEN_VLM[try=1] ConnectionError: refused"
                    ]
                }
            ),
        )
        self.assertTrue(
            vlm_backend_failure_reason(
                {
                    "vlm_json_fallback": True,
                    "vlm_json_last_error": "no JSON object found",
                }
            ).startswith("invalid_vlm_json:")
        )

    def test_executed_path_tangents_capture_curve_entry_and_exit(self):
        summary = executed_path_tangent_summary(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [0.50, 0.0, 0.0],
                [0.50, 0.0, 0.25],
            ]
        )

        self.assertTrue(summary["available"])
        self.assertAlmostEqual(summary["departure_heading_world_deg"], 0.0)
        self.assertAlmostEqual(summary["arrival_heading_world_deg"], 90.0)
        self.assertAlmostEqual(summary["path_length_m"], 0.75)
        self.assertEqual(len(summary["position_trace_xyz"]), 4)

    def test_runner_env_bool_supports_ablation_switches(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"VOCA_PIXELNAV_CANDIDATE_SCORING": "0"}):
            self.assertFalse(
                _runner_env_bool("VOCA_PIXELNAV_CANDIDATE_SCORING", True)
            )
        with patch.dict(os.environ, {"VOCA_PIXELNAV_CANDIDATE_SCORING": "1"}):
            self.assertTrue(
                _runner_env_bool("VOCA_PIXELNAV_CANDIDATE_SCORING", False)
            )

    def test_target_approach_uses_short_receding_horizon(self):
        self.assertEqual(
            go_step_limit_for_decision(
                {"proactive_target_approach": True},
                16,
                target_approach_steps=4,
            ),
            4,
        )
        self.assertEqual(
            go_step_limit_for_decision(
                {
                    "backend_action_override": {
                        "reason": "stop_confirmation_requires_approach_motion"
                    }
                },
                16,
                target_approach_steps=5,
            ),
            5,
        )
        self.assertEqual(go_step_limit_for_decision({"action": "go"}, 16), 16)
        self.assertEqual(
            go_step_limit_for_decision(
                {"action": "go", "target_context_approach": True},
                16,
                target_approach_steps=6,
            ),
            6,
        )

    def test_memory_verdict_metrics_separates_vlm_output_from_backend_defer(self):
        calls = [
            {
                "raw_vlm_output": {
                    "memory_ops": [
                        {
                            "op": "confirm_revisit_node",
                            "candidate_ref": "revisit_001",
                        }
                    ]
                },
                "memory_contract_validation": {
                    "required": True,
                    "passed": True,
                    "backend_inserted_defer": False,
                },
            },
            {
                "raw_vlm_output": {},
                "memory_contract_validation": {
                    "required": True,
                    "passed": False,
                    "backend_inserted_defer": True,
                },
                "vlm_output": {
                    "memory_ops": [
                        {
                            "op": "defer_revisit_candidate",
                            "candidate_ref": "revisit_002",
                            "source": "backend_memory_contract_guard",
                        }
                    ]
                },
            },
        ]

        metrics = _memory_verdict_metrics(calls)

        self.assertEqual(metrics["required_calls"], 2)
        self.assertEqual(metrics["explicit_calls"], 1)
        self.assertEqual(metrics["main_output_explicit_calls"], 1)
        self.assertEqual(metrics["dedicated_verifier_explicit_calls"], 0)
        self.assertEqual(metrics["dedicated_verifier_unresolved_calls"], 0)
        self.assertEqual(metrics["missing_calls"], 1)
        self.assertEqual(metrics["backend_deferred_calls"], 1)
        self.assertAlmostEqual(metrics["explicit_rate"], 0.5)
        self.assertEqual(metrics["operation_counts"]["confirm_revisit_node"], 1)
        self.assertEqual(metrics["operation_counts"]["defer_revisit_candidate"], 0)

    def test_memory_verdict_metrics_count_valid_dedicated_verifier_output(self):
        calls = [
            {
                "raw_vlm_output": {},
                "memory_contract_validation": {
                    "required": True,
                    "passed": True,
                    "backend_inserted_defer": False,
                    "dedicated_verifier_triggered": True,
                    "dedicated_verifier_operation": "reject_revisit_candidate",
                },
                "revisit_verification": {
                    "triggered": True,
                    "valid": True,
                    "passed": True,
                    "memory_op": {
                        "op": "reject_revisit_candidate",
                        "candidate_ref": "revisit_001",
                        "source": "dedicated_revisit_verifier",
                    },
                },
            },
            {
                "raw_vlm_output": {},
                "memory_contract_validation": {
                    "required": True,
                    "passed": False,
                    "backend_inserted_defer": False,
                    "dedicated_verifier_triggered": True,
                    "dedicated_verifier_operation": "defer_revisit_candidate",
                },
                "revisit_verification": {
                    "triggered": True,
                    "valid": False,
                    "passed": False,
                    "error": "response truncated",
                    "memory_op": {
                        "op": "defer_revisit_candidate",
                        "candidate_ref": "revisit_002",
                        "source": "dedicated_revisit_verifier",
                    },
                },
            },
        ]

        metrics = _memory_verdict_metrics(calls)

        self.assertEqual(metrics["required_calls"], 2)
        self.assertEqual(metrics["explicit_calls"], 1)
        self.assertEqual(metrics["main_output_explicit_calls"], 0)
        self.assertEqual(metrics["dedicated_verifier_explicit_calls"], 1)
        self.assertEqual(metrics["dedicated_verifier_unresolved_calls"], 1)
        self.assertEqual(metrics["missing_calls"], 1)
        self.assertEqual(
            metrics["operation_counts"]["reject_revisit_candidate"],
            1,
        )

    def test_memory_verdict_metrics_skip_backend_proactive_actions(self):
        metrics = _memory_verdict_metrics(
            [
                {
                    "proactive_stop_probe": True,
                    "action": "stop",
                    "raw_vlm_output": {"memory_ops": []},
                    "vlm_input": {
                        "memory": {
                            "candidate_refs": {
                                "revisits": [{"candidate_ref": "revisit_001"}]
                            }
                        }
                    },
                }
            ]
        )

        self.assertEqual(metrics["required_calls"], 0)
        self.assertEqual(metrics["missing_calls"], 0)
        self.assertEqual(metrics["explicit_rate"], 1.0)

    def test_pixel_candidate_metrics_tracks_rgb_gate_and_selected_evidence(self):
        calls = [
            {
                "vlm_input": {
                    "pixel_candidates": {
                        "candidates": [
                            {
                                "candidate_ref": "px_1",
                                "avoid": False,
                                "status": "rgb_verification_required",
                                "requires_rgb_verification": True,
                                "negative_memory_saturation_recovery": True,
                            },
                            {
                                "candidate_ref": "px_2",
                                "avoid": True,
                                "status": "verified_failed_pixel_neighborhood",
                            },
                        ],
                        "policy_conditioning": {
                            "negative_saturation_recovery": {"triggered": True}
                        },
                    },
                },
                "raw_vlm_output": {"action": "go"},
                "action": "go",
                "pixel_candidate_validation": {
                    "passed": True,
                    "ref_alias_applied": True,
                    "candidate": {
                        "candidate_ref": "px_1",
                        "requires_rgb_verification": True,
                        "negative_memory_saturation_recovery": True,
                        "visual_evidence": {"support_score": 0.25},
                    },
                },
                "selected_point_verification": {
                    "triggered": True,
                    "passed": True,
                },
            },
            {
                "vlm_input": {
                    "pixel_candidates": {
                        "candidates": [
                            {"candidate_ref": "px_3", "status": "rgb_visually_supported"}
                        ]
                    }
                },
                "raw_vlm_output": {"action": "go"},
                "pixel_candidate_validation": {"passed": False},
            },
        ]

        metrics = _pixel_candidate_metrics(calls)

        self.assertEqual(metrics["offered"], 3)
        self.assertEqual(metrics["generated"], 3)
        self.assertEqual(metrics["avoided"], 1)
        self.assertEqual(metrics["verification_required"], 1)
        self.assertEqual(metrics["visually_supported"], 1)
        self.assertEqual(metrics["raw_go_count"], 2)
        self.assertEqual(metrics["gate_passes"], 1)
        self.assertEqual(metrics["gate_rejections"], 1)
        self.assertEqual(metrics["selected_verification_required"], 1)
        self.assertEqual(metrics["saturation_recovery_calls"], 1)
        self.assertEqual(metrics["saturation_recovery_offered"], 1)
        self.assertEqual(metrics["saturation_recovery_selected"], 1)
        self.assertEqual(metrics["saturation_recovery_executed"], 1)
        self.assertEqual(metrics["saturation_recovery_verifier_passes"], 1)
        self.assertEqual(metrics["saturation_recovery_verifier_rejections"], 0)
        self.assertEqual(metrics["marker_alias_repairs"], 1)
        self.assertAlmostEqual(metrics["selected_support_avg"], 0.25)

    def test_attach_runner_action_audit_updates_decision_and_qwen_log(self):
        class Planner:
            def __init__(self):
                self.qwen_call_log = [{"action": "request_observation"}]

        planner = Planner()
        decision = {"action": "request_observation"}

        attach_runner_action_audit(
            planner,
            decision=decision,
            vlm_requested_action="request_observation",
            runner_action="rotate",
            override_reason="multi_view_observation_already_available",
            observation_view_count=5,
        )

        self.assertEqual(decision["vlm_requested_action"], "request_observation")
        self.assertEqual(decision["runner_action"], "rotate")
        self.assertEqual(planner.qwen_call_log[-1]["runner_action"], "rotate")
        self.assertEqual(
            planner.qwen_call_log[-1]["runner_override_reason"],
            "multi_view_observation_already_available",
        )

    def test_agent_heading_uses_agent_to_world_rotation(self):
        import math
        from habitat_sim.utils.common import quat_from_angle_axis

        class Sim:
            def __init__(self, rotation):
                self._state = type("State", (), {"rotation": rotation})()

            def get_agent_state(self):
                return self._state

        class Env:
            def __init__(self, rotation):
                self.sim = Sim(rotation)

        initial = quat_from_angle_axis(0.0, np.array([0.0, 1.0, 0.0]))
        turned = quat_from_angle_axis(math.radians(30.0), np.array([0.0, 1.0, 0.0]))
        delta = _agent_heading_rad(Env(turned)) - _agent_heading_rad(Env(initial))

        self.assertAlmostEqual(math.degrees(delta), -30.0, places=5)

    def test_turn_actions_for_yaw_uses_habitat_30_degree_turns(self):
        self.assertEqual(turn_actions_for_yaw(90), [RIGHT_ACTION, RIGHT_ACTION, RIGHT_ACTION])
        self.assertEqual(turn_actions_for_yaw(-60), [LEFT_ACTION, LEFT_ACTION])
        self.assertEqual(turn_actions_for_yaw(0), [])
        self.assertEqual(turn_actions_for_yaw(44), [RIGHT_ACTION])

    def test_action_from_decision_prefers_vlm_output_action(self):
        decision = {"action": "go", "vlm_output": {"action": "request_observation"}}

        self.assertEqual(action_from_decision(decision), "request_observation")

    def test_observation_offsets_from_directed_request(self):
        decision = {
            "vlm_output": {
                "action": "request_observation",
                "observation_request": {
                    "mode": "directed_sweep",
                    "yaw_offsets_deg": [-30, 0, 30],
                },
            }
        }

        self.assertEqual(observation_offsets_from_decision(decision), [-30, 0, 30])

    def test_observation_offsets_full_sweep_is_explicit_but_not_default(self):
        decision = {
            "vlm_output": {
                "action": "request_observation",
                "observation_request": {"mode": "full_sweep"},
            }
        }

        with patch.dict(
            os.environ,
            {"VOCA_EFFICIENT_CIRCULAR_SWEEP": "0"},
            clear=False,
        ):
            self.assertEqual(
                observation_offsets_from_decision(decision),
                [-180, -135, -90, -45, 0, 45, 90, 135],
            )

    def test_efficient_circular_sweep_is_an_explicit_ablation(self):
        with patch.dict(
            os.environ,
            {"VOCA_EFFICIENT_CIRCULAR_SWEEP": "1"},
            clear=False,
        ):
            self.assertEqual(
                full_sweep_observation_offsets(),
                [0, 60, 120, -180, -120, -60],
            )

    def test_full_sweep_uses_shortest_circular_path_in_twelve_turns(self):
        offsets = [0, 60, 120, -180, -120, -60]
        current = 0
        action_count = 0
        for offset in offsets:
            delta = shortest_yaw_delta_deg(offset, current)
            action_count += len(turn_actions_for_yaw(delta))
            current = offset
        action_count += len(turn_actions_for_yaw(shortest_yaw_delta_deg(0, current)))

        self.assertEqual(shortest_yaw_delta_deg(-120, -180), 60.0)
        self.assertEqual(action_count, 12)

    def test_full_sweep_is_presented_opposite_first_to_the_policy(self):
        physical_angles = [0, 60, 120, -180, -120, -60]
        physical_views = [
            np.full((2, 2, 3), index, dtype=np.uint8)
            for index in range(len(physical_angles))
        ]

        ordered_views, ordered_angles = order_full_sweep_views_for_policy(
            physical_views,
            physical_angles,
        )

        self.assertEqual(ordered_angles, [-180, -120, -60, 0, 60, 120])
        self.assertEqual(int(ordered_views[0][0, 0, 0]), 3)
        self.assertEqual(int(ordered_views[3][0, 0, 0]), 0)

    def test_observation_offsets_default_to_small_directed_sweep(self):
        decision = {"vlm_output": {"action": "request_observation", "observation_request": {}}}

        self.assertEqual(observation_offsets_from_decision(decision), [-30, 0, 30])

    def test_repeated_rotate_is_promoted_to_observation_sweep(self):
        self.assertEqual(runner_action_after_repeat("rotate", 0, rotate_to_observe_after=2), ("rotate", 1, False))
        self.assertEqual(
            runner_action_after_repeat("rotate", 1, rotate_to_observe_after=2),
            ("request_observation", 0, True),
        )
        self.assertEqual(runner_action_after_repeat("go", 1, rotate_to_observe_after=2), ("go", 0, False))

    def test_rotate_stall_observation_offsets_are_wider_than_default_request(self):
        self.assertEqual(rotate_stall_observation_offsets(), [-60, -30, 0, 30, 60])

    def test_multi_view_observation_request_is_overridden_to_rotate(self):
        self.assertEqual(
            runner_action_with_context(
                "request_observation",
                0,
                current_view_count=5,
                rotate_to_observe_after=2,
            ),
            ("rotate", 0, False, "multi_view_request_observation_suppressed"),
        )
        self.assertEqual(
            runner_action_with_context(
                "request_observation",
                0,
                current_view_count=1,
                rotate_to_observe_after=2,
            ),
            ("request_observation", 0, False, ""),
        )

    def test_explicit_full_sweep_is_allowed_after_partial_multi_view(self):
        self.assertEqual(
            runner_action_with_context(
                "request_observation",
                0,
                current_view_count=5,
                rotate_to_observe_after=2,
                observation_mode="full_sweep",
            ),
            ("request_observation", 0, False, ""),
        )

    def test_multi_view_observation_override_rotate_uses_nonzero_yaw(self):
        decision = {"vlm_output": {"action": "request_observation", "control": {"rotate_yaw_deg": 0}}}

        self.assertEqual(
            rotate_yaw_for_runner_action(decision, "multi_view_request_observation_suppressed"),
            30.0,
        )

    def test_explicit_zero_yaw_rotate_is_canonicalized_to_real_turn(self):
        decision = {
            "vlm_output": {
                "action": "rotate",
                "control": {"rotate_yaw_deg": 0},
            }
        }

        self.assertEqual(rotate_yaw_from_decision(decision), 0.0)
        self.assertEqual(rotate_yaw_for_runner_action(decision), 30.0)

    def test_verified_target_memory_override_uses_micro_translation_cap(self):
        decision = {
            "vlm_output": {
                "target_approach_memory_override": {
                    "authority": "current_identity_verified_target_evidence"
                }
            }
        }

        self.assertEqual(
            target_approach_translation_cap(decision, default_cap_m=0.5),
            0.25,
        )
        self.assertEqual(target_approach_translation_cap({}, default_cap_m=0.5), 0.5)

    def test_verified_stop_refinement_uses_single_step_translation_cap(self):
        decision = {
            "vlm_output": {
                "target_terminal_refinement": {
                    "schema_version": "verified_stop_terminal_refinement_v1"
                }
            }
        }

        with patch.dict(
            os.environ,
            {"VOCA_TARGET_STOP_REFINEMENT_MAX_TRANSLATION_M": "0.15"},
        ):
            self.assertEqual(
                target_approach_translation_cap(decision, default_cap_m=0.5),
                0.15,
            )

    def test_target_terminal_micro_approach_requires_verified_scale_recovery(self):
        decision = {
            "proactive_target_approach": True,
            "proactive_stop_probe_result": {
                "target_evidence_passed": True,
                "confidence": "high",
                "target_center_guard_satisfied": True,
                "identity_critic": {
                    "triggered": True,
                    "passed": True,
                    "exact_target": True,
                },
                "near_goal_visual_latch_approach_count": 3,
                "near_goal_visual_latch_approaches_required": 3,
                "target_scale_consistent": False,
                "target_approach_execution_satisfied": False,
            },
        }

        self.assertTrue(
            should_execute_target_terminal_micro_approach(
                decision,
                executed_count=0,
                max_executions=2,
            )
        )
        self.assertFalse(
            should_execute_target_terminal_micro_approach(
                decision,
                executed_count=2,
                max_executions=2,
            )
        )
        decision["proactive_stop_probe_result"]["identity_critic"]["passed"] = False
        self.assertFalse(
            should_execute_target_terminal_micro_approach(
                decision,
                executed_count=0,
                max_executions=2,
            )
        )

    def test_evaluate_go_progress_uses_observable_translation(self):
        progress = evaluate_go_progress(
            {"distance_to_goal": 4.6},
            {"distance_to_goal": 4.9},
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.2, 0.0, 0.0],
            collision_count=0,
            controller_reached_waypoint=False,
            supervisor_mode="goal_seek",
            min_progress_m=0.05,
        )

        self.assertTrue(progress["success"])
        self.assertTrue(progress["execution_success"])
        self.assertFalse(progress["no_progress"])
        self.assertAlmostEqual(progress["translation_m"], 0.2)
        self.assertAlmostEqual(progress["audit_goal_distance_delta_m"], -0.3)

    def test_evaluate_go_progress_marks_stationary_execution_as_no_progress(self):
        progress = evaluate_go_progress(
            {"distance_to_goal": 4.6},
            {"distance_to_goal": 4.0},
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.0, 0.0, 0.0],
            collision_count=0,
            controller_reached_waypoint=False,
            supervisor_mode="goal_seek",
            min_progress_m=0.05,
        )

        self.assertFalse(progress["success"])
        self.assertTrue(progress["no_progress"])
        self.assertAlmostEqual(progress["translation_m"], 0.0)
        self.assertAlmostEqual(progress["audit_goal_distance_delta_m"], 0.6)

    def test_go_no_progress_observation_offsets_are_directed_sweep(self):
        self.assertEqual(go_no_progress_observation_offsets(), [-60, -30, 0, 30, 60])
        with patch.dict(
            os.environ,
            {"VOCA_EFFICIENT_CIRCULAR_SWEEP": "0"},
            clear=False,
        ):
            self.assertEqual(
                go_no_progress_observation_offsets("escape_deadlock"),
                [-180, -135, -90, -45, 0, 45, 90, 135],
            )

    def test_strategic_evidence_detects_departure_from_latest_failed_sector(self):
        class Sidecar:
            directional_failures = [{"angle_deg": 0.0}]

        evidence = build_strategic_progress_evidence(
            {"Angle": 90, "selected_topological_relation_type": "frontier"},
            Sidecar(),
        )

        self.assertTrue(evidence["sector_departure_selected"])
        self.assertEqual(evidence["direction_change_from_last_failure_deg"], 90.0)
        self.assertFalse(evidence["backtrack_edge_selected"])

    def test_backtrack_selection_updates_effective_supervisor_mode(self):
        class Sidecar:
            supervisor_mode = "escape_deadlock"

            def update_supervisor_mode(self, mode, reason):
                self.supervisor_mode = mode
                self.reason = reason

        sidecar = Sidecar()
        mode = effective_go_supervisor_mode(
            sidecar,
            {"selected_topological_relation_type": "backtrack"},
        )

        self.assertEqual(mode, "backtrack")
        self.assertEqual(sidecar.supervisor_mode, "backtrack")
        self.assertEqual(sidecar.reason, "selected_verified_backtrack_edge")

    def test_policy_action_names_match_habitat_action_space(self):
        self.assertEqual(policy_action_name(0), "stop")
        self.assertEqual(policy_action_name(1), "move_forward")
        self.assertEqual(policy_action_name(2), "turn_left")
        self.assertEqual(policy_action_name(3), "turn_right")
        self.assertEqual(policy_action_name(4), "look_up")
        self.assertEqual(policy_action_name(5), "look_down")

    def test_pitch_offset_tracks_pixelnav_camera_pitch_actions(self):
        self.assertEqual(
            pitch_offset_from_actions([LOOK_DOWN_ACTION, LOOK_DOWN_ACTION, FORWARD_ACTION, LOOK_UP_ACTION]),
            -1,
        )
        self.assertEqual(pitch_offset_from_actions([LOOK_UP_ACTION, LOOK_UP_ACTION]), 2)
        self.assertEqual(pitch_offset_from_actions([FORWARD_ACTION, RIGHT_ACTION]), 0)

    def test_pitch_reset_actions_neutralize_policy_pitch_offset(self):
        self.assertEqual(pitch_reset_actions_for_offset(-3), [LOOK_UP_ACTION, LOOK_UP_ACTION, LOOK_UP_ACTION])
        self.assertEqual(pitch_reset_actions_for_offset(2), [LOOK_DOWN_ACTION, LOOK_DOWN_ACTION])
        self.assertEqual(pitch_reset_actions_for_offset(0), [])

    def test_build_pitch_reset_audit_reports_residual_offset(self):
        audit = build_pitch_reset_audit(
            policy_actions=[LOOK_DOWN_ACTION, LOOK_DOWN_ACTION, FORWARD_ACTION],
            pitch_reset_actions=[LOOK_UP_ACTION],
            pitch_reset_truncated=True,
        )

        self.assertEqual(audit["policy_pitch_offset"], -2)
        self.assertEqual(audit["pitch_reset_actions"], [LOOK_UP_ACTION])
        self.assertEqual(audit["pitch_reset_action_names"], ["look_up"])
        self.assertTrue(audit["pitch_reset_truncated"])
        self.assertEqual(audit["residual_pitch_offset"], -1)

    def test_build_go_no_progress_feedback_captures_failed_waypoint(self):
        decision = {
            "action": "go",
            "Angle": 30,
            "Point": [320, 360],
            "selected_view_id": 2,
            "vlm_output": {"selected_view_type": "front"},
        }
        progress = {
            "success": False,
            "no_progress": True,
            "translation_m": 0.02,
            "collision_count": 0,
            "min_translation_m": 0.05,
            "audit_goal_distance_delta_m": -0.02,
        }

        feedback = build_go_no_progress_feedback(decision, progress)

        self.assertEqual(feedback["event"], "go_no_progress")
        self.assertEqual(feedback["selected_view_id"], 2)
        self.assertEqual(feedback["selected_view_type"], "front")
        self.assertEqual(feedback["angle_deg"], 30)
        self.assertEqual(feedback["point_px"], [320, 360])
        self.assertEqual(feedback["progress"]["translation_m"], 0.02)
        self.assertNotIn("success", feedback["progress"])
        self.assertNotIn("audit_goal_distance_delta_m", feedback["progress"])

    def test_classify_go_failure_marks_collision_as_primary_blocker(self):
        classification = classify_go_failure(
            {
                "policy_action_names": ["move_forward", "turn_right", "move_forward"],
                "policy_steps": 3,
                "collision_count": 2,
                "stop_action_seen": False,
                "max_go_steps": 8,
                "go_progress": {"no_progress": True, "distance_delta_m": -0.32},
            }
        )

        self.assertEqual(classification["primary"], "collision_blocked")
        self.assertIn("moved_away_from_goal", classification["labels"])
        self.assertTrue(classification["needs_replan"])

    def test_classify_go_failure_marks_look_only_policy(self):
        classification = classify_go_failure(
            {
                "policy_action_names": ["look_down", "look_down"],
                "policy_steps": 2,
                "collision_count": 0,
                "stop_action_seen": False,
                "max_go_steps": 8,
                "go_progress": {"no_progress": True, "distance_delta_m": 0.0},
            }
        )

        self.assertEqual(classification["primary"], "look_only_policy")
        self.assertIn("no_translation", classification["labels"])

    def test_classify_go_failure_marks_progress_when_distance_decreases(self):
        classification = classify_go_failure(
            {
                "policy_action_names": ["move_forward", "move_forward"],
                "policy_steps": 2,
                "collision_count": 0,
                "stop_action_seen": False,
                "max_go_steps": 8,
                "go_progress": {"no_progress": False, "distance_delta_m": 0.24},
            }
        )

        self.assertEqual(classification["primary"], "progress")
        self.assertFalse(classification["needs_replan"])

    def test_sync_go_progress_feedback_records_no_progress_and_clears_on_success(self):
        class Planner:
            def __init__(self):
                self.recorded = []
                self.clear_count = 0

            def record_navigation_feedback(self, feedback):
                self.recorded.append(feedback)

            def clear_navigation_feedback(self):
                self.clear_count += 1

        planner = Planner()
        decision = {"Angle": 0, "Point": [320, 360], "selected_view_id": 0}

        sync_go_progress_feedback(planner, decision, {"no_progress": True, "distance_delta_m": -0.01})
        sync_go_progress_feedback(planner, decision, {"no_progress": False, "distance_delta_m": 0.2})

        self.assertEqual(len(planner.recorded), 1)
        self.assertEqual(planner.recorded[0]["event"], "go_no_progress")
        self.assertEqual(planner.clear_count, 1)

    def test_build_go_execution_audit_captures_policy_actions_and_progress(self):
        decision = {
            "Angle": 30,
            "Point": [320, 360],
            "selected_view_id": 2,
            "selected_candidate_ref": "px_v02_r1_c1",
            "selected_topological_candidate_ref": "exit_004",
            "selected_topological_edge_id": "e_00004",
            "selected_topological_relation_type": "backtrack",
        }
        progress = {
            "success": False,
            "no_progress": True,
            "distance_delta_m": -0.25,
            "start_distance_to_goal": 4.2,
            "final_distance_to_goal": 4.45,
            "min_progress_m": 0.05,
        }

        audit = build_go_execution_audit(
            decision=decision,
            start_metrics={"distance_to_goal": 4.2, "num_steps": 10, "top_down_map": np.zeros((2, 2), dtype=np.uint8)},
            final_metrics={"distance_to_goal": 4.45, "num_steps": 14, "top_down_map": np.ones((2, 2), dtype=np.uint8)},
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.1, 0.0, 0.0],
            policy_actions=[RIGHT_ACTION, 1, 1, 0],
            collision_count=1,
            stop_action_seen=True,
            max_go_steps=8,
            turn_yaw_deg=30,
            turn_truncated=False,
            go_progress=progress,
            pitch_reset={
                "policy_pitch_offset": -2,
                "pitch_reset_actions": [LOOK_UP_ACTION, LOOK_UP_ACTION],
                "pitch_reset_truncated": False,
                "residual_pitch_offset": 0,
            },
        )

        self.assertEqual(audit["selected_angle_deg"], 30)
        self.assertEqual(audit["selected_view_id"], 2)
        self.assertEqual(audit["selected_point_px"], [320, 360])
        self.assertEqual(audit["selected_candidate_ref"], "px_v02_r1_c1")
        self.assertEqual(audit["selected_topological_candidate_ref"], "exit_004")
        self.assertEqual(audit["selected_topological_edge_id"], "e_00004")
        self.assertEqual(audit["selected_topological_relation_type"], "backtrack")
        self.assertEqual(audit["policy_actions"], [RIGHT_ACTION, 1, 1, 0])
        self.assertEqual(audit["policy_action_names"], ["turn_right", "move_forward", "move_forward", "stop"])
        self.assertEqual(audit["policy_steps"], 4)
        self.assertEqual(audit["collision_count"], 1)
        self.assertTrue(audit["stop_action_seen"])
        self.assertTrue(audit["go_progress"]["no_progress"])
        self.assertEqual(audit["failure_classification"]["primary"], "collision_blocked")
        self.assertEqual(audit["pitch_reset"]["policy_pitch_offset"], -2)
        self.assertEqual(audit["pitch_reset"]["pitch_reset_action_names"], ["look_up", "look_up"])
        self.assertEqual(audit["pitch_reset"]["residual_pitch_offset"], 0)
        json.dumps(audit)

    def test_attach_go_execution_audit_updates_decision_qwen_log_and_steps_runtime(self):
        decision = {
            "Reason": "front floor",
            "Angle": 0,
            "Point": [32, 36],
            "Confidence": "medium",
            "selected_view_id": 0,
        }
        go_execution = {
            "selected_point_px": [32, 36],
            "policy_actions": [1, 1],
            "go_progress": {"no_progress": False, "distance_delta_m": 0.2},
        }

        class Planner:
            def __init__(self):
                self.qwen_call_log = [dict(decision)]

        planner = Planner()
        logger = NavigationAuditLogger()
        logger.record_decision(
            decision=decision,
            target_object="chair",
            image_shape=(96, 128, 3),
            frame_index=0,
            position_xyz=[0.0, 0.0, 0.0],
            metrics={"distance_to_goal": 4.0},
            angles=[0],
            call_type="qwen_action_loop",
            selected_view_id=0,
        )

        attach_go_execution_audit(planner, logger, decision, go_execution)

        self.assertEqual(decision["runner_go_execution"], go_execution)
        self.assertEqual(decision["runner_go_progress"], go_execution["go_progress"])
        self.assertEqual(planner.qwen_call_log[-1]["runner_go_execution"], go_execution)
        self.assertEqual(planner.qwen_call_log[-1]["runner_go_progress"], go_execution["go_progress"])
        self.assertEqual(logger.steps[0]["runtime"]["go_execution"], go_execution)
        self.assertEqual(logger.steps[0]["runtime"]["go_progress"], go_execution["go_progress"])

    def test_strategic_metrics_count_intents_and_forced_exit_guard(self):
        metrics = _strategic_state_metrics(
            [
                {
                    "strategic_state": {
                        "current_room": "living_room",
                        "navigation_intent": "search_current_room",
                    },
                    "strategic_validation": {
                        "triggered": True,
                        "passed": False,
                    },
                },
                {
                    "strategic_state": {
                        "current_room": "living_room",
                        "navigation_intent": "traverse_gateway",
                    },
                    "strategic_validation": {
                        "triggered": True,
                        "passed": True,
                    },
                },
            ]
        )

        self.assertEqual(metrics["intent_counts"]["search_current_room"], 1)
        self.assertEqual(metrics["intent_counts"]["traverse_gateway"], 1)
        self.assertEqual(metrics["forced_exit_checks"], 2)
        self.assertEqual(metrics["forced_exit_passes"], 1)
        self.assertEqual(metrics["forced_exit_rejections"], 1)
        self.assertEqual(
            metrics["last_state"]["navigation_intent"],
            "traverse_gateway",
        )

    def test_strategic_metrics_count_novel_sector_execution_outcome(self):
        metrics = _strategic_state_metrics(
            [
                {
                    "action": "go",
                    "strategic_state": {
                        "current_room": "living_room",
                        "navigation_intent": "escape_deadlock",
                    },
                    "strategic_validation": {
                        "triggered": True,
                        "passed": True,
                        "backend_novel_sector_escape": {
                            "triggered": True,
                            "selected_candidate_ref": "px_v05_r1_c1",
                        },
                    },
                    "selected_point_verification": {
                        "triggered": True,
                        "passed": True,
                    },
                    "runner_strategic_go_execution": {
                        "triggered": True,
                        "passed": False,
                    },
                }
            ]
        )

        self.assertEqual(metrics["novel_sector_attempts"], 1)
        self.assertEqual(metrics["novel_sector_executed"], 1)
        self.assertEqual(metrics["novel_sector_no_progress"], 1)

    def test_build_metrics_row_exposes_priors_health(self):
        class Episode:
            object_category = "chair"

        class Env:
            current_episode = Episode()

            def get_metrics(self):
                return {
                    "success": 0.0,
                    "spl": 0.0,
                    "distance_to_goal": 3.0,
                    "num_steps": 4,
                }

        class Planner:
            planner_name = "qwen_vlm"
            qwen_call_log = []
            llm_call_count = 3
            llm_success_count = 2
            llm_error_count = 0
            llm_durations = []
            priors_durations = [22.0]
            navigation_vlm_durations = [1.0, 2.0]
            llm_last_error = ""
            yoloe_durations = []
            priors_success_count = 0
            priors_parse_fail_count = 2
            priors_fallback_count = 1
            priors_last_error = "priors_parse_failed"
            vlm_json_fallback_count = 2
            vlm_json_last_error = "no JSON object found"
            point_verification_calls = 3
            point_verification_pass_count = 1
            point_verification_rejection_count = 2
            point_verification_error_count = 0
            point_verification_durations = [0.5, 0.7, 0.6]
            latest_priors = {
                "Supports": [],
                "StrongCooccurs": ["table"],
                "Gateways": ["gateway"],
                "Lookalikes": ["sofa"],
            }

        row = _build_metrics_row(
            habitat_env=Env(),
            nav_planner=Planner(),
            episode_index=0,
            start_geodesic_m=4.0,
            episode_time_sec=1.5,
            truncated_by_max_steps=False,
            audit_paths={},
            stats={"dist_m": 0.5},
            action_counts={
                "vlm_circuit_breaker_tripped": 1,
                "vlm_max_consecutive_failures": 3,
            },
            configured_max_episode_steps=1000,
            benchmark_invalid_reason="vlm_backend_unavailable",
        )

        self.assertEqual(row["configured_max_episode_steps"], 1000)
        self.assertEqual(row["priors_success_count"], 0)
        self.assertEqual(row["priors_parse_fail_count"], 2)
        self.assertEqual(row["priors_fallback_count"], 1)
        self.assertEqual(row["priors_context_item_count"], 3)
        self.assertEqual(row["priors_last_error"], "priors_parse_failed")
        self.assertEqual(row["vlm_json_fallback_count"], 2)
        self.assertEqual(row["vlm_json_last_error"], "no JSON object found")
        self.assertEqual(row["qwen_point_verification_calls"], 3)
        self.assertEqual(row["qwen_point_verification_passes"], 1)
        self.assertEqual(row["qwen_point_verification_rejections"], 2)
        self.assertEqual(row["qwen_point_verification_errors"], 0)
        self.assertAlmostEqual(row["qwen_point_verification_avg_time_sec"], 0.6)
        self.assertEqual(row["benchmark_valid"], 0)
        self.assertEqual(
            row["benchmark_invalid_reason"],
            "vlm_backend_unavailable",
        )
        self.assertEqual(row["qwen_vlm_circuit_breaker_tripped"], 1)
        self.assertEqual(row["qwen_vlm_max_consecutive_failures"], 3)
        self.assertEqual(row["priors_calls"], 1)
        self.assertAlmostEqual(row["priors_avg_time_sec"], 22.0)
        self.assertEqual(row["qwen_navigation_vlm_calls"], 2)
        self.assertAlmostEqual(row["qwen_navigation_vlm_avg_time_sec"], 1.5)

    def test_sync_memory_sidecar_go_execution_passes_positions_and_frame_index(self):
        class Sidecar:
            def __init__(self):
                self.calls = []

            def record_go_execution(self, **kwargs):
                self.calls.append(kwargs)

        class Planner:
            def __init__(self):
                self.memory_sidecar = Sidecar()

        planner = Planner()
        go_execution = {"go_progress": {"no_progress": False}}

        sync_memory_sidecar_go_execution(
            planner,
            go_execution=go_execution,
            frame_index=5,
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.4, 0.0, 0.0],
            start_heading_rad=0.25,
            final_heading_rad=0.5,
        )

        self.assertEqual(len(planner.memory_sidecar.calls), 1)
        self.assertIs(planner.memory_sidecar.calls[0]["go_execution"], go_execution)
        self.assertEqual(planner.memory_sidecar.calls[0]["frame_index"], 5)
        self.assertEqual(planner.memory_sidecar.calls[0]["final_position_xyz"], [0.4, 0.0, 0.0])
        self.assertEqual(planner.memory_sidecar.calls[0]["start_heading_rad"], 0.25)
        self.assertEqual(planner.memory_sidecar.calls[0]["final_heading_rad"], 0.5)

    def test_sync_memory_sidecar_hides_pose_when_policy_localization_is_disabled(self):
        class Sidecar:
            def __init__(self):
                self.calls = []

            def record_go_execution(self, **kwargs):
                self.calls.append(kwargs)

        class Planner:
            localization_contract = {"source": "none", "pose_available_to_policy": False}

            def __init__(self):
                self.memory_sidecar = Sidecar()

        planner = Planner()
        sync_memory_sidecar_go_execution(
            planner,
            go_execution={"go_progress": {"no_progress": False}},
            frame_index=3,
            start_position_xyz=[1.0, 0.0, 2.0],
            final_position_xyz=[1.5, 0.0, 2.0],
            start_heading_rad=0.0,
            final_heading_rad=0.0,
        )

        self.assertEqual(planner.memory_sidecar.calls[0]["start_position_xyz"], [])
        self.assertEqual(planner.memory_sidecar.calls[0]["final_position_xyz"], [])

    def test_benchmark_manifest_records_localization_contract(self):
        import os
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        class Planner:
            planner_name = "qwen_vlm"
            memory_sidecar = None
            localization_contract = {
                "schema_version": "voca_localization_contract_v1",
                "source": "habitat_sim_pose_declared",
                "pose_available_to_policy": True,
                "uses_sim_ground_truth": True,
                "sim_pose_explicitly_allowed": True,
            }

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "QWEN_MODEL": "qwen3-vl-32b-thinking-awq",
                "VOCA_QWEN_MODEL_ROOT": "QuantTrio/Qwen3-VL-32B-Thinking-AWQ",
            },
        ):
            path = save_benchmark_manifest(
                trajectory_dir=tmpdir,
                episode_index=0,
                nav_planner=Planner(),
                audit_paths={},
                action_counts={},
                metrics={},
            )
            data = json.loads(Path(path).read_text(encoding="utf-8"))

        self.assertEqual(
            data["localization_contract"]["source"],
            "habitat_sim_pose_declared",
        )
        self.assertTrue(data["localization_contract"]["uses_sim_ground_truth"])
        self.assertEqual(data["schema_version"], "voca_qwen_benchmark_manifest_v2")
        self.assertEqual(
            data["qwen_model_root"],
            "QuantTrio/Qwen3-VL-32B-Thinking-AWQ",
        )
        self.assertEqual(len(data["reproducibility"]["experiment_fingerprint_sha256"]), 64)
        self.assertIn("qwen_vlm_planner.py", data["reproducibility"]["source_files"])

    def test_save_memory_sidecar_artifacts_updates_memory_graph_path(self):
        class Sidecar:
            def save_artifacts(self, out_dir):
                return {"memory_graph_json": str(out_dir / "memory_graph.json"), "feedback_md": str(out_dir / "Feedback.md")}

        class Planner:
            memory_sidecar = Sidecar()

        paths = save_memory_sidecar_artifacts(Planner(), {"steps_json": "/tmp/steps.json"}, "/tmp/memory")

        self.assertEqual(paths["memory_graph_json"], "/tmp/memory/memory_graph.json")
        self.assertEqual(paths["feedback_md"], "/tmp/memory/Feedback.md")
        self.assertEqual(paths["steps_json"], "/tmp/steps.json")


if __name__ == "__main__":
    unittest.main()
