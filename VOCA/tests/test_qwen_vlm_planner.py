import json
import hashlib
import math
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qwen_vlm_planner import (
    CompactQwenNavVLMClient,
    QwenSelectedPointVerifier,
    QwenRevisitVerifier,
    QwenTargetStopVerifier,
    QwenVLMPlanner,
    _compact_nav_prompt,
    _compact_nav_retry_prompt,
    enforce_memory_verdict_contract,
    normalize_strategic_state,
    selected_point_visual_risk,
    _NAV_VLM_OUTPUT_GUIDED_JSON,
)
from qwen_memory_prompt import compact_memory_json
from qwen_point_planner import QwenPointPlannerConfig
from pixel_candidate_gate import validate_pixel_candidate_selection
from voca_s2e_bridge import import_nav_memory_qwen


class FakeVLMClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.inputs = []

    def decide(self, vlm_input):
        self.inputs.append(vlm_input)
        if self.outputs:
            return self.outputs.pop(0)
        raise AssertionError("FakeVLMClient received more calls than expected")


class FakeOpenAIClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []
        self.model = "qwen3-vl-32b-thinking"
        self.temperature = 0.0
        self.max_tokens = 2048
        self.image_max_side = 512
        self.jpeg_quality = 70
        self.extra_payload = {}

    def _post_chat_completion(self, payload):
        self.payloads.append(payload)
        if self.responses:
            return self.responses.pop(0)
        raise AssertionError("FakeOpenAIClient received more calls than expected")

    def _collect_image_refs(self, vlm_input):
        return []

    @staticmethod
    def _response_text_candidates(data):
        message = data["choices"][0]["message"]
        return [value for value in (message.get("content"), message.get("reasoning")) if value]


class FakeOpenAIClientV6Shape:
    def __init__(self):
        self.base_url = "http://qwen.test/v1"
        self.api_key = "secret"
        self.model = "qwen3-vl-32b-thinking"
        self.temperature = 0.0
        self.max_tokens = 2048
        self.timeout_s = 123
        self.image_max_side = 512
        self.jpeg_quality = 70
        self.extra_payload = {}
        self.max_json_retries = 0

    def _collect_image_refs(self, vlm_input):
        return []


class FakeOpenAIClientWithImage(FakeOpenAIClient):
    def _collect_image_refs(self, vlm_input):
        return [("front", "/tmp/fake-front.jpg")]


class FakePointVerifier:
    def __init__(self, result, delay_sec=0.0):
        self.result = dict(result)
        self.delay_sec = float(delay_sec)
        self.calls = []

    def verify(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)
        return dict(self.result)


class FakeStopVerifier(FakePointVerifier):
    pass


class FakeRevisitVerifier(FakePointVerifier):
    pass


class QwenVLMPlannerTests(unittest.TestCase):
    def test_repeated_unresolved_target_approach_enters_detour_cooldown(self):
        modules = import_nav_memory_qwen()
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([modules.schema.make_rotate_output(30)]),
        )
        planner.memory_sidecar = None
        planner.reset("chair")
        execution = {
            "selected_candidate_ref": "px_v00_r1_c1",
            "selected_view_id": 0,
            "selected_angle_deg": 0,
            "selected_point_px": [320, 402],
            "stop_action_seen": False,
            "go_progress": {
                "translation_m": 0.0,
                "collision_count": 0,
                "controller_reached_waypoint": False,
                "no_progress": True,
                "execution_success": False,
            },
        }

        outcome = None
        for frame_index in (10, 20, 30):
            outcome = planner.record_target_approach_execution(
                execution,
                frame_index=frame_index,
            )

        self.assertEqual(planner._target_approach_failure_streak, 3)
        self.assertEqual(planner._target_direct_approach_cooldown, 4)
        self.assertEqual(
            planner._runtime_feedback[-1]["event"],
            "target_approach_unresolved",
        )
        self.assertTrue(outcome["detour_triggered"])

    def test_repeated_collision_free_target_progress_does_not_enter_detour(self):
        modules = import_nav_memory_qwen()
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([modules.schema.make_rotate_output(30)]),
        )
        planner.memory_sidecar = None
        planner.reset("chair")
        execution = {
            "selected_candidate_ref": "px_v00_r1_c1",
            "stop_action_seen": False,
            "go_progress": {
                "translation_m": 0.5,
                "collision_count": 0,
                "controller_reached_waypoint": False,
                "no_progress": False,
                "execution_success": True,
            },
        }

        outcomes = [
            planner.record_target_approach_execution(
                execution,
                frame_index=frame_index,
            )
            for frame_index in (10, 20, 30)
        ]

        self.assertEqual(planner._target_approach_failure_streak, 0)
        self.assertEqual(planner._target_direct_approach_cooldown, 0)
        self.assertTrue(all(item["progress_satisfied"] for item in outcomes))
        self.assertFalse(any(item["detour_triggered"] for item in outcomes))

    def test_guided_schema_and_prompt_expose_verified_memory_ops(self):
        self.assertIn("memory_ops", _NAV_VLM_OUTPUT_GUIDED_JSON["properties"])
        self.assertIn("memory_ops", _NAV_VLM_OUTPUT_GUIDED_JSON["required"])
        self.assertIn("strategic_state", _NAV_VLM_OUTPUT_GUIDED_JSON["properties"])
        self.assertIn("strategic_state", _NAV_VLM_OUTPUT_GUIDED_JSON["required"])
        prompt = _compact_nav_prompt(
            {
                "task": {"target_object": "chair", "target_context": {}},
                "observation": {"image_width": 20, "image_height": 20, "views": []},
                "memory": {
                    "schema_version": "nav_memory_context_v6",
                    "candidate_refs": {"exits": [], "revisits": []},
                },
                "pixel_candidates": {"candidates": []},
            }
        )
        self.assertIn("Mandatory memory_ops policy", prompt)
        self.assertIn("defer_revisit_candidate", prompt)
        self.assertIn("backend independently verifies", prompt)
        self.assertIn("force_leave_room=true", prompt)
        self.assertIn("Target context cues are hypotheses", prompt)

    def test_strategic_state_normalizes_invalid_values_without_promoting_context(self):
        state = normalize_strategic_state(
            {
                "current_room": "spaceship",
                "target_evidence": "probably",
                "navigation_intent": "teleport",
                "room_search_status": "maybe",
                "selected_exit_ref": "",
            },
            action="go",
            reasoning={"short_text": "context suggests a chair may be nearby"},
        )

        self.assertEqual(state["current_room"], "unknown")
        self.assertEqual(state["target_evidence"], "none")
        self.assertEqual(state["navigation_intent"], "search_current_room")
        self.assertEqual(state["room_search_status"], "partial")
        self.assertIsNone(state["selected_exit_ref"])

    def test_force_leave_guard_rejects_generic_go_and_accepts_bound_exit(self):
        modules = import_nav_memory_qwen()
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.memory_sidecar = None
        planner.reset("chair")
        generic = modules.schema.make_go_output(
            view_id=0,
            view_type="front",
            point_px=(320, 420),
            width=640,
            height=480,
            decision_reason="G02_VISIBLE_FLOOR_TOWARD_GOAL",
            goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
            short_text="generic interior floor",
            confidence="medium",
            selected_candidate_ref="px_v00_r0_c1",
        )
        generic["strategic_state"] = {
            "current_room": "living_room",
            "target_evidence": "context_only",
            "navigation_intent": "search_current_room",
            "room_search_status": "partial",
            "selected_exit_ref": None,
            "reason_code": "GENERIC_FLOOR",
        }
        vlm_input = {
            "runtime": {
                "strategic_runtime": {
                    "force_leave_room": True,
                    "force_reason": "generic_floor_go_streak",
                }
            },
            "observation": {"views": [{"view_id": 0, "view_type": "front"}]},
        }

        rejected, validation, warnings = planner._apply_strategic_exit_guard(
            generic,
            vlm_input,
        )
        self.assertEqual(rejected["action"], "request_observation")
        self.assertFalse(validation["passed"])
        self.assertIn("strategic_exit_guard", warnings[0])

        explicit_exit = dict(generic)
        explicit_exit["strategic_state"] = {
            **generic["strategic_state"],
            "navigation_intent": "traverse_gateway",
            "selected_exit_ref": "px_v00_r0_c1",
        }
        accepted, validation, warnings = planner._apply_strategic_exit_guard(
            explicit_exit,
            vlm_input,
        )
        self.assertEqual(accepted["action"], "go")
        self.assertTrue(validation["passed"])
        self.assertEqual(warnings, [])

        topological_exit = dict(generic)
        topological_exit["strategic_state"] = {
            **generic["strategic_state"],
            "navigation_intent": "leave_current_room",
            "selected_exit_ref": "exit_001",
        }
        topological_input = {
            **vlm_input,
            "pixel_candidates": {
                "candidates": [
                    {
                        "candidate_ref": "px_v00_r0_c1",
                        "topological_candidate_ref": "exit_001",
                        "topological_edge_id": "edge_001",
                    }
                ]
            },
        }
        accepted, validation, warnings = planner._apply_strategic_exit_guard(
            topological_exit,
            topological_input,
        )
        self.assertEqual(accepted["action"], "go")
        self.assertTrue(validation["passed"])
        self.assertEqual(validation["reason"], "explicit_exit_strategy_valid")
        self.assertIn("exit_001", validation["selected_exit_aliases"])
        self.assertEqual(warnings, [])

    def test_forced_exit_failures_escalate_to_full_sweep_without_immediate_repeat(self):
        modules = import_nav_memory_qwen()
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.memory_sidecar = None
        planner.reset("chair")
        generic = modules.schema.make_go_output(
            view_id=0,
            view_type="front",
            point_px=(320, 420),
            width=640,
            height=480,
            decision_reason="G02_VISIBLE_FLOOR_TOWARD_GOAL",
            goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
            short_text="generic interior floor",
            confidence="medium",
            selected_candidate_ref="px_v00_r0_c1",
        )
        generic["strategic_state"] = {
            "current_room": "bedroom",
            "target_evidence": "none",
            "navigation_intent": "search_current_room",
            "room_search_status": "exhausted",
            "selected_exit_ref": None,
            "reason_code": "GENERIC_FLOOR",
        }

        def input_for(headings):
            return {
                "runtime": {
                    "strategic_runtime": {
                        "force_leave_room": True,
                        "force_reason": "room_search_exhausted",
                    }
                },
                "observation": {
                    "views": [
                        {
                            "view_id": index,
                            "view_type": "sweep_{:02d}".format(index),
                            "relative_heading_deg": heading,
                        }
                        for index, heading in enumerate(headings)
                    ]
                },
            }

        first, first_validation, _ = planner._apply_strategic_exit_guard(
            generic,
            input_for([0]),
        )
        self.assertEqual(first["action"], "request_observation")
        self.assertEqual(first["observation_request"]["mode"], "directed_sweep")
        self.assertEqual(first_validation["forced_exit_failure_streak"], 1)

        second, second_validation, _ = planner._apply_strategic_exit_guard(
            generic,
            input_for([-60, -30, 0, 30, 60]),
        )
        self.assertEqual(second["action"], "request_observation")
        self.assertEqual(second["observation_request"]["mode"], "full_sweep")
        self.assertEqual(second_validation["forced_exit_failure_streak"], 2)
        self.assertEqual(planner._forced_exit_full_sweep_count, 1)

        third, third_validation, _ = planner._apply_strategic_exit_guard(
            generic,
            input_for([-180, -135, -90, -45, 0, 45, 90, 135]),
        )
        self.assertEqual(third["action"], "rotate")
        self.assertEqual(
            third_validation["recovery_action"],
            "rotate.search_visible_gateway",
        )
        self.assertEqual(planner._forced_exit_full_sweep_count, 1)

    def test_coarse_direction_guard_rejects_unsupported_opposite_generic_go(self):
        modules = import_nav_memory_qwen()
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.memory_sidecar = None
        planner.reset("chair")
        output = modules.schema.make_go_output(
            view_id=0,
            view_type="front",
            point_px=(320, 420),
            width=640,
            height=480,
            decision_reason="G02_VISIBLE_FLOOR_TOWARD_GOAL",
            goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
            short_text="generic interior floor",
            confidence="medium",
            selected_candidate_ref="px_front",
        )
        output["strategic_state"] = {
            "current_room": "living_room",
            "target_evidence": "none",
            "navigation_intent": "search_current_room",
            "room_search_status": "partial",
            "selected_exit_ref": None,
            "reason_code": "GENERIC_FLOOR",
        }
        vlm_input = {
            "task": {
                "coarse_goal": {
                    "goal_geometry_available": True,
                    "source": "s2e_backbone",
                    "uncertainty": "medium",
                    "distance_m": 4.0,
                    "relative_bearing_deg": 170.0,
                }
            },
            "runtime": {"strategic_runtime": {"force_leave_room": False}},
            "pixel_candidates": {
                "candidates": [
                    {
                        "candidate_ref": "px_front",
                        "bearing_deg_robot": 0.0,
                        "point_norm": [0.5, 0.875],
                    }
                ]
            },
        }

        guarded, validation, warnings = planner._apply_coarse_goal_direction_guard(
            output,
            vlm_input,
        )

        self.assertEqual(guarded["action"], "request_observation")
        self.assertFalse(validation["passed"])
        self.assertEqual(
            validation["classification"],
            "unsupported_generic_go_away_from_coarse_goal",
        )
        self.assertEqual(
            guarded["observation_request"]["yaw_offsets_deg"],
            [110, 140, 170, -160, -130],
        )
        self.assertIn("coarse_goal_direction_guard", warnings[0])
        self.assertEqual(planner._coarse_direction_rejection_count, 1)
        self.assertTrue(validation["force_leave_armed_next_cycle"])
        self.assertEqual(
            guarded["strategic_state"]["navigation_intent"],
            "escape_deadlock",
        )
        self.assertTrue(planner._strategic_runtime_snapshot()["force_leave_room"])

    def test_coarse_direction_guard_allows_explicit_exit_detour(self):
        modules = import_nav_memory_qwen()
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.memory_sidecar = None
        planner.reset("chair")
        output = modules.schema.make_go_output(
            view_id=0,
            view_type="front",
            point_px=(320, 420),
            width=640,
            height=480,
            decision_reason="G04_VISIBLE_GATEWAY",
            goal_reason="F04_TEMPORARY_DETOUR_TO_EXIT_ROOM",
            short_text="leave through the opposite doorway",
            confidence="medium",
            selected_candidate_ref="px_exit",
        )
        output["strategic_state"] = {
            "current_room": "living_room",
            "target_evidence": "none",
            "navigation_intent": "traverse_gateway",
            "room_search_status": "exhausted",
            "selected_exit_ref": "px_exit",
            "reason_code": "EXIT_DETOUR",
        }
        vlm_input = {
            "task": {
                "coarse_goal": {
                    "goal_geometry_available": True,
                    "source": "s2e_backbone",
                    "uncertainty": "medium",
                    "distance_m": 8.0,
                    "relative_bearing_deg": 170.0,
                }
            },
            "runtime": {"strategic_runtime": {"force_leave_room": False}},
            "pixel_candidates": {
                "candidates": [
                    {
                        "candidate_ref": "px_exit",
                        "bearing_deg_robot": 0.0,
                        "point_norm": [0.5, 0.875],
                    }
                ]
            },
        }

        guarded, validation, warnings = planner._apply_coarse_goal_direction_guard(
            output,
            vlm_input,
        )

        self.assertEqual(guarded["action"], "go")
        self.assertTrue(validation["passed"])
        self.assertEqual(validation["classification"], "explicit_detour_authorized")
        self.assertEqual(warnings, [])
        self.assertEqual(planner._coarse_direction_detour_exemption_count, 1)

    def test_coarse_direction_guard_allows_local_search_inside_stop_region(self):
        modules = import_nav_memory_qwen()
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.memory_sidecar = None
        planner.reset("chair")
        output = modules.schema.make_go_output(
            view_id=0,
            view_type="front",
            point_px=(320, 420),
            width=640,
            height=480,
            decision_reason="G02_VISIBLE_FLOOR_TOWARD_GOAL",
            goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
            short_text="search locally",
            confidence="medium",
            selected_candidate_ref="px_front",
        )
        vlm_input = {
            "task": {
                "coarse_goal": {
                    "goal_geometry_available": True,
                    "source": "s2e_backbone",
                    "uncertainty": "medium",
                    "distance_m": 2.0,
                    "relative_bearing_deg": 170.0,
                }
            }
        }

        guarded, validation, warnings = planner._apply_coarse_goal_direction_guard(
            output,
            vlm_input,
        )

        self.assertEqual(guarded["action"], "go")
        self.assertEqual(validation["classification"], "inside_local_search_region")
        self.assertEqual(warnings, [])

    def test_full_sweep_uses_verified_memory_backtrack_after_repeated_failures(self):
        modules = import_nav_memory_qwen()
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.memory_sidecar = None
        planner.reset("toilet")
        planner._forced_exit_failure_streak = 4
        rotate = modules.schema.make_rotate_output(
            45,
            reason="R02_NO_VISIBLE_NAVIGABLE_FLOOR",
            confidence="low",
        )
        rotate["strategic_state"] = {
            "current_room": "living_room",
            "target_evidence": "none",
            "navigation_intent": "search_current_room",
            "room_search_status": "exhausted",
            "selected_exit_ref": None,
            "reason_code": "NO_EXIT",
        }
        headings = [-180, -135, -90, -45, 0, 45, 90, 135]
        backtrack_candidate = {
            "candidate_ref": "px_v03_r1_c1",
            "view_id": 3,
            "view_type_hint": "left",
            "point_px": [320, 402],
            "point_norm": [0.5, 0.84],
            "avoid": False,
            "score": 0.8,
            "topological_candidate_ref": "exit_003",
            "topological_edge_id": "edge_back",
            "topological_relation_type": "backtrack",
            "pixelnav_feasibility": {"policy_feasibility_score": 0.9},
        }
        alternate_backtrack_candidate = {
            **dict(backtrack_candidate),
            "candidate_ref": "px_v04_r1_c1",
            "view_id": 4,
            "view_type_hint": "front",
            "score": 0.7,
            "pixelnav_feasibility": {"policy_feasibility_score": 0.8},
        }
        novel_candidate = {
            "candidate_ref": "px_v05_r1_c1",
            "view_id": 5,
            "view_type_hint": "sweep_05",
            "relative_heading_deg": 45,
            "point_px": [320, 402],
            "point_norm": [0.5, 0.84],
            "avoid": False,
            "score": 1.0,
            "visual_evidence": {
                "support_score": 0.99,
                "hard_reject": False,
            },
            "pixelnav_feasibility": {"policy_feasibility_score": 1.0},
        }
        vlm_input = {
            "runtime": {
                "strategic_runtime": {
                    "force_leave_room": True,
                    "force_reason": "generic_floor_go_streak",
                }
            },
            "observation": {
                "image_width": 640,
                "image_height": 480,
                "views": [
                    {
                        "view_id": index,
                        "view_type": "sweep_{:02d}".format(index),
                        "relative_heading_deg": heading,
                    }
                    for index, heading in enumerate(headings)
                ],
            },
            "pixel_candidates": {
                "candidates": [
                    backtrack_candidate,
                    alternate_backtrack_candidate,
                    novel_candidate,
                ]
            },
        }

        output, validation, warnings = planner._apply_strategic_exit_guard(
            rotate,
            vlm_input,
        )

        self.assertEqual(output["action"], "go")
        self.assertEqual(output["selected_candidate_ref"], "px_v03_r1_c1")
        self.assertEqual(
            output["strategic_state"]["navigation_intent"],
            "escape_deadlock",
        )
        self.assertTrue(validation["passed"])
        self.assertTrue(validation["backend_memory_backtrack"]["triggered"])
        self.assertIn("verified_memory_backtrack_override", warnings[0])
        self.assertEqual(planner._forced_exit_memory_backtrack_count, 1)

        verification = planner._verify_selected_point_if_needed(
            safe_output=output,
            pixel_candidate_validation={
                "passed": True,
                "candidate": backtrack_candidate,
            },
            pano_images=[np.zeros((480, 640, 3), dtype=np.uint8) for _ in headings],
        )
        self.assertTrue(verification["triggered"])
        self.assertFalse(verification["passed"])
        self.assertEqual(
            verification["reason"],
            "memory_backtrack_floor_verifier_unavailable_fail_closed",
        )

        alternate = planner._forced_exit_memory_backtrack_output(
            vlm_input,
            rotate["strategic_state"],
            excluded_candidate_refs=["px_v03_r1_c1"],
        )
        self.assertEqual(
            alternate["selected_candidate_ref"],
            "px_v04_r1_c1",
        )

    def test_full_sweep_uses_novel_safe_sector_when_backtrack_is_unavailable(self):
        modules = import_nav_memory_qwen()
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.memory_sidecar = None
        planner.reset("toilet")
        planner._forced_exit_failure_streak = 4
        rotate = modules.schema.make_rotate_output(
            45,
            reason="R02_NO_VISIBLE_NAVIGABLE_FLOOR",
            confidence="low",
        )
        rotate["strategic_state"] = {
            "current_room": "living_room",
            "target_evidence": "none",
            "navigation_intent": "search_current_room",
            "room_search_status": "exhausted",
            "selected_exit_ref": None,
            "reason_code": "NO_EXIT",
        }
        headings = [-180, -135, -90, -45, 0, 45, 90, 135]
        candidate = {
            "candidate_ref": "px_v05_r1_c1",
            "view_id": 5,
            "view_type_hint": "sweep_05",
            "relative_heading_deg": 45,
            "point_px": [336, 404],
            "point_norm": [0.525, 0.842],
            "avoid": False,
            "score": 0.82,
            "visual_evidence": {
                "support_score": 0.88,
                "hard_reject": False,
            },
            "pixelnav_feasibility": {"policy_feasibility_score": 0.91},
        }
        alternate_candidate = {
            **dict(candidate),
            "candidate_ref": "px_v06_r1_c1",
            "view_id": 6,
            "view_type_hint": "sweep_06",
            "relative_heading_deg": 90,
            "point_px": [320, 402],
            "score": 0.76,
            "visual_evidence": {
                "support_score": 0.79,
                "hard_reject": False,
            },
            "pixelnav_feasibility": {"policy_feasibility_score": 0.82},
        }
        vlm_input = {
            "runtime": {
                "strategic_runtime": {
                    "force_leave_room": True,
                    "force_reason": "room_search_exhausted",
                }
            },
            "observation": {
                "image_width": 640,
                "image_height": 480,
                "views": [
                    {
                        "view_id": index,
                        "view_type": "sweep_{:02d}".format(index),
                        "relative_heading_deg": heading,
                    }
                    for index, heading in enumerate(headings)
                ],
            },
            "pixel_candidates": {
                "candidates": [candidate, alternate_candidate]
            },
        }

        with patch.dict(
            os.environ,
            {"VOCA_FORCE_LEAVE_NOVEL_SECTOR_AFTER": "4"},
        ):
            output, validation, warnings = planner._apply_strategic_exit_guard(
                rotate,
                vlm_input,
            )

        self.assertEqual(output["action"], "go")
        self.assertEqual(output["selected_candidate_ref"], "px_v05_r1_c1")
        self.assertEqual(
            output["strategic_state"]["navigation_intent"],
            "escape_deadlock",
        )
        self.assertTrue(validation["passed"])
        self.assertTrue(validation["backend_novel_sector_escape"]["triggered"])
        self.assertIn("novel_safe_sector_override", warnings[0])
        self.assertEqual(planner._forced_exit_novel_sector_count, 1)

        alternate = planner._forced_exit_novel_sector_output(
            vlm_input,
            rotate["strategic_state"],
            excluded_candidate_refs=["px_v05_r1_c1"],
        )
        self.assertEqual(
            alternate["selected_candidate_ref"],
            "px_v06_r1_c1",
        )

    def test_forced_exit_streak_commits_only_after_go_execution(self):
        modules = import_nav_memory_qwen()
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.memory_sidecar = None
        planner.reset("chair")
        planner._forced_exit_failure_streak = 3
        output = modules.schema.make_go_output(
            view_id=0,
            view_type="front",
            point_px=(320, 402),
            width=640,
            height=480,
            decision_reason="G03_EXIT_ROUTE",
            goal_reason="F03_EXIT_ROUTE",
            short_text="traverse the visible doorway",
            confidence="high",
            selected_candidate_ref="exit_01",
        )
        output["strategic_state"] = {
            "current_room": "bedroom",
            "target_evidence": "none",
            "navigation_intent": "traverse_gateway",
            "room_search_status": "exhausted",
            "selected_exit_ref": "exit_01",
            "reason_code": "VISIBLE_GATEWAY",
        }
        vlm_input = {
            "runtime": {
                "strategic_runtime": {
                    "force_leave_room": True,
                    "force_reason": "room_search_exhausted",
                }
            }
        }

        accepted, validation, _ = planner._apply_strategic_exit_guard(
            output,
            vlm_input,
        )
        self.assertTrue(validation["passed"])
        self.assertEqual(planner._forced_exit_failure_streak, 3)

        decision = {
            "strategic_validation": validation,
            "strategic_state": accepted["strategic_state"],
            "vlm_output": accepted,
        }
        failed = planner.record_strategic_go_execution(
            decision,
            {"strategic_progress": False},
        )
        self.assertTrue(failed["triggered"])
        self.assertFalse(failed["passed"])
        self.assertEqual(planner._forced_exit_failure_streak, 4)

        passed = planner.record_strategic_go_execution(
            decision,
            {"strategic_progress": True},
        )
        self.assertTrue(passed["passed"])
        self.assertEqual(planner._forced_exit_failure_streak, 0)

    def test_candidate_canonicalization_preserves_backend_recovery_authority(self):
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        canonical = {
            "action": "go",
            "selected_candidate_ref": "px_v05_r1_c1",
            "memory_ops": [],
        }
        recovery = {
            "action": "go",
            "selected_candidate_ref": "px_v05_r1_c1",
            "backend_novel_sector_escape": {
                "triggered": True,
                "selected_candidate_ref": "px_v05_r1_c1",
            },
            "strategic_state": {
                "navigation_intent": "escape_deadlock",
            },
            "memory_ops": [{"op": "defer_revisit_candidate"}],
        }

        merged = planner._preserve_backend_recovery_metadata(
            canonical,
            recovery,
        )

        self.assertTrue(merged["backend_novel_sector_escape"]["triggered"])
        self.assertEqual(
            merged["strategic_state"]["navigation_intent"],
            "escape_deadlock",
        )
        self.assertEqual(merged["memory_ops"], recovery["memory_ops"])

    def test_normal_goal_seek_rejects_unjustified_backtrack_selection(self):
        modules = import_nav_memory_qwen()
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.memory_sidecar = None
        planner.reset("chair")
        output = modules.schema.make_go_output(
            view_id=0,
            view_type="front",
            point_px=(320, 402),
            width=640,
            height=480,
            decision_reason="G02_VISIBLE_FLOOR_TOWARD_GOAL",
            goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
            short_text="visible floor",
            confidence="medium",
            selected_candidate_ref="px_v00_r1_c1",
        )
        output["strategic_state"] = {
            "current_room": "living_room",
            "target_evidence": "context_only",
            "navigation_intent": "search_current_room",
            "room_search_status": "partial",
            "selected_exit_ref": None,
            "reason_code": "SAFE_INTERIOR_PROGRESS",
        }
        vlm_input = {
            "runtime": {
                "strategic_runtime": {
                    "force_leave_room": False,
                    "force_reason": "",
                }
            },
            "pixel_candidates": {
                "candidates": [
                    {
                        "candidate_ref": "px_v00_r1_c1",
                        "topological_edge_id": "e_reverse",
                        "topological_relation_type": "backtrack",
                    }
                ]
            },
        }

        guarded, validation, warnings = planner._apply_strategic_exit_guard(
            output,
            vlm_input,
        )

        self.assertEqual(guarded["action"], "request_observation")
        self.assertEqual(
            guarded["observation_request"]["yaw_offsets_deg"],
            [120.0, 150.0, 180.0, -150.0, -120.0],
        )
        self.assertFalse(validation["passed"])
        self.assertEqual(
            validation["reason"],
            "normal_goal_seek_backtrack_requires_recovery_intent",
        )
        self.assertIn("normal_goal_seek_backtrack_rejected", warnings[0])
        self.assertEqual(
            planner._normal_goal_seek_backtrack_rejection_count,
            1,
        )

        output["strategic_state"] = {
            **output["strategic_state"],
            "navigation_intent": "revisit_promising_place",
        }
        allowed, allowed_validation, _ = planner._apply_strategic_exit_guard(
            output,
            vlm_input,
        )
        self.assertEqual(allowed["action"], "go")
        self.assertTrue(allowed_validation["passed"])

    def test_strategic_exit_guard_repairs_verified_topological_alias(self):
        modules = import_nav_memory_qwen()
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.memory_sidecar = None
        planner.reset("toilet")
        output = modules.schema.make_go_output(
            view_id=2,
            view_type="right",
            point_px=(320, 402),
            width=640,
            height=480,
            decision_reason="G03_VISIBLE_GATEWAY_EXIT",
            goal_reason="F03_VISIBLE_GATEWAY_EXIT",
            short_text="leave through the visible doorway",
            confidence="high",
            selected_candidate_ref="px_v02_r1_c1",
        )
        output["strategic_state"] = {
            "current_room": "bedroom",
            "target_evidence": "context_only",
            "navigation_intent": "leave_current_room",
            "room_search_status": "exhausted",
            "selected_exit_ref": "exit_001",
            "reason_code": "GATEWAY_EXIT",
        }
        vlm_input = {
            "runtime": {
                "strategic_runtime": {
                    "force_leave_room": True,
                    "force_reason": "same_room_cycle_limit",
                }
            },
            "pixel_candidates": {
                "candidates": [
                    {
                        "candidate_ref": "px_v02_r1_c1",
                        "topological_candidate_ref": "exit_001",
                        "topological_relation_type": None,
                    }
                ]
            },
        }

        accepted, validation, warnings = planner._apply_strategic_exit_guard(
            output,
            vlm_input,
        )

        self.assertEqual(accepted["action"], "go")
        self.assertTrue(validation["passed"])
        self.assertEqual(
            accepted["strategic_state"]["selected_exit_ref"],
            "px_v02_r1_c1",
        )
        self.assertEqual(
            validation["selected_exit_ref_alias_repair"]["from"],
            "exit_001",
        )
        self.assertEqual(warnings, [])
        self.assertEqual(planner._strategic_exit_alias_repair_count, 1)

    def test_forced_exit_missing_ref_uses_visual_repair_and_stays_fail_closed(self):
        modules = import_nav_memory_qwen()
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.memory_sidecar = None
        planner.reset("toilet")
        output = modules.schema.make_go_output(
            view_id=0,
            view_type="front",
            point_px=(160, 465),
            width=640,
            height=480,
            decision_reason="G03_VISIBLE_GATEWAY_EXIT",
            goal_reason="F03_VISIBLE_GATEWAY_EXIT",
            short_text="leave through the visible doorway",
            confidence="medium",
            selected_candidate_ref="px_v00_r0_c0",
        )
        output["strategic_state"] = {
            "current_room": "hallway",
            "target_evidence": "context_only",
            "navigation_intent": "leave_current_room",
            "room_search_status": "partial",
            "selected_exit_ref": None,
            "reason_code": "SAFE_EXIT_TO_NEXT_ROOM",
        }
        candidate = {
            "candidate_ref": "px_v00_r0_c0",
            "view_id": 0,
            "view_type_hint": "front",
            "point_px": [160, 465],
            "point_norm": [0.25, 0.97],
            "avoid": False,
        }
        vlm_input = {
            "runtime": {
                "strategic_runtime": {
                    "force_leave_room": True,
                    "force_reason": "generic_floor_go_streak",
                }
            },
            "pixel_candidates": {"candidates": [candidate]},
        }

        accepted, validation, warnings = planner._apply_strategic_exit_guard(
            output,
            vlm_input,
        )

        self.assertEqual(accepted["action"], "go")
        self.assertTrue(validation["passed"])
        self.assertEqual(
            accepted["strategic_state"]["selected_exit_ref"],
            "px_v00_r0_c0",
        )
        self.assertTrue(validation["selected_exit_ref_visual_repair"]["triggered"])
        self.assertEqual(planner._strategic_visual_exit_ref_repair_count, 1)
        self.assertEqual(warnings, [])

        verification = planner._verify_selected_point_if_needed(
            safe_output=accepted,
            pixel_candidate_validation={"passed": True, "candidate": candidate},
            pano_images=[np.zeros((480, 640, 3), dtype=np.uint8)],
        )
        self.assertTrue(verification["triggered"])
        self.assertFalse(verification["passed"])
        self.assertEqual(
            verification["reason"],
            "doorway_floor_verifier_unavailable_fail_closed",
        )

    def test_exit_floor_relaxation_keeps_walls_blocked_but_allows_verified_progress(self):
        modules = import_nav_memory_qwen()
        output = modules.schema.make_go_output(
            view_id=0,
            view_type="front",
            point_px=(320, 402),
            width=640,
            height=480,
            decision_reason="G03_VISIBLE_GATEWAY_EXIT",
            goal_reason="F03_VISIBLE_GATEWAY_EXIT",
            short_text="continue through the visible exit",
            confidence="medium",
            selected_candidate_ref="px_v00_r1_c1",
        )
        output["strategic_state"] = {
            "current_room": "bedroom",
            "target_evidence": "context_only",
            "navigation_intent": "leave_current_room",
            "room_search_status": "exhausted",
            "selected_exit_ref": "exit_001",
            "reason_code": "GATEWAY_EXIT",
        }
        candidate = {
            "candidate_ref": "px_v00_r1_c1",
            "view_id": 0,
            "view_type_hint": "front",
            "point_px": [320, 402],
            "topological_candidate_ref": "exit_001",
            "visual_evidence": {
                "available": True,
                "hard_reject": False,
                "support_score": 0.82,
            },
        }
        floor_verifier = FakePointVerifier(
            {
                "triggered": True,
                "passed": False,
                "valid": False,
                "surface": "floor",
                "confidence": "high",
                "leads_out_of_current_room": False,
                "reason": "hallway floor, not exactly on threshold",
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([]),
            point_verifier=floor_verifier,
        )
        planner.memory_sidecar = None
        planner.reset("toilet")

        relaxed = planner._verify_selected_point_if_needed(
            safe_output=output,
            pixel_candidate_validation={"passed": True, "candidate": candidate},
            pano_images=[np.zeros((480, 640, 3), dtype=np.uint8)],
        )

        self.assertTrue(relaxed["passed"])
        self.assertTrue(relaxed["semantic_exit_floor_relaxation"]["triggered"])

        planner.point_verifier = FakePointVerifier(
            {
                "triggered": True,
                "passed": False,
                "valid": False,
                "surface": "wall",
                "confidence": "high",
                "reason": "marked point is on a wall",
            }
        )
        blocked = planner._verify_selected_point_if_needed(
            safe_output=output,
            pixel_candidate_validation={"passed": True, "candidate": candidate},
            pano_images=[np.zeros((480, 640, 3), dtype=np.uint8)],
        )

        self.assertFalse(blocked["passed"])
        self.assertNotIn("semantic_exit_floor_relaxation", blocked)

    def test_generic_floor_go_streak_activates_force_leave_until_exit_intent(self):
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.memory_sidecar = None
        planner.reset("chair")
        generic = {
            "action": "go",
            "strategic_state": {
                "current_room": "living_room",
                "target_evidence": "context_only",
                "navigation_intent": "search_current_room",
                "room_search_status": "partial",
                "selected_exit_ref": None,
                "reason_code": "GENERIC_FLOOR",
            },
        }
        for frame_index in range(3):
            planner._record_strategic_state(generic, frame_index=frame_index)

        snapshot = planner._strategic_runtime_snapshot()
        self.assertEqual(snapshot["generic_floor_go_streak"], 3)
        self.assertTrue(snapshot["force_leave_room"])

        explicit_exit = {
            "action": "go",
            "strategic_state": {
                **generic["strategic_state"],
                "navigation_intent": "traverse_gateway",
                "selected_exit_ref": "px_v00_r0_c1",
            },
        }
        planner._record_strategic_state(explicit_exit, frame_index=4)
        self.assertEqual(
            planner._strategic_runtime_snapshot()["generic_floor_go_streak"],
            0,
        )

    def test_same_room_stagnation_resets_after_spatial_progress(self):
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.memory_sidecar = None
        planner.reset("toilet")
        rotate = {
            "action": "rotate",
            "strategic_state": {
                "current_room": "bathroom",
                "target_evidence": "none",
                "navigation_intent": "search_current_room",
                "room_search_status": "partial",
                "selected_exit_ref": None,
                "reason_code": "SEARCH",
            },
        }

        planner.update_runtime_context(position_xyz=[0.0, 1.0, 0.0])
        planner._record_strategic_state(rotate, frame_index=1)
        planner._record_strategic_state(rotate, frame_index=2)
        self.assertEqual(planner._same_room_cycles, 2)

        planner.update_runtime_context(position_xyz=[0.4, 1.0, 0.0])
        state = planner._record_strategic_state(rotate, frame_index=3)

        self.assertEqual(state["current_room"], "bathroom")
        self.assertEqual(planner._same_room_cycles, 1)
        self.assertTrue(planner._strategic_state_history[-1]["spatial_progress"])
        self.assertAlmostEqual(
            planner._strategic_state_history[-1]["movement_since_last_cycle_m"],
            0.4,
            places=3,
        )

    def test_missing_revisit_verdict_is_explicitly_deferred(self):
        output, validation, warnings = enforce_memory_verdict_contract(
            {"action": "go"},
            {
                "memory": {
                    "candidate_refs": {
                        "revisits": [
                            {
                                "candidate_ref": "revisit_001",
                                "spatial_plausibility": {"accepted": True},
                            }
                        ]
                    }
                }
            },
        )

        self.assertFalse(validation["passed"])
        self.assertTrue(validation["backend_inserted_defer"])
        self.assertEqual(output["memory_ops"][0]["op"], "defer_revisit_candidate")
        self.assertEqual(output["memory_ops"][0]["candidate_ref"], "revisit_001")
        self.assertIn("memory_verdict_missing_backend_deferred", warnings)

    def test_empty_revisit_set_normalizes_memory_ops_without_warning(self):
        output, validation, warnings = enforce_memory_verdict_contract(
            {"action": "rotate"},
            {"memory": {"candidate_refs": {"revisits": []}}},
        )

        self.assertTrue(validation["passed"])
        self.assertEqual(output["memory_ops"], [])
        self.assertEqual(warnings, [])

    def test_dedicated_revisit_verifier_replaces_backend_defer(self):
        verifier = FakeRevisitVerifier(
            {
                "triggered": True,
                "valid": True,
                "passed": True,
                "verdict": "same_place",
                "latency_sec": 0.01,
                "memory_op": {
                    "op": "confirm_revisit_node",
                    "candidate_ref": "revisit_001",
                    "confidence": 0.9,
                    "reason": "layout and openings match",
                    "source": "dedicated_revisit_verifier",
                },
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([]),
            revisit_verifier=verifier,
        )
        safe_output, validation, _warnings = enforce_memory_verdict_contract(
            {"action": "rotate"},
            {"memory": {"candidate_refs": {"revisits": [{"candidate_ref": "revisit_001"}]}}},
        )

        result = planner._verify_revisit_if_needed(
            safe_output=safe_output,
            vlm_input={},
            contract_validation=validation,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(safe_output["memory_ops"][0]["op"], "confirm_revisit_node")
        self.assertFalse(validation["backend_inserted_defer"])
        self.assertTrue(validation["dedicated_verifier_triggered"])
        self.assertEqual(planner.revisit_verification_calls, 1)

    def test_revisit_verifier_maps_same_place_to_confirm_only_after_spatial_gate(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "verdict": "same_place",
                                "confidence": "high",
                                "reason": "same doorway and wall layout",
                            }
                        )
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        base = FakeOpenAIClient([response])
        modules = import_nav_memory_qwen()
        verifier = QwenRevisitVerifier(
            CompactQwenNavVLMClient(base, modules),
            modules,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "current.jpg"
            memory_path = Path(tmpdir) / "memory.jpg"
            cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))
            cv2.imwrite(str(memory_path), np.full((32, 32, 3), 8, dtype=np.uint8))
            vlm_input = {
                "metadata": {"raw_observation_images": [str(image_path)]},
                "memory": {
                    "place_recognition": {
                        "revisit_candidates": [
                            {
                                "candidate_ref": "revisit_001",
                                "candidate_image_ref": str(memory_path),
                                "visual_retrieval_score": 0.99,
                                "spatial_plausibility": {
                                    "accepted": True,
                                    "reason": "short composed graph path",
                                },
                            }
                        ]
                    }
                },
            }

            result = verifier.verify(vlm_input=vlm_input, candidate_ref="revisit_001")

        self.assertTrue(result["passed"])
        self.assertEqual(result["memory_op"]["op"], "confirm_revisit_node")
        self.assertEqual(len(base.payloads), 1)

    def test_revisit_verifier_conservatively_defers_truncated_response(self):
        response = {
            "choices": [
                {
                    "message": {"content": ""},
                    "finish_reason": "length",
                }
            ]
        }
        base = FakeOpenAIClient([response])
        modules = import_nav_memory_qwen()
        verifier = QwenRevisitVerifier(
            CompactQwenNavVLMClient(base, modules),
            modules,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "current.jpg"
            memory_path = Path(tmpdir) / "memory.jpg"
            cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))
            cv2.imwrite(str(memory_path), np.full((32, 32, 3), 8, dtype=np.uint8))
            result = verifier.verify(
                vlm_input={
                    "metadata": {"raw_observation_images": [str(image_path)]},
                    "memory": {
                        "place_recognition": {
                            "revisit_candidates": [
                                {
                                    "candidate_ref": "revisit_001",
                                    "candidate_image_ref": str(memory_path),
                                    "spatial_plausibility": {
                                        "accepted": True,
                                        "reason": "short graph path",
                                    },
                                }
                            ]
                        }
                    },
                },
                candidate_ref="revisit_001",
            )

        self.assertTrue(result["valid"])
        self.assertTrue(result["passed"])
        self.assertEqual(result["verdict"], "uncertain")
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(result["memory_op"]["op"], "defer_revisit_candidate")
        self.assertNotIn("error", result)

    def test_unverified_stop_becomes_observation_request(self):
        modules = import_nav_memory_qwen()
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": False,
                "target_visible": True,
                "target_match": False,
                "close_enough": False,
                "confidence": "low",
                "reason": "possible lookalike",
                "latency_sec": 0.01,
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([modules.schema.make_stop_output()]),
            stop_verifier=verifier,
            stop_confirmations_required=1,
        )
        planner.reset("chair")
        planner.make_plan_from_views(
            [np.zeros((20, 20, 3), dtype=np.uint8)],
            [0],
            call_type="qwen_action_loop",
        )

        self.assertEqual(planner.last_decision["action"], "request_observation")
        self.assertFalse(planner.last_decision["stop_verification"]["passed"])
        self.assertEqual(planner.stop_verification_rejection_count, 1)

    def test_coarse_goal_guard_rejects_distant_same_category_instance(self):
        modules = import_nav_memory_qwen()
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": True,
                "target_visible": True,
                "target_match": True,
                "target_evidence_passed": True,
                "close_enough": True,
                "confidence": "high",
                "target_bbox_height": 0.5,
                "target_bbox_area": 0.15,
                "target_center_x": 0.5,
                "target_center_guard_satisfied": True,
                "identity_critic": {
                    "triggered": True,
                    "passed": True,
                    "exact_target": True,
                    "confidence": "high",
                },
                "reason": "a real chair is visible nearby",
                "latency_sec": 0.01,
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([modules.schema.make_rotate_output(30)]),
            stop_verifier=verifier,
        )
        planner.memory_sidecar = None
        planner.reset("chair")
        planner.update_runtime_context(
            frame_index=10,
            position_xyz=[0.0, 0.0, 0.0],
            heading_rad=0.0,
            spatial_coarse_goal={
                "source": "s2e_backbone",
                "type": "map_waypoint",
                "map_xy": [0.0, 9.0],
                "relative_bearing_deg": 90.0,
                "distance_m": 9.0,
                "distance_range_m": [7.5, 10.5],
                "uncertainty": "medium",
            },
        )
        image = np.random.default_rng(1167).integers(
            0,
            256,
            size=(100, 200, 3),
            dtype=np.uint8,
        )

        planner.make_plan_from_views(
            [image],
            [0],
            call_type="coarse_distractor_guard",
        )

        verification = planner.last_decision["proactive_stop_probe_result"]
        guard = verification["coarse_goal_consistency_guard"]
        self.assertEqual(planner.last_decision["action"], "rotate")
        self.assertFalse(verification["target_evidence_passed"])
        self.assertFalse(guard["passed"])
        self.assertEqual(
            guard["classification"],
            "same_category_instance_outside_coarse_goal_region",
        )
        self.assertEqual(planner._target_lock_remaining_cycles, 0)
        self.assertEqual(
            planner._runtime_feedback[-1]["event"],
            "coarse_inconsistent_target_instance",
        )

    def test_coarse_goal_guard_rejects_directionally_inconsistent_instance(self):
        modules = import_nav_memory_qwen()
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": True,
                "target_visible": True,
                "target_match": True,
                "target_evidence_passed": True,
                "close_enough": False,
                "confidence": "high",
                "target_view_index": 0,
                "target_bbox_height": 0.2,
                "target_bbox_area": 0.02,
                "target_center_x": 0.45,
                "target_center_guard_satisfied": True,
                "identity_critic": {
                    "triggered": True,
                    "passed": True,
                    "exact_target": True,
                    "confidence": "high",
                },
                "reason": "a real chair is visible in front",
                "latency_sec": 0.01,
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([modules.schema.make_rotate_output(30)]),
            stop_verifier=verifier,
        )
        planner.memory_sidecar = None
        planner.reset("chair")
        planner.update_runtime_context(
            frame_index=10,
            position_xyz=[0.0, 0.0, 0.0],
            heading_rad=0.0,
            spatial_coarse_goal={
                "source": "s2e_backbone",
                "type": "map_waypoint",
                "map_xy": [3.0, 1.0],
                "relative_bearing_deg": 110.0,
                "distance_m": 3.5,
                "distance_range_m": [2.5, 4.5],
                "uncertainty": "medium",
            },
        )
        image = np.random.default_rng(1168).integers(
            0,
            256,
            size=(100, 200, 3),
            dtype=np.uint8,
        )

        planner.make_plan_from_views(
            [image],
            [0],
            call_type="coarse_directional_distractor_guard",
        )

        verification = planner.last_decision["proactive_stop_probe_result"]
        guard = verification["coarse_goal_consistency_guard"]
        self.assertEqual(planner.last_decision["action"], "rotate")
        self.assertFalse(verification["target_evidence_passed"])
        self.assertFalse(guard["passed"])
        self.assertEqual(
            guard["classification"],
            "same_category_instance_directionally_inconsistent",
        )
        self.assertAlmostEqual(guard["target_bearing_robot_deg"], -4.5)
        self.assertAlmostEqual(guard["target_alignment_error_deg"], 114.5)
        self.assertEqual(
            planner._runtime_feedback[-1]["event"],
            "coarse_direction_inconsistent_target_instance",
        )

    def test_proactive_stop_probe_requires_approach_motion_before_stopping(self):
        modules = import_nav_memory_qwen()
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": True,
                "target_visible": True,
                "target_match": True,
                "close_enough": True,
                "confidence": "high",
                "target_bbox_height": 0.3,
                "target_bbox_area": 0.06,
                "reason": "target clearly visible and close",
                "latency_sec": 0.01,
            }
        )
        client = FakeVLMClient([modules.schema.make_rotate_output(30)])
        planner = QwenVLMPlanner(
            vlm_client=client,
            stop_verifier=verifier,
            stop_confirmations_required=2,
        )
        planner.memory_sidecar = None
        planner.reset("houseplant")
        planner.update_runtime_context(
            frame_index=1,
            position_xyz=[0.0, 0.0, 0.0],
        )

        planner.make_plan_from_views(
            [np.zeros((20, 20, 3), dtype=np.uint8)],
            [0],
            call_type="proactive_stop_first",
        )
        first = dict(planner.last_decision)
        planner.record_target_approach_execution(
            {
                "go_progress": {
                    "translation_m": 0.3,
                    "collision_count": 0,
                    "controller_reached_waypoint": True,
                    "no_progress": False,
                    "execution_success": True,
                }
            },
            frame_index=2,
        )
        planner.update_runtime_context(
            frame_index=2,
            position_xyz=[0.3, 0.0, 0.0],
        )
        planner.make_plan_from_views(
            [np.zeros((20, 20, 3), dtype=np.uint8)],
            [0],
            call_type="proactive_stop_second",
        )
        second = dict(planner.last_decision)

        self.assertEqual(first["action"], "request_observation")
        self.assertTrue(first["target_approach_floor_reacquisition"])
        self.assertEqual(
            first["proactive_stop_probe_result"]["consecutive_confirmations"], 1
        )
        self.assertEqual(second["action"], "stop")
        self.assertTrue(second["proactive_stop_probe"])
        self.assertEqual(second["stop_verification"]["consecutive_confirmations"], 2)
        self.assertEqual(len(client.inputs), 0)

    def test_proactive_target_detection_forces_supported_approach_before_main_call(self):
        modules = import_nav_memory_qwen()
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": True,
                "target_visible": True,
                "target_match": True,
                "close_enough": True,
                "confidence": "high",
                "target_view_index": 0,
                "target_center_x": 0.75,
                "reason": "target visible and close",
                "latency_sec": 0.01,
                "backend_call_count": 1,
                "backend_error_count": 0,
            }
        )
        client = FakeVLMClient([modules.schema.make_rotate_output(30)])
        planner = QwenVLMPlanner(
            vlm_client=client,
            stop_verifier=verifier,
            stop_confirmations_required=2,
        )
        planner.memory_sidecar = None
        planner.reset("chair")
        planner.update_runtime_context(frame_index=1, position_xyz=[0.0, 0.0, 0.0])
        image = np.random.default_rng(11).integers(
            0,
            256,
            size=(100, 200, 3),
            dtype=np.uint8,
        )

        planner.make_plan_from_views([image], [0], call_type="proactive_target_approach")

        decision = planner.last_decision
        self.assertEqual(decision["action"], "go")
        self.assertTrue(decision["proactive_target_approach"])
        self.assertTrue(decision["pixel_candidate_validation"]["passed"])
        self.assertTrue(decision["stop_verification"]["forced_approach"]["triggered"])
        self.assertEqual(client.inputs, [])

    def test_verified_target_restores_same_view_candidate_instead_of_other_heading(self):
        modules = import_nav_memory_qwen()
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([modules.schema.make_rotate_output(30)])
        )
        planner.memory_sidecar = None
        planner.reset("houseplant")
        other_heading = {
            "candidate_ref": "px_v04_r1_c1",
            "view_id": 4,
            "view_type_hint": "right",
            "point_px": [100, 80],
            "point_norm": [0.5, 0.8],
            "avoid": False,
            "requires_rgb_verification": False,
            "score": 0.95,
        }
        excluded_target_heading = {
            "candidate_ref": "px_v03_r1_c1",
            "view_id": 3,
            "view_type_hint": "front",
            "point_px": [100, 80],
            "point_norm": [0.5, 0.8],
            "avoid": True,
            "executable": False,
            "status": "deadlock_entry_candidate",
            "reason": "known negative branch",
            "topological_edge_id": "e_failed",
            "requires_rgb_verification": False,
            "visual_evidence": {
                "available": True,
                "hard_reject": False,
                "support_score": 0.95,
            },
        }
        vlm_input = {
            "observation": {"image_width": 200, "image_height": 100},
            "pixel_candidates": {
                "candidates": [other_heading],
                "excluded_candidates": [excluded_target_heading],
                "executable_count": 1,
            },
        }
        verification = {
            "target_view_index": 3,
            "target_center_x": 0.5,
            "target_evidence_passed": True,
            "confidence": "high",
            "identity_critic": {
                "triggered": True,
                "passed": True,
                "exact_target": True,
            },
            "near_goal_visual_latch_active": True,
        }

        output = planner._approach_output_after_stop_guard(vlm_input, verification)
        validation = validate_pixel_candidate_selection(output, vlm_input)

        self.assertIsNotNone(output)
        self.assertEqual(output["selected_candidate_ref"], "px_v03_r1_c1")
        self.assertEqual(output["selected_view_id"], 3)
        self.assertTrue(validation["passed"])
        self.assertEqual(
            output["target_approach_memory_override"]["original_edge_id"],
            "e_failed",
        )
        restored = vlm_input["pixel_candidates"]["candidates"][-1]
        self.assertFalse(restored["avoid"])
        self.assertEqual(restored["status"], "verified_target_evidence_override")

    def test_proactive_edge_target_rotates_to_center_before_approach(self):
        modules = import_nav_memory_qwen()
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": False,
                "target_visible": True,
                "target_match": True,
                "close_enough": False,
                "confidence": "high",
                "target_view_index": 0,
                "target_center_x": 0.1,
                "target_evidence_passed": True,
                "target_center_guard_satisfied": False,
                "target_bbox_guard_satisfied": False,
                "reason": "target evidence outside central stop band",
                "latency_sec": 0.01,
                "backend_call_count": 1,
                "backend_error_count": 0,
            }
        )
        client = FakeVLMClient([modules.schema.make_rotate_output(30)])
        planner = QwenVLMPlanner(
            vlm_client=client,
            stop_verifier=verifier,
            stop_confirmations_required=2,
        )
        planner.memory_sidecar = None
        planner.reset("houseplant")
        planner.update_runtime_context(frame_index=1, position_xyz=[0.0, 0.0, 0.0])

        planner.make_plan_from_views(
            [np.zeros((100, 200, 3), dtype=np.uint8)],
            [0],
            call_type="proactive_target_centering",
        )

        decision = planner.last_decision
        self.assertEqual(decision["action"], "rotate")
        self.assertTrue(decision["proactive_target_centering"])
        self.assertEqual(decision["vlm_output"]["control"]["rotate_yaw_deg"], -30.0)
        self.assertTrue(decision["stop_verification"]["forced_centering"]["triggered"])
        self.assertEqual(client.inputs, [])

    def test_verified_stop_reaches_runner(self):
        modules = import_nav_memory_qwen()
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": True,
                "target_visible": True,
                "target_match": True,
                "close_enough": True,
                "confidence": "high",
                "target_bbox_height": 0.3,
                "target_bbox_area": 0.06,
                "reason": "target clearly visible and close",
                "latency_sec": 0.01,
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([modules.schema.make_stop_output()]),
            stop_verifier=verifier,
            stop_confirmations_required=1,
        )
        planner.reset("chair")
        planner.make_plan_from_views(
            [np.zeros((20, 20, 3), dtype=np.uint8)],
            [0],
            call_type="qwen_action_loop",
        )

        self.assertEqual(planner.last_decision["action"], "stop")
        self.assertTrue(planner.last_decision["stop_verification"]["passed"])
        self.assertEqual(planner.stop_verification_pass_count, 1)

    def test_stop_without_approach_motion_is_overridden_to_supported_go_candidate(self):
        modules = import_nav_memory_qwen()
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": True,
                "target_visible": True,
                "target_match": True,
                "close_enough": True,
                "confidence": "high",
                "reason": "target visible but approach not yet confirmed",
                "latency_sec": 0.01,
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([modules.schema.make_stop_output()]),
            stop_verifier=verifier,
            stop_confirmations_required=2,
        )
        planner.memory_sidecar = None
        planner.reset("chair")
        planner.update_runtime_context(position_xyz=[0.0, 0.0, 0.0])
        image = np.random.default_rng(7).integers(
            0,
            256,
            size=(100, 200, 3),
            dtype=np.uint8,
        )

        planner.make_plan_from_views([image], [0], call_type="stop_motion_guard")

        decision = planner.last_decision
        self.assertEqual(decision["raw_vlm_output"]["action"], "stop")
        self.assertEqual(decision["action"], "go")
        self.assertTrue(decision["pixel_candidate_validation"]["passed"])
        self.assertEqual(decision["backend_action_override"]["from"], "stop")
        self.assertEqual(decision["backend_action_override"]["to"], "go")
        self.assertTrue(decision["stop_verification"]["forced_approach"]["triggered"])

    def test_stop_requires_two_consecutive_verified_observations(self):
        modules = import_nav_memory_qwen()
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": True,
                "target_visible": True,
                "target_match": True,
                "close_enough": True,
                "confidence": "high",
                "target_bbox_height": 0.3,
                "target_bbox_area": 0.06,
                "reason": "target clearly visible and close",
                "latency_sec": 0.01,
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient(
                [modules.schema.make_stop_output(), modules.schema.make_stop_output()]
            ),
            stop_verifier=verifier,
            stop_confirmations_required=2,
        )
        planner.reset("chair")
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        planner.update_runtime_context(frame_index=0, position_xyz=[0.0, 0.0, 0.0])

        planner.make_plan_from_views([image], [0], call_type="qwen_action_loop")
        self.assertEqual(planner.last_decision["action"], "request_observation")
        self.assertEqual(
            planner.last_decision["stop_verification"]["consecutive_confirmations"], 1
        )

        planner.record_target_approach_execution(
            {
                "go_progress": {
                    "translation_m": 0.3,
                    "collision_count": 0,
                    "controller_reached_waypoint": True,
                    "no_progress": False,
                    "execution_success": True,
                }
            },
            frame_index=1,
        )
        planner.update_runtime_context(frame_index=1, position_xyz=[0.3, 0.0, 0.0])
        planner.make_plan_from_views([image], [0], call_type="qwen_action_loop")
        self.assertEqual(planner.last_decision["action"], "stop")
        self.assertEqual(
            planner.last_decision["stop_verification"]["consecutive_confirmations"], 2
        )

    def test_compact_target_stops_after_collision_free_post_confirmation_approach(self):
        modules = import_nav_memory_qwen()
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": True,
                "target_visible": True,
                "target_match": True,
                "close_enough": True,
                "confidence": "high",
                "target_bbox_height": 0.3,
                "target_bbox_area": 0.06,
                "target_center_x": 0.5,
                "target_center_guard_satisfied": True,
                "identity_critic": {
                    "triggered": True,
                    "passed": True,
                    "exact_target": True,
                },
                "reason": "exact chair is visible and close",
                "latency_sec": 0.01,
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([modules.schema.make_rotate_output(30)]),
            stop_verifier=verifier,
            stop_confirmations_required=2,
        )
        planner.memory_sidecar = None
        planner.reset("chair")
        image = np.random.default_rng(139).integers(
            0,
            256,
            size=(100, 200, 3),
            dtype=np.uint8,
        )
        planner.update_runtime_context(
            frame_index=134,
            position_xyz=[0.0, 0.0, 0.0],
        )
        planner.make_plan_from_views([image], [0], call_type="chair_first_exact")

        outcome = planner.record_target_approach_execution(
            {
                "selected_candidate_ref": "px_v00_r1_c1",
                "go_progress": {
                    "translation_m": 0.5,
                    "collision_count": 0,
                    "controller_reached_waypoint": False,
                    "no_progress": False,
                    "execution_success": True,
                },
            },
            frame_index=139,
        )
        verifier.result.update(
            target_bbox_height=0.4,
            target_bbox_area=0.1,
            reason="same exact chair is larger after approach",
        )
        planner.update_runtime_context(
            frame_index=139,
            position_xyz=[0.5, 0.0, 0.0],
        )
        planner.make_plan_from_views([image], [0], call_type="chair_second_exact")

        refinement_decision = dict(planner.last_decision)
        verification = refinement_decision["stop_verification"]
        self.assertTrue(outcome["progress_satisfied"])
        self.assertEqual(outcome["failure_streak"], 0)
        self.assertEqual(refinement_decision["action"], "go")
        self.assertTrue(
            refinement_decision["verified_stop_terminal_refinement"]
        )
        self.assertTrue(verification["target_approach_after_confirmation"])
        self.assertTrue(verification["target_approach_progress_satisfied"])
        self.assertTrue(
            verification["post_confirmation_compact_target_evidence"]
        )
        self.assertTrue(verification["target_approach_execution_satisfied"])

        planner.record_target_approach_execution(
            {
                "selected_candidate_ref": "px_v00_r1_c1",
                "go_progress": {
                    "translation_m": 0.25,
                    "collision_count": 0,
                    "controller_reached_waypoint": False,
                    "no_progress": False,
                    "execution_success": True,
                },
            },
            frame_index=140,
        )
        planner.update_runtime_context(
            frame_index=140,
            position_xyz=[0.75, 0.0, 0.0],
        )
        planner.make_plan_from_views([image], [0], call_type="chair_after_refinement")
        self.assertEqual(planner.last_decision["action"], "stop")

    def test_verified_near_coarse_goal_skips_terminal_refinement(self):
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.memory_sidecar = None
        planner.reset("chair")
        verification = {
            "passed": True,
            "single_frame_passed": True,
            "coarse_goal_consistency_guard": {
                "enabled": True,
                "triggered": True,
                "passed": True,
                "stop_consistent": True,
                "distance_m": 0.2,
            },
            "identity_critic": {
                "triggered": True,
                "passed": True,
                "exact_target": True,
            },
            "target_approach_execution": {
                "controller_reached_waypoint": False,
            },
        }

        with patch.dict(
            os.environ,
            {"VOCA_COARSE_GOAL_TERMINAL_STOP_RADIUS_M": "0.75"},
            clear=False,
        ):
            output = planner._terminal_refinement_output_after_stop_guard(
                {"pixel_candidates": {"candidates": []}},
                verification,
            )

        self.assertIsNone(output)
        self.assertTrue(verification["coarse_terminal_stop_authorized"])
        self.assertEqual(verification["coarse_terminal_stop_radius_m"], 0.75)

    def test_stop_accepts_stationary_confirmation_after_blocked_target_approach(self):
        modules = import_nav_memory_qwen()
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": True,
                "target_visible": True,
                "target_match": True,
                "close_enough": True,
                "confidence": "high",
                "target_bbox_height": 0.3,
                "target_bbox_area": 0.06,
                "reason": "target clearly visible and close",
                "latency_sec": 0.01,
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient(
                [modules.schema.make_stop_output(), modules.schema.make_stop_output()]
            ),
            stop_verifier=verifier,
            stop_confirmations_required=2,
        )
        planner.reset("chair")
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        planner.update_runtime_context(
            frame_index=10,
            position_xyz=[0.0, 0.0, 0.0],
        )
        planner.make_plan_from_views([image], [0], call_type="stationary_stop_first")

        planner.record_target_approach_execution(
            {
                "go_progress": {
                    "translation_m": 0.0,
                    "collision_count": 1,
                    "controller_reached_waypoint": False,
                    "no_progress": True,
                    "execution_success": False,
                }
            },
            frame_index=11,
        )

        planner.update_runtime_context(
            frame_index=11,
            position_xyz=[0.0, 0.0, 0.0],
        )
        planner.make_plan_from_views([image], [0], call_type="stationary_stop_second")

        verification = planner.last_decision["stop_verification"]
        self.assertEqual(planner.last_decision["action"], "stop")
        self.assertTrue(verification["confirmation_frame_change_satisfied"])
        self.assertFalse(verification["confirmation_motion_satisfied"])
        self.assertEqual(verification["confirmation_frame_gap"], 1)
        self.assertTrue(verification["strong_visual_proximity_satisfied"])
        self.assertTrue(verification["target_approach_execution_satisfied"])

    def test_stop_rejects_stationary_temporal_confirmation_when_target_stays_small(self):
        modules = import_nav_memory_qwen()
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": True,
                "target_visible": True,
                "target_match": True,
                "close_enough": True,
                "confidence": "high",
                "target_bbox_height": 0.1,
                "target_bbox_area": 0.01,
                "reason": "small target appears visually close",
                "latency_sec": 0.01,
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient(
                [modules.schema.make_stop_output(), modules.schema.make_stop_output()]
            ),
            stop_verifier=verifier,
            stop_confirmations_required=2,
        )
        planner.reset("plant")
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        planner.update_runtime_context(frame_index=10, position_xyz=[0.0, 0.0, 0.0])
        planner.make_plan_from_views([image], [0], call_type="weak_scale_first")
        planner.update_runtime_context(frame_index=11, position_xyz=[0.0, 0.0, 0.0])
        planner.make_plan_from_views([image], [0], call_type="weak_scale_second")

        verification = planner.last_decision["stop_verification"]
        self.assertNotEqual(planner.last_decision["action"], "stop")
        self.assertFalse(verification["strong_visual_proximity_satisfied"])
        self.assertFalse(verification["confirmation_independence_satisfied"])
        self.assertEqual(verification["consecutive_confirmations"], 1)

    def test_near_goal_visual_latch_rejects_post_approach_scale_drop(self):
        modules = import_nav_memory_qwen()
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": True,
                "target_visible": True,
                "target_match": True,
                "close_enough": True,
                "confidence": "high",
                "target_bbox_height": 0.5,
                "target_bbox_area": 0.12,
                "reason": "large target evidence near the robot",
                "latency_sec": 0.01,
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([modules.schema.make_rotate_output(30)]),
            stop_verifier=verifier,
            stop_confirmations_required=2,
        )
        planner.memory_sidecar = None
        planner.reset("houseplant")
        image = np.random.default_rng(71).integers(
            0,
            256,
            size=(100, 200, 3),
            dtype=np.uint8,
        )

        with patch.dict(
            os.environ,
            {
                "VOCA_STOP_ALLOW_TRANSLATION_ONLY_CONFIRM": "0",
                "VOCA_STOP_NEAR_LATCH_MIN_BBOX_HEIGHT": "0.45",
                "VOCA_STOP_NEAR_LATCH_MIN_BBOX_AREA": "0.08",
            },
        ):
            planner.update_runtime_context(
                frame_index=10,
                position_xyz=[0.0, 0.0, 0.0],
            )
            planner.make_plan_from_views(
                [image],
                [0],
                call_type="near_latch_first",
            )
            first_verification = planner.last_decision["stop_verification"]
            self.assertTrue(first_verification["near_goal_visual_latch_evidence"])
            self.assertNotEqual(planner.last_decision["action"], "stop")

            planner.record_target_approach_execution(
                {
                    "go_progress": {
                        "translation_m": 0.3,
                        "collision_count": 0,
                        "controller_reached_waypoint": False,
                        "no_progress": False,
                        "execution_success": True,
                    }
                },
                frame_index=11,
            )
            verifier.result.update(
                target_bbox_height=0.3,
                target_bbox_area=0.09,
                reason="same target remains visible after approach",
            )
            planner.update_runtime_context(
                frame_index=11,
                position_xyz=[0.3, 0.0, 0.0],
            )
            planner.make_plan_from_views(
                [image],
                [0],
                call_type="near_latch_second",
            )

        verification = planner.last_decision["stop_verification"]
        self.assertNotEqual(planner.last_decision["action"], "stop")
        self.assertFalse(verification["target_scale_consistent"])
        self.assertFalse(
            verification["near_goal_visual_latch_confirmation_satisfied"]
        )
        self.assertFalse(verification["target_approach_execution_satisfied"])
        self.assertFalse(
            verification["near_goal_visual_latch_scale_guard_satisfied"]
        )
        self.assertEqual(verification["near_goal_visual_latch_approach_count"], 1)
        self.assertEqual(verification["near_goal_visual_latch_approaches_required"], 4)
        self.assertEqual(
            verification["near_goal_visual_latch_category_min_approaches"],
            4,
        )
        self.assertTrue(verification["approach_preferred_before_centering"])
        self.assertIn("target scale decreased", verification["reason"])

    def test_near_goal_latch_survives_intermediate_confirmation_reset(self):
        modules = import_nav_memory_qwen()
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": True,
                "target_visible": True,
                "target_match": True,
                "close_enough": True,
                "confidence": "high",
                "target_bbox_height": 0.5,
                "target_bbox_area": 0.12,
                "reason": "same compact chair is visible after approach",
                "latency_sec": 0.01,
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([modules.schema.make_rotate_output(30)]),
            stop_verifier=verifier,
            stop_confirmations_required=2,
        )
        planner.memory_sidecar = None
        planner.reset("chair")
        image = np.random.default_rng(711).integers(
            0,
            256,
            size=(100, 200, 3),
            dtype=np.uint8,
        )
        planner.update_runtime_context(
            frame_index=10,
            position_xyz=[0.0, 0.0, 0.0],
        )
        planner.make_plan_from_views([image], [0], call_type="latch_before_reset")
        planner.record_target_approach_execution(
            {
                "go_progress": {
                    "translation_m": 0.3,
                    "collision_count": 0,
                    "controller_reached_waypoint": False,
                    "no_progress": False,
                    "execution_success": True,
                }
            },
            frame_index=11,
        )

        planner.consecutive_stop_confirmations = 0
        planner._last_stop_confirmation_position = []
        planner._last_stop_confirmation_frame_index = None
        planner._last_stop_confirmation_bbox_height = 0.0
        planner._last_stop_confirmation_bbox_area = 0.0
        planner.update_runtime_context(
            frame_index=11,
            position_xyz=[0.3, 0.0, 0.0],
        )
        planner.make_plan_from_views([image], [0], call_type="latch_after_reset")

        verification = planner.last_decision["stop_verification"]
        self.assertEqual(planner.last_decision["action"], "stop")
        self.assertTrue(
            verification["near_goal_visual_latch_confirmation_satisfied"]
        )
        self.assertEqual(
            verification["temporal_confirmation_source"],
            "near_goal_visual_latch",
        )
        self.assertEqual(verification["consecutive_confirmations"], 1)
        self.assertEqual(verification["effective_confirmations"], 2)

    def test_plant_scale_drop_requires_visual_recovery_after_three_approaches(self):
        modules = import_nav_memory_qwen()
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": True,
                "target_visible": True,
                "target_match": True,
                "close_enough": True,
                "confidence": "high",
                "target_bbox_height": 0.5,
                "target_bbox_area": 0.12,
                "reason": "large plant evidence",
                "latency_sec": 0.01,
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([modules.schema.make_rotate_output(30)]),
            stop_verifier=verifier,
            stop_confirmations_required=2,
        )
        planner.memory_sidecar = None
        planner.reset("houseplant")
        image = np.random.default_rng(72).integers(
            0,
            256,
            size=(100, 200, 3),
            dtype=np.uint8,
        )

        with patch.dict(
            os.environ,
            {
                "VOCA_STOP_NEAR_LATCH_MIN_BBOX_HEIGHT": "0.40",
                "VOCA_STOP_NEAR_LATCH_MIN_BBOX_AREA": "0.08",
                "VOCA_STOP_NEAR_LATCH_PLANT_MIN_APPROACHES": "3",
            },
        ):
            planner.update_runtime_context(frame_index=10, position_xyz=[0.0, 0.0, 0.0])
            planner.make_plan_from_views([image], [0], call_type="plant_latch_first")
            verifier.result.update(target_bbox_height=0.3, target_bbox_area=0.09)
            for approach_index in range(1, 4):
                planner.record_target_approach_execution(
                    {
                        "go_progress": {
                            "translation_m": 0.3,
                            "collision_count": 0,
                            "controller_reached_waypoint": approach_index == 2,
                            "no_progress": False,
                            "execution_success": True,
                        }
                    },
                    frame_index=10 + approach_index,
                )
                planner.update_runtime_context(
                    frame_index=10 + approach_index,
                    position_xyz=[0.3 * approach_index, 0.0, 0.0],
                )
                planner.make_plan_from_views(
                    [image],
                    [0],
                    call_type="plant_latch_approach_{}".format(approach_index),
                )
                if approach_index < 3:
                    self.assertNotEqual(planner.last_decision["action"], "stop")
                if approach_index == 2:
                    interim = planner.last_decision["stop_verification"]
                    self.assertTrue(interim["target_approach_terminal_satisfied"])
                    self.assertFalse(
                        interim["near_goal_visual_latch_terminal_bypass_allowed"]
                    )
                    self.assertFalse(
                        interim["near_goal_visual_latch_confirmation_satisfied"]
                    )

            verification = planner.last_decision["stop_verification"]
            self.assertNotEqual(planner.last_decision["action"], "stop")
            self.assertEqual(verification["near_goal_visual_latch_approach_count"], 3)
            self.assertEqual(verification["near_goal_visual_latch_approaches_required"], 3)
            self.assertFalse(verification["near_goal_visual_latch_scale_guard_satisfied"])
            self.assertFalse(
                verification["near_goal_visual_latch_scale_drop_bypass_allowed"]
            )

            verifier.result.update(target_bbox_height=0.5, target_bbox_area=0.12)
            planner.update_runtime_context(frame_index=14, position_xyz=[0.9, 0.0, 0.0])
            planner.make_plan_from_views(
                [image],
                [0],
                call_type="plant_latch_scale_recovered",
            )

        verification = planner.last_decision["stop_verification"]
        self.assertEqual(planner.last_decision["action"], "stop")
        self.assertTrue(verification["target_scale_consistent"])
        self.assertTrue(verification["near_goal_visual_latch_scale_guard_satisfied"])

    def test_continuous_plant_evidence_refreshes_latch_position_anchor(self):
        modules = import_nav_memory_qwen()
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": True,
                "target_visible": True,
                "target_match": True,
                "close_enough": True,
                "confidence": "high",
                "target_bbox_height": 0.5,
                "target_bbox_area": 0.12,
                "reason": "same large plant remains visible",
                "latency_sec": 0.01,
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([modules.schema.make_rotate_output(30)]),
            stop_verifier=verifier,
            stop_confirmations_required=2,
        )
        planner.memory_sidecar = None
        planner.reset("houseplant")
        image = np.random.default_rng(721).integers(
            0,
            256,
            size=(100, 200, 3),
            dtype=np.uint8,
        )

        with patch.dict(
            os.environ,
            {
                "VOCA_STOP_NEAR_LATCH_PLANT_MIN_APPROACHES": "4",
                "VOCA_STOP_NEAR_LATCH_RADIUS_M": "1.0",
            },
        ):
            planner.update_runtime_context(frame_index=10, position_xyz=[0.0, 0.0, 0.0])
            planner.make_plan_from_views([image], [0], call_type="moving_latch_first")
            for approach_index in range(1, 5):
                planner.record_target_approach_execution(
                    {
                        "go_progress": {
                            "translation_m": 0.35,
                            "collision_count": 0,
                            "controller_reached_waypoint": False,
                            "no_progress": False,
                            "execution_success": True,
                        }
                    },
                    frame_index=10 + approach_index,
                )
                planner.update_runtime_context(
                    frame_index=10 + approach_index,
                    position_xyz=[0.35 * approach_index, 0.0, 0.0],
                )
                if approach_index == 4:
                    verifier.result.update(
                        target_bbox_height=0.44,
                        target_bbox_area=0.10,
                    )
                planner.make_plan_from_views(
                    [image],
                    [0],
                    call_type="moving_latch_approach_{}".format(approach_index),
                )

        verification = planner.last_decision["stop_verification"]
        self.assertEqual(planner.last_decision["action"], "stop")
        self.assertEqual(verification["near_goal_visual_latch_approach_count"], 4)
        self.assertTrue(verification["target_scale_consistent"])
        self.assertEqual(
            verification["target_scale_consistency_threshold"],
            {"min_height_ratio": 0.85, "min_area_ratio": 0.75},
        )
        self.assertLessEqual(
            verification["near_goal_visual_latch_displacement_m"],
            0.35 + 1e-6,
        )
        self.assertEqual(
            planner._near_goal_visual_latch["position_xyz"],
            [1.4, 0.0, 0.0],
        )

    def test_small_visual_target_does_not_create_near_goal_latch(self):
        modules = import_nav_memory_qwen()
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": True,
                "target_visible": True,
                "target_match": True,
                "close_enough": True,
                "confidence": "high",
                "target_bbox_height": 0.2,
                "target_bbox_area": 0.04,
                "reason": "small target could still be across an obstacle",
                "latency_sec": 0.01,
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient(
                [modules.schema.make_rotate_output(30)] * 2
            ),
            stop_verifier=verifier,
            stop_confirmations_required=2,
        )
        planner.memory_sidecar = None
        planner.reset("houseplant")
        image = np.random.default_rng(72).integers(
            0,
            256,
            size=(100, 200, 3),
            dtype=np.uint8,
        )

        with patch.dict(
            os.environ,
            {
                "VOCA_STOP_ALLOW_TRANSLATION_ONLY_CONFIRM": "0",
                "VOCA_STOP_NEAR_LATCH_MIN_BBOX_HEIGHT": "0.45",
                "VOCA_STOP_NEAR_LATCH_MIN_BBOX_AREA": "0.08",
            },
        ):
            planner.update_runtime_context(frame_index=20, position_xyz=[0.0, 0.0, 0.0])
            planner.make_plan_from_views([image], [0], call_type="small_latch_first")
            planner.record_target_approach_execution(
                {
                    "go_progress": {
                        "translation_m": 0.3,
                        "collision_count": 0,
                        "controller_reached_waypoint": False,
                        "no_progress": False,
                        "execution_success": True,
                    }
                },
                frame_index=21,
            )
            planner.update_runtime_context(frame_index=21, position_xyz=[0.3, 0.0, 0.0])
            planner.make_plan_from_views([image], [0], call_type="small_latch_second")

        verification = planner.last_decision["stop_verification"]
        self.assertNotEqual(planner.last_decision["action"], "stop")
        self.assertFalse(verification["near_goal_visual_latch_evidence"])
        self.assertFalse(verification["near_goal_visual_latch_active"])
        self.assertFalse(
            verification["near_goal_visual_latch_confirmation_satisfied"]
        )

    def test_area_only_near_goal_latch_overrides_approach_cooldown(self):
        modules = import_nav_memory_qwen()
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": True,
                "target_visible": True,
                "target_match": True,
                "close_enough": True,
                "confidence": "high",
                "target_bbox_height": 0.3,
                "target_bbox_area": 0.15,
                "reason": "wide sofa fills a large part of the view",
                "latency_sec": 0.01,
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([modules.schema.make_rotate_output(30)]),
            stop_verifier=verifier,
            stop_confirmations_required=2,
        )
        planner.memory_sidecar = None
        planner.reset("sofa")
        planner._target_direct_approach_cooldown = 4
        image = np.random.default_rng(73).integers(
            0,
            256,
            size=(100, 200, 3),
            dtype=np.uint8,
        )

        with patch.dict(
            os.environ,
            {
                "VOCA_STOP_NEAR_LATCH_MIN_BBOX_HEIGHT": "0.40",
                "VOCA_STOP_NEAR_LATCH_MIN_BBOX_AREA": "0.08",
            },
        ):
            planner.update_runtime_context(frame_index=30, position_xyz=[0.0, 0.0, 0.0])
            planner.make_plan_from_views([image], [0], call_type="area_latch_cooldown")

        verification = planner.last_decision["stop_verification"]
        self.assertEqual(planner.last_decision["action"], "go")
        self.assertTrue(planner.last_decision["proactive_target_approach"])
        self.assertTrue(verification["near_goal_visual_latch_evidence"])
        self.assertTrue(verification["direct_target_approach_cooldown_overridden"])
        self.assertEqual(planner._target_direct_approach_cooldown, 0)

    def test_area_only_near_goal_latch_requires_two_approaches(self):
        modules = import_nav_memory_qwen()
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": True,
                "target_visible": True,
                "target_match": True,
                "close_enough": True,
                "confidence": "high",
                "target_bbox_height": 0.3,
                "target_bbox_area": 0.15,
                "reason": "wide target evidence",
                "latency_sec": 0.01,
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([modules.schema.make_rotate_output(30)]),
            stop_verifier=verifier,
            stop_confirmations_required=2,
        )
        planner.memory_sidecar = None
        planner.reset("sofa")
        image = np.random.default_rng(74).integers(
            0,
            256,
            size=(100, 200, 3),
            dtype=np.uint8,
        )

        with patch.dict(
            os.environ,
            {
                "VOCA_STOP_NEAR_LATCH_MIN_BBOX_HEIGHT": "0.40",
                "VOCA_STOP_NEAR_LATCH_MIN_BBOX_AREA": "0.08",
                "VOCA_STOP_NEAR_LATCH_SINGLE_APPROACH_HEIGHT": "0.45",
                "VOCA_STOP_NEAR_LATCH_SINGLE_APPROACH_AREA": "0.10",
            },
        ):
            planner.update_runtime_context(frame_index=40, position_xyz=[0.0, 0.0, 0.0])
            planner.make_plan_from_views([image], [0], call_type="moderate_latch_first")
            planner.record_target_approach_execution(
                {
                    "go_progress": {
                        "translation_m": 0.3,
                        "collision_count": 0,
                        "controller_reached_waypoint": False,
                        "no_progress": False,
                        "execution_success": True,
                    }
                },
                frame_index=41,
            )
            planner.update_runtime_context(frame_index=41, position_xyz=[0.3, 0.0, 0.0])
            planner.make_plan_from_views([image], [0], call_type="moderate_latch_second")
            second = dict(planner.last_decision)
            planner.record_target_approach_execution(
                {
                    "go_progress": {
                        "translation_m": 0.3,
                        "collision_count": 0,
                        "controller_reached_waypoint": False,
                        "no_progress": False,
                        "execution_success": True,
                    }
                },
                frame_index=42,
            )
            planner.update_runtime_context(frame_index=42, position_xyz=[0.6, 0.0, 0.0])
            planner.make_plan_from_views([image], [0], call_type="moderate_latch_third")

        second_verification = second["stop_verification"]
        final_verification = planner.last_decision["stop_verification"]
        self.assertNotEqual(second["action"], "stop")
        self.assertEqual(
            second_verification["near_goal_visual_latch_approach_count"],
            1,
        )
        self.assertEqual(
            second_verification["near_goal_visual_latch_approaches_required"],
            2,
        )
        self.assertEqual(planner.last_decision["action"], "stop")
        self.assertEqual(
            final_verification["near_goal_visual_latch_approach_count"],
            2,
        )
        self.assertTrue(
            final_verification["near_goal_visual_latch_confirmation_satisfied"]
        )

    def test_qwen_selected_point_verifier_parses_conservative_surface_result(self):
        modules = import_nav_memory_qwen()
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"valid":false,"surface":"wall","confidence":"high",'
                            '"reason":"selected point is on a vertical wall"}'
                        )
                    },
                }
            ]
        }
        base = FakeOpenAIClient([response])
        compact = CompactQwenNavVLMClient(base, modules)
        verifier = QwenSelectedPointVerifier(compact, modules)
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "selected.jpg"
            modules.utils.Image.new("RGB", (20, 20), color=(220, 220, 220)).save(image_path)

            result = verifier.verify(
                image_path=str(image_path),
                point_px=[10, 16],
                candidate_ref="px_v00_r1_c1",
                visual_risk={"requires_verification": True, "risk_reasons": ["low_texture"]},
            )

        self.assertFalse(result["passed"])
        self.assertEqual(result["surface"], "wall")
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(base.payloads[0]["max_tokens"], 1024)
        self.assertEqual(base.payloads[0]["response_format"], {"type": "json_object"})

    def test_exit_intent_verifier_rejects_ordinary_floor(self):
        modules = import_nav_memory_qwen()
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"valid":true,"surface":"floor","confidence":"high",'
                            '"reason":"marked point is interior floor"}'
                        )
                    },
                }
            ]
        }
        base = FakeOpenAIClient([response])
        verifier = QwenSelectedPointVerifier(
            CompactQwenNavVLMClient(base, modules),
            modules,
        )
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "selected.jpg"
            modules.utils.Image.new("RGB", (20, 20), color=(120, 120, 120)).save(
                image_path
            )
            result = verifier.verify(
                image_path=str(image_path),
                point_px=[10, 16],
                candidate_ref="px_v00_r1_c1",
                visual_risk={"requires_verification": False},
                required_surface="doorway_floor",
                current_room="bathroom",
            )

        self.assertFalse(result["passed"])
        self.assertEqual(result["surface"], "floor")
        self.assertEqual(result["required_surface"], "doorway_floor")
        prompt = base.payloads[0]["messages"][1]["content"][0]["text"]
        self.assertIn("camera-side near foreground", prompt)
        self.assertIn("visibly beyond a doorframe/opening", prompt)
        self.assertIn("Current strategic room estimate: bathroom", prompt)

    def test_exit_intent_verifier_accepts_safe_floor_beyond_visible_doorway(self):
        modules = import_nav_memory_qwen()
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"valid":true,"surface":"floor",'
                            '"leads_out_of_current_room":true,'
                            '"confidence":"high",'
                            '"reason":"safe floor immediately beyond visible doorway"}'
                        )
                    },
                }
            ]
        }
        base = FakeOpenAIClient([response])
        verifier = QwenSelectedPointVerifier(
            CompactQwenNavVLMClient(base, modules),
            modules,
        )
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "selected.jpg"
            modules.utils.Image.new("RGB", (20, 20), color=(120, 120, 120)).save(
                image_path
            )
            result = verifier.verify(
                image_path=str(image_path),
                point_px=[10, 16],
                candidate_ref="px_v00_r1_c1",
                visual_risk={"requires_verification": False},
                required_surface="doorway_floor",
                current_room="bathroom",
            )

        self.assertTrue(result["passed"])
        self.assertTrue(result["leads_out_of_current_room"])
        self.assertEqual(result["surface"], "floor")
        self.assertEqual(result["current_room_context"], "bathroom")

    def test_exit_intent_verifier_retries_internally_contradictory_json(self):
        modules = import_nav_memory_qwen()
        contradictory = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"valid":false,"surface":"doorway_floor",'
                            '"leads_out_of_current_room":true,'
                            '"confidence":"high",'
                            '"reason":"safe floor beyond visible doorway"}'
                        )
                    },
                }
            ]
        }
        corrected = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"valid":true,"surface":"doorway_floor",'
                            '"leads_out_of_current_room":true,'
                            '"confidence":"high",'
                            '"reason":"safe floor beyond visible doorway"}'
                        )
                    },
                }
            ]
        }
        base = FakeOpenAIClient([contradictory, corrected])
        verifier = QwenSelectedPointVerifier(
            CompactQwenNavVLMClient(base, modules),
            modules,
        )
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "selected.jpg"
            modules.utils.Image.new(
                "RGB",
                (20, 20),
                color=(120, 120, 120),
            ).save(image_path)
            result = verifier.verify(
                image_path=str(image_path),
                point_px=[10, 16],
                candidate_ref="px_v00_r1_c1",
                visual_risk={"requires_verification": False},
                required_surface="doorway_floor",
                current_room="bedroom",
            )

        self.assertTrue(result["passed"])
        self.assertTrue(result["consistency_retry_triggered"])
        self.assertFalse(result["consistency_first_attempt"]["valid"])
        self.assertEqual(result["backend_call_count"], 2)
        self.assertEqual(len(base.payloads), 2)
        self.assertIn(
            "internally contradictory",
            base.payloads[1]["messages"][-1]["content"],
        )

    def test_target_stop_verifier_uses_strict_schema_and_response_fallback(self):
        modules = import_nav_memory_qwen()
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "",
                        "reasoning": (
                            '{"target_visible":false,"target_match":false,'
                            '"close_enough":false,"confidence":"high",'
                            '"target_view_index":null,"target_center_x":null,'
                            '"target_bbox_norm":null,'
                            '"reason":"target is absent"}'
                        ),
                    },
                }
            ]
        }
        base = FakeOpenAIClient([response])
        compact = CompactQwenNavVLMClient(base, modules)
        verifier = QwenTargetStopVerifier(compact, modules)
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "front.jpg"
            modules.utils.Image.new("RGB", (20, 20), color=(100, 100, 100)).save(image_path)
            result = verifier.verify(target="chair", image_paths=[str(image_path)])

        self.assertFalse(result["passed"])
        self.assertFalse(result["target_visible"])
        response_format = base.payloads[0]["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertFalse(
            response_format["json_schema"]["schema"]["additionalProperties"]
        )
        self.assertNotIn("thinking_token_budget", base.payloads[0])

    def test_target_stop_verifier_checks_multiple_views_one_at_a_time_until_positive(self):
        modules = import_nav_memory_qwen()
        negative_response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"target_visible":false,"target_match":false,'
                            '"close_enough":false,"confidence":"low",'
                            '"target_view_index":null,"target_center_x":null,'
                            '"target_bbox_norm":null,'
                            '"reason":"target absent"}'
                        )
                    },
                }
            ]
        }
        positive_response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"target_visible":true,"target_match":true,'
                            '"close_enough":true,"confidence":"high",'
                            '"target_view_index":0,"target_center_x":0.7,'
                            '"target_bbox_norm":[0.55,0.3,0.85,0.9],'
                            '"reason":"chair visible and close"}'
                        )
                    },
                }
            ]
        }
        base = FakeOpenAIClient([negative_response, positive_response])
        verifier = QwenTargetStopVerifier(CompactQwenNavVLMClient(base, modules), modules)
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
                path = Path(tmp) / "view_{}.jpg".format(index)
                modules.utils.Image.new("RGB", (40, 30), color=color).save(path)
                paths.append(str(path))
            with patch.dict(os.environ, {"VOCA_QWEN_STOP_IDENTITY_CRITIC": "0"}):
                result = verifier.verify(target="chair", image_paths=paths)

        self.assertEqual(len(base.payloads), 2)
        self.assertTrue(result["passed"])
        self.assertEqual(result["target_view_index"], 1)
        self.assertEqual(result["verification_input_mode"], "sequential_single_view")
        self.assertEqual(result["verification_view_count"], 3)
        self.assertEqual(result["backend_call_count"], 2)

    def test_target_identity_critic_rejects_exercise_bench_as_chair(self):
        modules = import_nav_memory_qwen()
        verifier_response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"target_visible":true,"target_match":true,'
                            '"close_enough":true,"confidence":"high",'
                            '"target_view_index":0,"target_center_x":0.55,'
                            '"target_bbox_norm":[0.4,0.5,0.7,0.8],'
                            '"reason":"chair visible and close"}'
                        )
                    },
                }
            ]
        }
        critic_response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"exact_target":false,"lookalike_detected":true,'
                            '"lookalike_type":"incline exercise bench",'
                            '"confidence":"high","reason":"gym bench lacks upright chair geometry"}'
                        )
                    },
                }
            ]
        }
        base = FakeOpenAIClient([verifier_response, critic_response])
        verifier = QwenTargetStopVerifier(
            CompactQwenNavVLMClient(base, modules),
            modules,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.jpg"
            modules.utils.Image.new("RGB", (40, 30), color=(80, 80, 80)).save(path)
            result = verifier.verify(target="chair", image_paths=[str(path)])

        self.assertFalse(result["passed"])
        self.assertFalse(result["target_evidence_passed"])
        self.assertTrue(result["identity_critic"]["triggered"])
        self.assertFalse(result["identity_critic_passed"])
        self.assertEqual(result["identity_critic"]["lookalike_type"], "incline exercise bench")
        self.assertTrue(result["identity_critic"]["identity_crop"]["used"])
        self.assertEqual(
            result["identity_critic"]["identity_crop"]["input_mode"],
            "marked_full_scene_plus_expanded_crop",
        )
        self.assertEqual(result["identity_critic"]["identity_crop"]["image_count"], 2)
        self.assertEqual(result["identity_critic"]["identity_crop"]["padding_ratio"], 0.45)
        self.assertEqual(result["backend_call_count"], 2)
        critic_content = base.payloads[1]["messages"][1]["content"]
        critic_prompt = critic_content[0]["text"]
        self.assertIn("incline workout benches", critic_prompt)
        self.assertIn("red box is an approximate detector box", critic_prompt)
        self.assertEqual(base.payloads[1]["thinking_token_budget"], 256)
        self.assertEqual(
            len([item for item in critic_content if item.get("type") == "image_url"]),
            2,
        )

    def test_identity_critic_runs_before_off_center_lookalike_can_create_target_lock(self):
        modules = import_nav_memory_qwen()
        verifier_response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"target_visible":true,"target_match":true,'
                            '"close_enough":true,"confidence":"high",'
                            '"target_view_index":0,"target_center_x":0.9,'
                            '"target_bbox_norm":[0.78,0.4,0.98,0.85],'
                            '"reason":"chair-like object at image edge"}'
                        )
                    },
                }
            ]
        }
        critic_response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"exact_target":false,"lookalike_detected":true,'
                            '"lookalike_type":"exercise equipment",'
                            '"confidence":"high","reason":"not a chair"}'
                        )
                    },
                }
            ]
        }
        base = FakeOpenAIClient([verifier_response, critic_response])
        verifier = QwenTargetStopVerifier(
            CompactQwenNavVLMClient(base, modules),
            modules,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "edge_lookalike.jpg"
            modules.utils.Image.new("RGB", (40, 30), color=(80, 80, 80)).save(path)
            result = verifier.verify(target="chair", image_paths=[str(path)])

        self.assertEqual(len(base.payloads), 2)
        self.assertFalse(result["passed"])
        self.assertFalse(result["target_evidence_passed"])
        self.assertTrue(result["identity_critic"]["triggered"])

    def test_identity_rejection_clears_existing_target_lock_and_latch(self):
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": False,
                "target_visible": True,
                "target_match": True,
                "target_evidence_passed": False,
                "identity_critic": {
                    "triggered": True,
                    "passed": False,
                    "reason": "lookalike",
                },
                "confidence": "high",
                "reason": "identity critic rejected target",
                "latency_sec": 0.01,
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([]),
            stop_verifier=verifier,
        )
        planner.memory_sidecar = None
        planner.reset("chair")
        planner._target_lock_remaining_cycles = 3
        planner._near_goal_visual_latch = {"frame_index": 1}
        planner._proactive_stop_pending = True
        planner.consecutive_stop_confirmations = 1

        result = planner._verify_stop_if_needed(
            safe_output={"action": "stop"},
            vlm_input={"metadata": {"raw_observation_images": ["fake.jpg"]}},
        )

        self.assertTrue(result["identity_rejection_cleared_target_state"])
        self.assertEqual(planner._target_lock_remaining_cycles, 0)
        self.assertEqual(planner._near_goal_visual_latch, {})
        self.assertFalse(planner._proactive_stop_pending)
        self.assertEqual(planner.consecutive_stop_confirmations, 0)
        self.assertEqual(
            planner._runtime_feedback[-1]["event"],
            "target_lookalike_rejected",
        )

    def test_identity_rejection_preserves_recent_verified_target_memory_lock(self):
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": False,
                "target_visible": True,
                "target_match": True,
                "target_evidence_passed": False,
                "identity_critic": {
                    "triggered": True,
                    "passed": False,
                    "lookalike_type": "sink",
                    "reason": "new centered object is a sink",
                },
                "confidence": "high",
                "reason": "identity critic rejected current object",
                "latency_sec": 0.01,
            }
        )

        class Sidecar:
            last_verified_target_evidence = {
                "updated": True,
                "frame_index": 20,
                "position_xyz": [0.0, 0.0, 0.0],
                "target_object": "toilet",
            }

        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([]),
            stop_verifier=verifier,
        )
        planner.memory_sidecar = None
        planner.reset("toilet")
        planner.memory_sidecar = Sidecar()
        planner.update_runtime_context(
            frame_index=22,
            position_xyz=[0.0, 0.0, 0.0],
        )
        planner._target_lock_remaining_cycles = 3
        planner._target_lock_reacquisition_attempts = 2
        planner._near_goal_visual_latch = {
            "frame_index": 20,
            "last_evidence_frame_index": 20,
            "position_xyz": [0.0, 0.0, 0.0],
            "approach_count": 2,
        }

        result = planner._verify_stop_if_needed(
            safe_output={"action": "stop"},
            vlm_input={"metadata": {"raw_observation_images": ["fake.jpg"]}},
        )

        self.assertTrue(
            result["identity_rejection_preserved_verified_target_lock"]
        )
        self.assertNotIn("identity_rejection_cleared_target_state", result)
        self.assertTrue(
            result["identity_rejection_preserved_near_goal_latch"]
        )
        self.assertEqual(
            planner._near_goal_visual_latch["approach_count"],
            2,
        )
        self.assertGreaterEqual(planner._target_lock_remaining_cycles, 3)
        self.assertEqual(planner._target_lock_reacquisition_attempts, 2)
        self.assertEqual(
            result["identity_rejection_preserved_reacquisition_attempts"],
            2,
        )
        self.assertEqual(planner._identity_rejection_cooldown_cycles, 0)
        self.assertEqual(
            planner._runtime_feedback[-1]["event"],
            "target_lookalike_rejected_preserve_verified_lock",
        )

    def test_target_lock_reacquisition_expands_to_full_sweep(self):
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": False,
                "target_visible": False,
                "target_match": False,
                "close_enough": False,
                "confidence": "low",
                "target_evidence_passed": False,
                "reason": "target absent",
                "latency_sec": 0.01,
                "backend_call_count": 1,
                "backend_error_count": 0,
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([]),
            stop_verifier=verifier,
        )
        planner.memory_sidecar = None
        planner.reset("chair")
        planner._target_lock_remaining_cycles = 4
        image = np.zeros((20, 20, 3), dtype=np.uint8)

        with patch.dict(
            os.environ,
            {"VOCA_TARGET_LOCK_FULL_SWEEP_AFTER": "2"},
        ):
            planner.update_runtime_context(frame_index=10, position_xyz=[0.0, 0.0, 0.0])
            planner.make_plan_from_views([image], [0], call_type="reacquire_directed")
            first_request = planner.last_decision["vlm_output"]["observation_request"]
            self.assertEqual(first_request["mode"], "directed_sweep")

            planner.update_runtime_context(frame_index=11, position_xyz=[0.0, 0.0, 0.0])
            planner.make_plan_from_views([image], [0], call_type="reacquire_full")
            second_request = planner.last_decision["vlm_output"]["observation_request"]

        self.assertEqual(second_request["mode"], "full_sweep")
        self.assertEqual(len(second_request["yaw_offsets_deg"]), 8)
        self.assertEqual(planner._target_lock_reacquisition_attempts, 2)

    def test_target_lock_budget_breaks_visible_target_reobservation_loop(self):
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": False,
                "target_visible": True,
                "target_match": True,
                "target_evidence_passed": True,
                "close_enough": False,
                "confidence": "high",
                "target_view_index": 0,
                "target_center_x": 0.5,
                "target_center_guard_satisfied": True,
                "target_bbox_height": 0.2,
                "target_bbox_area": 0.03,
                "reason": "target visible but approach floor unresolved",
                "latency_sec": 0.01,
                "backend_call_count": 1,
                "backend_error_count": 0,
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([]),
            stop_verifier=verifier,
        )
        planner.memory_sidecar = None
        planner.reset("chair")
        image = np.zeros((20, 20, 3), dtype=np.uint8)

        with patch.dict(
            os.environ,
            {
                "VOCA_TARGET_LOCK_MAX_CYCLES": "2",
                "VOCA_TARGET_LOCK_EXHAUSTION_COOLDOWN_CYCLES": "3",
            },
            clear=False,
        ):
            for frame_index in (10, 11, 12):
                planner.update_runtime_context(
                    frame_index=frame_index,
                    position_xyz=[0.0, 0.0, 0.0],
                )
                planner.make_plan_from_views(
                    [image],
                    [0],
                    call_type="bounded_target_lock",
                )

        self.assertEqual(planner.last_decision["action"], "rotate")
        self.assertTrue(planner.last_decision["target_lock_budget_exhausted"])
        self.assertEqual(planner._target_lock_session_cycles, 0)
        self.assertEqual(planner._target_lock_cooldown_cycles, 3)
        self.assertEqual(planner._target_lock_exhaustion_count, 1)
        self.assertEqual(
            planner._runtime_feedback[-1]["event"],
            "target_lock_budget_exhausted",
        )

    def test_verified_target_anchor_reacquisition_rotates_toward_memory_pose(self):
        modules = import_nav_memory_qwen()
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([modules.schema.make_rotate_output(30)]),
        )
        planner.reset("chair")
        self.assertIsNotNone(planner.memory_sidecar)
        planner.memory_sidecar.last_verified_target_evidence = {
            "updated": True,
            "node_id": "n_target",
            "target_object": "chair",
            "frame_index": 20,
            "position_xyz": [0.0, 0.0, 1.0],
            "target_bearing_world_rad": math.pi / 2.0,
        }
        planner.update_runtime_context(
            frame_index=22,
            position_xyz=[0.0, 0.0, 0.0],
            heading_rad=0.0,
        )

        output = planner._verified_target_anchor_reacquisition_output()

        self.assertIsNotNone(output)
        self.assertEqual(output["action"], "rotate")
        self.assertEqual(output["control"]["rotate_yaw_deg"], 90.0)
        audit = output["backend_verified_target_anchor_reacquisition"]
        self.assertEqual(audit["bearing_source"], "verified_target_visual_bearing")
        self.assertEqual(audit["bearing_deg_robot"], 90.0)
        self.assertEqual(audit["node_id"], "n_target")

    def test_target_lock_escalation_prefers_verified_anchor_rotation(self):
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": False,
                "target_visible": False,
                "target_match": False,
                "close_enough": False,
                "confidence": "low",
                "target_evidence_passed": False,
                "reason": "target absent from current view",
                "latency_sec": 0.01,
                "backend_call_count": 1,
                "backend_error_count": 0,
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([]),
            stop_verifier=verifier,
        )
        planner.reset("chair")
        self.assertIsNotNone(planner.memory_sidecar)
        planner.memory_sidecar.last_verified_target_evidence = {
            "updated": True,
            "node_id": "n_target",
            "target_object": "chair",
            "frame_index": 20,
            "position_xyz": [0.0, 0.0, 1.0],
            "target_bearing_world_rad": math.pi / 2.0,
        }
        planner._target_lock_remaining_cycles = 4
        planner._target_lock_reacquisition_attempts = 1
        planner.update_runtime_context(
            frame_index=22,
            position_xyz=[0.0, 0.0, 0.0],
            heading_rad=0.0,
        )

        with patch.dict(
            os.environ,
            {"VOCA_TARGET_LOCK_FULL_SWEEP_AFTER": "2"},
        ):
            planner.make_plan_from_views(
                [np.zeros((20, 20, 3), dtype=np.uint8)],
                [0],
                call_type="anchor_escalation",
            )

        self.assertEqual(planner.last_decision["action"], "rotate")
        self.assertTrue(
            planner.last_decision["verified_target_anchor_reacquisition"]
        )
        self.assertEqual(planner.last_decision["Angle"], 90)
        self.assertEqual(planner._target_anchor_reacquisition_count, 1)

    def test_verified_target_lookalikes_do_not_reset_full_sweep_escalation(self):
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": False,
                "target_visible": True,
                "target_match": False,
                "close_enough": False,
                "confidence": "high",
                "target_evidence_passed": False,
                "identity_critic": {
                    "triggered": True,
                    "passed": False,
                    "exact_target": False,
                    "lookalike_type": "cabinet",
                    "reason": "the selected crop is nearby furniture",
                },
                "reason": "identity critic rejected current crop",
                "latency_sec": 0.01,
                "backend_call_count": 1,
                "backend_error_count": 0,
            }
        )

        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([]),
            stop_verifier=verifier,
        )
        planner.reset("chair")
        self.assertIsNotNone(planner.memory_sidecar)
        planner.memory_sidecar.last_verified_target_evidence = {
            "updated": True,
            "frame_index": 20,
            "position_xyz": [0.0, 0.0, 0.0],
            "target_object": "chair",
        }
        image = np.zeros((20, 20, 3), dtype=np.uint8)

        with patch.dict(
            os.environ,
            {"VOCA_TARGET_LOCK_FULL_SWEEP_AFTER": "2"},
        ):
            planner.update_runtime_context(
                frame_index=22,
                position_xyz=[0.0, 0.0, 0.0],
            )
            planner.make_plan_from_views(
                [image],
                [0],
                call_type="lookalike_reacquire_directed",
            )
            first_request = planner.last_decision["vlm_output"][
                "observation_request"
            ]

            planner.update_runtime_context(
                frame_index=23,
                position_xyz=[0.0, 0.0, 0.0],
            )
            planner.make_plan_from_views(
                [image],
                [0],
                call_type="lookalike_reacquire_full",
            )
            second_request = planner.last_decision["vlm_output"][
                "observation_request"
            ]

        self.assertEqual(first_request["mode"], "directed_sweep")
        self.assertEqual(second_request["mode"], "full_sweep")
        self.assertEqual(planner._target_lock_reacquisition_attempts, 2)

    def test_identity_rejection_cooldown_skips_repeat_backend_call_until_moved(self):
        verifier = FakeStopVerifier(
            {
                "triggered": True,
                "passed": False,
                "target_visible": True,
                "target_match": True,
                "target_evidence_passed": False,
                "identity_critic": {
                    "triggered": True,
                    "passed": False,
                    "lookalike_type": "incline workout bench",
                    "reason": "known chair lookalike",
                },
                "confidence": "high",
                "reason": "identity critic rejected target",
                "latency_sec": 0.01,
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([]),
            stop_verifier=verifier,
        )
        planner.memory_sidecar = None
        planner.reset("chair")
        planner._runtime_frame_index = 10
        planner._runtime_position_xyz = [0.0, 0.0, 0.0]
        vlm_input = {"metadata": {"raw_observation_images": ["fake.jpg"]}}

        first = planner._verify_stop_if_needed(
            safe_output={"action": "stop"},
            vlm_input=vlm_input,
        )
        planner._runtime_frame_index = 11
        second = planner._verify_stop_if_needed(
            safe_output={"action": "stop"},
            vlm_input=vlm_input,
        )

        self.assertFalse(first["passed"])
        self.assertTrue(second["cached_identity_rejection"])
        self.assertEqual(len(verifier.calls), 1)
        self.assertEqual(planner._identity_rejection_cooldown_skip_count, 1)

        planner._runtime_frame_index = 12
        planner._runtime_position_xyz = [2.0, 0.0, 0.0]
        third = planner._verify_stop_if_needed(
            safe_output={"action": "stop"},
            vlm_input=vlm_input,
        )

        self.assertFalse(third.get("cached_identity_rejection", False))
        self.assertEqual(len(verifier.calls), 2)

    def test_target_stop_verifier_uses_small_target_bbox_profile_for_houseplant(self):
        modules = import_nav_memory_qwen()
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"target_visible":true,"target_match":true,'
                            '"close_enough":true,"confidence":"high",'
                            '"target_view_index":0,"target_center_x":0.6,'
                            '"target_bbox_norm":[0.55,0.55,0.65,0.65],'
                            '"reason":"houseplant visible and close"}'
                        )
                    },
                }
            ]
        }
        base = FakeOpenAIClient([response])
        verifier = QwenTargetStopVerifier(CompactQwenNavVLMClient(base, modules), modules)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plant.jpg"
            modules.utils.Image.new("RGB", (40, 30), color=(0, 180, 0)).save(path)
            with patch.dict(os.environ, {"VOCA_QWEN_STOP_IDENTITY_CRITIC": "0"}):
                result = verifier.verify(target="plant", image_paths=[str(path)])

        self.assertTrue(result["passed"])
        self.assertEqual(result["target_bbox_threshold_profile"], "small_physical_target")
        self.assertEqual(result["target_min_bbox_height"], 0.04)

    def test_small_target_bbox_profile_still_rejects_edge_grounding(self):
        modules = import_nav_memory_qwen()
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"target_visible":true,"target_match":true,'
                            '"close_enough":true,"confidence":"high",'
                            '"target_view_index":0,"target_center_x":0.2,'
                            '"target_bbox_norm":[0.15,0.55,0.25,0.65],'
                            '"reason":"houseplant visible at image edge"}'
                        )
                    },
                }
            ]
        }
        base = FakeOpenAIClient([response])
        verifier = QwenTargetStopVerifier(CompactQwenNavVLMClient(base, modules), modules)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "edge_plant.jpg"
            modules.utils.Image.new("RGB", (40, 30), color=(0, 180, 0)).save(path)
            result = verifier.verify(target="houseplant", image_paths=[str(path)])

        self.assertFalse(result["passed"])
        self.assertFalse(result["target_center_guard_satisfied"])
        self.assertTrue(result["target_bbox_guard_satisfied"])

    def test_selected_point_visual_risk_separates_blank_wall_from_textured_floor(self):
        blank_wall = np.full((100, 200, 3), 208, dtype=np.uint8)
        textured_floor = np.zeros((100, 200, 3), dtype=np.uint8)
        textured_floor[:, ::2] = 220

        wall_risk = selected_point_visual_risk(blank_wall, [100, 90])
        floor_risk = selected_point_visual_risk(textured_floor, [100, 90])

        self.assertTrue(wall_risk["requires_verification"])
        self.assertIn("low_texture", wall_risk["risk_reasons"])
        self.assertFalse(floor_risk["requires_verification"])

    def test_ambiguous_selected_point_is_rejected_by_rgb_verifier_before_pixelnav(self):
        modules = import_nav_memory_qwen()
        output = modules.schema.make_go_output(
            view_id=0,
            view_type="front",
            point_px=(10, 16),
            width=20,
            height=20,
            decision_reason="G02_VISIBLE_FLOOR_TOWARD_GOAL",
            goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
            short_text="blank region might be floor",
            confidence="medium",
            selected_candidate_ref="px_v00_r1_c1",
        )
        verifier = FakePointVerifier(
            {
                "schema_version": "qwen_rgb_point_verification_v1",
                "triggered": True,
                "passed": False,
                "valid": False,
                "surface": "unknown",
                "confidence": "low",
                "reason": "no clear floor support",
                "latency_sec": 0.05,
            },
            delay_sec=0.05,
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([output]),
            point_verifier=verifier,
        )
        planner.reset("chair")

        planner.make_plan_from_views(
            [np.full((20, 20, 3), 208, dtype=np.uint8)],
            [0],
            call_type="qwen_action_loop",
        )

        self.assertEqual(len(verifier.calls), 1)
        self.assertEqual(planner.last_decision["action"], "request_observation")
        self.assertFalse(planner.last_decision["selected_point_verification"]["passed"])
        self.assertIn("point_verifier", planner.last_decision["fallback_reason"])
        self.assertEqual(planner.llm_call_count, 2)
        self.assertEqual(len(planner.llm_durations), 2)
        self.assertLess(sum(planner.llm_durations), 0.075)

    def test_textured_selected_point_skips_optional_rgb_verifier(self):
        modules = import_nav_memory_qwen()
        output = modules.schema.make_go_output(
            view_id=0,
            view_type="front",
            point_px=(10, 16),
            width=20,
            height=20,
            decision_reason="G02_VISIBLE_FLOOR_TOWARD_GOAL",
            goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
            short_text="textured doorway floor",
            confidence="medium",
            selected_candidate_ref="px_v00_r1_c1",
        )
        verifier = FakePointVerifier({"passed": False})
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([output]),
            point_verifier=verifier,
        )
        planner.reset("chair")
        textured = np.zeros((20, 20, 3), dtype=np.uint8)
        textured[:, ::2] = 220

        planner.make_plan_from_views([textured], [0], call_type="qwen_action_loop")

        self.assertEqual(verifier.calls, [])
        self.assertEqual(planner.last_decision["action"], "go")
        self.assertFalse(planner.last_decision["selected_point_verification"]["triggered"])

    def test_valid_candidate_ref_canonicalizes_qwen_point_before_pixelnav(self):
        modules = import_nav_memory_qwen()
        output = modules.schema.make_go_output(
            view_id=0,
            view_type="front",
            point_px=(1, 2),
            width=16,
            height=12,
            decision_reason="G02_VISIBLE_FLOOR_TOWARD_GOAL",
            goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
            short_text="choose marked floor",
            confidence="medium",
            selected_candidate_ref="px_v00_r0_c0",
        )
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([output]))
        planner.reset("chair")

        _goal_rgb, goal_mask, _debug, _vis, _direction, _pri, _detected = planner.make_plan_from_views(
            [np.zeros((12, 16, 3), dtype=np.uint8)],
            [0],
            call_type="candidate_gate_test",
        )

        self.assertEqual(planner.last_decision["action"], "go")
        self.assertEqual(planner.last_decision["selected_candidate_ref"], "px_v00_r0_c0")
        self.assertTrue(planner.last_decision["selected_topological_candidate_ref"].startswith("exit_"))
        self.assertEqual(planner.last_decision["Point"], [4, 11])
        self.assertEqual(planner.last_decision["vlm_output"]["selected_image_point"], [4, 11])
        self.assertTrue(planner.last_decision["pixel_candidate_validation"]["passed"])
        self.assertTrue(planner.last_decision["pixel_candidate_validation"]["coordinate_corrected"])
        self.assertEqual(goal_mask[11, 4], 255)

    def test_missing_candidate_ref_becomes_observation_request_not_blind_go(self):
        modules = import_nav_memory_qwen()
        output = modules.schema.make_go_output(
            view_id=0,
            view_type="front",
            point_px=(4, 6),
            width=16,
            height=12,
            decision_reason="G02_VISIBLE_FLOOR_TOWARD_GOAL",
            goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
            short_text="free form point without ref",
            confidence="medium",
        )
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([output]))
        planner.reset("chair")

        planner.make_plan_from_views(
            [np.zeros((12, 16, 3), dtype=np.uint8)],
            [0],
            call_type="candidate_gate_test",
        )

        self.assertEqual(planner.last_decision["action"], "request_observation")
        self.assertEqual(planner.last_decision["vlm_output"]["action"], "request_observation")
        self.assertFalse(planner.last_decision["pixel_candidate_validation"]["passed"])
        self.assertEqual(
            planner.last_decision["pixel_candidate_validation"]["reason"],
            "missing_selected_candidate_ref",
        )
        self.assertIn("pixel_candidate_gate", planner.last_decision["fallback_reason"])

    def test_vlm_input_uses_marked_rgb_and_preserves_raw_memory_evidence(self):
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.reset("chair")

        with tempfile.TemporaryDirectory() as tmpdir:
            planner.set_qwen_call_log_path(str(Path(tmpdir) / "qwen_calls.jsonl"))
            vlm_input = planner._build_vlm_input(
                [np.zeros((100, 200, 3), dtype=np.uint8)],
                [0],
                "pixel_candidate_input_test",
            )
            raw_path = Path(vlm_input["metadata"]["raw_observation_images"][0])
            marked_path = Path(vlm_input["observation"]["views"][0]["image"])
            prompt = _compact_nav_prompt(vlm_input)
            retry_prompt = _compact_nav_retry_prompt(vlm_input)

            self.assertTrue(raw_path.exists())
            self.assertTrue(marked_path.exists())
            self.assertNotEqual(raw_path.read_bytes(), marked_path.read_bytes())

        candidates = vlm_input["pixel_candidates"]["candidates"]
        self.assertEqual(vlm_input["pixel_candidates"]["schema_version"], "rgb_pixel_candidates_v1")
        self.assertTrue(vlm_input["pixel_candidates"]["required_for_go"])
        self.assertEqual(len(candidates), 6)
        self.assertEqual(candidates[0]["candidate_ref"], "px_v00_r0_c0")
        self.assertEqual(candidates[0]["marker_id"], "M01")
        self.assertEqual(candidates[0]["point_px"], [50, 96])
        self.assertIn("topological_candidate_refs", vlm_input["memory"])
        self.assertEqual(
            vlm_input["memory"]["candidate_refs"]["exits"],
            candidates,
        )
        for text in (prompt, retry_prompt):
            self.assertIn("Pixel waypoint candidates", text)
            self.assertIn("M01 ref=px_v00_r0_c0", text)
            self.assertIn("point=[50,96]", text)
            self.assertIn("selected_candidate_ref", text)
            self.assertIn("Never invent a point", text)
            self.assertNotIn(str(raw_path), text)
            self.assertNotIn(str(marked_path), text)

    def test_vlm_input_blends_and_exposes_attached_pixelnav_candidate_scores(self):
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.reset("chair")

        def scorer(_images, candidates):
            return [
                {
                    "candidate_ref": item["candidate_ref"],
                    "policy_feasibility_score": 0.9 if index == 1 else 0.2,
                    "predicted_action": "forward" if index == 1 else "look_down",
                }
                for index, item in enumerate(candidates)
            ]

        planner.set_pixel_candidate_scorer(scorer)
        vlm_input = planner._build_vlm_input(
            [np.zeros((100, 200, 3), dtype=np.uint8)],
            [0],
            "pixelnav_candidate_score_test",
        )

        block = vlm_input["pixel_candidates"]
        best = block["candidates"][0]
        self.assertEqual(block["policy_conditioning"]["status"], "available")
        self.assertEqual(block["policy_conditioning"]["scored_count"], 6)
        self.assertEqual(best["pixelnav_feasibility"]["predicted_action"], "forward")
        self.assertEqual(best["score_source"], "rgb_memory_pixelnav_blend_v1")
        prompt = _compact_nav_prompt(vlm_input)
        self.assertIn("pixelnav=0.9", prompt)
        self.assertIn("policy_action=forward", prompt)

    def test_pixelnav_immediate_stop_candidate_is_audit_only_not_vlm_visible(self):
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.reset("chair")

        def scorer(_images, candidates):
            records = []
            for index, item in enumerate(candidates):
                stop = index == 0
                records.append(
                    {
                        "candidate_ref": item["candidate_ref"],
                        "policy_feasibility_score": 0.0 if stop else 0.8,
                        "predicted_action": "stop" if stop else "forward",
                        "stop_probability": 0.9 if stop else 0.05,
                        "locomotion_probability": 0.02 if stop else 0.8,
                    }
                )
            return records

        planner.set_pixel_candidate_scorer(scorer)
        vlm_input = planner._build_vlm_input(
            [np.zeros((100, 200, 3), dtype=np.uint8)],
            [0],
            "pixelnav_stop_exclusion_test",
        )

        block = vlm_input["pixel_candidates"]
        visible_refs = {item["candidate_ref"] for item in block["candidates"]}
        excluded = {
            item["candidate_ref"]: item for item in block["excluded_candidates"]
        }
        self.assertNotIn("px_v00_r0_c0", visible_refs)
        self.assertEqual(
            excluded["px_v00_r0_c0"]["exclusion_reason"],
            "pixelnav_immediate_stop_risk",
        )
        self.assertEqual(block["policy_conditioning"]["policy_excluded_count"], 1)

    def test_negative_memory_saturation_opens_only_verified_far_floor_probes(self):
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.reset("toilet")
        headings = [-120, -60, 0, 60, 120]
        blocked = [
            {
                "candidate_ref": "px_v{:02d}_r0_c0".format(view_id),
                "view_id": view_id,
                "point_px": [50, 96],
                "avoid": True,
                "status": "verified_failed_pixel_neighborhood",
            }
            for view_id in range(len(headings))
        ]
        blocked_count = len(blocked)

        with patch.dict(
            os.environ,
            {"VOCA_NEGATIVE_SATURATION_RECOVERY": "1"},
            clear=False,
        ), patch(
            "qwen_vlm_planner.build_pixel_candidates",
            return_value=blocked,
        ):
            vlm_input = planner._build_vlm_input(
                [
                    np.zeros((100, 200, 3), dtype=np.uint8)
                    for _ in headings
                ],
                headings,
                "negative_memory_saturation_test",
            )

        block = vlm_input["pixel_candidates"]
        recovery = block["candidates"]
        audit = block["policy_conditioning"]["negative_saturation_recovery"]
        self.assertTrue(audit["triggered"])
        self.assertEqual(audit["negative_exclusion_count"], blocked_count)
        self.assertEqual(audit["offered_count"], 15)
        self.assertEqual(audit["executable_after_pixelnav_count"], 15)
        self.assertEqual(len(recovery), 15)
        self.assertTrue(
            all(item["negative_memory_saturation_recovery"] for item in recovery)
        )
        self.assertTrue(all(item["requires_rgb_verification"] for item in recovery))
        self.assertTrue(all(item["topological_candidate_ref"] is None for item in recovery))
        self.assertEqual(len(block["excluded_candidates"]), blocked_count)

        selected = next(item for item in recovery if item["marker_id"] == "S11")
        validation = validate_pixel_candidate_selection(
            {
                "action": "go",
                "selected_candidate_ref": "S11",
                "selected_view_id": selected["view_id"],
                "selected_view_type": selected["view_type_hint"],
                "selected_image_point": [1, 2],
            },
            vlm_input,
        )
        self.assertTrue(validation["passed"])
        self.assertTrue(validation["ref_alias_applied"])
        self.assertEqual(
            validation["canonical_output"]["selected_candidate_ref"],
            selected["candidate_ref"],
        )
        self.assertEqual(
            validation["canonical_output"]["selected_image_point"],
            selected["point_px"],
        )

    def test_negative_saturation_probe_forces_verifier_on_textured_rgb(self):
        verifier = FakePointVerifier(
            {
                "schema_version": "qwen_rgb_point_verification_v1",
                "triggered": True,
                "passed": True,
                "valid": True,
                "surface": "floor",
                "confidence": "high",
                "reason": "far probe lands on traversable floor",
                "latency_sec": 0.01,
            }
        )
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([]),
            point_verifier=verifier,
        )
        planner.reset("toilet")
        textured = np.zeros((100, 200, 3), dtype=np.uint8)
        textured[:, ::2] = 220
        self.assertFalse(
            selected_point_visual_risk(textured, [100, 71])["requires_verification"]
        )
        candidate = {
            "candidate_ref": "px_v00_r2_c1",
            "view_id": 0,
            "point_px": [100, 71],
            "negative_memory_saturation_recovery": True,
            "requires_rgb_verification": True,
        }

        result = planner._verify_selected_point_if_needed(
            safe_output={"action": "go", "strategic_state": {}},
            pixel_candidate_validation={"passed": True, "candidate": candidate},
            pano_images=[textured],
        )

        self.assertEqual(len(verifier.calls), 1)
        self.assertTrue(result["triggered"])
        self.assertTrue(result["passed"])
        self.assertEqual(result["candidate_ref"], candidate["candidate_ref"])

    def test_compact_prompts_use_same_verified_memory_projection(self):
        vlm_input = {
            "task": {
                "target_object": "chair",
                "target_context": {"Gateways": ["living room doorway"]},
            },
            "observation": {
                "image_width": 640,
                "image_height": 480,
                "views": [
                    {
                        "view_id": 0,
                        "view_type": "front",
                        "relative_heading_deg": 0,
                        "image": "/tmp/private-current.jpg",
                    }
                ],
            },
            "memory": {
                "schema_version": "nav_memory_context_v6",
                "goal_context": {"supervisor_mode": "escape_deadlock"},
                "policy_harness_state": {"current_stage": "escape_deadlock"},
                "graph_summary": {"num_nodes": 3, "num_edges": 2, "num_deadlock_edges": 1},
                "deadlock_state": {"status": "confirmed", "incoming_edge_id": "edge_failed"},
                "candidate_refs": {
                    "exits": [
                        {
                            "candidate_ref": "exit_left",
                            "view_type_hint": "left",
                            "bearing_deg_robot": -90,
                            "status": "unknown_frontier",
                            "avoid": False,
                            "score": 0.8,
                            "reason": "alternate doorway",
                        }
                    ]
                },
                "voca_sidecar": {
                    "no_progress_count": 2,
                    "supervisor_mode": "escape_deadlock",
                    "position_xyz": [20.0, 0.0, 30.0],
                    "directional_failures": [
                        {
                            "frame_index": 7,
                            "angle_deg": 0,
                            "selected_view_id": 0,
                            "point_px": [320, 360],
                            "failure_class": "collision_blocked",
                            "collision_count": 1,
                            "translation_m": 0.0,
                        }
                    ],
                },
                "event_log": [{"event_type": "private_event"}],
            },
            "runtime": {
                "recent_navigation_feedback": [
                    {"event": "raw_runtime_should_not_appear", "point_px": [111, 222]}
                ]
            },
        }

        expected_memory = compact_memory_json(vlm_input)
        prompt = _compact_nav_prompt(vlm_input)
        retry_prompt = _compact_nav_retry_prompt(vlm_input)

        for text in (prompt, retry_prompt):
            self.assertIn("Verified navigation memory: {}".format(expected_memory), text)
            self.assertIn('"supervisor_mode":"escape_deadlock"', text)
            self.assertIn('"failure_class":"collision_blocked"', text)
            self.assertIn('"candidate_ref":"exit_left"', text)
            self.assertIn("Temporary motion opposite", text)
            self.assertNotIn("raw_runtime_should_not_appear", text)
            self.assertNotIn("private_event", text)
            self.assertNotIn("/tmp/private-current.jpg", text)
            self.assertNotIn("position_xyz", text)

    def test_vlm_input_audits_exact_compact_memory_projection_and_digest(self):
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.reset("chair")
        planner.memory_sidecar.update_supervisor_mode("escape_deadlock")
        planner.memory_sidecar.directional_failures.append(
            {
                "frame_index": 3,
                "angle_deg": 0.0,
                "selected_view_id": 0,
                "point_px": [320, 360],
                "failure_class": "no_progress",
                "collision_count": 0,
                "translation_m": 0.0,
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            planner.set_qwen_call_log_path(str(Path(tmpdir) / "qwen_calls.jsonl"))
            first = planner._build_vlm_input(
                [np.zeros((12, 16, 3), dtype=np.uint8)],
                [0],
                "memory_digest_test",
            )
            first_json = compact_memory_json(first)
            first_digest = hashlib.sha256(first_json.encode("utf-8")).hexdigest()

            planner.memory_sidecar.update_supervisor_mode("goal_seek", reason="test_transition")
            second = planner._build_vlm_input(
                [np.zeros((12, 16, 3), dtype=np.uint8)],
                [0],
                "memory_digest_test",
            )

        self.assertEqual(first["metadata"]["compact_memory_projection"], json.loads(first_json))
        self.assertEqual(first["metadata"]["compact_memory_sha256"], first_digest)
        self.assertTrue(first["metadata"]["compact_memory_used"])
        self.assertNotEqual(
            first["metadata"]["compact_memory_sha256"],
            second["metadata"]["compact_memory_sha256"],
        )

    def test_vlm_input_excludes_runtime_evaluation_oracles(self):
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.reset("chair")

        with tempfile.TemporaryDirectory() as tmpdir:
            planner.set_qwen_call_log_path(str(Path(tmpdir) / "qwen_calls.jsonl"))
            planner.update_runtime_context(
                position_xyz=[1.0, 0.0, 2.0],
                metrics={
                    "distance_to_goal": 4.321987,
                    "success": 0.0,
                    "spl": 0.0,
                    "top_down_map": "evaluation-only",
                },
            )
            vlm_input = planner._build_vlm_input(
                [np.zeros((12, 16, 3), dtype=np.uint8)],
                [0],
                "policy_boundary_test",
            )

        serialized = json.dumps(vlm_input, sort_keys=True)
        self.assertNotIn("distance_to_goal", serialized)
        self.assertNotIn("4.321987", serialized)
        self.assertNotIn("top_down_map", serialized)
        self.assertFalse(vlm_input["task"]["coarse_goal"]["goal_geometry_available"])
        self.assertIsNone(vlm_input["task"]["coarse_goal"]["relative_bearing_deg"])
        self.assertIsNone(vlm_input["task"]["coarse_goal"]["distance_m"])
        self.assertFalse(vlm_input["memory"]["goal_context"]["goal_geometry_available"])
        self.assertIsNone(vlm_input["memory"]["goal_context"]["goal_distance_m"])
        self.assertEqual(
            vlm_input["metadata"]["localization_contract"]["source"],
            "none",
        )

    def test_declared_policy_pose_populates_vlm_robot_state(self):
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.localization_contract = {
            "source": "habitat_sim_pose_declared",
            "pose_available_to_policy": True,
            "uses_sim_ground_truth_pose": True,
        }
        planner.reset("chair")
        planner.update_runtime_context(
            position_xyz=[1.25, 0.1, -2.5],
            heading_rad=0.75,
            frame_index=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            planner.set_qwen_call_log_path(str(Path(tmpdir) / "qwen_calls.jsonl"))
            vlm_input = planner._build_vlm_input(
                [np.zeros((12, 16, 3), dtype=np.uint8)],
                [0],
                "declared_pose_contract_test",
            )

        self.assertEqual(vlm_input["robot_state"]["position_xyz"], [1.25, 0.1, -2.5])
        self.assertEqual(vlm_input["robot_state"]["map_xy"], [1.25, -2.5])
        self.assertEqual(vlm_input["robot_state"]["heading_rad"], 0.75)

    def test_vlm_input_and_prompt_include_declared_spatial_coarse_goal(self):
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.reset("chair")
        planner.update_runtime_context(
            spatial_coarse_goal={
                "type": "relative_waypoint",
                "source": "upstream_global_planner",
                "relative_bearing_deg": 70.0,
                "distance_range_m": [3.0, 6.0],
                "uncertainty": "high",
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            planner.set_qwen_call_log_path(str(Path(tmpdir) / "qwen_calls.jsonl"))
            vlm_input = planner._build_vlm_input(
                [np.zeros((12, 16, 3), dtype=np.uint8)],
                [0],
                "spatial_coarse_goal_test",
            )

        coarse = vlm_input["task"]["coarse_goal"]
        self.assertEqual(vlm_input["task"]["task_mode"], "CoarseGoalNav")
        self.assertEqual(coarse["relative_bearing_deg"], 70.0)
        self.assertEqual(coarse["distance_range_m"], [3.0, 6.0])
        memory_goal = vlm_input["memory"]["goal_context"]
        self.assertEqual(memory_goal["task_mode"], "CoarseGoalNav")
        self.assertTrue(memory_goal["goal_geometry_available"])
        self.assertEqual(memory_goal["goal_bearing_from_current_deg"], 70.0)
        self.assertEqual(memory_goal["coarse_goal"], coarse)
        prompt = _compact_nav_prompt(vlm_input)
        self.assertIn("source=upstream_global_planner", prompt)
        self.assertIn("relative_bearing_deg=70.0", prompt)
        self.assertIn("temporary_detour_allowed=True", prompt)
        self.assertIn("soft objective", prompt)

    def test_compact_client_parses_json_content_without_memory_heavy_prompt(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": '{"schema_version":"nav_vlm_waypoint_v1","action":"rotate"}',
                        "reasoning": "long private analysis that should not be parsed first",
                    }
                }
            ]
        }
        base_client = FakeOpenAIClient([response])
        client = CompactQwenNavVLMClient(base_client, import_nav_memory_qwen())

        output = client.decide(
            {
                "task": {"target_object": "toilet", "target_context": {"Gateways": ["bathroom doorway"]}},
                "observation": {
                    "image_width": 640,
                    "image_height": 480,
                    "views": [{"view_id": 0, "view_type": "front", "relative_heading_deg": 0, "image": "<test>"}],
                },
            }
        )

        prompt_text = base_client.payloads[0]["messages"][1]["content"][0]["text"]
        system_text = base_client.payloads[0]["messages"][0]["content"]
        self.assertEqual(output["schema_version"], "nav_vlm_waypoint_v1")
        self.assertEqual(output["action"], "rotate")
        self.assertTrue(prompt_text.startswith("/no_think"))
        self.assertIn("Do not include reasoning", system_text)
        self.assertIn("Target object: toilet", prompt_text)
        self.assertIn("Target context cues: Gateways=bathroom doorway", prompt_text)
        self.assertNotIn("{'Gateways'", prompt_text)
        self.assertIn("Return ONLY final JSON", prompt_text)
        self.assertIn("Return one JSON object only", prompt_text)
        self.assertIn("Do not describe the views", prompt_text)
        self.assertIn("Prefer action go", prompt_text)
        self.assertIn("Return action stop", prompt_text)
        self.assertIn("target object is clearly visible", prompt_text)
        self.assertIn("outer 10% left/right image edges", prompt_text)
        self.assertIn("visible doorway", prompt_text)
        self.assertIn("Rotate only", prompt_text)
        self.assertIn("request_observation", prompt_text)
        self.assertIn("directed_sweep", prompt_text)
        self.assertIn("Avoid repeating rotate", prompt_text)
        self.assertIn("When multiple views are provided", prompt_text)
        self.assertIn("do not request_observation again", prompt_text)
        self.assertNotIn("Place recognition", prompt_text)
        self.assertNotIn("Deadlock memory policy", prompt_text)

    def test_compact_client_posts_chat_completion_for_v6_client_shape(self):
        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"schema_version":"nav_vlm_waypoint_v1","action":"rotate"}'
                            }
                        }
                    ]
                }

        posted = []

        def fake_post(url, headers, json, timeout):
            posted.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
            return FakeResponse()

        base_client = FakeOpenAIClientV6Shape()
        client = CompactQwenNavVLMClient(base_client, import_nav_memory_qwen())

        with patch("qwen_vlm_planner.requests.post", side_effect=fake_post):
            output = client.decide(
                {
                    "task": {"target_object": "chair", "target_context": {}},
                    "observation": {
                        "image_width": 640,
                        "image_height": 480,
                        "views": [{"view_id": 0, "view_type": "front", "relative_heading_deg": 0}],
                    },
                }
            )

        self.assertEqual(output["action"], "rotate")
        self.assertEqual(posted[0]["url"], "http://qwen.test/v1/chat/completions")
        self.assertEqual(posted[0]["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(posted[0]["timeout"], 123)

    def test_compact_client_retry_instruction_rejects_prose_analysis(self):
        modules = import_nav_memory_qwen()
        prose_response = {
            "choices": [
                {
                    "message": {
                        "content": "We are given 5 views. I should inspect the corridor and then choose a point."
                    }
                }
            ]
        }
        json_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            modules.schema.make_rotate_output(
                                yaw_deg=30,
                                reason="R02_NO_VISIBLE_NAVIGABLE_FLOOR",
                                confidence="low",
                            )
                        )
                    }
                }
            ]
        }
        base_client = FakeOpenAIClient([prose_response, json_response])
        base_client.max_json_retries = 1
        client = CompactQwenNavVLMClient(base_client, modules)

        output = client.decide(
            {
                "task": {"target_object": "toilet"},
                "observation": {
                    "image_width": 640,
                    "image_height": 480,
                    "views": [
                        {"view_id": 0, "view_type": "left_far", "relative_heading_deg": -60, "image": "<left_far>"},
                        {"view_id": 1, "view_type": "left", "relative_heading_deg": -30, "image": "<left>"},
                        {"view_id": 2, "view_type": "front", "relative_heading_deg": 0, "image": "<front>"},
                        {"view_id": 3, "view_type": "right", "relative_heading_deg": 30, "image": "<right>"},
                        {"view_id": 4, "view_type": "right_far", "relative_heading_deg": 60, "image": "<right_far>"},
                    ],
                },
            }
        )

        retry_prompt = base_client.payloads[1]["messages"][1]["content"][0]["text"]
        self.assertEqual(output["action"], "rotate")
        self.assertIn("Previous response was prose or invalid JSON", retry_prompt)
        self.assertIn("Do not include analysis", retry_prompt)
        self.assertIn("Do not include reasoning", retry_prompt)
        self.assertIn("STOP JSON shape", retry_prompt)
        self.assertIn("outer 10% left/right image edges", retry_prompt)
        self.assertEqual(base_client.payloads[0]["thinking_token_budget"], 1024)
        self.assertEqual(base_client.payloads[1]["thinking_token_budget"], 256)

    def test_compact_client_retry_uses_short_json_function_prompt(self):
        modules = import_nav_memory_qwen()
        prose_response = {
            "choices": [{"message": {"content": "We are given multiple views and should reason about them."}}]
        }
        json_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            modules.schema.make_rotate_output(
                                yaw_deg=45,
                                reason="R02_NO_VISIBLE_NAVIGABLE_FLOOR",
                                confidence="low",
                            )
                        )
                    }
                }
            ]
        }
        base_client = FakeOpenAIClient([prose_response, json_response])
        base_client.max_json_retries = 1
        client = CompactQwenNavVLMClient(base_client, modules)

        client.decide(
            {
                "task": {"target_object": "toilet", "target_context": {"Gateways": ["bathroom doorway"]}},
                "observation": {
                    "image_width": 640,
                    "image_height": 480,
                    "views": [
                        {"view_id": 0, "view_type": "left", "relative_heading_deg": -60, "image": "<left>"},
                        {"view_id": 1, "view_type": "front", "relative_heading_deg": 0, "image": "<front>"},
                        {"view_id": 2, "view_type": "right", "relative_heading_deg": 60, "image": "<right>"},
                    ],
                },
            }
        )

        first_prompt = base_client.payloads[0]["messages"][1]["content"][0]["text"]
        retry_prompt = base_client.payloads[1]["messages"][1]["content"][0]["text"]
        self.assertIn("JSON function call", retry_prompt)
        self.assertIn("Do not solve in natural language", retry_prompt)
        self.assertIn("If the target object is clearly visible", retry_prompt)
        self.assertIn("Target context cues: Gateways=bathroom doorway", retry_prompt)
        self.assertLess(len(retry_prompt), len(first_prompt))
        self.assertNotIn("Use this exact schema shape for go", retry_prompt)

    def test_compact_client_places_json_only_reminder_after_images(self):
        modules = import_nav_memory_qwen()
        response = {
            "choices": [
                {
                    "message": {
                        "content": '{"schema_version":"nav_vlm_waypoint_v1","action":"rotate"}',
                    }
                }
            ]
        }
        base_client = FakeOpenAIClientWithImage([response])
        client = CompactQwenNavVLMClient(base_client, modules)

        with patch.object(modules.utils, "image_to_data_url", return_value="data:image/jpeg;base64,test"):
            client.decide(
                {
                    "task": {"target_object": "toilet"},
                    "observation": {
                        "image_width": 640,
                        "image_height": 480,
                        "views": [
                            {"view_id": 0, "view_type": "front", "relative_heading_deg": 0, "image": "/tmp/fake-front.jpg"}
                        ],
                    },
                }
            )

        content = base_client.payloads[0]["messages"][1]["content"]
        self.assertEqual(content[-2]["type"], "image_url")
        self.assertEqual(content[-1]["type"], "text")
        self.assertIn("FINAL ANSWER JSON ONLY", content[-1]["text"])
        self.assertIn("first character must be {", content[-1]["text"])
        self.assertIn("Do not describe the views", content[-1]["text"])

    def test_compact_prompt_omits_runtime_feedback_to_avoid_prose_drift(self):
        vlm_input = {
            "task": {"target_object": "toilet", "target_context": {"Gateways": ["bathroom doorway"]}},
            "observation": {
                "image_width": 640,
                "image_height": 480,
                "views": [
                    {"view_id": 0, "view_type": "left", "relative_heading_deg": -60, "image": "<left>"},
                    {"view_id": 1, "view_type": "front", "relative_heading_deg": 0, "image": "<front>"},
                    {"view_id": 2, "view_type": "right", "relative_heading_deg": 60, "image": "<right>"},
                ],
            },
            "runtime": {
                "recent_navigation_feedback": [
                    {
                        "event": "go_no_progress",
                        "selected_view_id": 1,
                        "selected_view_type": "front",
                        "angle_deg": 0,
                        "point_px": [320, 360],
                        "progress": {"distance_delta_m": -0.02, "min_progress_m": 0.05},
                    }
                ]
            },
        }

        prompt = _compact_nav_prompt(vlm_input)
        retry_prompt = _compact_nav_retry_prompt(vlm_input)

        for text in (prompt, retry_prompt):
            self.assertNotIn("Waypoint blacklist", text)
            self.assertNotIn("reason=no_progress", text)
            self.assertNotIn("point=[320, 360]", text)
            self.assertNotIn("Recent navigation feedback", text)

    def test_compact_client_repairs_fine_goal_fragment_into_go_output(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"valid":true,"view_id":1,"view_type":"front",'
                            '"point_px":[4,5],"point_norm":[0.25,0.45],'
                            '"projected_map_xy":null,"navigability":"likely_free"}'
                        ),
                    }
                }
            ]
        }
        base_client = FakeOpenAIClient([response])
        client = CompactQwenNavVLMClient(base_client, import_nav_memory_qwen())

        output = client.decide(
            {
                "observation": {
                    "image_width": 16,
                    "image_height": 12,
                    "views": [
                        {"view_id": 0, "view_type": "left", "relative_heading_deg": -30, "image": "<left>"},
                        {"view_id": 1, "view_type": "front", "relative_heading_deg": 0, "image": "<front>"},
                    ],
                },
            }
        )

        self.assertEqual(output["schema_version"], "nav_vlm_waypoint_v1")
        self.assertEqual(output["action"], "go")
        self.assertEqual(output["selected_view_id"], 1)
        self.assertEqual(output["selected_view_type"], "front")
        self.assertEqual(output["selected_image_point"], [4, 5])
        self.assertEqual(output["fine_goal"]["point_px"], [4, 5])

    def test_default_planner_uses_compact_qwen_nav_client_for_openai_backend(self):
        modules = import_nav_memory_qwen()
        fake_base = FakeOpenAIClient([])
        fake_base.image_max_side = 1024
        fake_base.jpeg_quality = 85
        fake_base.max_tokens = 4096
        fake_base.max_json_retries = 1

        with patch.object(modules.vlm_client.OpenAICompatibleVLMClient, "from_env", return_value=fake_base):
            planner = QwenVLMPlanner()

        self.assertIsInstance(planner.vlm_client, CompactQwenNavVLMClient)
        self.assertIs(planner.vlm_client.base_client, fake_base)
        self.assertEqual(fake_base.image_max_side, 512)
        self.assertEqual(fake_base.jpeg_quality, 70)
        self.assertEqual(fake_base.max_tokens, 4096)
        self.assertEqual(fake_base.max_json_retries, 1)

    def test_default_direct_vlm_retries_are_shorter_than_legacy_point_mode(self):
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))

        self.assertEqual(planner.config.retries, 1)

    def test_openai_compatible_client_gets_structured_nav_schema_and_thinking_budget(self):
        modules = import_nav_memory_qwen()
        client = modules.vlm_client.OpenAICompatibleVLMClient(
            base_url="http://qwen.test/v1",
            api_key="test-key",
            extra_payload={},
        )

        with patch.dict("os.environ", {}, clear=False):
            QwenVLMPlanner(vlm_client=client)

        response_format = client.extra_payload["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["name"], "nav_vlm_waypoint")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertEqual(client.extra_payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(client.extra_payload["seed"], 20260710)
        schema = response_format["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["schema_version"]["const"], "nav_vlm_waypoint_v1")
        self.assertIn("selected_image_point", schema["properties"])
        self.assertIn("selected_candidate_ref", schema["properties"])
        self.assertIn("selected_candidate_ref", schema["required"])
        self.assertEqual(client.extra_payload["thinking_token_budget"], 1024)
        self.assertNotIn("guided_json", client.extra_payload)

    def test_planner_injects_runtime_navigation_feedback_into_vlm_input(self):
        modules = import_nav_memory_qwen()
        output = modules.schema.make_go_output(
            view_id=0,
            view_type="front",
            point_px=(4, 5),
            width=16,
            height=12,
            decision_reason="G02_VISIBLE_FLOOR_TOWARD_GOAL",
            goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
            short_text="new waypoint after failed progress",
            confidence="medium",
            selected_candidate_ref="px_v00_r0_c0",
        )
        client = FakeVLMClient([output])
        planner = QwenVLMPlanner(vlm_client=client)
        planner.reset("toilet")
        planner.record_navigation_feedback(
            {
                "event": "go_no_progress",
                "selected_view_id": 2,
                "selected_view_type": "front",
                "angle_deg": 30,
                "point_px": [320, 360],
                "progress": {"distance_delta_m": -0.04, "min_progress_m": 0.05},
            }
        )

        planner.make_plan_from_views([np.zeros((12, 16, 3), dtype=np.uint8)], [0], call_type="qwen_action_loop")

        feedback = client.inputs[0]["runtime"]["recent_navigation_feedback"]
        self.assertEqual(feedback[0]["event"], "go_no_progress")
        self.assertEqual(feedback[0]["selected_view_id"], 2)
        self.assertEqual(feedback[0]["point_px"], [320, 360])

    def test_planner_reset_clears_runtime_navigation_feedback(self):
        planner = QwenVLMPlanner(vlm_client=FakeVLMClient([]))
        planner.record_navigation_feedback({"event": "go_no_progress", "point_px": [320, 360]})

        planner.reset("chair")
        vlm_input = planner._build_vlm_input([np.zeros((12, 16, 3), dtype=np.uint8)], [0], "qwen_action_loop")

        self.assertNotIn("recent_navigation_feedback", vlm_input["runtime"])
        strategy = vlm_input["runtime"]["strategic_runtime"]
        self.assertEqual(strategy["same_room_cycles"], 0)
        self.assertEqual(strategy["generic_floor_go_streak"], 0)
        self.assertFalse(strategy["force_leave_room"])

    def test_verified_failed_pixel_candidate_is_rejected_for_observation(self):
        modules = import_nav_memory_qwen()
        output = modules.schema.make_go_output(
            view_id=0,
            view_type="front",
            point_px=(10, 16),
            width=20,
            height=20,
            decision_reason="G02_VISIBLE_FLOOR_TOWARD_GOAL",
            goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
            short_text="same point again",
            confidence="medium",
            selected_candidate_ref="px_v00_r1_c1",
        )
        client = FakeVLMClient([output])
        planner = QwenVLMPlanner(vlm_client=client)
        planner.reset("chair")
        planner.memory_sidecar.build_context(
            target_object="chair",
            priors={},
            image_ref=None,
            frame_index=0,
            image_shape=(20, 20, 3),
            position_xyz=[0.0, 0.0, 0.0],
            heading_rad=0.0,
        )
        planner.memory_sidecar.record_go_execution(
            go_execution={
                "frame_index": 1,
                "selected_view_id": 0,
                "selected_angle_deg": 0,
                "selected_candidate_ref": "px_v00_r1_c1",
                "selected_point_px": [10, 16],
                "failure_classification": {"primary": "no_progress"},
                "collision_count": 0,
                "go_progress": {
                    "execution_success": False,
                    "strategic_progress": False,
                    "translation_m": 0.0,
                },
            },
            frame_index=1,
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.0, 0.0, 0.0],
            start_heading_rad=0.0,
            final_heading_rad=0.0,
        )

        _goal_rgb, goal_mask, _debug_image, _vis_rgb, _direction, _pri_flag, _obj_detected = planner.make_plan_from_views(
            [np.zeros((20, 20, 3), dtype=np.uint8)],
            [0],
            call_type="qwen_action_loop",
        )

        self.assertEqual(planner.last_decision["action"], "request_observation")
        self.assertFalse(planner.last_decision["pixel_candidate_validation"]["passed"])
        self.assertEqual(
            planner.last_decision["pixel_candidate_validation"]["reason"],
            "selected_candidate_excluded",
        )

    def test_make_plan_uses_voca_s2e_client_schema_and_builds_pixelnav_goal(self):
        modules = import_nav_memory_qwen()
        output = modules.schema.make_go_output(
            view_id=1,
            view_type="front",
            point_px=(4, 5),
            width=16,
            height=12,
            decision_reason="G02_VISIBLE_FLOOR_TOWARD_GOAL",
            goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
            short_text="doorway floor likely leads to bathroom",
            confidence="high",
            selected_candidate_ref="px_v01_r0_c0",
        )
        client = FakeVLMClient([output])
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = QwenVLMPlanner(vlm_client=client)
            planner.set_qwen_call_log_path(str(Path(tmpdir) / "qwen_calls.jsonl"))
            planner.reset("toilet")
            planner.latest_priors = {
                "Supports": ["sink"],
                "StrongCooccurs": ["mirror"],
                "Gateways": ["bathroom doorway"],
                "Lookalikes": [],
            }
            planner.memory_sidecar.update_supervisor_mode("escape_deadlock")
            pano = [np.zeros((12, 16, 3), dtype=np.uint8) for _ in range(3)]

            goal_rgb, goal_mask, debug_image, vis_rgb, direction, pri_flag, obj_detected = planner.make_plan(pano)

            lines = (Path(tmpdir) / "qwen_calls.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertTrue(Path(client.inputs[0]["observation"]["views"][0]["image"]).exists())

        self.assertEqual(direction, 1)
        self.assertEqual(goal_rgb.shape, (12, 16, 3))
        self.assertEqual(debug_image.shape, (12, 16, 3))
        self.assertEqual(vis_rgb.shape, (12, 16, 3))
        self.assertEqual(goal_mask[11, 4], 255)
        self.assertFalse(pri_flag)
        self.assertFalse(obj_detected)
        self.assertEqual(len(client.inputs), 1)
        vlm_input = client.inputs[0]
        self.assertEqual(vlm_input["schema_version"], "nav_vlm_waypoint_v1")
        self.assertEqual(vlm_input["task"]["task_mode"], "ObjNav")
        self.assertEqual(vlm_input["task"]["target_context"]["Gateways"], ["bathroom doorway"])
        self.assertEqual(vlm_input["memory"]["schema_version"], "nav_memory_context_v6")
        self.assertIn("candidate_refs", vlm_input["memory"])
        self.assertEqual(vlm_input["memory"]["goal_context"]["supervisor_mode"], "escape_deadlock")
        self.assertEqual(vlm_input["memory"]["policy_harness_state"]["current_stage"], "escape_deadlock")
        self.assertTrue(vlm_input["memory"]["policy_harness_state"]["requirements"]["go_requires_selected_candidate_ref"])
        self.assertEqual(planner.last_decision["supervisor_mode"], "escape_deadlock")
        self.assertEqual(planner.last_decision["schema_version"], "nav_vlm_waypoint_v1")
        self.assertEqual(planner.last_decision["memory_visualizer"]["schema_version"], "nav_memory_context_v6")
        self.assertGreaterEqual(planner.last_decision["memory_visualizer"]["num_nodes"], 1)
        self.assertEqual(planner.last_decision["selected_candidate_ref"], "px_v01_r0_c0")
        # The selected 30-degree view overlaps the 0-degree bearing within the
        # camera coverage threshold, independent of its coarse view label.
        self.assertTrue(
            planner.last_decision["selected_topological_candidate_ref"].startswith(
                "exit_"
            )
        )
        self.assertTrue(planner.last_decision["pixel_candidate_validation"]["passed"])
        self.assertEqual(planner.last_decision["Point"], [4, 11])
        self.assertEqual(planner.last_decision["vlm_output"]["selected_image_point"], [4, 11])
        self.assertEqual(planner.last_decision["Reason"], "doorway floor likely leads to bathroom")
        self.assertEqual(len(lines), 1)
        self.assertIn('"schema_version": "nav_vlm_waypoint_v1"', lines[0])

    def test_rotate_output_reports_commanded_yaw_and_uses_lower_center_fallback_point(self):
        modules = import_nav_memory_qwen()
        output = modules.schema.make_rotate_output(
            yaw_deg=45,
            reason="R02_NO_VISIBLE_NAVIGABLE_FLOOR",
            confidence="low",
        )
        client = FakeVLMClient([output])
        planner = QwenVLMPlanner(vlm_client=client)
        planner.reset("chair")

        planner.make_plan([np.zeros((10, 20, 3), dtype=np.uint8) for _ in range(2)])

        self.assertTrue(planner.last_decision["fallback"])
        self.assertEqual(planner.last_decision["Point"], [10, 8])
        self.assertEqual(planner.last_decision["vlm_output"]["action"], "rotate")
        self.assertEqual(planner.last_decision["Angle"], 45)
        self.assertIn("rotate", planner.last_decision["fallback_reason"])

    def test_strategic_state_survives_sanitizer_and_updates_memory_visualizer(self):
        modules = import_nav_memory_qwen()
        output = modules.schema.make_go_output(
            view_id=0,
            view_type="front",
            point_px=(4, 11),
            width=16,
            height=12,
            decision_reason="G03_DOORWAY_OR_CORRIDOR_TOWARD_GOAL",
            goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
            short_text="leave through the visible doorway",
            confidence="high",
            selected_candidate_ref="px_v00_r0_c0",
        )
        output["strategic_state"] = {
            "current_room": "living_room",
            "target_evidence": "context_only",
            "navigation_intent": "traverse_gateway",
            "room_search_status": "exhausted",
            "selected_exit_ref": "px_v00_r0_c0",
            "reason_code": "VISIBLE_EXIT_SELECTED",
        }
        planner = QwenVLMPlanner(
            vlm_client=FakeVLMClient([output]),
            point_verifier=FakePointVerifier(
                {
                    "triggered": True,
                    "passed": True,
                    "valid": True,
                    "surface": "doorway_floor",
                    "confidence": "high",
                    "reason": "marked point is on the doorway threshold",
                    "latency_sec": 0.01,
                }
            ),
        )
        planner.reset("chair")

        planner.make_plan([np.zeros((12, 16, 3), dtype=np.uint8)])

        state = planner.last_decision["strategic_state"]
        self.assertEqual(state["current_room"], "living_room")
        self.assertEqual(state["navigation_intent"], "traverse_gateway")
        self.assertEqual(
            planner.last_decision["vlm_output"]["strategic_state"],
            state,
        )
        self.assertEqual(
            planner.last_decision["memory_visualizer"]["last_strategic_state"]["current_room"],
            "living_room",
        )
        self.assertEqual(
            planner.last_decision["memory_visualizer"]["graph_layout"]["nodes"][0]["place_category"],
            "living_room",
        )

    def test_go_point_guard_rejects_point_above_floor_band(self):
        self.assertEqual(
            QwenVLMPlanner._go_point_guard_reason([5, 2], (20, 20, 3)),
            "go_point_outside_floor_band",
        )

    def test_go_point_guard_rejects_point_near_image_edge(self):
        self.assertEqual(
            QwenVLMPlanner._go_point_guard_reason([18, 16], (20, 20, 3)),
            "go_point_near_image_edge",
        )

    def test_invalid_json_response_uses_audited_fallback_without_llm_error(self):
        class InvalidJsonClient:
            def decide(self, vlm_input):
                raise ValueError("failed to extract compact VLM JSON after 1 attempt(s): no JSON object found")

        planner = QwenVLMPlanner(vlm_client=InvalidJsonClient(), config=QwenPointPlannerConfig(retries=1))
        planner.reset("chair")

        decision = planner._query_point_decision(
            [np.zeros((20, 20, 3), dtype=np.uint8)],
            [0],
            call_type="qwen_action_loop",
        )

        self.assertEqual(decision["action"], "request_observation")
        self.assertEqual(decision["vlm_output"]["action"], "request_observation")
        self.assertTrue(decision["fallback"])
        self.assertIn("qwen_vlm_invalid_json", decision["fallback_reason"])
        self.assertEqual(
            decision["pixel_candidate_validation"]["reason"],
            "invalid_vlm_json_no_candidate_selection",
        )
        self.assertEqual(planner.llm_error_count, 0)
        self.assertEqual(planner.vlm_json_fallback_count, 1)
        self.assertIn("no JSON object found", planner.vlm_json_last_error)


if __name__ == "__main__":
    unittest.main()
