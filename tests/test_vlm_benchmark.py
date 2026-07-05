import json
import tempfile
import unittest
from pathlib import Path

from goal_adapter.habitat_smoke import write_rgb_png
from goal_adapter.schema import ActionType, Confidence, GoalAdapterOutput
from goal_adapter.vlm_benchmark import (
    evaluate_vlm_benchmark_case,
    load_vlm_benchmark_cases,
    run_vlm_decision_benchmark,
)
from goal_adapter.vlm_benchmark_dataset import write_sample_vlm_benchmark_cases


class FakeRawProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.inputs = []

    def generate_raw_decision(self, adapter_input):
        self.inputs.append(adapter_input)
        if not self.responses:
            raise RuntimeError("no fake response left")
        return self.responses.pop(0)


class FakeDecisionProvider:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.inputs = []

    def decide(self, adapter_input):
        self.inputs.append(adapter_input)
        if not self.outputs:
            raise RuntimeError("no fake output left")
        return self.outputs.pop(0)


class VLMBenchmarkTest(unittest.TestCase):
    def test_sample_dataset_loads_four_required_categories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_input_path = _write_base_input(root)

            case_paths = write_sample_vlm_benchmark_cases(base_input_path, root / "cases")
            cases = load_vlm_benchmark_cases(root / "cases")

            self.assertEqual(len(case_paths), 4)
            self.assertEqual(
                [case.category for case in cases],
                ["coarse_gps", "coarse_object_point", "tracking_loss", "deadlock"],
            )
            self.assertEqual(cases[0].expected_action_type, ActionType.NAVIGATE)
            self.assertEqual(cases[1].expected_goal_xy, [10.5, -6.1])
            self.assertEqual(cases[2].expected_action_type, ActionType.LOOK_AROUND)
            self.assertEqual(cases[2].reasoning_terms, ["tracking_loss"])
            self.assertEqual(cases[3].forbidden_goal_xy, [[10.5, -6.1]])
            self.assertTrue(Path(cases[0].input_json["current_rgb"]).is_absolute())

    def test_benchmark_runs_all_categories_and_writes_reports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_input_path = _write_base_input(root)
            write_sample_vlm_benchmark_cases(base_input_path, root / "cases")
            cases = load_vlm_benchmark_cases(root / "cases")
            provider = FakeRawProvider(
                [
                    json.dumps(
                        {
                            "action_type": "NAVIGATE",
                            "refined_goal_xy": [10.55, -6.05],
                            "selected_image_point": [82, 62],
                            "controller_action": None,
                            "reasoning": "The open floor ahead leads toward the coarse GPS target.",
                            "confidence": "high",
                        }
                    ),
                    json.dumps(
                        {
                            "action_type": "NAVIGATE",
                            "refined_goal_xy": [10.45, -6.08],
                            "selected_image_point": [88, 70],
                            "controller_action": None,
                            "reasoning": "Choose the floor in front of the object, not its center.",
                            "confidence": "high",
                        }
                    ),
                    json.dumps(
                        {
                            "action_type": "LOOK_AROUND",
                            "refined_goal_xy": None,
                            "selected_image_point": None,
                            "controller_action": None,
                            "reasoning": "tracking_loss is active, so look around for a new visual cue.",
                            "confidence": "medium",
                        }
                    ),
                    json.dumps(
                        {
                            "action_type": "RESELECT_GOAL",
                            "refined_goal_xy": [11.25, -5.35],
                            "selected_image_point": [126, 52],
                            "controller_action": None,
                            "reasoning": "Avoid the blocked repeated goal and select a different opening.",
                            "confidence": "medium",
                        }
                    ),
                ]
            )

            summary = run_vlm_decision_benchmark(
                cases=cases,
                output_dir=root / "report",
                decision_provider=provider,
                model_name="fake-vlm",
            )

            self.assertEqual(summary.total_cases, 4)
            self.assertEqual(summary.metrics["case_success_rate"], 1.0)
            self.assertEqual(summary.metrics["action_type_accuracy"], 1.0)
            self.assertEqual(summary.metrics["waypoint_success_rate"], 1.0)
            self.assertEqual(summary.metrics["forbidden_goal_repeat_rate"], 0.0)
            self.assertEqual(set(summary.category_metrics), set(case.category for case in cases))
            self.assertTrue((root / "report" / "summary.json").exists())
            self.assertTrue((root / "report" / "summary.csv").exists())
            self.assertTrue((root / "report" / "index.html").exists())
            self.assertIn(
                "VLM 의사결정 벤치마크",
                (root / "report" / "index.html").read_text(encoding="utf-8"),
            )
            for case in cases:
                case_dir = root / "report" / "cases" / case.case_id
                case_html = (case_dir / "index.html").read_text(encoding="utf-8")
                self.assertTrue((case_dir / "index.html").exists())
                self.assertTrue((case_dir / "current_rgb_overlay.png").exists())
                self.assertTrue((case_dir / "topdown_overlay.png").exists())
                self.assertTrue((case_dir / "goal_adapter_input.json").exists())
                self.assertTrue((case_dir / "vlm_decision.json").exists())
                self.assertTrue((case_dir / "vlm_raw_response.txt").exists())
                self.assertTrue((case_dir / "result.json").exists())
                self.assertIn("RGB 오버레이", case_html)
                self.assertIn("탑다운 오버레이", case_html)
                self.assertIn("시야 범위", case_html)
                self.assertIn("선택 지점 주행 가능", case_html)
            self.assertEqual(len(provider.inputs), 4)
            self.assertTrue(str(provider.inputs[0].current_rgb).endswith("current_rgb.png"))

    def test_benchmark_records_parse_errors_without_stopping_batch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_input_path = _write_base_input(root)
            write_sample_vlm_benchmark_cases(base_input_path, root / "cases")
            cases = load_vlm_benchmark_cases(root / "cases")[:2]
            provider = FakeRawProvider(
                [
                    '{"action_type": "NAVIGATE"}',
                    json.dumps(
                        {
                            "action_type": "NAVIGATE",
                            "refined_goal_xy": [10.5, -6.1],
                            "selected_image_point": [88, 70],
                            "controller_action": None,
                            "reasoning": "Choose the object-front floor point.",
                            "confidence": "high",
                        }
                    ),
                ]
            )

            summary = run_vlm_decision_benchmark(
                cases=cases,
                output_dir=root / "report",
                decision_provider=provider,
                model_name="fake-vlm",
            )

            self.assertEqual(summary.total_cases, 2)
            self.assertEqual(summary.metrics["parse_success_rate"], 0.5)
            self.assertEqual(summary.case_results[0].status, "error")
            self.assertFalse(summary.case_results[0].passed)
            self.assertEqual(summary.case_results[1].status, "success")
            self.assertTrue((root / "report" / "cases" / cases[0].case_id / "vlm_raw_response.txt").exists())

    def test_benchmark_can_retry_empty_vlm_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_input_path = _write_base_input(root)
            write_sample_vlm_benchmark_cases(base_input_path, root / "cases")
            cases = load_vlm_benchmark_cases(root / "cases")[:1]
            provider = FakeRawProvider(
                [
                    "",
                    json.dumps(
                        {
                            "action_type": "NAVIGATE",
                            "refined_goal_xy": [10.5, -6.1],
                            "selected_image_point": [82, 62],
                            "controller_action": None,
                            "reasoning": "The open floor ahead is the best candidate.",
                            "confidence": "high",
                        }
                    ),
                ]
            )

            summary = run_vlm_decision_benchmark(
                cases=cases,
                output_dir=root / "report",
                decision_provider=provider,
                model_name="fake-vlm",
                max_attempts=2,
            )

            self.assertEqual(len(provider.inputs), 2)
            self.assertEqual(summary.metrics["parse_success_rate"], 1.0)
            self.assertEqual(summary.case_results[0].attempts, 2)
            self.assertTrue(summary.case_results[0].passed)

    def test_case_evaluation_fails_when_deadlock_goal_repeats_forbidden_goal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_input_path = _write_base_input(root)
            write_sample_vlm_benchmark_cases(base_input_path, root / "cases")
            deadlock_case = load_vlm_benchmark_cases(root / "cases")[-1]
            output = GoalAdapterOutput(
                refined_goal_xy=[10.5, -6.1],
                selected_image_point=[80, 60],
                action_type=ActionType.RESELECT_GOAL,
                reasoning="The route is blocked.",
                confidence=Confidence.MEDIUM,
            )

            result = evaluate_vlm_benchmark_case(deadlock_case, output)

            self.assertFalse(result.passed)
            self.assertTrue(result.forbidden_goal_repeated)
            self.assertFalse(result.waypoint_success)

    def test_case_evaluation_fails_when_selected_image_point_is_not_navigable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_input_path = _write_base_input(root)
            write_sample_vlm_benchmark_cases(base_input_path, root / "cases")
            object_case = load_vlm_benchmark_cases(root / "cases")[1]
            output = GoalAdapterOutput(
                refined_goal_xy=[10.5, -6.1],
                selected_image_point=[88, 48],
                action_type=ActionType.NAVIGATE,
                reasoning="Select the floor in front of the object.",
                confidence=Confidence.HIGH,
            )

            result = evaluate_vlm_benchmark_case(object_case, output)

            self.assertFalse(result.passed)
            self.assertFalse(result.selected_point_navigable)
            self.assertTrue(result.waypoint_success)

    def test_benchmark_safely_snaps_raw_vlm_wall_selection_to_navigable_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_input_path = _write_base_input(root)
            write_sample_vlm_benchmark_cases(base_input_path, root / "cases")
            object_case = load_vlm_benchmark_cases(root / "cases")[1]
            provider = FakeRawProvider(
                [
                    json.dumps(
                        {
                            "action_type": "NAVIGATE",
                            "refined_goal_xy": [10.9, -6.4],
                            "selected_image_point": [88, 48],
                            "controller_action": None,
                            "reasoning": "The object center is visible.",
                            "confidence": "high",
                        }
                    )
                ]
            )

            summary = run_vlm_decision_benchmark(
                cases=[object_case],
                output_dir=root / "report",
                decision_provider=provider,
                model_name="fake-vlm",
            )

            decision = json.loads(
                (root / "report" / "cases" / object_case.case_id / "vlm_decision.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(summary.case_results[0].passed)
            self.assertEqual(decision["refined_goal_xy"], [10.5, -6.1])
            self.assertEqual(decision["selected_image_point"], [88, 70])
            self.assertIn("safety", decision["reasoning"].lower())

    def test_reasoning_terms_match_space_and_underscore_variants(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_input_path = _write_base_input(root)
            write_sample_vlm_benchmark_cases(base_input_path, root / "cases")
            tracking_loss_case = load_vlm_benchmark_cases(root / "cases")[2]
            output = GoalAdapterOutput(
                refined_goal_xy=None,
                selected_image_point=None,
                action_type=ActionType.LOOK_AROUND,
                reasoning="Target point is missing due to tracking loss.",
                confidence=Confidence.HIGH,
            )

            result = evaluate_vlm_benchmark_case(tracking_loss_case, output)

            self.assertTrue(result.reasoning_correct)
            self.assertTrue(result.passed)


def _write_base_input(root: Path) -> Path:
    image_path = root / "base_rgb.png"
    write_rgb_png(
        [
            [[225, 231, 238] for _column in range(16)]
            for _row in range(12)
        ],
        image_path,
    )
    input_path = root / "goal_adapter_input.json"
    input_path.write_text(
        json.dumps(
            {
                "target_type": "language",
                "high_level_target": "inspect the room",
                "current_rgb": str(image_path),
                "current_pose": {"x": 10.0, "y": -6.8},
                "heading": 0.1,
                "progress_state": "normal",
                "s2e_status": "unknown",
                "memory_summary": {},
            }
        ),
        encoding="utf-8",
    )
    return input_path


if __name__ == "__main__":
    unittest.main()
