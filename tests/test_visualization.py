import tempfile
import unittest
from pathlib import Path

from goal_adapter.habitat_smoke import PNG_SIGNATURE, write_rgb_png
from goal_adapter.schema import ActionType, Confidence, GoalAdapterOutput
from goal_adapter.visualization import _read_png_rgb, render_case_visualizations
from goal_adapter.vlm_benchmark import VLMDecisionBenchmarkCase


class VisualizationTest(unittest.TestCase):
    def test_render_case_visualizations_writes_rgb_and_topdown_overlays(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "current_rgb.png"
            write_rgb_png(_solid_image(width=16, height=12), image_path)
            case = VLMDecisionBenchmarkCase.from_json(
                {
                    "case_id": "coarse_gps_visual",
                    "category": "coarse_gps",
                    "input": {
                        "target_type": "gps",
                        "high_level_target": {"coarse_goal_xy": [2.0, 1.0]},
                        "current_rgb": str(image_path),
                        "current_pose": {"x": 0.0, "y": 0.0},
                        "heading": 0.0,
                        "current_goal_xy": [2.0, 1.0],
                        "progress_state": "normal",
                        "memory_summary": {
                            "candidate_waypoints": [
                                {
                                    "goal_xy": [1.0, 0.2],
                                    "image_point": [9, 7],
                                    "navigable": True,
                                }
                            ],
                            "failed_goal_xy": [[-0.5, 0.4]],
                        },
                    },
                    "expected": {
                        "action_type": "NAVIGATE",
                        "goal_xy": [1.0, 0.2],
                        "waypoint_tolerance": 0.75,
                    },
                }
            )
            output = GoalAdapterOutput(
                refined_goal_xy=[1.1, 0.1],
                selected_image_point=[10, 7],
                action_type=ActionType.NAVIGATE,
                reasoning="The visible open floor is navigable.",
                confidence=Confidence.HIGH,
            )

            artifacts = render_case_visualizations(case, output, root)

            self.assertEqual(artifacts.rgb_overlay_path, root / "current_rgb_overlay.png")
            self.assertEqual(artifacts.topdown_overlay_path, root / "topdown_overlay.png")
            self.assertTrue(artifacts.rgb_overlay_path.exists())
            self.assertTrue(artifacts.topdown_overlay_path.exists())
            self.assertEqual(artifacts.rgb_overlay_path.read_bytes()[:8], PNG_SIGNATURE)
            self.assertEqual(artifacts.topdown_overlay_path.read_bytes()[:8], PNG_SIGNATURE)

    def test_render_case_visualizations_uses_metric_map_background(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "current_rgb.png"
            metric_map_path = root / "metric_map.png"
            write_rgb_png(_solid_image(width=16, height=12), image_path)
            write_rgb_png(_metric_map_image(width=24, height=18), metric_map_path)
            case = VLMDecisionBenchmarkCase.from_json(
                {
                    "case_id": "metric_map_visual",
                    "category": "coarse_gps",
                    "input": {
                        "target_type": "gps",
                        "high_level_target": {"coarse_goal_xy": [1.8, 1.3]},
                        "current_rgb": str(image_path),
                        "current_pose": {"x": 0.4, "y": 0.3},
                        "heading": 0.2,
                        "current_goal_xy": [1.8, 1.3],
                        "progress_state": "normal",
                        "memory_summary": {
                            "metric_map": {
                                "path": str(metric_map_path),
                                "meters_per_pixel": 0.1,
                                "bounds": {
                                    "min_x": 0.0,
                                    "max_x": 2.4,
                                    "min_y": 0.0,
                                    "max_y": 1.8,
                                },
                            },
                            "candidate_waypoints": [
                                {
                                    "goal_xy": [1.0, 0.8],
                                    "image_point": [9, 7],
                                    "navigable": True,
                                }
                            ],
                        },
                    },
                    "expected": {
                        "action_type": "NAVIGATE",
                        "goal_xy": [1.0, 0.8],
                        "waypoint_tolerance": 0.75,
                    },
                }
            )
            output = GoalAdapterOutput(
                refined_goal_xy=[1.0, 0.8],
                selected_image_point=[9, 7],
                action_type=ActionType.NAVIGATE,
                reasoning="The metric map floor corridor is navigable.",
                confidence=Confidence.HIGH,
            )

            artifacts = render_case_visualizations(case, output, root)
            topdown_image = _read_png_rgb(artifacts.topdown_overlay_path)

            self.assertEqual(len(topdown_image), 18)
            self.assertEqual(len(topdown_image[0]), 24)

    def test_render_case_visualizations_draws_metric_audit_layers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "current_rgb.png"
            metric_map_path = root / "metric_map.png"
            write_rgb_png(_solid_image(width=24, height=18), image_path)
            write_rgb_png(_metric_map_image(width=48, height=48), metric_map_path)
            case = VLMDecisionBenchmarkCase.from_json(
                {
                    "case_id": "metric_audit_visual",
                    "category": "coarse_object_point",
                    "input": {
                        "target_type": "object_point",
                        "high_level_target": {
                            "object_label": "target object",
                            "coarse_image_point": [12, 7],
                        },
                        "current_rgb": str(image_path),
                        "current_pose": {"x": 1.0, "y": 1.0},
                        "heading": 0.0,
                        "current_goal_xy": [3.4, 1.0],
                        "progress_state": "normal",
                        "memory_summary": {
                            "metric_map": {
                                "path": str(metric_map_path),
                                "meters_per_pixel": 0.1,
                                "bounds": {
                                    "min_x": 0.0,
                                    "max_x": 4.8,
                                    "min_y": 0.0,
                                    "max_y": 4.8,
                                },
                            },
                            "camera_fov": {
                                "hfov_degrees": 70.0,
                                "range_meters": 2.2,
                            },
                            "trajectory_xy": [[0.4, 0.4], [0.7, 0.7], [1.0, 1.0]],
                            "candidate_waypoints": [
                                {
                                    "goal_xy": [2.2, 2.5],
                                    "image_point": [12, 13],
                                    "navigable": True,
                                    "rationale": "object-front navigable floor point",
                                },
                                {
                                    "goal_xy": [3.4, 1.0],
                                    "image_point": [12, 7],
                                    "navigable": False,
                                    "rationale": "object visual center is not a floor waypoint",
                                },
                            ],
                        },
                    },
                    "expected": {
                        "action_type": "NAVIGATE",
                        "goal_xy": [2.2, 2.5],
                        "waypoint_tolerance": 0.75,
                    },
                }
            )
            output = GoalAdapterOutput(
                refined_goal_xy=[2.2, 2.5],
                selected_image_point=[12, 13],
                action_type=ActionType.NAVIGATE,
                reasoning="Select the object-front navigable floor point.",
                confidence=Confidence.HIGH,
            )

            artifacts = render_case_visualizations(case, output, root)
            topdown_image = _read_png_rgb(artifacts.topdown_overlay_path)

            self.assertGreater(_count_color(topdown_image, [191, 219, 254]), 0)
            self.assertGreater(_count_color(topdown_image, [34, 197, 94]), 0)
            self.assertGreater(_count_color(topdown_image, [14, 165, 233]), 0)
            self.assertGreater(_count_color(topdown_image, [239, 68, 68]), 0)


def _solid_image(width, height):
    return [
        [[230, 235, 240] for _column in range(width)]
        for _row in range(height)
    ]


def _metric_map_image(width, height):
    image = _solid_image(width, height)
    for row in range(5, 13):
        for column in range(3, 21):
            image[row][column] = [70, 74, 80]
    return image


def _count_color(image, color):
    return sum(1 for row in image for pixel in row if pixel == color)


if __name__ == "__main__":
    unittest.main()
