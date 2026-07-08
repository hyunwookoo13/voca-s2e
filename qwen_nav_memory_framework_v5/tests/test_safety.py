import unittest

from nav_memory_qwen.safety import sanitize_vlm_output


class SanitizeVLMOutputTest(unittest.TestCase):
    def test_recovers_qwen_go_output_with_nested_selected_image_point(self):
        raw_output = {
            "schema_version": "nav_vlm_waypoint_v1",
            "action": "go",
            "fine_goal": {
                "valid": True,
                "selected_image_point": [120, 240],
                "view_id": 0,
                "reason": "F01_SAFE_LOCAL_FINE_WAYPOINT",
            },
            "reasoning": {
                "reason_code": "F01_SAFE_LOCAL_FINE_WAYPOINT",
                "reason": "The front view shows a clear navigable floor point.",
            },
            "memory_ops": [],
        }
        vlm_input = {
            "observation": {
                "image_width": 640,
                "image_height": 480,
                "views": [
                    {"view_id": 0, "view_type": "front"},
                    {"view_id": 1, "view_type": "left"},
                ],
            }
        }

        safe, warnings = sanitize_vlm_output(raw_output, vlm_input)

        self.assertEqual(safe["action"], "go")
        self.assertEqual(safe["selected_view_id"], 0)
        self.assertEqual(safe["selected_view_type"], "front")
        self.assertEqual(safe["selected_image_point"], [120, 240])
        self.assertEqual(safe["fine_goal"]["point_px"], [120, 240])
        self.assertEqual(safe["memory_ops"], [])
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
