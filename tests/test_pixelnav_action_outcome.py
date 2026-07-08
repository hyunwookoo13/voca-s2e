import sys
import unittest
import json
from pathlib import Path

from goal_adapter import habitat_pixelnav_execute as hpe
from goal_adapter.habitat_pixelnav_execute import PixelNavRolloutSummary, PixelNavStep


QWEN_ROOT = Path(__file__).resolve().parents[1] / "qwen_nav_memory_framework_v5"
if str(QWEN_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN_ROOT))

from nav_memory_qwen.robot_backend import ActionOutcome


class PixelNavActionOutcomeTest(unittest.TestCase):
    def _summary(
        self,
        *,
        moved_distance_m=0.3,
        stopped_by_policy=False,
        unsupported_action=None,
        collision_count=0,
        steps=None,
    ):
        if steps is None:
            steps = [
                PixelNavStep(
                    step_index=0,
                    pixelnav_action=1,
                    sim_action="move_forward",
                    collision_before=collision_count > 0,
                    position_before_xyz=[0.0, 0.0, 0.0],
                    position_after_xyz=[moved_distance_m, 0.0, 0.0],
                    moved_distance_m=moved_distance_m,
                )
            ]
        return PixelNavRolloutSummary(
            steps=steps,
            total_moved_distance_m=moved_distance_m,
            stopped_by_policy=stopped_by_policy,
            unsupported_action=unsupported_action,
            collision_count=collision_count,
        )

    def test_forward_rollout_builds_successful_go_action_outcome(self):
        summary = self._summary()

        outcome = hpe.build_action_outcome_from_rollout(summary)

        self.assertIsInstance(outcome, ActionOutcome)
        self.assertEqual(outcome.action, "go")
        self.assertTrue(outcome.success)
        self.assertFalse(outcome.collision)
        self.assertFalse(outcome.no_progress)
        self.assertAlmostEqual(outcome.moved_distance_m, 0.3)
        self.assertAlmostEqual(outcome.odom_delta.dx_m, 0.3)
        self.assertAlmostEqual(outcome.odom_delta.dy_m, 0.0)
        self.assertAlmostEqual(outcome.odom_delta.dyaw_deg, 0.0)
        self.assertTrue(outcome.raw["local_waypoint_reached"])

    def test_collision_rollout_fails_and_marks_no_progress(self):
        outcome = hpe.build_action_outcome_from_rollout(self._summary(collision_count=1))

        self.assertFalse(outcome.success)
        self.assertTrue(outcome.collision)
        self.assertTrue(outcome.no_progress)
        self.assertIn("collision", outcome.message)

    def test_unsupported_action_fails_and_preserves_raw_action_id(self):
        summary = self._summary(
            moved_distance_m=0.0,
            unsupported_action=5,
            steps=[
                PixelNavStep(
                    step_index=0,
                    pixelnav_action=5,
                    sim_action=None,
                    collision_before=False,
                    position_before_xyz=[0.0, 0.0, 0.0],
                    position_after_xyz=[0.0, 0.0, 0.0],
                    moved_distance_m=0.0,
                )
            ],
        )

        outcome = hpe.build_action_outcome_from_rollout(summary)

        self.assertFalse(outcome.success)
        self.assertEqual(outcome.raw["unsupported_action"], 5)
        self.assertIn("unsupported action 5", outcome.message)

    def test_stop_immediately_fails_as_no_progress(self):
        summary = self._summary(
            moved_distance_m=0.0,
            stopped_by_policy=True,
            steps=[
                PixelNavStep(
                    step_index=0,
                    pixelnav_action=0,
                    sim_action=None,
                    collision_before=False,
                    position_before_xyz=[0.0, 0.0, 0.0],
                    position_after_xyz=[0.0, 0.0, 0.0],
                    moved_distance_m=0.0,
                )
            ],
        )

        outcome = hpe.build_action_outcome_from_rollout(summary)

        self.assertFalse(outcome.success)
        self.assertTrue(outcome.no_progress)
        self.assertFalse(outcome.raw["local_waypoint_reached"])

    def test_stop_after_movement_counts_as_local_waypoint_reached(self):
        summary = self._summary(
            moved_distance_m=0.3,
            stopped_by_policy=True,
            steps=[
                PixelNavStep(
                    step_index=0,
                    pixelnav_action=1,
                    sim_action="move_forward",
                    collision_before=False,
                    position_before_xyz=[0.0, 0.0, 0.0],
                    position_after_xyz=[0.3, 0.0, 0.0],
                    moved_distance_m=0.3,
                ),
                PixelNavStep(
                    step_index=1,
                    pixelnav_action=0,
                    sim_action=None,
                    collision_before=False,
                    position_before_xyz=[0.3, 0.0, 0.0],
                    position_after_xyz=[0.3, 0.0, 0.0],
                    moved_distance_m=0.0,
                ),
            ],
        )

        outcome = hpe.build_action_outcome_from_rollout(summary)

        self.assertTrue(outcome.success)
        self.assertTrue(outcome.raw["local_waypoint_reached"])

    def test_empty_step_log_fails_with_invalid_message(self):
        summary = self._summary(moved_distance_m=0.0, steps=[])

        outcome = hpe.build_action_outcome_from_rollout(summary)

        self.assertFalse(outcome.success)
        self.assertIn("empty step log", outcome.message)

    def test_step_c_rollout_result_json_smoke_builds_expected_action_outcome(self):
        step_c_result = (
            Path(__file__).resolve().parents[1]
            / "reports"
            / "habitat_pixelnav_integration"
            / "step_c_rollout_single"
            / "pixelnav_rollout_result.json"
        )

        outcome = hpe.build_action_outcome_from_rollout_result(step_c_result)

        self.assertEqual(outcome.action, "go")
        self.assertTrue(outcome.success)
        self.assertFalse(outcome.collision)
        self.assertAlmostEqual(outcome.moved_distance_m, 3.001126)
        self.assertAlmostEqual(outcome.odom_delta.dx_m, 3.001126)
        self.assertEqual(outcome.raw["selected_view"], "left")
        self.assertEqual(outcome.raw["selected_image_point"], [320, 300])
        self.assertEqual(outcome.raw["rollout_result_json"], str(step_c_result))

    def test_action_outcome_to_json_preserves_backend_contract_fields(self):
        outcome = hpe.build_action_outcome_from_rollout(self._summary())

        payload = hpe.action_outcome_to_json(outcome)

        self.assertEqual(payload["action"], "go")
        self.assertTrue(payload["success"])
        self.assertFalse(payload["collision"])
        self.assertEqual(payload["moved_distance_m"], 0.3)
        self.assertEqual(payload["rotated_deg"], 0.0)
        self.assertEqual(payload["odom_delta"]["dx_m"], 0.3)
        self.assertEqual(payload["odom_delta"]["dy_m"], 0.0)
        self.assertEqual(payload["odom_delta"]["dyaw_deg"], 0.0)
        self.assertTrue(payload["raw"]["local_waypoint_reached"])
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
