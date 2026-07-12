import unittest

from navigation_supervisor import (
    apply_policy_safe_navigation_context,
    assert_policy_input_safe,
    build_localization_contract,
    build_policy_coarse_goal,
    build_policy_goal_context,
    evaluate_execution_progress,
    normalize_supervisor_mode,
)


class NavigationSupervisorTests(unittest.TestCase):
    def test_sim_pose_requires_explicit_opt_in(self):
        with self.assertRaises(ValueError):
            build_localization_contract(
                source="habitat_sim_pose_declared",
                allow_sim_pose=False,
            )
        contract = build_localization_contract(
            source="habitat_sim_pose_declared",
            allow_sim_pose=True,
        )
        self.assertTrue(contract["uses_sim_ground_truth"])
        self.assertTrue(contract["pose_available_to_policy"])

    def test_no_localization_disables_policy_pose(self):
        contract = build_localization_contract(source="none")
        self.assertFalse(contract["pose_available_to_policy"])
        self.assertFalse(contract["uses_sim_ground_truth"])
    def test_spatial_coarse_goal_is_explicit_and_policy_safe(self):
        coarse = build_policy_coarse_goal(
            {
                "type": "relative_waypoint",
                "source": "upstream_global_planner",
                "relative_bearing_deg": 65.0,
                "distance_range_m": [4.0, 7.0],
                "uncertainty": "medium",
            }
        )
        vlm_input = {"task": {}}
        apply_policy_safe_navigation_context(
            vlm_input,
            target_object="chair",
            supervisor_mode="goal_seek",
            spatial_goal=coarse,
        )

        self.assertEqual(vlm_input["task"]["task_mode"], "CoarseGoalNav")
        self.assertTrue(vlm_input["task"]["coarse_goal"]["goal_geometry_available"])
        self.assertEqual(vlm_input["task"]["coarse_goal"]["relative_bearing_deg"], 65.0)
        self.assertTrue(vlm_input["task"]["coarse_goal"]["temporary_detour_allowed"])
        assert_policy_input_safe(vlm_input)

    def test_spatial_coarse_goal_rejects_oracle_source(self):
        with self.assertRaises(ValueError):
            build_policy_coarse_goal(
                {
                    "source": "habitat_ground_truth",
                    "relative_bearing_deg": 20.0,
                }
            )

    def test_objectnav_mode_rejects_spatial_geometry(self):
        with self.assertRaises(ValueError):
            assert_policy_input_safe(
                {
                    "task": {
                        "task_mode": "ObjNav",
                        "coarse_goal": {
                            "goal_geometry_available": True,
                            "relative_bearing_deg": 20.0,
                        },
                    }
                }
            )
    def test_objectnav_goal_context_explicitly_hides_oracle_geometry(self):
        context = build_policy_goal_context(
            task_mode="ObjNav",
            target_object="chair",
            supervisor_mode="goal_seek",
        )

        self.assertEqual(context["task_mode"], "ObjNav")
        self.assertEqual(context["target_object"], "chair")
        self.assertFalse(context["goal_geometry_available"])
        self.assertIsNone(context["goal_bearing_from_current_deg"])
        self.assertIsNone(context["goal_distance_m"])
        self.assertEqual(context["supervisor_mode"], "goal_seek")

    def test_policy_input_guard_rejects_nested_evaluation_metrics(self):
        for forbidden_key in (
            "distance_to_goal",
            "success",
            "spl",
            "soft_spl",
            "top_down_map",
            "shortest_path",
        ):
            with self.subTest(forbidden_key=forbidden_key):
                with self.assertRaises(ValueError):
                    assert_policy_input_safe({"memory": {"runtime": {forbidden_key: 1.0}}})

    def test_policy_input_guard_allows_semantic_success_labels(self):
        assert_policy_input_safe(
            {
                "memory": {
                    "candidate_exits": [
                        {"traversal_status": "success", "success_count": 2}
                    ]
                }
            }
        )

    def test_supervisor_mode_normalizes_v6_stage_names(self):
        self.assertEqual(normalize_supervisor_mode("normal_goal_seek"), "goal_seek")
        self.assertEqual(normalize_supervisor_mode("escape_deadlock"), "escape_deadlock")
        self.assertEqual(normalize_supervisor_mode("verify_revisit"), "goal_seek")
        self.assertEqual(normalize_supervisor_mode("unknown"), "goal_seek")

    def test_stationary_go_is_no_progress_even_if_audit_goal_distance_improves(self):
        progress = evaluate_execution_progress(
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.0, 0.0, 0.0],
            collision_count=0,
            controller_reached_waypoint=False,
            supervisor_mode="goal_seek",
            min_translation_m=0.05,
            audit_goal_distance_delta_m=0.8,
        )

        self.assertFalse(progress["execution_success"])
        self.assertFalse(progress["strategic_progress"])
        self.assertTrue(progress["no_progress"])
        self.assertEqual(progress["translation_m"], 0.0)
        self.assertEqual(progress["audit_goal_distance_delta_m"], 0.8)
        self.assertNotIn("goal_distance", progress["evidence"])

    def test_immediate_controller_stop_without_motion_is_not_waypoint_progress(self):
        progress = evaluate_execution_progress(
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.0, 0.0, 0.0],
            collision_count=0,
            controller_reached_waypoint=True,
            supervisor_mode="goal_seek",
            min_translation_m=0.05,
        )

        self.assertFalse(progress["execution_success"])
        self.assertFalse(progress["strategic_progress"])
        self.assertFalse(progress["evidence"]["controller_reached_with_motion"])

    def test_escape_translation_is_progress_when_audit_goal_distance_gets_worse(self):
        progress = evaluate_execution_progress(
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.4, 0.0, 0.0],
            collision_count=0,
            controller_reached_waypoint=False,
            supervisor_mode="escape_deadlock",
            min_translation_m=0.05,
            audit_goal_distance_delta_m=-0.7,
            strategic_evidence={"sector_departure_selected": True},
        )

        self.assertTrue(progress["execution_success"])
        self.assertTrue(progress["strategic_progress"])
        self.assertFalse(progress["no_progress"])
        self.assertEqual(progress["supervisor_mode"], "escape_deadlock")
        self.assertEqual(progress["audit_goal_distance_delta_m"], -0.7)

    def test_escape_translation_in_same_failed_sector_is_not_strategic_progress(self):
        progress = evaluate_execution_progress(
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.4, 0.0, 0.0],
            collision_count=0,
            controller_reached_waypoint=False,
            supervisor_mode="escape_deadlock",
            min_translation_m=0.05,
            strategic_evidence={"sector_departure_selected": False},
        )

        self.assertTrue(progress["execution_success"])
        self.assertFalse(progress["strategic_progress"])
        self.assertEqual(progress["strategic_rule_id"], "PROGRESS_ESCAPE_001")

    def test_backtrack_requires_verified_backtrack_edge_and_translation(self):
        rejected = evaluate_execution_progress(
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.4, 0.0, 0.0],
            collision_count=0,
            controller_reached_waypoint=False,
            supervisor_mode="backtrack",
            min_translation_m=0.05,
        )
        accepted = evaluate_execution_progress(
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.4, 0.0, 0.0],
            collision_count=0,
            controller_reached_waypoint=False,
            supervisor_mode="backtrack",
            min_translation_m=0.05,
            strategic_evidence={"backtrack_edge_selected": True},
        )

        self.assertFalse(rejected["strategic_progress"])
        self.assertTrue(accepted["strategic_progress"])

    def test_verify_target_requires_temporally_stable_target_evidence(self):
        pending = evaluate_execution_progress(
            start_position_xyz=[],
            final_position_xyz=[],
            collision_count=0,
            controller_reached_waypoint=False,
            supervisor_mode="verify_target",
            min_translation_m=0.05,
        )
        verified = evaluate_execution_progress(
            start_position_xyz=[],
            final_position_xyz=[],
            collision_count=0,
            controller_reached_waypoint=False,
            supervisor_mode="verify_target",
            min_translation_m=0.05,
            strategic_evidence={"target_verification_stable": True},
        )

        self.assertFalse(pending["strategic_progress"])
        self.assertTrue(verified["strategic_progress"])

    def test_goal_seek_uses_translation_not_audit_goal_distance(self):
        progress = evaluate_execution_progress(
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.2, 0.0, 0.0],
            collision_count=0,
            controller_reached_waypoint=False,
            supervisor_mode="goal_seek",
            min_translation_m=0.05,
            audit_goal_distance_delta_m=-0.3,
        )

        self.assertTrue(progress["execution_success"])
        self.assertTrue(progress["strategic_progress"])
        self.assertFalse(progress["no_progress"])

    def test_collision_blocks_execution_progress_despite_translation(self):
        progress = evaluate_execution_progress(
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.4, 0.0, 0.0],
            collision_count=1,
            controller_reached_waypoint=False,
            supervisor_mode="backtrack",
            min_translation_m=0.05,
        )

        self.assertFalse(progress["execution_success"])
        self.assertFalse(progress["strategic_progress"])
        self.assertTrue(progress["no_progress"])

    def test_long_translation_with_one_terminal_collision_counts_as_progress(self):
        progress = evaluate_execution_progress(
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[1.8, 0.0, 0.0],
            collision_count=1,
            controller_reached_waypoint=False,
            supervisor_mode="goal_seek",
            min_translation_m=0.05,
            collision_tolerant_min_translation_m=0.75,
            collision_tolerant_max_collisions=1,
        )

        self.assertTrue(progress["execution_success"])
        self.assertTrue(progress["strategic_progress"])
        self.assertFalse(progress["no_progress"])
        self.assertTrue(progress["collision_tolerant_progress"])

    def test_short_translation_with_collision_remains_no_progress(self):
        progress = evaluate_execution_progress(
            start_position_xyz=[0.0, 0.0, 0.0],
            final_position_xyz=[0.4, 0.0, 0.0],
            collision_count=1,
            controller_reached_waypoint=False,
            supervisor_mode="goal_seek",
            min_translation_m=0.05,
            collision_tolerant_min_translation_m=0.75,
            collision_tolerant_max_collisions=1,
        )

        self.assertFalse(progress["execution_success"])
        self.assertFalse(progress["strategic_progress"])
        self.assertTrue(progress["no_progress"])
        self.assertFalse(progress["collision_tolerant_progress"])


if __name__ == "__main__":
    unittest.main()
