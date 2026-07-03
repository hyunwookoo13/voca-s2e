FIRST_STAGE_REFINEMENT_CASES = [
    {
        "name": "coarse_gps_target_uses_nearby_navigable_waypoint",
        "case_group": "coarse_gps",
        "input": {
            "target_type": "gps",
            "high_level_target": {
                "label": "bathroom",
                "coarse_goal_xy": [4.0, 0.0],
            },
            "current_rgb": "fixtures/room_start.png",
            "current_pose": {"x": 0.0, "y": 0.0},
            "heading": 0.0,
            "current_goal_xy": [4.0, 0.0],
            "progress_state": "normal",
            "memory_summary": {
                "candidate_waypoints": [
                    {
                        "kind": "blocked direct bathroom center",
                        "goal_xy": [4.0, 0.0],
                        "image_point": [520, 220],
                        "navigable": False,
                        "semantic_score": 0.9,
                    },
                    {
                        "kind": "doorway toward corridor",
                        "goal_xy": [1.4, 0.6],
                        "image_point": [350, 240],
                        "navigable": True,
                        "semantic_score": 0.8,
                    },
                ]
            },
            "system_prompt": "Refine coarse GPS targets into S2E local goals.",
        },
        "expected_goal_xy": [1.4, 0.6],
        "expected_image_point": [350, 240],
        "expected_action_type": "NAVIGATE",
        "expected_confidence": "medium",
        "reasoning_terms": ["coarse", "navigable"],
    },
    {
        "name": "coarse_gps_target_uses_corridor_turn_before_far_goal",
        "case_group": "coarse_gps",
        "input": {
            "target_type": "gps",
            "high_level_target": {
                "label": "kitchen",
                "coarse_goal_xy": [5.5, -1.0],
            },
            "current_rgb": "fixtures/corridor_wall_ahead.png",
            "current_pose": {"x": 0.0, "y": 0.0},
            "heading": 0.2,
            "current_goal_xy": [5.5, -1.0],
            "progress_state": "normal",
            "memory_summary": {
                "candidate_waypoints": [
                    {
                        "kind": "straight line through wall",
                        "goal_xy": [3.0, -0.4],
                        "image_point": [430, 210],
                        "navigable": False,
                        "semantic_score": 0.95,
                    },
                    {
                        "kind": "right corridor turn",
                        "goal_xy": [1.2, -0.9],
                        "image_point": [420, 245],
                        "navigable": True,
                        "semantic_score": 0.82,
                    },
                ]
            },
            "system_prompt": "Choose the next navigable waypoint toward the coarse GPS target.",
        },
        "expected_goal_xy": [1.2, -0.9],
        "expected_image_point": [420, 245],
        "expected_action_type": "NAVIGATE",
        "expected_confidence": "medium",
        "reasoning_terms": ["coarse", "navigable"],
    },
    {
        "name": "coarse_gps_target_exits_room_before_open_area",
        "case_group": "coarse_gps",
        "input": {
            "target_type": "gps",
            "high_level_target": {
                "label": "living room",
                "coarse_goal_xy": [3.8, 1.7],
            },
            "current_rgb": "fixtures/bedroom_doorway.png",
            "current_pose": {"x": -0.5, "y": 0.2},
            "heading": -0.1,
            "current_goal_xy": [3.8, 1.7],
            "progress_state": "normal",
            "memory_summary": {
                "candidate_waypoints": [
                    {
                        "kind": "bedroom doorway exit",
                        "goal_xy": [0.9, 0.8],
                        "image_point": [300, 250],
                        "navigable": True,
                        "semantic_score": 0.88,
                    },
                    {
                        "kind": "far living room center",
                        "goal_xy": [3.8, 1.7],
                        "image_point": [500, 210],
                        "navigable": True,
                        "semantic_score": 0.45,
                    },
                ]
            },
            "system_prompt": "Prefer a reachable intermediate local goal over the coarse target center.",
        },
        "expected_goal_xy": [0.9, 0.8],
        "expected_image_point": [300, 250],
        "expected_action_type": "NAVIGATE",
        "expected_confidence": "medium",
        "reasoning_terms": ["coarse", "navigable"],
    },
    {
        "name": "coarse_object_point_uses_approach_waypoint",
        "case_group": "coarse_object_point",
        "input": {
            "target_type": "object_point",
            "high_level_target": {
                "label": "toilet",
                "object_image_point": [430, 180],
            },
            "current_rgb": "fixtures/bathroom_view.png",
            "current_goal_xy": [2.0, -0.5],
            "progress_state": "normal",
            "memory_summary": {
                "candidate_waypoints": [
                    {
                        "kind": "in front of toilet",
                        "goal_xy": [1.0, -0.3],
                        "image_point": [410, 260],
                        "navigable": True,
                        "semantic_score": 0.95,
                    }
                ]
            },
            "system_prompt": "Do not navigate to the object center itself.",
        },
        "expected_goal_xy": [1.0, -0.3],
        "expected_image_point": [410, 260],
        "expected_action_type": "NAVIGATE",
        "expected_confidence": "medium",
        "reasoning_terms": ["object", "approach"],
    },
    {
        "name": "coarse_object_point_sofa_uses_front_access_space",
        "case_group": "coarse_object_point",
        "input": {
            "target_type": "object_point",
            "high_level_target": {
                "label": "sofa",
                "object_image_point": [210, 170],
            },
            "current_rgb": "fixtures/living_room_sofa.png",
            "current_goal_xy": [1.6, 0.4],
            "progress_state": "normal",
            "memory_summary": {
                "candidate_waypoints": [
                    {
                        "kind": "sofa object center",
                        "goal_xy": [1.6, 0.4],
                        "image_point": [210, 170],
                        "navigable": False,
                        "semantic_score": 0.96,
                    },
                    {
                        "kind": "front access space of sofa",
                        "goal_xy": [0.9, 0.2],
                        "image_point": [240, 255],
                        "navigable": True,
                        "semantic_score": 0.86,
                    },
                ]
            },
            "system_prompt": "Select an approach point near the object.",
        },
        "expected_goal_xy": [0.9, 0.2],
        "expected_image_point": [240, 255],
        "expected_action_type": "NAVIGATE",
        "expected_confidence": "medium",
        "reasoning_terms": ["object", "approach"],
    },
    {
        "name": "coarse_object_point_sink_uses_counter_front_waypoint",
        "case_group": "coarse_object_point",
        "input": {
            "target_type": "object_point",
            "high_level_target": {
                "label": "sink",
                "object_image_point": [380, 160],
            },
            "current_rgb": "fixtures/kitchen_sink.png",
            "current_goal_xy": [2.4, -0.2],
            "progress_state": "normal",
            "memory_summary": {
                "candidate_waypoints": [
                    {
                        "kind": "inside sink basin",
                        "goal_xy": [2.4, -0.2],
                        "image_point": [380, 160],
                        "navigable": False,
                        "semantic_score": 0.91,
                    },
                    {
                        "kind": "counter front approach",
                        "goal_xy": [1.1, -0.6],
                        "image_point": [365, 265],
                        "navigable": True,
                        "semantic_score": 0.84,
                    },
                ]
            },
            "system_prompt": "Use a navigable point in front of the target object.",
        },
        "expected_goal_xy": [1.1, -0.6],
        "expected_image_point": [365, 265],
        "expected_action_type": "NAVIGATE",
        "expected_confidence": "medium",
        "reasoning_terms": ["object", "approach"],
    },
    {
        "name": "missing_point_tracking_loss_uses_lookaround_candidate",
        "case_group": "tracking_loss",
        "input": {
            "target_type": "missing_point",
            "high_level_target": "bathroom",
            "current_rgb": "fixtures/no_target_visible.png",
            "optional_lookaround_images": [
                "fixtures/look_front.png",
                "fixtures/look_left.png",
            ],
            "current_goal_xy": None,
            "progress_state": "tracking_loss",
            "memory_summary": {
                "candidate_waypoints": [
                    {
                        "kind": "left hallway doorway",
                        "goal_xy": [0.8, 1.1],
                        "image_point": [280, 230],
                        "navigable": True,
                        "semantic_score": 0.7,
                    }
                ],
                "failed_goal_xy": [[1.0, 0.0]],
            },
            "system_prompt": "Recover a waypoint when the target point is missing.",
        },
        "expected_goal_xy": [0.8, 1.1],
        "expected_image_point": [280, 230],
        "expected_action_type": "LOOK_AROUND",
        "expected_confidence": "medium",
        "reasoning_terms": ["tracking", "look-around"],
    },
    {
        "name": "missing_point_uses_visible_corridor_after_detector_loss",
        "case_group": "tracking_loss",
        "input": {
            "target_type": "missing_point",
            "high_level_target": "toilet",
            "current_rgb": "fixtures/detector_lost_toilet.png",
            "optional_lookaround_images": [
                "fixtures/look_right_bathroom_door.png",
                "fixtures/look_back_bedroom.png",
            ],
            "current_goal_xy": None,
            "progress_state": "tracking_loss",
            "memory_summary": {
                "candidate_waypoints": [
                    {
                        "kind": "right bathroom corridor",
                        "goal_xy": [0.6, -1.0],
                        "image_point": [390, 235],
                        "navigable": True,
                        "semantic_score": 0.78,
                    },
                    {
                        "kind": "previous bedroom corner",
                        "goal_xy": [-0.4, 0.2],
                        "image_point": [130, 240],
                        "navigable": True,
                        "semantic_score": 0.25,
                    },
                ],
                "failed_goal_xy": [[1.0, 0.0]],
            },
            "system_prompt": "Use look-around views when the object point is gone.",
        },
        "expected_goal_xy": [0.6, -1.0],
        "expected_image_point": [390, 235],
        "expected_action_type": "LOOK_AROUND",
        "expected_confidence": "medium",
        "reasoning_terms": ["tracking", "look-around"],
    },
    {
        "name": "tracking_loss_language_goal_uses_unexplored_hallway",
        "case_group": "tracking_loss",
        "input": {
            "target_type": "language",
            "high_level_target": "find the bathroom past the hallway",
            "current_rgb": "fixtures/ambiguous_hallway.png",
            "optional_lookaround_images": [
                "fixtures/look_left_closed_room.png",
                "fixtures/look_front_open_hallway.png",
            ],
            "current_goal_xy": [1.0, 0.0],
            "progress_state": "tracking_loss",
            "memory_summary": {
                "candidate_waypoints": [
                    {
                        "kind": "unexplored open hallway",
                        "goal_xy": [1.3, 0.3],
                        "image_point": [330, 230],
                        "navigable": True,
                        "semantic_score": 0.74,
                    },
                    {
                        "kind": "closed side room",
                        "goal_xy": [0.2, 1.1],
                        "image_point": [180, 220],
                        "navigable": True,
                        "semantic_score": 0.3,
                    },
                ],
                "failed_goal_xy": [[0.2, 1.1]],
            },
            "system_prompt": "When tracking is lost, choose a promising unexplored waypoint.",
        },
        "expected_goal_xy": [1.3, 0.3],
        "expected_image_point": [330, 230],
        "expected_action_type": "LOOK_AROUND",
        "expected_confidence": "medium",
        "reasoning_terms": ["tracking", "look-around"],
    },
    {
        "name": "blocked_deadlock_reselects_unfailed_waypoint",
        "case_group": "deadlock",
        "input": {
            "target_type": "language",
            "high_level_target": "go to the bathroom",
            "current_rgb": "fixtures/blocked_corridor.png",
            "current_goal_xy": [1.0, 0.0],
            "progress_state": "blocked",
            "memory_summary": {
                "failed_goal_xy": [[1.0, 0.0]],
                "candidate_waypoints": [
                    {
                        "kind": "previously blocked hallway",
                        "goal_xy": [1.0, 0.0],
                        "image_point": [320, 220],
                        "navigable": True,
                        "semantic_score": 0.9,
                    },
                    {
                        "kind": "alternate doorway",
                        "goal_xy": [0.2, 1.2],
                        "image_point": [180, 230],
                        "navigable": True,
                        "semantic_score": 0.75,
                    },
                ],
            },
            "system_prompt": "Avoid repeated failed directions.",
        },
        "expected_goal_xy": [0.2, 1.2],
        "expected_image_point": [180, 230],
        "expected_action_type": "RESELECT_GOAL",
        "expected_confidence": "medium",
        "reasoning_terms": ["blocked", "failed"],
    },
    {
        "name": "low_progress_deadlock_reselects_side_doorway",
        "case_group": "deadlock",
        "input": {
            "target_type": "gps",
            "high_level_target": {
                "label": "bathroom",
                "coarse_goal_xy": [2.0, 0.0],
            },
            "current_rgb": "fixtures/low_progress_near_wall.png",
            "current_goal_xy": [1.2, 0.0],
            "progress_state": "low_progress",
            "memory_summary": {
                "failed_goal_xy": [[1.2, 0.0]],
                "candidate_waypoints": [
                    {
                        "kind": "same low progress wall direction",
                        "goal_xy": [1.2, 0.0],
                        "image_point": [315, 220],
                        "navigable": True,
                        "semantic_score": 0.9,
                    },
                    {
                        "kind": "side doorway around obstacle",
                        "goal_xy": [0.4, -1.1],
                        "image_point": [430, 235],
                        "navigable": True,
                        "semantic_score": 0.71,
                    },
                ],
            },
            "system_prompt": "If progress is low, reselect a different waypoint.",
        },
        "expected_goal_xy": [0.4, -1.1],
        "expected_image_point": [430, 235],
        "expected_action_type": "RESELECT_GOAL",
        "expected_confidence": "medium",
        "reasoning_terms": ["low_progress", "failed"],
    },
    {
        "name": "repeated_view_deadlock_reselects_unvisited_corridor",
        "case_group": "deadlock",
        "input": {
            "target_type": "language",
            "high_level_target": "go to the kitchen",
            "current_rgb": "fixtures/repeated_view_hallway.png",
            "current_goal_xy": [0.8, 0.0],
            "progress_state": "repeated_view",
            "memory_summary": {
                "visited": [[0.0, 0.0], [0.5, 0.0]],
                "failed_goal_xy": [[0.8, 0.0]],
                "candidate_waypoints": [
                    {
                        "kind": "repeated view forward loop",
                        "goal_xy": [0.8, 0.0],
                        "image_point": [320, 220],
                        "navigable": True,
                        "semantic_score": 0.85,
                    },
                    {
                        "kind": "unvisited corridor branch",
                        "goal_xy": [-0.3, 1.0],
                        "image_point": [160, 235],
                        "navigable": True,
                        "semantic_score": 0.69,
                    },
                ],
            },
            "system_prompt": "When the same view repeats, avoid the failed loop.",
        },
        "expected_goal_xy": [-0.3, 1.0],
        "expected_image_point": [160, 235],
        "expected_action_type": "RESELECT_GOAL",
        "expected_confidence": "medium",
        "reasoning_terms": ["repeated_view", "failed"],
    },
]
