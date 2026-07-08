import math
import sys
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np


QWEN_ROOT = Path(__file__).resolve().parents[1] / "qwen_nav_memory_framework_v5"
if str(QWEN_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN_ROOT))

from nav_memory_qwen.schema import RobotState
from nav_memory_qwen.robot_backend import ActionOutcome
from nav_memory_qwen.schema import Observation, make_go_output
from nav_memory_qwen.agent import NavMemoryAgent, NavAgentConfig
from nav_memory_qwen.vlm_client import BaseVLMClient

from goal_adapter.habitat_pixelnav_backend import HabitatPixelNavBackend, HabitatPixelNavBackendConfig


class FakeRunner:
    def __init__(self):
        self.position = [1.0, 0.1, 2.0]
        self.heading = 0.5
        self.poses = []
        self.actions = []
        self.collided = False

    def set_pose(self, position_xyz, heading):
        self.position = [float(v) for v in position_xyz]
        self.heading = float(heading)
        self.poses.append((list(self.position), float(heading)))

    def position_xyz(self):
        return list(self.position)

    def rgb(self):
        return np.full((4, 6, 3), 127, dtype=np.uint8)

    def previous_step_collided(self):
        return self.collided

    def supported_action_names(self):
        return {"move_forward", "turn_left", "turn_right"}

    def step(self, action_name):
        self.actions.append(action_name)
        if action_name == "move_forward":
            self.position = [self.position[0] + 0.25, self.position[1], self.position[2]]


class FakeGoal:
    def __init__(self):
        self.goal_mask = np.ones((4, 6), dtype=np.uint8)


class FakeExecutor:
    def __init__(self, actions):
        self.actions = list(actions)
        self.reset_args = None

    def reset_from_point(self, goal_rgb, selected_point_uv, radius=5):
        self.reset_args = (goal_rgb.copy(), tuple(selected_point_uv), radius)
        return FakeGoal()

    def step(self, obs_rgb, collide=False):
        action = self.actions.pop(0) if self.actions else 0
        return type(
            "FakeStep",
            (),
            {
                "action": action,
                "overlay_image": np.zeros_like(obs_rgb),
            },
        )()


class FrontGoVLM(BaseVLMClient):
    def decide(self, vlm_input):
        observation = vlm_input["observation"]
        front = next(view for view in observation["views"] if view["view_type"] == "front")
        return make_go_output(
            view_id=front["view_id"],
            view_type="front",
            point_px=(observation["image_width"] // 2, observation["image_height"] // 2),
            width=observation["image_width"],
            height=observation["image_height"],
            decision_reason="G02_VISIBLE_FLOOR_TOWARD_GOAL",
            goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
            short_text="unit test front go",
        )


class HabitatPixelNavBackendTest(unittest.TestCase):
    def test_get_robot_state_uses_runner_position_and_backend_heading(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = HabitatPixelNavBackend(
                HabitatPixelNavBackendConfig(
                    scene_path="scene.glb",
                    output_dir=temp_dir,
                    start_position_xyz=[1.0, 0.1, 2.0],
                    start_heading_rad=0.5,
                ),
                runner=FakeRunner(),
            )

            state = backend.get_robot_state()

        self.assertIsInstance(state, RobotState)
        self.assertEqual(state.position_xyz, (1.0, 0.1, 2.0))
        self.assertEqual(state.map_xy, (1.0, 2.0))
        self.assertAlmostEqual(state.heading_rad, 0.5)

    def test_backend_can_create_runner_from_factory_when_not_injected(self):
        created = []

        def runner_factory(config):
            created.append(config.scene_path)
            return FakeRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            backend = HabitatPixelNavBackend(
                HabitatPixelNavBackendConfig(
                    scene_path="scene.glb",
                    output_dir=temp_dir,
                    start_position_xyz=[1.0, 0.1, 2.0],
                    start_heading_rad=0.5,
                ),
                runner_factory=runner_factory,
            )

            state = backend.get_robot_state()

        self.assertEqual(created, ["scene.glb"])
        self.assertEqual(state.position_xyz, (1.0, 0.1, 2.0))

    def test_capture_views_writes_observation_images_for_requested_offsets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            backend = HabitatPixelNavBackend(
                HabitatPixelNavBackendConfig(
                    scene_path="scene.glb",
                    output_dir=temp_dir,
                    start_position_xyz=[1.0, 0.1, 2.0],
                    start_heading_rad=0.5,
                    image_width=6,
                    image_height=4,
                ),
                runner=runner,
            )

            observation = backend.capture_views([-90.0, 0.0, 90.0], mode="directed_sweep")
            image_paths_exist = all(Path(view.image).exists() for view in observation.views)

        self.assertIsInstance(observation, Observation)
        self.assertEqual(observation.mode, "directed_sweep")
        self.assertEqual(observation.image_width, 6)
        self.assertEqual(observation.image_height, 4)
        self.assertEqual([view.view_type for view in observation.views], ["left", "front", "right"])
        self.assertTrue(image_paths_exist)
        self.assertAlmostEqual(runner.heading, 0.5)

    def test_rotate_updates_backend_heading_and_returns_rotate_outcome(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            backend = HabitatPixelNavBackend(
                HabitatPixelNavBackendConfig(
                    scene_path="scene.glb",
                    output_dir=temp_dir,
                    start_position_xyz=[1.0, 0.1, 2.0],
                    start_heading_rad=0.5,
                ),
                runner=runner,
            )

            outcome = backend.rotate(-90.0)

        self.assertIsInstance(outcome, ActionOutcome)
        self.assertEqual(outcome.action, "rotate")
        self.assertTrue(outcome.success)
        self.assertAlmostEqual(outcome.rotated_deg, -90.0)
        self.assertAlmostEqual(outcome.odom_delta.dyaw_deg, -90.0)
        self.assertAlmostEqual(runner.heading, 0.5 - math.pi / 2.0)

    def test_execute_waypoint_runs_pixelnav_rollout_and_returns_action_outcome(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            executor = FakeExecutor([1, 1])
            backend = HabitatPixelNavBackend(
                HabitatPixelNavBackendConfig(
                    scene_path="scene.glb",
                    output_dir=temp_dir,
                    start_position_xyz=[1.0, 0.1, 2.0],
                    start_heading_rad=0.5,
                    image_width=6,
                    image_height=4,
                    max_steps=2,
                    mask_radius=1,
                ),
                runner=runner,
                executor_factory=lambda: executor,
            )

            outcome = backend.execute_waypoint(
                view_type="front",
                view_id=0,
                point_px=(3, 2),
                ttl_ms=1000,
            )
            rollout_result_path = Path(outcome.raw["rollout_result_json"])
            rollout_result_exists = rollout_result_path.exists()

        self.assertIsInstance(outcome, ActionOutcome)
        self.assertEqual(outcome.action, "go")
        self.assertTrue(outcome.success)
        self.assertFalse(outcome.collision)
        self.assertAlmostEqual(outcome.moved_distance_m, 0.5)
        self.assertEqual(outcome.raw["selected_view"], "front")
        self.assertEqual(outcome.raw["selected_image_point"], [3, 2])
        self.assertEqual(runner.actions, ["move_forward", "move_forward"])
        self.assertEqual(executor.reset_args[1], (3, 2))
        self.assertEqual(executor.reset_args[2], 1)
        self.assertTrue(rollout_result_exists)

    def test_execute_waypoint_artifact_keeps_base_and_selected_rollout_heading(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            executor = FakeExecutor([1])
            backend = HabitatPixelNavBackend(
                HabitatPixelNavBackendConfig(
                    scene_path="scene.glb",
                    output_dir=temp_dir,
                    start_position_xyz=[1.0, 0.1, 2.0],
                    start_heading_rad=0.5,
                    image_width=6,
                    image_height=4,
                    max_steps=1,
                ),
                runner=runner,
                executor_factory=lambda: executor,
            )

            outcome = backend.execute_waypoint(
                view_type="left",
                view_id=1,
                point_px=(3, 2),
                ttl_ms=1000,
            )
            payload = json.loads(Path(outcome.raw["rollout_result_json"]).read_text(encoding="utf-8"))

        self.assertAlmostEqual(payload["request"]["base_heading"], 0.5)
        self.assertAlmostEqual(payload["request"]["rollout_heading"], 0.5 - math.pi / 2.0)

    def test_nav_memory_agent_can_execute_front_go_through_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            executor = FakeExecutor([1, 1])
            backend = HabitatPixelNavBackend(
                HabitatPixelNavBackendConfig(
                    scene_path="scene.glb",
                    output_dir=temp_dir,
                    start_position_xyz=[0.0, 0.0, 0.0],
                    image_width=6,
                    image_height=4,
                    max_steps=2,
                ),
                runner=runner,
                executor_factory=lambda: executor,
            )
            agent = NavMemoryAgent(
                robot=backend,
                vlm_client=FrontGoVLM(),
                config=NavAgentConfig(max_steps=1, force_front_view_waypoint=True),
            )

            result = agent.step(goal_map_xy=(5.0, 0.0), step_index=0)

        self.assertEqual(result.action, "go")
        self.assertIsNotNone(result.outcome)
        self.assertTrue(result.outcome.success)
        self.assertAlmostEqual(result.outcome.moved_distance_m, 0.5)
        self.assertEqual(runner.actions, ["move_forward", "move_forward"])


if __name__ == "__main__":
    unittest.main()
