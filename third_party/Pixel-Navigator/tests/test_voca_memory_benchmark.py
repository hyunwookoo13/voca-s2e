import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


PIXEL_ROOT = Path(__file__).resolve().parents[1]
if str(PIXEL_ROOT) not in sys.path:
    sys.path.insert(0, str(PIXEL_ROOT))


class FakeEpisode:
    object_category = "chair"
    episode_id = "fake_objnav_0001"
    scene_id = "fake_scene.glb"
    goals = []


class FakeGoal:
    position = [1.0, 0.0, 0.0]


class FakeAgentState:
    def __init__(self, position, heading_rad=0.0):
        self.position = np.asarray(position, dtype=np.float32)
        self.heading_rad = float(heading_rad)
        self.rotation = None


class FakeSim:
    previous_step_collided = False

    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.heading_rad = 0.0

    def get_agent_state(self):
        return FakeAgentState(self.position.copy(), self.heading_rad)


class FakeHabitatEnv:
    def __init__(self):
        self.sim = FakeSim()
        self.current_episode = FakeEpisode()
        self.episode_over = False
        self.step_count = 0
        self.metrics = {
            "success": 0.0,
            "spl": 0.0,
            "distance_to_goal": 3.0,
            "top_down_map": {
                "map": np.zeros((2, 2), dtype=np.uint8),
                "agent_map_coord": np.asarray([1, 1], dtype=np.int64),
            },
        }

    def reset(self):
        self.sim.position[:] = 0.0
        self.sim.heading_rad = 0.0
        self.episode_over = False
        self.step_count = 0
        self.step_actions = []
        self.metrics.update(success=0.0, spl=0.0, distance_to_goal=3.0)
        return {"rgb": np.full((48, 64, 3), 127, dtype=np.uint8)}

    def step(self, action):
        self.step_count += 1
        action = int(action)
        self.step_actions.append(action)
        if action == 0:
            self.episode_over = True
            self.metrics.update(success=1.0, spl=0.8, distance_to_goal=0.2)
        elif action == 1:
            self.sim.position[0] += 0.25
            self.metrics["distance_to_goal"] = max(0.0, self.metrics["distance_to_goal"] - 0.25)
        elif action == 2:
            self.sim.heading_rad -= np.deg2rad(30.0)
        elif action == 3:
            self.sim.heading_rad += np.deg2rad(30.0)
        return {"rgb": np.full((48, 64, 3), 127, dtype=np.uint8)}

    def get_metrics(self):
        return dict(self.metrics)


class FakePointNavEpisode(FakeEpisode):
    episode_id = "fake_pointnav_0001"
    goals = [FakeGoal()]


class FakePointNavEnv(FakeHabitatEnv):
    def __init__(self):
        super().__init__()
        self.current_episode = FakePointNavEpisode()


class FakePixelNavPolicy:
    def __init__(self):
        self.reset_calls = 0
        self.calls = 0

    def reset(self, goal_image, goal_mask):
        self.reset_calls += 1

    def step(self, image, collide=False):
        self.calls += 1
        if self.calls == 1:
            return 1, image
        return 0, image


class UnsupportedActionPixelPolicy:
    def reset(self, goal_image, goal_mask):
        pass

    def step(self, image, collide=False):
        return 5, image


class ObservationAwarePixelPolicy:
    def __init__(self):
        self.seen_observation = None

    def reset(self, goal_image, goal_mask):
        pass

    def step(self, image, collide=False):
        raise AssertionError("backend should call step_from_observation")

    def step_from_observation(self, obs, image, collide=False):
        self.seen_observation = obs
        return 0, image


class GoalAwarePixelPolicy(FakePixelNavPolicy):
    def __init__(self):
        super().__init__()
        self.bound_env = None
        self.goal_position = None

    def bind_env(self, env):
        self.bound_env = env

    def set_goal_position(self, goal_position):
        self.goal_position = list(goal_position)


class CenterGoThenStopVLM:
    def __init__(self):
        self.calls = 0

    def decide(self, vlm_input):
        self.calls += 1
        if self.calls == 1:
            return {
                "schema_version": "nav_vlm_waypoint_v1",
                "action": "go",
                "selected_view_id": 0,
                "selected_view_type": "front",
                "selected_image_point": [32, 36],
                "fine_goal": {"valid": True, "point_px": [32, 36]},
                "reasoning": {"short_text": "fake visible floor"},
                "confidence": "medium",
            }
        return {
            "schema_version": "nav_vlm_waypoint_v1",
            "action": "stop",
            "reasoning": {"short_text": "fake target reached"},
            "confidence": "medium",
        }


class RaisingVLM:
    def decide(self, vlm_input):
        raise ValueError("bad qwen json")


class StaticFallbackVLM:
    def decide(self, vlm_input):
        return {
            "schema_version": "nav_vlm_waypoint_v1",
            "action": "stop",
            "reasoning": {"short_text": "fallback stop"},
            "confidence": "low",
        }


class DummyVideoWriter:
    def __init__(self, path, fps=4):
        self.path = Path(path)
        self.fps = fps
        self.frame_count = 0

    def append_data(self, frame):
        self.frame_count += 1

    def close(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(f"dummy video frames={self.frame_count} fps={self.fps}\n".encode("utf-8"))


class VocaMemoryBenchmarkTests(unittest.TestCase):
    def test_ensure_voca_imports_accepts_v6_framework_selector(self):
        from voca_memory_benchmark import ensure_voca_imports

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            framework = root / "qwen_nav_memory_framework_v6"
            framework.mkdir()

            before = list(sys.path)
            try:
                ensure_voca_imports(root, memory_framework="v6")
                self.assertEqual(sys.path[0], str(framework))
                self.assertIn(str(root), sys.path)
            finally:
                sys.path[:] = before

    def test_ensure_voca_imports_honors_memory_framework_env(self):
        from voca_memory_benchmark import ensure_voca_imports

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            framework = root / "qwen_nav_memory_framework_v6"
            framework.mkdir()

            before = list(sys.path)
            try:
                with patch.dict(os.environ, {"VOCA_MEMORY_FRAMEWORK": "v6"}):
                    ensure_voca_imports(root)
                self.assertEqual(sys.path[0], str(framework))
            finally:
                sys.path[:] = before

    def test_habitat_env_memory_backend_executes_pixelnav_rollout(self):
        from voca_memory_benchmark import HabitatEnvMemoryBackend

        with self.subTest("backend rollout"):
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                env = FakeHabitatEnv()
                obs = env.reset()
                backend = HabitatEnvMemoryBackend(
                    env=env,
                    initial_obs=obs,
                    output_dir=tmp,
                    pixelnav_policy=FakePixelNavPolicy(),
                    max_pixelnav_steps=4,
                )

                outcome = backend.execute_waypoint(view_type="front", view_id=0, point_px=(32, 36), ttl_ms=1000)

                self.assertEqual(outcome.action, "go")
                self.assertTrue(outcome.success)
                self.assertFalse(outcome.collision)
                self.assertEqual(outcome.moved_distance_m, 0.25)
                self.assertEqual(float(env.sim.get_agent_state().position[0]), 0.25)
                self.assertTrue(Path(outcome.raw["rollout_result_json"]).exists())

    def test_backend_rotate_uses_habitat_turn_direction_convention(self):
        from voca_memory_benchmark import HabitatEnvMemoryBackend
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            env = FakeHabitatEnv()
            obs = env.reset()
            backend = HabitatEnvMemoryBackend(
                env=env,
                initial_obs=obs,
                output_dir=tmp,
                pixelnav_policy=FakePixelNavPolicy(),
                max_pixelnav_steps=4,
            )

            backend.rotate(30)
            backend.rotate(-30)

            self.assertEqual(env.step_actions, [2, 3])

    def test_backend_reports_unsupported_pixelnav_action_without_crashing(self):
        from voca_memory_benchmark import HabitatEnvMemoryBackend
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            env = FakeHabitatEnv()
            obs = env.reset()
            backend = HabitatEnvMemoryBackend(
                env=env,
                initial_obs=obs,
                output_dir=tmp,
                pixelnav_policy=UnsupportedActionPixelPolicy(),
                max_pixelnav_steps=4,
            )

            outcome = backend.execute_waypoint(view_type="front", view_id=0, point_px=(32, 36), ttl_ms=1000)

            self.assertFalse(outcome.success)
            self.assertEqual(outcome.raw["rollout"]["unsupported_action"], 5)

    def test_backend_caps_rollout_steps_when_pointnav_goal_is_near(self):
        from voca_memory_benchmark import ForwardOnlyPixelPolicy, HabitatEnvMemoryBackend
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            env = FakeHabitatEnv()
            obs = env.reset()
            backend = HabitatEnvMemoryBackend(
                env=env,
                initial_obs=obs,
                output_dir=tmp,
                pixelnav_policy=ForwardOnlyPixelPolicy(),
                max_pixelnav_steps=12,
            )
            backend.pointnav_goal_distance_m = 0.60

            outcome = backend.execute_waypoint(view_type="front", view_id=0, point_px=(32, 36), ttl_ms=1000)

            self.assertEqual(len(outcome.raw["rollout"]["steps"]), 1)
            self.assertEqual(outcome.moved_distance_m, 0.25)
            self.assertEqual(env.step_count, 1)

    def test_backend_uses_observation_aware_pixel_policy_hook(self):
        from voca_memory_benchmark import HabitatEnvMemoryBackend
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            env = FakeHabitatEnv()
            obs = env.reset()
            policy = ObservationAwarePixelPolicy()
            backend = HabitatEnvMemoryBackend(
                env=env,
                initial_obs=obs,
                output_dir=tmp,
                pixelnav_policy=policy,
                max_pixelnav_steps=4,
            )

            backend.execute_waypoint(view_type="front", view_id=0, point_px=(32, 36), ttl_ms=1000)

            self.assertIs(policy.seen_observation, backend._last_obs)

    def test_backend_binds_env_to_goal_aware_pixel_policy(self):
        from voca_memory_benchmark import HabitatEnvMemoryBackend
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            env = FakeHabitatEnv()
            obs = env.reset()
            policy = GoalAwarePixelPolicy()
            HabitatEnvMemoryBackend(
                env=env,
                initial_obs=obs,
                output_dir=tmp,
                pixelnav_policy=policy,
                max_pixelnav_steps=4,
            )

            self.assertIs(policy.bound_env, env)

    def test_pointnav_benchmark_sets_distance_aware_rollout_cap(self):
        from voca_memory_benchmark import ForwardOnlyPixelPolicy, run_pointnav_memory_benchmark
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            env = FakePointNavEnv()
            result = run_pointnav_memory_benchmark(
                env=env,
                output_dir=tmp,
                eval_episodes=1,
                max_agent_steps=1,
                max_pixelnav_steps=12,
                max_env_steps=250,
                pixelnav_policy_factory=ForwardOnlyPixelPolicy,
                vlm_client_factory=CenterGoThenStopVLM,
            )

            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
            self.assertEqual(env.step_count, 3)
            self.assertEqual(summary["episodes"][0]["env_steps"], 3)

    def test_pointnav_benchmark_passes_goal_position_to_goal_aware_policy(self):
        from voca_memory_benchmark import run_pointnav_memory_benchmark
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            policy = GoalAwarePixelPolicy()
            run_pointnav_memory_benchmark(
                env=FakePointNavEnv(),
                output_dir=tmp,
                eval_episodes=1,
                max_agent_steps=1,
                max_pixelnav_steps=4,
                max_env_steps=8,
                pixelnav_policy_factory=lambda: policy,
                vlm_client_factory=CenterGoThenStopVLM,
            )

            self.assertEqual(policy.goal_position, [1.0, 0.0, 0.0])

    def test_run_objnav_memory_benchmark_writes_official_metrics_and_memory_graph(self):
        from voca_memory_benchmark import run_objnav_memory_benchmark
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            result = run_objnav_memory_benchmark(
                env=FakeHabitatEnv(),
                output_dir=tmp,
                eval_episodes=1,
                max_agent_steps=2,
                max_pixelnav_steps=4,
                pixelnav_policy_factory=FakePixelNavPolicy,
                vlm_client_factory=CenterGoThenStopVLM,
            )

            summary_path = Path(result["summary_json"])
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["benchmark_type"], "official_habitat_objnav_with_voca_memory")
            self.assertEqual(summary["aggregate"]["episodes"], 1)
            self.assertEqual(summary["aggregate"]["success"], 1.0)
            self.assertEqual(summary["aggregate"]["spl"], 0.8)
            self.assertEqual(summary["aggregate"]["mean_distance_to_goal_delta"], 2.8)
            self.assertEqual(summary["episodes"][0]["initial_distance_to_goal"], 3.0)
            self.assertEqual(summary["episodes"][0]["distance_to_goal_delta"], 2.8)
            self.assertTrue(summary["episodes"][0]["habitat_metrics"]["top_down_map"]["present"])
            self.assertEqual(summary["episodes"][0]["habitat_metrics"]["top_down_map"]["map_shape"], [2, 2])
            self.assertEqual(summary["episodes"][0]["habitat_metrics"]["top_down_map"]["agent_map_coord"], [1, 1])
            self.assertGreaterEqual(summary["aggregate"]["mean_memory_nodes"], 1)
            self.assertTrue(Path(summary["episodes"][0]["memory_graph_json"]).exists())

    def test_run_pointnav_memory_benchmark_uses_episode_goal_position(self):
        from voca_memory_benchmark import run_pointnav_memory_benchmark
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            result = run_pointnav_memory_benchmark(
                env=FakePointNavEnv(),
                output_dir=tmp,
                eval_episodes=1,
                max_agent_steps=2,
                max_pixelnav_steps=4,
                pixelnav_policy_factory=FakePixelNavPolicy,
                vlm_client_factory=CenterGoThenStopVLM,
            )

            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
            self.assertEqual(summary["benchmark_type"], "official_habitat_pointnav_with_voca_memory")
            self.assertEqual(summary["episodes"][0]["memory_framework"], "v5")
            self.assertEqual(summary["episodes"][0]["goal_position_xyz"], [1.0, 0.0, 0.0])
            self.assertEqual(summary["aggregate"]["success"], 1.0)
            self.assertEqual(summary["aggregate"]["mean_distance_to_goal_delta"], 2.8)
            self.assertEqual(summary["episodes"][0]["initial_distance_to_goal"], 3.0)
            self.assertEqual(summary["episodes"][0]["distance_to_goal_delta"], 2.8)

    def test_pointnav_sensor_goal_source_projects_relative_goal_to_map_frame(self):
        from nav_memory_qwen.schema import RobotState
        from voca_memory_benchmark import _pointnav_goal_map_xy_from_observation

        state = RobotState(map_xy=(1.0, 2.0), heading_rad=np.deg2rad(30.0), position_xyz=(1.0, 0.0, 2.0))
        obs = {"pointgoal_with_gps_compass": np.asarray([2.0, np.deg2rad(60.0)], dtype=np.float32)}

        goal_xy = _pointnav_goal_map_xy_from_observation(obs, state, fallback_goal_map_xy=(0.0, 0.0))

        self.assertAlmostEqual(goal_xy[0], 1.0, places=5)
        self.assertAlmostEqual(goal_xy[1], 4.0, places=5)

    def test_pointnav_bearing_heuristic_attaches_safe_front_candidate_ref(self):
        from voca_memory_benchmark import PointNavBearingHeuristicVLMClient

        client = PointNavBearingHeuristicVLMClient(rotate_threshold_deg=90.0)
        output = client.decide({
            "task": {
                "task_mode": "PointNav",
                "coarse_goal": {"relative_bearing_deg": 0.0, "distance_m": 3.0},
            },
            "observation": {
                "image_width": 64,
                "image_height": 48,
                "views": [{"view_id": 0, "view_type": "front"}],
            },
            "memory": {
                "candidate_refs": {
                    "exits": [
                        {"candidate_ref": "exit_bad", "view_type_hint": "front", "avoid": True, "score": 9.0},
                        {"candidate_ref": "exit_ok", "view_type_hint": "front", "avoid": False, "score": 1.0},
                    ]
                }
            },
        })

        self.assertEqual(output["action"], "go")
        self.assertEqual(output["selected_candidate_ref"], "exit_ok")

    def test_pointnav_benchmark_respects_low_level_env_step_budget(self):
        from voca_memory_benchmark import run_pointnav_memory_benchmark
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            env = FakePointNavEnv()
            result = run_pointnav_memory_benchmark(
                env=env,
                output_dir=tmp,
                eval_episodes=1,
                max_agent_steps=10,
                max_pixelnav_steps=4,
                max_env_steps=1,
                pixelnav_policy_factory=FakePixelNavPolicy,
                vlm_client_factory=CenterGoThenStopVLM,
            )

            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
            episode = summary["episodes"][0]
            self.assertEqual(env.step_count, 1)
            self.assertEqual(episode["env_steps"], 1)
            self.assertEqual(episode["max_env_steps"], 1)
            self.assertTrue(episode["env_step_budget_exhausted"])

    def test_pointnav_benchmark_writes_episode_video_artifacts_when_enabled(self):
        from voca_memory_benchmark import run_pointnav_memory_benchmark
        import tempfile

        def fake_memory_video(graph_path, episode_dir, *, fps, **kwargs):
            path = Path(episode_dir) / "memory_graph_reconstruction.mp4"
            path.write_bytes(b"dummy memory graph video\n")
            return {
                "video_path": str(path),
                "summary_json": str(Path(episode_dir) / "memory_graph_video_summary.json"),
                "fps": fps,
                "frame_count": 1,
            }

        with tempfile.TemporaryDirectory() as tmp:
            with patch("imageio.get_writer", side_effect=lambda path, fps=4: DummyVideoWriter(path, fps=fps)):
                with patch(
                    "voca_memory_benchmark._render_memory_graph_video_artifact",
                    side_effect=fake_memory_video,
                    create=True,
                ):
                    result = run_pointnav_memory_benchmark(
                        env=FakePointNavEnv(),
                        output_dir=tmp,
                        eval_episodes=1,
                        max_agent_steps=2,
                        max_pixelnav_steps=4,
                        max_env_steps=8,
                        write_videos=True,
                        video_fps=4,
                        memory_video_fps=2,
                        pixelnav_policy_factory=FakePixelNavPolicy,
                        vlm_client_factory=CenterGoThenStopVLM,
                    )

            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
            artifacts = summary["episodes"][0]["video_artifacts"]
            self.assertTrue(Path(artifacts["fps_mp4"]).exists())
            self.assertTrue(Path(artifacts["metric_mp4"]).exists())
            self.assertTrue(Path(artifacts["memory_graph_video"]).exists())

    def test_official_pointnav_config_uses_downloaded_hm3d_assets(self):
        from voca_memory_benchmark import _official_pointnav_config

        config = _official_pointnav_config("hm3d", eval_episodes=3, split="val")

        self.assertEqual(config.habitat.dataset.type, "PointNav-v1")
        self.assertEqual(config.habitat.dataset.split, "val")
        self.assertEqual(
            config.habitat.dataset.data_path,
            "/home/icra/habitat_data/datasets/pointnav/hm3d/v1/{split}/{split}.json.gz",
        )
        self.assertEqual(config.habitat.dataset.scenes_dir, "/home/icra/habitat_data/scene_datasets")
        self.assertEqual(config.habitat.environment.iterator_options.num_episode_sample, 3)
        self.assertEqual(config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width, 640)
        self.assertIn("look_up", config.habitat.task.actions)
        self.assertIn("look_down", config.habitat.task.actions)

    def test_fallback_vlm_client_marks_runtime_fallback(self):
        from voca_memory_benchmark import FallbackVLMClient

        client = FallbackVLMClient(RaisingVLM(), StaticFallbackVLM())
        output = client.decide({"observation": {"image_width": 64, "image_height": 48}})

        self.assertEqual(output["action"], "stop")
        self.assertEqual(client.fallback_count, 1)
        self.assertIn("bad qwen json", output["runtime_fallback"]["reason"])

    def test_pixelnav_friendly_heuristic_uses_mid_floor_point(self):
        from voca_memory_benchmark import PixelNavFriendlyHeuristicVLMClient

        client = PixelNavFriendlyHeuristicVLMClient(y_ratio=0.625)
        output = client.decide(
            {
                "task": {"coarse_goal": {"relative_bearing_deg": 0.0, "distance_m": 5.0}},
                "observation": {
                    "image_width": 256,
                    "image_height": 256,
                    "views": [{"view_id": 0, "view_type": "front", "relative_heading_deg": 0.0}],
                },
                "memory": {"local_topology": {"candidate_exits": []}},
            }
        )

        self.assertEqual(output["action"], "go")
        self.assertEqual(output["selected_image_point"], [128, 160])

    def test_pointnav_bearing_heuristic_rotates_before_goal_is_in_front(self):
        from voca_memory_benchmark import PointNavBearingHeuristicVLMClient

        client = PointNavBearingHeuristicVLMClient(rotate_threshold_deg=25.0)
        output = client.decide(
            {
                "task": {
                    "task_mode": "PointNav",
                    "coarse_goal": {"relative_bearing_deg": 72.0, "distance_m": 5.0},
                },
                "observation": {
                    "image_width": 256,
                    "image_height": 256,
                    "views": [{"view_id": 0, "view_type": "front", "relative_heading_deg": 0.0}],
                },
                "memory": {"local_topology": {"candidate_exits": [{"view_type_hint": "front", "score": 1.0}]}},
            }
        )

        self.assertEqual(output["action"], "rotate")
        self.assertEqual(output["control"]["rotate_yaw_deg"], 72.0)
        self.assertIn("bearing", output["reasoning"]["short_text"])

    def test_pointnav_bearing_heuristic_goes_when_goal_is_in_front(self):
        from voca_memory_benchmark import PointNavBearingHeuristicVLMClient

        client = PointNavBearingHeuristicVLMClient(y_ratio=0.625, rotate_threshold_deg=25.0)
        output = client.decide(
            {
                "task": {
                    "task_mode": "PointNav",
                    "coarse_goal": {"relative_bearing_deg": 12.0, "distance_m": 5.0},
                },
                "observation": {
                    "image_width": 256,
                    "image_height": 256,
                    "views": [{"view_id": 0, "view_type": "front", "relative_heading_deg": 0.0}],
                },
                "memory": {"local_topology": {"candidate_exits": [{"view_type_hint": "left", "score": 1.0}]}},
            }
        )

        self.assertEqual(output["action"], "go")
        self.assertEqual(output["selected_view_type"], "front")
        self.assertEqual(output["selected_image_point"], [128, 160])
        self.assertIn("bearing-aligned", output["reasoning"]["short_text"])

    def test_pointnav_bearing_heuristic_does_not_stop_before_official_success_radius(self):
        from voca_memory_benchmark import PointNavBearingHeuristicVLMClient

        client = PointNavBearingHeuristicVLMClient(y_ratio=0.625, rotate_threshold_deg=25.0)
        output = client.decide(
            {
                "task": {
                    "task_mode": "PointNav",
                    "coarse_goal": {"relative_bearing_deg": 0.0, "distance_m": 0.30},
                },
                "observation": {
                    "image_width": 256,
                    "image_height": 256,
                    "views": [{"view_id": 0, "view_type": "front", "relative_heading_deg": 0.0}],
                },
                "memory": {"local_topology": {"candidate_exits": []}},
            }
        )

        self.assertEqual(output["action"], "go")

    def test_pointnav_bearing_heuristic_escapes_after_collision_before_realigning(self):
        from voca_memory_benchmark import PointNavBearingHeuristicVLMClient

        client = PointNavBearingHeuristicVLMClient(y_ratio=0.625, rotate_threshold_deg=25.0)
        collision_output = client.decide(
            {
                "task": {
                    "task_mode": "PointNav",
                    "coarse_goal": {"relative_bearing_deg": 0.0, "distance_m": 5.0},
                },
                "observation": {
                    "image_width": 256,
                    "image_height": 256,
                    "views": [{"view_id": 0, "view_type": "front", "relative_heading_deg": 0.0}],
                },
                "memory": {
                    "runtime_state": {
                        "last_action_outcome": {
                            "action": "go",
                            "collision": True,
                            "moved_distance_m": 0.05,
                        }
                    }
                },
            }
        )
        recovery_output = client.decide(
            {
                "task": {
                    "task_mode": "PointNav",
                    "coarse_goal": {"relative_bearing_deg": 70.0, "distance_m": 5.0},
                },
                "observation": {
                    "image_width": 256,
                    "image_height": 256,
                    "views": [{"view_id": 0, "view_type": "front", "relative_heading_deg": 0.0}],
                },
                "memory": {
                    "runtime_state": {
                        "last_action_outcome": {
                            "action": "rotate",
                            "collision": False,
                            "moved_distance_m": 0.0,
                        }
                    }
                },
            }
        )

        self.assertEqual(collision_output["action"], "rotate")
        self.assertEqual(abs(collision_output["control"]["rotate_yaw_deg"]), 60.0)
        self.assertEqual(recovery_output["action"], "go")
        self.assertIn("collision recovery", recovery_output["reasoning"]["short_text"])

    def test_make_vlm_client_factory_supports_pointnav_bearing(self):
        from voca_memory_benchmark import PointNavBearingHeuristicVLMClient, make_vlm_client_factory

        client = make_vlm_client_factory("pointnav-bearing", y_ratio=0.7)()

        self.assertIsInstance(client, PointNavBearingHeuristicVLMClient)
        self.assertEqual(client.y_ratio, 0.7)

    def test_make_pixelnav_policy_factory_supports_forward_policy(self):
        from voca_memory_benchmark import ForwardOnlyPixelPolicy, make_pixelnav_policy_factory

        policy = make_pixelnav_policy_factory(None, None, kind="forward")()
        image = np.full((16, 16, 3), 127, dtype=np.uint8)

        policy.reset(image, np.zeros((16, 16), dtype=np.uint8))
        action, overlay = policy.step(image, collide=False)

        self.assertIsInstance(policy, ForwardOnlyPixelPolicy)
        self.assertEqual(action, 1)
        self.assertEqual(overlay.shape, image.shape)

    def test_make_pixelnav_policy_factory_supports_reactive_forward_policy(self):
        from voca_memory_benchmark import ReactiveForwardPixelPolicy, make_pixelnav_policy_factory

        policy = make_pixelnav_policy_factory(None, None, kind="reactive-forward")()
        image = np.full((16, 16, 3), 127, dtype=np.uint8)

        policy.reset(image, np.zeros((16, 16), dtype=np.uint8))
        action_clear, _ = policy.step(image, collide=False)
        action_collide, _ = policy.step(image, collide=True)

        self.assertIsInstance(policy, ReactiveForwardPixelPolicy)
        self.assertEqual(action_clear, 1)
        self.assertIn(action_collide, {2, 3})

    def test_pointgoal_reactive_policy_uses_pointgoal_sensor(self):
        from voca_memory_benchmark import PointGoalReactivePixelPolicy

        policy = PointGoalReactivePixelPolicy(success_distance_m=0.2, turn_threshold_deg=15.0)
        image = np.full((16, 16, 3), 127, dtype=np.uint8)
        policy.reset(image, np.zeros((16, 16), dtype=np.uint8))

        stop_action, _ = policy.step_from_observation(
            {"pointgoal_with_gps_compass": np.asarray([0.1, 0.0], dtype=np.float32)},
            image,
        )
        forward_action, _ = policy.step_from_observation(
            {"pointgoal_with_gps_compass": np.asarray([1.0, 0.0], dtype=np.float32)},
            image,
        )
        left_action, _ = policy.step_from_observation(
            {"pointgoal_with_gps_compass": np.asarray([1.0, np.deg2rad(45.0)], dtype=np.float32)},
            image,
        )
        right_action, _ = policy.step_from_observation(
            {"pointgoal_with_gps_compass": np.asarray([1.0, np.deg2rad(-45.0)], dtype=np.float32)},
            image,
        )

        self.assertEqual(stop_action, 0)
        self.assertEqual(forward_action, 1)
        self.assertEqual(left_action, 2)
        self.assertEqual(right_action, 3)

    def test_make_pixelnav_policy_factory_supports_pointgoal_reactive_policy(self):
        from voca_memory_benchmark import PointGoalReactivePixelPolicy, make_pixelnav_policy_factory

        policy = make_pixelnav_policy_factory(None, None, kind="pointgoal-reactive")()

        self.assertIsInstance(policy, PointGoalReactivePixelPolicy)

    def test_make_pixelnav_policy_factory_supports_shortest_path_policy(self):
        from voca_memory_benchmark import ShortestPathFollowerPixelPolicy, make_pixelnav_policy_factory

        policy = make_pixelnav_policy_factory(None, None, kind="shortest-path")()

        self.assertIsInstance(policy, ShortestPathFollowerPixelPolicy)


if __name__ == "__main__":
    unittest.main()
