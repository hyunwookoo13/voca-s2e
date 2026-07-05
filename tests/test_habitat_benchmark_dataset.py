import json
import tempfile
import unittest
from pathlib import Path

from goal_adapter.habitat_benchmark_dataset import (
    HabitatBenchmarkConfig,
    _point_component,
    generate_habitat_benchmark_cases,
)
from goal_adapter.vlm_benchmark import load_vlm_benchmark_cases


class FakeHabitatSimModule:
    class SensorType:
        COLOR = "color"
        DEPTH = "depth"

    class SimulatorConfiguration:
        def __init__(self):
            self.scene_id = None

    class CameraSensorSpec:
        def __init__(self):
            self.uuid = None
            self.sensor_type = None
            self.resolution = None
            self.position = None
            self.hfov = None

    class Configuration:
        def __init__(self, simulator_config, agent_configs):
            self.simulator_config = simulator_config
            self.agent_configs = agent_configs

    class agent:
        class AgentConfiguration:
            def __init__(self):
                self.sensor_specifications = []

    def __init__(self):
        self.simulators = []

    def Simulator(self, config):
        simulator = FakeSimulator(config)
        self.simulators.append(simulator)
        return simulator


class FakePathfinder:
    is_loaded = True

    def __init__(self):
        self.points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.2],
            [2.0, 0.0, 0.4],
            [3.0, 0.0, 0.6],
            [4.0, 0.0, 0.8],
            [5.0, 0.0, 1.0],
            [6.0, 0.0, 1.2],
            [7.0, 0.0, 1.4],
            [8.0, 0.0, 1.6],
            [9.0, 0.0, 1.8],
            [10.0, 0.0, 2.0],
            [11.0, 0.0, 2.2],
        ]
        self.index = 0

    def seed(self, value):
        self.seed_value = value

    def get_random_navigable_point(self):
        point = self.points[self.index % len(self.points)]
        self.index += 1
        return list(point)

    def snap_point(self, point):
        return list(point)

    def is_navigable(self, point):
        return float(point[0]) >= 0.0

    def get_bounds(self):
        return (FakeVector3([-1.0, 0.0, -1.0]), FakeVector3([12.0, 0.0, 3.0]))


class FakeVector3:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        return self.values[index]


class FakeAgentState:
    def __init__(self):
        self.position = [0.0, 0.0, 0.0]


class FakeAgent:
    def __init__(self):
        self.state = FakeAgentState()

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state


class FakeSimulator:
    def __init__(self, config):
        self.config = config
        self.pathfinder = FakePathfinder()
        self.agent = FakeAgent()
        self.closed = False
        self.observation_count = 0

    def initialize_agent(self, agent_id):
        self.agent_id = agent_id
        return self.agent

    def get_sensor_observations(self):
        self.observation_count += 1
        value = 20 + self.observation_count
        depth = [
            [0.2 for _column in range(96)]
            for _row in range(72)
        ]
        for row in range(55, 66):
            for column in range(68, 82):
                depth[row][column] = 5.0
        return {
            "rgb": [
                [[value, 30, 40], [50, 60, 70], [80, 90, 100]],
                [[110, 120, 130], [140, 150, 160], [170, 180, 190]],
            ],
            "depth": depth,
        }

    def close(self):
        self.closed = True


class HabitatBenchmarkDatasetTest(unittest.TestCase):
    def test_point_component_reads_indexable_vector_objects(self):
        try:
            component = _point_component(FakeVector3([-1.0, 0.0, 3.5]), 2)
        except ValueError as exc:
            self.fail(str(exc))

        self.assertEqual(component, 3.5)

    def test_generate_habitat_benchmark_cases_writes_four_categories(self):
        fake_habitat_sim = FakeHabitatSimModule()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "cases"
            dataset = generate_habitat_benchmark_cases(
                HabitatBenchmarkConfig(
                    scene_paths=["scene_a.glb"],
                    output_dir=output_dir,
                    cases_per_category=1,
                    image_width=96,
                    image_height=72,
                    seed=7,
                ),
                habitat_sim_module=fake_habitat_sim,
            )

            self.assertEqual(len(dataset.case_paths), 4)
            self.assertEqual(dataset.total_cases, 4)
            self.assertTrue((output_dir / "assets").exists())
            cases = load_vlm_benchmark_cases(output_dir)

            self.assertEqual(
                [case.category for case in cases],
                ["coarse_gps", "coarse_object_point", "tracking_loss", "deadlock"],
            )
            self.assertEqual(cases[0].expected_action_type.value, "NAVIGATE")
            self.assertEqual(cases[1].expected_action_type.value, "NAVIGATE")
            self.assertEqual(cases[2].expected_action_type.value, "LOOK_AROUND")
            self.assertEqual(cases[2].reasoning_terms, ["tracking_loss"])
            self.assertEqual(cases[3].expected_action_type.value, "RESELECT_GOAL")
            self.assertTrue(cases[3].forbidden_goal_xy)

            for path in dataset.case_paths:
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertIn("scene_id", payload["input"]["memory_summary"])
                self.assertIn("metric_map", payload["input"]["memory_summary"])
                metric_map = payload["input"]["memory_summary"]["metric_map"]
                self.assertEqual(metric_map["meters_per_pixel"], 0.1)
                self.assertTrue((path.parent / metric_map["path"]).exists())
                self.assertIn("bounds", metric_map)
                self.assertIn("camera_fov", payload["input"]["memory_summary"])
                self.assertEqual(
                    payload["input"]["memory_summary"]["camera_fov"],
                    {"hfov_degrees": 79.0, "range_meters": 3.0},
                )
                self.assertIn("trajectory_xy", payload["input"]["memory_summary"])
                self.assertGreaterEqual(len(payload["input"]["memory_summary"]["trajectory_xy"]), 2)
                for candidate in payload["input"]["memory_summary"].get("candidate_waypoints", []):
                    image_point = candidate.get("image_point")
                    if candidate.get("navigable") is True:
                        self.assertGreaterEqual(image_point[1], int(72 * 0.75))
                        self.assertGreaterEqual(image_point[0], int(96 * 0.65))
                    else:
                        self.assertLess(image_point[1], int(72 * 0.75))
                self.assertTrue((path.parent / payload["input"]["current_rgb"]).exists())

        self.assertTrue(fake_habitat_sim.simulators[0].closed)


if __name__ == "__main__":
    unittest.main()
