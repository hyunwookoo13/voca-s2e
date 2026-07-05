import json
import tempfile
import unittest
from pathlib import Path

from goal_adapter.habitat_sim_smoke import HabitatSimSmokeConfig, capture_habitat_sim_smoke_sample


class FakeHabitatSimModule:
    class SensorType:
        COLOR = "color"

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
        self.last_simulator = None

    def Simulator(self, config):
        self.last_simulator = FakeSimulator(config)
        return self.last_simulator


class FakePathfinder:
    is_loaded = True

    def get_random_navigable_point(self):
        return [1.0, 2.0, 3.0]


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

    def initialize_agent(self, agent_id):
        self.agent_id = agent_id
        return self.agent

    def get_sensor_observations(self):
        return {"rgb": [[[11, 22, 33], [44, 55, 66]]]}

    def close(self):
        self.closed = True


class HabitatSimSmokeRunnerTest(unittest.TestCase):
    def test_capture_habitat_sim_scene_writes_rgb_and_input_json(self):
        fake_habitat_sim = FakeHabitatSimModule()

        with tempfile.TemporaryDirectory() as temp_dir:
            sample = capture_habitat_sim_smoke_sample(
                HabitatSimSmokeConfig(
                    scene_path="scene.glb",
                    output_dir=Path(temp_dir),
                    target_type="language",
                    high_level_target="inspect the room",
                    image_width=320,
                    image_height=240,
                    sensor_height=0.88,
                    image_hfov=79.0,
                ),
                habitat_sim_module=fake_habitat_sim,
            )

            self.assertTrue(sample.image_path.exists())
            self.assertTrue(sample.input_path.exists())
            payload = json.loads(sample.input_path.read_text(encoding="utf-8"))

        simulator_config = fake_habitat_sim.last_simulator.config.simulator_config
        sensor_spec = fake_habitat_sim.last_simulator.config.agent_configs[0].sensor_specifications[0]
        self.assertEqual(simulator_config.scene_id, "scene.glb")
        self.assertEqual(sensor_spec.uuid, "rgb")
        self.assertEqual(sensor_spec.resolution, [240, 320])
        self.assertEqual(sensor_spec.position, [0.0, 0.88, 0.0])
        self.assertEqual(payload["target_type"], "language")
        self.assertEqual(payload["high_level_target"], "inspect the room")
        self.assertEqual(payload["current_pose"], {"x": 1.0, "y": 3.0, "z": 2.0})
        self.assertTrue(fake_habitat_sim.last_simulator.closed)


if __name__ == "__main__":
    unittest.main()
