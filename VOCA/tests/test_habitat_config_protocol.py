import os
import unittest
from unittest.mock import patch

from habitat_config import hm3d_config


class HabitatBenchmarkProtocolTests(unittest.TestCase):
    def test_hm3d_defaults_to_standard_success_radius_and_no_sliding(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VOCA_HM3D_SUCCESS_DISTANCE_M", None)
            os.environ.pop("VOCA_HM3D_ALLOW_SLIDING", None)
            config = hm3d_config(episodes=1)

        self.assertAlmostEqual(
            float(config.habitat.task.measurements.success.success_distance),
            0.10,
        )
        self.assertFalse(config.habitat.simulator.habitat_sim_v0.allow_sliding)

    def test_hm3d_integration_profile_must_be_explicit(self):
        with patch.dict(
            os.environ,
            {
                "VOCA_HM3D_SUCCESS_DISTANCE_M": "1.0",
                "VOCA_HM3D_ALLOW_SLIDING": "1",
            },
        ):
            config = hm3d_config(episodes=1)

        self.assertAlmostEqual(
            float(config.habitat.task.measurements.success.success_distance),
            1.0,
        )
        self.assertTrue(config.habitat.simulator.habitat_sim_v0.allow_sliding)

    def test_requested_episode_step_limit_updates_habitat_environment(self):
        config = hm3d_config(episodes=1, max_episode_steps=1000)

        self.assertEqual(
            int(config.habitat.environment.max_episode_steps),
            1000,
        )


if __name__ == "__main__":
    unittest.main()
