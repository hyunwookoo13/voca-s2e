import unittest
from pathlib import Path

import settings


class SetupPathTests(unittest.TestCase):
    def test_default_habitat_objectnav_configs_exist(self):
        if not Path(settings.HM3D_CONFIG_PATH).is_file():
            self.skipTest("Habitat ObjectNav configs are not installed")
        self.assertTrue(Path(settings.HM3D_CONFIG_PATH).is_file(), settings.HM3D_CONFIG_PATH)
        self.assertTrue(Path(settings.MP3D_CONFIG_PATH).is_file(), settings.MP3D_CONFIG_PATH)

    def test_default_voca_data_links_exist(self):
        if not Path(settings.POLICY_CHECKPOINT).is_file():
            self.skipTest("VOCA datasets/checkpoints are not installed")
        self.assertTrue(Path(settings.SCENE_PREFIX, "hm3d_v0.2").exists())
        self.assertTrue(Path(settings.EPISODE_PREFIX, "objectnav/hm3d/v2/val/val.json.gz").is_file())
        self.assertTrue(Path(settings.POLICY_CHECKPOINT).is_file(), settings.POLICY_CHECKPOINT)
        self.assertTrue(Path(settings.YOLOE_CHECKPOINT_PATH).is_file(), settings.YOLOE_CHECKPOINT_PATH)


if __name__ == "__main__":
    unittest.main()
