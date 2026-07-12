import json
import math
import tempfile
import unittest
from pathlib import Path

from coarse_goal_provider import CoarseGoalProvider


class _Episode:
    episode_id = "ep-10"
    scene_id = "/data/00844-room/room.basis.glb"
    object_category = "chair"


class CoarseGoalProviderTests(unittest.TestCase):
    def test_external_s2e_goal_resolves_and_updates_robot_relative_geometry(self):
        payload = {
            "schema_version": "voca_coarse_goal_dataset_v1",
            "records": [
                {
                    "episode_index": 0,
                    "episode_id": "ep-10",
                    "scene_id": "room",
                    "object_goal": "chair",
                    "source": "s2e_backbone",
                    "type": "map_waypoint",
                    "map_xy": [0.0, 2.0],
                    "uncertainty": "medium",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "goals.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            provider = CoarseGoalProvider.from_path(str(path))
            record = provider.resolve(episode_index=0, episode=_Episode())

        runtime = provider.runtime_goal(
            record,
            position_xyz=[0.0, 0.0, 0.0],
            heading_rad=0.0,
        )
        self.assertEqual(runtime["source"], "s2e_backbone")
        self.assertEqual(runtime["map_xy"], [0.0, 2.0])
        self.assertEqual(runtime["relative_bearing_deg"], 90.0)
        self.assertEqual(runtime["distance_m"], 2.0)
        self.assertEqual(runtime["distance_range_m"], [0.5, 3.5])
        self.assertEqual(len(record["provider_sha256"]), 64)

        rotated = provider.runtime_goal(
            record,
            position_xyz=[0.0, 0.0, 0.0],
            heading_rad=math.pi / 2.0,
        )
        self.assertEqual(rotated["relative_bearing_deg"], 0.0)

    def test_missing_episode_fails_closed(self):
        provider = CoarseGoalProvider(
            [
                {
                    "episode_index": 2,
                    "source": "user_instruction",
                    "map_xy": [1.0, 2.0],
                }
            ]
        )
        with self.assertRaises(KeyError):
            provider.resolve(episode_index=0, episode=_Episode())

    def test_oracle_source_is_rejected(self):
        with self.assertRaises(ValueError):
            CoarseGoalProvider(
                [
                    {
                        "episode_index": 0,
                        "source": "habitat_ground_truth",
                        "map_xy": [1.0, 2.0],
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
