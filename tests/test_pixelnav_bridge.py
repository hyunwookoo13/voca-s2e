import unittest
from unittest.mock import patch

import numpy as np

from goal_adapter.pixelnav_bridge import (
    PixelNavPolicyExecutor,
    _default_torch_device,
    clamp_pixel_point,
    make_goal_mask,
    make_pixelnav_goal,
)


class PixelNavGoalMaskTest(unittest.TestCase):
    def test_make_pixelnav_goal_creates_mask_centered_on_selected_point(self):
        rgb = np.zeros((8, 10, 3), dtype=np.uint8)

        goal = make_pixelnav_goal(rgb, [5, 4], radius=1)

        self.assertEqual(goal.goal_image.shape, (8, 10, 3))
        self.assertEqual(goal.goal_mask.dtype, np.uint8)
        self.assertEqual(goal.goal_mask.shape, (8, 10))
        self.assertEqual(goal.goal_mask[3:6, 4:7].min(), 255)
        self.assertEqual(goal.goal_mask.sum(), 9 * 255)
        self.assertEqual(goal.selected_point_uv, (5, 4))

    def test_make_goal_mask_clips_at_image_boundary(self):
        mask = make_goal_mask((6, 7), [0, 0], radius=2)

        self.assertEqual(mask.shape, (6, 7))
        self.assertEqual(mask[0:3, 0:3].min(), 255)
        self.assertEqual(mask.sum(), 9 * 255)

    def test_clamp_pixel_point_rejects_malformed_point(self):
        with self.assertRaisesRegex(ValueError, "selected point"):
            clamp_pixel_point([3], width=10, height=8)

    def test_make_pixelnav_goal_rejects_non_rgb_image(self):
        with self.assertRaisesRegex(ValueError, "RGB image"):
            make_pixelnav_goal(np.zeros((8, 10), dtype=np.uint8), [5, 4])


class FakePolicy:
    def __init__(self):
        self.reset_args = None
        self.step_args = None

    def reset(self, goal_image, goal_mask):
        self.reset_args = (goal_image, goal_mask)

    def step(self, obs_rgb, collide=False):
        self.step_args = (obs_rgb, collide)
        return 1, obs_rgb.copy()


class PixelNavPolicyExecutorTest(unittest.TestCase):
    def test_reset_from_point_passes_goal_image_and_mask_to_policy(self):
        fake_policy = FakePolicy()
        executor = PixelNavPolicyExecutor(policy=fake_policy)
        rgb = np.zeros((8, 10, 3), dtype=np.uint8)

        goal = executor.reset_from_point(rgb, [5, 4], radius=1)

        self.assertIsNotNone(fake_policy.reset_args)
        reset_image, reset_mask = fake_policy.reset_args
        self.assertTrue(np.array_equal(reset_image, goal.goal_image))
        self.assertTrue(np.array_equal(reset_mask, goal.goal_mask))
        self.assertEqual(reset_mask[3:6, 4:7].min(), 255)

    def test_step_returns_action_overlay_and_forwards_collision_flag(self):
        fake_policy = FakePolicy()
        executor = PixelNavPolicyExecutor(policy=fake_policy)
        rgb = np.zeros((8, 10, 3), dtype=np.uint8)

        result = executor.step(rgb, collide=True)

        self.assertEqual(result.action, 1)
        self.assertTrue(np.array_equal(result.overlay_image, rgb))
        self.assertIs(fake_policy.step_args[1], True)

    def test_default_torch_device_honors_pixelnav_device_env(self):
        with patch.dict("os.environ", {"PIXELNAV_DEVICE": "cpu"}, clear=False):
            device = _default_torch_device()

        self.assertEqual(device, "cpu")


if __name__ == "__main__":
    unittest.main()
