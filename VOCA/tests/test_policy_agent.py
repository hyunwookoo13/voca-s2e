import unittest
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policy_agent import PolicyAgent


class _FakePolicyNetwork:
    def __init__(self, token_length):
        self.token_length = token_length
        self.grad_enabled_seen = None
        self.sequence_lengths_seen = []

    def __call__(self, goal_mask, goal_image, input_image):
        self.grad_enabled_seen = torch.is_grad_enabled()
        batch_size, sequence_length = input_image.shape[:2]
        self.sequence_lengths_seen.append(sequence_length)
        action_pred = torch.zeros((batch_size, sequence_length, 6), dtype=torch.float32)
        action_pred[:, 0, 1] = 1.0
        action_pred[:, 0, 2] = 0.5
        distance_pred = torch.zeros((batch_size, sequence_length, 1), dtype=torch.float32)
        goal_pred = torch.zeros((batch_size, sequence_length, 2), dtype=torch.float32)
        goal_pred[:, 0] = torch.tensor([0.5, 0.5])
        return action_pred, distance_pred, goal_pred


class PolicyAgentTests(unittest.TestCase):
    def test_step_runs_policy_forward_without_gradient_tracking(self):
        agent = PolicyAgent.__new__(PolicyAgent)
        agent.image_size = 224
        agent.max_token_length = 64
        agent.device = "cpu"
        agent.network = _FakePolicyNetwork(agent.max_token_length)

        goal_image = np.zeros((32, 32, 3), dtype=np.uint8)
        goal_mask = np.zeros((32, 32), dtype=np.uint8)
        obs_image = np.zeros((480, 640, 3), dtype=np.uint8)

        agent.reset(goal_image, goal_mask)
        action, skill_image = agent.step(obs_image)

        self.assertEqual(action, 1)
        self.assertEqual(skill_image.shape, cv2.cvtColor(obs_image, cv2.COLOR_BGR2RGB).shape)
        self.assertFalse(agent.network.grad_enabled_seen)
        self.assertEqual(agent.network.sequence_lengths_seen, [1])
        self.assertEqual(agent.history_image.dtype, np.uint8)

        agent.step(obs_image)
        self.assertEqual(agent.network.sequence_lengths_seen, [1, 2])

    def test_step_can_adjust_collision_action_after_inference_forward(self):
        agent = PolicyAgent.__new__(PolicyAgent)
        agent.image_size = 224
        agent.max_token_length = 64
        agent.device = "cpu"
        agent.network = _FakePolicyNetwork(agent.max_token_length)

        goal_image = np.zeros((32, 32, 3), dtype=np.uint8)
        goal_mask = np.zeros((32, 32), dtype=np.uint8)
        obs_image = np.zeros((480, 640, 3), dtype=np.uint8)

        agent.reset(goal_image, goal_mask)
        action, _ = agent.step(obs_image, collide=True)

        self.assertEqual(action, 2)

    def test_scores_pixel_candidates_in_one_non_mutating_batch(self):
        agent = PolicyAgent.__new__(PolicyAgent)
        agent.image_size = 32
        agent.max_token_length = 64
        agent.device = "cpu"
        agent.network = _FakePolicyNetwork(agent.max_token_length)
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        candidates = [
            {"candidate_ref": "px_1", "view_id": 0, "point_px": [15, 35]},
            {"candidate_ref": "px_2", "view_id": 0, "point_px": [45, 35]},
        ]

        results = agent.score_pixel_candidates([image], candidates)

        self.assertEqual([item["candidate_ref"] for item in results], ["px_1", "px_2"])
        self.assertTrue(all(item["available"] for item in results))
        self.assertTrue(all(item["predicted_action"] == "forward" for item in results))
        self.assertTrue(
            all(0.0 <= item["policy_feasibility_score"] <= 1.0 for item in results)
        )
        self.assertFalse(agent.network.grad_enabled_seen)


if __name__ == "__main__":
    unittest.main()
