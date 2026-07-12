import tempfile
from pathlib import Path
import unittest

import numpy as np

from nav_memory_qwen.memory_graph import MemoryGraph
from nav_memory_qwen.schema import RelativePose2D
from nav_memory_qwen.robot_backend import StaticImageBackend
from nav_memory_qwen.agent import NavMemoryAgent, NavAgentConfig
from nav_memory_qwen.vlm_client import HeuristicVLMClient


class MemoryGraphTest(unittest.TestCase):
    def test_relative_pose_composition(self):
        a = RelativePose2D(1.0, 0.0, 90.0)
        b = RelativePose2D(1.0, 0.0, 0.0)
        c = a.compose(b)
        self.assertAlmostEqual(c.dx_m, 1.0, places=5)
        self.assertAlmostEqual(c.dy_m, 1.0, places=5)
        self.assertAlmostEqual(c.dyaw_deg, 90.0, places=5)

    def test_add_deadlock_edge(self):
        g = MemoryGraph(embedding_dim=4)
        n1 = g.add_node(frame_index=1, image_ref=None, embedding=np.array([1, 0, 0, 0], dtype=np.float32))
        n2 = g.add_node(frame_index=2, image_ref=None, embedding=np.array([0, 1, 0, 0], dtype=np.float32))
        eid = g.add_or_update_edge(n1, n2, RelativePose2D(1, 0, 0), status="success")
        g.mark_deadlock(n2, eid)
        self.assertTrue(g.edges[eid].is_negative())
        ctx = g.build_vlm_memory_context(goal_bearing_deg=0, goal_distance_m=3)
        self.assertGreaterEqual(ctx["graph_summary"]["num_deadlock_edges"], 1)

    def test_mock_episode_runs(self):
        with tempfile.TemporaryDirectory() as td:
            img = StaticImageBackend.create_demo_image(Path(td) / "img.jpg")
            robot = StaticImageBackend(img, step_m=1.0)
            agent = NavMemoryAgent(robot=robot, vlm_client=HeuristicVLMClient(), config=NavAgentConfig(max_steps=8))
            result = agent.run_until_done(goal_map_xy=(2.0, 0.0), max_steps=8)
            self.assertGreaterEqual(result.steps, 1)
            self.assertGreaterEqual(len(result.graph.nodes), 1)


if __name__ == "__main__":
    unittest.main()
