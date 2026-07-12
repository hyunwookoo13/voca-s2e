import unittest
from pathlib import Path

from voca_s2e_bridge import (
    NAV_MEMORY_QWEN_ROOT,
    VOCA_ROOT,
    import_memory_graph_visualizer,
    import_nav_memory_qwen,
)


class VocaS2EBridgeTests(unittest.TestCase):
    def test_bridge_loads_memory_graph_visualizer(self):
        visualizer = import_memory_graph_visualizer()

        self.assertEqual(Path(visualizer.__file__).resolve().parent, VOCA_ROOT)
        self.assertTrue(hasattr(visualizer, "render_memory_graph_snapshot"))
        self.assertTrue(hasattr(visualizer, "render_memory_graph_video"))

    def test_bridge_loads_canonical_v6_nav_memory_modules(self):
        modules = import_nav_memory_qwen()

        self.assertTrue(NAV_MEMORY_QWEN_ROOT.is_dir())
        self.assertIn(str(NAV_MEMORY_QWEN_ROOT), modules.schema.__file__)
        self.assertIn(str(NAV_MEMORY_QWEN_ROOT), modules.safety.__file__)
        self.assertIn(str(NAV_MEMORY_QWEN_ROOT), modules.vlm_client.__file__)
        self.assertIn(str(NAV_MEMORY_QWEN_ROOT), modules.memory_graph.__file__)
        self.assertIn(str(NAV_MEMORY_QWEN_ROOT), modules.policy.__file__)
        self.assertEqual(modules.schema.ALLOWED_ACTIONS, ["go", "rotate", "stop", "request_observation"])
        self.assertTrue(hasattr(modules.memory_graph, "MemoryGraph"))
        self.assertTrue(hasattr(modules.policy, "apply_gap_lite_validation"))


if __name__ == "__main__":
    unittest.main()
