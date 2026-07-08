import json
import tempfile
import unittest
from pathlib import Path

from goal_adapter.memory_graph_visualizer import (
    MemoryGraphPose,
    reconstruct_memory_graph_layout,
    render_memory_graph_snapshot,
    render_memory_graph_video,
)


class MemoryGraphVisualizerTest(unittest.TestCase):
    def test_reconstructs_node_positions_from_relative_edges(self):
        graph = {
            "schema_version": "relative_topometric_memory_graph_v5",
            "current_node_id": "n_00003",
            "nodes": {
                "n_00001": {"node_id": "n_00001"},
                "n_00002": {"node_id": "n_00002"},
                "n_00003": {"node_id": "n_00003"},
            },
            "edges": {
                "e_00001": {
                    "edge_id": "e_00001",
                    "src_node_id": "n_00001",
                    "dst_node_id": "n_00002",
                    "edge_type": "temporal_transition",
                    "traversal": {"status": "success"},
                    "relative_pose_src_to_dst": {"dx_m": 2.0, "dy_m": 0.0, "dyaw_deg": 90.0},
                },
                "e_00002": {
                    "edge_id": "e_00002",
                    "src_node_id": "n_00002",
                    "dst_node_id": "n_00003",
                    "edge_type": "temporal_transition",
                    "traversal": {"status": "success"},
                    "relative_pose_src_to_dst": {"dx_m": 1.0, "dy_m": 0.0, "dyaw_deg": 0.0},
                },
            },
        }

        layout = reconstruct_memory_graph_layout(graph)

        self.assertEqual(layout.node_count, 3)
        self.assertEqual(layout.edge_count, 2)
        self.assertEqual(layout.current_node_id, "n_00003")
        self.assertAlmostEqual(layout.poses["n_00001"].x_m, 0.0)
        self.assertAlmostEqual(layout.poses["n_00001"].y_m, 0.0)
        self.assertAlmostEqual(layout.poses["n_00002"].x_m, 2.0)
        self.assertAlmostEqual(layout.poses["n_00002"].y_m, 0.0)
        self.assertAlmostEqual(layout.poses["n_00002"].yaw_deg, 90.0)
        self.assertAlmostEqual(layout.poses["n_00003"].x_m, 2.0)
        self.assertAlmostEqual(layout.poses["n_00003"].y_m, 1.0)
        self.assertEqual(layout.poses["n_00001"], MemoryGraphPose(0.0, 0.0, 0.0, 0))

    def test_render_memory_graph_snapshot_writes_png_html_and_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            graph_path = temp_path / "memory_graph.json"
            graph_path.write_text(
                json.dumps(
                    {
                        "schema_version": "relative_topometric_memory_graph_v5",
                        "current_node_id": "n_00002",
                        "nodes": {
                            "n_00001": {"node_id": "n_00001", "place_category": "unknown"},
                            "n_00002": {"node_id": "n_00002", "place_category": "corridor"},
                        },
                        "edges": {
                            "e_00001": {
                                "edge_id": "e_00001",
                                "src_node_id": "n_00001",
                                "dst_node_id": "n_00002",
                                "edge_type": "temporal_transition",
                                "traversal": {"status": "success"},
                                "relative_pose_src_to_dst": {"dx_m": 1.25, "dy_m": 0.25, "dyaw_deg": 0.0},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = render_memory_graph_snapshot(
                graph_path,
                output_dir=temp_path / "viz",
                title="test graph",
            )

            self.assertTrue(Path(result["png_path"]).exists())
            self.assertTrue(Path(result["html_path"]).exists())
            self.assertTrue(Path(result["summary_json"]).exists())
            self.assertGreater(Path(result["png_path"]).stat().st_size, 1000)
            html = Path(result["html_path"]).read_text(encoding="utf-8")
            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))

        self.assertIn("test graph", html)
        self.assertIn("memory_graph.png", html)
        self.assertEqual(summary["node_count"], 2)
        self.assertEqual(summary["edge_count"], 1)
        self.assertEqual(summary["current_node_id"], "n_00002")

    def test_render_memory_graph_video_writes_animation_and_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            graph_path = temp_path / "memory_graph.json"
            graph_path.write_text(
                json.dumps(
                    {
                        "schema_version": "relative_topometric_memory_graph_v5",
                        "current_node_id": "n_00003",
                        "nodes": {
                            "n_00001": {"node_id": "n_00001", "place_category": "unknown"},
                            "n_00002": {"node_id": "n_00002", "place_category": "corridor"},
                            "n_00003": {"node_id": "n_00003", "place_category": "doorway"},
                        },
                        "edges": {
                            "e_00001": {
                                "edge_id": "e_00001",
                                "src_node_id": "n_00001",
                                "dst_node_id": "n_00002",
                                "edge_type": "temporal_transition",
                                "traversal": {"status": "success"},
                                "relative_pose_src_to_dst": {"dx_m": 1.25, "dy_m": 0.0, "dyaw_deg": 0.0},
                            },
                            "e_00002": {
                                "edge_id": "e_00002",
                                "src_node_id": "n_00002",
                                "dst_node_id": "n_00003",
                                "edge_type": "same_place_constraint",
                                "relation_type": "same_place",
                                "relative_pose_src_to_dst": {"dx_m": 0.2, "dy_m": 0.1, "dyaw_deg": 5.0},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = render_memory_graph_video(
                graph_path,
                output_dir=temp_path / "viz",
                output_video="memory_graph_reconstruction.gif",
                title="test graph replay",
                fps=2,
                hold_frames=2,
            )

            video_path = Path(result["video_path"])
            summary_path = Path(result["summary_json"])
            self.assertTrue(video_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertGreater(video_path.stat().st_size, 1000)
            self.assertEqual(result["video_format"], "gif")
            self.assertGreaterEqual(result["frame_count"], 4)

            from PIL import Image

            with Image.open(video_path) as image:
                self.assertGreaterEqual(getattr(image, "n_frames", 1), 4)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["node_count"], 3)
        self.assertEqual(summary["edge_count"], 2)
        self.assertEqual(summary["video_path"], str(video_path))


if __name__ == "__main__":
    unittest.main()
