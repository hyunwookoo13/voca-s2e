from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class MemoryGraphPose:
    x_m: float
    y_m: float
    yaw_deg: float
    component_index: int


@dataclass(frozen=True)
class MemoryGraphLayout:
    poses: dict[str, MemoryGraphPose]
    nodes: dict[str, dict[str, Any]]
    edges: dict[str, dict[str, Any]]
    schema_version: str | None
    current_node_id: str | None

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def to_summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "current_node_id": self.current_node_id,
            "components": max((pose.component_index for pose in self.poses.values()), default=-1) + 1,
            "nodes": {
                node_id: {
                    "x_m": round(pose.x_m, 4),
                    "y_m": round(pose.y_m, 4),
                    "yaw_deg": round(_normalize_deg(pose.yaw_deg), 3),
                    "component_index": pose.component_index,
                    "place_category": self.nodes.get(node_id, {}).get("place_category"),
                }
                for node_id, pose in sorted(self.poses.items())
            },
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize reconstructed v5 memory graph nodes from relative SE(2) edges.")
    parser.add_argument("--graph", required=True, help="Path to memory_graph.json")
    parser.add_argument("--out", required=True, help="Output directory for PNG/HTML/summary artifacts")
    parser.add_argument("--title", default="Habitat + PixelNav Memory Graph")
    parser.add_argument("--serve", action="store_true", help="Serve an auto-refreshing visualizer process")
    parser.add_argument("--video", action="store_true", help="Render a node/edge reveal animation")
    parser.add_argument("--video-out", default="memory_graph_reconstruction.mp4", help="Video filename or path")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--hold-frames", type=int, default=4)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--watch-interval", type=float, default=2.0)
    args = parser.parse_args(argv)

    if args.serve:
        serve_memory_graph_visualizer(
            args.graph,
            output_dir=args.out,
            title=args.title,
            host=args.host,
            port=args.port,
            watch_interval_s=args.watch_interval,
        )
        return 0

    result = render_memory_graph_snapshot(args.graph, output_dir=args.out, title=args.title)
    if args.video:
        video_result = render_memory_graph_video(
            args.graph,
            output_dir=args.out,
            output_video=args.video_out,
            title=args.title,
            fps=args.fps,
            hold_frames=args.hold_frames,
        )
        result = {**result, "video": video_result}
    print(json.dumps({"status": "rendered", **result}, ensure_ascii=False))
    return 0


def load_memory_graph(graph_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(graph_path).read_text(encoding="utf-8"))


def reconstruct_memory_graph_layout(graph: dict[str, Any]) -> MemoryGraphLayout:
    nodes = _normalize_items(graph.get("nodes"), id_key="node_id")
    edges = _normalize_items(graph.get("edges"), id_key="edge_id")
    poses: dict[str, MemoryGraphPose] = {}
    adjacency = _build_adjacency(edges)

    component_index = 0
    for root_id in _ordered_node_ids(nodes, graph.get("current_node_id")):
        if root_id in poses:
            continue
        x_offset = float(component_index) * 4.0
        poses[root_id] = MemoryGraphPose(x_offset, 0.0, 0.0, component_index)
        queue = [root_id]
        while queue:
            node_id = queue.pop(0)
            src_pose = poses[node_id]
            for edge_id, neighbor_id, rel_pose, forward in adjacency.get(node_id, []):
                if neighbor_id in poses:
                    continue
                next_pose = _compose(src_pose, rel_pose) if forward else _compose(src_pose, _inverse(rel_pose))
                poses[neighbor_id] = MemoryGraphPose(
                    next_pose.x_m,
                    next_pose.y_m,
                    next_pose.yaw_deg,
                    component_index,
                )
                queue.append(neighbor_id)
        component_index += 1

    return MemoryGraphLayout(
        poses=poses,
        nodes=nodes,
        edges=edges,
        schema_version=graph.get("schema_version"),
        current_node_id=graph.get("current_node_id"),
    )


def render_memory_graph_snapshot(
    graph_path: str | Path,
    *,
    output_dir: str | Path,
    title: str = "Habitat + PixelNav Memory Graph",
) -> dict[str, Any]:
    graph_path = Path(graph_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = load_memory_graph(graph_path)
    layout = reconstruct_memory_graph_layout(graph)
    png_path = output_dir / "memory_graph.png"
    html_path = output_dir / "memory_graph_visualizer.html"
    summary_path = output_dir / "memory_graph_visualizer_summary.json"

    _render_png(layout, png_path, title=title, graph_path=graph_path)
    summary = {
        **layout.to_summary(),
        "graph_path": str(graph_path),
        "png_path": str(png_path),
        "html_path": str(html_path),
        "updated_at_unix_s": round(time.time(), 3),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    html_path.write_text(_render_html(title=title, summary=summary, refresh_ms=2000), encoding="utf-8")
    return {
        "png_path": str(png_path),
        "html_path": str(html_path),
        "summary_json": str(summary_path),
        "node_count": layout.node_count,
        "edge_count": layout.edge_count,
        "current_node_id": layout.current_node_id,
    }


def render_memory_graph_video(
    graph_path: str | Path,
    *,
    output_dir: str | Path,
    output_video: str | Path = "memory_graph_reconstruction.mp4",
    title: str = "Habitat + PixelNav Memory Graph",
    fps: int = 2,
    hold_frames: int = 4,
) -> dict[str, Any]:
    graph_path = Path(graph_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = max(1, int(fps))
    hold_frames = max(0, int(hold_frames))

    output_video_path = Path(output_video)
    if not output_video_path.is_absolute():
        output_video_path = output_dir / output_video_path
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    graph = load_memory_graph(graph_path)
    final_layout = reconstruct_memory_graph_layout(graph)
    frames_dir = output_dir / "memory_graph_video_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("frame_*.png"):
        old_frame.unlink()

    frame_graphs = _build_reveal_graphs(graph)
    if frame_graphs:
        frame_graphs.extend([frame_graphs[-1]] * hold_frames)

    frame_paths: list[Path] = []
    total_frames = len(frame_graphs)
    for index, frame_graph in enumerate(frame_graphs):
        layout = reconstruct_memory_graph_layout(frame_graph)
        frame_path = frames_dir / f"frame_{index:04d}.png"
        frame_title = f"{title} | frame {index + 1}/{total_frames}"
        _render_png(layout, frame_path, title=frame_title, graph_path=graph_path)
        frame_paths.append(frame_path)

    encoder, video_format, actual_video_path = _encode_video_frames(frame_paths, output_video_path, fps=fps)
    summary_path = output_dir / "memory_graph_video_summary.json"
    summary = {
        **final_layout.to_summary(),
        "graph_path": str(graph_path),
        "video_path": str(actual_video_path),
        "video_format": video_format,
        "encoder": encoder,
        "fps": fps,
        "frame_count": len(frame_paths),
        "frames_dir": str(frames_dir),
        "updated_at_unix_s": round(time.time(), 3),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "video_path": str(actual_video_path),
        "summary_json": str(summary_path),
        "video_format": video_format,
        "encoder": encoder,
        "fps": fps,
        "frame_count": len(frame_paths),
        "node_count": final_layout.node_count,
        "edge_count": final_layout.edge_count,
        "current_node_id": final_layout.current_node_id,
    }


def serve_memory_graph_visualizer(
    graph_path: str | Path,
    *,
    output_dir: str | Path,
    title: str,
    host: str,
    port: int,
    watch_interval_s: float,
) -> None:
    from flask import Flask, Response, send_file

    graph_path = Path(graph_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    app = Flask(__name__)
    state: dict[str, Any] = {"last_error": None, "last_result": None}

    def refresh() -> dict[str, Any]:
        try:
            state["last_result"] = render_memory_graph_snapshot(graph_path, output_dir=output_dir, title=title)
            state["last_error"] = None
        except Exception as exc:  # pragma: no cover - server resilience path
            state["last_error"] = repr(exc)
        return dict(state)

    @app.get("/")
    def index() -> Response:
        state_snapshot = refresh()
        error = state_snapshot.get("last_error")
        result = state_snapshot.get("last_result") or {}
        body = _render_live_html(
            title=title,
            graph_path=str(graph_path),
            png_name="memory_graph.png",
            summary_name="memory_graph_visualizer_summary.json",
            refresh_ms=max(500, int(watch_interval_s * 1000)),
            error=error,
            result=result,
        )
        return Response(body, mimetype="text/html")

    @app.get("/memory_graph.png")
    def image() -> Any:
        refresh()
        return send_file(output_dir / "memory_graph.png", mimetype="image/png")

    @app.get("/summary.json")
    def summary() -> Any:
        refresh()
        return send_file(output_dir / "memory_graph_visualizer_summary.json", mimetype="application/json")

    @app.get("/memory_graph_visualizer_summary.json")
    def summary_alias() -> Any:
        refresh()
        return send_file(output_dir / "memory_graph_visualizer_summary.json", mimetype="application/json")

    refresh()
    app.run(host=host, port=int(port), debug=False, use_reloader=False)


def _build_reveal_graphs(graph: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = _normalize_items(graph.get("nodes"), id_key="node_id")
    edges = _normalize_items(graph.get("edges"), id_key="edge_id")
    ordered_nodes = _ordered_node_ids(nodes, graph.get("current_node_id"))
    if not ordered_nodes:
        return [dict(graph, nodes={}, edges={})]

    visible_nodes: set[str] = {ordered_nodes[0]}
    visible_edges: dict[str, dict[str, Any]] = {}
    frames = [_subset_graph(graph, nodes, edges, visible_nodes, visible_edges)]

    for edge_id, edge in sorted(edges.items()):
        src = edge.get("src_node_id")
        dst = edge.get("dst_node_id")
        if src:
            visible_nodes.add(str(src))
        if dst:
            visible_nodes.add(str(dst))
        visible_edges[edge_id] = edge
        frames.append(_subset_graph(graph, nodes, edges, visible_nodes, visible_edges))

    for node_id in ordered_nodes:
        if node_id in visible_nodes:
            continue
        visible_nodes.add(node_id)
        frames.append(_subset_graph(graph, nodes, edges, visible_nodes, visible_edges))

    return frames


def _subset_graph(
    graph: dict[str, Any],
    all_nodes: dict[str, dict[str, Any]],
    all_edges: dict[str, dict[str, Any]],
    visible_node_ids: set[str],
    visible_edges: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    nodes = {node_id: all_nodes[node_id] for node_id in sorted(visible_node_ids) if node_id in all_nodes}
    edges = {
        edge_id: edge
        for edge_id, edge in sorted(visible_edges.items())
        if str(edge.get("src_node_id")) in nodes and str(edge.get("dst_node_id")) in nodes and edge_id in all_edges
    }
    frame_graph = dict(graph)
    frame_graph["nodes"] = nodes
    frame_graph["edges"] = edges
    return frame_graph


def _encode_video_frames(frame_paths: list[Path], output_video_path: Path, *, fps: int) -> tuple[str, str, Path]:
    if not frame_paths:
        raise ValueError("cannot render memory graph video without frames")

    suffix = output_video_path.suffix.lower()
    if suffix == ".gif":
        _write_gif(frame_paths, output_video_path, fps=fps)
        return "pillow", "gif", output_video_path

    if suffix == ".mp4":
        ffmpeg_path = _find_ffmpeg()
        if ffmpeg_path:
            frame_pattern = str(frame_paths[0].parent / "frame_%04d.png")
            command = [
                ffmpeg_path,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                frame_pattern,
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
                str(output_video_path),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            return "ffmpeg", "mp4", output_video_path

    fallback_path = output_video_path.with_suffix(".gif")
    _write_gif(frame_paths, fallback_path, fps=fps)
    return "pillow_fallback", "gif", fallback_path


def _find_ffmpeg() -> str | None:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _write_gif(frame_paths: list[Path], output_path: Path, *, fps: int) -> None:
    from PIL import Image

    images = [Image.open(frame_path).convert("P", palette=Image.Palette.ADAPTIVE) for frame_path in frame_paths]
    try:
        first, rest = images[0], images[1:]
        first.save(
            output_path,
            save_all=True,
            append_images=rest,
            duration=max(1, int(1000 / max(1, fps))),
            loop=0,
            optimize=False,
        )
    finally:
        for image in images:
            image.close()


def _normalize_items(value: Any, *, id_key: str) -> dict[str, dict[str, Any]]:
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, list):
        items = ((str(item.get(id_key, index)), item) for index, item in enumerate(value) if isinstance(item, dict))
    else:
        items = []

    normalized: dict[str, dict[str, Any]] = {}
    for fallback_id, payload in items:
        if not isinstance(payload, dict):
            continue
        item_id = str(payload.get(id_key) or fallback_id)
        normalized[item_id] = dict(payload)
        normalized[item_id][id_key] = item_id
    return normalized


def _ordered_node_ids(nodes: dict[str, dict[str, Any]], current_node_id: Any) -> list[str]:
    return sorted(nodes.keys())


def _build_adjacency(edges: dict[str, dict[str, Any]]) -> dict[str, list[tuple[str, str, MemoryGraphPose, bool]]]:
    adjacency: dict[str, list[tuple[str, str, MemoryGraphPose, bool]]] = {}
    for edge_id, edge in edges.items():
        src = edge.get("src_node_id")
        dst = edge.get("dst_node_id")
        if not src or not dst:
            continue
        rel = _relative_pose(edge)
        adjacency.setdefault(str(src), []).append((edge_id, str(dst), rel, True))
        adjacency.setdefault(str(dst), []).append((edge_id, str(src), rel, False))
    return adjacency


def _relative_pose(edge: dict[str, Any]) -> MemoryGraphPose:
    pose = edge.get("relative_pose_src_to_dst") or {}
    return MemoryGraphPose(
        x_m=float(pose.get("dx_m", 0.0) or 0.0),
        y_m=float(pose.get("dy_m", 0.0) or 0.0),
        yaw_deg=float(pose.get("dyaw_deg", 0.0) or 0.0),
        component_index=0,
    )


def _compose(src: MemoryGraphPose, rel: MemoryGraphPose) -> MemoryGraphPose:
    theta = math.radians(src.yaw_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    return MemoryGraphPose(
        x_m=src.x_m + c * rel.x_m - s * rel.y_m,
        y_m=src.y_m + s * rel.x_m + c * rel.y_m,
        yaw_deg=_normalize_deg(src.yaw_deg + rel.yaw_deg),
        component_index=src.component_index,
    )


def _inverse(rel: MemoryGraphPose) -> MemoryGraphPose:
    theta = math.radians(rel.yaw_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    return MemoryGraphPose(
        x_m=-(c * rel.x_m + s * rel.y_m),
        y_m=-(-s * rel.x_m + c * rel.y_m),
        yaw_deg=_normalize_deg(-rel.yaw_deg),
        component_index=rel.component_index,
    )


def _normalize_deg(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


def _render_png(layout: MemoryGraphLayout, png_path: Path, *, title: str, graph_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7), dpi=140)
    ax.set_title(title)
    ax.set_xlabel("reconstructed x (m)")
    ax.set_ylabel("reconstructed y (m)")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_aspect("equal", adjustable="datalim")

    for edge_id, edge in sorted(layout.edges.items()):
        src = edge.get("src_node_id")
        dst = edge.get("dst_node_id")
        if src not in layout.poses or dst not in layout.poses:
            continue
        src_pose = layout.poses[str(src)]
        dst_pose = layout.poses[str(dst)]
        color, linestyle, linewidth = _edge_style(edge)
        ax.annotate(
            "",
            xy=(dst_pose.x_m, dst_pose.y_m),
            xytext=(src_pose.x_m, src_pose.y_m),
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": linewidth,
                "linestyle": linestyle,
                "shrinkA": 13,
                "shrinkB": 13,
            },
        )
        mx = (src_pose.x_m + dst_pose.x_m) / 2.0
        my = (src_pose.y_m + dst_pose.y_m) / 2.0
        ax.text(mx, my, edge_id, fontsize=7, color=color, ha="center", va="bottom")

    for node_id, pose in sorted(layout.poses.items()):
        node = layout.nodes.get(node_id, {})
        is_current = node_id == layout.current_node_id
        is_failed_frontier = str(node.get("node_type") or "") == "failed_frontier"
        color = "#dc2626" if is_failed_frontier else ("#ef4444" if is_current else "#2563eb")
        size = 180 if is_current else (150 if is_failed_frontier else 110)
        marker = "X" if is_failed_frontier else "o"
        ax.scatter(
            [pose.x_m],
            [pose.y_m],
            s=size,
            c=color,
            marker=marker,
            edgecolors="#111827",
            linewidths=0.8,
            zorder=3,
        )
        dx = 0.25 * math.cos(math.radians(pose.yaw_deg))
        dy = 0.25 * math.sin(math.radians(pose.yaw_deg))
        ax.arrow(pose.x_m, pose.y_m, dx, dy, color="#111827", width=0.01, head_width=0.08, zorder=4)
        label = f"{node_id}"
        category = node.get("place_category")
        if category:
            label += f"\n{category}"
        if is_failed_frontier:
            label += "\nFAILED FRONTIER"
        if is_current:
            label += "\nCURRENT"
        ax.text(pose.x_m, pose.y_m + 0.12, label, fontsize=8, ha="center", va="bottom")

    if not layout.poses:
        ax.text(0.5, 0.5, "No nodes in memory graph", transform=ax.transAxes, ha="center", va="center")

    footer = f"{layout.schema_version or 'unknown schema'} | nodes={layout.node_count} edges={layout.edge_count} | {graph_path}"
    fig.text(0.01, 0.01, footer, fontsize=7, color="#4b5563")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(png_path)
    plt.close(fig)


def _edge_style(edge: dict[str, Any]) -> tuple[str, str, float]:
    edge_type = str(edge.get("edge_type") or "")
    relation_type = str(edge.get("relation_type") or "")
    traversal = edge.get("traversal") if isinstance(edge.get("traversal"), dict) else {}
    status = str(traversal.get("status") or "")
    if status in {"deadlock_entry", "deadlock_entry_candidate", "blocked", "risky"}:
        return "#dc2626", "--", 2.0
    if edge_type in {"same_place", "same_place_constraint"} or relation_type == "same_place":
        return "#7c3aed", ":", 2.0
    if status == "success":
        return "#059669", "-", 1.8
    return "#6b7280", "-", 1.2


def _render_html(*, title: str, summary: dict[str, Any], refresh_ms: int) -> str:
    return _render_live_html(
        title=title,
        graph_path=str(summary.get("graph_path")),
        png_name="memory_graph.png",
        summary_name="memory_graph_visualizer_summary.json",
        refresh_ms=refresh_ms,
        error=None,
        result=summary,
    )


def _render_live_html(
    *,
    title: str,
    graph_path: str,
    png_name: str,
    summary_name: str,
    refresh_ms: int,
    error: str | None,
    result: dict[str, Any],
) -> str:
    error_html = f"<p class='error'>{_escape(error)}</p>" if error else ""
    node_count = result.get("node_count", result.get("node_count", "unknown"))
    edge_count = result.get("edge_count", result.get("edge_count", "unknown"))
    current = result.get("current_node_id", "unknown")
    stamp = result.get("updated_at_unix_s", time.time())
    image_src = f"{png_name}?t={int(time.time() * 1000)}"
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="{max(1, int(refresh_ms / 1000))}">
  <title>{_escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 20px; color: #111827; }}
    .meta {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0; color: #374151; }}
    .pill {{ border: 1px solid #d1d5db; padding: 4px 8px; border-radius: 6px; background: #f9fafb; }}
    img {{ max-width: 100%; border: 1px solid #d1d5db; border-radius: 6px; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
    .error {{ color: #b91c1c; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>{_escape(title)}</h1>
  {error_html}
  <div class="meta">
    <span class="pill">nodes: {node_count}</span>
    <span class="pill">edges: {edge_count}</span>
    <span class="pill">current: {current}</span>
    <span class="pill">updated: {stamp}</span>
  </div>
  <p>Source graph: <code>{_escape(graph_path)}</code></p>
  <p><a href="{_escape(summary_name)}">summary json</a></p>
  <img src="{_escape(image_src)}" alt="memory graph reconstruction">
</body>
</html>
"""


def _escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    raise SystemExit(main())
