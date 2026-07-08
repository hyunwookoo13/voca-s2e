from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from goal_adapter.habitat_goal_audit import (
    HabitatGoalAuditConfig,
    Image,
    run_habitat_goal_audit,
)
from goal_adapter.habitat_smoke import write_rgb_png
from goal_adapter.visualization import _read_png_rgb


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run random Habitat VLM goal audits and create a contact sheet.")
    parser.add_argument("--scene-root", required=True, help="Directory containing Habitat .glb/.basis.glb scenes.")
    parser.add_argument("--output-dir", required=True, help="Directory where batch audit outputs are written.")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", default="qwen3-vl-30b")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=79.0)
    parser.add_argument("--sensor-height", type=float, default=0.88)
    args = parser.parse_args(argv)

    scene_paths = _discover_scene_paths(Path(args.scene_root))
    if not scene_paths:
        raise RuntimeError(f"no Habitat scenes found under {args.scene_root}")

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_scenes = [rng.choice(scene_paths) for _ in range(args.count)]

    records = []
    layout_paths = []
    for index, scene_path in enumerate(selected_scenes):
        trial_dir = output_dir / f"trial_{index:03d}_{scene_path.stem}"
        heading = rng.uniform(-3.141592653589793, 3.141592653589793)
        config = HabitatGoalAuditConfig(
            scene_path=scene_path,
            output_dir=trial_dir,
            heading=heading,
            image_width=args.width,
            image_height=args.height,
            sensor_height=args.sensor_height,
            image_hfov=args.hfov,
            endpoint=args.endpoint,
            model=args.model,
        )
        print(f"[{index + 1}/{args.count}] scene={scene_path} heading={heading:.3f}", flush=True)
        result = run_habitat_goal_audit(config)
        records.append(
            {
                "index": index,
                "scene_path": str(scene_path),
                "heading": heading,
                "result": {
                    **asdict(result),
                    "output_dir": str(result.output_dir),
                    "current_rgb_path": str(result.current_rgb_path),
                    "rgb_overlay_path": str(result.rgb_overlay_path),
                    "topdown_path": str(result.topdown_path),
                    "audit_path": str(result.audit_path),
                    "result_json_path": str(result.result_json_path),
                },
            }
        )
        layout_paths.append(result.audit_path)

    contact_sheet_path = output_dir / "contact_sheet.png"
    write_rgb_png(_contact_sheet([_read_png_rgb(path) for path in layout_paths], columns=2), contact_sheet_path)
    summary_path = output_dir / "batch_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "scene_root": str(args.scene_root),
                "available_scene_count": len(scene_paths),
                "requested_count": args.count,
                "seed": args.seed,
                "contact_sheet": str(contact_sheet_path),
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote contact_sheet={contact_sheet_path}")
    print(f"wrote summary={summary_path}")
    return 0


def _discover_scene_paths(scene_root: Path) -> list[Path]:
    scene_paths = []
    for root, _dirs, filenames in os.walk(scene_root, followlinks=True):
        for filename in filenames:
            if filename.endswith(".glb") or filename.endswith(".basis.glb"):
                scene_paths.append(Path(root) / filename)
    return sorted(scene_paths)


def _contact_sheet(images: list[Image], columns: int) -> Image:
    if not images:
        return [[[255, 255, 255]]]
    cell_height = len(images[0])
    cell_width = len(images[0][0])
    rows = (len(images) + columns - 1) // columns
    sheet: Image = [
        [[255, 255, 255] for _ in range(cell_width * columns)]
        for _ in range(cell_height * rows)
    ]
    for index, image in enumerate(images):
        row_index = index // columns
        column_index = index % columns
        for y, image_row in enumerate(image):
            target_y = (row_index * cell_height) + y
            target_x = column_index * cell_width
            sheet[target_y][target_x : target_x + cell_width] = [list(pixel[:3]) for pixel in image_row]
    return sheet


if __name__ == "__main__":
    raise SystemExit(main())
