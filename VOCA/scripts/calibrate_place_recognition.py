#!/usr/bin/env python
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from place_recognition import build_place_embedder
from voca_s2e_bridge import import_nav_memory_qwen


def _position(record: Dict[str, Any]) -> List[float]:
    position = (
        record.get("vlm_input", {})
        .get("memory", {})
        .get("voca_sidecar", {})
        .get("position_xyz", [])
    )
    if not isinstance(position, list) or len(position) < 3:
        return []
    try:
        return [float(position[0]), float(position[1]), float(position[2])]
    except Exception:
        return []


def _image_paths(record: Dict[str, Any]) -> List[str]:
    vlm_input = record.get("vlm_input") if isinstance(record.get("vlm_input"), dict) else {}
    metadata = vlm_input.get("metadata") if isinstance(vlm_input.get("metadata"), dict) else {}
    paths = metadata.get("raw_observation_images")
    if not isinstance(paths, list) or not paths:
        observation = vlm_input.get("observation") if isinstance(vlm_input.get("observation"), dict) else {}
        paths = [view.get("image") for view in observation.get("views", []) if isinstance(view, dict)]
    return [str(path) for path in paths if path and Path(path).exists()]


def load_trajectory(path: Path) -> List[Dict[str, Any]]:
    observations = []
    for line_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        position = _position(record)
        images = _image_paths(record)
        if not position or not images:
            continue
        observations.append(
            {
                "source": str(path),
                "line_index": int(line_index),
                "frame_index": int(
                    record.get("vlm_input", {}).get("observation", {}).get("frame_index", line_index)
                    or line_index
                ),
                "position_xyz": position,
                "image_paths": images,
            }
        )
    return observations


def aggregate_descriptor(embedder: Any, image_paths: Sequence[str]) -> np.ndarray:
    descriptors = [np.asarray(embedder.embed_image(path), dtype=np.float32) for path in image_paths]
    descriptor = np.mean(np.stack(descriptors, axis=0), axis=0).astype(np.float32)
    norm = float(np.linalg.norm(descriptor))
    return descriptor if norm <= 1e-8 else descriptor / norm


def image_descriptors(embedder: Any, image_paths: Sequence[str]) -> List[np.ndarray]:
    return [
        np.asarray(embedder.embed_image(path), dtype=np.float32)
        for path in image_paths
    ]


def build_pairs(
    trajectories: Iterable[List[Dict[str, Any]]],
    *,
    embedder: Any,
    same_max_m: float,
    different_min_m: float,
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for observations in trajectories:
        for observation in observations:
            observation["descriptors"] = image_descriptors(
                embedder, observation["image_paths"]
            )
        for index, first in enumerate(observations):
            for second in observations[index + 1 :]:
                distance = math.sqrt(
                    sum(
                        (float(first["position_xyz"][axis]) - float(second["position_xyz"][axis])) ** 2
                        for axis in range(3)
                    )
                )
                if distance <= float(same_max_m):
                    label = "same_place"
                elif distance >= float(different_min_m):
                    label = "different_place"
                else:
                    continue
                similarity = max(
                    float(np.dot(first_descriptor, second_descriptor))
                    for first_descriptor in first["descriptors"]
                    for second_descriptor in second["descriptors"]
                )
                pairs.append(
                    {
                        "source": first["source"],
                        "first_line": first["line_index"],
                        "second_line": second["line_index"],
                        "first_frame": first["frame_index"],
                        "second_frame": second["frame_index"],
                        "distance_m": round(distance, 6),
                        "similarity": round(similarity, 6),
                        "label": label,
                        "first_view_count": len(first["image_paths"]),
                        "second_view_count": len(second["image_paths"]),
                    }
                )
    return pairs


def threshold_metrics(pairs: Sequence[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    for pair in pairs:
        positive = pair["label"] == "same_place"
        predicted = float(pair["similarity"]) >= float(threshold)
        tp += int(positive and predicted)
        fp += int(not positive and predicted)
        tn += int(not positive and not predicted)
        fn += int(positive and not predicted)
    precision = tp / float(tp + fp) if tp + fp else 1.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_positive_rate = fp / float(fp + tn) if fp + tn else 0.0
    return {
        "threshold": round(float(threshold), 6),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "false_positive_rate": round(false_positive_rate, 6),
    }


def select_threshold(
    pairs: Sequence[Dict[str, Any]],
    *,
    target_precision: float = 0.98,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not pairs:
        raise ValueError("no labeled place-recognition pairs")
    scores = sorted({float(pair["similarity"]) for pair in pairs})
    candidates = sorted(set([0.0, 1.000001] + scores + [min(1.0, score + 1e-6) for score in scores]))
    table = [threshold_metrics(pairs, threshold) for threshold in candidates]
    eligible = [
        row
        for row in table
        if row["precision"] >= float(target_precision) and row["tp"] > 0
    ]
    if eligible:
        selected = max(eligible, key=lambda row: (row["recall"], -row["false_positive_rate"], row["threshold"]))
        policy = "precision_constrained_max_recall"
    else:
        selected = max(table, key=lambda row: (row["f1"], row["precision"], row["threshold"]))
        policy = "fallback_max_f1"
    selected = dict(selected)
    selected["selection_policy"] = policy
    selected["target_precision"] = float(target_precision)
    return selected, table


def _distribution(pairs: Sequence[Dict[str, Any]], label: str) -> Dict[str, Any]:
    values = [float(pair["similarity"]) for pair in pairs if pair["label"] == label]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "mean": round(float(np.mean(values)), 6),
        "median": round(float(np.median(values)), 6),
        "max": round(max(values), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate VOCA DINOv2 place-recognition threshold.")
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--same-max-m", type=float, default=0.75)
    parser.add_argument("--different-min-m", type=float, default=2.0)
    parser.add_argument("--target-precision", type=float, default=0.98)
    parser.add_argument("--max-trajectories", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(Path(args.root).glob("**/qwen_calls.jsonl"))
    trajectories = [load_trajectory(path) for path in paths]
    trajectories = [trajectory for trajectory in trajectories if len(trajectory) >= 2]
    if args.max_trajectories > 0:
        trajectories = trajectories[: args.max_trajectories]
    embedder = build_place_embedder(import_nav_memory_qwen())
    pairs = build_pairs(
        trajectories,
        embedder=embedder,
        same_max_m=args.same_max_m,
        different_min_m=args.different_min_m,
    )
    selected, table = select_threshold(pairs, target_precision=args.target_precision)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "place_recognition_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0].keys()))
        writer.writeheader()
        writer.writerows(pairs)
    with (output / "place_recognition_thresholds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    status_fn = getattr(embedder, "status", None)
    summary = {
        "schema_version": "voca_place_recognition_calibration_v1",
        "trajectory_count": len(trajectories),
        "observation_count": sum(len(items) for items in trajectories),
        "pair_count": len(pairs),
        "same_max_m": float(args.same_max_m),
        "different_min_m": float(args.different_min_m),
        "same_place_distribution": _distribution(pairs, "same_place"),
        "different_place_distribution": _distribution(pairs, "different_place"),
        "selected": selected,
        "embedder": status_fn() if callable(status_fn) else {"backend": type(embedder).__name__},
    }
    (output / "place_recognition_calibration.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
