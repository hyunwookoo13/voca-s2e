#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from navigation_stress_suite import (
    deterministic_guard_cases,
    evaluate_revisit_pairs,
    summarize_stop_cases,
    write_stress_outputs,
)


MISMATCH_TARGET = {
    "chair": "toilet",
    "bed": "plant",
    "plant": "tv_monitor",
    "toilet": "chair",
    "sofa": "toilet",
    "tv_monitor": "bed",
    "tv": "bed",
}


def _write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR))


def generate_hm3d_stop_cases(output_dir: Path, count: int) -> List[Dict[str, Any]]:
    import habitat
    from habitat_sim.utils.common import quat_from_coeffs

    from habitat_config import hm3d_config

    env = habitat.Env(hm3d_config(episodes=max(count * 3, count)))
    cases: List[Dict[str, Any]] = []
    accepted = 0
    attempts = 0
    try:
        while accepted < count and attempts < max(4, count * 3):
            attempts += 1
            observation = env.reset()
            episode = env.current_episode
            target = str(episode.object_category)
            metrics = env.get_metrics()
            try:
                start_distance = float(metrics.get("distance_to_goal"))
            except Exception:
                start_distance = 0.0
            goals = list(getattr(episode, "goals", []) or [])
            viewpoints = [
                viewpoint
                for goal in goals
                for viewpoint in list(getattr(goal, "view_points", []) or [])
            ]
            if not viewpoints or start_distance < 2.0:
                continue
            case_dir = output_dir / "stop_cases" / "episode_{:03d}".format(accepted)
            negative_path = case_dir / "negative_far_start.jpg"
            _write_rgb(negative_path, observation["rgb"])
            cases.append(
                {
                    "suite": "stop_verifier",
                    "case": "far_start",
                    "episode": accepted,
                    "episode_id": str(episode.episode_id),
                    "scene_id": str(episode.scene_id),
                    "target": target,
                    "image_paths": [str(negative_path)],
                    "expected_pass": False,
                    "label_source": "HM3D start distance {:.3f}m > 2.0m".format(start_distance),
                }
            )

            viewpoint = max(viewpoints, key=lambda item: float(getattr(item, "iou", 0.0)))
            state = viewpoint.agent_state
            env.sim.set_agent_state(
                state.position,
                quat_from_coeffs(np.asarray(state.rotation, dtype=np.float32)),
            )
            positive_observation = env.sim.get_sensor_observations()
            positive_path = case_dir / "positive_goal_viewpoint.jpg"
            _write_rgb(positive_path, positive_observation["rgb"])
            cases.append(
                {
                    "suite": "stop_verifier",
                    "case": "official_goal_viewpoint",
                    "episode": accepted,
                    "episode_id": str(episode.episode_id),
                    "scene_id": str(episode.scene_id),
                    "target": target,
                    "image_paths": [str(positive_path)],
                    "expected_pass": True,
                    "label_source": "official HM3D ObjectNav goal viewpoint iou {:.4f}".format(
                        float(getattr(viewpoint, "iou", 0.0))
                    ),
                }
            )
            cases.append(
                {
                    "suite": "stop_verifier",
                    "case": "wrong_category_lookalike",
                    "episode": accepted,
                    "episode_id": str(episode.episode_id),
                    "scene_id": str(episode.scene_id),
                    "target": MISMATCH_TARGET.get(target, "toilet" if target != "toilet" else "chair"),
                    "image_paths": [str(positive_path)],
                    "expected_pass": False,
                    "label_source": "official viewpoint image queried with a different target category",
                }
            )
            accepted += 1
    finally:
        env.close()
    return cases


def run_online_verifiers(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    from qwen_vlm_planner import QwenVLMPlanner

    planner = QwenVLMPlanner()
    if planner.stop_verifier is None:
        raise RuntimeError("VOCA_QWEN_STOP_VERIFY must be enabled")
    for case in cases:
        result = planner.stop_verifier.verify(
            target=case["target"],
            image_paths=case["image_paths"],
        )
        case["actual_pass"] = bool(result.get("passed"))
        case["verifier_result"] = result

    revisit_cases: List[Dict[str, Any]] = []
    positives = [case for case in cases if case["case"] == "official_goal_viewpoint"]
    if planner.revisit_verifier is not None and positives:
        current = positives[0]["image_paths"][0]
        comparisons = [
            ("same_image", current, True, "confirm_revisit_node"),
            ("spatial_gate_rejects_visual_match", current, False, "not_confirm"),
        ]
        if len(positives) > 1:
            comparisons.append(
                ("different_episode", positives[1]["image_paths"][0], True, "not_confirm")
            )
        for index, (name, memory_image, spatial_accepted, expectation) in enumerate(comparisons):
            candidate_ref = "revisit_stress_{:03d}".format(index)
            vlm_input = {
                "metadata": {"raw_observation_images": [current]},
                "memory": {
                    "place_recognition": {
                        "revisit_candidates": [
                            {
                                "candidate_ref": candidate_ref,
                                "candidate_image_ref": memory_image,
                                "visual_retrieval_score": 1.0 if current == memory_image else 0.5,
                                "spatial_plausibility": {
                                    "accepted": spatial_accepted,
                                    "reason": "stress_suite_controlled_spatial_gate",
                                },
                            }
                        ]
                    }
                },
            }
            result = planner.revisit_verifier.verify(
                vlm_input=vlm_input,
                candidate_ref=candidate_ref,
            )
            operation = result.get("memory_op", {}).get("op")
            passed = (
                operation == expectation
                if expectation != "not_confirm"
                else operation != "confirm_revisit_node"
            )
            revisit_cases.append(
                {
                    "suite": "revisit_verifier",
                    "case": name,
                    "expected_operation": expectation,
                    "actual_operation": operation,
                    "passed": passed,
                    "verifier_result": result,
                }
            )
    return {
        "planner": planner,
        "revisit_cases": revisit_cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VOCA pre-benchmark navigation stress tests.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--revisit-pairs",
        default=str(
            PROJECT_ROOT
            / "reports"
            / "reproducibility_20260712"
            / "place_recognition"
            / "place_recognition_pairs.csv"
        ),
    )
    parser.add_argument("--revisit-threshold", type=float, default=0.84)
    parser.add_argument("--stop-episodes", type=int, default=3)
    parser.add_argument("--reuse-cases-jsonl", default="")
    parser.add_argument("--offline-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    started = time.perf_counter()
    summaries: List[Dict[str, Any]] = []
    all_cases = deterministic_guard_cases()
    guard_summary = {
        "suite": "deterministic_guards",
        "samples": len(all_cases),
        "passed_cases": sum(bool(case.get("passed")) for case in all_cases),
        "passed": all(bool(case.get("passed")) for case in all_cases),
        "required": True,
    }
    summaries.append(guard_summary)

    revisit_path = Path(args.revisit_pairs)
    if revisit_path.exists():
        revisit_summary = evaluate_revisit_pairs(revisit_path, args.revisit_threshold)
        revisit_summary["required"] = True
        summaries.append(revisit_summary)

    model = os.environ.get("QWEN_MODEL", "")
    base_url = os.environ.get("QWEN_BASE_URL", "")
    if not args.offline_only and args.stop_episodes > 0:
        if args.reuse_cases_jsonl:
            stop_cases = []
            for line in Path(args.reuse_cases_jsonl).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                case = json.loads(line)
                if case.get("suite") != "stop_verifier":
                    continue
                case.pop("actual_pass", None)
                case.pop("verifier_result", None)
                stop_cases.append(case)
        else:
            stop_cases = generate_hm3d_stop_cases(output_dir, args.stop_episodes)
        online = run_online_verifiers(stop_cases)
        all_cases.extend(stop_cases)
        all_cases.extend(online["revisit_cases"])
        stop_summary = summarize_stop_cases(stop_cases)
        stop_summary["required"] = True
        summaries.append(stop_summary)
        revisit_verifier_summary = {
            "suite": "revisit_verifier",
            "samples": len(online["revisit_cases"]),
            "passed_cases": sum(
                bool(case.get("passed")) for case in online["revisit_cases"]
            ),
            "passed": bool(online["revisit_cases"])
            and all(bool(case.get("passed")) for case in online["revisit_cases"]),
            "required": True,
        }
        summaries.append(revisit_verifier_summary)

    paths = write_stress_outputs(
        output_dir,
        summaries=summaries,
        cases=all_cases,
        metadata={
            "model": model,
            "base_url": base_url,
            "duration_sec": round(time.perf_counter() - started, 6),
            "offline_only": bool(args.offline_only),
            "stop_episodes_requested": int(args.stop_episodes),
        },
    )
    print(json.dumps({"summaries": summaries, "paths": paths}, indent=2))
    return 0 if all(bool(item.get("passed")) for item in summaries if item.get("required")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
