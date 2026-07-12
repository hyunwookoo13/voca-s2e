#!/usr/bin/env python
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _scene_key(value):
    return Path(str(value or "")).stem.replace(".basis", "")


def _finite_distance(sim, start, goal):
    try:
        distance = float(sim.geodesic_distance(start, goal))
    except Exception:
        return None
    return distance if math.isfinite(distance) else None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Derive a noisy HM3D CoarseGoalNav task file. This is a separate "
            "task protocol and must not be reported as category-only ObjectNav."
        )
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--pool-size", type=int, default=0)
    parser.add_argument("--stage", default="val")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--goal-noise-std-m", type=float, default=0.5)
    parser.add_argument("--source", default="upstream_global_planner")
    args = parser.parse_args()

    import habitat
    from habitat_config import hm3d_config

    episode_count = max(1, int(args.episodes))
    start_index = max(0, int(args.start_index))
    pool_size = max(
        start_index + episode_count,
        int(args.pool_size or 0),
    )
    config = hm3d_config(
        stage=str(args.stage),
        episodes=pool_size,
        seed=int(args.seed),
    )
    env = habitat.Env(config)
    records = []
    try:
        for _ in range(start_index):
            env.reset()
        for episode_index in range(start_index, start_index + episode_count):
            env.reset()
            episode = env.current_episode
            start_position = list(env.sim.get_agent_state().position)
            ranked = []
            for goal_index, goal in enumerate(episode.goals):
                goal_position = [float(value) for value in goal.position]
                distance = _finite_distance(env.sim, start_position, goal_position)
                if distance is not None:
                    ranked.append((distance, goal_index, goal_position))
            if not ranked:
                raise RuntimeError(
                    "episode {} has no reachable coarse goal viewpoint".format(
                        episode_index
                    )
                )
            distance, goal_index, goal_position = min(ranked)
            rng = np.random.default_rng(int(args.seed) + int(episode_index))
            noise = rng.normal(
                0.0,
                max(0.0, float(args.goal_noise_std_m)),
                size=2,
            )
            map_xy = [
                float(goal_position[0] + noise[0]),
                float(goal_position[2] + noise[1]),
            ]
            records.append(
                {
                    "episode_index": int(episode_index),
                    "episode_id": str(episode.episode_id),
                    "scene_id": _scene_key(episode.scene_id),
                    "object_goal": str(episode.object_category),
                    "source": str(args.source),
                    "type": "map_waypoint",
                    "map_xy": [round(value, 6) for value in map_xy],
                    "uncertainty": "medium",
                    "distance_uncertainty_m": max(
                        1.0,
                        2.0 * max(0.0, float(args.goal_noise_std_m)),
                    ),
                    "task_protocol": "hm3d_derived_coarse_goalnav_v1",
                    "derivation": "closest_reachable_episode_goal_viewpoint_from_start",
                    "goal_viewpoint_index": int(goal_index),
                    "derivation_start_geodesic_m": round(float(distance), 6),
                    "goal_noise_std_m": float(args.goal_noise_std_m),
                    "reporting_constraint": (
                        "derived CoarseGoalNav only; do not report as category-only ObjectNav"
                    ),
                }
            )
    finally:
        env.close()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "voca_coarse_goal_dataset_v1",
        "task_protocol": "hm3d_derived_coarse_goalnav_v1",
        "stage": str(args.stage),
        "seed": int(args.seed),
        "goal_noise_std_m": float(args.goal_noise_std_m),
        "records": records,
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(str(output))
    print("records={}".format(len(records)))


if __name__ == "__main__":
    main()
