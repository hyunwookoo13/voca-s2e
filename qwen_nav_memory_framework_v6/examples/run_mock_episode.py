"""Run a local smoke-test episode without a real VLM or robot.

Usage:
    python examples/run_mock_episode.py
"""

from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from nav_memory_qwen import NavMemoryAgent, NavAgentConfig, StaticImageBackend, HeuristicVLMClient


def main() -> None:
    out_dir = Path("runs/mock_episode")
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = StaticImageBackend.create_demo_image(out_dir / "demo_view.jpg")

    # Block the left sector to demonstrate negative-memory/deadlock handling.
    robot = StaticImageBackend(
        image_path,
        start_xy=(0.0, 0.0),
        start_heading_rad=0.0,
        step_m=0.65,
        blocked_bearing_sectors=[(-110, -70)],
    )
    agent = NavMemoryAgent(
        robot=robot,
        vlm_client=HeuristicVLMClient(),
        config=NavAgentConfig(max_steps=30, force_new_node_translation_m=0.5, log_full_vlm_input=False),
    )

    # Goal is straight ahead in the simulated map.
    result = agent.run_until_done(goal_map_xy=(4.0, 0.0), max_steps=30)
    agent.save_run(out_dir)

    print("Episode summary:")
    print(result.summary())
    print(f"Saved logs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
