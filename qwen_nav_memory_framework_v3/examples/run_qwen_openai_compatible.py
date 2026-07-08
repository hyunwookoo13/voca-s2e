"""Run the framework with an OpenAI-compatible Qwen3-VL endpoint.

Set environment variables first:
    export QWEN_BASE_URL="https://your-qwen-endpoint/v1"
    export QWEN_API_KEY="..."
    export QWEN_MODEL="qwen3-vl-32b-thinking"

Then run:
    python examples/run_qwen_openai_compatible.py --image path/to/front.jpg --goal-x 4.0 --goal-y 0.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from nav_memory_qwen import NavMemoryAgent, NavAgentConfig, StaticImageBackend, OpenAICompatibleVLMClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Initial/front RGB image path for the demo backend")
    parser.add_argument("--goal-x", type=float, required=True)
    parser.add_argument("--goal-y", type=float, required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--out", default="runs/qwen_episode")
    args = parser.parse_args()

    robot = StaticImageBackend(args.image, start_xy=(0.0, 0.0), start_heading_rad=0.0, step_m=0.65)
    vlm = OpenAICompatibleVLMClient.from_env()
    agent = NavMemoryAgent(
        robot=robot,
        vlm_client=vlm,
        config=NavAgentConfig(max_steps=args.max_steps, log_full_vlm_input=False),
    )
    result = agent.run_until_done(goal_map_xy=(args.goal_x, args.goal_y), max_steps=args.max_steps)
    out_dir = Path(args.out)
    agent.save_run(out_dir)
    print(result.summary())
    print(f"Saved logs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
