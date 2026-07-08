#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/icra/voca-s2e/reports/pointnav_hm3d_fast_ablation_$(date +%Y%m%d_%H%M%S)}"
EPISODES="${POINTNAV_FAST_EPISODES:-30}"
CHECKPOINT_EPISODES="${POINTNAV_CHECKPOINT_EPISODES:-3}"
MAX_ENV_STEPS="${POINTNAV_MAX_ENV_STEPS:-250}"
MAX_AGENT_STEPS="${POINTNAV_MAX_AGENT_STEPS:-80}"
CHECKPOINT="${PIXELNAV_POLICY_CHECKPOINT:-/home/icra/Pixel-Navigator/checkpoints/navigator.pth}"

mkdir -p "$ROOT"

run_one() {
  local name="$1"
  local policy="$2"
  local episodes="$3"
  local max_pixelnav_steps="$4"
  local out="$ROOT/$name"
  mkdir -p "$out"
  cat > "$out/run_config.json" <<EOF
{
  "name": "$name",
  "pixelnav_policy": "$policy",
  "vlm": "pointnav-bearing",
  "heuristic_y_ratio": 0.75,
  "bearing_rotate_threshold_deg": 20,
  "max_pixelnav_steps": $max_pixelnav_steps,
  "pointnav_goal_source": "sensor",
  "eval_episodes": $episodes,
  "max_env_steps": $MAX_ENV_STEPS,
  "max_agent_steps": $MAX_AGENT_STEPS
}
EOF
  echo "[run] $name episodes=$episodes"
  PIXELNAV_DEVICE="${PIXELNAV_DEVICE:-cpu}" \
  HABITAT_DATA_DIR="${HABITAT_DATA_DIR:-/home/icra/habitat_data}" \
  PIXELNAV_POLICY_CHECKPOINT="$CHECKPOINT" \
  PYTHONPATH=/home/icra/voca-s2e:/home/icra/voca-s2e/qwen_nav_memory_framework_v5:. \
  /home/icra/micromamba/bin/micromamba run -r /home/icra/micromamba-root -n habitat \
    python voca_memory_benchmark.py \
      --task pointnav \
      --dataset hm3d \
      --vlm pointnav-bearing \
      --heuristic-y-ratio 0.75 \
      --bearing-rotate-threshold-deg 20 \
      --eval-episodes "$episodes" \
      --max-env-steps "$MAX_ENV_STEPS" \
      --max-agent-steps "$MAX_AGENT_STEPS" \
      --max-pixelnav-steps "$max_pixelnav_steps" \
      --pointnav-goal-source sensor \
      --checkpoint "$CHECKPOINT" \
      --pixelnav-device "${PIXELNAV_DEVICE:-cpu}" \
      --pixelnav-policy "$policy" \
      --out "$out" \
      > "$out/run.log" 2>&1
}

cd /home/icra/voca-s2e/third_party/Pixel-Navigator

run_one "checkpoint_smoke_y075_p8_r20" "checkpoint" "$CHECKPOINT_EPISODES" 8
run_one "proposed_reactive_forward_y075_p12_r20" "reactive-forward" "$EPISODES" 12
run_one "pointgoal_reactive_y075_p12_r20" "pointgoal-reactive" "$EPISODES" 12
run_one "oracle_shortest_path_y075_p12_r20" "shortest-path" "$EPISODES" 12

/home/icra/micromamba/bin/micromamba run -r /home/icra/micromamba-root -n habitat \
  python scripts/summarize_pointnav_sweep.py "$ROOT" | tee "$ROOT/summary.log"

echo "$ROOT"
