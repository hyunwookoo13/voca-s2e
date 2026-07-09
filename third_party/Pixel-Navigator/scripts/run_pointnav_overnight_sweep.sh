#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/icra/voca-s2e/reports/pointnav_hm3d_overnight_$(date +%Y%m%d_%H%M%S)}"
EPISODES="${POINTNAV_SWEEP_EPISODES:-30}"
MAX_ENV_STEPS="${POINTNAV_MAX_ENV_STEPS:-250}"
MAX_AGENT_STEPS="${POINTNAV_MAX_AGENT_STEPS:-80}"
CHECKPOINT="${PIXELNAV_POLICY_CHECKPOINT:-/home/icra/Pixel-Navigator/checkpoints/navigator.pth}"

mkdir -p "$ROOT"

run_one() {
  local name="$1"
  local policy="$2"
  local vlm="$3"
  local y_ratio="$4"
  local rotate_threshold="$5"
  local max_pixelnav_steps="$6"
  local out="$ROOT/$name"
  mkdir -p "$out"
  cat > "$out/run_config.json" <<EOF
{
  "name": "$name",
  "pixelnav_policy": "$policy",
  "vlm": "$vlm",
  "heuristic_y_ratio": $y_ratio,
  "bearing_rotate_threshold_deg": $rotate_threshold,
  "max_pixelnav_steps": $max_pixelnav_steps,
  "pointnav_goal_source": "sensor",
  "eval_episodes": $EPISODES,
  "max_env_steps": $MAX_ENV_STEPS,
  "max_agent_steps": $MAX_AGENT_STEPS
}
EOF
  echo "[run] $name"
  PIXELNAV_DEVICE="${PIXELNAV_DEVICE:-cpu}" \
  HABITAT_DATA_DIR="${HABITAT_DATA_DIR:-/home/icra/habitat_data}" \
  PIXELNAV_POLICY_CHECKPOINT="$CHECKPOINT" \
  PYTHONPATH=/home/icra/voca-s2e:/home/icra/voca-s2e/qwen_nav_memory_framework_v5:. \
  /home/icra/micromamba/bin/micromamba run -r /home/icra/micromamba-root -n habitat \
    python voca_memory_benchmark.py \
      --task pointnav \
      --dataset hm3d \
      --vlm "$vlm" \
      --heuristic-y-ratio "$y_ratio" \
      --bearing-rotate-threshold-deg "$rotate_threshold" \
      --eval-episodes "$EPISODES" \
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

run_one "bearing_checkpoint_y075_p8_r25" "checkpoint" "pointnav-bearing" 0.75 25 8
run_one "bearing_checkpoint_y065_p8_r25" "checkpoint" "pointnav-bearing" 0.65 25 8
run_one "bearing_checkpoint_y085_p8_r20" "checkpoint" "pointnav-bearing" 0.85 20 8
run_one "bearing_forward_y075_p8_r25" "forward" "pointnav-bearing" 0.75 25 8
run_one "bearing_reactive_forward_y075_p8_r25" "reactive-forward" "pointnav-bearing" 0.75 25 8
run_one "bearing_reactive_forward_y075_p12_r20" "reactive-forward" "pointnav-bearing" 0.75 20 12
run_one "bearing_pointgoal_reactive_y075_p12_r20" "pointgoal-reactive" "pointnav-bearing" 0.75 20 12
run_one "oracle_shortest_path_y075_p12_r20" "shortest-path" "pointnav-bearing" 0.75 20 12

/home/icra/micromamba/bin/micromamba run -r /home/icra/micromamba-root -n habitat \
  python scripts/summarize_pointnav_sweep.py "$ROOT" | tee "$ROOT/summary.log"

echo "$ROOT"
