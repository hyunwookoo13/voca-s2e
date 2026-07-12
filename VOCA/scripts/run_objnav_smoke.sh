#!/usr/bin/env bash
set -euo pipefail

VOCA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${VOCA_ROOT}"

source "${VOCA_ROOT}/scripts/voca_env.sh"

MICROMAMBA_BIN="${MICROMAMBA_BIN:-${HOME}/micromamba/bin/micromamba}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/micromamba-root}"
MAMBA_ENV="${MAMBA_ENV:-habitat}"

mkdir -p "${VOCA_OUTPUT_DIR}"

MAX_STEP_ARGS=()
if [[ -n "${VOCA_MAX_EPISODE_STEPS:-}" ]]; then
  MAX_STEP_ARGS=(--max_episode_steps "${VOCA_MAX_EPISODE_STEPS}")
fi

exec "${MICROMAMBA_BIN}" run -r "${MAMBA_ROOT_PREFIX}" -n "${MAMBA_ENV}" \
  python objnav_benchmark.py \
    --eval_episodes "${VOCA_EVAL_EPISODES:-1}" \
    --stage "${VOCA_STAGE:-val}" \
    --device "${VOCA_DEVICE}" \
    --policy_checkpoint "${VOCA_POLICY_CHECKPOINT}" \
    --yoloe_checkpoint "${VOCA_YOLOE_CHECKPOINT}" \
    --output_dir "${VOCA_TRAJECTORY_DIR}" \
    --metrics_path "${VOCA_OBJNAV_METRICS_PATH}" \
    --planner "${VOCA_PLANNER}" \
    --seed "${VOCA_BENCHMARK_SEED:-20260710}" \
    --episode_start_index "${VOCA_EPISODE_START_INDEX:-0}" \
    --episode_pool_size "${VOCA_EPISODE_POOL_SIZE:-0}" \
    "${MAX_STEP_ARGS[@]}" \
    "$@"
