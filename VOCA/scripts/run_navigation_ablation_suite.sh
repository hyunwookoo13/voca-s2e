#!/usr/bin/env bash
set -euo pipefail

VOCA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${VOCA_ROOT}/scripts/voca_env.sh"
OUTPUT_ROOT="${1:-${VOCA_ROOT}/outputs/navigation_ablation_$(date +%Y%m%d_%H%M%S)}"
EPISODES="${VOCA_ABLATION_EPISODES:-30}"
START_INDEX="${VOCA_ABLATION_START_INDEX:-0}"
POOL_SIZE="${VOCA_ABLATION_POOL_SIZE:-${EPISODES}}"
MAX_STEPS="${VOCA_ABLATION_MAX_STEPS:-1000}"
SEED="${VOCA_ABLATION_SEED:-20260710}"

MODEL_LIST_URL="${QWEN_BASE_URL%/}/models"
SERVED_MODEL_ROOT="$(
  curl -fsS --max-time 15 "${MODEL_LIST_URL}" |
    jq -r --arg model "${QWEN_MODEL}" \
      '.data[] | select(.id == $model) | .root' |
    head -n 1
)"
if [[ -z "${SERVED_MODEL_ROOT}" ]]; then
  printf 'required model id is not served: %s at %s\n' \
    "${QWEN_MODEL}" "${MODEL_LIST_URL}" >&2
  exit 2
fi
if [[ "${SERVED_MODEL_ROOT}" != "${VOCA_QWEN_MODEL_ROOT}" ]]; then
  printf 'served model root mismatch: expected=%s actual=%s\n' \
    "${VOCA_QWEN_MODEL_ROOT}" "${SERVED_MODEL_ROOT}" >&2
  exit 2
fi
printf 'locked VLM: %s (%s) at %s\n' \
  "${QWEN_MODEL}" "${SERVED_MODEL_ROOT}" "${QWEN_BASE_URL}"

mkdir -p "${OUTPUT_ROOT}"

run_condition() {
  local label="$1"
  local memory_enabled="$2"
  local pixelnav_scoring_enabled="$3"
  local run_dir="${OUTPUT_ROOT}/${label}"
  local metrics="${run_dir}/objnav_hm3d.csv"

  if [[ -f "${metrics}" ]]; then
    local completed
    completed="$(awk 'END {print NR > 0 ? NR - 1 : 0}' "${metrics}")"
    if [[ "${completed}" -ge "${EPISODES}" ]]; then
      printf 'skip %s: %s/%s episodes already complete\n' "${label}" "${completed}" "${EPISODES}"
      return
    fi
    printf 'refusing partial overwrite for %s (%s/%s); choose a new output root\n' \
      "${label}" "${completed}" "${EPISODES}" >&2
    return 2
  fi

  mkdir -p "${run_dir}"
  printf 'run %s: memory=%s pixelnav_scoring=%s episodes=%s\n' \
    "${label}" "${memory_enabled}" "${pixelnav_scoring_enabled}" "${EPISODES}"
  env \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    VOCA_PLANNER=qwen_vlm \
    VOCA_QWEN_MEMORY_SIDECAR="${memory_enabled}" \
    VOCA_PIXELNAV_CANDIDATE_SCORING="${pixelnav_scoring_enabled}" \
    VOCA_EVAL_EPISODES="${EPISODES}" \
    VOCA_EPISODE_START_INDEX="${START_INDEX}" \
    VOCA_EPISODE_POOL_SIZE="${POOL_SIZE}" \
    VOCA_MAX_EPISODE_STEPS="${MAX_STEPS}" \
    VOCA_BENCHMARK_SEED="${SEED}" \
    VOCA_OUTPUT_DIR="${run_dir}" \
    VOCA_TRAJECTORY_DIR="${run_dir}/trajectories" \
    VOCA_OBJNAV_METRICS_PATH="${metrics}" \
    bash "${VOCA_ROOT}/scripts/run_objnav_smoke.sh"
}

# A paired 2x2 factorial isolates verified memory, PixelNav-conditioned scoring,
# and their interaction while all target-completion guards remain fixed.
run_condition full 1 1
run_condition no_memory 0 1
run_condition no_pixelnav_scoring 1 0
run_condition no_memory_no_pixelnav_scoring 0 0

"${MICROMAMBA_BIN:-${HOME}/micromamba/bin/micromamba}" run \
  -r "${MAMBA_ROOT_PREFIX:-${HOME}/micromamba-root}" \
  -n "${MAMBA_ENV:-habitat}" \
  python "${VOCA_ROOT}/scripts/summarize_navigation_ablation.py" \
    --run "full=${OUTPUT_ROOT}/full" \
    --run "no_memory=${OUTPUT_ROOT}/no_memory" \
    --run "no_pixelnav_scoring=${OUTPUT_ROOT}/no_pixelnav_scoring" \
    --run "no_memory_no_pixelnav_scoring=${OUTPUT_ROOT}/no_memory_no_pixelnav_scoring" \
    --output-dir "${OUTPUT_ROOT}/summary"

printf 'ablation suite complete: %s\n' "${OUTPUT_ROOT}"
