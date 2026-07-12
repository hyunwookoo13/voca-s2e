#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_TARGET_REPO="$(cd "${SOURCE_ROOT}/.." && pwd)/voca-s2e"
TARGET_REPO="${VOCA_S2E_REPO:-${DEFAULT_TARGET_REPO}}"
TARGET_VOCA="${TARGET_REPO}/VOCA"
SOURCE_MEMORY="${SOURCE_ROOT}/qwen_nav_memory_framework_v6"
TARGET_MEMORY="${TARGET_REPO}/qwen_nav_memory_framework_v6"

DRY_RUN=0
case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=1 ;;
  *)
    printf 'Usage: %s [--dry-run]\n' "$0" >&2
    exit 2
    ;;
esac

if [[ ! -d "${TARGET_REPO}/.git" ]]; then
  printf 'Target is not a Git checkout: %s\n' "${TARGET_REPO}" >&2
  exit 1
fi

if [[ ! -d "${SOURCE_MEMORY}/nav_memory_qwen" ]]; then
  printf 'VOCA v6 memory source is missing: %s\n' "${SOURCE_MEMORY}" >&2
  exit 1
fi

RSYNC_ARGS=(
  --archive
  --checksum
  --delete
  --itemize-changes
  --omit-dir-times
  --no-times
  --exclude=/.git/
  --exclude=/.env
  --exclude=/data
  --exclude=/datasets
  --exclude=/scene_datasets
  --exclude=/checkpoints
  --exclude=/outputs
  --exclude=/results
  --exclude=/tmp
  --exclude=/qwen_nav_memory_framework_v6
  --exclude=/monitor-panoramic.jpg
  --exclude='/*.ts'
  --exclude=__pycache__/
  --exclude='*.pyc'
)

MEMORY_RSYNC_ARGS=(
  --archive
  --checksum
  --delete
  --itemize-changes
  --omit-dir-times
  --no-times
  --exclude=__pycache__/
  --exclude='*.pyc'
)

FILE_RSYNC_ARGS=(--archive --checksum --itemize-changes --no-times)
if (( DRY_RUN )); then
  RSYNC_ARGS+=(--dry-run)
  MEMORY_RSYNC_ARGS+=(--dry-run)
  FILE_RSYNC_ARGS+=(--dry-run)
fi

mkdir -p "${TARGET_VOCA}" "${TARGET_MEMORY}" "${TARGET_REPO}/goal_adapter"

rsync "${RSYNC_ARGS[@]}" "${SOURCE_ROOT}/" "${TARGET_VOCA}/"
rsync "${MEMORY_RSYNC_ARGS[@]}" "${SOURCE_MEMORY}/" "${TARGET_MEMORY}/"
rsync "${FILE_RSYNC_ARGS[@]}" \
  "${SOURCE_ROOT}/memory_graph_visualizer.py" \
  "${TARGET_REPO}/goal_adapter/memory_graph_visualizer.py"

if (( DRY_RUN )); then
  printf 'Dry run complete: %s -> %s\n' "${SOURCE_ROOT}" "${TARGET_REPO}"
else
  printf 'Sync complete: %s -> %s\n' "${SOURCE_ROOT}" "${TARGET_REPO}"
fi
