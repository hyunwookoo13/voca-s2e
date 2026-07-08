#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/icra/habitat_data}"
MICROMAMBA="${MICROMAMBA:-/home/icra/micromamba/bin/micromamba}"
MAMBA_ROOT="${MAMBA_ROOT:-/home/icra/micromamba-root}"
HABITAT_ENV="${HABITAT_ENV:-habitat}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DOWNLOAD_DIR="${DATA_ROOT}/downloads"
mkdir -p "${DOWNLOAD_DIR}"

download_zip() {
  local url="$1"
  local zip_path="$2"
  local extract_dir="$3"

  mkdir -p "${extract_dir}"
  wget -c -O "${zip_path}" "${url}"
  unzip -n "${zip_path}" -d "${extract_dir}"
}

ensure_objectnav_symlinks() {
  mkdir -p "${DATA_ROOT}/datasets/objectnav/hm3d" "${DATA_ROOT}/datasets/objectnav/mp3d"

  if [ -d "${DATA_ROOT}/datasets/objectnav_hm3d_v2" ] && [ ! -e "${DATA_ROOT}/datasets/objectnav/hm3d/v2" ]; then
    ln -s "${DATA_ROOT}/datasets/objectnav_hm3d_v2" "${DATA_ROOT}/datasets/objectnav/hm3d/v2"
  fi

  if [ -d "${DATA_ROOT}/datasets/objectnav_mp3d_v1" ] && [ ! -e "${DATA_ROOT}/datasets/objectnav/mp3d/v1" ]; then
    ln -s "${DATA_ROOT}/datasets/objectnav_mp3d_v1" "${DATA_ROOT}/datasets/objectnav/mp3d/v1"
  fi
}

ensure_hm3d_compat_symlink() {
  local hm3d_link="${DATA_ROOT}/scene_datasets/hm3d"
  local hm3d_v02="${DATA_ROOT}/scene_datasets/hm3d_v0.2"

  if [ -e "${hm3d_link}" ] && [ ! -e "${hm3d_v02}" ]; then
    ln -s "${hm3d_link}" "${hm3d_v02}"
  fi
}

ensure_repo_data_symlink() {
  local repo_data="${REPO_ROOT}/data"

  if [ ! -e "${repo_data}" ]; then
    ln -s "${DATA_ROOT}" "${repo_data}"
  fi
}

download_pointnav_splits() {
  download_zip \
    "https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/hm3d/v1/pointnav_hm3d_v1.zip" \
    "${DOWNLOAD_DIR}/pointnav_hm3d_v1.zip" \
    "${DATA_ROOT}/datasets/pointnav/hm3d/v1"

  download_zip \
    "https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/mp3d/v1/pointnav_mp3d_v1.zip" \
    "${DOWNLOAD_DIR}/pointnav_mp3d_v1.zip" \
    "${DATA_ROOT}/datasets/pointnav/mp3d/v1"
}

download_hm3d_scene_assets() {
  local uids="${HM3D_UIDS:-hm3d_val_v0.2}"
  local matterport_username="${MATTERPORT_USERNAME:-${MATTERPORT_TOKEN_ID:-}}"
  local matterport_password="${MATTERPORT_PASSWORD:-${MATTERPORT_TOKEN_SECRET:-}}"

  if [ -z "${matterport_username}" ] || [ -z "${matterport_password}" ]; then
    echo "HM3D scene asset download requires a Matterport API token." >&2
    echo "Create it at: https://my.matterport.com/settings/account/devtools" >&2
    echo "Use the API token ID as MATTERPORT_TOKEN_ID and the API token secret as MATTERPORT_TOKEN_SECRET." >&2
    echo "Example:" >&2
    echo "  MATTERPORT_TOKEN_ID=... MATTERPORT_TOKEN_SECRET=... HM3D_UIDS='hm3d_val_v0.2' $0 hm3d-scenes" >&2
    return 2
  fi

  HM3D_DOWNLOAD_UIDS="${uids}" \
  HM3D_DOWNLOAD_DATA_ROOT="${DATA_ROOT}" \
  MATTERPORT_DOWNLOAD_USERNAME="${matterport_username}" \
  MATTERPORT_DOWNLOAD_PASSWORD="${matterport_password}" \
  "${MICROMAMBA}" run -r "${MAMBA_ROOT}" -n "${HABITAT_ENV}" python - <<'PY'
import os

from habitat_sim.utils.datasets_download import main

main(
    [
        "--uids",
        *os.environ["HM3D_DOWNLOAD_UIDS"].split(),
        "--data-path",
        os.environ["HM3D_DOWNLOAD_DATA_ROOT"],
        "--username",
        os.environ["MATTERPORT_DOWNLOAD_USERNAME"],
        "--password",
        os.environ["MATTERPORT_DOWNLOAD_PASSWORD"],
        "--no-replace",
    ]
)
PY

  ensure_hm3d_compat_symlink
}

print_status() {
  echo "DATA_ROOT=${DATA_ROOT}"
  echo
  echo "[Episode splits]"
  du -shL \
    "${DATA_ROOT}/datasets/pointnav/hm3d/v1" \
    "${DATA_ROOT}/datasets/pointnav/mp3d/v1" \
    "${DATA_ROOT}/datasets/objectnav/hm3d/v2" \
    "${DATA_ROOT}/datasets/objectnav/mp3d/v1" 2>/dev/null || true

  echo
  echo "[Scene assets]"
  if [ -e "${DATA_ROOT}/scene_datasets/hm3d" ]; then
    echo "HM3D val basis scenes: $(find -L "${DATA_ROOT}/scene_datasets/hm3d" -path '*/val/*/*.basis.glb' | wc -l)"
    echo "HM3D val semantic scenes: $(find -L "${DATA_ROOT}/scene_datasets/hm3d" -path '*/val/*/*.semantic.glb' | wc -l)"
  fi
  find "${DATA_ROOT}/scene_datasets" "${DATA_ROOT}/versioned_data" \
    -maxdepth 4 -type f \( -name '*.glb' -o -name '*.basis.glb' -o -name '*.scene_dataset_config.json' \) \
    | sort | sed -n '1,80p'
}

cmd="${1:-all}"

case "${cmd}" in
  all)
    ensure_repo_data_symlink
    ensure_objectnav_symlinks
    download_pointnav_splits
    download_hm3d_scene_assets || true
    print_status
    ;;
  episodes)
    ensure_repo_data_symlink
    ensure_objectnav_symlinks
    download_pointnav_splits
    print_status
    ;;
  hm3d-scenes)
    ensure_repo_data_symlink
    download_hm3d_scene_assets
    print_status
    ;;
  status)
    ensure_repo_data_symlink
    ensure_objectnav_symlinks
    ensure_hm3d_compat_symlink
    print_status
    ;;
  *)
    echo "Usage: $0 [all|episodes|hm3d-scenes|status]" >&2
    exit 64
    ;;
esac
