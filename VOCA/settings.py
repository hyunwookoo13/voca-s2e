import os
import importlib.util
from pathlib import Path
from typing import Optional


def _path_from_env(name: str, default: str) -> str:
    return str(Path(os.getenv(name, default)).expanduser())


def _dir_from_env(name: str, default: str) -> str:
    path = _path_from_env(name, default)
    return path if path.endswith(os.sep) else path + os.sep


def _list_from_env(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _first_existing_path(candidates: list[Path], fallback: Path) -> str:
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(fallback)


def _habitat_package_dir() -> Optional[Path]:
    for env_name in ("VOCA_HABITAT_PACKAGE_DIR", "HABITAT_PACKAGE_DIR"):
        value = os.getenv(env_name)
        if value:
            return Path(value).expanduser()

    spec = importlib.util.find_spec("habitat")
    locations = getattr(spec, "submodule_search_locations", None) if spec else None
    if locations:
        return Path(next(iter(locations))).expanduser()
    return None


PROJECT_ROOT = Path(__file__).resolve().parent

# Device/runtime defaults. Override with VOCA_DEVICE or VOCA_CUDA_VISIBLE_DEVICES.
DEFAULT_DEVICE = os.getenv("VOCA_DEVICE", "cuda:0")
DEFAULT_CUDA_VISIBLE_DEVICES = os.getenv("VOCA_CUDA_VISIBLE_DEVICES", "0")

# Habitat and dataset paths.
HABITAT_ROOT_DIR = _path_from_env("VOCA_HABITAT_ROOT", "~/habitat-lab")
_HABITAT_PACKAGE_DIR = _habitat_package_dir()
_HABITAT_CONFIG_CANDIDATES = [
    Path(HABITAT_ROOT_DIR) / "habitat-lab" / "habitat" / "config",
    Path(HABITAT_ROOT_DIR) / "habitat" / "config",
]
if _HABITAT_PACKAGE_DIR is not None:
    _HABITAT_CONFIG_CANDIDATES.insert(0, _HABITAT_PACKAGE_DIR / "config")

HM3D_CONFIG_PATH = _path_from_env(
    "VOCA_HM3D_CONFIG_PATH",
    _first_existing_path(
        [base / "benchmark/nav/objectnav/objectnav_hm3d.yaml" for base in _HABITAT_CONFIG_CANDIDATES],
        Path(HABITAT_ROOT_DIR) / "habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml",
    ),
)
MP3D_CONFIG_PATH = _path_from_env(
    "VOCA_MP3D_CONFIG_PATH",
    _first_existing_path(
        [base / "benchmark/nav/objectnav/objectnav_mp3d.yaml" for base in _HABITAT_CONFIG_CANDIDATES],
        Path(HABITAT_ROOT_DIR) / "habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_mp3d.yaml",
    ),
)
DATA_ROOT = _path_from_env("VOCA_DATA_ROOT", str(PROJECT_ROOT / "data"))
SCENE_PREFIX = _dir_from_env("VOCA_SCENE_DATASETS_DIR", f"{DATA_ROOT}/scene_datasets")
EPISODE_PREFIX = _dir_from_env("VOCA_DATASETS_DIR", f"{DATA_ROOT}/datasets")

# Checkpoints.
CHECKPOINT_DIR = _path_from_env("VOCA_CHECKPOINT_DIR", str(PROJECT_ROOT / "checkpoints"))
GROUNDING_DINO_CONFIG_PATH = _path_from_env(
    "VOCA_GROUNDING_DINO_CONFIG",
    f"{CHECKPOINT_DIR}/GroundingDINO_SwinB_cfg.py",
)
GROUNDING_DINO_CHECKPOINT_PATH = _path_from_env(
    "VOCA_GROUNDING_DINO_CHECKPOINT",
    f"{CHECKPOINT_DIR}/groundingdino_swinb_cogcoor.pth",
)
SAM_ENCODER_VERSION = os.getenv("VOCA_SAM_ENCODER_VERSION", "vit_h")
SAM_CHECKPOINT_PATH = _path_from_env("VOCA_SAM_CHECKPOINT", f"{CHECKPOINT_DIR}/sam_vit_h_4b8939.pth")
POLICY_CHECKPOINT = _path_from_env("VOCA_POLICY_CHECKPOINT", f"{CHECKPOINT_DIR}/pixelnav_A.ckpt")
YOLOE_CHECKPOINT_PATH = _path_from_env("VOCA_YOLOE_CHECKPOINT", f"{CHECKPOINT_DIR}/yoloe-11l-seg.pt")

# Evaluation outputs.
OUTPUT_DIR = _path_from_env("VOCA_OUTPUT_DIR", str(PROJECT_ROOT / "outputs"))
TRAJECTORY_DIR = _path_from_env("VOCA_TRAJECTORY_DIR", f"{OUTPUT_DIR}/trajectories")
OBJNAV_METRICS_PATH = _path_from_env("VOCA_OBJNAV_METRICS_PATH", f"{OUTPUT_DIR}/objnav_hm3d.csv")

# Default detector prompt classes for ObjNav evaluation.
DETECT_OBJECTS = _list_from_env(
    "VOCA_DETECT_OBJECTS",
    ["bed", "sofa", "chair", "plant", "tv", "toilet", "floor"],
)

# LLM/VLM backend. Supported values in this repository: gemini, qwen, ollama.
LLM_BACKEND = os.getenv("VOCA_LLM_BACKEND", "qwen").lower()
