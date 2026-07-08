# data-collection related directory
import os

HABITAT_DATA_DIR = os.getenv("HABITAT_DATA_DIR", "/home/icra/habitat_data")
HABITAT_PACKAGE_DIR = os.getenv(
    "HABITAT_PACKAGE_DIR",
    "/home/icra/micromamba-root/envs/habitat/lib/python3.9/site-packages/habitat",
)
HM3D_CONFIG_PATH = f"{HABITAT_PACKAGE_DIR}/config/benchmark/nav/objectnav/objectnav_hm3d.yaml"
MP3D_CONFIG_PATH = f"{HABITAT_PACKAGE_DIR}/config/benchmark/nav/objectnav/objectnav_mp3d.yaml"
SCENE_PREFIX = f"{HABITAT_DATA_DIR}/scene_datasets/"
EPISODE_PREFIX = f"{HABITAT_DATA_DIR}/datasets/"
# detection & segmentation related configs and checkpoints
PIXELNAV_CHECKPOINT_DIR = os.getenv("PIXELNAV_CHECKPOINT_DIR", "./checkpoints")
GROUNDING_DINO_CONFIG_PATH = os.getenv(
    "GROUNDING_DINO_CONFIG_PATH",
    f"{PIXELNAV_CHECKPOINT_DIR}/GroundingDINO_SwinB_cfg.py",
)
GROUNDING_DINO_CHECKPOINT_PATH = os.getenv(
    "GROUNDING_DINO_CHECKPOINT_PATH",
    f"{PIXELNAV_CHECKPOINT_DIR}/groundingdino_swinb_cogcoor.pth",
)
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.getenv("SAM_CHECKPOINT_PATH", f"{PIXELNAV_CHECKPOINT_DIR}/sam_vit_h_4b8939.pth")
# policy checkpoint
POLICY_CHECKPOINT = os.getenv("PIXELNAV_POLICY_CHECKPOINT", f"{PIXELNAV_CHECKPOINT_DIR}/navigator.pth")
