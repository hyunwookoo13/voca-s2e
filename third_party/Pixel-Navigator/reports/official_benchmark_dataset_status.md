# Official Benchmark Dataset Status

Updated: 2026-07-08

## Goal

Prepare paper-grade Habitat benchmark inputs for SR / SPL / Success reporting.

Target assets:

- HM3D / MP3D scene assets
- PointNav official episode split
- ObjectNav official episode split

## Completed Locally

### HM3D official val scene assets

Downloaded:

```text
/home/icra/habitat_data/versioned_data/hm3d-0.2/hm3d/
```

Symlinks:

```text
/home/icra/habitat_data/scene_datasets/hm3d -> /home/icra/habitat_data/versioned_data/hm3d-0.2/hm3d
/home/icra/habitat_data/scene_datasets/hm3d_v0.2 -> /home/icra/habitat_data/scene_datasets/hm3d
/home/icra/Pixel-Navigator/data -> /home/icra/habitat_data
```

Counts:

```text
HM3D val basis scenes: 100
HM3D val semantic scenes: 36
```

Verified:

```text
Habitat ObjectNav HM3D v2 val env reset succeeded.
Observation keys: compass, depth, gps, objectgoal, rgb
Metric keys: collisions, distance_to_goal, distance_to_goal_reward, soft_spl, spl, success, top_down_map
```

### PointNav episode splits

Downloaded and extracted:

```text
/home/icra/habitat_data/datasets/pointnav/hm3d/v1/
/home/icra/habitat_data/datasets/pointnav/mp3d/v1/
```

Sizes:

```text
995M  /home/icra/habitat_data/datasets/pointnav/hm3d/v1
400M  /home/icra/habitat_data/datasets/pointnav/mp3d/v1
```

### ObjectNav episode splits

Already present and canonical symlinks are in place:

```text
/home/icra/habitat_data/datasets/objectnav/hm3d/v2 -> /home/icra/habitat_data/datasets/objectnav_hm3d_v2
/home/icra/habitat_data/datasets/objectnav/mp3d/v1 -> /home/icra/habitat_data/datasets/objectnav_mp3d_v1
```

These are episode JSON splits only. They are not scene meshes.

### Existing test/example scenes

Available locally:

```text
/home/icra/habitat_data/versioned_data/habitat_test_scenes/
/home/icra/habitat_data/versioned_data/mp3d_example_scene_1.1/
```

These are useful for smoke checks, but they are not sufficient for official HM3D/MP3D SR/SPL reporting.

## HM3D Download Method

Attempted:

```bash
python -m habitat_sim.utils.datasets_download \
  --uids hm3d_val_v0.2 \
  --data-path /home/icra/habitat_data \
  --no-replace
```

Result:

```text
AssertionError: Username required, please enter with --username
```

HM3D scene assets require Matterport/HM3D access and a Matterport API token.

Create the API token at:

```text
https://my.matterport.com/settings/account/devtools
```

Use:

```text
API token ID     -> username
API token secret -> password
```

Working command:

```bash
cd /home/icra/Pixel-Navigator
MATTERPORT_TOKEN_ID='...' \
MATTERPORT_TOKEN_SECRET='...' \
HM3D_UIDS='hm3d_val_v0.2' \
./scripts/download_official_benchmark_assets.sh hm3d-scenes
```

For broader experiments:

```bash
HM3D_UIDS='hm3d_minival_v0.2 hm3d_val_v0.2'
```

or full Habitat BASIS-compressed HM3D:

```bash
HM3D_UIDS='hm3d'
```

## Blocked: MP3D Scene Assets

Habitat-Sim downloader lists only:

```text
mp3d_example_scene
```

It does not expose full MP3D scene assets. Full MP3D scene meshes must be obtained through the Matterport3D license/download process and placed under:

```text
/home/icra/habitat_data/scene_datasets/mp3d/{scene}/{scene}.glb
/home/icra/habitat_data/scene_datasets/mp3d/mp3d.scene_dataset_config.json
```

## Pixel-Navigator Expected Paths

Current Pixel-Navigator constants:

```text
HABITAT_DATA_DIR = /home/icra/habitat_data
SCENE_PREFIX = /home/icra/habitat_data/scene_datasets/
EPISODE_PREFIX = /home/icra/habitat_data/datasets/
```

ObjectNav HM3D config expects:

```text
/home/icra/habitat_data/datasets/objectnav/hm3d/v2/{split}/{split}.json.gz
/home/icra/habitat_data/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json
```

The helper script creates:

```text
/home/icra/habitat_data/scene_datasets/hm3d_v0.2 -> /home/icra/habitat_data/scene_datasets/hm3d
```

after HM3D download completes.

## Helper Script

```bash
cd /home/icra/Pixel-Navigator
./scripts/download_official_benchmark_assets.sh status
./scripts/download_official_benchmark_assets.sh episodes
./scripts/download_official_benchmark_assets.sh hm3d-scenes
```

## Current Readiness

| Item | Status |
|---|---|
| PointNav HM3D episode split | Ready |
| PointNav MP3D episode split | Ready |
| ObjectNav HM3D v2 episode split | Ready |
| ObjectNav MP3D v1 episode split | Ready |
| Habitat test scenes | Ready for smoke only |
| MP3D example scene | Ready for smoke only |
| HM3D official val scene assets | Ready |
| HM3D ObjectNav v2 val env reset | Verified |
| MP3D official scene assets | Blocked by Matterport3D licensed scene download |

HM3D val SR/SPL/Success benchmark runs are now unblocked. MP3D SR/SPL/Success reporting still requires the licensed MP3D scene meshes.
