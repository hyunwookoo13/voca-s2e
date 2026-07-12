# VOCA Integration Notes

This vendored copy keeps the Pixel-Navigator code needed to run Habitat + PixelNav + VOCA memory experiments from the `voca-s2e` repository.

## What Is Included

- PixelNav policy/network/runtime source code
- Habitat ObjectNav/PointNav config helpers
- VOCA memory benchmark runner:

```text
voca_memory_benchmark.py
```

- Dataset download notes:

```text
checkpoints/DATASETS.md
```

- Lightweight integration reports under `reports/`
- Unit tests under `tests/`

## What Is Not Included

The following are intentionally excluded from git:

```text
data/
checkpoints/*.pth
checkpoints/*.pt
checkpoints/*.ckpt
assets/*.mp4
assets/*.gif
runtime benchmark outputs
```

Use local environment variables to point to data/checkpoints:

```bash
export HABITAT_DATA_DIR=/home/icra/habitat_data
export PIXELNAV_CHECKPOINT_DIR=/home/icra/Pixel-Navigator/checkpoints
export PIXELNAV_POLICY_CHECKPOINT=/home/icra/Pixel-Navigator/checkpoints/navigator.pth
```

## PointNav Smoke Example

From `voca-s2e`:

```bash
cd third_party/Pixel-Navigator
PYTHONPATH=/home/icra/voca-s2e:/home/icra/voca-s2e/qwen_nav_memory_framework_v5:. \
PIXELNAV_DEVICE=cpu \
python voca_memory_benchmark.py \
  --task pointnav \
  --dataset hm3d \
  --memory-framework v5 \
  --vlm pointnav-bearing \
  --eval-episodes 3 \
  --max-env-steps 250 \
  --max-agent-steps 40 \
  --max-pixelnav-steps 8 \
  --write-videos \
  --out /home/icra/voca-s2e/reports/pointnav_hm3d_voca_memory_e3_video
```

Each episode writes:

```text
episode_000*/fps.mp4
episode_000*/metric.mp4
episode_000*/memory_graph_reconstruction.mp4
episode_000*/memory/memory_graph.json
```

## v6 GaP-lite Ablation

The default memory framework remains `qwen_nav_memory_framework_v5`. To test the v6 GaP-lite sidecar, switch the framework explicitly:

```bash
cd third_party/Pixel-Navigator
PYTHONPATH=/home/icra/voca-s2e:/home/icra/voca-s2e/qwen_nav_memory_framework_v6:. \
PIXELNAV_DEVICE=cpu \
python voca_memory_benchmark.py \
  --task pointnav \
  --dataset hm3d \
  --memory-framework v6 \
  --vlm pointnav-bearing \
  --eval-episodes 3 \
  --max-env-steps 250 \
  --max-agent-steps 40 \
  --max-pixelnav-steps 8 \
  --write-videos \
  --out /home/icra/voca-s2e/reports/pointnav_hm3d_voca_memory_v6_e3_video
```

v6 adds `candidate_refs`, `nav_skill_cards`, `validation_checkpoints`, `validation_feedback`, and `Feedback.md`. It should be treated as an ablation axis until SR/SPL confirms it is not worse than the v5 baseline.

## Repository Boundary

`voca-s2e` is the integration repository. This directory is a vendored runtime dependency so that reviewers can inspect how Habitat official envs, PixelNav rollouts, VLM grounded goals, and `qwen_nav_memory_framework_v5`/`qwen_nav_memory_framework_v6` connect end-to-end.
