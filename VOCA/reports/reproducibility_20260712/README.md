# Reproducibility Snapshot (2026-07-12)

This directory preserves the lightweight, reviewable artifacts used by the
weekend integration report. Full trajectory frames, model inputs, checkpoints,
and generated benchmark directories are intentionally excluded from Git.

## Contents

- `final_v3/`
  - Metrics and readiness summaries for the completed chair and plant episodes.
  - Per-episode benchmark manifests and final memory graphs.
  - The interrupted toilet episode is excluded from the CSV and report.
- `ablation/`
  - Three-episode paired summaries for memory and PixelNav candidate-scoring toggles.
- `stress/`
  - Deterministic guard, revisit, and stop-verifier stress cases and summary.
- `place_recognition/`
  - DINOv2 revisit calibration pairs, thresholds, and selected operating point.
- `model_comparison/`
  - Raw decision records and summaries for the Qwen-VL variants reported in
    `../qwen_vlm_model_comparison_report_20260710.md`.

## Reporting Boundary

The final-v3 episodes use the derived
`hm3d_derived_coarse_goalnav_v1_sim_pose_noise0p5` protocol. They must be
reported as RGB plus declared simulator pose with a derived coarse goal, not as
category-only RGB ObjectNav. The small three-episode ablation is diagnostic and
does not establish statistical superiority for memory or candidate scoring.

## Large Artifact Policy

Selected presentation videos live under `../weekend_20260711_12/media/`.
Datasets, checkpoints, complete `outputs/` trees, per-step RGB frames, and full
Qwen call logs remain local artifacts and should be distributed separately if
needed.
