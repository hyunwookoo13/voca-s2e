# Overnight PointNav Benchmark Tuning Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify the `voca-s2e` integrated Habitat + PixelNav + v5 memory runner and run enough HM3D PointNav trials to compare SR/SPL before tomorrow's ablation work.

**Architecture:** Use `third_party/Pixel-Navigator/voca_memory_benchmark.py` as the official Habitat runner. Generate a small video run for qualitative inspection, then run larger no-video sweeps over PointNav controller settings and summarize metrics into reports.

**Tech Stack:** Habitat-Lab/Habitat-Sim in the `habitat` micromamba env, PixelNav checkpoint from local disk, `qwen_nav_memory_framework_v5`, Python unittest, JSON/CSV reports.

## Global Constraints

- Run from `/home/icra/voca-s2e/third_party/Pixel-Navigator`.
- Keep datasets, checkpoints, JSON result dumps, CSV files, and videos out of git.
- Use `/home/icra/habitat_data` for official HM3D data.
- Use `/home/icra/Pixel-Navigator/checkpoints/navigator.pth` as the local PixelNav checkpoint.
- Use `--max-env-steps 250` as the PointNav low-level action budget.
- Treat SR/SPL as empirical metrics only after multi-episode runs; do not infer performance from 1-3 episode smoke runs.

---

### Task 1: Qualitative Video Smoke

**Files:**
- Read: `third_party/Pixel-Navigator/voca_memory_benchmark.py`
- Output only: `/home/icra/voca-s2e/reports/pointnav_hm3d_voca_memory_e3_video_*`

**Interfaces:**
- Consumes: CLI option `--write-videos`
- Produces: per-episode `fps.mp4`, `metric.mp4`, `memory_graph_reconstruction.mp4`

- [ ] Run 3 HM3D PointNav episodes with videos and `--max-env-steps 250`.
- [ ] Verify each episode directory has first-person, metric, and memory graph videos.
- [ ] Read `voca_memory_pointnav_benchmark_summary.json` and record SR/SPL/distance deltas.

### Task 2: SR/SPL Sweep Without Videos

**Files:**
- Create: `third_party/Pixel-Navigator/scripts/run_pointnav_overnight_sweep.sh`
- Output only: `/home/icra/voca-s2e/reports/pointnav_hm3d_overnight_*`

**Interfaces:**
- Consumes: benchmark CLI settings `--vlm`, `--heuristic-y-ratio`, `--bearing-rotate-threshold-deg`, `--max-pixelnav-steps`
- Produces: one summary JSON/CSV per configuration

- [ ] Create a shell script that runs multiple configs with `--eval-episodes` large enough to estimate SR/SPL.
- [ ] Start with cheap configs first so early failures appear quickly.
- [ ] Save logs under the output root.

### Task 3: Result Aggregation

**Files:**
- Create: `third_party/Pixel-Navigator/scripts/summarize_pointnav_sweep.py`
- Output only: sweep summary markdown/json under the selected output root

**Interfaces:**
- Consumes: `voca_memory_pointnav_benchmark_summary.json` files
- Produces: sorted table by success, SPL, distance delta, env steps

- [ ] Parse every summary JSON under the sweep root.
- [ ] Print and write a compact table sorted by success then SPL.
- [ ] Include best config command for reproducibility.
