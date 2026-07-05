# VLM Benchmark Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add RGB and top-down visual audit overlays to each VLM benchmark case report.

**Architecture:** Add a focused `goal_adapter/visualization.py` module that renders simple PNG overlays from benchmark case metadata and parsed VLM decisions. Integrate it into `goal_adapter/vlm_benchmark.py` so every case report gets image artifacts and HTML references.

**Tech Stack:** Python standard library only, existing `goal_adapter.habitat_smoke.write_rgb_png`, `unittest`, static HTML reports.

---

### Task 1: Visualization Renderer

**Files:**
- Create: `goal_adapter/visualization.py`
- Test: `tests/test_visualization.py`

- [ ] **Step 1: Write failing tests**

Create tests that build a tiny RGB PNG, call `render_case_visualizations`, and assert that `current_rgb_overlay.png` and `topdown_overlay.png` exist and are valid PNG files.

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -B -m unittest tests.test_visualization`

Expected: fail because `goal_adapter.visualization` does not exist.

- [ ] **Step 3: Implement minimal renderer**

Implement:

- `VisualizationPoint`
- `CaseVisualizationArtifacts`
- `render_case_visualizations(case, output, case_dir)`

The renderer should read simple RGB/RGBA PNG files, draw marker circles, generate a top-down canvas, and write PNG files.

- [ ] **Step 4: Run tests**

Run: `python3 -B -m unittest tests.test_visualization`

Expected: pass.

### Task 2: Benchmark Integration

**Files:**
- Modify: `goal_adapter/vlm_benchmark.py`
- Modify: `tests/test_vlm_benchmark.py`

- [ ] **Step 1: Write failing integration test**

Extend the benchmark report test to assert each successful case directory includes:

- `current_rgb_overlay.png`
- `topdown_overlay.png`

- [ ] **Step 2: Run test to verify failure**

Run: `python3 -B -m unittest tests.test_vlm_benchmark`

Expected: fail because overlays are not written yet.

- [ ] **Step 3: Integrate renderer**

Call `render_case_visualizations` after parsing a successful VLM output. Pass generated artifact paths to the per-case HTML renderer and display overlays beside the raw RGB fallback.

- [ ] **Step 4: Run benchmark tests**

Run: `python3 -B -m unittest tests.test_vlm_benchmark`

Expected: pass.

### Task 3: Full Verification

**Files:**
- No new source files.

- [ ] **Step 1: Run all unit tests**

Run: `python3 -B -m unittest discover -s tests`

Expected: all tests pass.

- [ ] **Step 2: Regenerate actual VLM report**

Run:

```bash
python3 -B -m goal_adapter.vlm_benchmark \
  --cases-dir reports/vlm_benchmark/cases_smoke \
  --output-dir reports/vlm_benchmark/latest \
  --model gemma4:26b \
  --timeout-seconds 420 \
  --max-attempts 2
```

Expected: `case_success_rate=1.0` and each case directory contains both overlay PNGs.
