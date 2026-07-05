# Habitat Benchmark Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Habitat-Sim pathfinder/navmesh-based generator for VLM benchmark case JSON and RGB assets.

**Architecture:** Create `goal_adapter/habitat_benchmark_dataset.py` as a focused generator that reuses `goal_adapter.habitat_sim_smoke` simulator setup helpers and writes existing `vlm_benchmark` case JSON format. Tests use a fake Habitat-Sim module so default Python does not need Habitat installed.

**Tech Stack:** Python standard library, existing PNG writer, optional Habitat-Sim runtime, `unittest`.

---

### Task 1: Generator API

**Files:**
- Create: `goal_adapter/habitat_benchmark_dataset.py`
- Test: `tests/test_habitat_benchmark_dataset.py`

- [ ] **Step 1: Write failing test**

Test that `generate_habitat_benchmark_cases` with one fake scene and `cases_per_category=1` writes four JSON files plus RGB assets.

- [ ] **Step 2: Run test**

Run: `python3 -B -m unittest tests.test_habitat_benchmark_dataset`

Expected: fail because module does not exist.

- [ ] **Step 3: Implement generator**

Implement:

- `HabitatBenchmarkConfig`
- `HabitatBenchmarkDataset`
- `generate_habitat_benchmark_cases`
- CLI `python -m goal_adapter.habitat_benchmark_dataset`

- [ ] **Step 4: Run test**

Run: `python3 -B -m unittest tests.test_habitat_benchmark_dataset`

Expected: pass.

### Task 2: Integration Verification

**Files:**
- Existing benchmark modules only.

- [ ] **Step 1: Run all unit tests**

Run: `python3 -B -m unittest discover -s tests`

Expected: all tests pass.

- [ ] **Step 2: Generate small real Habitat dataset**

Run the generator in the SRM conda environment against one HM3D scene with `--cases-per-category 1`.

- [ ] **Step 3: Load/run benchmark on generated cases**

Run the VLM benchmark on the generated small dataset and verify report/overlay files are written.
