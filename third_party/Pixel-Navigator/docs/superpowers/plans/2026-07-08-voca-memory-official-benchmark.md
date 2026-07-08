# VOCA Memory Official Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Pixel-Navigator-side official Habitat benchmark runner that executes Habitat episodes while using VOCA/Qwen grounded goals, PixelNav local rollout, and v5 memory graph logging.

**Architecture:** Keep Pixel-Navigator as the benchmark entrypoint. Add a new runner module that wraps a `habitat.Env` as a `RobotBackend`, delegates VLM/memory decisions to `qwen_nav_memory_framework_v5`, delegates low-level motion to PixelNav `Policy_Agent`, and records official Habitat metrics plus memory artifacts.

**Tech Stack:** Python, Habitat-Lab `habitat.Env`, Pixel-Navigator `Policy_Agent`, VOCA `nav_memory_qwen`, pytest.

## Global Constraints

- Do not modify existing `objnav_benchmark.py` or `evaluate_policy.py` behavior.
- Keep VOCA imports lazy so unit tests do not require Habitat, torch, or Qwen.
- Official metrics must come from `habitat_env.get_metrics()` when Habitat is available.
- Save memory graph artifacts per episode and aggregate benchmark summary.
- Support ObjNav first because Pixel-Navigator already has an official ObjNav benchmark loop.

---

### Task 1: Importable Runner and Fake-Env Contract

**Files:**
- Create: `voca_memory_benchmark.py`
- Create: `tests/test_voca_memory_benchmark.py`

**Interfaces:**
- Produces: `HabitatEnvMemoryBackend`, `run_objnav_memory_benchmark`, `write_benchmark_summary`

- [x] Write failing tests for fake-env backend stepping and benchmark summary.
- [x] Implement lazy imports, backend state/observation/rotate/waypoint execution, and summary writing.
- [x] Verify tests pass without Habitat/Qwen.

### Task 2: CLI Entry Point

**Files:**
- Modify: `voca_memory_benchmark.py`

**Interfaces:**
- Produces: `python voca_memory_benchmark.py --dataset hm3d --eval-episodes N --out DIR`

- [x] Add CLI options matching Pixel-Navigator conventions.
- [x] Wire `hm3d_config`/`mp3d_config`, `Policy_Agent`, Qwen client, and `NavMemoryAgent`.
- [x] Keep `--dry-run-fake-env` for smoke testing without assets.

### Task 3: Verification and Report

**Files:**
- Create: `reports/voca_memory_official_benchmark_readiness.md`

**Interfaces:**
- Consumes: pytest results and dry-run fake-env output.

- [x] Run unit tests.
- [x] Run fake-env dry-run.
- [x] Document exact command for official ObjNav run and remaining asset/Qwen requirements.
