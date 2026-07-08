# Habitat PixelNav Rollout Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute a short PixelNav action rollout inside Habitat-Sim from a VLM-selected fine goal point.

**Architecture:** Reuse Step B audit JSON as input. Convert the selected point to PixelNav `goal_image + goal_mask`, align the simulator camera to `selected_view`, run PixelNav actions for a small horizon, and save action logs plus visual rollout artifacts.

**Tech Stack:** Python 3.9 habitat env, Habitat-Sim, Pixel-Navigator `Policy_Agent`, NumPy, PIL/OpenCV, unittest.

## Global Constraints

- Keep Pixel-Navigator source unchanged.
- Use Habitat-Sim GPU rendering.
- Match PixelNav action scale: `move_forward=0.25m`, `turn_left/right=30deg`.
- Stop safely if PixelNav emits unsupported low-level actions in low-level Habitat-Sim.
- Write failing unittest before production code.

---

### Task 1: Testable Rollout Core

**Files:**
- Create: `/home/icra/voca-s2e/goal_adapter/habitat_pixelnav_execute.py`
- Create: `/home/icra/voca-s2e/tests/test_habitat_pixelnav_execute.py`

**Interfaces:**
- Consumes: executor with `step(rgb, collide)`, runner with `rgb()`, `position_xyz()`, `step(action_name)`.
- Produces: `PixelNavRolloutSummary` with step log, total movement, collision count, stop/unsupported status.

### Task 2: Habitat Runtime Entry Point

**Files:**
- Modify: `/home/icra/voca-s2e/goal_adapter/habitat_pixelnav_execute.py`

**Interfaces:**
- Consumes: Step B audit result JSON.
- Produces: `pixelnav_rollout_result.json`, `rollout_contact_sheet.png`, overlay frames.

### Task 3: Single Visual Rollout

**Command:**
- Run Step C on `/home/icra/voca-s2e/reports/habitat_pixelnav_integration/step_b_farthest_policy/farthest_goal_audit_result.json`

**Output:**
- `/home/icra/voca-s2e/reports/habitat_pixelnav_integration/step_c_rollout_single/`
