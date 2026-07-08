# Farthest Waypoint Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run the Step B VLM-to-PixelNav bridge report with a prompt policy that prefers the farthest visible navigable local subgoal.

**Architecture:** Keep the Habitat multiview audit pipeline intact and add a small policy switch to prompt construction. Reuse the existing PixelNav bridge to visualize the new selected point as `goal_image + goal_mask`.

**Tech Stack:** Python 3.9 habitat env, Habitat-Sim, Qwen OpenAI-compatible endpoint, NumPy, PIL/OpenCV, unittest.

## Global Constraints

- Preserve existing default multiview behavior unless `waypoint_policy="farthest_visible"` is explicitly selected.
- Reuse the same scene, robot pose, heading, and coarse goal from the prior Step B trial for fair comparison.
- Write a failing unittest before editing production code.
- Save visual and report artifacts under `/home/icra/voca-s2e/reports/habitat_pixelnav_integration/step_b_farthest_policy/`.

---

### Task 1: Prompt Policy Switch

**Files:**
- Modify: `/home/icra/voca-s2e/goal_adapter/habitat_multiview_goal_audit.py`
- Create: `/home/icra/voca-s2e/tests/test_habitat_multiview_goal_audit_policy.py`

**Interfaces:**
- Consumes: `MultiViewGoalAuditConfig.waypoint_policy`.
- Produces: prompt text that includes farthest visible navigable waypoint instructions only when requested.

- [ ] **Step 1: Write failing test**

```python
def test_farthest_visible_policy_prompt_prioritizes_far_free_space():
    config = MultiViewGoalAuditConfig(scene_path="scene.glb", output_dir="out", waypoint_policy="farthest_visible")
    prompt = _build_multiview_prompt(config, [0.0, 0.0], [5.0, 0.0])
    assert "farthest visible navigable floor point" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/icra/voca-s2e /home/icra/micromamba/bin/micromamba run -r /home/icra/micromamba-root -n habitat python /home/icra/voca-s2e/tests/test_habitat_multiview_goal_audit_policy.py`
Expected: FAIL because `_build_multiview_prompt` or `waypoint_policy` does not exist.

- [ ] **Step 3: Implement minimal prompt builder**

Add `waypoint_policy` to `MultiViewGoalAuditConfig`, add `_build_multiview_prompt`, and call it from `_query_qwen_for_multiview_point`.

- [ ] **Step 4: Run tests**

Run the policy test and the existing pixelnav bridge test. Both should pass.

### Task 2: Re-run Same Trial With Farthest Policy

**Files:**
- Read: `/home/icra/voca-s2e/reports/habitat_goal_audit/multiview_random10/trial_004_skokloster-castle/multiview_goal_audit_result.json`
- Output: `/home/icra/voca-s2e/reports/habitat_pixelnav_integration/step_b_farthest_policy/farthest_goal_audit_result.json`

**Interfaces:**
- Consumes: existing result JSON scene/pose/heading/coarse goal.
- Produces: new VLM selected view/point and layout under the farthest policy.

- [ ] **Step 1: Run `run_multiview_goal_audit` with `waypoint_policy="farthest_visible"`**
- [ ] **Step 2: Generate PixelNav bridge visual check for the new point**
- [ ] **Step 3: Generate a markdown report comparing old and new results**
