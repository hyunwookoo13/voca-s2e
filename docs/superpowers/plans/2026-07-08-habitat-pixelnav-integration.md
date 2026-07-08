# Habitat PixelNav Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect VLM-selected fine image points to PixelNav-style local execution in Habitat, with visual checkpoints after each stage.

**Architecture:** Keep VLM semantic decision, PixelNav execution, and memory update as separate layers. The first bridge converts a selected RGB pixel into PixelNav's expected `goal_image` and `goal_mask`; later stages run PixelNav actions in Habitat and log an `ActionOutcome` that can update memory.

**Tech Stack:** Python 3.9 habitat env, Habitat-Sim/Lab, Pixel-Navigator `Policy_Agent`, NumPy, OpenCV, unittest/pytest.

## Global Constraints

- Work in the existing `/home/icra/voca-s2e` workspace so the user can inspect outputs directly from the IDE.
- Do not commit unless the user explicitly asks.
- Keep integration code thin and reversible; do not modify Pixel-Navigator source unless required.
- Use GPU when loading PixelNav if CUDA is available.
- Tests must be written and observed failing before production code is added.
- Each development stage must produce either a test result or a visual artifact the user can inspect.

---

### Task 1: VLM Point To PixelNav Goal Mask

**Files:**
- Create: `/home/icra/voca-s2e/goal_adapter/pixelnav_bridge.py`
- Create: `/home/icra/voca-s2e/tests/test_pixelnav_bridge.py`

**Interfaces:**
- Consumes: RGB image as `np.ndarray` with shape `(H, W, 3)`, selected point `[u, v]`.
- Produces: `PixelNavGoal(goal_image: np.ndarray, goal_mask: np.ndarray, selected_point_uv: tuple[int, int])`.

- [ ] **Step 1: Write failing tests**

```python
def test_make_pixelnav_goal_creates_mask_centered_on_selected_point():
    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    goal = make_pixelnav_goal(rgb, [5, 4], radius=1)
    assert goal.goal_mask.dtype == np.uint8
    assert goal.goal_mask.shape == (8, 10)
    assert goal.goal_mask[3:6, 4:7].min() == 255
    assert goal.selected_point_uv == (5, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/icra/voca-s2e pytest /home/icra/voca-s2e/tests/test_pixelnav_bridge.py -q`
Expected: FAIL because `goal_adapter.pixelnav_bridge` does not exist.

- [ ] **Step 3: Write minimal implementation**

Implement `PixelNavGoal`, `clamp_pixel_point`, `make_goal_mask`, and `make_pixelnav_goal`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/icra/voca-s2e pytest /home/icra/voca-s2e/tests/test_pixelnav_bridge.py -q`
Expected: PASS.

### Task 2: PixelNav Policy Runtime Wrapper

**Files:**
- Modify: `/home/icra/voca-s2e/goal_adapter/pixelnav_bridge.py`
- Modify: `/home/icra/voca-s2e/tests/test_pixelnav_bridge.py`

**Interfaces:**
- Consumes: Pixel-Navigator repo path and checkpoint path.
- Produces: lazy `PixelNavPolicyExecutor` that calls `Policy_Agent.reset(goal_image, goal_mask)` and `Policy_Agent.step(obs_rgb, collide)`.

- [ ] **Step 1: Write failing tests with a fake policy class**

```python
class FakePolicy:
    def __init__(self):
        self.reset_args = None
    def reset(self, goal_image, goal_mask):
        self.reset_args = (goal_image, goal_mask)
    def step(self, obs_rgb, collide=False):
        return 1, obs_rgb
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/icra/voca-s2e pytest /home/icra/voca-s2e/tests/test_pixelnav_bridge.py -q`
Expected: FAIL because `PixelNavPolicyExecutor` does not exist.

- [ ] **Step 3: Write minimal implementation**

Add a wrapper that accepts an injected policy for tests and lazily imports `Policy_Agent` for runtime.

- [ ] **Step 4: Run unit tests and a checkpoint smoke test**

Run: `PYTHONPATH=/home/icra/voca-s2e pytest /home/icra/voca-s2e/tests/test_pixelnav_bridge.py -q`
Run: `/home/icra/micromamba/bin/micromamba run -r /home/icra/micromamba-root -n habitat python -m goal_adapter.pixelnav_bridge --smoke`
Expected: unit tests PASS and smoke prints one valid action id.

### Task 3: Habitat Local Execution Harness

**Files:**
- Create: `/home/icra/voca-s2e/goal_adapter/habitat_pixelnav_execute.py`
- Create: `/home/icra/voca-s2e/tests/test_habitat_pixelnav_execute.py`

**Interfaces:**
- Consumes: an existing VLM audit result containing selected view, selected point, RGB path, and scene metadata.
- Produces: one local PixelNav rollout with action log, overlay frames, and a contact sheet.

- [ ] **Step 1: Write tests around fake env and fake executor**

Validate that the harness resets PixelNav with a mask, runs at most `max_steps`, records collisions, and stops on action `0`.

- [ ] **Step 2: Implement harness**

Implement fake-testable control flow first, then add Habitat runtime loading.

- [ ] **Step 3: Run one visual checkpoint**

Run a single scene/trial and save output under `/home/icra/voca-s2e/reports/habitat_pixelnav_integration/step_c_single/`.

### Task 4: Memory ActionOutcome Hook

**Files:**
- Modify: `/home/icra/voca-s2e/goal_adapter/habitat_pixelnav_execute.py`
- Create: `/home/icra/voca-s2e/tests/test_pixelnav_action_outcome.py`

**Interfaces:**
- Consumes: rollout log from Task 3.
- Produces: `ActionOutcome`-style summary with `moved_distance_m`, `collision`, `no_progress`, `odom_delta`, and `local_waypoint_reached`.

- [ ] **Step 1: Write failing outcome tests**

Cover success, collision/no-progress, and stop-before-progress cases.

- [ ] **Step 2: Implement summary builder**

Keep it independent from Habitat so it can be tested with synthetic poses/actions.

- [ ] **Step 3: Insert into memory framework later**

Only after the summary is correct, pass it to `qwen_nav_memory_framework_v3.NavMemoryAgent`.
