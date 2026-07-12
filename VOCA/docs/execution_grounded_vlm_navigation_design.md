# Execution-Grounded VLM Navigation Design

## Research Claim

VOCA will treat PixelNav as a frozen local execution skill and Qwen-VL as an
execution-grounded semantic supervisor. Given an ObjectNav category and RGB
observations, the supervisor selects an executable image-space waypoint or
issues a bounded controller interrupt (`rotate`, `request_observation`, or
`stop`). A verified episodic graph records directional success, failure,
deadlock, escape, and target evidence so later decisions can avoid repeating
failed branches.

The intended standalone contribution is:

> Policy-conditioned pixel candidates plus execution-grounded directional
> graph memory and adaptive observation for coarse-goal detour and recovery.

The same supervisor contract must later accept S2E as a drop-in execution
backend without changing memory semantics or benchmark policy inputs.

## Non-Negotiable Benchmark Boundary

The online policy may consume only observations and state that a deployed
ObjectNav robot could receive under the declared Habitat sensor configuration.
Habitat measures such as `distance_to_goal`, `success`, `spl`, shortest-path
geometry, and top-down maps are evaluation-only. They may be written to audit
artifacts after an action, but they must never appear in a Qwen prompt, waypoint
candidate score, supervisor transition, or memory retrieval score.

Agent pose may be used only when the benchmark configuration explicitly
declares localization as an input. It is used for relative execution evidence
and graph geometry, never to derive hidden target geometry.

## Architecture

### 1. Policy-Safe Context Adapter

The adapter constructs the VLM input from the object category, target-context
priors, RGB views, policy-provided candidate points, recent execution feedback,
and retrieved memory. ObjectNav goal geometry is explicitly represented as
unavailable:

```json
{
  "task_mode": "ObjNav",
  "target_object": "chair",
  "goal_geometry_available": false,
  "goal_bearing_from_current_deg": null,
  "goal_distance_m": null
}
```

Evaluation metrics remain in a separate audit channel. A recursive leakage
test rejects forbidden metric names and known sentinel substitutions in every
serialized policy input.

### 2. Explicit Supervisor State Machine

The online supervisor has four states:

- `goal_seek`: select evidence-supported progress toward the object context.
- `escape_deadlock`: prioritize leaving a blocked sector; temporary motion away
  from the hidden goal is valid.
- `backtrack`: follow a previously verified reverse/escape edge.
- `verify_target`: collect evidence and decide whether Habitat `stop` is
  justified.

State transitions use observable execution evidence: translation, collision,
controller termination reason, visual novelty, repeated candidate/sector, and
verified memory events. Ground-truth goal-distance deltas are audit-only and do
not drive transitions.

### 3. Policy-Conditioned Candidate Interface

PixelNav first proposes a small set of executable image-space points from the
current RGB view or requested views. Each candidate has a stable reference,
pixel coordinate, view/yaw, policy feasibility, visible-floor evidence,
clearance/safety evidence, and source. Qwen must select a supplied candidate
reference for `go`; it may not invent an unchecked coordinate.

The initial candidate generator is RGB/policy-conditioned and does not expose
depth to Qwen. The backend may use PixelNav's own learned representation and
local controller guards. A deterministic emergency fallback is observation or
rotation, not a blind lower-center waypoint.

### 4. Execution-Grounded Progress

Every `go` yields two separate outcomes:

- `execution_success`: the local skill produced meaningful, collision-safe
  translation or reached its local waypoint.
- `strategic_progress`: the result advances the active supervisor state.

In `goal_seek`, repeated stationary/collision outcomes trigger recovery. In
`escape_deadlock` and `backtrack`, collision-safe translation and departure
from a failed sector count as progress even when evaluation-only goal distance
increases. Evaluation distance deltas are retained only under an explicitly
named audit field.

### 5. Verified Episodic Memory

Memory stores raw RGB keyframe references as evidence plus a relative
topometric graph. Writes are proposed by Qwen but committed by the backend only
after execution validation. Directional edges distinguish successful travel,
suspected failure, confirmed deadlock entry, and verified escape. Object
beliefs carry positive and negative visual evidence with timestamps and source
confidence.

Retrieval is coarse-to-fine: inexpensive place/semantic retrieval proposes a
small set, then Qwen verifies only those candidates. False-negative duplicate
nodes are preferred over false-positive merges. Controller-specific failures
are labeled separately so changing PixelNav to S2E does not poison semantic
memory.

### 6. Adaptive Observation and Stop Verification

`request_observation` is an information action with a declared reason and
bounded yaw set. It is triggered by candidate ambiguity, target verification,
place-recognition uncertainty, or deadlock diagnosis, rather than by a fixed
360-degree ritual.

`stop` enters `verify_target` first. A stop is committed only when current RGB
evidence supports the target category and the local execution state is stable.
The benchmark's success signal is never read to approve the stop.

## Data and Audit Contract

For each supervisor decision, artifacts record the policy-safe input hash,
action, selected candidate, controlled reason code, confidence, active state,
retrieved memory references, and validation result. For each execution, they
record controller actions, motion/collision evidence, termination reason,
state transition, memory mutation, and evaluation metrics in a separate audit
section.

`fps.mp4` shows the RGB trajectory and selected point on the left. The right
panel shows action, supervisor state, concise reason, confidence, validation,
and memory events. `metric.mp4` remains an evaluation-only visualization and is
never fed back to policy.

## Evaluation

The primary benchmark uses official HM3D ObjectNav episode definitions and
reports SR, SPL, SoftSPL, path length, collisions, VLM calls, wall-clock time,
and stop precision/recall. A curated stress suite separately measures doorway
detours, opposite-direction room exits, dead-end escape, repeated-branch
avoidance, and target re-verification.

Required ablations are: frozen PixelNav alone, VLM without memory, flat recent
history, positive graph memory, graph plus directional negative edges, adaptive
observation, stop verification, and the full method. Seeds and episode IDs are
fixed across variants. Development smoke episodes cannot be reported as final
evidence.

## Implementation Phases

1. Enforce benchmark-valid policy inputs and detour-aware progress semantics.
2. Add typed PixelNav candidate generation and mandatory candidate selection.
3. Implement the complete supervisor state machine and bounded observations.
4. Close the execution-verified v6 memory write/retrieval loop.
5. Add target-stop verification and controller-independent failure taxonomy.
6. Run multi-seed ablations, stress tests, and video/audit validation.
7. Replace PixelNav with S2E through the unchanged executor interface.

## Acceptance Criteria

- Serialized Qwen inputs contain no evaluation-only fields or derived target
  geometry.
- A detour can be execution-successful without being mislabeled as goal
  failure.
- Every `go` references a backend-validated candidate before full evaluation.
- Memory writes are traceable to observable evidence and validation outcomes.
- Stop and deadlock recovery have explicit states and measurable error rates.
- Multi-episode runs are reproducible, complete, and produce synchronized
  videos plus machine-readable audit artifacts.
