# qwen-nav-memory-framework v0.3

Relative-pose topo-metric episodic memory framework for a Qwen3-VL semantic navigation supervisor.

The framework keeps the original `nav_vlm_waypoint_v1` envelope:

```text
RGB observation + robot state + coarse goal + memory
    -> Qwen3-VL semantic supervisor
    -> go / rotate / stop / request_observation
    -> backend validation + fast local controller
    -> verified memory update
    -> repeat
```

The VLM is the slow semantic supervisor. It does not drive motors directly. It either selects a visible local fine waypoint for the fast S2E/PixelNav-style module or requests rotation / additional observation when the current view is insufficient.

## What v0.3 adds

```text
VLM-grounded place recognition
  backend retrieval proposes revisit candidates
  Qwen compares current image and memory images/summaries
  backend verifies confirm_revisit_node / request_merge_nodes before committing

Live latest-node pose relation
  the agent maintains runtime T_latest_node_to_current_robot
  this is not a stored global node pose
  graph exits are transformed from node frame into current robot frame
  new temporal edges use cumulative latest-node-to-robot pose, not only the last odom delta

Rotate-to-front fine-goal policy
  if Qwen selects left/right/back view and force_front_view_waypoint=True
  backend rotates first, obtains a new front observation, and asks Qwen again
  the fast module receives front-view fine goals only

Deadlock state machine
  none -> suspected -> confirmed -> escaping -> confirmed_escaped
  suspected: scan/rotate and gather evidence
  confirmed: mark directional incoming edge as negative
  escaped: preserve escape edge and resume goal-directed planning

ObjNav memory hook
  object_context and MemoryNode.semantic.object_belief
  VLM can update object belief through memory_ops

Pose graph optimization
  intentionally not enabled
  explicit TODO hook only
```

## Installation

```bash
cd qwen_nav_memory_framework
pip install -e .
python -m unittest discover -s tests -v
```

## Run local smoke test

```bash
python examples/run_mock_episode.py
```

This uses `StaticImageBackend` and `HeuristicVLMClient`. It validates the recursive loop, schema validation, memory graph construction, compression, and log saving without a real robot or Qwen endpoint.

## Run with Qwen/OpenAI-compatible endpoint

Set your endpoint:

```bash
export QWEN_BASE_URL="https://your-qwen-endpoint/v1"
export QWEN_API_KEY="..."
export QWEN_MODEL="qwen3-vl-32b-thinking"
```

Then:

```bash
python examples/run_qwen_openai_compatible.py \
  --image path/to/front_rgb.jpg \
  --goal-x 4.0 \
  --goal-y 0.0 \
  --max-steps 20
```

The example backend uses one static image and simulated motion. For real navigation, implement `RobotBackend` so each loop obtains a fresh observation and executes the selected local waypoint with your fast module.

## Real robot / Habitat integration

Implement this protocol:

```python
class MyBackend:
    def get_robot_state(self) -> RobotState:
        ...

    def get_observation(self) -> Observation:
        ...

    def execute_waypoint(self, *, view_type: str, view_id: int, point_px: tuple[int, int], ttl_ms: int) -> ActionOutcome:
        ...

    def rotate(self, yaw_deg: float) -> ActionOutcome:
        ...

    def capture_views(self, yaw_offsets_deg: Sequence[float], mode: str = "directed_sweep") -> Observation:
        ...
```

`execute_waypoint` should call your S2E/PixelNav-style module. It must return collision/progress/odometry information. Without action outcome and odometry, Qwen can still choose actions, but the framework cannot robustly form relative-pose edges or confirm deadlocks.

### Latest-node live pose relation

The graph still does not store global node poses. The agent keeps a transient robot-state anchor for the current/latest node and exposes this runtime relation in memory:

```json
"current_pose_relation_to_latest_node": {
  "pose_type": "live_relative_SE2_not_global_node_pose",
  "latest_node_to_robot": {"dx_m": 0.65, "dy_m": 0.0, "dyaw_deg": 0.0},
  "robot_to_latest_node": {"dx_m": -0.65, "dy_m": 0.0, "dyaw_deg": 0.0}
}
```

This floating transform lets the backend convert `edge.relative_pose_src_to_dst` from the latest node frame into the current robot frame before Qwen chooses a view or fine goal. When a new node is later committed, the temporal edge uses the cumulative latest-node-to-robot pose.

## Minimal PointNav code

```python
from nav_memory_qwen import (
    NavMemoryAgent, NavAgentConfig,
    StaticImageBackend, OpenAICompatibleVLMClient,
)

robot = StaticImageBackend("front_rgb.jpg", start_xy=(0.0, 0.0), start_heading_rad=0.0)
vlm = OpenAICompatibleVLMClient.from_env()
agent = NavMemoryAgent(
    robot=robot,
    vlm_client=vlm,
    config=NavAgentConfig(
        max_steps=50,
        force_front_view_waypoint=True,
        vlm_confirms_place_recognition=True,
    ),
)

result = agent.run_until_done(goal_map_xy=(6.418, 21.904))
agent.save_run("runs/episode_001")
print(result.summary())
```

## Minimal ObjNav code

```python
result = agent.run_until_done(target_object="mug", max_steps=50)
```

ObjNav is implemented as a memory and prompt hook. Production use should connect object detection, open-vocabulary detection, or Qwen object evidence to `memory_ops` such as `update_object_belief` or `mark_object_seen`.

## VLM-facing memory context

The top-level input remains `nav_vlm_waypoint_v1`, but `memory` is upgraded to `nav_memory_context_v4`:

```json
{
  "schema_version": "nav_memory_context_v4",
  "current_localization": {
    "final_place_recognition_policy": "VLM_verifies_backend_candidates_backend_commits"
  },
  "place_recognition": {
    "revisit_candidates": [
      {
        "candidate_node_id": "n_00001",
        "candidate_image_ref": "...",
        "visual_retrieval_score": 0.94,
        "semantic_summary": "hallway intersection",
        "negative_memory": null
      }
    ]
  },
  "local_topology": {
    "candidate_exits": [
      {
        "view_type_hint": "left",
        "status": "deadlock_entry",
        "avoid": true
      }
    ]
  },
  "deadlock_state": {
    "status": "suspected | confirmed | escaping | confirmed_escaped"
  },
  "object_context": {
    "enabled": true,
    "target_object": "mug"
  }
}
```

## Supported memory_ops

The VLM may request these operations. The backend verifies before applying topology-changing operations.

```text
confirm_revisit_node
reject_revisit_candidate
request_merge_nodes / merge_nodes
mark_incoming_edge
mark_deadlock
mark_blocked_edge
mark_escape_edge
compress_node
update_object_belief / mark_object_seen
```

## Why pose graph optimization is not included

This version keeps relative-pose edges and supports path composition. It does not solve global consistency after loop closure. `MemoryGraph.optimize_pose_graph()` is an explicit TODO hook and raises `NotImplementedError` if called. This keeps the current version centered on VLM-prompted reasoning, verified memory writes, and directional negative memory rather than algorithmic SLAM optimization.

## Test coverage

```text
test_relative_pose_composition
test_add_deadlock_edge
test_mock_episode_runs
test_vlm_confirmed_revisit_merges_provisional_node
test_context_exposes_place_recognition_candidates
test_live_pose_relation_updates_after_go_without_new_node
test_new_edge_uses_cumulative_latest_node_to_robot_pose
test_candidate_exit_bearing_uses_current_robot_frame
test_rotate_to_front_policy_prevents_non_front_waypoint_execution
test_deadlock_suspected_on_repeated_scan_request
```

## Important files

```text
nav_memory_qwen/agent.py          Observe -> Memory -> VLM -> Act loop
nav_memory_qwen/memory_graph.py   Relative-pose memory, revisit verification, deadlock state
nav_memory_qwen/prompts.py        Qwen prompt with place-recognition and deadlock policies
nav_memory_qwen/vlm_client.py     Qwen/OpenAI-compatible adapter and heuristic test client
nav_memory_qwen/robot_backend.py  Robot/simulator backend protocol
nav_memory_qwen/safety.py         VLM output validation and conservative repair
```
