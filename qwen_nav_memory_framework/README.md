# Qwen3-VL Relative-Pose Navigation Memory Framework

이 프로젝트는 **Qwen3-VL 32B dense thinking / reasoning 계열 VLM**을 느린 semantic supervisor로 두고, 실제 이동은 **S2E / PixelNav-style fast navigation skill** 또는 로봇 backend가 수행하는 구조의 Python framework입니다.

핵심 목표는 다음 세 시나리오입니다.

1. 로봇이 방/공간에 들어가 막혔을 때, rotate 또는 추가 observation으로 탈출구를 찾는다.
2. deadlock에서 탈출한 뒤, 같은 deadlock entry branch에 다시 들어가지 않는다.
3. deadlock 탈출을 위해 일시적으로 goal 반대 방향으로 움직여도, coarse goal을 잊지 않고 escape 이후 goal-directed planning으로 복귀한다.

본 framework는 기존 `nav_vlm_waypoint_v1` input/output envelope를 유지합니다. 단, 기존 schema의 `memory` 필드 내부에 graph-derived context를 확장해서 넣습니다. VLM output에는 기존 필드만 있어도 동작하고, optional `memory_ops`를 추가로 낼 수 있습니다. `memory_ops`는 backend가 검증한 뒤에만 반영합니다.

---

## Architecture

```text
RobotBackend / Simulator / ROS2
  ├─ robot_state: pose, heading
  ├─ observation: RGB views
  └─ action outcome: success, collision, odometry delta
          │
          ▼
NavMemoryAgent
  ├─ Visual place retrieval
  ├─ Relative-pose topo-metric memory graph
  ├─ Deadlock / negative edge memory
  ├─ Node compression / marginalization
  ├─ VLM input builder: nav_vlm_waypoint_v1
  ├─ VLM output sanitizer
  └─ Action execution through RobotBackend
          │
          ▼
Qwen3-VL supervisor
  ├─ go: select visible navigable fine waypoint
  ├─ rotate
  ├─ stop
  └─ request_observation: directed/full sweep
```

---

## What is implemented

### Memory graph

`nav_memory_qwen.memory_graph.MemoryGraph` implements:

- node: place/situation memory
- keyframe: image reference, embedding id, caption, storage tier
- directed edge: relative `SE(2)` pose from source node to destination node
- traversal outcome: success, blocked, deadlock entry, escape success
- negative edge index: fast lookup for failed branches
- graph context builder: compact VLM-facing memory
- node compression: raw active images leave the VLM prompt, but summaries, embeddings, relative constraints, and negative memories stay

No global node pose is stored. Subgoals can be reconstructed by composing edge transforms from the currently localized node.

### VLM clients

- `HeuristicVLMClient`: deterministic rule-based client for smoke tests.
- `OpenAICompatibleVLMClient`: generic Qwen/OpenAI-compatible multimodal chat-completions adapter.

### Robot backends

- `RobotBackend` protocol: implement this for Habitat, ROS2, Isaac Sim, real robot, or your S2E/PixelNav module.
- `StaticImageBackend`: runnable demo backend using one input image and simulated odometry.

---

## Install

```bash
cd qwen_nav_memory_framework
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Smoke test

```bash
python examples/run_mock_episode.py
python -m unittest discover -s tests
```

Expected output:

```text
Episode summary:
{'success': True/False, 'done': ..., 'steps': ..., 'final_distance_m': ..., 'num_nodes': ..., 'num_edges': ...}
```

The mock backend is intentionally simple. It verifies that schema construction, memory graph updates, VLM output validation, action execution, and log saving work end-to-end.

---

## Run with Qwen/OpenAI-compatible endpoint

Set your endpoint first:

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

---

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

`execute_waypoint` should call your S2E/PixelNav-style module. It must return collision/progress/odometry information. Without odometry/action outcome, the framework can still ask Qwen for decisions, but it cannot build reliable relative-pose edges or detect deadlock robustly.

---

## Minimal code

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
    config=NavAgentConfig(max_steps=50),
)

result = agent.run_until_done(goal_map_xy=(6.418, 21.904))
agent.save_run("runs/episode_001")
print(result.summary())
```

---

## Schema compatibility

The VLM input builder returns:

```json
{
  "schema_version": "nav_vlm_waypoint_v1",
  "task": {...},
  "coordinate_frame": {...},
  "robot_state": {...},
  "observation": {...},
  "memory": {...},
  "constraints": {...}
}
```

The top-level output accepted by the framework is the original v1 action schema:

```json
{
  "schema_version": "nav_vlm_waypoint_v1",
  "action": "go | rotate | stop | request_observation",
  "selected_view_id": 0,
  "selected_view_type": "front",
  "selected_image_point": [320, 360],
  "fine_goal": {...},
  "observation_request": {...},
  "reasoning": {...},
  "control": {...},
  "confidence": "medium"
}
```

Optional extension:

```json
{
  "memory_ops": [
    {
      "op": "mark_deadlock",
      "trigger": "possible_deadlock_state",
      "confidence": 0.72
    }
  ]
}
```

The framework treats these as requests only. Merge/remove operations are not automatically applied without project-specific verification.

---

## Memory lifecycle

```text
HOT:
  current/recent node, active keyframe image can be sent to VLM

WARM:
  can still retrieve image if relevant

COMPRESSED:
  no active image in VLM context
  keeps embedding, summary, edge constraints, negative memory

ARCHIVED / TOMBSTONE:
  hidden from VLM prompt and active planning
  retained for audit/merge history
```

Compression policy is implemented in `MemoryGraph.compress_old_nodes`.

---

## How the research ideas map to code

| Idea | Code location | Implementation |
|---|---|---|
| SPTM-style topological place graph and retrieval | `memory_graph.py`, `indexing.py` | keyframe embeddings, visual retrieval, topological adjacency |
| Lookahead / graph planning | `MemoryGraph.score_candidate_exit`, `dijkstra_path`, `compose_path_pose` | candidate exit scoring and relative-pose composition |
| Neural Topological SLAM-style semantic node + coarse geometry edge | `MemoryNode.semantic`, `MemoryEdge.relative_pose_src_to_dst` | node semantic summary and edge relative SE(2) |
| MapGPT-style VLM-facing map prompt | `build_vlm_memory_context` | compact local topology, negative memory, route hints |
| PRISM-TopoMap-style no global node pose | `RelativePose2D`, `MemoryGraph.compose_path_pose` | local relative constraints only |
| VLMaps-style semantic/object extension | `MemoryNode.semantic`, docs hooks | object belief can be added to node semantic field |
| SLAM marginalization analogy | `compress_old_nodes` | raw active image dropped; constraints and negative edges kept |
| ReAct without free-form chain-of-thought | `prompts.py` | structured Observe/Diagnose/Act/Write Memory via JSON reason codes |

---

## Safety notes

- This is a research framework, not a certified robot safety stack.
- Do not allow VLM output to directly drive motors. Always route through a collision-aware fast module.
- Keep hardware emergency stop, depth/proximity checks, speed limits, and obstacle avoidance outside the VLM.
- `sanitize_vlm_output` repairs invalid model JSON into conservative rotate/scan fallbacks.
- The VLM is never trusted to directly mutate the graph; memory operation requests are verified by backend outcomes.

---

## File layout

```text
nav_memory_qwen/
  agent.py          closed-loop orchestration
  memory_graph.py   relative-pose topo-metric graph
  indexing.py       lightweight visual retrieval index
  embedding.py      pluggable image embedding interface
  vlm_client.py     Qwen/OpenAI-compatible and heuristic VLM clients
  robot_backend.py  backend protocol and static demo backend
  schema.py         nav_vlm_waypoint_v1 builders and pose utilities
  safety.py         output validation and fallback repair
  prompts.py        structured Qwen navigation prompt
examples/
  run_mock_episode.py
  run_qwen_openai_compatible.py
tests/
  test_memory_graph.py
```
