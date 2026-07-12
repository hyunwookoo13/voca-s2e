# VLM Grounded Goal Prompt Policy

## 1. One-line Summary

본 문서는 VLM grounded goal 아이디어를 실제 구현 가능한 prompt policy로 정의한다. 목표는 coarse GPS/map goal, robot state, RGB observation, memory context를 입력으로 받아, VLM이 PixelNav/S2E가 실행할 수 있는 local fine waypoint 또는 필요한 interrupt action을 안정적으로 출력하도록 만드는 것이다.

핵심 질문은 다음과 같다.

```text
coarse goal이 현재 RGB에서 직접 보이지 않아도,
VLM은 robot pose, goal bearing, multi-view observation, memory graph context를 이용해
goal 방향으로 이어질 가능성이 높은 visible navigable fine point를 선택할 수 있는가?
```

여기서 VLM은 low-level controller가 아니다. VLM은 느린 semantic supervisor / goal adapter이며, 실제 이동은 PixelNav/S2E-style fast navigation module이 수행한다.

---

## 2. Definition of VLM Grounded Goal

VLM grounded goal은 coarse GPS/map goal을 현재 robot observation에서 실행 가능한 local goal로 바꾸는 중간 표현이다.

```text
Coarse goal:
  map/global frame에서 주어진 먼 목표점
  현재 RGB에서 보이지 않을 수 있음

Grounded fine goal:
  현재 또는 추가 관측 RGB 안에서 선택한 visible navigable image point
  PixelNav/S2E가 local waypoint로 사용할 수 있음

VLM grounded goal:
  coarse goal direction + RGB semantics + memory context를 결합하여
  지금 이동해야 할 local fine point를 선택하는 과정
```

즉, VLM의 역할은 "목표 GPS 자체를 보는 것"이 아니라, 현재 관측 안에서 그 목표로 이어질 가능성이 높은 floor, doorway, corridor entry, safe branch를 고르는 것이다.

---

## 3. Prompt Input Sequence

VLM input은 `nav_vlm_waypoint_v1` envelope를 사용한다.

```text
1. task
   task_mode
   instruction
   coarse_goal
   target_object

2. coordinate_frame
   map_frame
   robot_pose_source
   heading_source
   pose_noise

3. robot_state
   position_xyz
   map_xy
   heading_rad

4. observation
   mode
   sequence_id
   timestamp_ms
   frame_index
   image_width / image_height
   views

5. memory
   nav_memory_context_v5

6. constraints
   allowed_actions
   allowed_observation_modes
   allowed_views
   allowed_step_deg
   pixel_coordinate_rule
```

이 순서는 의도적으로 top-down이다. 먼저 task와 coarse goal을 알려주고, 그 다음 robot pose와 observation을 주며, 마지막으로 memory와 constraints를 제공한다. VLM은 이 구조를 통해 "무엇을 해야 하는가", "현재 어디에 있는가", "무엇을 보고 있는가", "과거에 무엇을 했는가", "무엇을 출력할 수 있는가"를 순서대로 해석한다.

---

## 4. Spatio-temporal Prompt Binding

VLM prompt는 이미지 여러 장과 텍스트 context를 섞어서 사용한다. 각 view는 numeric `view_id`와 semantic `view_type`을 동시에 가진다.

```json
{
  "view_id": 0,
  "view_type": "front",
  "timestamp_ms": 391245,
  "relative_heading_deg": 0,
  "image": "<front_rgb>"
}
```

spatial binding은 다음 필드로 수행한다.

```text
relative_heading_deg:
  현재 robot heading 기준 각 view의 방향

coarse_goal.relative_bearing_deg:
  현재 robot heading 기준 coarse goal 방향

memory.local_topology.candidate_exits[*].bearing_deg_robot:
  memory graph의 candidate exit를 현재 robot frame 기준으로 변환한 방향

selected_view_id / selected_view_type:
  VLM이 선택한 view를 downstream parser가 명확히 식별하기 위한 출력
```

temporal binding은 다음 필드로 수행한다.

```text
sequence_id:
  같은 episode / sequence에 속하는 observation 묶음

frame_index:
  memory node, keyframe, action outcome과 연결되는 step index

timestamp_ms:
  observation과 keyframe의 시간 순서

memory.current_localization.last_seen_frame_index:
  과거 node가 얼마나 최근에 관측되었는지 판단
```

중요한 점은 모든 과거 frame을 VLM에 넣지 않는다는 것이다. 과거 정보는 `nav_memory_context_v5`로 압축하고, 필요한 경우에만 `retrieved_memory_images`로 제한된 keyframe 이미지를 첨부한다.

---

## 5. Required Memory Context

VLM prompt에 들어가는 memory는 전체 graph가 아니라 `build_vlm_memory_context()`가 만든 compact context이다.

필수 memory field는 다음과 같다.

```text
schema_version:
  nav_memory_context_v5

v6 ablation:
  nav_memory_context_v6
  candidate_refs / nav_skill_cards / policy_harness_state 추가
  go action은 가능하면 selected_candidate_ref를 함께 출력

current_pose_relation_to_latest_node:
  latest/current node 기준 현재 robot pose relation
  candidate exit를 current robot frame으로 해석하기 위해 필요

current_localization:
  현재 robot이 어떤 node 근처인지

place_recognition:
  backend retrieval이 제안한 revisit candidates
  VLM은 confirm/reject/merge request 가능

goal_context:
  goal bearing, goal distance, detour status

local_topology.candidate_exits:
  현재 node에서 선택 가능한 graph-aware direction 후보

deadlock_state:
  suspected / confirmed / escaping / escaped 상태

retrieved_memory_images:
  VLM이 비교할 수 있는 제한된 keyframe evidence

compressed_negative_memories:
  이미지 없이 요약으로만 유지하는 실패 memory
```

이 중 prompt decision에 가장 직접적인 필드는 `goal_context`, `candidate_exits`, `current_pose_relation_to_latest_node`, `deadlock_state`이다.

---

## 6. Decision Policy

VLM은 다음 우선순위로 decision을 내린다.

```text
1. Safety
   hazard, collision risk, blocked path, drop-off를 피한다.

2. Memory avoidance
   confirmed negative edge 또는 deadlock branch는 goal 방향과 맞더라도 피한다.

3. Goal alignment
   coarse_goal.relative_bearing_deg와 selected view / candidate exit bearing을 비교한다.

4. Grounded navigability
   selected_image_point는 visible navigable floor, doorway entry, corridor branch 위여야 한다.

5. Information sufficiency
   현재 view만으로 부족하면 guessing하지 않고 request_observation을 출력한다.

6. Recovery
   deadlock escape가 필요하면 일시적으로 goal 반대 방향 detour도 허용한다.

7. Resume
   escape 이후에는 goal bearing과 다시 정렬되는 candidate를 선택한다.
```

이 정책은 VLM이 "아무 바닥"을 찍는 것을 막고, coarse goal과 memory context에 의해 grounded된 fine point를 선택하도록 만든다.

---

## 7. Observation Policy

VLM은 항상 즉시 `go`를 출력하지 않는다. 정보가 부족하면 추가 관측을 요청할 수 있다.

```text
current_only:
  현재 입력된 view만 사용

directed_view:
  특정 yaw 방향 하나만 추가 관측

directed_sweep:
  goal bearing 또는 불확실한 branch 주변을 여러 장 관측

full_sweep:
  loop, deadlock, severe uncertainty 상황에서 360도 관측
```

선택 기준은 다음과 같다.

```text
go:
  visible navigable point가 있고, goal/memory 방향과 일치할 때

request_observation:
  goal이 current view 밖에 있거나, doorway/branch 판단이 애매할 때

rotate:
  controller가 front-only fine goal을 요구하거나, 현재 heading 자체를 바꿔야 할 때

stop:
  collision/hazard/task done/sensor failure 같은 interrupt 상황
```

`force_front_view_waypoint`가 켜져 있으면, VLM이 left/right/back view를 선택해도 backend가 먼저 rotate 후 re-observe하여 front-view fine goal을 다시 요청한다.

---

## 8. Controlled ReAct Policy

VLM 내부 reasoning은 자유로운 장문 chain-of-thought가 아니라 controlled ReAct 형태로 제한한다.

```text
OBSERVE:
  현재 RGB view, view direction, visible navigable floor를 확인한다.

LOCALIZE:
  robot heading, coarse goal bearing, current_pose_relation을 확인한다.

RECALL:
  candidate_exits, negative edges, deadlock_state, place_recognition 후보를 확인한다.

PLAN:
  safety, goal alignment, memory avoidance, exploration value를 함께 고려한다.

ACT:
  go / rotate / stop / request_observation 중 하나를 선택한다.

WRITE_MEMORY_REQUEST:
  필요하면 memory_ops를 제안한다.
  단, 실제 graph update는 backend verification 이후에만 수행된다.
```

최종 출력에는 hidden chain-of-thought를 쓰지 않는다. 대신 `reasoning.decision_reason`, `reasoning.goal_reason`, `reasoning.failure_mode`, `reasoning.short_text`를 사용한다.

---

## 9. Output Policy

VLM output은 반드시 `nav_vlm_waypoint_v1` JSON object 하나여야 한다.

### Case A. go

```json
{
  "schema_version": "nav_vlm_waypoint_v1",
  "action": "go",
  "selected_view_id": 1,
  "selected_view_type": "left",
  "selected_image_point": [320, 360],
  "fine_goal": {
    "valid": true,
    "view_id": 1,
    "view_type": "left",
    "point_px": [320, 360],
    "point_norm": [0.5, 0.75],
    "projected_map_xy": null,
    "navigability": "likely_free"
  },
  "observation_request": {
    "valid": false,
    "mode": null,
    "center_yaw_deg": null,
    "step_deg": null,
    "num_views": null,
    "yaw_offsets_deg": null,
    "reason": null
  },
  "reasoning": {
    "decision_reason": "G02_VISIBLE_FLOOR_TOWARD_GOAL",
    "goal_reason": "F02_VISIBLE_FLOOR_TOWARD_GOAL",
    "failure_mode": null,
    "short_text": "visible floor aligns with goal"
  },
  "control": {
    "vlm_control_mode": "resume_async_navigation",
    "rotate_yaw_deg": 0,
    "ttl_ms": 1000
  },
  "confidence": "high"
}
```

### Case B. request_observation

```json
{
  "schema_version": "nav_vlm_waypoint_v1",
  "action": "request_observation",
  "selected_view_id": null,
  "selected_view_type": null,
  "selected_image_point": null,
  "fine_goal": {
    "valid": false,
    "view_id": null,
    "view_type": null,
    "point_px": null,
    "point_norm": null,
    "projected_map_xy": null,
    "navigability": "unknown"
  },
  "observation_request": {
    "valid": true,
    "mode": "directed_sweep",
    "center_yaw_deg": -90,
    "step_deg": 30,
    "num_views": 3,
    "yaw_offsets_deg": [-120, -90, -60],
    "reason": "goal_bearing_left_but_not_visible"
  },
  "reasoning": {
    "decision_reason": "R01_GOAL_OUTSIDE_CURRENT_VIEW",
    "goal_reason": "F08_NONE_ROTATE_OR_STOP",
    "failure_mode": "insufficient_visual_context",
    "short_text": "need left-side observations"
  },
  "control": {
    "vlm_control_mode": "pause_for_observation",
    "rotate_yaw_deg": 0,
    "ttl_ms": 1000
  },
  "confidence": "medium"
}
```

### Optional memory_ops

VLM은 memory update를 직접 수행하지 않는다. 필요하면 `memory_ops`를 제안한다.

```json
{
  "memory_ops": [
    {
      "op": "confirm_revisit_node",
      "node_id": "n_00002",
      "confidence": 0.82,
      "reason": "same doorway layout and corridor geometry"
    }
  ]
}
```

backend는 confidence, retrieval score, topology consistency, critical memory conflict를 검증한 뒤에만 graph를 수정한다.

---

## 10. System Prompt Policy

구현 기준 system prompt는 `qwen_nav_memory_framework_v5/nav_memory_qwen/prompts.py`의 정책을 따른다.

핵심 내용은 다음과 같다.

```text
You are a semantic navigation supervisor for a robot.
You are not a low-level controller.
Your job is to convert a coarse navigation objective and current RGB observations into:
1. a safe local fine waypoint for S2E / PixelNav-style navigation, or
2. rotate / request_observation / stop.

Return only one valid JSON object matching schema_version nav_vlm_waypoint_v1.
Do not output hidden chain-of-thought.
Use controlled reason codes instead of free-form long reasoning.
```

필수 정책은 다음과 같다.

```text
Safety first:
  avoid hazards, collisions, blocked paths, negative memory edges

Memory-aware:
  avoid confirmed failed/deadlock branches even if aligned with goal

Robot-frame spatial reasoning:
  use candidate_exits[*].bearing_deg_robot
  use relative_pose_robot_to_dst
  do not assume robot is at node origin

Observation-aware:
  if information is insufficient, request_observation rather than guessing

Place recognition:
  confirm same place only when stable layout cues match
  false merge is worse than missed merge

Deadlock:
  deadlock is directional, not room-level
```

---

## 11. User Prompt Policy

User prompt는 VLM input JSON과 attached images를 연결한다.

```text
Given the following JSON input, choose the next navigation action.
Images are attached as multimodal inputs and referenced in JSON by placeholders.

Rules for output:
- Output JSON only.
- Use selected_image_point only when action is go and the point is on visible navigable floor.
- For rotate/request_observation/stop, fine_goal.valid must be false.
- Preserve schema_version: nav_vlm_waypoint_v1.
- Optional memory_ops are allowed, but only as requests; backend will verify them.
- Use memory.current_pose_relation_to_latest_node to understand robot offset from latest graph node.
- Use memory.local_topology.candidate_exits[*].bearing_deg_robot for robot-relative direction selection.
- If memory.place_recognition.revisit_candidates is non-empty, include memory_ops judgement when confident enough.

VLM_INPUT_JSON:
{vlm_input_json}
```

이미지는 JSON 안의 `<front_rgb>`, `<left_rgb>`, `<memory_image_id>` placeholder와 실제 multimodal attachment를 매칭한다.

---

## 12. Context Budget Policy

Qwen3-VL은 multi-image / interleaved context를 사용할 수 있지만, 구현에서는 long-context capacity에 의존하지 않는다. navigation loop에서는 latency와 JSON 안정성이 중요하기 때문이다.

따라서 prompt budget은 다음 원칙으로 제한한다.

```text
Current observation:
  current_only: 1 image
  directed_view: 1 additional image
  directed_sweep: usually 3 images
  full_sweep: 4~8 images depending on step_deg

Memory images:
  current node keyframe
  neighbor node keyframe
  negative memory keyframe
  revisit candidate keyframe
  max_memory_images로 hard cap

Text memory:
  full graph를 넣지 않음
  nav_memory_context_v5 compact context만 삽입

Reasoning:
  hidden chain-of-thought 금지
  controlled reason code와 short_text만 사용
```

32B Thinking model에서는 reasoning이 길어져 `message.content`가 비거나 JSON completion이 늦어질 수 있다. 따라서 VLM client는 `content fallback`, `reasoning fallback`, `JSON extraction`, `retry/request_observation fallback`을 가져야 한다.

---

## 13. Prompt-to-Backend Data Flow

실제 closed-loop에서는 다음 순서로 동작한다.

```text
1. HabitatPixelNavBackend.get_robot_state()
2. HabitatPixelNavBackend.get_observation()
3. NavMemoryAgent updates MemoryGraph
4. MemoryGraph.build_vlm_memory_context(nav_memory_context_v5)
5. build_vlm_input_v1(task, robot_state, observation, memory, constraints)
6. Qwen3-VL decides action JSON
7. Safety parser validates JSON
8. NavMemoryAgent applies verified memory_ops
9. Backend executes go / rotate / request_observation / stop
10. ActionOutcome updates MemoryGraph
```

이 flow에서 VLM prompt policy는 5~6번을 정의하지만, 7~10번의 backend validation을 전제로 설계된다. 즉 VLM output은 제안이고, 실제 graph mutation과 robot action은 backend safety layer를 통과해야 한다.

---

## 14. Initial Implementation Scope

이번 주 구현 범위는 다음과 같다.

```text
I1. Prompt policy 문서 확정
  nav_vlm_waypoint_v1
  nav_memory_context_v5
  controlled reason code
  observation request policy

I2. qwen_nav_memory_framework_v5 prompt와 정합성 확인
  SYSTEM_PROMPT
  USER_PROMPT_TEMPLATE
  memory_ops policy

I3. Habitat static audit prompt에 schema 반영
  기존 selected_view / selected_image_point prompt를
  nav_vlm_waypoint_v1 구조에 맞게 확장

I4. HabitatPixelNavBackend integration에서 사용
  VLM input 생성
  VLM output parsing
  ActionOutcome feedback
```

즉 이 문서는 integration 구현 전에 VLM이 어떤 정보를 어떤 순서로 읽고, 어떤 규칙으로 fine goal을 출력해야 하는지를 고정하는 기준 문서이다.

---

## 15. Report Summary

보고용 요약은 다음과 같다.

```text
VLM grounded goal은 coarse GPS/map goal을 현재 RGB에서 실행 가능한 local fine waypoint로 변환하는 중간 표현입니다.
VLM은 low-level controller가 아니라 semantic supervisor이며,
PixelNav/S2E가 실행할 수 있는 visible navigable point 또는 rotate/request_observation/stop을 출력합니다.

Prompt input은 nav_vlm_waypoint_v1 envelope를 사용하고,
task, coordinate frame, robot state, observation, nav_memory_context_v5, constraints 순서로 구성합니다.
Observation은 view_id, view_type, relative_heading_deg, timestamp를 갖고,
memory는 current_pose_relation_to_latest_node, candidate_exits, place_recognition, deadlock_state를 포함합니다.
v6 ablation에서는 여기에 candidate_refs, nav_skill_cards, policy_harness_state가 추가되며,
go 출력은 selected_candidate_ref를 통해 backend가 제안한 exit 후보와 연결됩니다.

VLM decision은 safety, memory avoidance, goal alignment, grounded navigability, information sufficiency 순서로 수행합니다.
특히 candidate_exits는 current robot frame 기준 bearing_deg_robot으로 제공되므로,
VLM은 latest node origin이 아니라 현재 robot pose 기준으로 방향을 선택합니다.

출력은 반드시 nav_vlm_waypoint_v1 JSON이며,
go일 때만 selected_image_point와 fine_goal.valid=true를 허용합니다.
정보가 부족하면 request_observation을 사용하고,
memory update는 memory_ops로 제안만 하며 backend verification 이후에만 반영합니다.

이 prompt policy를 기준으로 Habitat + PixelNav integration에서는
VLM selected point를 local action으로 실행하고,
ActionOutcome을 memory graph에 되먹이는 closed-loop 검증을 진행합니다.
```
