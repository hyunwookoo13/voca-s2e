# Habitat + PixelNav Verification Scenario and Metric Proposal

## 1. One-line Summary

본 문서는 VLM grounded goal과 memory를 Habitat + PixelNav/S2E에 연결하기 전에, 어떤 시나리오에서 무엇을 검증하고 어떤 metric으로 성공 여부를 판단할지 정의한다.

핵심 검증 질문은 다음과 같다.

```text
현재 robot pose, heading, coarse goal, RGB observation, memory context가 주어졌을 때,
VLM은 PixelNav/S2E가 실행 가능한 local fine waypoint를 제안하고,
그 실행 결과를 memory graph에 안정적으로 되먹일 수 있는가?
```

여기서 VLM은 low-level controller가 아니다. VLM은 semantic supervisor / goal adapter로 동작하며, 실제 이동은 PixelNav/S2E-style fast navigation module이 수행한다.

---

## 2. Current Validation Status

현재까지 확인된 부분은 다음과 같다.

```text
Habitat static audit:
  Habitat scene에서 robot RGB, topdown map, coarse goal, VLM fine point layout 생성 완료

Single-view random10:
  10/10 trial에서 front RGB 기준 selected_image_point 생성

Multi-view random10:
  10/10 trial에서 front/left/right/back 중 selected_view와 selected_image_point 생성
  selected_view 분포: front 3, back 3, right 2, left 2

Qwen3-VL-32B-Thinking random5:
  4/5 trial에서 valid selected_view/point 생성
  1개 trial은 long reasoning / JSON completion issue로 실패

Memory framework v5:
  qwen_nav_memory_framework_v5 unittest 통과
  place recognition / merge / negative edge / live latest-node pose relation 확인
```

즉, 현재 완료된 것은 **static VLM grounded fine goal feasibility**와 **memory framework standalone feasibility**이다. 아직 완료되지 않은 것은 PixelNav/S2E가 VLM point를 실제로 실행하고, 그 outcome을 memory로 되먹이는 closed-loop integration이다.

---

## 3. Verification Scope

이번 검증은 최종 성능 논문 실험이 아니라, Habitat + PixelNav integration 전후로 시스템이 닫힌 loop로 동작할 수 있는지 확인하는 feasibility study이다.

검증 범위는 네 단계로 나눈다.

```text
Level 0. Static VLM audit
  Habitat RGB/topdown에서 VLM이 fine point를 생성하는지 확인

Level 1. PixelNav local execution
  VLM selected point를 PixelNav/S2E local waypoint로 변환하고 실행 가능한지 확인

Level 2. ActionOutcome logging
  moved_distance_m, collision, no_progress, odom_delta를 수집할 수 있는지 확인

Level 3. Memory closed loop
  ActionOutcome 기반으로 node/edge/negative memory를 갱신하고,
  다음 VLM prompt에 nav_memory_context_v5를 삽입할 수 있는지 확인
```

이번 주의 최소 목표는 Level 2까지 안정적으로 만들고, Level 3의 memory context 삽입을 최소 시나리오에서 확인하는 것이다.

---

## 4. System Under Test

검증 대상 pipeline은 다음과 같다.

```text
Habitat Simulator
  -> RobotState
  -> RGB Observation
  -> coarse goal

NavMemoryAgent
  -> MemoryGraph update
  -> build_vlm_memory_context(nav_memory_context_v5)
  -> nav_vlm_waypoint_v1 input 생성

Qwen3-VL Supervisor
  -> go / rotate / stop / request_observation
  -> selected_view_id/type
  -> selected_image_point
  -> optional memory_ops

HabitatPixelNavBackend
  -> selected image point를 PixelNav/S2E local goal로 변환
  -> action 실행
  -> ActionOutcome 반환

MemoryGraph
  -> edge success/failure update
  -> negative edge / deadlock / escape edge 기록
  -> 다음 VLM prompt에 compact memory context 제공
```

`HabitatPixelNavBackend`가 이번 integration의 핵심 adapter이다.

```python
class HabitatPixelNavBackend:
    def get_robot_state(self):
        # Habitat pose / GPS-map aligned pose / heading -> RobotState
        ...

    def get_observation(self):
        # current RGB or multi-view RGB -> Observation
        ...

    def execute_waypoint(self, *, view_type, view_id, point_px, ttl_ms):
        # VLM image point -> PixelNav/S2E local waypoint -> ActionOutcome
        ...

    def rotate(self, yaw_deg):
        # Habitat agent yaw rotation -> ActionOutcome
        ...

    def capture_views(self, yaw_offsets_deg, mode="directed_sweep"):
        # VLM requested observation -> multi-view Observation
        ...
```

---

## 5. Verification Scenarios

### S0. Static Grounded Goal Sanity

목적은 VLM이 Habitat RGB에서 visible navigable floor point를 생성할 수 있는지 확인하는 것이다.

```text
Input:
  current RGB or multi-view RGB
  robot pose / heading
  coarse goal bearing / distance

Expected:
  valid selected_image_point
  selected point가 visible floor 또는 doorway/corridor entry에 위치
  topdown + RGB overlay로 사람이 확인 가능

Current status:
  single-view random10 완료
  multi-view random10 완료
  thinking32b random5 일부 완료
```

이 시나리오는 이미 static feasibility 확인에 사용되었다.

### S1. Visible Goal Direction Local Move

목적은 coarse goal 방향과 현재 view가 대체로 일치할 때, VLM point를 PixelNav/S2E가 실제 local move로 실행할 수 있는지 확인하는 것이다.

```text
Condition:
  goal bearing이 front 또는 selected view와 가까움
  selected point가 visible floor 위에 있음

Expected:
  action = go
  PixelNav/S2E execution success
  moved_distance_m > threshold
  collision = false
  distance_to_coarse_goal 감소
```

이 시나리오는 integration의 첫 성공 기준이다.

### S2. Coarse Goal Not Visible, Multi-view Selection

목적은 coarse GPS goal이 현재 front RGB에 보이지 않을 때, VLM이 아무 바닥이 아니라 goal 방향으로 이어질 가능성이 높은 view/point를 고르는지 확인하는 것이다.

```text
Condition:
  coarse goal이 벽/복도/방문 뒤에 있음
  front-only로는 판단이 애매함
  multi-view 또는 request_observation 허용

Expected:
  selected_view_type이 goal bearing과 대체로 일치
  selected point가 selected view의 navigable floor 또는 doorway/corridor entry에 있음
  PixelNav/S2E 실행 후 goal progress가 양수
```

이 시나리오는 VLM grounded goal의 핵심 검증이다.

### S3. Dead-end / No-progress Detection

목적은 VLM point가 local controller 관점에서 실패했을 때, memory가 실패 branch를 negative edge로 기록할 수 있는지 확인하는 것이다.

```text
Condition:
  VLM selected point 실행 후 collision 또는 no_progress 발생

Expected:
  ActionOutcome.no_progress = true 또는 collision = true
  current node deadlock_status가 suspected 또는 confirmed로 변경
  incoming edge traversal.status가 blocked/deadlock_entry로 기록
  다음 prompt의 compressed_negative_memories 또는 candidate_exits에 avoid 정보 포함
```

이 시나리오는 memory integration의 최소 필요 조건이다.

### S4. Branch Recovery / Backtracking

목적은 실패한 branch를 다시 선택하지 않고 다른 exit 또는 frontier를 선택하는지 확인하는 것이다.

```text
Condition:
  n1 분기점에서 branch A로 이동
  branch A에서 dead-end 또는 no_progress 발생
  n1 또는 근처 node로 돌아옴

Expected:
  branch A는 negative edge로 유지
  VLM이 branch A와 겹치는 selected_view를 피함
  candidate_exits 중 unvisited frontier 또는 escape edge를 선택
```

이 시나리오는 memory의 효과를 보여주는 핵심 사례이다.

### S5. Place Recognition / Revisit

목적은 현재 관측이 과거 node와 유사할 때, VLM이 revisit candidate를 보고 같은 장소 여부를 판단하고 backend가 보수적으로 merge하는지 확인하는 것이다.

```text
Condition:
  loop 또는 재방문 상황
  backend retrieval이 revisit_candidates 제공

Expected:
  VLM이 confirm_revisit_node 또는 reject_revisit_candidate 제안
  backend verification을 통과한 경우에만 commit_revisit 또는 merge_nodes 수행
  false merge 방지
```

이 시나리오는 v3 memory framework의 place recognition / merge 기능 검증이다.

---

## 6. Metrics

Metric은 VLM output, PixelNav execution, memory update, end-to-end navigation 네 층으로 나눈다.

### 6.1 VLM Output Metrics

```text
Schema Validity:
  VLM output이 nav_vlm_waypoint_v1 JSON으로 parse되는 비율

Action Validity:
  action이 go / rotate / stop / request_observation 중 하나인지

Point Validity:
  selected_image_point가 이미지 범위 안에 있는 비율

Visible Navigable Point Rate:
  selected point가 visible floor 또는 traversable region에 있는 비율

View Selection Alignment:
  selected_view_type의 relative heading이 coarse goal bearing 또는 candidate_exit bearing과 일치하는 정도

Observation Request Validity:
  request_observation의 mode, step_deg, yaw_offsets_deg가 constraints를 만족하는 비율

VLM Latency:
  request start부터 valid JSON 추출까지 걸린 시간
```

### 6.2 PixelNav / Local Execution Metrics

```text
Local Execution Success:
  selected point 실행 후 success=true인 비율

Moved Distance:
  moved_distance_m 평균 및 분포

Collision Rate:
  collision=true 비율

No-progress Rate:
  no_progress=true 비율

Local Waypoint Reachability:
  VLM selected point가 PixelNav/S2E가 실제로 접근 가능한 waypoint였는지

Execution TTL Failure:
  ttl_ms 안에 충분히 이동하지 못한 비율
```

### 6.3 Goal Progress Metrics

```text
Euclidean Goal Progress:
  before distance_to_goal - after distance_to_goal

Geodesic Goal Progress:
  Habitat shortest path distance 기준 progress

Progress Success Rate:
  one step 또는 N step 이후 goal distance가 감소한 비율

Path Efficiency:
  실제 이동 거리 대비 goal distance 감소량
```

초기 feasibility에서는 Euclidean progress를 우선 사용하고, 이후 navmesh shortest path를 안정적으로 얻으면 geodesic progress로 확장한다.

### 6.4 Memory Metrics

```text
Memory Context Validity:
  nav_memory_context_v5 생성 성공률

Candidate Exit Coverage:
  local_topology.candidate_exits가 현재 선택 가능한 방향을 포함하는 비율

Negative Edge Precision:
  실제 실패한 branch가 negative edge로 기록되는 비율

Deadlock Re-entry Rate:
  known negative edge 방향을 다시 선택하는 비율

Recovery Success:
  deadlock 이후 escape edge 또는 unvisited frontier로 빠져나오는 비율

Revisit Recognition Accuracy:
  revisit candidate를 VLM/backend가 올바르게 confirm/reject하는 비율

False Merge Rate:
  서로 다른 장소를 같은 node로 잘못 merge한 비율

Candidate Exit Bearing Error:
  node frame 기준 bearing과 current robot frame 보정 bearing의 차이 및 보정 후 정확도

v6 GaP-lite Validation:
  selected_candidate_ref가 memory.candidate_refs.exits에 존재하는 비율
  NAV_GO_001/NAV_GO_002/NAV_GO_003 failure rate
  request_observation recovery 이후 progress 회복률
```

### 6.5 End-to-End Episode Metrics

```text
Episode Success:
  goal threshold 안에 도달한 episode 비율

SPL-like Efficiency:
  shortest path 대비 실제 path length 효율

Step Count:
  episode당 VLM call 수, PixelNav action 수

Intervention Count:
  request_observation, rotate, stop 발생 횟수

Memory Size:
  node 수, edge 수, compressed node 수, retrieved image 수
```

End-to-end metric은 integration이 안정화된 뒤 사용한다. 이번 단계에서는 local execution과 memory update metric을 우선한다.

---

## 7. Logging Schema

각 step은 최소한 다음 정보를 저장한다.

```json
{
  "episode_id": "hm3d_scene_x_trial_0001",
  "step_index": 4,
  "scene_id": "apartment_1",
  "robot_state_before": {
    "map_xy": [0.216, 0.127],
    "heading_rad": 2.68
  },
  "coarse_goal": {
    "map_xy": [6.527, 6.565],
    "relative_bearing_deg": -109.1,
    "distance_m": 9.01
  },
  "memory_summary": {
    "schema_version": "nav_memory_context_v5",
    "num_nodes": 3,
    "num_edges": 2,
    "num_deadlock_edges": 1
  },
  "vlm_output": {
    "action": "go",
    "selected_view_type": "left",
    "selected_image_point": [300, 350],
    "confidence": "high"
  },
  "action_outcome": {
    "success": true,
    "moved_distance_m": 0.62,
    "collision": false,
    "no_progress": false,
    "odom_delta": {
      "dx_m": 0.62,
      "dy_m": 0.01,
      "dyaw_deg": 0.0
    }
  },
  "robot_state_after": {
    "map_xy": [0.81, 0.31],
    "heading_rad": 2.68
  },
  "metrics": {
    "euclidean_goal_progress_m": 0.44,
    "schema_valid": true,
    "point_in_image": true,
    "selected_point_executed": true
  }
}
```

이 step log가 있어야 나중에 ablation, failure analysis, qualitative figure를 한 번에 만들 수 있다.

---

## 8. Initial Pass / Fail Criteria

이번 주 integration feasibility 기준은 다음처럼 둔다.

```text
Static VLM audit:
  random scene에서 valid selected point 생성률 >= 80%

PixelNav local execution:
  selected point 실행 후 moved_distance_m > 0.2m인 step 비율 >= 60%
  collision rate <= 30%

ActionOutcome logging:
  success, moved_distance_m, collision, no_progress, odom_delta가 모든 step에서 기록

Memory context insertion:
  nav_memory_context_v5가 매 step VLM input에 포함
  current_pose_relation_to_latest_node가 valid 또는 명시적으로 invalid로 기록

v6 GaP-lite ablation:
  nav_memory_context_v6 사용 시 candidate_refs / nav_skill_cards가 VLM input에 포함
  steps.json에 validation_checkpoints / validation_feedback 기록
  Feedback.md에 rule pass/fail summary 기록

Negative memory sanity:
  collision/no_progress 발생 시 negative edge 또는 deadlock suspected가 기록

Report artifact:
  topdown + RGB + selected point + trajectory + memory event log 저장
```

이 기준은 최종 논문 성능 기준이 아니라, integration이 제대로 연결되었는지 확인하기 위한 engineering gate이다.

---

## 9. Expected Artifacts

각 scenario 실행 후 다음 artifact를 저장한다.

```text
Per-step:
  RGB observation
  selected point overlay
  VLM input JSON
  VLM output JSON
  ActionOutcome JSON
  memory context JSON

Per-episode:
  topdown trajectory visualization
  coarse goal / robot heading / selected fine goals
  node-edge memory graph summary
  metric summary JSON
  contact sheet
```

현재 static audit에서 사용한 `topdown + RGB + fine point layout`을 closed-loop episode용으로 확장한다.

---

## 10. Implementation Priority

우선순위는 다음과 같다.

```text
P0. Static audit 유지
  기존 Habitat topdown/RGB/VLM fine point layout 유지

P1. HabitatPixelNavBackend skeleton
  get_robot_state
  get_observation
  rotate
  capture_views

P2. execute_waypoint 연결
  selected image point -> PixelNav/S2E local goal
  fixed horizon 실행
  ActionOutcome 생성

P3. NavMemoryAgent 연결
  build_vlm_memory_context(nav_memory_context_v5)
  memory_ops 검증
  negative edge / deadlock update

P4. Scenario runner
  S1/S2/S3 최소 시나리오 반복 실행
  metric summary와 visualization 저장
```

처음부터 full HM3D benchmark를 목표로 하지 않는다. 먼저 3개 test scene 또는 사용 가능한 HM3D subset에서 curated scenario를 만든 뒤, random scene batch로 확장한다.

---

## 11. Risks and Mitigations

```text
Risk 1. VLM selected image point가 PixelNav local goal로 바로 변환되지 않음
  Mitigation:
    depth/navmesh projection 또는 PixelNav expected input format을 먼저 adapter로 고정

Risk 2. Qwen3-VL-32B-Thinking이 JSON completion 전에 token을 소진
  Mitigation:
    max_tokens 증가, content/reasoning fallback, JSON extraction retry

Risk 3. Multi-view point는 좋지만 robot은 front-only controller를 사용
  Mitigation:
    force_front_view_waypoint 정책 사용
    selected_view가 left/right/back이면 먼저 rotate 후 re-observe

Risk 4. False merge가 memory graph를 오염
  Mitigation:
    VLM은 request만 하고 backend verification 통과 시에만 merge
    qwen_nav_memory_framework_v5 기준 soft merge 정책 사용
      revisit node를 삭제하지 않고 보존
      canonical node와 revisit node를 same-place relative edge로 연결
      first 방문과 revisit 사이의 relative baseline/parallax를 유지
    spatial plausibility gate 사용
      graph-composed relative pose로 candidate-current/revisit 간 baseline 확인
      corridor/doorway/unknown 등 place category별 conservative threshold 적용
      far visual alias / outlier는 same-place merge 대신 reject 또는 request_observation
    false-positive merge 방지를 duplicate node 제거보다 우선
    qwen_nav_memory_framework_v6 ablation에서는 GaP-lite validation 사용
      go action은 selected_candidate_ref를 요구
      unknown/avoid=true candidate_ref는 request_observation으로 recover
      Feedback.md로 validation failure를 prompt/policy 개선 신호로 사용

Risk 5. CPU Habitat rendering이 느림
  Mitigation:
    GPU Habitat-Sim 사용, scene/trial 수를 작은 batch부터 시작
```

---

## 12. Report Summary

보고용 요약은 다음과 같다.

```text
Habitat + PixelNav 검증은 최종 benchmark 성능보다 먼저,
VLM grounded goal이 실제 local navigation loop에 들어갈 수 있는지를 확인하는 단계입니다.

우리는 먼저 Habitat에서 RGB/topdown/coarse goal/robot pose를 생성하고,
Qwen3-VL이 visible fine point와 selected view를 안정적으로 출력하는지 확인했습니다.
single-view random10과 multi-view random10에서는 모두 selected point가 생성되었고,
32B Thinking random5에서는 4/5가 valid하게 생성되었습니다.

다음 단계는 VLM selected point를 PixelNav/S2E local waypoint로 실행하고,
그 결과를 ActionOutcome으로 기록하는 것입니다.
ActionOutcome에는 moved_distance_m, collision, no_progress, odom_delta가 포함되어야 하며,
이 값이 memory graph의 success edge, negative edge, deadlock state를 갱신합니다.

metric은 VLM output validity, PixelNav local execution success, goal progress,
memory update correctness, deadlock recovery, false merge rate로 나누어 관리합니다.
이를 통해 schema / memory / multiview ablation으로 넘어가기 전에,
integration 자체가 닫힌 loop로 작동하는지 먼저 검증합니다.
```
