# Memory Structure and VLM Prompt Definition

## 1. One-line Summary

본 문서는 VLM waypoint adapter가 사용할 memory 구조와 VLM prompt 주입 방식을 정의한다. 현재 구현 기준은 `/home/icra/voca-s2e/qwen_nav_memory_framework_v5`이며, memory는 raw trajectory나 image history가 아니라 **relative-pose topo-metric episodic memory graph with directional negative edges, VLM-grounded place recognition, live latest-node pose relation, backend-verified soft merge**로 저장한다.

핵심은 다음과 같다.

```text
VLM:
  느린 semantic waypoint / interrupt supervisor

S2E / PixelNav:
  빠른 local navigation skill

Memory:
  장소 node, 상대 pose edge, negative edge, compressed keyframe,
  place recognition candidate, live latest-node pose relation을 저장/관리하는 graph

Prompt:
  전체 graph가 아니라 현재 판단에 필요한 compact memory context만 주입
```

---

## 2. Why Memory Is Needed

VLM이 현재 RGB만 보고 waypoint를 선택하면 다음 문제가 발생한다.

- 같은 corridor나 branch에서 반복적으로 같은 선택을 할 수 있다.
- dead-end에 들어갔을 때 이전 분기점으로 돌아가는 전략을 만들기 어렵다.
- 과거에 실패한 branch와 성공한 escape route를 다음 판단에 활용하기 어렵다.
- 모든 과거 image/frame을 prompt에 넣으면 token cost가 커지고, 중요한 정보가 희석된다.
- VLM이 goal 방향만 따라가면 일시적으로 goal 반대 방향으로 나가야 하는 deadlock escape를 설명하기 어렵다.

따라서 memory는 "많은 과거 정보를 저장하는 로그"가 아니라, **현재 decision에 필요한 navigation constraints와 route hints를 복원하는 graph memory**로 설계한다.

---

## 3. System Role Split

선임님 framework의 구조는 다음 역할 분리를 전제로 한다.

```text
RobotBackend / Habitat / PixelNav / S2E
  - robot_state 제공
  - RGB observation 제공
  - waypoint 실행
  - rotate 실행
  - action outcome 반환
    - success
    - collision
    - no_progress
    - moved_distance_m
    - odom_delta

NavMemoryAgent
  - observation 수집
  - visual place localization / revisit candidate 생성
  - latest/current node와 현재 robot pose 사이의 live relative pose 갱신
  - MemoryGraph update
  - nav_vlm_waypoint_v1 input 생성
  - VLM output sanitize
  - backend action 실행

Qwen3-VL Supervisor
  - go / rotate / stop / request_observation 결정
  - selected_view_id/type + selected_image_point 출력
  - optional memory_ops 제안
  - place recognition 후보를 보고 confirm/reject/merge 요청
```

여기서 VLM은 low-level controller가 아니다. VLM은 S2E/PixelNav가 사용할 local waypoint와 interrupt action을 제안하고, 실제 이동과 collision handling은 fast navigation module이 담당한다.

---

## 4. Core Memory Design Claim

Memory는 **relative-pose topo-metric episodic graph**로 유지한다.

```text
Topological:
  node는 장소/place/situation을 나타낸다.

Topo-metric:
  edge는 node 간 연결뿐 아니라 relative SE(2) pose를 저장한다.

Episodic:
  node, keyframe, edge traversal에는 frame/time 정보가 들어간다.

Directional negative memory:
  실패는 방 전체가 아니라 특정 directed edge 또는 entry branch에 기록한다.

Live latest-node pose relation:
  node에 global pose를 저장하지 않되,
  runtime에는 latest/current node 기준 현재 robot의 상대 pose를 유지한다.
```

중요한 설계 선택은 **memory node에 global pose를 직접 저장하지 않는 것**이다. Robot state와 coarse goal은 VLM input에서 GPS/map-aligned 좌표계로 제공될 수 있지만, memory graph 자체는 node 간 relative pose를 중심으로 유지한다.

v5에서 유지되는 중요한 보완은 `T_latest_node_to_current_robot`이다. 로봇이 latest node를 만든 뒤 아직 새 node를 만들 정도로 멀리 가지 않았더라도 실제 로봇 pose는 계속 변한다. 따라서 candidate exit를 고를 때는 edge pose를 node frame 그대로 쓰지 않고, latest node 기준 현재 robot의 live relative pose를 반영해 **현재 robot frame**으로 변환한다. 이 값은 runtime state이며, node의 persistent global pose가 아니다.

이렇게 하면 localization noise가 있더라도 다음과 같은 복원이 가능하다.

```text
current localized node
  -> outgoing edge relative pose
  -> candidate exit direction
  -> VLM view_type hint
  -> PixelNav/S2E local waypoint
```

---

## 5. Memory Graph Schema

Framework의 graph serialization은 다음 정책을 갖는다.

```json
{
  "schema_version": "relative_topometric_memory_graph_v5",
  "pose_policy": {
    "node_global_pose_stored": false,
    "edge_pose_type": "relative_SE2",
    "edge_pose_direction": "src_to_dst",
    "supports_path_composition": true,
    "runtime_live_pose_relation": true,
    "pose_graph_optimization": "TODO_not_enabled"
  },
  "nodes": {},
  "edges": {},
  "current_node_id": "n_00004",
  "previous_node_id": "n_00003",
  "runtime_current_pose_relation_to_latest_node": {
    "valid": true,
    "pose_type": "live_relative_SE2_not_global_node_pose",
    "latest_node_id": "n_00004",
    "latest_node_to_robot": {
      "dx_m": 0.65,
      "dy_m": 0.0,
      "dyaw_deg": 0.0
    },
    "robot_to_latest_node": {
      "dx_m": -0.65,
      "dy_m": 0.0,
      "dyaw_deg": 0.0
    }
  },
  "negative_edge_index": {}
}
```

이 graph는 VLM input의 `memory` field에 그대로 들어가지 않는다. 전체 graph는 내부 memory state이고, VLM prompt에는 `build_vlm_memory_context`로 만든 compact context만 들어간다.

주의할 점은 VLM에 전체 graph 원본을 넣지 않고, `qwen_nav_memory_framework_v5`의 `MemoryGraph.build_vlm_memory_context()`가 만든 `nav_memory_context_v5` compact context만 넣는다는 점이다.

`qwen_nav_memory_framework_v6`는 v5 graph memory를 대체하지 않고, GaP-lite policy sidecar를 ablation 옵션으로 추가한다. v6 context는 `nav_memory_context_v6`이며 `candidate_refs.exits`, `candidate_refs.revisits`, `nav_skill_cards`, `policy_harness_state`를 추가한다. VLM은 `go` action에서 가능하면 backend가 제안한 `selected_candidate_ref`를 함께 출력하고, backend는 이를 `validation_checkpoints`로 검증한다.

---

## 6. MemoryNode

`MemoryNode`는 로봇이 방문한 의미 있는 장소 또는 상황을 나타낸다.

```json
{
  "node_id": "n_00004",
  "node_type": "place",
  "place_category": "unknown",
  "created_frame_index": 42,
  "created_timestamp_ms": 391245,
  "last_seen_frame_index": 51,
  "last_seen_timestamp_ms": 398120,
  "visit_count": 2,
  "keyframes": [],
  "semantic": {
    "short_description": "corridor junction with a doorway on the left"
  },
  "navigation_state": {
    "deadlock_status": "none",
    "loop_status": "none",
    "risk_level": "low",
    "memory_importance": "medium"
  },
  "negative_memory": null,
  "lifecycle": {
    "storage_tier": "hot",
    "vlm_visible": true,
    "can_retrieve_image_for_vlm": true,
    "compression_level": "none"
  },
  "visual_signature": {}
}
```

Node 생성 조건은 다음과 같다.

```text
새로운 장소로 visual localization된 경우
의미 있는 branch, doorway, junction, dead-end에 도달한 경우
VLM이 waypoint decision을 내린 위치인 경우
PixelNav/S2E outcome이 collision, no_progress, deadlock인 경우
기존 node와 visual similarity가 낮거나 이동량이 threshold를 넘은 경우
```

Node에는 시간 정보가 반드시 들어간다. `created_frame_index`, `created_timestamp_ms`, `last_seen_frame_index`, `last_seen_timestamp_ms`, `visit_count`는 recency, loop detection, compression, revisit 판단에 사용된다.

---

## 7. Keyframe and Image Lifecycle

Image는 영구 prompt memory가 아니라 **active evidence**로 취급한다.

```json
{
  "keyframe_id": "kf_00012",
  "node_id": "n_00004",
  "frame_index": 42,
  "timestamp_ms": 391245,
  "view_type": "front",
  "relative_heading_deg": 0,
  "image_ref": "front_rgb.png",
  "embedding_id": "emb_kf_00012",
  "active_for_vlm": true,
  "storage_tier": "hot",
  "caption": "open floor ahead, doorway on left"
}
```

Lifecycle은 다음처럼 관리한다.

```text
HOT:
  current/recent node
  active keyframe image를 VLM prompt에 첨부 가능

WARM:
  필요하면 image retrieval 가능

COMPRESSED:
  image_ref는 prompt에서 제거
  embedding, semantic summary, relative constraints, negative memory는 유지

ARCHIVED:
  active planning과 VLM prompt에서는 제외
  audit/merge history로만 보존
```

이는 SLAM marginalization과 비슷하다. 오래된 raw image는 prompt window에서 빠지지만, graph constraint와 negative memory는 남는다.

---

## 8. MemoryEdge and Relative Pose

`MemoryEdge`는 node 간 directed connection과 이동 결과를 저장한다.

```json
{
  "edge_id": "e_00003",
  "src_node_id": "n_00003",
  "dst_node_id": "n_00004",
  "relative_pose_src_to_dst": {
    "dx_m": 1.4,
    "dy_m": 1.5,
    "dyaw_deg": 45.0,
    "covariance_diag": [0.05, 0.05, 4.0]
  },
  "edge_type": "temporal_transition",
  "relation_type": "transition",
  "directed": true,
  "traversal": {
    "status": "success",
    "success_count": 1,
    "failure_count": 0,
    "last_outcome": "success",
    "last_attempt_frame": 51,
    "escape_edge_id": null
  },
  "planning_cost": {
    "base_cost": 2.1,
    "deadlock_penalty": 0.0,
    "loop_penalty": 0.0,
    "uncertainty_penalty": 0.4,
    "final_cost": 2.5
  }
}
```

Relative pose는 `src_node` 기준 local frame에서 `dst_node`가 어디에 있는지를 뜻한다.

```text
dx_m, dy_m:
  source node 기준 local displacement

dyaw_deg:
  source heading에서 destination heading까지의 상대 yaw

covariance_diag:
  odometry/action outcome 불확실성
```

이 edge들을 compose하면 과거 경로의 상대 subgoal을 현재 node 기준으로 복원할 수 있다.

```text
n1 -> n2 -> n3 path
  compose(edge n1->n2, edge n2->n3)
  -> n1 기준 n3의 relative pose
```

v5에서는 여기에 runtime live pose relation이 추가된다.

```text
Stored edge:
  T_node_src_to_node_dst

Runtime relation:
  T_latest_node_to_current_robot

Candidate exit for VLM:
  T_current_robot_to_candidate_exit
  = inverse(T_latest_node_to_current_robot) compose T_latest_node_to_exit
```

즉, edge 자체는 여전히 node 기준 relative pose로 저장하지만, VLM이 보는 `bearing_deg_robot`과 `relative_pose_robot_to_dst`는 현재 robot frame 기준으로 변환된 값이다. 이 보완이 없으면 로봇이 latest node에서 조금 이동한 뒤에도 candidate exit가 마치 node origin에서 보이는 방향처럼 계산되는 문제가 생긴다.

---

## 9. Directional Negative Edge Memory

Deadlock이나 blocked path는 node 전체가 아니라 **directed edge**에 기록한다.

```text
좋지 않은 방식:
  "이 방은 실패한 방이다"

좋은 방식:
  "n2에서 n3로 들어간 이 directed branch가 deadlock_entry이다"
```

이렇게 해야 특정 entry branch만 피하면서 다른 exit나 doorway는 계속 탐색할 수 있다.

```json
{
  "node_id": "n_00004",
  "navigation_state": {
    "deadlock_status": "confirmed",
    "risk_level": "high",
    "memory_importance": "critical"
  },
  "negative_memory": {
    "is_negative_region": true,
    "reason": "no_progress_or_collision",
    "avoid_scope": "incoming_edge_only",
    "avoid_until": "episode_end",
    "severity": 0.9,
    "failed_entry_edge_id": "e_00003"
  }
}
```

Edge traversal도 함께 바뀐다.

```json
{
  "edge_id": "e_00003",
  "traversal": {
    "status": "deadlock_entry",
    "failure_count": 1,
    "last_outcome": "no_progress_or_collision",
    "escape_edge_id": "e_00005"
  }
}
```

이 설계는 우리가 말한 failed waypoint보다 더 graph-oriented한 표현이다. 즉 실패한 `[u, v]` point 하나보다, 그 point 실행 결과가 만든 **entry edge의 실패 상태**를 더 중요하게 저장한다.

---

## 10. Backtracking and Branch Recovery

Graph memory의 장점은 막힌 길에서 이전 분기점으로 돌아갈 수 있다는 점이다.

예시는 다음과 같다.

```text
n1 branch
  -> n2 branch_1 corridor
      -> n3 dead_end
  -> branch_2 아직 미시도
```

로봇이 `n1 -> n2 -> n3`로 갔다가 `n3`에서 no_progress/dead-end를 만났다면 memory는 다음처럼 갱신된다.

```text
1. n3.navigation_state.deadlock_status = confirmed
2. edge n2 -> n3 traversal.status = deadlock_entry
3. edge n2 -> n3 negative_edge_index에 등록
4. escape 성공 시 n3 -> n2 또는 n3 -> n1 방향 edge를 escape_success로 기록
5. n1로 돌아오면 negative edge를 피하고 unvisited/unknown frontier branch를 우선 고려
```

VLM prompt에는 다음처럼 들어간다.

```text
Memory:
- Current node is near n3, marked as confirmed deadlock.
- Incoming edge n2 -> n3 is a known negative branch.
- Avoid re-entering the same edge.
- Use escape edge or request observation to recover.
- After escape, resume toward coarse goal bearing.
```

따라서 VLM은 goal 반대 방향으로 움직이는 detour도 합리화할 수 있다. 이 detour는 goal을 잊은 것이 아니라, deadlock escape 후 goal-directed planning으로 복귀하기 위한 임시 행동이다.

---

## 11. VLM Memory Context

VLM prompt에는 전체 graph를 넣지 않는다. `MemoryGraph.build_vlm_memory_context`가 현재 decision에 필요한 subset만 만든다.

```json
{
  "schema_version": "nav_memory_context_v5",
  "graph_summary": {
    "num_nodes": 5,
    "num_edges": 4,
    "num_deadlock_edges": 1,
    "num_compressed_nodes": 2,
    "pose_graph_optimization": "TODO_not_enabled"
  },
  "current_pose_relation_to_latest_node": {
    "valid": true,
    "latest_node_id": "n_00004",
    "pose_type": "live_relative_SE2_not_global_node_pose",
    "latest_node_to_robot": {
      "dx_m": 0.65,
      "dy_m": 0.0,
      "dyaw_deg": 0.0
    },
    "robot_to_latest_node": {
      "dx_m": -0.65,
      "dy_m": 0.0,
      "dyaw_deg": 0.0
    },
    "usage": "transform node-frame graph exits into current robot frame before VLM fine-goal selection"
  },
  "current_localization": {
    "current_node_id": "n_00004",
    "match_status": "localized",
    "match_confidence": 0.91,
    "candidate_node_ids": ["n_00004", "n_00002"],
    "matched_keyframe_ids": ["kf_00012"],
    "revisit_likelihood": 0.91,
    "final_place_recognition_policy": "VLM_verifies_backend_candidates_backend_commits"
  },
  "place_recognition": {
    "policy": "backend_retrieval_proposes_candidates; VLM verifies; backend commits",
    "revisit_candidates": [
      {
        "candidate_node_id": "n_00002",
        "visual_retrieval_score": 0.94,
        "semantic_summary": "hallway intersection",
        "negative_memory": null
      }
    ],
    "false_merge_warning": "prefer uncertain/request_observation over false positive merge"
  },
  "goal_context": {
    "task_mode": "PointNav",
    "goal_bearing_from_current_deg": -55.2,
    "goal_distance_m": 17.4,
    "detour_status": "normal_goal_seek",
    "goal_resume_hint": "avoid known negative edges; after escape re-align toward goal bearing"
  },
  "local_topology": {
    "current_node": "...",
    "candidate_exits": []
  },
  "deadlock_state": {
    "current_node_id": "n_00004",
    "status": "none",
    "incoming_edge_id": "e_00003"
  },
  "retrieved_memory_images": [],
  "compressed_negative_memories": [],
  "loop_warning": {
    "is_looping": false,
    "repeated_branch_count": 0
  }
}
```

Prompt에 들어가는 우선순위는 다음과 같다.

```text
1. current_pose_relation_to_latest_node
2. current_localization
3. place_recognition.revisit_candidates
4. goal_context
5. current node summary
6. local_topology.candidate_exits
7. failed/deadlock edges
8. retrieved_memory_images
9. compressed_negative_memories
10. loop_warning
```

---

## 12. Candidate Exits

`candidate_exits`는 현재 node에서 VLM이 선택할 수 있는 graph-aware local direction 후보이다.

```json
{
  "edge_id": "e_00003",
  "dst_node_id": "n_00005",
  "bearing_deg_robot": -90.0,
  "view_type_hint": "left",
  "status": "deadlock_entry",
  "avoid": true,
  "goal_alignment": 0.82,
  "deadlock_risk": 1.0,
  "exploration_value": 0.2,
  "score": -0.25,
  "reason": "known negative branch",
  "relative_pose_node_to_dst": {
    "dx_m": 0.0,
    "dy_m": -2.0,
    "dyaw_deg": 0.0
  },
  "relative_pose_robot_to_dst": {
    "dx_m": -0.65,
    "dy_m": -2.0,
    "dyaw_deg": 0.0
  },
  "pose_relation_used": {
    "valid": true,
    "latest_node_id": "n_00004"
  }
}
```

Graph에 없는 방향은 frontier candidate로 들어간다.

```json
{
  "exit_id": "frontier_right",
  "edge_id": null,
  "view_type_hint": "right",
  "bearing_deg_robot": 90.0,
  "status": "unknown_frontier",
  "avoid": false,
  "reason": "unvisited or unmodeled local view"
}
```

VLM은 이 정보를 이용해 다음과 같은 결정을 할 수 있다.

```text
known negative branch:
  goal 방향과 맞더라도 피한다.

known success edge:
  goal alignment가 좋으면 우선 고려한다.

unknown frontier:
  negative edge보다 우선 고려할 수 있다.

insufficient context:
  request_observation으로 directed/full sweep을 요청한다.
```

v5 기준으로 `candidate_exits`의 핵심은 `bearing_deg_robot`이 현재 robot frame 기준이라는 점이다. 내부 graph edge는 node frame의 `relative_pose_node_to_dst`를 유지하지만, VLM prompt에는 live pose relation을 반영한 `relative_pose_robot_to_dst`를 함께 제공한다.

---

## 13. VLM Prompt Policy

Prompt는 자유로운 chain-of-thought가 아니라 controlled ReAct 형태를 따른다.

```text
OBSERVE:
  현재 RGB view, robot heading, coarse goal bearing을 확인한다.

RECALL:
  current_pose_relation, current_localization, place_recognition,
  candidate_exits, negative edges, compressed memories를 확인한다.

PLAN:
  goal alignment, deadlock risk, exploration value를 함께 고려한다.
  failed/deadlock branch는 피한다.
  candidate_exits는 현재 robot frame 기준 bearing으로 해석한다.
  deadlock escape가 필요하면 goal 반대 방향 detour도 허용한다.

ACT:
  go / rotate / stop / request_observation 중 하나를 선택한다.
  go라면 selected_view_id/type과 selected_image_point를 출력한다.

WRITE_MEMORY_REQUEST:
  필요하면 memory_ops를 제안한다.
  단, 실제 graph update는 backend가 검증 후 수행한다.
```

실제 prompt 정책은 다음 원칙을 강제한다.

```text
1. VLM은 low-level controller가 아니다.
2. output은 nav_vlm_waypoint_v1 JSON이어야 한다.
3. failed/deadlock branch는 goal 방향과 맞더라도 피한다.
4. deadlock escape를 위해 goal 반대 방향 detour도 허용한다.
5. 정보가 부족하면 guessing하지 말고 request_observation을 사용한다.
6. memory_ops는 요청일 뿐이며 backend가 검증한다.
7. revisit/merge는 false positive가 더 위험하므로 uncertain이면 request_observation을 사용한다.
8. 자유로운 긴 reasoning 대신 controlled reason code를 사용한다.
```

---

## 14. VLM Memory Operations

VLM은 memory update에 참여할 수 있지만, graph를 직접 수정하지 않는다. VLM은 optional `memory_ops`를 제안하고, `NavMemoryAgent`와 backend가 검증 후 반영한다.

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

현재 framework 기준으로 안전하게 반영 가능한 operation은 다음과 같다.

```text
confirm_revisit_node:
  backend retrieval 후보 중 하나를 VLM이 같은 장소로 확인
  backend가 score, node 상태, critical memory conflict를 검증한 뒤 commit

reject_revisit_candidate:
  retrieval 후보가 현재 장소와 다르다고 판단
  backend는 후보를 즉시 삭제하지 않고 이번 decision에서 제외

request_merge_nodes / merge_nodes:
  VLM은 merge를 요청할 수 있지만,
  backend가 visual similarity, odometry consistency, topology conflict를 검증한 뒤 수행

mark_deadlock / mark_incoming_edge:
  confidence threshold를 넘고,
  current node와 incoming edge가 존재할 때만 반영

mark_blocked_edge / mark_escape_edge:
  action outcome과 deadlock state machine이 일치할 때만 반영

compress_node:
  current node가 아닌 node에 대해서만 image_ref를 제거하고 compressed 상태로 변경

update_object_belief / mark_object_seen:
  ObjNav 확장 시 semantic evidence로만 반영
```

아래 operation은 VLM 요청만으로 수행하지 않는다.

```text
delete_node:
  graph consistency와 auditability 문제 때문에 즉시 삭제하지 않음

edge_rewrite:
  pose graph consistency 검증 필요
```

이 구조는 **VLM semantic reasoning을 memory update에 활용하면서도 hallucination으로 graph가 오염되는 것을 막는 hybrid memory manager**이다.

---

## 15. Memory Update Flow

Closed-loop update 흐름은 다음과 같다.

```text
Step 1. RobotBackend.get_observation()
  RGB views, frame_index, timestamp 획득

Step 2. Visual place localization
  current observation embedding을 기존 keyframe index에서 검색
  localized / uncertain / new_place 판단
  revisit candidate를 VLM context에 포함

Step 3. MemoryGraph before-decision update
  localized node로 current_node 갱신
  또는 new node 생성
  latest/current node 기준 current robot pose relation 갱신

Step 3.5. VLM-grounded place recognition 준비
  backend retrieval 후보를 place_recognition.revisit_candidates로 제공
  VLM은 confirm_revisit_node / reject_revisit_candidate / request_merge_nodes 제안 가능
  backend는 false merge를 막기 위해 보수적으로 검증

Step 4. VLM input 생성
  task
  coordinate_frame
  robot_state
  observation
  memory_context
  constraints

Step 5. Qwen3-VL decision
  action
  selected waypoint
  optional memory_ops

Step 6. Output sanitize
  invalid action이면 request_observation fallback
  invalid point이면 rotate fallback
  invalid view이면 safe view로 repair

Step 7. RobotBackend action execution
  go / rotate / capture_views / stop

Step 8. ActionOutcome 기반 memory update
  success_count / failure_count
  deadlock mark
  escape edge
  latest-node-to-current-robot live pose 갱신
  pending relative pose edge는 마지막 action delta가 아니라 cumulative live pose를 사용
  node compression
```

이 구조에서 memory의 신뢰성은 VLM이 아니라 backend outcome이 보장한다.

---

## 16. Habitat + PixelNav Integration Plan

`qwen_nav_memory_framework_v5`는 Habitat + PixelNav 연결에서 사용하는 현재 구현 기준이다. 우리 실험에서는 `RobotBackend` protocol을 Habitat/PixelNav용으로 구현해 official Habitat env에 붙인다.

```python
class HabitatPixelNavBackend:
    def get_robot_state(self):
        # Habitat agent pose / GPS-map aligned pose / heading
        # v5 live pose relation 계산의 기준 입력
        ...

    def get_observation(self):
        # current RGB or multi-view RGB -> Observation schema
        ...

    def execute_waypoint(self, *, view_type, view_id, point_px, ttl_ms):
        # selected image point를 PixelNav/S2E local goal로 전달
        # 일정 step 실행 후 ActionOutcome 반환
        ...

    def rotate(self, yaw_deg):
        # Habitat agent 회전
        ...

    def capture_views(self, yaw_offsets_deg, mode="directed_sweep"):
        # VLM이 요청한 directed/full sweep RGB 수집
        ...
```

필수로 연결해야 하는 outcome은 다음과 같다.

```text
moved_distance_m:
  local waypoint 수행 후 실제 이동량

collision:
  Habitat collision 여부

no_progress:
  이동량이 작거나 같은 위치 반복

odom_delta:
  action 단위 odometry delta
  단, 새 edge 생성 시에는 v5 agent가 latest-node-to-robot cumulative live pose를 사용

success:
  local waypoint가 실행 가능한 방향이었는지
```

이 outcome과 `get_robot_state()`가 있어야 memory graph가 reliable edge와 negative edge를 구분하고, latest node 기준 현재 robot pose를 계속 갱신할 수 있다.

---

## 17. Qwen3-VL-32B-Thinking Integration Issue

현재 서버에 올라온 `qwen3-vl-32b-thinking`은 reasoning이 길어질 수 있다. 실제 실험에서 다음 문제가 관찰되었다.

```text
message.content가 null
message.reasoning만 길게 생성
max_tokens 부족으로 최종 JSON까지 도달하지 못함
```

따라서 VLM client에는 다음 보강이 필요하다.

```text
1. max_tokens 증가
2. message.content fallback
3. message.reasoning fallback
4. JSON extraction fallback
5. 실패 시 retry 또는 request_observation fallback
```

즉 32B Thinking을 사용하지 못하는 것은 아니다. 다만 실시간 low-level control이 아니라 async semantic supervisor로 사용하고, structured output reliability를 metric으로 관리해야 한다.

---

## 18. Ablation Plan

Memory ablation은 다음 단계로 비교한다.

```text
A1. no memory
  현재 RGB + robot pose + coarse goal만 사용

A2. flat recent memory
  최근 selected waypoint와 실패 기록만 list로 제공

A3. relative-pose graph memory
  node/edge/candidate_exits 제공

A4. graph memory + negative edge
  failed/deadlock branch 회피 추가

A5. graph memory + negative edge + retrieved memory images
  current/neighbor/negative keyframe image를 prompt에 함께 제공

A6. graph memory + place recognition / merge
  revisit candidate와 VLM-confirmed merge를 사용

A7. graph memory + live latest-node pose relation
  candidate exit를 current robot frame으로 변환
```

측정 metric은 다음과 같다.

```text
Schema Validity:
  VLM output JSON parse 성공률

Waypoint Validity:
  selected point가 visible navigable floor에 있는 비율

View Selection Accuracy:
  selected_view_type이 coarse goal bearing 또는 candidate_exit와 맞는 비율

Deadlock Re-entry Rate:
  known negative edge로 다시 들어가는 비율

False Merge Rate:
  서로 다른 장소를 같은 node로 잘못 merge하는 비율

Revisit Recognition Accuracy:
  revisit candidate를 VLM/backend가 올바르게 confirm/reject하는 비율

Candidate Exit Bearing Error:
  node frame 그대로 계산한 bearing 대비 current robot frame 보정 후 bearing의 정확도

Recovery Success:
  deadlock 이후 escape edge 또는 unvisited frontier로 빠져나오는 비율

Goal Progress:
  VLM/S2E 실행 후 coarse goal geodesic/euclidean distance 감소 여부

Latency:
  VLM 호출 시간, Thinking model JSON completion time
```

---

## 19. Implementation Scope

이번 주 구현 범위는 전체 graph planner 완성이 아니라, Habitat + PixelNav에서 memory가 실제로 작동하는 최소 경로를 만드는 것이다.

```text
Phase 1. Framework validation
  qwen_nav_memory_framework_v5 smoke test 확인
  MemoryGraph / NavMemoryAgent 동작 확인
  v5 unittest 통과 확인
  nav_memory_context_v5 출력 확인
  Qwen3-VL-32B-Thinking client 보강

Phase 2. HabitatPixelNavBackend
  Habitat RGB observation -> Observation schema
  Habitat pose/heading -> RobotState
  VLM selected point -> PixelNav/S2E local waypoint

Phase 3. ActionOutcome logging
  moved_distance_m
  collision
  no_progress
  odom_delta
  local waypoint reachability

Phase 4. Memory context prompt insertion
  build_vlm_memory_context(nav_memory_context_v5)를 nav_vlm_waypoint_v1 memory field에 삽입
  current_pose_relation_to_latest_node 확인
  place_recognition.revisit_candidates 확인
  candidate_exits / negative edge / compressed memory 확인

Phase 5. Ablation
  no memory
  flat memory
  graph memory
  graph memory + negative edge
  graph memory + place recognition
  graph memory + live pose relation
```

---

## 20. Report Summary

보고용 요약은 다음과 같다.

```text
Memory는 raw trajectory나 이미지 히스토리를 그대로 저장하지 않고,
relative-pose topo-metric episodic graph로 유지합니다.
Node는 장소/place/situation과 keyframe, semantic summary, navigation state를 저장하고,
Edge는 node 간 directed connection과 relative SE(2) pose, traversal outcome, planning cost를 저장합니다.

최종 구현 기준은 qwen_nav_memory_framework_v5입니다.
v5에서는 VLM-grounded place recognition, backend-verified revisit/merge, latest/current node와 현재 robot pose 사이의 live relative pose relation, soft merge, geometric filtering이 함께 사용됩니다.
따라서 node에는 global pose를 저장하지 않으면서도,
candidate exit는 current robot frame 기준으로 VLM에 제공할 수 있습니다.

특히 실패는 장소 전체가 아니라 directional negative edge로 기록합니다.
따라서 dead-end나 no-progress가 발생해도 방 전체를 금지하지 않고,
문제가 된 entry branch만 피하면서 다른 exit나 unvisited frontier를 선택할 수 있습니다.

VLM prompt에는 전체 graph가 아니라 build_vlm_memory_context가 만든 compact context만 넣습니다.
이 context schema는 nav_memory_context_v5이며,
current_pose_relation_to_latest_node, current localization, place recognition candidates,
goal context, candidate exits, retrieved memory images, compressed negative memories가 포함됩니다.

VLM은 memory_ops를 통해 confirm_revisit_node, request_merge_nodes, mark_deadlock,
mark_escape_edge, compress_node 같은 update를 제안할 수 있지만,
실제 graph update는 backend outcome과 rule-based verification을 거쳐 수행합니다.
따라서 VLM의 semantic reasoning을 memory 구축에 활용하면서도,
hallucination으로 graph가 오염되는 것을 방지합니다.

현재 기본 구현은 qwen_nav_memory_framework_v5에 Habitat/PixelNav backend를 붙인 상태입니다.
v6는 GaP-lite ablation으로 `candidate_refs`, `nav_skill_cards`, `validation_checkpoints`, `Feedback.md`를 추가해 memory를 policy처럼 안전하게 사용하는지 검증합니다.
Qwen3-VL-32B-Thinking과 heuristic policy의 차이, candidate exit/negative edge 활용, v5/v6 memory ablation을 수행하는 것입니다.
```
