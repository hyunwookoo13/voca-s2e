# VLM schema정의

### VLM Input/Output Schema 정의

### **==1\. 목적==**

본 schema는 https://chatgpt.com/share/6a4b1bc6-bad0-83ee-b21f-2711e61ce4a4의 async navigation system 구상을 기반으로 한다.
시스템은 느리게 동작하는 **VLM semantic supervisor**와 빠르게 동작하는 **S2E / PixelNav-style navigation skill**로 구성된다. VLM은 low-level controller가 아니라, coarse goal을 local fine waypoint로 변환하는 **goal adapter** 역할을 수행한다.

```
Input:
robot pose, heading, coarse goal, RGB observation, memory

Output:
action, optional observation request, selected view, selected image point, controlled reasoning
```

VLM이 항상 waypoint를 바로 찍는 것이 아니라, 정보가 부족할 경우 **추가 관측을 요청할 수 있다**는 점이다.

```
VLM decision:
1. 현재 관측만으로 waypoint 선택
2. 추가 view 관측 요청
3. rotate / stop 같은 interrupt action 요청
```

핵심 질문은 다음과 같다.

→ 현재 robot pose, heading, coarse goal, RGB 관측, memory가 주어졌을 때, VLM은 S2E 또는 PixelNav가 사용할 수 있는 local fine waypoint와 필요한 interrupt action을 안정적으로 생성할 수 있는가?

---

### **==2\. System-level Assumption==**

- **Fast Module::** S2E / PixelNav-style navigation skill
  - 역할:
    - 실시간 local control
    - 동적 장애물 회피
    - VLM이 제안한 local waypoint 추종
- **Slow Module::** `Qwen3-VL 기반 VLM semantic supervisor`
  - 역할:
    - coarse goal과 현재 관측 해석
    - 현재 정보만으로 충분한지 판단
    - 필요 시 추가 observation request 생성
    - S2E/PixelNav가 사용할 fine waypoint 생성
    - 위험 상황에서는 rotate/stop action 생성

---

### **==3\. VLM Input Schema==**

이번 주 구현에서는 `nav_vlm_waypoint_v1`을 사용한다. (앞으로 추가 업데이트 사항을 고려)

- `coarse_goal.map_xy`와 `robot_state.map_xy`는 같은 GPS/map-aligned 좌표계에 존재한다.
- 실제 로봇에서는 GPS/IMU 기반 pose를 사용한다고 가정하고, Habitat에서는 GT pose에 noise를 추가해 평가한다.
- coarse goal의 visibility는 VLM이 추론하는 값이 아니라, 호출 전 전처리로 계산한다.
- 각 RGB view는 numeric `view_id`와 semantic `view_type`을 분리한다.
- observation에는 시간 흐름을 위해 `timestamp_ms`, `frame_index`, view별 timestamp를 포함한다.
- `memory`에는 전체 graph 원본을 넣지 않고, `MemoryGraph.build_vlm_memory_context()`가 만든 compact context만 넣는다.
- 최종 memory 기준은 `qwen_nav_memory_framework_v3`이며, VLM input에 들어가는 memory context schema는 `nav_memory_context_v4`를 사용한다.

```json
{
  "schema_version": "nav_vlm_waypoint_v1",
  "task": {
    "task_mode": "PointNav",
    "instruction": "move toward the coarse GPS goal",
    "target_object": null,
    "coarse_goal": {
      "type": "gps",
      "map_xy": [6.418, 21.904],
      "relative_bearing_deg": -55.2,
      "distance_m": 17.4,
      "camera_relation": {
        "in_front_camera_frame": false,
        "bearing_in_fov": false,
        "estimated_visible": false,
        "preprocess_method": "transform_goal_to_camera_frame"
      }
    }
  },
  "coordinate_frame": {
    "map_frame": "habitat_world_xy_or_gps_aligned_local_map",
    "robot_pose_source": "gps_or_sim_gt_with_noise",
    "heading_source": "imu_or_sim_gt_with_noise",
    "pose_noise": {
      "enabled": true,
      "xy_std_m": 0.10,
      "heading_std_deg": 2.0
    }
  },
  "robot_state": {
    "position_xyz": [-3.013, 0.046, 7.306],
    "map_xy": [-3.013, 7.306],
    "heading_rad": 1.537
  },
  "observation": {
    "mode": "current_only",
    "sequence_id": "seq_000042",
    "timestamp_ms": 391245,
    "frame_index": 128,
    "image_width": 640,
    "image_height": 480,
    "views": [
      {
        "view_id": 0,
        "view_type": "front",
        "timestamp_ms": 391245,
        "relative_heading_deg": 0,
        "image": "<front_rgb>"
      }
    ]
  },
  "memory": {
    "schema_version": "nav_memory_context_v4",
    "graph_summary": {
      "num_nodes": 0,
      "num_edges": 0,
      "num_deadlock_edges": 0,
      "num_compressed_nodes": 0,
      "pose_graph_optimization": "TODO_not_enabled"
    },
    "current_pose_relation_to_latest_node": {
      "valid": false,
      "latest_node_id": null,
      "pose_type": "live_relative_SE2_not_global_node_pose",
      "latest_node_to_robot": {
        "dx_m": 0.0,
        "dy_m": 0.0,
        "dyaw_deg": 0.0
      },
      "robot_to_latest_node": {
        "dx_m": 0.0,
        "dy_m": 0.0,
        "dyaw_deg": 0.0
      },
      "usage": "transform node-frame graph exits into current robot frame before VLM fine-goal selection"
    },
    "current_localization": {
      "current_node_id": null,
      "match_status": "new_place",
      "match_confidence": 0.0,
      "candidate_node_ids": [],
      "matched_keyframe_ids": [],
      "revisit_likelihood": 0.0,
      "final_place_recognition_policy": "VLM_verifies_backend_candidates_backend_commits"
    },
    "place_recognition": {
      "policy": "backend_retrieval_proposes_candidates; VLM verifies; backend commits",
      "revisit_candidates": [],
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
      "current_node": null,
      "candidate_exits": []
    },
    "deadlock_state": {
      "current_node_id": null,
      "status": "none",
      "incoming_edge_id": null
    },
    "retrieved_memory_images": [],
    "compressed_negative_memories": [],
    "loop_warning": {
      "is_looping": false,
      "repeated_branch_count": 0
    }
  },
  "constraints": {
    "allowed_actions": [
      "go",
      "rotate",
      "stop",
      "request_observation"
    ],
    "allowed_observation_modes": [
      "current_only",
      "directed_view",
      "directed_sweep",
      "full_sweep"
    ],
    "allowed_views": [
      "front",
      "left",
      "right",
      "back"
    ],
    "allowed_step_deg": [30, 45, 60, 90],
    "pixel_coordinate_rule": "u in [0,639], v in [0,479]",
    "must_select_visible_navigable_floor": true
  }
}
```

---

### **4\. Input Field 설명**

> `task:` **현재 navigation task와 coarse goal 정보를 담는다.**

```json
"task": {
  "task_mode": "PointNav",
  "instruction": "move toward the coarse GPS goal",
  "coarse_goal": {
    "type": "gps",
    "map_xy": [6.418, 21.904],
    "relative_bearing_deg": -55.2,
    "distance_m": 17.4,
    "camera_relation": {
        "in_front_camera_frame": false,
        "bearing_in_fov": false,
        "estimated_visible": false,
        "preprocess_method": "transform_goal_to_camera_frame"
    }
  }
}
```

초기 구현은 `PointNav`에 집중한다.
추후 ObjNav으로 확장 가능하다.

```
PointNav:
- coarse GPS/map goal을 local waypoint로 변환

ObjNav:
- 목표 객체를 찾기 위한 semantic exploration
```

> `coordinate_frame:` **coarse goal과 robot pose가 어떤 좌표계에서 정렬되어 있는지 명시한다.**

```json
"coordinate_frame": {
  "map_frame": "habitat_world_xy_or_gps_aligned_local_map",
  "robot_pose_source": "gps_or_sim_gt_with_noise",
  "heading_source": "imu_or_sim_gt_with_noise",
  "pose_noise": {
    "enabled": true,
    "xy_std_m": 0.10,
    "heading_std_deg": 2.0
  }
}
```

`coarse_goal.map_xy`와 `robot_state.map_xy`는 반드시 같은 map-aligned 좌표계에 있어야 한다.
실제 로봇에서는 GPS/IMU 기반 pose를 사용한다고 가정하고, Habitat 실험에서는 GT pose에 noise를 추가해 GPS/IMU 오차를 근사한다.

> `robot_state:` **로봇의 현재 위치와 heading이다.**

```
"robot_state": {
  "position_xyz": [-3.013, 0.046, 7.306],
  "map_xy": [-3.013, 7.306],
  "heading_rad": 1.537
}
```

VLM은 `heading_rad`와 `coarse_goal.relative_bearing_deg`를 사용하여 goal이 현재 시야 기준 어느 방향에 있는지 판단한다.

> `observation:` **현재 VLM에 제공된 RGB 관측이다. 초기에는** `current_only`**로 시작한다.**

```
"observation": {
  "mode": "current_only",
  "sequence_id": "seq_000042",
  "timestamp_ms": 391245,
  "frame_index": 128,
  "image_width": 640,
  "image_height": 480,
  "views": [
    {
      "view_id": 0,
      "view_type": "front",
      "timestamp_ms": 391245,
      "relative_heading_deg": 0,
      "image": "<front_rgb>"
    }
  ]
}
```

VLM이 현재 관측만으로 waypoint를 선택하기 어렵다고 판단하면 `request_observation`을 출력한다.

예를 들어 goal이 왼쪽에 있을 가능성이 높지만 현재 front view만 들어온 경우:

```
{
  "action": "request_observation",
  "observation_request": {
    "valid": true,
    "mode": "directed_sweep",
    "center_yaw_deg": -90,
    "step_deg": 30,
    "num_views": 3,
    "yaw_offsets_deg": [-120, -90, -60],
    "reason": "goal_bearing_left_but_not_visible"
  }
}
```

그 후 시스템은 요청된 yaw 방향으로 RGB를 추가 캡처하여 VLM에 다시 입력한다.

> `memory:` **전체 memory graph가 아니라, VLM decision에 필요한 compact memory context를 담는다.**

```
"memory": {
  "schema_version": "nav_memory_context_v4",
  "graph_summary": {
    "num_nodes": 0,
    "num_edges": 0,
    "num_deadlock_edges": 0,
    "num_compressed_nodes": 0,
    "pose_graph_optimization": "TODO_not_enabled"
  },
  "current_pose_relation_to_latest_node": {
    "valid": false,
    "latest_node_id": null,
    "pose_type": "live_relative_SE2_not_global_node_pose",
    "latest_node_to_robot": {
      "dx_m": 0.0,
      "dy_m": 0.0,
      "dyaw_deg": 0.0
    },
    "robot_to_latest_node": {
      "dx_m": 0.0,
      "dy_m": 0.0,
      "dyaw_deg": 0.0
    }
  },
  "current_localization": {
    "current_node_id": null,
    "match_status": "new_place",
    "match_confidence": 0.0,
    "final_place_recognition_policy": "VLM_verifies_backend_candidates_backend_commits"
  },
  "place_recognition": {
    "policy": "backend_retrieval_proposes_candidates; VLM verifies; backend commits",
    "revisit_candidates": [],
    "false_merge_warning": "prefer uncertain/request_observation over false positive merge"
  },
  "goal_context": {
    "task_mode": "PointNav",
    "goal_bearing_from_current_deg": -55.2,
    "goal_distance_m": 17.4,
    "detour_status": "normal_goal_seek"
  },
  "local_topology": {
    "current_node": null,
    "candidate_exits": []
  },
  "deadlock_state": {
    "current_node_id": null,
    "status": "none",
    "incoming_edge_id": null
  },
  "retrieved_memory_images": [],
  "compressed_negative_memories": [],
  "loop_warning": {
    "is_looping": false,
    "repeated_branch_count": 0
  }
}
```

초기 episode에서는 빈 graph에서 시작하지만, navigation이 진행되면 backend가 node-edge graph를 갱신한다.
VLM prompt에는 전체 graph를 그대로 넣지 않고 `build_vlm_memory_context()` 결과만 넣는다. 이 context는 `nav_memory_context_v4`이며, `current_pose_relation_to_latest_node`, place recognition 후보, 현재 node, 후보 exit, 실패한 방향, loop warning, 관련 keyframe summary만 sparse하게 포함한다.
`current_pose_relation_to_latest_node`는 node에 global pose를 저장하지 않으면서도 candidate exit를 current robot frame 기준으로 해석하기 위한 runtime relative pose이다.
`place_recognition`은 backend retrieval이 제안한 revisit 후보를 VLM이 confirm/reject/merge 요청할 수 있도록 제공하되, 실제 graph update는 backend verification 이후에만 수행한다.
세부 memory 구조와 update rule은 별도 문서인 `memory_vlm_prompt.md`에서 정의한다.

> `constraints:` **VLM output을 안정적으로 제한하기 위한 필드다.**

```
"constraints": {
  "allowed_actions": ["go", "rotate", "stop", "request_observation"],
  "allowed_observation_modes": [
    "current_only",
    "directed_view",
    "directed_sweep",
    "full_sweep"
  ],
  "allowed_views": ["front", "left", "right", "back"],
  "allowed_step_deg": [30, 45, 60, 90],
  "pixel_coordinate_rule": "u in [0,639], v in [0,479]",
  "must_select_visible_navigable_floor": true
}
```

`allowed_views`는 `front/left/right/back`처럼 사람이 읽기 쉬운 semantic view label을 제한하기 위한 필드다.
반면 `directed_sweep`의 실제 회전 각도는 `allowed_step_deg`와 `yaw_offsets_deg`로 제어한다. 따라서 `-120, -90, -60` 같은 sweep 요청은 `view_type`이 아니라 yaw offset 기준으로 해석한다.

---

### **==5\. VLM Output Schema==**

VLM output은 네 가지 action 중 하나를 선택한다.

```
go
rotate
stop
request_observation
```

> **5.1 Case A: 현재 관측만으로 waypoint 선택 가능**

```
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
    "point_norm": [0.500, 0.750],
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

> **5.2 Case B: 추가 관측 요청**

현재 관측만으로 waypoint를 선택하기 어렵다면 VLM은 바로 point를 찍지 않고 추가 view를 요청한다.

```
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

> **5.3 Case C: rotate**

VLM이 waypoint를 선택하지 못하고, 실제 robot heading을 바꾸는 것이 필요하다고 판단한 경우다.

```
{
  "schema_version": "nav_vlm_waypoint_v1",
  "action": "rotate",
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
    "valid": false,
    "mode": null,
    "center_yaw_deg": null,
    "step_deg": null,
    "num_views": null,
    "yaw_offsets_deg": null,
    "reason": null
  },
  "reasoning": {
    "decision_reason": "R02_NO_VISIBLE_NAVIGABLE_FLOOR",
    "goal_reason": "F08_NONE_ROTATE_OR_STOP",
    "failure_mode": "no_visible_navigable_floor",
    "short_text": "rotate to find navigable floor"
  },
  "control": {
    "vlm_control_mode": "pause_until_rotation_done",
    "rotate_yaw_deg": -45,
    "ttl_ms": 1000
  },
  "confidence": "medium"
}
```

> **5.4 Case D: stop**

```
{
  "schema_version": "nav_vlm_waypoint_v1",
  "action": "stop",
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
    "navigability": "blocked"
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
    "decision_reason": "S01_COLLISION_OR_HAZARD_RISK",
    "goal_reason": "F08_NONE_ROTATE_OR_STOP",
    "failure_mode": "hazard",
    "short_text": "hazard detected"
  },
  "control": {
    "vlm_control_mode": "hard_stop",
    "rotate_yaw_deg": 0,
    "ttl_ms": 2000
  },
  "confidence": "high"
}
```

---

### ==6\. Observation Request 정의==

VLM은 필요한 관측을 직접 요청할 수 있다.

> **Observation Mode**

```
current_only
- 현재 입력된 view만 사용

directed_view
- 특정 yaw 방향 하나만 추가 관측

directed_sweep
- 특정 방향 주변을 여러 장 관측

full_sweep
- 360도 전체 관측
```

> **Example**

```
"observation_request": {
  "valid": true,
  "mode": "directed_sweep",
  "center_yaw_deg": -90,
  "step_deg": 30,
  "num_views": 3,
  "yaw_offsets_deg": [-120, -90, -60],
  "reason": "goal_bearing_left_but_not_visible"
}
```

> **허용 yaw step**

```
30도
45도
60도
90도
```

자유로운 continuous yaw를 허용하지 않고 discrete set으로 제한한다.
그래야 rule-based label 생성과 ablation study가 쉬워진다.

---

### ==7\. Controlled Reasoning Code==

> **Decision Reason Code**

```
G01_GOAL_ALIGNED_VIEW
- selected view의 relative heading이 coarse goal bearing과 가장 잘 맞음

G02_VISIBLE_FLOOR_TOWARD_GOAL
- goal 방향 view에 주행 가능해 보이는 바닥이 있음

G03_DOORWAY_OR_CORRIDOR_TOWARD_GOAL
- goal 방향으로 이어질 가능성이 높은 문/복도/통로가 있음

G04_CONTINUE_PREVIOUS_GOOD_DIRECTION
- memory상 이전 선택 방향이 유효하고 아직 실패하지 않음

R01_GOAL_OUTSIDE_CURRENT_VIEW
- 현재 관측만으로는 goal 방향을 확인하기 어려움

R02_NO_VISIBLE_NAVIGABLE_FLOOR
- 어떤 view에서도 신뢰할 만한 floor waypoint를 찾기 어려움

R03_LOW_CONFIDENCE_NEED_SCAN
- occlusion, darkness, blur, ambiguous layout으로 scan 필요

R04_LOOP_DETECTED_CHANGE_VIEW
- memory상 같은 branch 반복 또는 deadlock 가능성

S01_COLLISION_OR_HAZARD_RISK
- 충돌, 동적 객체, 낙하, 계단 등 safety risk

S02_TARGET_REACHED_OR_TASK_DONE
- goal 도달 또는 task 완료

S03_OOD_OR_SENSOR_FAILURE
- sensor blackout, severe blur, localization invalid 등
```

> Fine Goal Reason Code

```
F01_CENTER_FLOOR_CLEAR
- 선택 view 중앙 하단 바닥이 가장 안전함

F02_VISIBLE_FLOOR_TOWARD_GOAL
- goal bearing과 맞는 방향의 visible floor를 선택

F03_DOORWAY_ENTRY_POINT
- goal 방향으로 이어질 가능성이 높은 doorway 입구를 선택

F04_CORRIDOR_BRANCH_POINT
- corridor/intersection에서 goal 방향 branch를 선택

F05_AVOID_OBSTACLE_SHIFT_LEFT
- obstacle 때문에 좌측 free space로 shift

F06_AVOID_OBSTACLE_SHIFT_RIGHT
- obstacle 때문에 우측 free space로 shift

F07_MEMORY_AVOID_FAILED_BRANCH
- memory상 실패한 방향을 피해서 선택

F08_NONE_ROTATE_OR_STOP
- action이 rotate / stop / request_observation이라 fine goal 없음
```

---

### ==8\. Schema Validation and Fallback Rule==

VLM output은 반드시 JSON object 하나로 반환한다.
파서가 다음 조건을 만족하지 못하면 output을 invalid로 간주하고 fallback을 수행한다.

```
Invalid output 조건:
- JSON parsing 실패
- schema_version 불일치
- action이 allowed_actions에 없음
- selected_image_point가 이미지 범위를 벗어남
- fine_goal.valid=true인데 selected_view_id 또는 point_px가 없음
- observation_request.valid=true인데 mode, center_yaw_deg, step_deg, num_views가 없음
- request_observation의 step_deg가 allowed_step_deg에 없음
```

Fallback rule은 다음과 같다.

```
1. JSON parsing 실패:
   - 같은 prompt로 1회 retry

2. point 범위 초과:
   - point를 clamp하지 않고 invalid 처리
   - action을 request_observation 또는 rotate로 대체

3. navigable floor가 아닌 point:
   - depth / traversability check에서 reject
   - 해당 방향을 negative candidate로 memory에 기록

4. 반복적으로 같은 branch를 선택:
   - memory.loop_warning 또는 negative edge를 prompt에 포함
   - VLM에 다른 candidate_exit 또는 backtracking을 요구
```

이 validation layer는 VLM이 직접 robot을 제어하지 않고, S2E / PixelNav-style navigation skill에 넘길 수 있는 안전한 fine waypoint만 통과시키기 위한 장치다.
