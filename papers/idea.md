내가 다음의 로봇 navigation시스템을 만드려고 구상중에 있어.
async system인데, 느리게 돌고있는 VLM이 있고, 빠르게 돌고있는 e2e 모델이 있어.
VLM은 입력으로 이미지시퀀스와 어떤 object를 따라가거나 찾으라는 instruction, 또는 이미지 도메인의 coarse goal point가 주어지고 아웃풋은 {action, reasoning, fined goal point} 이렇게 나오게 하고싶어. action은 뭐에 해당하는거냐면 {stop, go, rotate x} 이렇게만 있어서 로봇의 동작을 통제하는거야. stop이 뜨면 무조건 멈추는거고, go가 나오면 정상적인 비동기 동작을 하는거고, rotate x면 비동기 멈추고 로봇이 무조건 목표 x만큼 회전만큼 수행한 뒤 다시 재개야.  보통 stop은 hazard발생시, rotate는 주행 가능 영역의 goal point를 찾지 못했을 때 다시 찾기 위해 수행하는 거라고 이해하면 돼.
reasoning은 action과 fined goal point에 대한 reason을 담으면 돼. 이때 나는 reasoning의 자유도를 너무 두고싶지 않아. 왜냐면 e2e의 강화학습을 수행할 때 VLM output으로 받은 입력을 넣어야하는데 강화학습 시 VLM을 같이 돌리면 병렬처리를 못하기 때문에 시뮬레이션 상황에서 action, reasoning, fined goal point를 룰 기반으로 얻어내고 싶은거거든. 따라서 reasoning이 어느정도 통제가 되어야해. ODD기반으로 왜 rotate를 해야하는지, 왜 fined goal이 정면인지, 왼쪽인지, 오른쪾인지, 왜 멈춰야하는지 등에 대한 reason을 몇개로 추려주고 data schema를 정해줘.

VLM으로 model은 qwen3 VL을 쓸거고, 최근 qwen3 VL 기반 sota논문인 qwen robonav가 나왔어. 이 두 논문 참고해서 가장 호환이 좋을 데이터 스키마, reasoning을 제안해줘

qwen robonav : https://qianwen-res.oss-accelerate.aliyuncs.com/qwenrobot/papers/Qwen_RobotNav.pdf?spm=a2ty_o06.30285417.0.0.1c2ec921fnVHyC&file=Qwen_RobotNav.pdf
qwen3 vl : https://www.alphaxiv.org/abs/2511.21631?chatId=019f32e3-0016-7757-a152-addf54eab0db

자료 구조와 RL에서 재현 가능한 “룰 기반 VLM 출력”까지 같이 맞춰야 해서, 먼저 두 논문의 입력/출력 포맷과 navigation 표현을 확인한 뒤 ODD 기반 reason taxonomy와 JSON schema로 정리하겠습니다.

확인해보니 Qwen-RobotNav 쪽 핵심은 “별도 policy 여러 개”가 아니라 task mode와 observation config를 자연어/파라미터로 바꿔 같은 Qwen3-VL backbone을 재사용하는 구조입니다. 그래서 출력도 자유서술보다 `task_mode + observation_config + enum reason + fine goal` 조합으로 맞추는 게 가장 호환성이 좋습니다.

아래처럼 설계하는 것을 추천합니다. 핵심은 **VLM의 reasoning을 자연어가 아니라 “enum reason code + template text”로 제한**하고, Qwen-RobotNav와 맞게 **task mode / observation config / timestep-view tag / waypoint-compatible goal**을 유지하는 것입니다. Qwen-RobotNav도 Qwen3-VL backbone 위에서 task mode와 observation parameter를 외부에서 바꾸는 구조이고, VLN·PointNav·ObjNav·Tracking 같은 task mode를 같은 모델 인터페이스로 다룹니다.   Qwen3-VL 자체도 긴 interleaved image/video/text context, spatial-temporal modeling, DeepStack, textual timestamp를 강조하므로 image sequence 기반 navigation supervisor와 잘 맞습니다.

## **1. 권장 구조**

VLM 출력은 다음 3계층으로 나누는 게 좋습니다.

**1) hard supervisor action**

`stop > rotate > go` 우선순위를 갖습니다.

**2) symbolic reasoning**

RL 입력에는 자연어 reasoning을 넣지 말고, `decision_reason.code`, `goal_reason.code`, `odd_status`, `hazard_flags`를 one-hot 또는 embedding으로 넣습니다. 자연어 문장은 사람이 보기 위한 template로만 생성합니다.

**3) fine goal point**

Qwen-RobotNav는 8개 waypoint, 각 waypoint가 2D position + heading인 3-DoF trajectory를 예측하는 구조입니다.   지금 시스템은 VLM이 full waypoint 대신 image-domain fine goal point를 주려는 구조이므로, schema에 `fine_goal_point`를 기본으로 두고, 나중에 RobotNav-style waypoint head로 확장할 수 있게 `trajectory_hint`를 optional로 두는 게 호환성이 가장 좋습니다.

---

## **2. 추천 JSON schema**

필드명과 enum 값은 **영어 고정 토큰**으로 두는 것을 추천합니다. Qwen-RobotNav가 timestep/viewpoint를 “Front View”, “Time step” 같은 일반 vocabulary tag로 serialize하는 방식을 쓰기 때문에, 출력도 영문 enum이 더 안정적입니다.

"""
{
  "schema_version": "nav_vlm_decision_v1",
  "frame_ref": {
    "sequence_id": "seq_000123",
    "start_frame_id": 120,
    "end_frame_id": 128,
    "decision_frame_id": 128,
    "timestamp_ms": 391245
  },
  "task": {
    "task_mode": "ObjNav",
    "instruction_type": "find_object",
    "instruction": "find the red chair",
    "target_object": "red chair",
    "coarse_goal_point_norm": [0.55, 0.72]
  },
  "observation_config": {
    "context_policy": "balanced",
    "frame_sample_mode": "latest",
    "visual_token_budget": 2048,
    "temporal_decay": 1.5,
    "camera_weights": {
      "front": 1.0,
      "left": 0.6,
      "right": 0.6,
      "back": 0.2
    }
  },
  "odd": {
    "domain": "indoor_mobile_robot",
    "scene_type": "corridor",
    "odd_status": "IN_ODD",
    "visibility": "clear",
    "traversability": "free",
    "dynamic_agents": "none",
    "sensor_quality": "nominal"
  },
  "decision": {
    "action": "go",
    "rotate_yaw_deg": 0,
    "priority": "mission",
    "confidence": 0.82
  },
  "reasoning": {
    "decision_reason": {
      "code": "G01_PATH_CLEAR_TO_GOAL",
      "class": "go",
      "template_id": "go_path_clear"
    },
    "goal_reason": {
      "code": "F01_CENTER_ALIGNED_CLEAR",
      "region": "center",
      "template_id": "goal_center_clear"
    },
    "evidence": {
      "target_visibility": "visible",
      "free_space": "center",
      "obstacle_relation": "none_blocking",
      "last_seen_target_bearing": "center",
      "hazard_flags": []
    },
    "nlg": "The path toward the target is clear, so continue with a center fine goal."
  },
  "fine_goal_point": {
    "valid": true,
    "source": "coarse_goal_refined",
    "point_norm": [0.53, 0.74],
    "point_px": [678, 533],
    "region": "center",
    "navigability": "free",
    "confidence": 0.79
  },
  "trajectory_hint": {
    "valid": false,
    "waypoints_2d_heading": []
  },
  "control": {
    "vlm_control_mode": "resume_async_e2e",
    "ttl_ms": 500,
    "stale_policy": "ignore_go_keep_stop"
  }
}
"""

task_mode는 Qwen-RobotNav 호환을 위해 아래 네 가지를 기본으로 두면 됩니다. Qwen-RobotNav도 VLN은 route instruction following, PointNav는 spatial target, ObjNav는 object/category search, Tracking은 최근 관측 기반 target lock으로 정의합니다. 
"""
VLN | PointNav | ObjNav | Tracking
"""
observation_config도 Qwen-RobotNav와 맞춰 visual_token_budget, temporal_decay, camera_weights, frame_sample_mode를 유지하는 것이 좋습니다. 논문에서 이 네 축을 task-adaptive observation encoding의 핵심 control parameter로 둡니다. 

## **3. Action reason code taxonomy**

아래 정도로 제한하면 RL에서 rule-based reproduction이 쉽고, VLM prompt도 안정적입니다.

code	action	rule-based condition	의미
G01_PATH_CLEAR_TO_GOAL	go	coarse/fine goal 방향에 free-space가 충분함	목표 방향으로 계속 진행
G02_TARGET_VISIBLE_APPROACHABLE	go	target bbox가 보이고 bbox bottom-center 주변이 traversable	보이는 target 쪽으로 접근
G03_EXPLORE_FRONTIER	go	target은 안 보이지만 전방/좌/우 중 frontier free-space가 있음	탐색 계속
G04_TRACK_TARGET_RECENT	go	tracking target이 최근 frame에서 안정적으로 관측됨	recency 기반 추적
G05_AVOIDANCE_LOCAL_SHIFT	go	중앙은 막혔지만 좌/우 회피 goal이 traversable	작은 lateral shift로 회피
R01_NO_FREE_SPACE_IN_FOV	rotate	현재 FOV에 navigable mask가 임계치 이하	주행 가능 영역 재탐색
R02_TARGET_NOT_VISIBLE_NEED_SCAN	rotate	ObjNav/Tracking에서 target 미검출, hazard 없음	target search 회전
R03_GOAL_BEARING_OUTSIDE_FOV	rotate	coarse goal 또는 last-seen target bearing이 FOV 밖	목표 방향으로 yaw 회전
R04_VIEW_OCCLUDED_OR_AMBIGUOUS	rotate	occlusion/blur/glare로 goal 판단 confidence 낮음	시야 재확보
R05_TRACK_LOST_RECENTLY	rotate	tracking target이 직전에는 보였지만 현재 lost	last-seen 방향으로 recovery rotate
S01_COLLISION_RISK	stop	TTC, depth, obstacle distance가 safety threshold 위반	즉시 충돌 위험
S02_HUMAN_ANIMAL_CLOSE	stop	사람/동물/동적 객체가 safety zone 내부	social/safety stop
S03_DROP_STAIRS_UNTRAVERSABLE	stop	stairs, cliff, drop-off, water 등 ODD 밖 traversability	주행 불가 지형
S04_SENSOR_DEGRADED_OUT_OF_ODD	stop	image blackout, severe blur, desync, localization invalid	인지 불능
S05_TARGET_REACHED_OR_TASK_DONE	stop	target reached 또는 instruction 완료	정상 종료 stop
S06_RULE_REQUIRED_STOP	stop	traffic light, closed gate, stop sign, restricted zone 등 domain rule	규칙 기반 정지

추천 우선순위는 아래처럼 고정하세요.
"""
S01/S02/S03/S04/S06 > S05 > R01/R03/R04/R05/R02 > G02/G01/G04/G05/G03
"""
즉, safety stop은 항상 rotate/go보다 우선입니다.

## **4. Fine goal reason code taxonomy**

`decision_reason`은 “왜 action인가”이고, `goal_reason`은 “왜 fine goal이 그 위치인가”입니다. 이 둘을 분리해야 RL에서 action interrupt와 local guidance를 따로 학습하기 쉽습니다.

code	region	rule-based condition	fine goal source
F01_CENTER_ALIGNED_CLEAR	center	coarse goal 또는 target이 중앙 free-space와 정렬	coarse_goal_refined
F02_LEFT_TARGET_OR_LANDMARK	left	target/landmark가 좌측에 보이고 path free	target_ground_contact
F03_RIGHT_TARGET_OR_LANDMARK	right	target/landmark가 우측에 보이고 path free	target_ground_contact
F04_LEFT_FREE_SPACE_BEST	left	좌측 navigable score가 가장 높음	frontier_exploration
F05_RIGHT_FREE_SPACE_BEST	right	우측 navigable score가 가장 높음	frontier_exploration
F06_SHIFT_LEFT_TO_AVOID	left	중앙/우측 obstacle, 좌측 corridor free	avoidance_shift
F07_SHIFT_RIGHT_TO_AVOID	right	중앙/좌측 obstacle, 우측 corridor free	avoidance_shift
F08_FRONTIER_SCAN_POINT	center/left/right	target 미검출, frontier 중 최고 score	frontier_exploration
F09_NONE_STOP_OR_ROTATE	none	action이 stop/rotate라 goal point 무효	none

`point_norm`은 image coordinate 기준 `[x, y]`, 원점은 top-left, 범위는 `[0, 1]`로 고정하세요. navigation fine goal은 보통 바닥/주행 가능 영역 위에 찍혀야 하므로, object target이면 bbox center가 아니라 **bbox bottom-center를 free-space mask 위로 project/snap**하는 것을 추천합니다.

## **5. ODD schema는 이렇게 작게 유지**

ODD는 너무 세분화하면 rule-based simulator가 어려워집니다. 아래 6개 축이면 충분합니다.
"""
"odd": {
  "domain": "indoor_mobile_robot",
  "scene_type": "corridor",
  "odd_status": "IN_ODD",
  "visibility": "clear",
  "traversability": "free",
  "dynamic_agents": "none",
  "sensor_quality": "nominal"
}
"""
허용 enum은 다음 정도가 적당합니다.
domain:
  indoor_mobile_robot | warehouse_robot | sidewalk_robot | autonomous_vehicle

scene_type:
  corridor | room | doorway | intersection | open_area | narrow_passage | stairs_area | unknown

odd_status:
  IN_ODD | DEGRADED_IN_ODD | OUT_OF_ODD

visibility:
  clear | dim | glare | motion_blur | occluded | low_visibility

traversability:
  free | narrow | partially_blocked | blocked | stairs_or_drop | slippery_or_water | unknown

dynamic_agents:
  none | far | crossing_path | near_safety_zone | unpredictable

sensor_quality:
  nominal | low_confidence | desynced | overexposed | underexposed | blackout

odd_status rule은 간단히 이렇게 두면 됩니다.
"""
OUT_OF_ODD:
  S03_DROP_STAIRS_UNTRAVERSABLE 또는 S04_SENSOR_DEGRADED_OUT_OF_ODD에 해당

DEGRADED_IN_ODD:
  visibility가 dim/glare/occluded/motion_blur이지만 partial decision 가능
  traversability가 narrow/partially_blocked
  target confidence 낮음

IN_ODD:
  visibility clear, traversability free/narrow, sensor nominal
"""

## **6. Rule-based simulator label 생성 규칙**

시뮬레이터에서 VLM 없이 label을 만들려면, 아래 순서로 deterministic하게 생성하면 됩니다.

"""
def make_vlm_label(obs, task, memory):
    # 1. Safety first
    if obs.collision_risk or obs.ttc < TTC_STOP:
        return stop("S01_COLLISION_RISK")
    if obs.human_or_animal_in_safety_zone:
        return stop("S02_HUMAN_ANIMAL_CLOSE")
    if obs.has_drop_or_stairs_or_water:
        return stop("S03_DROP_STAIRS_UNTRAVERSABLE")
    if obs.sensor_quality in ["blackout", "desynced", "severe_blur"]:
        return stop("S04_SENSOR_DEGRADED_OUT_OF_ODD")
    if task.target_reached:
        return stop("S05_TARGET_REACHED_OR_TASK_DONE")

    # 2. Compute free-space and target evidence
    free = obs.navigable_mask
    target = obs.target_detection
    coarse = task.coarse_goal_point_norm

    # 3. If no usable navigable region, rotate
    if free.area_ratio < MIN_FREE_AREA:
        yaw = choose_scan_yaw(memory, task)
        return rotate(yaw, "R01_NO_FREE_SPACE_IN_FOV")

    # 4. If target/search goal is outside FOV or lost, rotate
    if task.mode in ["ObjNav", "Tracking"] and not target.visible:
        yaw = choose_last_seen_or_sweep_yaw(memory)
        code = "R05_TRACK_LOST_RECENTLY" if memory.target_seen_recently else "R02_TARGET_NOT_VISIBLE_NEED_SCAN"
        return rotate(yaw, code)

    # 5. If goal visible/reachable, go with refined fine goal
    if target.visible and is_reachable(target.ground_contact, free):
        p = snap_to_free_space(target.ground_contact, free)
        return go(p, "G02_TARGET_VISIBLE_APPROACHABLE", goal_reason_from_region(p))

    if coarse is not None and is_reachable(coarse, free):
        p = snap_to_free_space(coarse, free)
        return go(p, "G01_PATH_CLEAR_TO_GOAL", goal_reason_from_region(p))

    # 6. Avoidance/frontier fallback
    p = best_frontier_or_lateral_goal(free, obs.obstacles)
    if p is not None:
        if is_lateral_avoidance(p, obs.obstacles):
            return go(p, "G05_AVOIDANCE_LOCAL_SHIFT", avoidance_goal_reason(p))
        return go(p, "G03_EXPLORE_FRONTIER", frontier_goal_reason(p))

    # 7. Last fallback
    return rotate(45, "R04_VIEW_OCCLUDED_OR_AMBIGUOUS")
"""

rotate_yaw_deg는 반드시 부호 convention을 고정하세요.

"""
positive yaw: left / counter-clockwise in robot frame
negative yaw: right / clockwise in robot frame

allowed values:
-180, -90, -60, -45, -30, 30, 45, 60, 90, 180
"""
처음에는 continuous rotate x보다 위 discrete set을 쓰는 게 RL과 rule label 모두 안정적입니다.

## **7. Prompt format 추천**

Qwen-RobotNav가 natural-language embodiment preamble과 viewpoint/timestep tag를 사용한다는 점을 따라가면 좋습니다. 논문에서도 robot/car 같은 embodiment를 system prompt preamble로 알려주며, 새 platform도 prompt template만 바꾸면 된다고 설명합니다.

"""
System:
Imagine you are a robot programmed for navigation tasks.
You are a slow semantic supervisor for an asynchronous navigation system.
Return only valid JSON following nav_vlm_decision_v1.
Do not output free-form chain-of-thought.
Use only the allowed reason codes.

Task:
task_mode: ObjNav
instruction: find the red chair
coarse_goal_point_norm: [0.55, 0.72]

Observation:
Time step -3 Front View <image>
Time step -2 Front View <image>
Time step -1 Front View <image>
Time step 0 Front View <image>

Allowed actions:
go, stop, rotate

Allowed decision_reason.code:
G01_PATH_CLEAR_TO_GOAL, ...
"""
Qwen3-VL에는 thinking / non-thinking variant가 있으므로, production supervisor에는 non-thinking 또는 “final JSON only” decoding을 쓰는 편이 좋습니다. Qwen3-VL technical report는 thinking variant가 복잡 reasoning에 강하다고 하지만, 여기서는 RL 재현성과 schema 안정성이 더 중요합니다.  

## **8. 예시 출력 3개**

### **Go**
{
  "schema_version": "nav_vlm_decision_v1",
  "task": {
    "task_mode": "ObjNav",
    "instruction_type": "find_object",
    "instruction": "find the red chair",
    "target_object": "red chair",
    "coarse_goal_point_norm": null
  },
  "odd": {
    "domain": "indoor_mobile_robot",
    "scene_type": "room",
    "odd_status": "IN_ODD",
    "visibility": "clear",
    "traversability": "free",
    "dynamic_agents": "none",
    "sensor_quality": "nominal"
  },
  "decision": {
    "action": "go",
    "rotate_yaw_deg": 0,
    "priority": "mission",
    "confidence": 0.86
  },
  "reasoning": {
    "decision_reason": {
      "code": "G02_TARGET_VISIBLE_APPROACHABLE",
      "class": "go",
      "template_id": "go_target_visible"
    },
    "goal_reason": {
      "code": "F03_RIGHT_TARGET_OR_LANDMARK",
      "region": "right",
      "template_id": "goal_right_target"
    },
    "evidence": {
      "target_visibility": "visible",
      "free_space": "right",
      "obstacle_relation": "none_blocking",
      "last_seen_target_bearing": "right",
      "hazard_flags": []
    },
    "nlg": "The target is visible on the right and the approach area is navigable."
  },
  "fine_goal_point": {
    "valid": true,
    "source": "target_ground_contact",
    "point_norm": [0.71, 0.76],
    "point_px": [909, 547],
    "region": "right",
    "navigability": "free",
    "confidence": 0.83
  },
  "control": {
    "vlm_control_mode": "resume_async_e2e",
    "ttl_ms": 500,
    "stale_policy": "ignore_go_keep_stop"
  }
}

Rotate
"""
{
  "schema_version": "nav_vlm_decision_v1",
  "task": {
    "task_mode": "ObjNav",
    "instruction_type": "find_object",
    "instruction": "find the red chair",
    "target_object": "red chair",
    "coarse_goal_point_norm": null
  },
  "odd": {
    "domain": "indoor_mobile_robot",
    "scene_type": "corridor",
    "odd_status": "DEGRADED_IN_ODD",
    "visibility": "occluded",
    "traversability": "unknown",
    "dynamic_agents": "none",
    "sensor_quality": "low_confidence"
  },
  "decision": {
    "action": "rotate",
    "rotate_yaw_deg": 45,
    "priority": "recovery",
    "confidence": 0.74
  },
  "reasoning": {
    "decision_reason": {
      "code": "R02_TARGET_NOT_VISIBLE_NEED_SCAN",
      "class": "rotate",
      "template_id": "rotate_scan_target"
    },
    "goal_reason": {
      "code": "F09_NONE_STOP_OR_ROTATE",
      "region": "none",
      "template_id": "goal_none_rotate"
    },
    "evidence": {
      "target_visibility": "not_visible",
      "free_space": "insufficient",
      "obstacle_relation": "unknown",
      "last_seen_target_bearing": "left",
      "hazard_flags": []
    },
    "nlg": "The target is not visible, so rotate left to scan for it."
  },
  "fine_goal_point": {
    "valid": false,
    "source": "none",
    "point_norm": null,
    "point_px": null,
    "region": "none",
    "navigability": "unknown",
    "confidence": 0.0
  },
  "control": {
    "vlm_control_mode": "pause_e2e_until_rotation_done",
    "ttl_ms": 1000,
    "stale_policy": "execute_once"
  }
}
"""

Stop
"""
{
  "schema_version": "nav_vlm_decision_v1",
  "task": {
    "task_mode": "VLN",
    "instruction_type": "follow_instruction",
    "instruction": "go through the hallway",
    "target_object": null,
    "coarse_goal_point_norm": [0.50, 0.75]
  },
  "odd": {
    "domain": "indoor_mobile_robot",
    "scene_type": "corridor",
    "odd_status": "OUT_OF_ODD",
    "visibility": "clear",
    "traversability": "blocked",
    "dynamic_agents": "near_safety_zone",
    "sensor_quality": "nominal"
  },
  "decision": {
    "action": "stop",
    "rotate_yaw_deg": 0,
    "priority": "safety",
    "confidence": 0.93
  },
  "reasoning": {
    "decision_reason": {
      "code": "S02_HUMAN_ANIMAL_CLOSE",
      "class": "stop",
      "template_id": "stop_dynamic_agent_close"
    },
    "goal_reason": {
      "code": "F09_NONE_STOP_OR_ROTATE",
      "region": "none",
      "template_id": "goal_none_stop"
    },
    "evidence": {
      "target_visibility": "irrelevant",
      "free_space": "blocked",
      "obstacle_relation": "dynamic_agent_in_safety_zone",
      "last_seen_target_bearing": "unknown",
      "hazard_flags": ["human_close"]
    },
    "nlg": "A dynamic agent is inside the safety zone, so the robot must stop."
  },
  "fine_goal_point": {
    "valid": false,
    "source": "none",
    "point_norm": null,
    "point_px": null,
    "region": "none",
    "navigability": "blocked",
    "confidence": 0.0
  },
  "control": {
    "vlm_control_mode": "hard_stop",
    "ttl_ms": 2000,
    "stale_policy": "keep_until_cleared"
  }
}
"""
## **9. 최종 추천**

가장 중요한 설계 결정은 이것입니다.

**VLM output의 `reasoning.nlg`는 사람이 읽는 로그일 뿐이고, RL에는 `decision_reason.code`, `goal_reason.code`, `odd_status`, `hazard_flags`, `fine_goal_point.point_norm`, `fine_goal_point.region`, `action`만 넣으세요.**

그러면 시뮬레이션에서는 segmentation/depth/object detector/GT state로 같은 code를 rule-based로 만들 수 있고, 실제 배포에서는 Qwen3-VL이 같은 schema를 채우게 할 수 있습니다. Qwen-RobotNav가 trajectory-only 학습만 하면 reactive action mapper로 collapse될 수 있어 vision-language reasoning co-training을 섞는다고 보고한 점을 보면, 당신의 구조에서도 VLM의 language reasoning을 완전히 자유서술로 두기보다 **작은 symbolic vocabulary로 남겨 e2e policy의 semantic prior로 쓰는 방식**이 가장 안전합니다.