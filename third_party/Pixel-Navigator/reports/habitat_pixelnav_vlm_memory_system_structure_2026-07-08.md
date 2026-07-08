# Habitat + PixelNav + VLM Grounded Goal + v5 Memory 구조 설명

Updated: 2026-07-08

## 1. 이 문서의 목적

이 문서는 현재까지 개발한 Habitat + PixelNav + VLM grounded goal + memory integration의 **구조**를 설명한다.

성능 결과 보고서가 아니라, 선임님이 코드를 보지 않아도 다음을 이해할 수 있게 정리하는 문서이다.

```text
무엇이 어디에 구현되어 있는지
어떤 데이터가 어떤 순서로 흐르는지
PixelNav와 VLM, memory graph가 어떤 contract로 연결되는지
official Habitat benchmark에서는 어떤 wrapper 구조로 실행되는지
```

현재 구조의 핵심은 다음 한 줄로 요약할 수 있다.

```text
Habitat official env를 실제 robot backend처럼 감싸고,
VLM이 image point goal을 고르면,
PixelNav가 local rollout을 수행하고,
그 결과를 ActionOutcome으로 변환하여
qwen_nav_memory_framework_v5의 memory graph를 업데이트한다.
```

## 2. 두 repository의 역할

이번 구조는 두 repository를 함께 사용한다.

### 2.1 Pixel-Navigator

위치:

```text
/home/icra/Pixel-Navigator
```

역할:

```text
Habitat official benchmark env 실행
PixelNav policy checkpoint 로드
goal image / goal mask 생성
PixelNav action rollout 수행
official SR/SPL metric 저장
benchmark 결과 artifact 저장
```

이번에 추가한 중심 entrypoint:

```text
/home/icra/Pixel-Navigator/voca_memory_benchmark.py
```

이 파일은 기존 `objnav_benchmark.py`를 직접 바꾸지 않고, VOCA memory integration용 별도 runner로 동작한다.

### 2.2 voca-s2e

위치:

```text
/home/icra/voca-s2e
```

역할:

```text
VLM output schema 정의
memory prompt / memory context 설계
ActionOutcome contract 정의
qwen_nav_memory_framework_v5 제공
memory graph node-edge update
memory graph visualization
```

이번 official benchmark runner가 import하는 memory framework:

```text
/home/icra/voca-s2e/qwen_nav_memory_framework_v5
```

중요한 내부 파일:

```text
nav_memory_qwen/agent.py
nav_memory_qwen/robot_backend.py
nav_memory_qwen/memory_graph.py
nav_memory_qwen/schema.py
nav_memory_qwen/vlm_client.py
```

## 3. 전체 실행 구조

전체 실행 흐름은 다음과 같다.

```text
Official Habitat episode
  |
  v
Habitat Env reset
  |
  v
HabitatEnvMemoryBackend
  |
  v
NavMemoryAgent.step()
  |
  +--> get_robot_state()
  |
  +--> get_observation()
  |
  +--> MemoryGraph pre-update / localization
  |
  +--> build VLM input
  |
  +--> Qwen or heuristic VLM decide()
  |
  +--> selected_view_type + selected_image_point
  |
  v
HabitatEnvMemoryBackend.execute_waypoint()
  |
  +--> current RGB를 goal image로 사용
  |
  +--> selected_image_point를 goal mask로 변환
  |
  +--> PixelNav Policy_Agent rollout
  |
  +--> Habitat env.step(action)
  |
  v
ActionOutcome
  |
  v
NavMemoryAgent._update_memory_after_outcome()
  |
  v
MemoryGraph node / edge update
  |
  v
Habitat official metrics read
  |
  v
summary.json / metrics.csv / memory_graph.json 저장
```

즉 Habitat가 실제 world 역할을 하고, `HabitatEnvMemoryBackend`가 memory agent 입장에서는 robot backend 역할을 한다.

## 4. 주요 runtime 객체

### 4.1 Habitat Env

생성 위치:

```text
voca_memory_benchmark.py
  create_official_objnav_env(...)
  create_official_pointnav_env(...)
```

역할:

```text
official episode reset
official action step
official collision / success / SPL metric 계산
RGB / depth / GPS / compass / objectgoal or pointgoal observation 제공
```

중요한 점은 movement가 별도 mock이 아니라 official Habitat env의 `env.step(...)`을 통해 발생한다는 것이다.

### 4.2 HabitatEnvMemoryBackend

위치:

```text
/home/icra/Pixel-Navigator/voca_memory_benchmark.py
  class HabitatEnvMemoryBackend
```

역할:

```text
qwen_nav_memory_framework_v5의 RobotBackend protocol을 Habitat env에 맞춰 구현
현재 robot pose를 Habitat agent state에서 읽음
현재 RGB observation을 memory framework의 Observation으로 변환
VLM이 선택한 image point를 PixelNav goal mask로 변환
PixelNav rollout을 Habitat env action으로 실행
rollout 결과를 ActionOutcome으로 반환
```

구현된 backend method:

```text
get_robot_state()
get_observation()
capture_views(...)
rotate(...)
execute_waypoint(...)
stop()
```

이 class가 두 시스템 사이의 adapter이다.

```text
Habitat env API
  <-> HabitatEnvMemoryBackend
  <-> qwen_nav_memory_framework_v5 RobotBackend protocol
```

### 4.3 NavMemoryAgent

위치:

```text
/home/icra/voca-s2e/qwen_nav_memory_framework_v5/nav_memory_qwen/agent.py
  class NavMemoryAgent
```

역할:

```text
Observe -> Memory -> VLM -> Act -> Memory update cycle 관리
VLM input 구성
VLM output sanitize
go / rotate / stop / request_observation 실행 분기
ActionOutcome을 memory graph update로 반영
episode step log 저장
```

핵심 method:

```text
NavMemoryAgent.step(...)
NavMemoryAgent.save_run(...)
```

`step(...)` 하나가 현재 구조에서 한 번의 closed-loop navigation decision이다.

### 4.4 VLM Client

위치:

```text
/home/icra/Pixel-Navigator/voca_memory_benchmark.py
  class PixelNavFriendlyHeuristicVLMClient
  class FallbackVLMClient

/home/icra/voca-s2e/qwen_nav_memory_framework_v5/nav_memory_qwen/vlm_client.py
  OpenAICompatibleVLMClient
```

역할:

```text
VLM input을 읽고 navigation action 결정
go인 경우 selected_view_type과 selected_image_point 출력
Qwen JSON 실패 시 heuristic fallback으로 episode crash 방지
```

현재 사용 가능한 모드:

```text
--vlm heuristic
--vlm qwen
```

`--vlm heuristic`은 benchmark runner 자체가 잘 도는지 확인하기 위한 deterministic fallback이다.

`--vlm qwen`은 OpenAI-compatible API endpoint를 통해 Qwen3-VL을 호출한다.

### 4.5 PixelNav Policy

생성 위치:

```text
voca_memory_benchmark.py
  make_pixelnav_policy_factory(...)
```

내부 로드:

```text
from policy_agent import Policy_Agent
from constants import POLICY_CHECKPOINT
```

역할:

```text
goal_image + goal_mask를 받아 local navigation policy reset
현재 RGB image를 입력으로 PixelNav action 출력
action id를 Habitat action으로 변환
```

현재 action mapping:

```text
0: stop
1: move_forward
2: turn_left
3: turn_right
4: look_up
5: look_down
```

official Habitat action space가 지원하지 않는 action은 crash시키지 않고 `unsupported_action`으로 기록한다.

### 4.6 ActionOutcome

위치:

```text
/home/icra/voca-s2e/qwen_nav_memory_framework_v5/nav_memory_qwen/robot_backend.py
  class ActionOutcome
```

역할:

```text
fast navigation skill 또는 robot backend 실행 결과를 memory framework가 이해하는 형태로 표현
```

주요 field:

```text
action
success
collision
moved_distance_m
rotated_deg
odom_delta
message
raw
```

PixelNav rollout은 최종적으로 이 구조로 변환된다.

```text
PixelNav rollout steps
  -> total_moved_distance_m
  -> collision_count
  -> unsupported_action
  -> RelativePose2D odom_delta
  -> ActionOutcome
```

`NavMemoryAgent`는 backend가 PixelNav인지 실제 robot인지 알 필요 없이 `ActionOutcome`만 보고 memory를 업데이트한다.

### 4.7 MemoryGraph

위치:

```text
/home/icra/voca-s2e/qwen_nav_memory_framework_v5/nav_memory_qwen/memory_graph.py
```

역할:

```text
place/situation node 저장
relative pose edge 저장
negative memory / deadlock edge 저장
same-place / revisit relation 저장
VLM prompt에 넣을 compact memory context 구성
```

v5 memory graph의 중요한 방향:

```text
global pose를 직접 저장하지 않음
node 간 relative SE(2) edge를 저장
revisit node를 무조건 삭제/merge하지 않고 soft merge constraint로 보존
false-positive merge를 줄이기 위해 geometric filtering 사용
```

즉 memory는 단순 방문 리스트가 아니라, node-edge 기반 topo-metric graph이다.

## 5. 데이터 contract

### 5.1 VLM output contract

VLM은 navigation action을 schema에 맞춰 반환해야 한다.

중요한 field:

```text
action: go | rotate | stop | request_observation
selected_view_type
selected_view_id
selected_image_point
fine_goal
reasoning
confidence
control
memory_ops
```

`go` action일 때 가장 중요한 값은 다음이다.

```text
selected_view_type
selected_image_point
```

이 값이 PixelNav local goal로 변환된다.

### 5.2 PixelNav goal contract

PixelNav는 다음 입력을 받는다.

```text
goal_image: current RGB image
goal_mask: selected_image_point 주변에 만든 binary mask
```

현재는 selected pixel 주변 반경을 mask로 만든다.

```text
selected_image_point = [u, v]
mask_radius = 5
goal_mask[v-radius:v+radius, u-radius:u+radius] = 255
```

### 5.3 Backend outcome contract

PixelNav rollout 결과는 바로 memory graph에 넣지 않고 `ActionOutcome`으로 변환한다.

```text
ActionOutcome(
  action="go",
  success=...,
  collision=...,
  moved_distance_m=...,
  odom_delta=RelativePose2D(...),
  raw={ rollout details }
)
```

이 contract 덕분에 이후 실제 robot backend로 바꿔도 memory framework 쪽은 같은 구조를 유지할 수 있다.

### 5.4 Memory graph artifact contract

episode 종료 후 다음 파일이 저장된다.

```text
episode_0000/memory/memory_graph.json
episode_0000/memory/steps.json
episode_0000/episode_summary.json
voca_memory_*_benchmark_summary.json
voca_memory_*_benchmark_metrics.csv
```

`memory_graph.json`은 visualizer와 후속 분석의 입력이다.

## 6. ObjectNav와 PointNav의 차이

두 task는 VLM-memory-agent 구조는 동일하고, coarse goal 생성만 다르다.

### 6.1 ObjectNav

ObjectNav에서는 Habitat episode의 target object category를 coarse goal로 사용한다.

```text
episode.object_category
  -> target_object
  -> NavMemoryAgent.step(target_object=...)
```

VLM은 object target을 향해 다음 fine goal을 고른다.

### 6.2 PointNav

PointNav에서는 Habitat episode의 goal position을 map goal로 사용한다.

```text
episode.goals[0].position
  -> goal_map_xy
  -> NavMemoryAgent.step(goal_map_xy=...)
```

agent는 현재 robot state와 goal position을 이용해 coarse bearing/distance를 만들고, VLM은 그 방향을 참고해 image point goal을 고른다.

### 6.3 공통 구조

두 task 모두 그 다음은 동일하다.

```text
coarse goal
  -> VLM selected image point
  -> PixelNav local rollout
  -> ActionOutcome
  -> MemoryGraph update
  -> Habitat metrics
```

## 7. Official benchmark runner 구조

`voca_memory_benchmark.py`의 상위 구조는 다음과 같다.

```text
main()
  |
  +-- create_official_objnav_env(...)
  |     -> Habitat ObjectNav env
  |
  +-- create_official_pointnav_env(...)
  |     -> Habitat PointNav env
  |
  +-- run_objnav_memory_benchmark(...)
  |
  +-- run_pointnav_memory_benchmark(...)
```

각 benchmark loop는 다음 순서로 돈다.

```text
for episode_index in eval_episodes:
    env.reset()
    HabitatEnvMemoryBackend 생성
    VLM client 생성
    NavMemoryAgent 생성

    for step_index in max_agent_steps:
        agent.step(...)
        if stop/error/done:
            break

    agent.save_run(memory_dir)
    env.get_metrics()
    episode_summary.json 저장

write_benchmark_summary(...)
```

## 8. Dataset path 구조

현재 Pixel-Navigator는 다음 root를 기준으로 Habitat data를 본다.

```text
/home/icra/habitat_data
```

Pixel-Navigator 내부에서는 다음 symlink로 접근할 수 있다.

```text
/home/icra/Pixel-Navigator/data
  -> /home/icra/habitat_data
```

HM3D scene asset:

```text
/home/icra/habitat_data/scene_datasets/hm3d
/home/icra/habitat_data/scene_datasets/hm3d_v0.2
```

ObjectNav episode split:

```text
/home/icra/habitat_data/datasets/objectnav/hm3d/v2/{split}/{split}.json.gz
```

PointNav episode split:

```text
/home/icra/habitat_data/datasets/pointnav/hm3d/v1/{split}/{split}.json.gz
```

## 9. Memory graph visualization과의 연결

memory graph visualizer는 benchmark runner와 직접 결합되어 있지 않고, output artifact를 후처리로 읽는다.

입력:

```text
episode_0000/memory/memory_graph.json
episode_0000/memory/steps.json
```

출력 예시:

```text
memory_graph.png
memory_graph_reconstruction.gif
memory_graph_reconstruction.mp4
```

즉 benchmark 실행과 visualization은 다음처럼 분리되어 있다.

```text
Benchmark runner
  -> memory_graph.json 저장
  -> visualizer가 graph json을 읽어 node-edge reconstruction 표시
```

이 구조 덕분에 benchmark를 돌리는 동안 graph를 계속 축적하고, 후처리로 graph가 어떻게 구성됐는지 확인할 수 있다.

## 10. 지금 구조에서 중요한 설계 선택

### 10.1 기존 Pixel-Navigator를 직접 뜯지 않음

기존 benchmark 파일을 직접 바꾸기보다 `voca_memory_benchmark.py`를 별도 entrypoint로 만들었다.

이유:

```text
기존 PixelNav baseline과 비교하기 쉬움
VOCA memory integration 실험을 독립적으로 관리 가능
문제가 생겼을 때 기존 코드 영향 최소화
```

### 10.2 HabitatEnvMemoryBackend를 adapter로 둠

memory framework는 robot/backend agnostic하게 유지한다.

```text
Memory framework는 RobotBackend protocol만 안다.
HabitatEnvMemoryBackend가 Habitat env를 그 protocol에 맞춘다.
```

이 구조는 이후 실제 robot이나 ROS backend로 옮길 때도 유리하다.

### 10.3 PixelNav rollout을 ActionOutcome으로 고정

PixelNav 세부 action log를 memory graph에 직접 넣지 않는다.

대신 다음 contract로 요약한다.

```text
success / collision / moved_distance / odom_delta / raw
```

이렇게 하면 memory update logic은 low-level controller 종류와 분리된다.

### 10.4 v5 memory는 false-positive merge 방지 중심

v5 memory는 revisit을 단순 duplicate 제거로 처리하지 않는다.

```text
revisit node 보존
same-place constraint edge 추가
relative baseline/parallax 유지
geometric plausibility gate 사용
```

목적은 graph를 예쁘게 압축하는 것보다 false-positive outlier merge를 피하는 것이다.

## 11. 현재 구조에서 아직 성능으로 말하면 안 되는 부분

현재 구조는 official benchmark를 돌릴 수 있는 형태까지 연결된 것이다.

하지만 다음은 아직 full evaluation으로 확인해야 한다.

```text
SR/SPL이 실제로 좋아지는지
memory graph가 long episode에서 유효하게 활용되는지
Qwen grounded goal이 heuristic보다 좋은지
PixelNav local rollout이 다양한 scene에서 안정적인지
memory merge/negative memory가 실패 경로를 줄이는지
```

따라서 현재 보고할 때는 다음 표현이 정확하다.

```text
HM3D official ObjectNav/PointNav benchmark runner 구조는 준비되었고,
end-to-end smoke에서 env 실행, PixelNav rollout, memory graph update,
official metric export까지 확인했다.
다만 SR/SPL 성능 평가는 multi-episode run 이후 판단해야 한다.
```

## 12. 한 장 요약

최종 구조:

```text
Habitat official env
  |
  | observation / agent state / official metric
  v
HabitatEnvMemoryBackend
  |
  | RobotBackend protocol
  v
NavMemoryAgent(v5)
  |
  | memory context + observation + coarse goal
  v
Qwen or heuristic VLM
  |
  | selected_view_type + selected_image_point
  v
PixelNav Policy_Agent
  |
  | local action rollout through Habitat env.step(...)
  v
ActionOutcome
  |
  | success / collision / odom_delta
  v
MemoryGraph node-edge update
  |
  v
memory_graph.json + steps.json + SR/SPL summary
```

이 구조가 현재까지의 개발 결과이다.
