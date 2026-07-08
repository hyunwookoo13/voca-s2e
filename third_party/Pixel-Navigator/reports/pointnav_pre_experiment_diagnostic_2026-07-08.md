# PointNav Pre-Experiment Diagnostic Report

Updated: 2026-07-08

## 1. 목적

내일 full benchmark / ablation을 돌리기 전에, HM3D PointNav에서 현재 Habitat + PixelNav + VLM grounded goal + v5 memory integration이 어떤 병목을 갖는지 먼저 확인했다.

PointNav를 먼저 본 이유는 다음과 같다.

```text
PointNav는 goal position이 명확하다.
ObjectNav보다 semantic ambiguity가 적다.
따라서 실패 원인을 VLM/object recognition 문제가 아니라 navigation loop 문제로 분리하기 쉽다.
```

이번 진단의 목표는 SR/SPL을 높게 보고하는 것이 아니라, 다음을 확인하는 것이다.

```text
official HM3D PointNav split에서 runner가 안정적으로 도는지
PixelNav rollout이 실제 이동을 만드는지
distance_to_goal이 줄어드는 방향으로 움직이는지
memory graph node/edge가 쌓이는지
내일 실험 전 바로 손봐야 할 병목이 무엇인지
```

## 2. 이번에 추가한 보강

### 2.1 PointNav bearing-first heuristic

기존 `--vlm heuristic`은 memory candidate/frontier를 먼저 고려하는 구조라, PointNav에서도 goal bearing보다 front candidate를 우선하는 경우가 있었다.

PointNav에서는 GPS goal bearing이 있으므로, 진단용으로 goal bearing을 먼저 따르는 heuristic을 추가했다.

추가 위치:

```text
/home/icra/Pixel-Navigator/voca_memory_benchmark.py
  PointNavBearingHeuristicVLMClient
```

사용법:

```bash
--vlm pointnav-bearing
```

동작:

```text
if goal bearing is outside rotate threshold:
    rotate toward goal
else:
    select front image point and execute PixelNav
```

추가 CLI:

```bash
--heuristic-y-ratio
--bearing-rotate-threshold-deg
```

### 2.2 Official distance delta 저장

기존 summary에는 final `distance_to_goal`만 저장되어 있어서, official Habitat geodesic metric 기준으로 얼마나 가까워졌는지 바로 보기 어려웠다.

다음 field를 추가했다.

```text
initial_distance_to_goal
distance_to_goal_delta
aggregate.mean_distance_to_goal_delta
```

해석:

```text
distance_to_goal_delta > 0이면 goal에 가까워짐
distance_to_goal_delta = 0이면 progress 없음
distance_to_goal_delta < 0이면 goal에서 멀어짐
```

## 3. 검증

Unit test:

```bash
cd /home/icra/Pixel-Navigator
/home/icra/micromamba/bin/micromamba run -r /home/icra/micromamba-root -n habitat \
python -m unittest /home/icra/Pixel-Navigator/tests/test_voca_memory_benchmark.py
```

결과:

```text
Ran 10 tests
OK
```

검증한 추가 항목:

```text
PointNav bearing-first heuristic이 goal bearing이 클 때 rotate를 먼저 반환
goal bearing이 front에 들어오면 front PixelNav point를 반환
make_vlm_client_factory가 pointnav-bearing client를 생성
initial/final distance_to_goal delta가 summary와 CSV에 저장됨
```

## 4. Diagnostic Run

### 4.1 기존 heuristic 짧은 비교

실행:

```bash
python voca_memory_benchmark.py \
  --task pointnav \
  --dataset hm3d \
  --vlm heuristic \
  --eval-episodes 1 \
  --max-agent-steps 3 \
  --max-pixelnav-steps 8 \
  --pixelnav-device cpu \
  --out reports/pointnav_hm3d_diag_e1_s3_p8
```

결과:

```text
final distance_to_goal: 13.089178
agent actions: go, go, go
PixelNav total moved: 1.5 m
memory nodes / edges: 1 / 0
```

해석:

```text
PixelNav는 움직였지만, goal bearing 정렬 없이 front로 진행하여 geodesic distance가 악화될 수 있음.
```

### 4.2 Bearing-first heuristic, y_ratio=0.625

실행:

```bash
python voca_memory_benchmark.py \
  --task pointnav \
  --dataset hm3d \
  --vlm pointnav-bearing \
  --eval-episodes 5 \
  --max-agent-steps 6 \
  --max-pixelnav-steps 8 \
  --pixelnav-device cpu \
  --out reports/pointnav_hm3d_diag_e5_s6_p8_bearing_delta
```

결과:

```text
episodes: 5
success: 0.0
spl: 0.0
mean_distance_to_goal: 9.920616
mean_distance_to_goal_delta: +0.933093
mean_agent_steps: 6.0
mean_memory_nodes: 1.6
mean_memory_edges: 0.6
```

action summary:

```text
agent actions:
  rotate: 13
  go: 17

PixelNav / Habitat sim actions:
  move_forward: 21
  look_down: 18
  turn_left: 13
  stop/None: 11
  look_up: 6
  turn_right: 3

rollouts: 17
zero-move rollouts: 10
total moved distance: 5.25 m
```

episode별 distance delta:

```text
ep0: +0.459131
ep1: +0.510613
ep2: +0.248328
ep3: +0.000000
ep4: +3.447395
```

### 4.3 Bearing-first heuristic, y_ratio=0.75

실행:

```bash
python voca_memory_benchmark.py \
  --task pointnav \
  --dataset hm3d \
  --vlm pointnav-bearing \
  --heuristic-y-ratio 0.75 \
  --eval-episodes 5 \
  --max-agent-steps 6 \
  --max-pixelnav-steps 8 \
  --pixelnav-device cpu \
  --out reports/pointnav_hm3d_diag_e5_s6_p8_bearing_y075
```

결과:

```text
episodes: 5
success: 0.0
spl: 0.0
mean_distance_to_goal: 9.718217
mean_distance_to_goal_delta: +1.135492
mean_agent_steps: 6.0
mean_memory_nodes: 2.0
mean_memory_edges: 1.0
```

action summary:

```text
agent actions:
  rotate: 13
  go: 17

PixelNav / Habitat sim actions:
  move_forward: 27
  look_down: 19
  turn_left: 14
  stop/None: 10
  look_up: 5
  turn_right: 3

rollouts: 17
zero-move rollouts: 9
total moved distance: 6.75 m
```

episode별 distance delta:

```text
ep0: +1.471124
ep1: +0.510613
ep2: +0.248328
ep3: +0.000000
ep4: +3.447395
```

## 5. 관찰된 이슈

### Issue 1. 기존 heuristic은 PointNav goal bearing보다 memory candidate를 빨리 믿음

기존 heuristic은 memory candidate/frontier를 먼저 고르면서 계속 front로 가는 경향이 있었다.

결과적으로 PixelNav는 이동해도 goal 방향이 아닐 수 있다.

조치:

```text
PointNav diagnostic에서는 --vlm pointnav-bearing을 기본으로 쓰는 것이 좋다.
```

### Issue 2. selected point를 더 바닥 쪽으로 내리면 이동성이 좋아짐

`y_ratio=0.625`보다 `y_ratio=0.75`가 이번 5-episode diagnostic에서 더 나았다.

비교:

```text
y=0.625: mean_distance_to_goal_delta +0.933093, total moved 5.25 m
y=0.75 : mean_distance_to_goal_delta +1.135492, total moved 6.75 m
```

조치:

```text
내일 PointNav smoke는 --heuristic-y-ratio 0.75부터 시작하는 것이 좋다.
```

### Issue 3. PixelNav가 look_down / turn / stop에 많은 step을 씀

PixelNav rollout에서 `move_forward`가 나오기 전에 `look_down`, `turn_left/right`, `look_up`이 꽤 많이 나온다.

따라서 `max_pixelnav_steps=1` 같은 smoke는 너무 짧다.

조치:

```text
PointNav 실험에서는 max_pixelnav_steps를 최소 8 이상으로 두는 것이 좋다.
처음부터 12 또는 16도 sweep 후보로 둔다.
```

### Issue 4. zero-move rollout이 아직 많음

`y_ratio=0.75`에서도 17개 rollout 중 9개가 zero-move였다.

특히 episode 3은 모든 rollout이 stop/None으로 끝나 distance progress가 없었다.

조치 후보:

```text
no_progress_count > 0일 때 selected point를 더 아래로 조정
연속 no_progress 시 rotate 또는 observation request 강제
PixelNav가 즉시 stop하면 같은 point를 반복하지 않도록 point jitter 추가
```

### Issue 5. memory graph는 이동 성공이 있어야 의미 있게 자람

`y_ratio=0.75`에서 평균 memory nodes/edges는 2.0/1.0까지 증가했다.

반대로 zero-move episode에서는 node/edge가 거의 늘지 않는다.

해석:

```text
memory 문제가 아니라 local rollout progress가 memory graph 성장의 선행 조건이다.
```

### Issue 6. HM3D semantic descriptor warning

PointNav 실행 중 다음 warning이 반복된다.

```text
active scene does not contain semantic annotations
*.basis.scn does not exist
```

PointNav는 semantic object annotation이 필요하지 않으므로 현재 geometry/navmesh 실행에는 치명적이지 않다. ObjectNav에서는 semantic asset 상태를 별도로 확인해야 한다.

## 6. 내일 실험 추천 기본값

먼저 다음 설정으로 10-episode diagnostic을 추천한다.

```bash
cd /home/icra/Pixel-Navigator
/home/icra/micromamba/bin/micromamba run -r /home/icra/micromamba-root -n habitat \
python voca_memory_benchmark.py \
  --task pointnav \
  --dataset hm3d \
  --vlm pointnav-bearing \
  --heuristic-y-ratio 0.75 \
  --eval-episodes 10 \
  --max-agent-steps 20 \
  --max-pixelnav-steps 12 \
  --pixelnav-device cpu \
  --out reports/pointnav_hm3d_diag_e10_s20_p12_bearing_y075
```

그 다음 sweep:

```text
max_pixelnav_steps: 8, 12, 16
heuristic_y_ratio: 0.70, 0.75, 0.80
bearing_rotate_threshold_deg: 15, 25, 35
```

## 7. 결론

PointNav부터 보는 전략은 맞다.

이번 진단에서 확인한 것은 다음이다.

```text
1. 기존 heuristic보다 PointNav bearing-first policy가 더 적합하다.
2. y_ratio를 0.75로 내리면 이동량과 distance delta가 개선된다.
3. PixelNav local rollout은 동작하지만 zero-move rollout이 아직 많다.
4. 내일 성능을 올리려면 no-progress recovery와 selected point sweep이 1순위다.
```

현재 추천 시작점:

```text
--task pointnav
--dataset hm3d
--vlm pointnav-bearing
--heuristic-y-ratio 0.75
--max-agent-steps 20
--max-pixelnav-steps 12
```

이 설정은 아직 success/SPL을 기대하는 최종 policy는 아니지만, 내일 ablation을 시작하기 위한 안정적인 diagnostic baseline으로 적합하다.
