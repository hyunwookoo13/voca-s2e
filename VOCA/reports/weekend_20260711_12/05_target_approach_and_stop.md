# 05. Target approach와 STOP/goal completion 안정화

## 목표

목표 물체를 한 번 봤다는 이유로 멀리서 false stop하지 않고, 반대로 충분히 가까워진 뒤에도 target을 잃어 영원히 움직이는 문제를 줄인다. 최종 `stop`은 VLM 판단, visual evidence, 실행 상태가 함께 일치할 때만 허용한다.

## 개발한 내용

- `stop` 요청을 바로 실행하지 않고 `verify_target` state로 전환한다.
- independent identity critic이 target과 lookalike를 구분한다.
- target bbox의 height, area, horizontal center를 검사한다.
- 두 시점의 temporal confirmation과 접근 중 scale 증가를 확인한다.
- small target와 strong target threshold를 분리했다.
- verified target의 pose/bearing/image evidence를 bounded target anchor memory에 저장한다.
- target을 잠시 잃으면 anchor 방향으로 제한적으로 reacquire하고, 횟수 초과 시 탐색으로 복귀한다.
- target이 보이지만 아직 먼 경우 bounded PixelNav approach를 실행한다.
- final approach는 translation cap, cooldown, failure limit, terminal micro-approach count로 제한한다.
- coarse goal stop-consistency가 충분하고 가까운 경우 불필요한 terminal refinement를 건너뛸 수 있게 했다.
- 실제 runner가 Habitat STOP을 실행했는지를 `qwen_stop_actions`로 별도 audit한다.

## 대표 검증 결과

| Target | Success | Final distance | SPL | Steps | Stop 특징 |
|---|---:|---:|---:|---:|---|
| plant | 1.0 | 0.0940 m | 0.831 | 42 | visible target 후 bounded approach, identity confirm, STOP |
| chair | 1.0 | 0.0287 m | 0.429 | 602 | target re-acquisition 후 final approach, STOP |

HM3D success threshold는 0.10 m로 설정했고 sliding은 비활성화했다.

## 개발 과정에서 해결한 문제

- target visible인데 멀리서 stop하는 false positive
- 목표 근처에서 stop verifier는 통과했지만 runner가 STOP을 실행하지 않는 문제
- target approach가 과도해 목표를 지나치거나 다시 잃는 문제
- plant처럼 형태가 가늘고 bbox area가 작은 category의 near-latch 문제
- 한 번의 target evidence만으로 temporal confirmation을 우회하는 문제
- verified target을 잃은 뒤 무한 reacquisition에 빠지는 문제

## 추천 영상

- `media/02_plant_verified_stop_12s.mp4`
  - 짧고 결과가 명확해 선임님께 가장 먼저 보여주기 좋은 stop demo다.
  - 마지막 프레임에서 `STOP`, `confirmed`, `verify_target`, `stop condition satisfied`를 동시에 확인할 수 있다.
- `media/01_chair_end_to_end_highlight_49s.mp4`
  - 긴 탐색 뒤에도 target evidence를 다시 확보하고 종료하는 사례다.

![Plant verified STOP final frame](media/02_plant_verified_stop_final.png)

## 주요 코드

- `qwen_vlm_planner.py`
- `qwen_action_runner.py`
- `navigation_supervisor.py`
- `voca_memory_sidecar.py`

## 현재 한계

- verifier는 VLM 호출을 추가하므로 episode latency가 크다.
- category별 bbox 특성이 달라 threshold가 완전히 category-agnostic하지 않다.
- stop precision/recall은 더 많은 positive/negative episode로 검증해야 한다.
- current stress set에서는 false positive 0, positive recall 0.667이지만 sample은 9개뿐이다.
