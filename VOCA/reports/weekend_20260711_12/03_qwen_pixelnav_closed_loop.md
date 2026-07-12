# 03. YOLOE-free Qwen action + PixelNav closed loop 구축

## 목표

YOLOE detection에 의존하지 않고, Qwen이 RGB와 target-context cues를 보고 실행 가능한 image point 또는 controller interrupt를 선택하도록 만들었다. PixelNav는 선택된 point까지 이동하는 frozen local skill로 사용한다.

## Action contract

| Action | 의미 | 실행 |
|---|---|---|
| `go` | 선택한 image-space waypoint로 이동 | PixelNav local policy |
| `rotate` | 지정 각도로 시야 변경 | Habitat controller |
| `request_observation` | 필요한 방향의 추가 RGB 수집 | bounded observation sweep |
| `stop` | 목표 완료 요청 | verifier 통과 후 Habitat STOP |

## 개발한 내용

- 기존 target-context priors call은 유지했다.
- 고정 360도 스캔 후 한 점으로 가는 VOCA loop 대신, 상황에 따라 observation 범위를 조절하는 receding-horizon action loop를 만들었다.
- `go`마다 backend-generated RGB Set-of-Marks candidate를 최대 30개 제공한다.
- Qwen은 `selected_candidate_ref`를 복사해야 하며, raw pixel을 임의로 바꿔도 backend가 canonical point로 교정한다.
- missing, unknown, avoided, view-mismatch candidate는 `request_observation`으로 fail-closed 처리한다.
- PixelNav candidate scoring을 연결해 local policy 관점의 실행 가능성을 candidate에 반영했다.
- 주행 뒤 camera pitch를 neutral pose로 복원해 천장/바닥 또는 뒤집힌 시점이 다음 Qwen 입력으로 누적되는 문제를 해결했다.
- zero-yaw rotate, repeated no-progress, collision-tolerant translation을 별도 분류했다.

## 주요 코드

- `qwen_vlm_planner.py`
- `qwen_action_runner.py`
- `pixel_candidate_gate.py`
- `policy_agent.py`
- `qwen_point_planner.py`

## 검증

- 모든 `go`는 candidate gate checkpoint를 남긴다.
- 최신 chair 성공 episode에서 raw go 28회 모두 candidate gate를 통과했다.
- 최신 plant 성공 episode는 42 step, collision 1회, final distance 0.0940 m로 종료됐다.
- YOLOE runtime call은 0회다.

## 추천 영상

- `media/02_plant_verified_stop_12s.mp4`
  - 좌측에서 selected point와 실제 RGB trajectory를 보고, 우측에서 action, candidate, gate, reasoning을 함께 확인할 수 있다.
- `media/01_chair_end_to_end_highlight_49s.mp4`
  - `go`만 반복하지 않고 `request_observation`과 detour action으로 전환되는 장면을 보여준다.

## 현재 한계

- candidate는 RGB/policy proxy로 생성되므로 완전한 geometric traversability를 보장하지 않는다.
- verifier가 안전성을 높이는 대신 추가 Qwen latency를 만든다.
- PixelNav가 local waypoint에서 멈췄다고 ObjectNav goal이 완료된 것은 아니다. STOP verifier가 별도로 필요하다.
