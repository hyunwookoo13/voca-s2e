# 04. Safe waypoint, room exit, deadlock recovery 안정화

## 목표

coarse goal 방향만 기계적으로 따라가 벽이나 가구를 반복 선택하는 문제를 막고, 막힌 경우 뒤로 물러나거나 목표와 반대 방향의 문으로라도 방을 나가 새 경로를 찾게 한다.

## 개발한 내용

- RGB candidate마다 visible-floor support, texture/structure, exposure, local policy score를 기록한다.
- ambiguous candidate는 independent RGB point verifier를 거친다.
- collision/no-progress가 발생한 angle, view, pixel, candidate ref를 directional failure로 memory에 저장한다.
- failed direction과 가까운 다음 candidate는 `avoid=true`로 표시한다.
- `escape_deadlock`에서는 coarse goal 반대 방향이라도 명시적 doorway/exit/backtrack이면 detour를 허용한다.
- `goal_seek`에서는 직전 verified transition을 즉시 반대로 되돌아가는 행동을 제한한다.
- repeated generic-floor GO, same-room cycle, forced-exit failure를 감지해 full sweep 또는 backtrack으로 승격한다.
- doorway alias와 visual exit ref를 backend가 교정할 수 있게 했다.
- target lock이 장시간 풀리지 않으면 cooldown 후 탐색으로 복귀한다.

## 실패 원인 taxonomy

- `collision_blocked`
- `no_progress`
- `stopped_early`
- `point_verifier:wall/furniture/unknown`
- `coarse_goal_direction_rejected`
- `repeated_branch`
- `deadlock_entry_candidate`
- `target_lock_exhausted`

## 최신 chair 성공 사례

- start distance: 8.472 m
- final distance: 0.0287 m
- success: 1.0
- SPL: 0.429
- steps: 602
- place nodes: 10
- failed frontiers: 5
- collision count: 28

성공 자체는 확인됐지만 602 step과 28 collision은 아직 효율이 낮다는 뜻이다. 이 영상은 성능 완성본이 아니라 recovery mechanism이 실제로 닫힌 loop에서 작동한다는 증거로 사용해야 한다.

## 추천 영상

- `media/01_chair_end_to_end_highlight_49s.mp4`
  - 0~14초: room exit unresolved와 observation request
  - 14~27초: 실패 간선 누적과 escape/backtrack
  - 27초 이후: 다른 방 진입, target approach, verified STOP
- `media/04_chair_topdown_highlight_41s.mp4`
  - 같은 구간의 실제 이동 경로를 보조 설명한다.

## 현재 한계

- recovery는 작동하지만 장시간 같은 구역을 다시 방문하는 비용이 크다.
- collision과 observation call을 더 줄이려면 S2E local executor와 더 강한 traversability prior가 필요하다.
- coarse goal, memory, candidate scoring의 결합 효과는 더 큰 paired benchmark로 검증해야 한다.
