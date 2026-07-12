# 07. Coarse goal benchmark protocol과 재현성 구축

## 목표

최종 S2E backbone이 제공할 coarse spatial goal을 미리 모사하고, Qwen이 coarse 방향을 활용하되 장애물이나 방 구조 때문에 필요한 detour는 허용하는 실험 protocol을 만든다.

## Coarse goal contract

```json
{
  "type": "map_waypoint",
  "map_xy": [7.882316, -2.743365],
  "relative_bearing_deg": 128.047,
  "distance_m": 1.0912,
  "distance_range_m": [0.0912, 2.0912],
  "uncertainty": "medium",
  "source": "upstream_global_planner"
}
```

## 개발한 내용

- episode ID, scene, object category에 맞는 coarse goal provider를 구현했다.
- provider file의 SHA256을 실행 전 검증한다.
- uncertainty에 따라 acquire radius, direction error, target alignment, stop radius를 다르게 적용한다.
- generic floor GO가 coarse direction과 크게 다르면 observation request로 바꾼다.
- doorway, explicit exit, deadlock escape, verified backtrack은 반대 방향 detour로 허용한다.
- target identity와 coarse goal이 충돌하면 distractor rejection을 기록한다.
- stop은 coarse goal proximity와 visual target evidence가 일치해야 한다.
- 최신 terminal rule은 coarse distance 0.75 m 이내에서 stop-consistent하면 불필요한 refinement를 생략한다.

## Benchmark validity

- protocol 이름: `hm3d_derived_coarse_goalnav_v1_sim_pose_noise0p5`
- seed: `20260710`
- max steps: 1000
- success distance: 0.10 m
- sliding: disabled
- localization: `habitat_sim_pose_declared`
- coarse goal source: episode goal viewpoint에서 noise를 추가한 derived upstream waypoint

## 반드시 지켜야 할 보고 문구

이 결과는 category-only official ObjectNav가 아니라 **derived CoarseGoalNav with RGB + declared simulator pose**다. top-down map과 Habitat goal metrics는 evaluation-only이며 Qwen prompt에는 들어가지 않는다.

## 주요 코드와 데이터

- `coarse_goal_provider.py`
- `scripts/generate_hm3d_coarse_goal_dataset.py`
- `scripts/check_benchmark_ready.py`
- `configs/coarse_goals/hm3d_val_seed20260710_ep0_2_noise0p5.json`
- provider SHA256: `be0dbc1a36227f9dabb226b7f0de19a1d53501560044420802d82262ca1739a7`

## 추천 영상

- `media/01_chair_end_to_end_highlight_49s.mp4`
  - coarse direction rejection과 explicit room-exit detour가 번갈아 나타난다.
- `media/04_chair_topdown_highlight_41s.mp4`
  - detour가 실제 path에 어떻게 반영됐는지 설명한다.

## 현재 한계

- coarse goal이 실제 S2E output이 아니라 derived proxy다.
- simulator pose를 사용한다.
- goal noise 0.5 m 한 설정만으로 일반화할 수 없다.
- 후속 실험에서는 S2E output을 같은 provider interface로 공급하고 noise/uncertainty ablation을 해야 한다.
