# 06. v6 episodic graph memory closed loop 통합

## 목표

memory를 단순 로그나 prompt history가 아니라, 실행 결과로 검증되고 다음 action을 실제로 바꾸는 episodic graph로 만든다. false merge보다 duplicate node를 허용하는 보수적인 설계를 택했다.

## 이식한 기반

- `qwen_nav_memory_framework_v6/` (`voca-s2e`의 canonical 구현)
- 독립 VOCA checkout은 인접한 `voca-s2e`를 자동 탐색하며, 필요하면 `VOCA_S2E_ROOT`로 위치를 지정
- VOCA와 v6 사이의 adapter: `voca_memory_sidecar.py`

## 저장하는 memory

| Memory | 내용 |
|---|---|
| Place node | visual keyframe, relative pose, room category |
| Transition edge | 성공 이동, escape success, reverse relation |
| Failed frontier | collision/no-progress 방향과 candidate/pixel |
| Revisit link | 같은 장소 후보와 confirm/reject/defer 결과 |
| Semantic state | current room, target evidence, navigation intent |
| Target evidence | identity-verified target image, bearing, pose anchor |

## closed-loop 동작

1. 실행 전 Qwen은 bounded compact memory projection을 받는다.
2. projection에는 supervisor mode, 최근 failure, safe exits, negative memory, revisit candidate만 포함한다.
3. exact projection과 SHA256을 `qwen_calls.jsonl`에 저장한다.
4. Qwen이 memory op를 제안해도 backend verifier가 `confirm/reject/defer`한 뒤에만 graph를 갱신한다.
5. PixelNav 실행 결과가 success/failure edge와 failed frontier를 만든다.
6. 다음 candidate generation과 Qwen decision이 이 memory를 다시 사용한다.

## Place recognition

- DINOv2 small embedding을 CPU에서 사용한다.
- threshold: 0.84
- labeled sample 112개에서 precision 1.0, false positive 0, recall 0.434
- 낮은 recall은 duplicate place를 늘리지만, 잘못된 merge로 topology가 오염되는 위험을 줄인다.

## Audit 가능한 항목

- `memory_ops_requested/accepted/rejected`
- `revisits_confirmed/rejected`
- `soft_merges`
- `directional_failure_count`
- `place/failed-frontier/negative/same-place edge count`
- `compact_memory_sha256`
- `last_event`, `last_mode_transition`, `last_verified_target_evidence`

## 추천 영상

- `media/03_chair_memory_graph_reconstruction.mp4`
  - graph node와 edge가 시간 순으로 추가되는 standalone 영상이다.
- `media/01_chair_end_to_end_highlight_49s.mp4`
  - 같은 graph를 RGB/action/reasoning과 동기화해 본다.

## 시각화 범례

- 주황 node: place
- 파랑 node: current place
- 빨강 X: failed frontier
- 초록 arrow: verified transition/escape
- 파랑 계열 line: revisit/same-place relation
- 우측 상단 mode: `GOAL SEEK`, `ESCAPE`, `BACKTRACK`, `VERIFY TARGET`

## 현재 한계

- 3-episode pilot ablation에서는 full memory가 명확한 success 우위를 보이지 않았다.
- full과 no-memory는 모두 1/3 success였고, no-memory/no-scoring이 2/3으로 더 높았다.
- McNemar exact p-value는 모두 1.0으로 표본이 너무 작다.
- 따라서 현재 주장할 수 있는 것은 memory closed loop와 측정 가능성이지, navigation 성능 향상 자체가 아니다.
