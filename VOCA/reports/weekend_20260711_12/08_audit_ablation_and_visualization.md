# 08. Audit, ablation, stress test, 시각화 정리

## 목표

성공/실패 숫자만 남기는 benchmark가 아니라, Qwen이 무엇을 보고 어떤 action을 요청했고 backend가 왜 승인/거부했는지 재현할 수 있는 artifact와 보고용 영상을 만든다.

## Episode별 artifact

| 파일 | 역할 |
|---|---|
| `qwen_calls.jsonl` | VLM input/output, candidate, verifier, memory projection, latency |
| `memory/steps.json` | action 이후 pose/motion/collision/evaluation audit |
| `memory/memory_graph.json` | 최종 episodic graph |
| `benchmark_manifest.json` | model, protocol, seed, config, code/checkpoint hash |
| `fps.mp4` | RGB + selected point + action/reasoning + memory graph |
| `metric.mp4` | evaluation-only top-down trajectory |
| `memory_graph_reconstruction.mp4` | memory graph 생성/갱신 과정 |

## FPS UI 개선

- 좌측: agent RGB trajectory와 selected image point
- 우측 상단: executed action badge
- 우측 중단: target, room/intent, angle/point, confidence, candidate gate, fallback, reasoning
- 우측 하단: v6 memory graph와 supervisor mode
- graph layout은 dominant path axis를 가로로 정렬하되 metric aspect를 보존한다.
- reverse duplicate edge를 숨기고 arrow head를 약 4 px로 고정했다.
- node가 많을 때 current/start label만 남기고 label collision을 회피한다.
- 4-node와 20-node 실제 state에서 별도 visual inspection을 수행했다.

## 자동 검증

- 전체 regression: 292 tests passed
- `git diff --check`: pass
- readiness checker:
  - HM3D data/config
  - PixelNav checkpoint load
  - Qwen served ID와 model root
  - coarse goal file과 SHA256
  - localization contract
  - max episode steps
- deterministic guard stress: 6/6 pass
- revisit verifier stress: 3/3 pass
- stop verifier stress: false positive 0/6, true positive 2/3

## 3-episode pilot ablation

| Variant | Success | SPL mean | Final distance mean | 해석 |
|---|---:|---:|---:|---|
| full | 1/3 | 0.277 | 3.836 m | plant 성공 |
| no memory | 1/3 | 0.277 | 1.668 m | success 동률 |
| no PixelNav scoring | 0/3 | 0.000 | 1.616 m | scoring 제거 시 success 하락 |
| no memory + no scoring | 2/3 | 0.442 | 0.094 m | 비단조 결과, 작은 표본 영향 큼 |

paired McNemar exact p-value는 모든 비교에서 1.0이다. 이 pilot으로 component superiority를 주장하면 안 된다. 대신 실험 harness가 동일 seed/config에서 component toggle과 paired report를 생성한다는 점이 이번 단계의 성과다.

## 추천 영상

| 우선순위 | 파일 | 길이 | 보여주는 내용 |
|---:|---|---:|---|
| 1 | `media/01_chair_end_to_end_highlight_49s.mp4` | 약 49 s | 전체 closed loop와 최종 success |
| 2 | `media/02_plant_verified_stop_12s.mp4` | 약 12 s | 빠른 target approach와 STOP |
| 3 | `media/03_chair_memory_graph_reconstruction.mp4` | 약 19 s | memory 생성/갱신 |
| 4 | `media/04_chair_topdown_highlight_41s.mp4` | 약 41 s | evaluation trajectory |

## 보존된 결과 위치

- `reports/reproducibility_20260712/final_v3/`
- `reports/reproducibility_20260712/ablation/`
- `reports/reproducibility_20260712/stress/`
- `reports/reproducibility_20260712/place_recognition/`

Full output tree는 Git 대상에서 제외했다. final-v3의 `trajectory_2`는 정리
작업으로 전환하면서 중단한 partial run이므로 경량 snapshot에도 포함하지
않았다.

## 다음 검증

1. 20~50 episode paired benchmark로 표본 확대
2. full, no-memory, no-negative-memory, no-adaptive-observation, no-stop-verifier ablation 분리
3. SR/SPL뿐 아니라 collision, VLM call, wall-clock, false stop, deadlock escape rate 보고
4. 32B Instruct 기본 + Thinking escalation routing 실험
5. PixelNav와 동일 executor contract로 S2E 연결
