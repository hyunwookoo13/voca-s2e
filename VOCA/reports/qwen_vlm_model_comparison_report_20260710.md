# VOCA Qwen3-VL 모델 비교 보고서

- 작성 시각: 2026-07-10 22:34 KST
- 대상 시스템: VOCA Qwen-VLM + PixelNav + v6 graph memory
- 목적: VOCA의 주행 판단 모델 후보를 속도, 출력 안정성, waypoint 안전성, memory 준수 관점에서 비교

## 1. 요약

동일한 세 종류의 저장 VOCA 입력을 사용한 정식 비교에서 가장 높은 안전성 proxy를 보인 모델은 `32B Thinking-AWQ`였고, 가장 빠른 모델은 `8B Instruct-AWQ-4bit`였다. 그러나 8B Instruct는 `avoid=true`로 명시된 deadlock 후보를 반복 선택했고, 8B Thinking은 실제 revisit 후보가 주어졌을 때 필수 memory verdict를 생략했다. 따라서 8B 계열은 현재 상태에서 VOCA의 단일 주 모델로 사용하기 어렵다.

속도와 판단 품질을 함께 고려한 실용적 주 모델 후보는 `32B Instruct-AWQ`다. 표준 memory 6-view 입력에서 유효한 `go`를 선택했고 평균 호출 시간은 2.33초였다. 단, panoramic 입력에서 low-texture waypoint를 선택했으므로 point verifier와 backend candidate guard를 유지해야 한다.

정확도 우선 기준 모델은 `32B Thinking-AWQ`다. 세 표준 상황에서 모든 선택이 candidate gate와 visual-risk 검사를 통과했지만 평균 9.82초가 필요하다. 최종 시스템은 `32B Instruct-AWQ`를 기본 모델로 사용하고, deadlock/revisit/stop 검증 등 고난도 상황에서 `32B Thinking-AWQ`로 승격하는 구성이 가장 합리적이다.

## 2. 비교 모델

| 표기 | 모델 |
|---|---|
| 32B Thinking | `Qwen/Qwen3-VL-32B-Thinking` |
| 32B Thinking-AWQ | `QuantTrio/Qwen3-VL-32B-Thinking-AWQ` |
| 32B Instruct-AWQ | `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` |
| 8B Thinking-AWQ | `cyankiwi/Qwen3-VL-8B-Thinking-AWQ-4bit` |
| 8B Instruct-AWQ | server ID `qwen3-vl-8b-instruct-awq-4bit` |

## 3. 실험 조건

정식 비교에는 다음 세 VOCA 판단 입력을 사용했다.

1. `initial_1view`: 단일 전방 RGB 입력
2. `panoramic_5view`: 실패 이후 수집된 5-view 입력과 negative-memory 후보
3. `memory_6view`: current multi-view와 memory image/context를 포함한 입력

각 상황을 3회 반복했고 model warm-up 호출은 통계에서 제외했다. 모든 모델의 temperature는 0이며 최대 출력은 4096 tokens로 설정했다.

입력 원본은 같지만 image preprocessing 크기는 완전히 동일하지 않다. 32B Thinking 계열은 최대 512px, Instruct 계열과 8B Thinking-AWQ는 최대 640px로 실행됐다. 32B Instruct-AWQ는 512px 구성에서 HTTP 500이 발생해 640px가 필요했다. 또한 32B Thinking 원본은 server-01, 나머지 quantized 모델은 server-02에서 측정됐다. 따라서 원본과 quantized 모델 사이의 속도 차이에는 hardware 영향도 포함될 수 있다.

## 4. 지표 정의

- `JSON/Schema`: JSON 파싱과 기본 schema 정규화 성공 여부다. 판단의 의미적 정확도를 뜻하지 않는다.
- `Candidate gate`: `go`가 backend가 제공한 실행 가능한 candidate ref를 선택했는지 검사한다. `rotate`와 같은 non-go action은 candidate가 필요하지 않아 통과로 처리된다.
- `Pipeline accept`: candidate gate를 통과하고 선택 point에 즉시 확인되는 visual risk가 없는 비율이다. Habitat Success/SPL이 아니다.
- `Risky point`: valid candidate이지만 low-texture, 과노출, 저노출 등의 이유로 추가 verifier가 필요한 point다.
- `Completion tokens`: reasoning을 포함한 평균 출력 token 수다.

## 5. 전체 정식 비교

| 모델 | 서버 | 평균 | 중앙값 | P95 | JSON/Schema | Gate | Pipeline | Risky point | 출력 tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32B Thinking | server-01 | 39.26초 | 39.07초 | 40.12초 | 100% | 100% | 100% | 0% | 1229.7 |
| 32B Thinking-AWQ | server-02 | 9.82초 | 8.44초 | 12.82초 | 100% | 100% | 100% | 0% | 937.0 |
| 8B Thinking-AWQ | server-02 | 5.20초 | 5.30초 | 6.09초 | 100% | 100% | 66.7% | 33.3% | 1168.2 |
| 32B Instruct-AWQ | server-02 | 2.33초 | 2.09초 | 3.62초 | 100% | 100% | 66.7% | 33.3% | 184.3 |
| 8B Instruct-AWQ | server-02 | 1.04초 | 0.86초 | 1.59초 | 100% | 66.7% | 66.7% | 0%* | 185.7 |

`*` 8B Instruct의 risky-point 비율이 0%인 것은 안전해서가 아니다. `panoramic_5view`에서 선택한 point가 `avoid=true` 후보였기 때문에 point-risk 검사 전에 candidate gate가 차단했다.

### 속도 배율

| 비교 | 속도 향상 |
|---|---:|
| 32B Thinking -> 32B Thinking-AWQ | 4.00배 |
| 32B Thinking -> 8B Thinking-AWQ | 7.55배 |
| 32B Thinking -> 32B Instruct-AWQ | 16.86배 |
| 32B Thinking -> 8B Instruct-AWQ | 37.81배 |
| 32B Thinking-AWQ -> 32B Instruct-AWQ | 4.22배 |
| 32B Instruct-AWQ -> 8B Instruct-AWQ | 2.24배 |

## 6. 상황별 비교

| 모델 | initial 1-view | panoramic 5-view | memory 6-view |
|---|---|---|---|
| 32B Thinking | 38.60초, `go`, `px_v00_r0_c0`, 통과 | 40.10초, `go`, `px_v04_r0_c1`, 통과 | 39.07초, `go`, `px_v03_r0_c1`, 통과 |
| 32B Thinking-AWQ | 8.44초, `go`, `px_v00_r0_c1`, 통과 | 12.82초, `go`, `px_v04_r0_c1`, 통과 | 8.20초, `go`, `px_v03_r0_c1`, 통과 |
| 8B Thinking-AWQ | 5.25초, `go`, `px_v00_r0_c1`, 통과 | 5.67초, `go`, `px_v00_r0_c1`, low-texture | 4.67초, `rotate` |
| 32B Instruct-AWQ | 1.77초, `rotate` | 2.58초, `go`, `px_v04_r1_c2`, low-texture | 2.64초, `go`, `px_v00_r1_c1`, 통과 |
| 8B Instruct-AWQ | 0.85초, `rotate` | 1.17초, `go`, `px_v03_r1_c2`, `avoid=true`로 차단 | 1.09초, `rotate` |

### 행동 차이 해석

32B Thinking 계열은 세 표준 상황에서 모두 `go`를 선택했다. 후보가 backend gate를 통과했고 local visual-risk도 검출되지 않았다. 다만 별도의 단일 post-upgrade smoke에서 32B Thinking 원본도 low-texture point를 선택한 사례가 있으므로 표준 세 입력의 100% Pipeline 수치를 일반적인 안전 성공률로 해석해서는 안 된다.

32B Instruct-AWQ는 초기 입력에서는 보수적으로 `rotate`했지만 memory 6-view에서는 유효한 `go`를 선택했다. panoramic 입력의 문제는 금지된 branch 선택이 아니라 시각 정보가 부족한 low-texture point 선택이다. 이 유형은 independent point verifier로 차단하거나 승인할 수 있다.

8B Instruct-AWQ는 가장 빠르지만 negative memory가 `deadlock_entry_candidate`, `avoid=true`, 음수 score로 표시한 `px_v03_r1_c2`를 세 번 모두 선택했다. backend가 실제 `go`를 `request_observation`으로 변환하므로 즉시 충돌하지는 않지만, 불필요한 VLM 재호출을 유발한다.

8B Thinking-AWQ는 8B Instruct의 hard policy violation을 피했다. 그러나 panoramic 입력에서 low-texture point를 선택했고 memory 6-view에서는 `rotate`했다. 평균 출력이 1168 tokens로 32B Thinking 계열과 유사해 8B 모델임에도 호출 시간이 5.20초다. 첫 engine warm-up은 28.24초였으며 persistent serving에서는 이후 호출에 포함되지 않는다.

## 7. 실제 Revisit Memory 검증

표준 `memory_6view`는 memory image/context를 포함하지만 `revisit_candidates`가 비어 있었다. 따라서 실제 place recognition 판단은 별도 closed-loop 입력으로 검사했다.

| 모델 | 호출 시간 | 입력 후보 | VLM memory verdict | Backend 결과 |
|---|---:|---|---|---|
| 32B Thinking 원본 | 약 41.0초 | `revisit_001`, DINO 0.9999, spatial pass | `confirm_revisit_node` | 검증 후 soft merge 성공 |
| 8B Thinking-AWQ | 5.81초 | `revisit_001`, DINO 0.9999, spatial pass | `memory_ops=[]` | 누락 감지 후 `defer`, topology 변경 없음 |
| 32B Thinking-AWQ | 미측정 | - | - | 실제 revisit 추가 검증 필요 |
| 32B Instruct-AWQ | 미측정 | - | - | 실제 revisit 추가 검증 필요 |
| 8B Instruct-AWQ | 미측정 | - | - | 실제 revisit 추가 검증 필요 |

8B Thinking-AWQ는 JSON schema를 만족했지만, revisit 후보가 있을 때 정확히 하나의 `confirm/reject/defer`를 출력하라는 의미적 계약을 지키지 않았다. backend contract guard가 누락을 감지해 자동 `defer`했고 false merge는 발생하지 않았다. 이 결과는 JSON 성공률과 memory reasoning 성공률을 분리해서 평가해야 함을 보여준다.

## 8. 최종 평가

### 정확도 및 안정성 우선

현재 표준 입력 기준으로는 `32B Thinking-AWQ`가 가장 안정적이다. 원본 Thinking보다 약 4배 빠르고 candidate 및 visual-risk proxy를 모두 통과했다. 단일 모델만 사용하고 latency보다 연구 성능 상한이 중요하다면 우선 선택할 수 있다.

### 속도와 성능 균형

`32B Instruct-AWQ`가 가장 현실적인 주 모델 후보다. 평균 2.33초이며 memory-context 입력에서 유효한 `go`를 선택했다. low-texture point는 이미 구축한 point verifier와 backend guard로 관리할 수 있다. 최종 채택 전 실제 revisit와 stop-verification 입력을 추가로 검사해야 한다.

### 8B 모델

8B Instruct-AWQ는 초고속이지만 negative-memory 위반이 확인됐다. 8B Thinking-AWQ는 그 위반을 줄였지만 latency가 5.20초로 증가했고 실제 revisit verdict를 생략했다. 현재 결과에서는 어느 8B 모델도 주 모델로 권장하기 어렵고, ablation 또는 향후 candidate-set pruning 실험용으로 적합하다.

## 9. 권고 구성

1. 기본 주행 판단: `32B Instruct-AWQ`
2. backend 보호: strict candidate gate + independent point verifier + stop verifier 유지
3. 고난도 승격: deadlock, low confidence, actual revisit, ambiguous stop에서 `32B Thinking-AWQ` 호출
4. 연구 baseline: `32B Thinking-AWQ`를 품질 상한, `8B Instruct-AWQ`를 latency 하한으로 사용
5. 추가 모델 탐색 중단: 다음 자원은 동일-seed HM3D multi-episode 검증에 사용

## 10. 남은 검증

현재 Pipeline accept는 navigation 성공률이 아니다. 최종 모델 선정에는 동일 HM3D episode seed로 다음을 비교해야 한다.

- Success, SPL, SoftSPL, final distance-to-goal
- collision 및 no-progress 빈도
- `avoid=true` 선택 시도와 backend rejection 빈도
- VLM 호출 횟수와 episode wall-clock time
- deadlock escape 성공률
- actual revisit confirm/reject/defer 정확도
- stop precision/recall 및 lookalike false stop

## 11. 결과 파일

- 32B Thinking 및 Thinking-AWQ: `reports/reproducibility_20260712/model_comparison/32b_thinking_original_vs_awq/`
- 32B Instruct-AWQ: `reports/reproducibility_20260712/model_comparison/32b_instruct_awq/`
- 8B Instruct-AWQ: `reports/reproducibility_20260712/model_comparison/8b_instruct_awq/`
- 8B Thinking-AWQ: `reports/reproducibility_20260712/model_comparison/8b_thinking_awq/`

Actual-revisit의 전체 image/prompt artifact는 크기와 로컬 경로 때문에 Git
대상에서 제외했다. 본 보고서에는 해당 closed-loop 결과와 해석만 남겼다.
