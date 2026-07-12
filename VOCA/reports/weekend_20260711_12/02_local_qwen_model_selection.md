# 02. Local Qwen-VLM 전환 및 모델 비교

## 목표

Gemini 기반 VOCA 호출을 OpenAI-compatible local Qwen-VL endpoint로 교체하고, 실제 VOCA decision payload에서 속도와 판단 안정성을 비교해 주 모델 후보를 정했다.

## 비교 조건

- `initial_1view`: 초기 단일 전방 RGB
- `panoramic_5view`: 실패 이후 수집한 5-view와 negative candidate
- `memory_6view`: current multi-view와 memory context/image를 포함한 입력
- 각 상황 3회 반복, temperature 0, warm-up 제외
- JSON 성공과 semantic safety를 분리 평가

## 핵심 결과

| 모델 | 평균 | P95 | JSON | Gate | Pipeline accept | 해석 |
|---|---:|---:|---:|---:|---:|---|
| 32B Thinking | 39.26 s | 40.12 s | 100% | 100% | 100% | 품질은 높지만 너무 느림 |
| 32B Thinking-AWQ | 9.82 s | 12.82 s | 100% | 100% | 100% | 현재 품질 우선 benchmark 모델 |
| 32B Instruct-AWQ | 2.33 s | 3.62 s | 100% | 100% | 66.7% | 실용적인 기본 모델 후보 |
| 8B Thinking-AWQ | 5.20 s | 6.09 s | 100% | 100% | 66.7% | revisit verdict 누락 사례 |
| 8B Instruct-AWQ | 1.04 s | 1.59 s | 100% | 66.7% | 66.7% | avoid=true candidate 선택 사례 |

## 결론

- 현재 최종 검증은 `QuantTrio/Qwen3-VL-32B-Thinking-AWQ`로 실행했다.
- 속도와 안정성의 균형은 `32B Instruct-AWQ`가 가장 좋았다.
- 장기적으로는 Instruct를 일반 주행에 사용하고, deadlock/revisit/stop 같은 어려운 판단만 Thinking으로 승격하는 routing이 합리적이다.
- 이 model routing은 아직 실제 runner에 적용하지 않았다.

## 개발한 내용

- `llm_utils/qwen_request.py`에 OpenAI-compatible chat completion client를 추가했다.
- model ID와 served root를 함께 검증해 잘못된 endpoint/model 조합을 차단했다.
- strict JSON parsing, retry, timeout, backend health, duration audit을 추가했다.
- priors와 navigation/verification call latency를 분리 기록했다.
- model comparison script와 JSON/CSV report를 추가했다.

## 근거 파일

- `reports/qwen_vlm_model_comparison_report_20260710.md`
- `scripts/benchmark_qwen_models.py`
- `reports/reproducibility_20260712/model_comparison/`

## 추천 영상

- `media/02_plant_verified_stop_12s.mp4`
  - 현재 32B Thinking-AWQ가 실제 VOCA action을 내고 PixelNav와 결합된 결과를 짧게 보여준다.

## 현재 한계

- 원본 Thinking과 AWQ는 서로 다른 서버에서 측정돼 hardware 영향이 섞여 있다.
- image preprocessing 크기가 일부 모델에서 달랐다.
- Pipeline accept는 Habitat Success/SPL이 아니다.
- 32B Thinking-AWQ의 navigation call은 최신 run에서 약 10~16초, priors는 약 41초였다.
