# HM3D Official Benchmark Integration Report

Updated: 2026-07-08

## 1. 요약

현재 `Pixel-Navigator` 안에서 Habitat official benchmark 환경에 VOCA/Qwen grounded goal, PixelNav local rollout, v5 memory graph update를 연결하는 runner를 준비했다.

결론적으로 HM3D 기준에서는 다음 두 경로가 모두 실행 가능한 상태이다.

```text
ObjectNav HM3D official split
  -> Habitat env
  -> VLM/heuristic grounded goal
  -> PixelNav rollout
  -> ActionOutcome
  -> qwen_nav_memory_framework_v5 memory graph
  -> Habitat official metrics summary

PointNav HM3D official split
  -> Habitat env
  -> VLM/heuristic grounded goal
  -> PixelNav rollout
  -> ActionOutcome
  -> qwen_nav_memory_framework_v5 memory graph
  -> Habitat official metrics summary
```

단, 현재 확인한 결과는 1 episode / 1 agent step smoke check이다. 따라서 SR, SPL, Success 성능 수치로 해석하기보다는 공식 benchmark를 돌릴 수 있는 실행 경로가 열렸는지 확인한 결과로 보는 것이 맞다.

## 2. 구현 위치

주요 구현 파일은 다음과 같다.

```text
/home/icra/Pixel-Navigator/voca_memory_benchmark.py
/home/icra/Pixel-Navigator/tests/test_voca_memory_benchmark.py
/home/icra/Pixel-Navigator/scripts/download_official_benchmark_assets.sh
```

`voca_memory_benchmark.py`는 기존 Pixel-Navigator benchmark entrypoint를 직접 바꾸지 않고 별도 runner로 추가했다. 이 runner는 official Habitat env를 기준으로 episode를 reset/step하고, official metric은 `env.get_metrics()`에서 읽어 summary로 저장한다.

지원 task는 다음과 같다.

```text
--task objnav
--task pointnav
--task pointnav-test
```

지원 dataset은 다음과 같다.

```text
--dataset hm3d
--dataset mp3d
```

현재 실제 benchmark 실행 가능 상태로 확인된 것은 HM3D이다. MP3D는 episode split은 있으나 full scene asset이 아직 없어 보류 상태이다.

## 3. 데이터셋 준비 상태

HM3D official val scene asset은 로컬에 준비되어 있다.

```text
/home/icra/habitat_data/versioned_data/hm3d-0.2/hm3d/
```

Habitat/Pixel-Navigator가 기대하는 경로로 symlink를 구성했다.

```text
/home/icra/habitat_data/scene_datasets/hm3d
  -> /home/icra/habitat_data/versioned_data/hm3d-0.2/hm3d

/home/icra/habitat_data/scene_datasets/hm3d_v0.2
  -> /home/icra/habitat_data/scene_datasets/hm3d

/home/icra/Pixel-Navigator/data
  -> /home/icra/habitat_data
```

episode split 준비 상태는 다음과 같다.

| 항목 | 상태 |
|---|---|
| ObjectNav HM3D v2 split | 준비됨 |
| PointNav HM3D v1 split | 준비됨 |
| ObjectNav MP3D v1 split | episode split만 준비됨 |
| PointNav MP3D v1 split | episode split만 준비됨 |
| HM3D val scene assets | 준비됨 |
| MP3D full scene assets | 아직 없음 |

HM3D val 기준 scene count는 다음과 같이 확인했다.

```text
HM3D val basis scenes: 100
HM3D val semantic scenes: 36
```

Matterport API token 정보는 보고서에 기록하지 않는다. 채팅에 노출된 토큰은 사용 후 revoke/rotate하는 것이 안전하다.

## 4. 검증 결과

### 4.1 Unit Test

실행 명령:

```bash
cd /home/icra/Pixel-Navigator
/home/icra/micromamba/bin/micromamba run -r /home/icra/micromamba-root -n habitat \
python -m unittest /home/icra/Pixel-Navigator/tests/test_voca_memory_benchmark.py
```

결과:

```text
Ran 7 tests
OK
```

검증한 내용:

```text
PixelNav rollout -> ActionOutcome 변환
unsupported PixelNav action 처리
ObjNav official metrics summary 저장
PointNav goal position -> goal_map_xy 변환
Qwen 실패 시 heuristic fallback 처리
PixelNav-friendly heuristic point 생성
official PointNav HM3D config가 다운로드된 split/scene root를 바라보는지 확인
Habitat metric 안의 numpy array를 JSON-safe summary로 저장
```

### 4.2 ObjectNav HM3D Official Smoke

실행 명령:

```bash
cd /home/icra/Pixel-Navigator
/home/icra/micromamba/bin/micromamba run -r /home/icra/micromamba-root -n habitat \
python voca_memory_benchmark.py \
  --task objnav \
  --dataset hm3d \
  --vlm heuristic \
  --eval-episodes 1 \
  --max-agent-steps 1 \
  --max-pixelnav-steps 1 \
  --pixelnav-device cpu \
  --out /home/icra/Pixel-Navigator/reports/voca_memory_objnav_hm3d_official_smoke_jsonfix
```

결과 summary:

```text
benchmark_type: official_habitat_objnav_with_voca_memory
episodes: 1
success: 0.0
spl: 0.0
soft_spl: 0.0
mean_distance_to_goal: 4.591166
mean_agent_steps: 1.0
mean_memory_nodes: 1.0
mean_memory_edges: 0.0
```

산출물:

```text
/home/icra/Pixel-Navigator/reports/voca_memory_objnav_hm3d_official_smoke_jsonfix/voca_memory_objnav_benchmark_summary.json
/home/icra/Pixel-Navigator/reports/voca_memory_objnav_hm3d_official_smoke_jsonfix/voca_memory_objnav_benchmark_metrics.csv
/home/icra/Pixel-Navigator/reports/voca_memory_objnav_hm3d_official_smoke_jsonfix/episode_0000/memory/memory_graph.json
```

확인된 scene:

```text
/home/icra/habitat_data/scene_datasets/hm3d_v0.2/val/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.basis.glb
```

### 4.3 PointNav HM3D Official Smoke

실행 명령:

```bash
cd /home/icra/Pixel-Navigator
/home/icra/micromamba/bin/micromamba run -r /home/icra/micromamba-root -n habitat \
python voca_memory_benchmark.py \
  --task pointnav \
  --dataset hm3d \
  --vlm heuristic \
  --eval-episodes 1 \
  --max-agent-steps 1 \
  --max-pixelnav-steps 1 \
  --pixelnav-device cpu \
  --out /home/icra/Pixel-Navigator/reports/voca_memory_pointnav_hm3d_official_smoke
```

결과 summary:

```text
benchmark_type: official_habitat_pointnav_with_voca_memory
episodes: 1
success: 0.0
spl: 0.0
soft_spl: 0.0
mean_distance_to_goal: 12.040049
mean_agent_steps: 1.0
mean_memory_nodes: 1.0
mean_memory_edges: 0.0
```

산출물:

```text
/home/icra/Pixel-Navigator/reports/voca_memory_pointnav_hm3d_official_smoke/voca_memory_pointnav_benchmark_summary.json
/home/icra/Pixel-Navigator/reports/voca_memory_pointnav_hm3d_official_smoke/voca_memory_pointnav_benchmark_metrics.csv
/home/icra/Pixel-Navigator/reports/voca_memory_pointnav_hm3d_official_smoke/episode_0000/memory/memory_graph.json
```

확인된 scene:

```text
/home/icra/habitat_data/scene_datasets/hm3d/val/00884-XMHNu9rRQ1y/XMHNu9rRQ1y.basis.glb
```

## 5. 해석

현재 확인된 것은 성능 검증이 아니라 실행성 검증이다.

확인된 것:

```text
HM3D ObjectNav official env가 생성되고 episode를 실행할 수 있음
HM3D PointNav official env가 생성되고 episode를 실행할 수 있음
PixelNav rollout 결과가 ActionOutcome으로 변환됨
v5 memory graph가 node/edge artifact로 저장됨
Habitat official metrics가 JSON/CSV로 저장됨
top_down_map 같은 numpy 기반 metric도 summary 저장에서 깨지지 않음
```

아직 확인되지 않은 것:

```text
충분한 episode 수에서의 SR/SPL/Success 성능
Qwen 실사용 시 JSON 안정성
memory graph가 장기 episode에서 실제로 성능을 개선하는지
VLM grounded goal이 PixelNav local action과 항상 잘 맞는지
GPU PixelNav 실행 안정성
```

따라서 현재 상태는 “내일부터 HM3D official benchmark 실험을 시작할 수 있는 상태”로 정리하는 것이 정확하다.

## 6. 다음 실행 명령

HM3D ObjectNav benchmark:

```bash
cd /home/icra/Pixel-Navigator
/home/icra/micromamba/bin/micromamba run -r /home/icra/micromamba-root -n habitat \
python voca_memory_benchmark.py \
  --task objnav \
  --dataset hm3d \
  --vlm heuristic \
  --eval-episodes 10 \
  --max-agent-steps 50 \
  --max-pixelnav-steps 12 \
  --pixelnav-device cpu \
  --out reports/objnav_hm3d_heuristic_e10
```

HM3D PointNav benchmark:

```bash
cd /home/icra/Pixel-Navigator
/home/icra/micromamba/bin/micromamba run -r /home/icra/micromamba-root -n habitat \
python voca_memory_benchmark.py \
  --task pointnav \
  --dataset hm3d \
  --vlm heuristic \
  --eval-episodes 10 \
  --max-agent-steps 50 \
  --max-pixelnav-steps 12 \
  --pixelnav-device cpu \
  --out reports/pointnav_hm3d_heuristic_e10
```

Qwen 사용 시:

```bash
export QWEN_BASE_URL="http://server-01.cgv:8000/v1"
export QWEN_API_KEY="EMPTY"
export QWEN_MODEL="qwen3-vl-32b-thinking"
export QWEN_TIMEOUT_S="600"
export QWEN_MAX_TOKENS="2048"

cd /home/icra/Pixel-Navigator
/home/icra/micromamba/bin/micromamba run -r /home/icra/micromamba-root -n habitat \
python voca_memory_benchmark.py \
  --task pointnav \
  --dataset hm3d \
  --vlm qwen \
  --eval-episodes 10 \
  --max-agent-steps 50 \
  --max-pixelnav-steps 12 \
  --pixelnav-device cpu \
  --out reports/pointnav_hm3d_qwen_e10
```

## 7. 남은 이슈

### Qwen JSON 안정성

이전 smoke에서 Qwen이 strict JSON이 아니라 prose reasoning을 반환한 경우가 있었다. 현재 runner는 fallback을 통해 episode 중단은 방지하지만, 성능 실험에서는 JSON extraction retry 또는 system prompt 강화가 필요하다.

### PixelNav device

현재 안정적으로 확인한 실행은 `--pixelnav-device cpu`이다. Qwen 서버가 GPU를 점유하는 상황에서는 Habitat rendering은 GPU/OpenGL을 사용하더라도 PixelNav policy inference는 CPU로 돌리는 구성이 안전하다.

### Full benchmark 성능

현재 SR/SPL 값은 1-step smoke 결과이므로 낮게 나오는 것이 정상이다. 성능 보고를 위해서는 최소 10 episode smoke, 이후 100+ episode 평가로 확장해야 한다.

### MP3D

MP3D episode split은 있으나 full MP3D scene meshes가 아직 없다. HM3D 실험을 우선 진행하고, MP3D는 Matterport3D licensed scene asset 확보 후 연결하면 된다.

## 8. 오늘 기준 결론

HM3D 기준 ObjectNav와 PointNav는 모두 official Habitat split 기반 runner로 실행 가능한 상태까지 왔다.

오늘까지의 핵심 성과는 다음이다.

```text
official Habitat env
  -> grounded goal selection
  -> PixelNav local execution
  -> ActionOutcome conversion
  -> qwen_nav_memory_framework_v5 graph update
  -> official SR/SPL metric export
```

이 end-to-end 실험 경로가 HM3D ObjectNav와 HM3D PointNav에서 모두 smoke 수준으로 검증되었다. 다음 단계는 episode 수를 늘려 실제 SR/SPL/Success를 측정하고, memory 사용 여부 및 VLM/heuristic 차이에 대한 ablation을 돌리는 것이다.
