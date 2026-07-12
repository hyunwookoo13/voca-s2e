# VOCA Memory Official Benchmark Readiness

## 결론

`Pixel-Navigator` 안에 VOCA/Qwen grounded goal + v5 memory integration을 붙인 official Habitat benchmark runner를 추가했다.

추가 파일:

```text
/home/icra/Pixel-Navigator/voca_memory_benchmark.py
/home/icra/Pixel-Navigator/tests/test_voca_memory_benchmark.py
```

이 runner는 기존 `objnav_benchmark.py`를 직접 수정하지 않고, 별도 entrypoint로 동작한다.

```text
Habitat official env
  -> NavMemoryAgent(v5)
  -> Qwen or heuristic VLM
  -> PixelNav Policy_Agent rollout
  -> ActionOutcome
  -> v5 memory graph
  -> Habitat official metrics(success/spl/distance_to_goal)
```

## 현재 검증 상태

### 1. Unit test

```bash
/home/icra/micromamba/bin/micromamba run -r /home/icra/micromamba-root -n habitat \
python -m unittest /home/icra/Pixel-Navigator/tests/test_voca_memory_benchmark.py
```

결과:

```text
Ran 6 tests
OK
```

검증 내용:

- fake Habitat env에서 PixelNav rollout이 ActionOutcome으로 변환됨
- unsupported PixelNav action이 crash가 아니라 rollout failure로 기록됨
- ObjNav benchmark summary가 official metric 형태로 저장됨
- PointNav episode goal position이 `goal_map_xy`로 변환됨
- Qwen JSON 실패 시 heuristic fallback으로 episode 중단을 방지함
- PixelNav-friendly fallback point를 사용함

### 2. Official PointNav test-scene smoke

실행 명령:

```bash
/home/icra/micromamba/bin/micromamba run -r /home/icra/micromamba-root -n habitat \
python /home/icra/Pixel-Navigator/voca_memory_benchmark.py \
  --task pointnav-test \
  --eval-episodes 1 \
  --max-agent-steps 2 \
  --max-pixelnav-steps 8 \
  --vlm heuristic \
  --pixelnav-device cpu \
  --out /home/icra/Pixel-Navigator/reports/voca_memory_pointnav_official_heuristic_640x480
```

결과:

```text
benchmark_type = official_habitat_pointnav_with_voca_memory
episodes = 1
success = 0.0
spl = 0.0
mean_distance_to_goal = 8.73196
mean_memory_nodes = 1.0
mean_memory_edges = 0.0
```

해석:

- Habitat official PointNav test env는 실제로 생성되고 reset/step/metric read까지 동작한다.
- v5 memory graph JSON이 생성된다.
- PixelNav rollout artifact가 생성된다.
- 단, 현재 smoke는 goal 도달 실험이 아니라 integration 실행성 확인이다.
- PixelNav policy가 해당 frame/point에서 `look_down`, `turn_left` 위주 action을 내서 위치 이동은 발생하지 않았다.

### 3. Qwen 포함 official PointNav smoke

실행 명령:

```bash
export QWEN_BASE_URL="http://localhost:8000/v1"
export QWEN_API_KEY="EMPTY"
export QWEN_MODEL="qwen3-vl-32b-thinking"
export QWEN_TIMEOUT_S="600"
export QWEN_MAX_TOKENS="2048"
export QWEN_MAX_JSON_RETRIES="0"
export QWEN_EXTRA_PAYLOAD_JSON='{"response_format":{"type":"json_object"}}'
export PIXELNAV_DEVICE="cpu"

/home/icra/micromamba/bin/micromamba run -r /home/icra/micromamba-root -n habitat \
python /home/icra/Pixel-Navigator/voca_memory_benchmark.py \
  --task pointnav-test \
  --eval-episodes 1 \
  --max-agent-steps 1 \
  --max-pixelnav-steps 1 \
  --vlm qwen \
  --pixelnav-device cpu \
  --out /home/icra/Pixel-Navigator/reports/voca_memory_pointnav_official_qwen_fallback_smoke_v2
```

결과:

```text
benchmark_type = official_habitat_pointnav_with_voca_memory
episodes = 1
success = 0.0
spl = 0.0
mean_memory_nodes = 1.0
vlm_fallback_count = 1
```

Qwen은 JSON 대신 prose reasoning을 반환했고, runner가 heuristic fallback으로 episode 중단을 방지했다.

```text
failed to extract VLM JSON ... no JSON object found in model response
```

## ObjNav 상태

ObjNav runner도 `--task objnav --dataset hm3d|mp3d` 경로로 연결했다.

다만 현재 로컬 환경에서는 HM3D scene asset이 없어 official ObjNav env 생성이 실패한다.

확인된 에러:

```text
No Stage Attributes exists for requested scene
/home/icra/habitat_data/scene_datasets/hm3d_v0.2/val/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.basis.glb
```

현재 있는 것:

```text
/home/icra/habitat_data/datasets/objectnav_hm3d_v2/...
/home/icra/habitat_data/datasets/objectnav_mp3d_v1/...
```

부족한 것:

```text
/home/icra/habitat_data/scene_datasets/hm3d_v0.2/...
/home/icra/habitat_data/scene_datasets/mp3d/...
```

즉 ObjectNav episode JSON은 있으나 scene render asset이 부족하다.

## 다음 이슈

1. Qwen JSON 안정화
   - 현재 Qwen Thinking이 JSON 대신 prose를 반환한다.
   - fallback은 넣었지만, 성능 실험에서는 strict JSON completion 또는 better extraction이 필요하다.

2. PixelNav local action 병목
   - 공식 PointNav test frame에서 PixelNav가 `look_down/turn_left` 위주로 동작해 이동이 없었다.
   - VLM fine point selection, camera pitch, sensor config, goal mask 위치를 추가 sweep해야 한다.

3. ObjNav asset 준비
   - HM3D/MP3D scene asset이 있어야 official ObjNav SR/SPL을 돌릴 수 있다.

4. full episode 성능
   - 현재는 1-2 step smoke다.
   - long episode에서 memory node/edge가 증가하고 활용되는지는 다음 실험으로 확인해야 한다.

## 현재 가능한 실행

PointNav test-scene official smoke:

```bash
/home/icra/micromamba/bin/micromamba run -r /home/icra/micromamba-root -n habitat \
python /home/icra/Pixel-Navigator/voca_memory_benchmark.py \
  --task pointnav-test \
  --eval-episodes 1 \
  --max-agent-steps 10 \
  --max-pixelnav-steps 12 \
  --vlm qwen \
  --pixelnav-device cpu \
  --out /home/icra/Pixel-Navigator/reports/manual_pointnav_qwen_run
```

ObjNav official run, after HM3D scene assets are installed:

```bash
/home/icra/micromamba/bin/micromamba run -r /home/icra/micromamba-root -n habitat \
python /home/icra/Pixel-Navigator/voca_memory_benchmark.py \
  --task objnav \
  --dataset hm3d \
  --eval-episodes 1 \
  --max-agent-steps 10 \
  --max-pixelnav-steps 12 \
  --vlm qwen \
  --pixelnav-device cpu \
  --out /home/icra/Pixel-Navigator/reports/manual_objnav_qwen_run
```
