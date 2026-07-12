# 01. 연구 경계와 execution-grounded architecture 정립

## 목표

VLM이 장면을 설명하는 보조 모듈이 아니라, 실행 가능한 local navigation skill을 감독하는 semantic supervisor가 되도록 시스템 경계를 다시 정의했다. 이후 PixelNav를 S2E로 교체해도 VLM action, memory, audit 계약은 유지하는 것이 핵심이다.

## 최종 구조

```text
Object category + target-context cues + RGB observations + coarse goal
                              |
                              v
                    Qwen-VLM supervisor
               go / rotate / observe / stop
                              |
              +---------------+----------------+
              |                                |
        RGB candidate gate              v6 graph memory
              |                                |
              +---------- PixelNav ------------+
                          execution
                              |
                     execution evidence
                              |
                 supervisor/memory update
```

## 개발한 내용

- `goal_seek`, `escape_deadlock`, `backtrack`, `verify_target`의 explicit supervisor mode를 정의했다.
- local execution success와 strategic progress를 분리했다.
- collision 없이 방을 빠져나가는 detour는 hidden goal distance가 증가해도 실패로 기록하지 않는다.
- Habitat의 `distance_to_goal`, `success`, `spl`, shortest path, top-down map을 Qwen prompt와 online memory decision에서 차단했다.
- simulator pose 사용은 `habitat_sim_pose_declared` localization contract로 명시했다.
- evaluation metrics는 action 이후 audit channel에만 기록한다.
- `voca_s2e_bridge.py`로 memory framework와 executor 경계를 분리했다. S2E backend 연결은 후속 단계다.

## 주요 코드

- `navigation_supervisor.py`
- `qwen_action_runner.py`
- `qwen_vlm_planner.py`
- `voca_s2e_bridge.py`
- `docs/superpowers/specs/2026-07-10-execution-grounded-vlm-navigation-design.md`

## 검증

- policy input 전체를 recursive 검사해 forbidden evaluation key가 들어가면 fail-closed 처리한다.
- detour 상황에서 execution success와 strategic progress가 분리되는 테스트를 추가했다.
- sidecar와 planner가 동일 supervisor mode를 보는 synchronization test를 추가했다.
- 전체 regression 292개가 통과했다.

## 추천 영상

- `media/01_chair_end_to_end_highlight_49s.mp4`
  - `GO`, `REQUEST_OBSERVATION`, `ESCAPE`, `VERIFY TARGET`, `STOP`이 하나의 closed loop에서 바뀌는 모습을 설명하기 좋다.
- `media/04_chair_topdown_highlight_41s.mp4`
  - evaluation-only map임을 명확히 말한 뒤 detour가 실제 이동으로 이어졌음을 보인다.

## 현재 한계

- S2E는 interface 수준에서 준비됐지만 실제 fast navigation backend로 연결되지 않았다.
- pose를 정책 입력으로 사용하므로 RGB-only claim은 불가능하다.
- architecture의 타당성은 확보했지만 대규모 multi-seed 성능 우위는 아직 검증하지 않았다.
