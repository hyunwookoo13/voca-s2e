import numpy as np
from settings import LLM_BACKEND

if LLM_BACKEND == "gemini":
    from llm_utils.gemini_request import text_response, vision_response
elif LLM_BACKEND == "qwen":
    from llm_utils.qwen_request import text_response, vision_response
elif LLM_BACKEND == "ollama":
    from llm_utils.ollama_request import text_response, vision_response
else:
    raise ValueError("Unsupported VOCA_LLM_BACKEND: {}".format(LLM_BACKEND))

from llm_utils.navigation_prompts import VLM_PROMPT, PRIORS_PROMPT, PRIOR_CLASS_LIST
from llm_utils.priors_parser import parse_llm_json, extract_priors, parse_decision_json, dedupe_preserve_order, parse_prior_class_block
from cv_utils.yoloe_detector import *
from typing import List, Dict, Any, Tuple, Optional, Union
import cv2
import time  # <-- 추가
# 변경/추가 import
import os   # NEW
import json # NEW
import math


class VLMPlanner:
    def __init__(self,yoloe_model):
        self.vlm_trajectory = []
        self.yoloe_model = yoloe_model
        self.detect_objects = ['bed','sofa','chair','houseplant','tv monitor','toilet']
        # ---- LLM/플래너/모듈별 시간 계측 저장소 ----
        self.llm_call_count = 0
        self.llm_success_count = 0      # 응답 문자열을 받은 호출 수
        self.llm_error_count = 0        # 예외로 실패한 호출 수
        self.llm_last_error = ""        # 마지막 예외 메시지
        self.llm_durations = []          # 각 LLM 호출 소요시간(초)
        self.planner_durations = []      # 각 make_plan 호출 전체 소요시간(초)
        self.yoloe_durations = []        # 각 YOLOE object detection 호출 소요시간(초)
        # ---- Priors 저장소 ----
        self.priors_log = []       # 에피소드 동안 쌓인 priors 히스토리 (리스트)
        self.latest_priors = None  # 가장 최근 priors 스냅샷 (딕셔너리)
        self._last_prompt = None
        # 🔧 내부 프롬프트/클래스 캐시
        self._prompt_classes = None
        self._floor_aliases = ['floor', 'ground', 'flooring']

        # 로깅 폴더 (끄기: None)
        self._logdir = None  # NEW


    # -------------------------
    # (선택) 로그 저장 활성화
    # -------------------------
    def enable_run_logger(self, out_dir: str):  # NEW
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
        self._logdir = out_dir

    def _dump_llm_call(self, snapshot: dict):  # NEW
        """각 LLM 호출 결과를 JSONL로 누적 저장 + 원본/파노라마/선택 이미지 파일로도 보관."""
        if not self._logdir:
            return
        # 1) JSONL
        path = os.path.join(self._logdir, "llm_calls.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        # 2) 텍스트(raw)
        if "raw" in snapshot and isinstance(snapshot["raw"], str):
            idx = snapshot.get("call_index", len(self.priors_log))
            with open(os.path.join(self._logdir, f"raw_{idx:04d}.txt"), "w", encoding="utf-8") as f:
                f.write(snapshot["raw"])

    def _record_llm_exception(self, stage: str, try_idx: int, exc: Exception):
        self.llm_error_count += 1
        self.llm_last_error = "{}[try={}] {}: {}".format(
            stage, try_idx + 1, type(exc).__name__, str(exc)
        )
        print("[LLM/{}] ERROR try {}/10: {}".format(
            stage, try_idx + 1, self.llm_last_error
        ))

    def export_priors_log(self, path: str):  # NEW
        """에피소드 종료 시 전체 priors_log를 한 번에 저장하고 싶다면 호출."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.priors_log, f, ensure_ascii=False, indent=2)


    def reset(self,object_goal):
        # translation to align for the detection model
        if object_goal == 'tv_monitor':
            self.object_goal = 'tv monitor'
        elif object_goal == 'plant':
            self.object_goal = 'houseplant'
        else:
            self.object_goal = object_goal

        self.vlm_trajectory = []
        self.panoramic_trajectory = []
        self.direction_image_trajectory = []
        self.direction_mask_trajectory = []

        # ---- 에피소드 시작 시 계측 초기화 ----
        self.llm_call_count = 0
        self.llm_success_count = 0
        self.llm_error_count = 0
        self.llm_last_error = ""
        self.llm_durations = []
        self.planner_durations = []
        self.yoloe_durations = []
        self.priors_log = []
        self.latest_priors = None

        # 🔧 캐시/프롬프트 상태 완전 초기화
        self._last_prompt = None
        self._prompt_classes = None
        self._floor_aliases = ['floor', 'ground', 'flooring']

        self._set_prompt_from_priors({})  # 빈 priors로 초기 클래스셋 구성

    def concat_panoramic(self,images,angles):
        try:
            height,width = images[0].shape[0],images[0].shape[1]
        except:
            height,width = 480,640
        background_image = np.zeros((2*height + 3*10, 3*width + 4*10,3),np.uint8)
        copy_images = np.array(images,dtype=np.uint8)
        for i in range(len(copy_images)):
            if i % 2 == 0:
               continue
            copy_images[i] = cv2.putText(copy_images[i],"Angle %d"%angles[i],(100,100),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6, cv2.LINE_AA)
            row = i // 6
            col = (i//2) % 3
            background_image[10*(row+1)+row*height:10*(row+1)+row*height+height:,10*(col+1)+col*width:10*(col+1)+col*width+width,:] = copy_images[i]
        return background_image

    def make_plan(self, pano_images):
        """
        Returns:
            goal_image_rgb (np.ndarray, RGB):
                정책 입력용 목표 이미지 (RGB). 선택된 방향 프레임 자체를 복사한 것.

            debug_mask (np.ndarray, uint8 HxW):
                정책 입력용 0/255 마스크. 현재 subgoal로 가야 할 픽셀 주변(작은 정사각형)만 255.

            debug_image (np.ndarray, RGB):
                선택된 방향 프레임에 subgoal 위치를 네모로 시각적으로 찍어둔 디버그용 이미지.

            vis_rgb (np.ndarray, RGB):
                YOLOE detection 결과와 subgoal 위치 마커를 그려둔 시각화 이미지.

            direction (int):
                파노라마에서 선택된 뷰의 인덱스. (ex. 0~11 또는 0~4 같은 로컬 yaw index)

            pri_flag (bool):
                "목표 카테고리로 분류된 박스가 프레임 안에 있었다"고 판단되면 True.
                즉 클래스 매칭만으로도 후보 타깃이라고 본 상태 (confidence 낮아도 True 가능).

            obj_detected (bool):
                실제 타깃이라고 확신할 수 있을 정도로 신뢰도가 높은 박스가 있다고 본 경우 True.
                (conf ≥ MIN_TARGET_CONF 이고 lookalike와 과도하게 겹치지 않는 경우)
        """

        _plan_t0 = time.perf_counter()

        # 1) LLM로 진행 방향/priors 결정
        direction, vlm_obj_detected = self.query_vlm(pano_images)
        direction_image = pano_images[direction]  # RGB

        # 2) YOLOE 클래스 프롬프트 보장 (priors 반영)
        self._set_prompt_from_priors(self.latest_priors or {})

        # 3) priors-기반 웨이포인트 선택을 apply_priors_on_image로 통일
        #    - 이 함수가 YOLOE 박스 → priors 스코어링 → 바닥 폴백까지 수행
        goal_image_rgb, debug_mask, pri_flag, bb_obj_detected, vis_rgb = self.apply_priors_on_image(direction_image, MIN_TARGET_CONF=0.10, MIN_TARGET_AREA=0.0625)

        obj_detected = vlm_obj_detected and bb_obj_detected

        # 4) 디버그 이미지에 사각형 찍기 (PixNav 스타일)  # <-- changed
        debug_image = np.array(direction_image)  # copy
        ys, xs = np.where(debug_mask > 0)
        if len(xs) > 0:
            px = int(np.mean(xs))
            py = int(np.mean(ys))
        else:
            H, W = debug_image.shape[:2]
            px, py = W // 2, H // 2

        r = 8
        cv2.rectangle(
            debug_image,
            (int(px - r), int(py - r)),
            (int(px + r), int(py + r)),
            (255, 0, 0),
            -1
        )

        # 5) 내부 로그/트래젝토리 업데이트
        self.direction_image_trajectory.append(direction_image)
        self.direction_mask_trajectory.append(debug_mask.copy())
        self.planner_durations.append(time.perf_counter() - _plan_t0)

        # 6) 리턴 (기존 시그니처 유지)
        return goal_image_rgb, debug_mask, debug_image, vis_rgb, direction, pri_flag, obj_detected


    def _set_prompt_from_priors(self, priors: dict):
        floor_aliases = ['floor', 'ground', 'flooring']

        supports   = set(map(str.lower, (priors or {}).get("Supports", []) or []))
        cooccurs   = set(map(str.lower, (priors or {}).get("StrongCooccurs", []) or []))
        gateways   = set(map(str.lower, (priors or {}).get("Gateways", []) or []))
        lookalikes = set(map(str.lower, (priors or {}).get("Lookalikes", []) or []))

        # 1) PRIOR_CLASS_LIST 정적 어휘
        static_list = parse_prior_class_block(PRIOR_CLASS_LIST)

        # 2) priors + 바닥 alias + 정적 어휘 결합 → 중복 제거
        extra_positive = list(
            supports | cooccurs | gateways | lookalikes | set(floor_aliases)
        )
        prompt_classes = dedupe_preserve_order(self.detect_objects+ static_list + extra_positive)

        # 모델 클래스셋 적용 (변경 시에만)
        if self._last_prompt != tuple(prompt_classes):
            set_yoloe_classes(self.yoloe_model, prompt_classes)
            self._last_prompt = tuple(prompt_classes)

        self._prompt_classes = prompt_classes
        self._floor_aliases = floor_aliases

    def query_priors_text(self):
        """
        이미지 없이 텍스트만으로 PRIORS를 받아온다.
        - 시스템 프롬프트: PRIORS_PROMPT
        - 유저 프롬프트: "<Target Object>:<name>"
        - 쿼리 시도: 10회
        반환: 표준 priors(dict)
            {
            "Supports": [...],
            "StrongCooccurs": [...],
            "Gateways": [...],
            "Lookalikes": [...]
            }
        """
        text_content = "<Target Object>:{}\n".format(self.object_goal)
        # (선택) 트래젝토리 로그에 입력 기록
        self.vlm_trajectory.append("\nInput(priors):\n%s \n" % text_content)

        raw_answer = None
        priors = None

        for try_idx in range(10):
            print("[LLM/PRIORS] try {}/10".format(try_idx + 1))
            t0 = time.perf_counter()
            try:
                # 이미지 없이 텍스트 전용 호출
                raw_answer = text_response(text_content + PRIOR_CLASS_LIST, system_prompt=PRIORS_PROMPT)
                if raw_answer:
                    self.llm_success_count += 1
            except Exception as e:
                raw_answer = None
                self._record_llm_exception("PRIORS", try_idx, e)
            finally:
                self.llm_call_count += 1
                dt = time.perf_counter() - t0
                self.llm_durations.append(dt)
                print("[LLM/PRIORS] elapsed={:.3f}s has_response={}".format(dt, bool(raw_answer)))

            if not raw_answer:
                continue

            # priors 전용 파서 (각도/플래그/리즌 미사용)
            parsed = None
            try:
                parsed = parse_llm_json(raw_answer)  # alias: priors만 반환
            except Exception:
                parsed = None
            if not parsed:
                print("[LLM/PRIORS] raw response:\n{}".format(raw_answer))
                print("[LLM/PRIORS] parse failed, retrying.")
                continue

            # 표준 PRIORS 딕셔너리만 추출
            priors = extract_priors(parsed, raw_answer) or {}

            # 키 보장
            for k in ("Supports", "StrongCooccurs", "Gateways", "Lookalikes"):
                priors.setdefault(k, [])

            # 내부 로그(원하면 파일 저장 훅도 사용 가능)
            snapshot = {
                "target": self.object_goal,
                "priors": priors,
                "raw": raw_answer,
                "call_index": try_idx,
                "call_type": "text_priors",
            }
            self.priors_log.append(snapshot)
            # self._dump_llm_call(snapshot)  # 주석 해제 시 파일로도 저장

            print("[LLM/PRIORS] sizes:",
                len(priors.get("Supports", [])),
                len(priors.get("StrongCooccurs", [])),
                len(priors.get("Gateways", [])),
                len(priors.get("Lookalikes", [])))
            break  # 성공 시도 종료

        # 실패 시 안전 디폴트
        if not isinstance(priors, dict):
            priors = {
                "Supports": [],
                "StrongCooccurs": [],
                "Gateways": [],
                "Lookalikes": [],
            }

        # 상태 업데이트 + YOLOE 클래스셋 적용
        self.latest_priors = priors
        self._set_prompt_from_priors(priors)

        # (선택) 응답 원문도 트래젝토리에 남김
        self.vlm_trajectory.append("PRIORS Answer:\n%s" % (raw_answer if raw_answer is not None else "<EMPTY>"))

        return priors


    def query_vlm(self, pano_images):
        """
        LLM에서 Reason/Angle/Flag만 받아와 진행 방향을 정한다.
        priors는 사용/저장하지 않는다.
        """
        angles = (np.arange(len(pano_images))) * 30
        inference_image = cv2.cvtColor(self.concat_panoramic(pano_images, angles), cv2.COLOR_BGR2RGB)

        # (옵션) 기록용 저장
        cv2.imwrite("monitor-panoramic.jpg", inference_image)

        text_content = "<Target Object>:{}\n".format(self.object_goal)
        self.vlm_trajectory.append("\nInput:\n%s \n" % text_content)
        self.panoramic_trajectory.append(inference_image)

        raw_answer = None
        parsed = None

        def _to_bool(x):
            if isinstance(x, bool):
                return x
            if isinstance(x, (int, float)):
                return x != 0
            if isinstance(x, str):
                return x.strip().lower() in ("true", "1", "yes", "y", "t")
            return False

        valid_angles = set(int(x) for x in angles.tolist())

        for try_idx in range(10):
            print("[LLM/VLM] try {}/10".format(try_idx + 1))
            t0 = time.perf_counter()
            try:
                raw_answer = vision_response(text_content, inference_image, VLM_PROMPT)
                if raw_answer:
                    self.llm_success_count += 1
            except Exception as e:
                raw_answer = None
                self._record_llm_exception("VLM", try_idx, e)
            finally:
                self.llm_call_count += 1
                dt = time.perf_counter() - t0
                self.llm_durations.append(dt)
                print("[LLM/VLM] elapsed={:.3f}s has_response={}".format(dt, bool(raw_answer)))

            if not raw_answer:
                continue

            # Reason/Angle/Flag만 파싱
            parsed = parse_decision_json(raw_answer, valid_angles=list(valid_angles))
            if not parsed:
                print("[LLM/VLM] raw response:\n{}".format(raw_answer))
                print("[LLM/VLM] decision parse failed, retrying.")
                continue

            # Angle 검증
            try:
                a = int(parsed.get("Angle"))
            except Exception:
                a = None
            if a is None or a not in valid_angles:
                continue

            # Flag 안정 변환
            flag = _to_bool(parsed.get("Flag", False))
            reason = parsed.get("Reason", "")

            # (선택) 호출 로그 저장 — priors는 비움
            snapshot = {
                "target": self.object_goal,
                "angles": list(map(int, angles.tolist())) if hasattr(angles, "tolist") else list(angles),
                "selected_angle": int(a),
                "flag": bool(flag),
                "reason": reason,
                "raw": raw_answer,
                "call_index": try_idx,
            }

            # 파일로 남기고 싶으면 주석 해제
            # self._dump_llm_call(snapshot)

            print("[LLM] raw response:\n{}".format(raw_answer))
            print("[LLM] angle:", a, "| flag:", flag)
            if reason:
                print("[LLM] reason:", reason)
            break  # 성공

        self.vlm_trajectory.append("VLM Answer:\n%s" % (raw_answer if raw_answer is not None else "<EMPTY>"))
        self.panoramic_trajectory.append(inference_image)

        try:
            idx = (int(parsed['Angle']) // 30) % max(1, len(pano_images))
            return idx, _to_bool(parsed.get('Flag', False))
        except Exception:
            return np.random.randint(0, max(1, len(pano_images))), False

    def apply_priors_on_image(
        self,
        image_or_pano,                          # np.ndarray (RGB) or Sequence[np.ndarray]
        conf_threshold: float = 0.05,
        MIN_TARGET_CONF: float = 0.5,
        MIN_TARGET_AREA: float = 0.02,
        iou_threshold: float = 0.50,
        return_boxes: bool = False
    ):
        """
        단일 이미지 또는 파노라믹 이미지 시퀀스를 입력으로 받아
        priors 기반 웨이포인트를 선택한다.

        모든 이미지는 RGB 기준으로 처리하고 RGB로 반환한다.

        - 단일 이미지 입력(np.ndarray, RGB):
            반환값
            => (goal_rgb, debug_mask, pri_flag, obj_detected, vis_rgb)                     # return_boxes=False
            => (goal_rgb, debug_mask, pri_flag, obj_detected, vis_rgb, boxes_std)          # return_boxes=True

        - 파노라믹 입력(list/tuple of RGB frames, 보통 12장: episode_images[-12:]):
            각 방향별로 동일 로직으로 웨이포인트/점수를 산출 후
            가장 점수가 높은 방향(idx)을 goal_rotate로 선택하여 반환.
            반환값
            => (goal_rgb, debug_mask, pri_flag, obj_detected, vis_rgb, goal_rotate)                     # return_boxes=False
            => (goal_rgb, debug_mask, pri_flag, obj_detected, vis_rgb, boxes_std, goal_rotate)         # return_boxes=True
        """
        # -------- 공통: 프롬프트/클래스셋 보장 --------
        prompt_classes = getattr(self, "_prompt_classes", None)
        floor_aliases  = getattr(self, "_floor_aliases", ['floor','ground','flooring'])
        if not prompt_classes:
            self._set_prompt_from_priors(self.latest_priors or {})
            prompt_classes = self._prompt_classes
            floor_aliases  = self._floor_aliases

        # 내부 상수 (기존 로직과 동일)
        WEIGHTS = {"supports": 0.4, "cooccurs": 0.2, "gateways": 0.4, "lookalikes": 0.6}
        BETA_AREA, GAMMA_BOTTOM = 0.03, 0.05
        LA_IOU_THRES    = 0.99


        # priors 셋
        pri = self.latest_priors or {"Supports": [], "StrongCooccurs": [], "Gateways": [], "Lookalikes": []}
        supports = set(map(str.lower, pri.get("Supports", [])))
        cooccurs = set(map(str.lower, pri.get("StrongCooccurs", [])))
        gateways = set(map(str.lower, pri.get("Gateways", [])))
        lookalikes  = set(map(str.lower, pri.get("Lookalikes", [])))
        floor_set = set(map(str.lower, floor_aliases))

        # ---- 단일 프레임 처리 유틸 (점수까지 함께 반환) ----
        def _process_single(rgb: np.ndarray):
            H, W = rgb.shape[:2]
            debug_mask = np.zeros((H, W), dtype=np.uint8)

            _det_t0 = time.perf_counter()
            det = yoloe_detection(
                image=rgb,
                target_classes=prompt_classes,
                model=self.yoloe_model,
                box_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                run_extra_nms=False,
                use_text_prompt=False,
                retina_masks=False,   # 박스만
            )
            self.yoloe_durations.append(time.perf_counter() - _det_t0)

            # 안전 추출 + numpy 표준화 헬퍼
            def _first_not_none(obj, names):
                import numpy as _np
                for n in names:
                    if hasattr(obj, n):
                        v = getattr(obj, n)
                        if v is None:
                            continue
                        # torch.Tensor -> numpy
                        try:
                            import torch
                            if isinstance(v, torch.Tensor):
                                v = v.detach().cpu().numpy()
                        except Exception:
                            pass
                        # list/tuple -> numpy
                        if isinstance(v, (list, tuple)):
                            v = _np.asarray(v)
                        return v
                return None

            xyxy     = _first_not_none(det, ["xyxy", "boxes"])         # (N,4) 기대
            cls_ids  = _first_not_none(det, ["class_id", "labels", "cls"])
            box_conf = _first_not_none(det, ["confidence", "scores", "score"])

            # 1D로 평탄화(있다면)
            if box_conf is not None:
                box_conf = box_conf.reshape(-1)

            # 시각화용
            goal_name = getattr(self, "object_goal", "") or ""
            try:
                goal_idx = prompt_classes.index(goal_name)
            except ValueError:
                goal_idx = -1
            vis_bgr = draw_detections_bgr(rgb, det, goal_idx=goal_idx)

            boxes_std = []
            def _push_box(i):
                x1, y1, x2, y2 = map(float, xyxy[i])
                x1 = max(0.0, min(x1, W - 1)); y1 = max(0.0, min(y1, H - 1))
                x2 = max(x1+1.0, min(x2, W));  y2 = max(y1+1.0, min(y2, H))
                w = x2 - x1; h = y2 - y1
                cx = x1 + 0.5 * w; cy = y1 + 0.5 * h
                ci = int(cls_ids[i]) if cls_ids is not None and np.size(cls_ids) > i else None
                cn = str(prompt_classes[ci]).lower() if (ci is not None and 0 <= ci < len(prompt_classes)) else None
                sc = float(box_conf[i]) if box_conf is not None and len(box_conf) > i else None
                boxes_std.append({
                    "cls": cn, "cls_id": ci, "score": sc,
                    "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                    "bbox_xywh": [int(x1), int(y1), int(w), int(h)],
                    "center": [float(cx), float(cy)],
                    "area": float(w*h),
                    "norm": {
                        "cx": float(cx / W), "cy": float(cy / H),
                        "w": float(w / W),  "h": float(h / H),
                        "area": float((w*h) / (W*H + 1e-12)),
                    }
                })

            def _safe():
                return (xyxy is not None) and (len(xyxy) > 0) and (cls_ids is not None) and (np.size(cls_ids) > 0)

            def _area(i):
                x1, y1, x2, y2 = map(float, xyxy[i])
                return max(0.0, x2 - x1) * max(0.0, y2 - y1)

            def _norm_area(i):
                x1, y1, x2, y2 = map(float, xyxy[i])
                return max(0.0, (x2 - x1) * (y2 - y1)) / float(W * H + 1e-6)

            def _bottom_bias(i):
                x1, y1, x2, y2 = map(float, xyxy[i])
                return (max(y1, y2) / H)

            def _center(i):
                x1, y1, x2, y2 = map(float, xyxy[i])
                return int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)

            def _iou(i, j):
                x1a, y1a, x2a, y2a = map(float, xyxy[i])
                x1b, y1b, x2b, y2b = map(float, xyxy[j])
                ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
                ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
                iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                inter = iw * ih
                union = _area(i) + _area(j) - inter + 1e-9
                return inter / union

            def _name(ci):
                return str(prompt_classes[int(ci)]).lower() if ci is not None and 0 <= int(ci) < len(prompt_classes) else ""

            # 기본값
            px, py = W // 2, H // 2
            pri_flag = False
            obj_detected = False
            dir_score = -1e9  # 파노라 방향 선택용 점수

            if _safe():
                for i in range(len(xyxy)):
                    _push_box(i)
                cls_np = np.asarray(cls_ids).astype(int)

                # 1) 목표 객체가 있으면 "우선 접근"
                if goal_idx >= 0 and np.any(cls_np == goal_idx):
                    t_inds = np.where(cls_np == goal_idx)[0]
                    if len(t_inds) > 0:
                        if box_conf is not None:
                            t_inds = sorted(t_inds, key=lambda i: (float(box_conf[i]), _area(i)), reverse=True)
                            top = int(t_inds[0])
                        else:
                            top = int(t_inds[np.argmax([_area(i) for i in t_inds])])
                        px, py = _center(top)
                        conf_ok = (box_conf is not None) and (float(box_conf[top]) >= MIN_TARGET_CONF)
                        la_inds = [k for k, ci in enumerate(cls_np) if _name(ci) in lookalikes]
                        la_conflict = any(_iou(top, lk) >= LA_IOU_THRES for lk in la_inds)
                        box_area_norm = _norm_area(top)      # box 픽셀수 / (W*H)
                        big_enough = (box_area_norm >= MIN_TARGET_AREA)
                        obj_detected = bool(conf_ok and not la_conflict and big_enough)
                        pri_flag = True
                        # 목표 발견 시 방향 점수는 높은 기준으로
                        dir_score = 1.0 + (float(box_conf[top]) if box_conf is not None else 0.0) * 0.5 \
                                    + 0.1 * _norm_area(top) + 0.05 * _bottom_bias(top)

                # 2) 목표가 없거나(혹은 pri_flag=False지만) priors 기반 폴백 스코어
                if not pri_flag:
                    best_i, best_score, best_area = None, -1e9, -1.0
                    for i, ci in enumerate(cls_np):
                        name = _name(ci)
                        if name in supports:
                            base = WEIGHTS["supports"]
                        elif name in cooccurs:
                            base = WEIGHTS["cooccurs"]
                        elif name in gateways:
                            base = WEIGHTS["gateways"]
                        elif name in lookalikes:
                            base = WEIGHTS["lookalikes"]
                        else:
                            base = 0.0

                        score = base \
                                + BETA_AREA * _norm_area(i) \
                                + GAMMA_BOTTOM * _bottom_bias(i)
                        a = _area(i)

                        if (score > best_score) or (np.isclose(score, best_score) and a > best_area):
                            best_i, best_score, best_area = i, score, a

                    if best_i is not None:
                        px, py = _center(best_i)
                        dir_score = best_score
                    else:
                        # floor 폴백 (아무 priors에도 안 걸릴 때)
                        f_inds = [i for i, ci in enumerate(cls_np) if _name(ci) in floor_set]
                        if len(f_inds) > 0:
                            top = int(f_inds[np.argmax([_area(k) for k in f_inds])])
                            px, py = _center(top)
                            dir_score = WEIGHTS["supports"] * 0.1  # 아주 낮은 기본값
                        else:
                            dir_score = 0  # 진짜로 쓸만한 힌트 없음

            # --- PixNav-style mask draw (cv2.rectangle, filled) ---  # <-- changed
            r = 8
            cv2.rectangle(
                debug_mask,
                (int(px - r), int(py - r)),
                (int(px + r), int(py + r)),
                (255,),
                -1
            )

            # 선택 포인트 오버레이(색: pri_flag True=초록, False=노랑)
            cv2.drawMarker(
                vis_bgr, (px, py),
                (0, 255, 0) if pri_flag else (0, 255, 255),
                markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2
            )

            vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
            goal_rgb = rgb.copy()
            return goal_rgb, debug_mask, pri_flag, obj_detected, vis_rgb, boxes_std, dir_score

        # ---- 분기: 단일 vs 파노라믹 ----
        if isinstance(image_or_pano, (list, tuple)):
            # 파노라믹: 각 프레임 처리 → 최고 점수 idx 선택 (+ 전방 보너스)
            best = None
            best_idx = 0
            best_cmp = -1e9

            for idx, frame in enumerate(image_or_pano):
                out = _process_single(frame)   # (..., dir_score) 포함
                cmp_score = out[-1]
                if (best is None) or (cmp_score > best_cmp):
                    best = (idx, *out)
                    best_idx = idx
                    best_cmp = cmp_score
            # 선택된 방향의 아웃풋 구성
            _, goal_rgb, debug_mask, pri_flag, obj_detected, vis_rgb, boxes_std, _ = best
            self._last_bboxes = boxes_std  # 최신 박스는 선택 방향 기준으로 유지

            if return_boxes:
                return goal_rgb, debug_mask, pri_flag, obj_detected, vis_rgb, boxes_std, int(best_idx)
            else:
                return goal_rgb, debug_mask, pri_flag, obj_detected, vis_rgb, int(best_idx)

        else:
            # 단일 이미지: 기존 반환 형태 유지 (하위호환)
            goal_rgb, debug_mask, pri_flag, obj_detected, vis_rgb, boxes_std, _ = _process_single(image_or_pano)
            self._last_bboxes = boxes_std
            if return_boxes:
                return goal_rgb, debug_mask, pri_flag, obj_detected, vis_rgb, boxes_std
            else:
                return goal_rgb, debug_mask, pri_flag, obj_detected, vis_rgb

    def are_bboxes_similar(
        self,
        prev_boxes: List[Dict[str, Any]],
        curr_boxes: List[Dict[str, Any]],
        *,
        iou_thresh: float = 0.70,
        center_tol: float = 0.15,
        area_tol: float = 0.20,
        min_match_ratio: float = 0.80,
        class_sensitive: bool = True,
        ignore_classes: Optional[List[str]] = None,
        max_count_delta: int = 0,
        return_detail: bool = False,
    ) -> Union[bool, Tuple[bool, Dict[str, Any]]]:


        ignore_set = set([c.lower() for c in (ignore_classes or [])])
        MIN_CONF = 0.0  # 필요 시 0.2~0.3으로 올리세요.

        def _ok(b):
            # ★ 버그 수정: cx만 두 번 보던 문제 → cx, cy, area 모두 확인
            if not (b and ("bbox_xyxy" in b) and ("norm" in b)):
                return False
            n = b["norm"]
            return all(k in n for k in ("cx", "cy", "area"))

        def _cls_name(b):
            # ★ 클래스 이름 없으면 cls_id로 폴백
            c = b.get("cls")
            if c is None:
                cid = b.get("cls_id")
                c = str(cid) if cid is not None else None
            return str(c).lower() if c is not None else None

        def _iou_xyxy(a, b) -> float:
            ax1, ay1, ax2, ay2 = map(float, a["bbox_xyxy"])
            bx1, by1, bx2, by2 = map(float, b["bbox_xyxy"])
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
            area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
            union = area_a + area_b - inter + 1e-9
            return inter / union

        def _center_dist(a, b) -> float:
            dx = float(a["norm"]["cx"]) - float(b["norm"]["cx"])
            dy = float(a["norm"]["cy"]) - float(b["norm"]["cy"])
            return math.hypot(dx, dy)

        def _area_change(a, b) -> float:
            aa = max(1e-9, float(a["norm"]["area"]))
            bb = max(1e-9, float(b["norm"]["area"]))
            return abs(aa - bb) / max(aa, bb)

        # --- 유효 / 필터링 ---
        def _filter(v):
            out = []
            for b in (v or []):
                if not _ok(b):
                    continue
                if _cls_name(b) in ignore_set:
                    continue
                sc = b.get("score", None)
                if (sc is not None) and (sc < MIN_CONF):
                    continue
                out.append(b)
            return out

        prev = _filter(prev_boxes)
        curr = _filter(curr_boxes)

        n_prev, n_curr = len(prev), len(curr)
        detail = {"n_prev": n_prev, "n_curr": n_curr, "matched": 0, "pairs": [], "reason": ""}

        if n_prev == 0 and n_curr == 0:
            return (True, {**detail, "reason": "both empty"}) if return_detail else True
        if n_prev == 0 or n_curr == 0:
            return (False, {**detail, "reason": "one side empty"}) if return_detail else False
        if abs(n_prev - n_curr) > max_count_delta:
            return (False, {**detail, "reason": "count delta too large"}) if return_detail else False

        # --- 후보 페어 만들기: 싼 검사(center, area) 먼저 → 통과 시 IoU 계산 ---
        candidates = []
        for i, pb in enumerate(prev):
            pn = _cls_name(pb)
            for j, cb in enumerate(curr):
                if class_sensitive:
                    cn = _cls_name(cb)
                    if (pn is None) or (cn is None) or (pn != cn):
                        continue
                cdist = _center_dist(pb, cb)
                if cdist > center_tol:
                    continue
                adiff = _area_change(pb, cb)
                if adiff > area_tol:
                    continue
                iou = _iou_xyxy(pb, cb)
                if iou >= iou_thresh:
                    candidates.append((iou, i, j, cdist, adiff))

        # --- 그리디 매칭 (IoU 내림차순) ---
        candidates.sort(key=lambda x: -x[0])
        used_prev, used_curr = set(), set()
        for iou, i, j, cdist, adiff in candidates:
            if i in used_prev or j in used_curr:
                continue
            used_prev.add(i); used_curr.add(j)
            detail["pairs"].append((i, j, float(iou), float(cdist), float(adiff)))

        matched = len(detail["pairs"])
        detail["matched"] = matched
        denom = max(n_prev, n_curr)
        ratio = matched / max(1, denom)
        similar = ratio >= min_match_ratio
        detail["match_ratio"] = float(ratio)
        detail["reason"] = "ok" if similar else "low match ratio"

        return (similar, detail) if return_detail else similar
