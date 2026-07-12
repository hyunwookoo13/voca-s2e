# cv_utils/yoloe_detector.py

from dataclasses import dataclass
from typing import List, Optional, Tuple
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLOE
from settings import DEFAULT_DEVICE, YOLOE_CHECKPOINT_PATH


@dataclass
class Detections:
    """
    Boxes + (optional) segmentation masks.
    - xyxy: (N, 4) float32 bounding boxes in pixel coords [x1,y1,x2,y2]
    - masks: (N, H, W) uint8 {0,1} or None
    - class_id: (N,) int, mapped to the prompt `class_names`
    - confidence: (N,) float
    - class_names: list of class names used for prompting (for reference)
    """
    xyxy: Optional[np.ndarray]           # (N,4) float32
    masks: Optional[np.ndarray]          # (N,H,W) uint8 {0,1}
    class_id: np.ndarray                 # (N,)
    confidence: np.ndarray               # (N,)
    class_names: List[str]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_weights_path(weights: Optional[str]) -> str:
    def _try_paths(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        p = Path(raw)
        if p.is_file():
            return str(p)
        pr = _project_root() / raw.lstrip("./")
        if pr.is_file():
            return str(pr)
        return None

    hit = _try_paths(weights) or _try_paths(os.environ.get("YOLOE_WEIGHTS")) or _try_paths(YOLOE_CHECKPOINT_PATH)
    if hit:
        return hit

    candidates = []
    for base in [(_project_root() / "checkpoints"), (Path.cwd() / "checkpoints")]:
        if base.is_dir():
            candidates += [str(x) for x in base.glob("yoloe*-seg.pt")]
    hint = "\nAvailable in checkpoints:\n  " + "\n  ".join(candidates) if candidates else ""
    raise FileNotFoundError(
        "YOLOE weights not found.\n"
        f"Tried:\n"
        f"  weights arg: {weights}\n"
        f"  env YOLOE_WEIGHTS: {os.environ.get('YOLOE_WEIGHTS')}\n"
        f"  settings.YOLOE_CHECKPOINT_PATH: {YOLOE_CHECKPOINT_PATH}\n"
        + hint
    )


def initialize_yoloe_model(
    weights: Optional[str] = None,
    device: str = DEFAULT_DEVICE,
    classes: Optional[List[str]] = None,
    prompt_mode: str = "text",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.50,
) -> YOLOE:
    weight_path = _resolve_weights_path(weights)
    model = YOLOE(weight_path)
    try:
        model.to(device)
    except Exception:
        pass
    model.eval()
    model.overrides = getattr(model, "overrides", {})
    model.overrides["conf"] = conf_threshold
    model.overrides["iou"] = iou_threshold

    if classes and prompt_mode == "text":
        set_yoloe_classes(model, classes)
    return model


def set_yoloe_classes(model: YOLOE, classes: List[str]) -> List[str]:
    model.set_classes(classes, model.get_text_pe(classes))
    return classes


def _ensure_hw(image: np.ndarray) -> Tuple[int, int]:
    h, w = (image.shape[0], image.shape[1]) if image.ndim >= 2 else (480, 640)
    return int(h), int(w)


def yoloe_detection(
    image: np.ndarray,
    target_classes: List[str],
    model: YOLOE,
    box_threshold: float = 0.25,
    iou_threshold: float = 0.50,
    run_extra_nms: bool = False,   # kept for API compat
    use_text_prompt: bool = True,
    retina_masks: bool = True,
) -> Detections:
    """
    Single-pass YOLOE inference returning BOXES (+ optional seg masks).
    - If `use_text_prompt=True`, sets model classes to `target_classes` each call.
    - Returns boxes even when masks are not produced/disabled.
    """
    if use_text_prompt:
        set_yoloe_classes(model, target_classes)

    results = model.predict(
        image,
        conf=box_threshold,
        iou=iou_threshold,
        retina_masks=retina_masks,
        verbose=False,
    )
    r = results[0]
    H, W = _ensure_hw(image)

    # No detections at all
    if not hasattr(r, "boxes") or r.boxes is None or len(r.boxes) == 0:
        return Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            masks=None,
            class_id=np.empty((0,), dtype=int),
            confidence=np.empty((0,), dtype=float),
            class_names=target_classes,
        )

    # Raw model classes & conf
    cls_raw = r.boxes.cls.detach().cpu().numpy().astype(int)      # (N,)
    conf = r.boxes.conf.detach().cpu().numpy().astype(float)      # (N,)
    xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32) # (N,4)

    # Map model class ids to our prompt indices
    names_map = r.names if isinstance(r.names, dict) else {i: n for i, n in enumerate(r.names or [])}
    inv = {name.lower(): i for i, name in enumerate(target_classes)}
    mapped = []
    for c in cls_raw:
        name = str(names_map.get(int(c), "")).lower()
        mapped.append(inv.get(name, -1))
    mapped = np.array(mapped, dtype=int)

    keep = mapped >= 0
    if not np.any(keep):
        return Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            masks=None,
            class_id=np.empty((0,), dtype=int),
            confidence=np.empty((0,), dtype=float),
            class_names=target_classes,
        )

    xyxy = xyxy[keep]
    conf = conf[keep]
    mapped = mapped[keep]

    # Optional masks
    masks = None
    if retina_masks and getattr(r, "masks", None) is not None and r.masks is not None and len(r.masks) > 0:
        m = r.masks.data.detach().cpu().numpy().astype(np.uint8)  # (N,h,w) {0,1}
        m = m[keep]
        # resize to (H,W) if needed
        if m.shape[-2:] != (H, W):
            m = np.stack([cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST) for mi in m])
        masks = m.astype(np.uint8)

    return Detections(
        xyxy=xyxy,                 # (N,4)
        masks=masks,               # (N,H,W) or None
        class_id=mapped,           # (N,)
        confidence=conf,           # (N,)
        class_names=target_classes,
    )

# file: cv_utils/yoloe_detector.py

from typing import Dict

def detections_to_boxes(det: Detections) -> List[Dict]:
    """
    Detections -> List[dict]로 변환.
    각 dict: {'bbox':[x1,y1,x2,y2], 'class':str, 'class_id':int, 'confidence':float, 'area':float, 'center':[x,y]}
    """
    out: List[Dict] = []
    if det.xyxy is None or len(det.xyxy) == 0:
        return out

    for i, (x1, y1, x2, y2) in enumerate(det.xyxy):
        ci = int(det.class_id[i]) if det.class_id is not None and i < len(det.class_id) else -1
        name = det.class_names[ci] if 0 <= ci < len(det.class_names) else f"id:{ci}"
        conf = float(det.confidence[i]) if det.confidence is not None and i < len(det.confidence) else None

        w = float(x2) - float(x1)
        h = float(y2) - float(y1)
        area = max(0.0, w) * max(0.0, h)
        cx = int((float(x1) + float(x2)) * 0.5)
        cy = int((float(y1) + float(y2)) * 0.5)

        out.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "class": str(name),
            "class_id": ci,
            "confidence": conf,
            "area": area,
            "center": [cx, cy],
        })
    return out


def draw_detections_bgr(image_rgb: np.ndarray,
                        det: Detections,
                        goal_idx: int = -1,
                        thickness: Optional[int] = None) -> np.ndarray:
    """
    RGB 입력에 Detections를 그려서 BGR로 반환.
    goal_idx(=prompt_classes 인덱스)가 있으면 그 클래스는 초록색, 나머지는 파란색.
    """
    vis = image_rgb.copy()
    H, W = (vis.shape[0], vis.shape[1]) if vis.ndim >= 2 else (480, 640)
    if thickness is None:
        thickness = max(1, int(round(0.002 * (H + W))))
    font = cv2.FONT_HERSHEY_SIMPLEX

    if det.xyxy is not None and len(det.xyxy) > 0:
        for i, (x1, y1, x2, y2) in enumerate(det.xyxy):
            ci = int(det.class_id[i]) if det.class_id is not None and i < len(det.class_id) else -1
            name = det.class_names[ci] if 0 <= ci < len(det.class_names) else f"id:{ci}"
            conf = float(det.confidence[i]) if det.confidence is not None and i < len(det.confidence) else None

            is_goal = (goal_idx >= 0 and ci == goal_idx)
            color_box = (0, 255, 0) if is_goal else (255, 0, 0)  # BGR

            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color_box, thickness)

            label = f"{name}" + (f" {conf:.2f}" if conf is not None else "")
            (tw, th), _ = cv2.getTextSize(label, font, 0.5, thickness)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color_box, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)


__all__ = [
    "Detections",
    "initialize_yoloe_model",
    "set_yoloe_classes",
    "yoloe_detection",
    "detections_to_boxes",       # ← 추가
    "draw_detections_bgr",       # ← 추가
]
