import base64
import json
import os
from mimetypes import guess_type
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests


def _env(name: str, default: str = "") -> str:
    return os.getenv(f"VOCA_{name}", os.getenv(name, default))


def _int_env(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


QWEN_BASE_URL = _env("QWEN_BASE_URL", "http://localhost:8000/v1").rstrip("/")
QWEN_API_KEY = _env("QWEN_API_KEY", "EMPTY")
DEFAULT_MODEL = _env("QWEN_MODEL", "qwen3-vl-32b-thinking")
TEXT_MODEL = _env("QWEN_TEXT_MODEL", DEFAULT_MODEL)
VISION_MODEL = _env("QWEN_VISION_MODEL", DEFAULT_MODEL)
TIMEOUT_S = _float_env("QWEN_TIMEOUT_S", 300.0)
MAX_TOKENS = _int_env("QWEN_MAX_TOKENS", 4096)
TEMPERATURE = _float_env("QWEN_TEMPERATURE", 0.0)
SEED = _int_env("QWEN_SEED", 20260710)
IMAGE_MAX_SIDE = _int_env("QWEN_IMAGE_MAX_SIDE", 1024)
JPEG_QUALITY = _int_env("QWEN_JPEG_QUALITY", 85)


def _extra_payload_from_env() -> Dict[str, Any]:
    raw = _env("QWEN_EXTRA_PAYLOAD_JSON", "").strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError("QWEN_EXTRA_PAYLOAD_JSON must decode to a JSON object.")
    return data


EXTRA_PAYLOAD = _extra_payload_from_env()


def _resize_if_needed(image: np.ndarray) -> np.ndarray:
    if IMAGE_MAX_SIDE <= 0:
        return image
    h, w = image.shape[:2]
    max_side = max(h, w)
    if max_side <= IMAGE_MAX_SIDE:
        return image
    scale = float(IMAGE_MAX_SIDE) / float(max_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def image_to_data_url(image_prompt) -> str:
    if isinstance(image_prompt, str):
        path = Path(image_prompt)
        mime_type, _ = guess_type(str(path))
        mime_type = mime_type or "image/jpeg"
        with open(path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    if isinstance(image_prompt, np.ndarray):
        image = _resize_if_needed(image_prompt)
        quality = max(1, min(100, JPEG_QUALITY))
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise ValueError("failed to encode image for Qwen request")
        encoded = base64.b64encode(buf).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    raise TypeError("image_prompt must be a file path or numpy.ndarray")


def _stringify_text_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts) if parts else None
    if isinstance(value, (dict, int, float, bool)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _response_text(data: Dict[str, Any]) -> str:
    candidates: List[str] = []
    choices = data.get("choices") or []
    if choices:
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else {}
        if isinstance(message, dict):
            for key in ("content", "reasoning_content", "reasoning"):
                text = _stringify_text_value(message.get(key))
                if text:
                    candidates.append(text)
        text = _stringify_text_value(first.get("text")) if isinstance(first, dict) else None
        if text:
            candidates.append(text)

    top_level_text = _stringify_text_value(data.get("output_text"))
    if top_level_text:
        candidates.append(top_level_text)

    if not candidates:
        return ""
    return candidates[0].strip()


def _chat_completion(
    messages: List[Dict[str, Any]],
    model: str,
    *,
    max_tokens: Optional[int] = None,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> str:
    url = f"{QWEN_BASE_URL}/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": int(max_tokens) if max_tokens is not None else MAX_TOKENS,
    }
    payload.update(EXTRA_PAYLOAD)
    if isinstance(extra_payload, dict):
        payload.update(extra_payload)
    payload.setdefault("seed", SEED)
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_S)
    response.raise_for_status()
    return _response_text(response.json())


def text_response(
    text_prompt,
    system_prompt="",
    *,
    max_tokens: Optional[int] = None,
    extra_payload: Optional[Dict[str, Any]] = None,
):
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text_prompt})
    return _chat_completion(
        messages,
        TEXT_MODEL,
        max_tokens=max_tokens,
        extra_payload=extra_payload,
    )


def vision_response(text_prompt, image_prompt, system_prompt=""):
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_prompt)}},
            ],
        }
    )
    return _chat_completion(messages, VISION_MODEL)
