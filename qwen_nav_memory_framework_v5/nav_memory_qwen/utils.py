"""Utility helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import base64
import io
import json
import os
import re
from PIL import Image


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def image_to_data_url(image_path: str | Path, max_side: int = 1024, jpeg_quality: int = 85) -> str:
    """Convert an image file to a compact data URL."""
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {p}")
    img = Image.open(p).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def strip_large_image_values(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-copy where local image paths/data are replaced by short tags.

    This prevents accidentally dumping huge base64 images into logs.
    """
    def rec(x: Any) -> Any:
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                if k == "image" and isinstance(v, str):
                    out[k] = v if v.startswith("<") and v.endswith(">") else "<image>"
                else:
                    out[k] = rec(v)
            return out
        if isinstance(x, list):
            return [rec(v) for v in x]
        return x
    return rec(schema)


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from a model response.

    The function accepts raw JSON, fenced code blocks, and responses with short
    preambles. It does balanced-brace extraction instead of a brittle regex.
    """
    if not text or not isinstance(text, str):
        raise ValueError("empty model response")
    text = text.strip()
    # Prefer fenced JSON if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fence:
        text = fence.group(1).strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in model response")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                return json.loads(candidate)
    raise ValueError("unbalanced JSON object in model response")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
