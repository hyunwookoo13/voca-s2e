"""Image embedding interfaces.

The default :class:`HashImageEmbedder` is deterministic and lightweight so the
framework can run immediately in tests. For real robots, replace it with a
strong place-recognition encoder such as DINOv2, SigLIP/CLIP, a Qwen image
encoder, or your internal visual descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple, Union
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image, ImageOps

ImageLike = Union[str, Path, Image.Image]


class ImageEmbedder(Protocol):
    """Protocol for pluggable image encoders."""

    dim: int

    def embed_image(self, image: ImageLike) -> np.ndarray:
        """Return a 1D float32 embedding."""
        ...


@dataclass
class HashImageEmbedder:
    """Deterministic image descriptor for smoke tests and prototyping.

    It combines a small grayscale thumbnail, RGB color statistics, and a stable
    random projection. It is **not** a good semantic place-recognition model,
    but it lets every component of the memory framework run without GPU or
    external model downloads.
    """

    dim: int = 256
    thumbnail_size: Tuple[int, int] = (32, 32)
    seed: int = 7

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        rng = np.random.default_rng(self.seed)
        # grayscale thumb + rgb means/stds + 16-bin luminance histogram
        in_dim = self.thumbnail_size[0] * self.thumbnail_size[1] + 6 + 16
        self._proj = rng.standard_normal((in_dim, self.dim), dtype=np.float32) / np.sqrt(float(in_dim))

    def _load(self, image: ImageLike) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        p = Path(image)
        if not p.exists():
            raise FileNotFoundError(f"image not found: {p}")
        return Image.open(p).convert("RGB")

    def embed_image(self, image: ImageLike) -> np.ndarray:
        img = self._load(image)
        thumb = ImageOps.fit(img.convert("L"), self.thumbnail_size, method=Image.Resampling.BILINEAR)
        g = np.asarray(thumb, dtype=np.float32).reshape(-1) / 255.0
        arr = np.asarray(img.resize((64, 64)), dtype=np.float32) / 255.0
        means = arr.reshape(-1, 3).mean(axis=0)
        stds = arr.reshape(-1, 3).std(axis=0)
        hist, _ = np.histogram(np.asarray(img.convert("L"), dtype=np.uint8), bins=16, range=(0, 255), density=True)
        feat = np.concatenate([g, means.astype(np.float32), stds.astype(np.float32), hist.astype(np.float32)], axis=0)
        emb = feat.astype(np.float32) @ self._proj
        norm = float(np.linalg.norm(emb))
        if norm <= 1e-8 or not np.isfinite(norm):
            return np.zeros(self.dim, dtype=np.float32)
        return (emb / norm).astype(np.float32)


def stable_text_hash_embedding(text: str, dim: int = 256) -> np.ndarray:
    """Small deterministic text hash embedding for semantic tags.

    This is useful for tests and indexing summaries. Replace with a proper text
    encoder for ObjNav production usage.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = [t for t in text.lower().replace("_", " ").split() if t]
    if not tokens:
        return vec
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    return vec if norm <= 1e-8 else (vec / norm).astype(np.float32)
