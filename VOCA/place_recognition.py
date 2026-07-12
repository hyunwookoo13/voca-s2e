import os
from pathlib import Path
from typing import Any, Optional

import numpy as np


class DinoV2PlaceEmbedder:
    """Lazy DINOv2 place descriptor with a deterministic offline fallback."""

    def __init__(
        self,
        *,
        fallback: Any,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        local_files_only: Optional[bool] = None,
    ):
        self.fallback = fallback
        self.model_name = model_name or os.environ.get(
            "VOCA_DINOV2_MODEL", "facebook/dinov2-small"
        )
        self.device = device or os.environ.get("VOCA_PLACE_EMBEDDER_DEVICE", "cpu")
        if local_files_only is None:
            local_files_only = os.environ.get(
                "VOCA_PLACE_EMBEDDER_LOCAL_ONLY", "1"
            ).strip().lower() in {"1", "true", "yes", "on"}
        self.local_files_only = bool(local_files_only)
        self.dim = 384
        self._processor = None
        self._model = None
        self._load_error = ""
        self.backend = "dinov2_pending"

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if self._load_error:
            return False
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            self._processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            self._model = AutoModel.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            ).eval().to(self.device)
            self.dim = int(getattr(self._model.config, "hidden_size", self.dim))
            self.backend = "dinov2"
            return True
        except Exception as exc:
            self._load_error = "{}: {}".format(type(exc).__name__, str(exc))[:240]
            self.backend = "deterministic_visual_fallback"
            return False

    def embed_image(self, image: Any) -> np.ndarray:
        path = Path(image) if isinstance(image, (str, Path)) else None
        if path is not None and not path.exists():
            raise FileNotFoundError("image not found: {}".format(path))
        if not self._load():
            fallback = np.asarray(self.fallback.embed_image(image), dtype=np.float32).reshape(-1)
            descriptor = np.zeros(self.dim, dtype=np.float32)
            descriptor[: min(self.dim, fallback.size)] = fallback[: self.dim]
            norm = float(np.linalg.norm(descriptor))
            return descriptor if norm <= 1e-8 else descriptor / norm

        import torch
        from PIL import Image

        rgb = image.convert("RGB") if isinstance(image, Image.Image) else Image.open(path).convert("RGB")
        inputs = self._processor(images=rgb, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self._model(**inputs)
            tokens = output.last_hidden_state
            descriptor = tokens[:, 1:, :].mean(dim=1)[0]
            descriptor = torch.nn.functional.normalize(descriptor.float(), dim=0)
        return descriptor.detach().cpu().numpy().astype(np.float32)

    def status(self) -> dict:
        return {
            "backend": self.backend,
            "model": self.model_name,
            "device": self.device,
            "dimension": int(self.dim),
            "local_files_only": self.local_files_only,
            "load_error": self._load_error or None,
        }


def build_place_embedder(nav_modules: Any) -> Any:
    fallback = nav_modules.embedding.HashImageEmbedder(dim=256)
    backend = os.environ.get("VOCA_PLACE_EMBEDDER", "dinov2").strip().lower()
    if backend in {"off", "none", "disabled"}:
        return None
    if backend in {"hash", "deterministic", "fallback"}:
        return fallback
    return DinoV2PlaceEmbedder(fallback=fallback)
