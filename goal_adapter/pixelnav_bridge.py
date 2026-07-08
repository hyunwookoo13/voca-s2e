from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
import os
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PixelNavGoal:
    goal_image: np.ndarray
    goal_mask: np.ndarray
    selected_point_uv: tuple[int, int]


@dataclass(frozen=True)
class PixelNavStep:
    action: int
    overlay_image: np.ndarray


class PixelNavPolicyExecutor:
    def __init__(
        self,
        policy: object | None = None,
        pixelnav_root: str | Path = "/home/icra/Pixel-Navigator",
        checkpoint_path: str | Path = "checkpoints/navigator.pth",
        device: str | None = None,
        max_token_length: int = 64,
        image_size: int = 224,
    ):
        self._policy = policy
        self.pixelnav_root = Path(pixelnav_root)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.max_token_length = max_token_length
        self.image_size = image_size
        self.last_goal: PixelNavGoal | None = None

    @property
    def policy(self):
        if self._policy is None:
            self._policy = self._load_policy_agent()
        return self._policy

    def reset_from_point(
        self,
        goal_image: np.ndarray,
        selected_point_uv: Sequence[Real],
        radius: int = 5,
    ) -> PixelNavGoal:
        self.last_goal = make_pixelnav_goal(goal_image, selected_point_uv, radius=radius)
        self.policy.reset(self.last_goal.goal_image, self.last_goal.goal_mask)
        return self.last_goal

    def step(self, obs_rgb: np.ndarray, collide: bool = False) -> PixelNavStep:
        action, overlay_image = self.policy.step(obs_rgb, collide=collide)
        return PixelNavStep(action=int(action), overlay_image=np.asarray(overlay_image))

    def _load_policy_agent(self):
        if not self.pixelnav_root.exists():
            raise FileNotFoundError(f"Pixel-Navigator root not found: {self.pixelnav_root}")
        checkpoint = self.checkpoint_path
        if not checkpoint.is_absolute():
            checkpoint = self.pixelnav_root / checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(f"PixelNav checkpoint not found: {checkpoint}")

        root_str = str(self.pixelnav_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        from policy_agent import Policy_Agent

        return Policy_Agent(
            model_path=str(checkpoint),
            max_token_length=self.max_token_length,
            image_size=self.image_size,
            device=self.device or _default_torch_device(),
        )


def _default_torch_device() -> str:
    configured = os.getenv("PIXELNAV_DEVICE")
    if configured:
        return configured
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def clamp_pixel_point(point_uv: Sequence[Real], width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    if isinstance(point_uv, (str, bytes, bytearray)) or len(point_uv) != 2:
        raise ValueError("selected point must be a two-element [u, v] coordinate")
    if not all(isinstance(component, Real) for component in point_uv):
        raise ValueError("selected point must contain numeric coordinates")

    u = int(round(float(point_uv[0])))
    v = int(round(float(point_uv[1])))
    return max(0, min(width - 1, u)), max(0, min(height - 1, v))


def make_goal_mask(
    image_shape: Sequence[int],
    point_uv: Sequence[Real],
    radius: int = 5,
) -> np.ndarray:
    if len(image_shape) < 2:
        raise ValueError("image_shape must include height and width")
    if radius < 0:
        raise ValueError("radius must be non-negative")

    height = int(image_shape[0])
    width = int(image_shape[1])
    u, v = clamp_pixel_point(point_uv, width=width, height=height)

    mask = np.zeros((height, width), dtype=np.uint8)
    x0 = max(0, u - radius)
    x1 = min(width - 1, u + radius)
    y0 = max(0, v - radius)
    y1 = min(height - 1, v + radius)
    mask[y0 : y1 + 1, x0 : x1 + 1] = 255
    return mask


def make_pixelnav_goal(
    rgb_image: np.ndarray,
    selected_point_uv: Sequence[Real],
    radius: int = 5,
) -> PixelNavGoal:
    if not isinstance(rgb_image, np.ndarray) or rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("PixelNav goal requires an RGB image with shape (H, W, 3)")

    height, width = rgb_image.shape[:2]
    selected_uv = clamp_pixel_point(selected_point_uv, width=width, height=height)
    return PixelNavGoal(
        goal_image=np.ascontiguousarray(rgb_image.copy()),
        goal_mask=make_goal_mask((height, width), selected_uv, radius=radius),
        selected_point_uv=selected_uv,
    )
