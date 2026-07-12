"""Robot backend interfaces.

The navigation agent is model/backend agnostic. Real deployment should implement
:class:`RobotBackend` for ROS2, Habitat, Isaac Sim, a custom controller, or any
S2E/PixelNav-style fast module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple
import math
import time
import uuid

from PIL import Image, ImageDraw

from .schema import Observation, ObservationView, RelativePose2D, RobotState, nearest_view_type, normalize_angle_deg, normalize_angle_rad, view_type_to_heading_deg


@dataclass
class ActionOutcome:
    """Result returned by the fast navigation skill or robot backend."""

    action: str
    success: bool
    collision: bool = False
    moved_distance_m: float = 0.0
    rotated_deg: float = 0.0
    odom_delta: RelativePose2D = field(default_factory=RelativePose2D)
    message: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def no_progress(self) -> bool:
        return (not self.success) or self.collision or self.moved_distance_m < 0.05


class RobotBackend(Protocol):
    """Protocol every real/sim backend should implement."""

    def get_robot_state(self) -> RobotState:
        ...

    def get_observation(self) -> Observation:
        ...

    def execute_waypoint(self, *, view_type: str, view_id: int, point_px: Tuple[int, int], ttl_ms: int) -> ActionOutcome:
        ...

    def rotate(self, yaw_deg: float) -> ActionOutcome:
        ...

    def capture_views(self, yaw_offsets_deg: Sequence[float], mode: str = "directed_sweep") -> Observation:
        ...


class StaticImageBackend:
    """Minimal backend that makes the framework runnable without a robot.

    It repeatedly returns the same image but simulates pose updates when a
    waypoint is executed. This is useful for smoke tests, prompt integration,
    and API debugging; it is not a physics simulator.
    """

    def __init__(
        self,
        image_path: str | Path,
        *,
        sequence_id: Optional[str] = None,
        start_xy: Tuple[float, float] = (0.0, 0.0),
        start_heading_rad: float = 0.0,
        step_m: float = 0.65,
        blocked_bearing_sectors: Optional[List[Tuple[float, float]]] = None,
    ):
        self.image_path = str(image_path)
        p = Path(self.image_path)
        if not p.exists():
            raise FileNotFoundError(f"image not found: {p}")
        with Image.open(p) as img:
            self.width, self.height = img.size
        self.sequence_id = sequence_id or f"seq_{uuid.uuid4().hex[:8]}"
        self.xy = [float(start_xy[0]), float(start_xy[1])]
        self.heading_rad = float(start_heading_rad)
        self.step_m = float(step_m)
        self.frame_index = 0
        self._last_extra_views: List[ObservationView] = []
        self.blocked_bearing_sectors = blocked_bearing_sectors or []

    @classmethod
    def create_demo_image(cls, path: str | Path, width: int = 640, height: int = 480) -> str:
        """Create a simple synthetic floor/corridor-like image."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (width, height), (210, 210, 210))
        draw = ImageDraw.Draw(img)
        # Wall/ceiling
        draw.rectangle([0, 0, width, int(height * 0.45)], fill=(160, 170, 180))
        # Floor trapezoid
        draw.polygon([(0, height), (width, height), (int(width * 0.62), int(height * 0.45)), (int(width * 0.38), int(height * 0.45))], fill=(120, 120, 120))
        # Center guide/corridor opening
        draw.rectangle([int(width * 0.43), int(height * 0.25), int(width * 0.57), int(height * 0.52)], fill=(80, 95, 105))
        draw.line([(width // 2, int(height * 0.52)), (width // 2, height)], fill=(230, 230, 230), width=3)
        img.save(p)
        return str(p)

    def _timestamp_ms(self) -> int:
        return int(time.time() * 1000)

    def get_robot_state(self) -> RobotState:
        return RobotState(map_xy=(self.xy[0], self.xy[1]), heading_rad=self.heading_rad)

    def get_observation(self) -> Observation:
        self.frame_index += 1
        ts = self._timestamp_ms()
        views = [ObservationView(view_id=0, view_type="front", relative_heading_deg=0.0, image=self.image_path, timestamp_ms=ts)]
        views.extend(self._last_extra_views)
        mode = "current_only" if not self._last_extra_views else "directed_sweep"
        self._last_extra_views = []
        return Observation(
            mode=mode,
            sequence_id=self.sequence_id,
            frame_index=self.frame_index,
            image_width=self.width,
            image_height=self.height,
            views=views,
            timestamp_ms=ts,
        )

    def _is_blocked(self, relative_bearing_deg: float) -> bool:
        b = normalize_angle_deg(relative_bearing_deg)
        for lo, hi in self.blocked_bearing_sectors:
            lo = normalize_angle_deg(lo)
            hi = normalize_angle_deg(hi)
            if lo <= hi:
                if lo <= b <= hi:
                    return True
            else:
                if b >= lo or b <= hi:
                    return True
        return False

    def execute_waypoint(self, *, view_type: str, view_id: int, point_px: Tuple[int, int], ttl_ms: int) -> ActionOutcome:
        rel_heading_deg = view_type_to_heading_deg(view_type)
        if self._is_blocked(rel_heading_deg):
            return ActionOutcome(action="go", success=False, collision=True, moved_distance_m=0.0, odom_delta=RelativePose2D(), message="simulated blocked sector")
        # Simulate turning toward selected view and moving one step.
        world_heading = normalize_angle_rad(self.heading_rad + math.radians(rel_heading_deg))
        dx = self.step_m * math.cos(world_heading)
        dy = self.step_m * math.sin(world_heading)
        self.xy[0] += dx
        self.xy[1] += dy
        self.heading_rad = world_heading
        odom = RelativePose2D(dx_m=self.step_m, dy_m=0.0, dyaw_deg=rel_heading_deg, covariance_diag=(0.04, 0.04, 4.0))
        return ActionOutcome(action="go", success=True, collision=False, moved_distance_m=self.step_m, odom_delta=odom, message="simulated waypoint success")

    def rotate(self, yaw_deg: float) -> ActionOutcome:
        yaw_deg = max(-180.0, min(180.0, float(yaw_deg)))
        self.heading_rad = normalize_angle_rad(self.heading_rad + math.radians(yaw_deg))
        return ActionOutcome(action="rotate", success=True, rotated_deg=yaw_deg, odom_delta=RelativePose2D(0.0, 0.0, yaw_deg), message="simulated rotation")

    def capture_views(self, yaw_offsets_deg: Sequence[float], mode: str = "directed_sweep") -> Observation:
        ts = self._timestamp_ms()
        views: List[ObservationView] = []
        used_types: set[str] = set()
        for i, yaw in enumerate(yaw_offsets_deg):
            vt = nearest_view_type(float(yaw))
            # v1 allowed views are semantic; avoid duplicates when offsets quantize to same view.
            if vt in used_types:
                continue
            used_types.add(vt)
            views.append(ObservationView(view_id=i + 1, view_type=vt, relative_heading_deg=float(yaw), image=self.image_path, timestamp_ms=ts))
        self._last_extra_views = views
        # Return an observation immediately for callers that want it.
        return self.get_observation()
