import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from navigation_supervisor import build_policy_coarse_goal


def _normalize_angle_deg(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


def _scene_key(value: Any) -> str:
    return Path(str(value or "")).stem.replace(".basis", "")


class CoarseGoalProvider:
    """Resolve externally supplied S2E/user coarse goals for benchmark episodes."""

    schema_version = "voca_coarse_goal_dataset_v1"

    def __init__(self, records: Sequence[Dict[str, Any]], *, source_path: str = ""):
        self.source_path = str(source_path or "")
        self.source_sha256 = ""
        if self.source_path:
            payload = Path(self.source_path).read_bytes()
            self.source_sha256 = hashlib.sha256(payload).hexdigest()
        self.records = [self._validate_record(item) for item in records]
        self._by_episode_index: Dict[int, Dict[str, Any]] = {}
        self._by_episode_id: Dict[str, List[Dict[str, Any]]] = {}
        for record in self.records:
            if record.get("episode_index") is not None:
                index = int(record["episode_index"])
                if index in self._by_episode_index:
                    raise ValueError("duplicate coarse goal episode_index: {}".format(index))
                self._by_episode_index[index] = record
            episode_id = str(record.get("episode_id") or "").strip()
            if episode_id:
                self._by_episode_id.setdefault(episode_id, []).append(record)

    @classmethod
    def from_path(cls, path: str) -> "CoarseGoalProvider":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError("coarse goal file not found: {}".format(source))
        if source.suffix.lower() == ".jsonl":
            records = [
                json.loads(line)
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            payload = json.loads(source.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                schema = str(payload.get("schema_version") or "")
                if schema and schema != cls.schema_version:
                    raise ValueError("unsupported coarse goal schema: {}".format(schema))
                records = payload.get("records")
            else:
                records = payload
            if not isinstance(records, list):
                raise ValueError("coarse goal JSON must contain a records list")
        return cls(records, source_path=str(source))

    @staticmethod
    def _validate_record(value: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("coarse goal record must be an object")
        record = dict(value)
        if record.get("episode_index") is None and not str(
            record.get("episode_id") or ""
        ).strip():
            raise ValueError("coarse goal record needs episode_index or episode_id")
        if record.get("episode_index") is not None:
            record["episode_index"] = int(record["episode_index"])
            if record["episode_index"] < 0:
                raise ValueError("coarse goal episode_index must be non-negative")
        map_xy = record.get("map_xy")
        if not isinstance(map_xy, (list, tuple)) or len(map_xy) != 2:
            raise ValueError("coarse goal record requires map_xy [world_x, world_z]")
        record["map_xy"] = [float(map_xy[0]), float(map_xy[1])]
        record.setdefault("type", "map_waypoint")
        record.setdefault("uncertainty", "medium")
        # Reuse the policy contract to reject oracle/evaluation provenance.
        build_policy_coarse_goal(
            {
                "source": record.get("source"),
                "type": record.get("type"),
                "map_xy": record["map_xy"],
                "uncertainty": record.get("uncertainty"),
            }
        )
        return record

    def resolve(self, *, episode_index: int, episode: Any) -> Dict[str, Any]:
        episode_id = str(getattr(episode, "episode_id", "") or "").strip()
        scene = _scene_key(getattr(episode, "scene_id", ""))
        object_goal = str(getattr(episode, "object_category", "") or "").strip()
        candidates = list(self._by_episode_id.get(episode_id, []))
        if not candidates and int(episode_index) in self._by_episode_index:
            candidates = [self._by_episode_index[int(episode_index)]]
        filtered = [
            item
            for item in candidates
            if not item.get("scene_id") or _scene_key(item.get("scene_id")) == scene
        ]
        if len(filtered) != 1:
            raise KeyError(
                "coarse goal resolution failed for episode_index={} episode_id={} scene={}".format(
                    int(episode_index),
                    episode_id,
                    scene,
                )
            )
        record = dict(filtered[0])
        expected_object = str(record.get("object_goal") or "").strip()
        if expected_object and expected_object != object_goal:
            raise ValueError(
                "coarse goal object mismatch: expected={} actual={}".format(
                    expected_object,
                    object_goal,
                )
            )
        record["resolved_episode_index"] = int(episode_index)
        record["resolved_episode_id"] = episode_id
        record["resolved_scene_id"] = scene
        record["resolved_object_goal"] = object_goal
        record["provider_path"] = self.source_path
        record["provider_sha256"] = self.source_sha256
        return record

    @staticmethod
    def runtime_goal(
        record: Dict[str, Any],
        *,
        position_xyz: Sequence[float],
        heading_rad: float,
    ) -> Dict[str, Any]:
        if len(position_xyz) < 3:
            raise ValueError("runtime coarse goal requires position_xyz")
        goal_xy = record.get("map_xy") or []
        if len(goal_xy) != 2:
            raise ValueError("resolved coarse goal is missing map_xy")
        dx_world = float(goal_xy[0]) - float(position_xyz[0])
        dz_world = float(goal_xy[1]) - float(position_xyz[2])
        heading = float(heading_rad)
        c = math.cos(heading)
        s = math.sin(heading)
        dx_robot = c * dx_world + s * dz_world
        dy_robot = -s * dx_world + c * dz_world
        distance_m = math.hypot(dx_world, dz_world)
        bearing_deg = _normalize_angle_deg(
            math.degrees(math.atan2(dy_robot, dx_robot))
        )
        uncertainty = str(record.get("uncertainty") or "medium").lower()
        default_padding = {"low": 0.5, "medium": 1.5, "high": 3.0}[uncertainty]
        padding_m = max(
            0.0,
            float(record.get("distance_uncertainty_m", default_padding)),
        )
        output = {
            "source": str(record.get("source") or ""),
            "type": str(record.get("type") or "map_waypoint"),
            "map_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "relative_bearing_deg": round(float(bearing_deg), 3),
            "distance_m": round(float(distance_m), 4),
            "distance_range_m": [
                round(max(0.0, distance_m - padding_m), 4),
                round(distance_m + padding_m, 4),
            ],
            "uncertainty": uncertainty,
        }
        build_policy_coarse_goal(output)
        return output
