from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any, Mapping, Sequence, TypeVar


class TargetType(str, Enum):
    GPS = "gps"
    OBJECT_POINT = "object_point"
    LANGUAGE = "language"
    MISSING_POINT = "missing_point"


class ProgressState(str, Enum):
    NORMAL = "normal"
    LOW_PROGRESS = "low_progress"
    BLOCKED = "blocked"
    REPEATED_VIEW = "repeated_view"
    TRACKING_LOSS = "tracking_loss"


class S2EStatus(str, Enum):
    UNKNOWN = "unknown"
    SUCCESS = "success"
    FAILED = "failed"
    LOW_CONFIDENCE = "low_confidence"
    COLLISION = "collision"
    NO_VALID_TRAJECTORY = "no_valid_trajectory"


class ActionType(str, Enum):
    NAVIGATE = "NAVIGATE"
    LOOK_AROUND = "LOOK_AROUND"
    RESELECT_GOAL = "RESELECT_GOAL"
    STOP = "STOP"
    DIRECT_CONTROL = "DIRECT_CONTROL"


class ControllerAction(str, Enum):
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    MOVE_BACK = "MOVE_BACK"
    WAIT = "WAIT"
    STOP = "STOP"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


EnumT = TypeVar("EnumT", bound=Enum)


def _parse_enum(enum_type: type[EnumT], value: Any, field_name: str) -> EnumT:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{field_name} must be one of: {allowed}") from exc


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _optional_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return dict(value)


def _optional_list(value: Any, field_name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return list(value)


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a number")
    return float(value)


def _xy(value: Any, field_name: str) -> list[float]:
    if _is_string_like(value) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{field_name} must be a two-element [x, y] list")
    if not all(isinstance(component, Real) for component in value):
        raise ValueError(f"{field_name} must contain numeric values")
    return [float(value[0]), float(value[1])]


def _optional_xy(value: Any, field_name: str) -> list[float] | None:
    if value is None:
        return None
    return _xy(value, field_name)


def _optional_image_point(value: Any, field_name: str) -> list[int] | None:
    if value is None:
        return None
    if _is_string_like(value) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{field_name} must be null or a two-element [u, v] list")
    if not all(isinstance(component, int) for component in value):
        raise ValueError(f"{field_name} must contain integer pixel coordinates")
    return [int(value[0]), int(value[1])]


def _is_string_like(value: Any) -> bool:
    return isinstance(value, (str, bytes, bytearray))


@dataclass
class GoalAdapterConfig:
    default_goal_xy: list[float] = field(default_factory=lambda: [0.0, 0.0])

    @classmethod
    def from_json(cls, payload: "GoalAdapterConfig | Mapping[str, Any]") -> "GoalAdapterConfig":
        if isinstance(payload, cls):
            return payload
        payload = _require_mapping(payload, "config")
        return cls(default_goal_xy=_xy(payload.get("default_goal_xy", [0.0, 0.0]), "default_goal_xy"))

    def to_json(self) -> dict[str, Any]:
        return {"default_goal_xy": list(self.default_goal_xy)}


@dataclass
class GoalAdapterInput:
    target_type: TargetType
    high_level_target: Any = None
    current_rgb: Any = None
    optional_lookaround_images: list[Any] = field(default_factory=list)
    current_pose: dict[str, Any] = field(default_factory=dict)
    heading: float | None = None
    current_goal_xy: list[float] | None = None
    previous_waypoint: list[float] | None = None
    progress_state: ProgressState = ProgressState.NORMAL
    s2e_status: S2EStatus = S2EStatus.UNKNOWN
    memory_summary: dict[str, Any] = field(default_factory=dict)
    system_prompt: str | None = None

    @classmethod
    def from_json(cls, payload: "GoalAdapterInput | Mapping[str, Any]") -> "GoalAdapterInput":
        if isinstance(payload, cls):
            return payload
        payload = _require_mapping(payload, "input_json")
        if "target_type" not in payload:
            raise ValueError("target_type is required")

        return cls(
            target_type=_parse_enum(TargetType, payload["target_type"], "target_type"),
            high_level_target=payload.get("high_level_target"),
            current_rgb=payload.get("current_rgb"),
            optional_lookaround_images=_optional_list(
                payload.get("optional_lookaround_images", []),
                "optional_lookaround_images",
            ),
            current_pose=_optional_mapping(payload.get("current_pose", {}), "current_pose"),
            heading=_optional_float(payload.get("heading"), "heading"),
            current_goal_xy=_optional_xy(payload.get("current_goal_xy"), "current_goal_xy"),
            previous_waypoint=_optional_xy(payload.get("previous_waypoint"), "previous_waypoint"),
            progress_state=_parse_enum(
                ProgressState,
                payload.get("progress_state", ProgressState.NORMAL.value),
                "progress_state",
            ),
            s2e_status=_parse_enum(
                S2EStatus,
                payload.get("s2e_status", S2EStatus.UNKNOWN.value),
                "s2e_status",
            ),
            memory_summary=_optional_mapping(payload.get("memory_summary", {}), "memory_summary"),
            system_prompt=payload.get("system_prompt"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "target_type": self.target_type.value,
            "high_level_target": self.high_level_target,
            "current_rgb": self.current_rgb,
            "optional_lookaround_images": list(self.optional_lookaround_images),
            "current_pose": dict(self.current_pose),
            "heading": self.heading,
            "current_goal_xy": self.current_goal_xy,
            "previous_waypoint": self.previous_waypoint,
            "progress_state": self.progress_state.value,
            "s2e_status": self.s2e_status.value,
            "memory_summary": dict(self.memory_summary),
            "system_prompt": self.system_prompt,
        }


@dataclass
class GoalAdapterOutput:
    refined_goal_xy: list[float] | None
    selected_image_point: list[int] | None
    action_type: ActionType
    reasoning: str
    confidence: Confidence
    controller_action: ControllerAction | None = None

    def __post_init__(self) -> None:
        self.action_type = _parse_enum(ActionType, self.action_type, "action_type")
        if self.refined_goal_xy is not None:
            self.refined_goal_xy = _xy(self.refined_goal_xy, "refined_goal_xy")
        if self.action_type in {ActionType.NAVIGATE, ActionType.RESELECT_GOAL} and self.refined_goal_xy is None:
            raise ValueError("refined_goal_xy is required for NAVIGATE and RESELECT_GOAL")
        self.selected_image_point = _optional_image_point(
            self.selected_image_point,
            "selected_image_point",
        )
        if self.controller_action is not None:
            self.controller_action = _parse_enum(
                ControllerAction,
                self.controller_action,
                "controller_action",
            )
        if self.action_type is ActionType.DIRECT_CONTROL and self.controller_action is None:
            raise ValueError("controller_action is required for DIRECT_CONTROL")
        self.confidence = _parse_enum(Confidence, self.confidence, "confidence")
        if not isinstance(self.reasoning, str) or not self.reasoning.strip():
            raise ValueError("reasoning must be a non-empty string")

    def to_json(self) -> dict[str, Any]:
        return {
            "refined_goal_xy": list(self.refined_goal_xy) if self.refined_goal_xy is not None else None,
            "selected_image_point": self.selected_image_point,
            "action_type": self.action_type.value,
            "controller_action": self.controller_action.value if self.controller_action is not None else None,
            "reasoning": self.reasoning,
            "confidence": self.confidence.value,
        }
