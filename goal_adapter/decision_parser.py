from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any, Mapping

from goal_adapter.schema import GoalAdapterOutput


class VLMDecisionParseError(ValueError):
    """Raised when a VLM decision cannot be parsed into GoalAdapterOutput."""


def parse_vlm_decision(raw_decision: str | Mapping[str, Any]) -> GoalAdapterOutput:
    payload = _normalize_payload(_coerce_payload(raw_decision))
    try:
        return GoalAdapterOutput(
            refined_goal_xy=payload.get("refined_goal_xy"),
            selected_image_point=payload.get("selected_image_point"),
            action_type=payload.get("action_type"),
            controller_action=payload.get("controller_action"),
            reasoning=payload.get("reasoning"),
            confidence=payload.get("confidence"),
        )
    except (TypeError, ValueError) as exc:
        raise VLMDecisionParseError(f"Invalid VLM decision schema: {exc}") from exc


def _coerce_payload(raw_decision: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(raw_decision, Mapping):
        return dict(raw_decision)
    if not isinstance(raw_decision, str):
        raise VLMDecisionParseError("VLM decision must be a JSON object or string")

    payload = _extract_json_object(raw_decision)
    if not isinstance(payload, dict):
        raise VLMDecisionParseError("VLM decision JSON must be an object")
    return payload


def _extract_json_object(text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise VLMDecisionParseError("No JSON object found in VLM decision text")


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    if normalized.get("refined_goal_xy") is None:
        refined_goal_xy = _first_non_null(
            normalized,
            (
                "refined_goal_calibrated_xy",
                "refined_goal_nav_xy",
                "local_goal_xy",
                "goal_xy",
            ),
        )
        if refined_goal_xy is not None:
            normalized["refined_goal_xy"] = refined_goal_xy

    if normalized.get("selected_image_point") is None:
        selected_image_point = _first_non_null(
            normalized,
            (
                "selected_image_post_point",
                "selected_action_point",
                "selected_pixel",
                "image_point",
            ),
        )
        if selected_image_point is not None:
            normalized["selected_image_point"] = selected_image_point
    return normalized


def _first_non_null(payload: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None
