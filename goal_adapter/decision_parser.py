from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any, Mapping

from goal_adapter.schema import GoalAdapterOutput


class VLMDecisionParseError(ValueError):
    """Raised when a VLM decision cannot be parsed into GoalAdapterOutput."""


def parse_vlm_decision(raw_decision: str | Mapping[str, Any]) -> GoalAdapterOutput:
    payload = _coerce_payload(raw_decision)
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
