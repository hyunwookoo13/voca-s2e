from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from goal_adapter.schema import (
    ActionType,
    ControllerAction,
    GoalAdapterInput,
)


@dataclass(frozen=True)
class VLMDecisionPrompt:
    system: str
    user: str
    image_paths: list[str]


def build_vlm_decision_prompt(input_json: GoalAdapterInput | Mapping[str, Any]) -> VLMDecisionPrompt:
    adapter_input = GoalAdapterInput.from_json(input_json)
    input_payload = json.dumps(
        adapter_input.to_json(),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )

    return VLMDecisionPrompt(
        system=_system_prompt(),
        user=(
            "Use the attached images and adapter context to choose the next "
            "single VLM decision.\n\n"
            "Adapter input JSON:\n"
            f"{input_payload}\n\n"
            "Return only the JSON object. Do not include markdown, prose, or "
            "hidden reasoning."
        ),
        image_paths=_image_paths(adapter_input),
    )


def _system_prompt() -> str:
    action_types = ", ".join(action.value for action in ActionType)
    controller_actions = ", ".join(action.value for action in ControllerAction)
    return (
        "You are the VOCA-side VLM decision module for an S2E-compatible "
        "navigation stack. Decide whether to produce a refined local goal for "
        "S2E or a controller-level recovery/stop action.\n\n"
        "Return exactly one JSON object with this schema:\n"
        "{\n"
        '  "action_type": "NAVIGATE | LOOK_AROUND | RESELECT_GOAL | STOP | DIRECT_CONTROL",\n'
        '  "refined_goal_xy": [x, y] | null,\n'
        '  "selected_image_point": [u, v] | null,\n'
        '  "controller_action": "TURN_LEFT | TURN_RIGHT | MOVE_BACK | WAIT | STOP" | null,\n'
        '  "reasoning": "brief observable reason",\n'
        '  "confidence": "high | medium | low"\n'
        "}\n\n"
        f"Allowed action_type values: {action_types}.\n"
        f"Allowed controller_action values: {controller_actions}.\n"
        "refined_goal_xy is required for NAVIGATE and RESELECT_GOAL.\n"
        "controller_action is required for DIRECT_CONTROL.\n"
        "LOOK_AROUND and STOP may set refined_goal_xy to null.\n"
        "For ObjectNav/object_point targets, choose an object-front navigable "
        "floor point, not the object's visual center.\n"
        "Use sparse memory as context: avoid repeated failed goals, account "
        "for progress_state and s2e_status, and prefer navigable candidate "
        "waypoints when they are visually and semantically consistent.\n\n"
        "Decision policy:\n"
        "- If target_type is missing_point or progress_state is tracking_loss, "
        "choose LOOK_AROUND unless a fresh reliable candidate waypoint is "
        "visible in the current input.\n"
        "- If progress_state is blocked, low_progress, or repeated_view, or "
        "s2e_status is failed, collision, low_confidence, or no_valid_trajectory, "
        "choose RESELECT_GOAL and avoid every failed_goal_xy entry.\n"
        "- If you select a candidate_waypoints item, copy its goal_xy exactly "
        "into refined_goal_xy and copy its image_point into selected_image_point "
        "when present.\n"
        "- Use exactly the schema keys shown above. Do not invent aliases such "
        "as calibrated_goal, post_point, or target_xy."
    )


def _image_paths(adapter_input: GoalAdapterInput) -> list[str]:
    paths: list[str] = []
    if isinstance(adapter_input.current_rgb, str) and adapter_input.current_rgb:
        paths.append(adapter_input.current_rgb)
    for image in adapter_input.optional_lookaround_images:
        if isinstance(image, str) and image:
            paths.append(image)
    return paths
