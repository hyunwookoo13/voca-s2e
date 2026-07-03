from __future__ import annotations

from typing import Any, Mapping

from goal_adapter.baseline import refine_with_baseline_rules
from goal_adapter.schema import (
    GoalAdapterConfig,
    GoalAdapterInput,
    GoalAdapterOutput,
)


class GoalAdapter:
    """Schema-stable entry point for future VOCA-side goal refinement."""

    def __init__(self, config: GoalAdapterConfig | Mapping[str, Any] | None = None):
        self.config = GoalAdapterConfig.from_json(config or {})

    def refine(self, input_json: GoalAdapterInput | Mapping[str, Any]) -> GoalAdapterOutput:
        adapter_input = (
            input_json
            if isinstance(input_json, GoalAdapterInput)
            else GoalAdapterInput.from_json(input_json)
        )

        return refine_with_baseline_rules(adapter_input, self.config.default_goal_xy)
