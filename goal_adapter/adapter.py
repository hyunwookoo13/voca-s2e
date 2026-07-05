from __future__ import annotations

from typing import Any, Mapping, Protocol

from goal_adapter.baseline import refine_with_baseline_rules
from goal_adapter.safety import enforce_navigation_safety
from goal_adapter.schema import (
    GoalAdapterConfig,
    GoalAdapterInput,
    GoalAdapterOutput,
)


class DecisionProvider(Protocol):
    def decide(self, adapter_input: GoalAdapterInput) -> GoalAdapterOutput:
        ...


class GoalAdapter:
    """Schema-stable entry point for future VOCA-side goal refinement."""

    def __init__(
        self,
        config: GoalAdapterConfig | Mapping[str, Any] | None = None,
        decision_provider: DecisionProvider | None = None,
    ):
        self.config = GoalAdapterConfig.from_json(config or {})
        self.decision_provider = decision_provider

    def refine(self, input_json: GoalAdapterInput | Mapping[str, Any]) -> GoalAdapterOutput:
        adapter_input = (
            input_json
            if isinstance(input_json, GoalAdapterInput)
            else GoalAdapterInput.from_json(input_json)
        )

        if self.decision_provider is not None:
            return enforce_navigation_safety(
                adapter_input,
                self.decision_provider.decide(adapter_input),
            )

        return enforce_navigation_safety(
            adapter_input,
            refine_with_baseline_rules(adapter_input, self.config.default_goal_xy),
        )
