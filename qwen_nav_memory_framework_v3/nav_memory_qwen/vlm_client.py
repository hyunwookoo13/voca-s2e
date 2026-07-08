"""VLM client adapters.

``OpenAICompatibleVLMClient`` works with services that expose an OpenAI-style
``/chat/completions`` endpoint and support image URL content. For local vLLM or
other runtimes, subclass :class:`BaseVLMClient` and return a JSON dictionary in
``nav_vlm_waypoint_v1`` format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import os

import requests

from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .schema import (
    make_go_output,
    make_observation_request_output,
    make_rotate_output,
    make_stop_output,
    nearest_view_type,
    normalize_angle_deg,
    view_type_to_heading_deg,
)
from .utils import extract_json_object, image_to_data_url, strip_large_image_values


class BaseVLMClient:
    """Abstract VLM decision client."""

    def decide(self, vlm_input: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class OpenAICompatibleVLMClient(BaseVLMClient):
    """Qwen/OpenAI-compatible multimodal chat-completions adapter.

    Environment variables used by :meth:`from_env`:
        QWEN_BASE_URL: e.g. https://your-endpoint/v1
        QWEN_API_KEY: bearer token
        QWEN_MODEL: model name, e.g. qwen3-vl-32b-thinking

    ``extra_payload`` can be used for runtime-specific fields such as thinking
    mode flags, response format, or sampling controls.
    """

    base_url: str
    api_key: str
    model: str = "qwen3-vl-32b-thinking"
    timeout_s: float = 120.0
    temperature: float = 0.0
    max_tokens: int = 2048
    image_max_side: int = 1024
    jpeg_quality: int = 85
    extra_payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "OpenAICompatibleVLMClient":
        base_url = os.getenv("QWEN_BASE_URL", "").rstrip("/")
        api_key = os.getenv("QWEN_API_KEY", "")
        model = os.getenv("QWEN_MODEL", "qwen3-vl-32b-thinking")
        if not base_url or not api_key:
            raise RuntimeError("Set QWEN_BASE_URL and QWEN_API_KEY before using OpenAICompatibleVLMClient.from_env().")
        return cls(base_url=base_url, api_key=api_key, model=model)

    def _collect_image_refs(self, vlm_input: Dict[str, Any]) -> List[Tuple[str, str]]:
        refs: List[Tuple[str, str]] = []
        for v in vlm_input.get("observation", {}).get("views", []) or []:
            image = v.get("image")
            if isinstance(image, str) and image and not image.startswith("<") and Path(image).exists():
                refs.append((f"observation view_id={v.get('view_id')} view_type={v.get('view_type')}", image))
        for mi in vlm_input.get("memory", {}).get("retrieved_memory_images", []) or []:
            image = mi.get("image_ref")
            if isinstance(image, str) and image and Path(image).exists():
                refs.append((f"memory node_id={mi.get('node_id')} keyframe_id={mi.get('keyframe_id')} reason={mi.get('reason_for_inclusion')}", image))
        mem = vlm_input.get("memory", {})
        revisit_candidates = list(mem.get("place_recognition", {}).get("revisit_candidates", []) or [])
        revisit_candidates.extend(list(mem.get("revisit_candidates", []) or []))  # backward-compatible v3 context
        for cand in revisit_candidates:
            image = cand.get("candidate_image_ref")
            if isinstance(image, str) and image and Path(image).exists():
                refs.append((f"revisit_candidate node_id={cand.get('candidate_node_id')} score={cand.get('visual_retrieval_score')}", image))
        return refs

    def decide(self, vlm_input: Dict[str, Any]) -> Dict[str, Any]:
        safe_json = strip_large_image_values(vlm_input)
        prompt = USER_PROMPT_TEMPLATE.format(vlm_input_json=json.dumps(safe_json, ensure_ascii=False, indent=2))
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for label, path in self._collect_image_refs(vlm_input):
            content.append({"type": "text", "text": f"Attached image: {label}"})
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path, max_side=self.image_max_side, jpeg_quality=self.jpeg_quality)}})

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        payload.update(self.extra_payload)
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        return extract_json_object(text)


@dataclass
class HeuristicVLMClient(BaseVLMClient):
    """Deterministic rule-based VLM substitute for tests and integration.

    It does not understand images. It uses goal bearing and graph memory context
    to produce schema-valid actions. Replace with Qwen for real deployment.
    """

    stop_distance_m: float = 0.4
    request_sweep_if_goal_outside_view: bool = True

    def decide(self, vlm_input: Dict[str, Any]) -> Dict[str, Any]:
        task = vlm_input.get("task", {})
        coarse = task.get("coarse_goal", {})
        goal_bearing = float(coarse.get("relative_bearing_deg", 0.0) or 0.0)
        goal_dist = float(coarse.get("distance_m", 999.0) or 999.0)
        obs = vlm_input.get("observation", {})
        width = int(obs.get("image_width", 640) or 640)
        height = int(obs.get("image_height", 480) or 480)
        views = obs.get("views", []) or [{"view_id": 0, "view_type": "front", "relative_heading_deg": 0}]
        available_by_type = {v.get("view_type"): v for v in views}
        memory = vlm_input.get("memory", {})
        memory_ops: List[Dict[str, Any]] = []
        revisit_candidates = list(memory.get("place_recognition", {}).get("revisit_candidates", []) or [])
        revisit_candidates.extend(list(memory.get("revisit_candidates", []) or []))  # backward-compatible v3 context
        for cand in revisit_candidates:
            try:
                score = float(cand.get("visual_retrieval_score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            # This is only a smoke-test stand-in for Qwen's image comparison.
            # The real VLM should emit this op after comparing the current image
            # with the candidate memory image.
            if score >= 0.92 and cand.get("candidate_node_id"):
                memory_ops.append({
                    "op": "confirm_revisit_node",
                    "node_id": cand.get("candidate_node_id"),
                    "confidence": min(0.99, score),
                    "reason": "heuristic_high_score_revisit_candidate",
                })
                break

        def with_memory_ops(output: Dict[str, Any]) -> Dict[str, Any]:
            if memory_ops:
                output = dict(output)
                output["memory_ops"] = list(memory_ops)
            return output

        if goal_dist <= self.stop_distance_m:
            return with_memory_ops(make_stop_output(reason="S02_TARGET_REACHED_OR_TASK_DONE"))

        if memory.get("loop_warning", {}).get("is_looping"):
            # A full sweep is safer than repeatedly picking the same branch.
            return with_memory_ops(make_observation_request_output(
                mode="full_sweep",
                center_yaw_deg=0,
                step_deg=45,
                num_views=8,
                yaw_offsets_deg=[-180, -135, -90, -45, 0, 45, 90, 135],
                reason="loop_or_deadlock_warning_need_full_sweep",
                confidence="medium",
            ))

        # Prefer backend-ranked memory candidates that are not negative.
        candidates = memory.get("local_topology", {}).get("candidate_exits", []) or []
        safe_candidates = [c for c in candidates if not c.get("avoid")]
        safe_candidates.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)
        for cand in safe_candidates:
            vt = cand.get("view_type_hint") or nearest_view_type(float(cand.get("bearing_deg_robot", goal_bearing)))
            if vt in available_by_type:
                v = available_by_type[vt]
                u = width // 2
                px_v = int(height * 0.75)
                goal_reason = "F07_MEMORY_AVOID_FAILED_BRANCH" if any(c.get("avoid") for c in candidates) else "F02_VISIBLE_FLOOR_TOWARD_GOAL"
                decision_reason = "G03_DOORWAY_OR_CORRIDOR_TOWARD_GOAL" if cand.get("status") != "unknown_frontier" else "G01_GOAL_ALIGNED_VIEW"
                return with_memory_ops(make_go_output(
                    view_id=int(v.get("view_id", 0)),
                    view_type=str(vt),
                    point_px=(u, px_v),
                    width=width,
                    height=height,
                    decision_reason=decision_reason,
                    goal_reason=goal_reason,
                    short_text=f"choose {vt} candidate; memory status={cand.get('status')}",
                    confidence="medium",
                ))

        # If the goal direction is not covered by current views, ask for a directed sweep.
        desired_view = nearest_view_type(goal_bearing)
        if self.request_sweep_if_goal_outside_view and desired_view not in available_by_type:
            return with_memory_ops(make_observation_request_output(
                mode="directed_sweep",
                center_yaw_deg=goal_bearing,
                step_deg=45,
                num_views=3,
                yaw_offsets_deg=[normalize_angle_deg(goal_bearing - 45), goal_bearing, normalize_angle_deg(goal_bearing + 45)],
                reason="goal_bearing_not_covered_by_current_views",
                confidence="medium",
            ))

        # Default: if a view near goal exists, go; otherwise rotate toward goal.
        if desired_view in available_by_type:
            v = available_by_type[desired_view]
            return with_memory_ops(make_go_output(
                view_id=int(v.get("view_id", 0)),
                view_type=desired_view,
                point_px=(width // 2, int(height * 0.75)),
                width=width,
                height=height,
                decision_reason="G02_VISIBLE_FLOOR_TOWARD_GOAL",
                goal_reason="F02_VISIBLE_FLOOR_TOWARD_GOAL",
                short_text="heuristic selects bottom-center navigable floor near goal bearing",
                confidence="medium",
            ))
        return with_memory_ops(make_rotate_output(max(-90.0, min(90.0, goal_bearing)), reason="R01_GOAL_OUTSIDE_CURRENT_VIEW", confidence="medium"))
