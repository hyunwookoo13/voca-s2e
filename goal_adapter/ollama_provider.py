from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, Mapping

from goal_adapter.decision_parser import parse_vlm_decision
from goal_adapter.prompting import build_vlm_decision_prompt
from goal_adapter.schema import GoalAdapterInput, GoalAdapterOutput


HttpPost = Callable[[str, Mapping[str, Any], float], Mapping[str, Any]]


class OllamaVLMError(RuntimeError):
    """Raised when Ollama cannot return a parseable VLM decision response."""


@dataclass(frozen=True)
class OllamaVLMConfig:
    model: str = "gemma4:26b"
    endpoint: str = "http://localhost:11434/api/chat"
    timeout_seconds: float = 120.0
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 64
    num_predict: int = 512
    think: bool = False
    response_format: str = "json"


class OllamaVLMProvider:
    def __init__(
        self,
        config: OllamaVLMConfig | None = None,
        http_post: HttpPost | None = None,
    ) -> None:
        self.config = config or OllamaVLMConfig()
        self._http_post = http_post or _post_json

    def decide(self, input_json: GoalAdapterInput | Mapping[str, Any]) -> GoalAdapterOutput:
        return parse_vlm_decision(self.generate_raw_decision(input_json))

    def generate_raw_decision(self, input_json: GoalAdapterInput | Mapping[str, Any]) -> str:
        adapter_input = GoalAdapterInput.from_json(input_json)
        prompt = build_vlm_decision_prompt(adapter_input)
        response = self._http_post(
            self.config.endpoint,
            self._payload(prompt.system, prompt.user, prompt.image_paths),
            self.config.timeout_seconds,
        )
        return _response_content(response)

    def _payload(self, system_prompt: str, user_prompt: str, image_paths: list[str]) -> dict[str, Any]:
        user_message: dict[str, Any] = {
            "role": "user",
            "content": user_prompt,
        }
        images = [_encode_image(path) for path in image_paths]
        if images:
            user_message["images"] = images

        return {
            "model": self.config.model,
            "stream": False,
            "format": self.config.response_format,
            "think": self.config.think,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                user_message,
            ],
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "num_predict": self.config.num_predict,
            },
        }


def _encode_image(path: str) -> str:
    try:
        return base64.b64encode(Path(path).read_bytes()).decode("ascii")
    except OSError as exc:
        raise OllamaVLMError(f"Unable to read image for Ollama VLM input: {path}") from exc


def _response_content(response: Mapping[str, Any]) -> str:
    message = response.get("message")
    if isinstance(message, Mapping) and isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(response.get("response"), str):
        return response["response"]
    raise OllamaVLMError("Ollama response did not include message.content")


def _post_json(endpoint: str, payload: Mapping[str, Any], timeout_seconds: float) -> Mapping[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw_body = response.read().decode("utf-8")
    except TimeoutError as exc:
        raise OllamaVLMError(f"Ollama request timed out: {exc}") from exc
    except urllib.error.URLError as exc:
        raise OllamaVLMError(f"Ollama request failed: {exc}") from exc

    try:
        decoded = json.loads(raw_body)
    except JSONDecodeError as exc:
        raise OllamaVLMError("Ollama response was not valid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise OllamaVLMError("Ollama response JSON must be an object")
    return decoded
