import json
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Union

import requests

DEFAULT_API_BASE = "https://chat.intern-ai.org.cn/api/v1/chat/completions"
DEFAULT_MODEL = "intern-s2-preview"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 8192

ChatMessage = Dict[str, Any]
ChatResponse = Union[str, ChatMessage]


class InternChatClient:
    """Small OpenAI-compatible client for the Intern challenge API.

    This intentionally stays close to the official baseline contract while
    exposing `thinking_mode`, which is useful for controlled baseline studies.
    """

    def __init__(
        self,
        timeout: int = 180,
        retry: int = 3,
        default_args: Optional[Mapping[str, Any]] = None,
        **request_args: Any,
    ) -> None:
        raw_api_key = os.environ.get("INTERN_API_KEY")
        if not raw_api_key:
            raise RuntimeError("Missing API key. Set INTERN_API_KEY.")
        self.authorization = (
            raw_api_key if raw_api_key.startswith("Bearer ") else f"Bearer {raw_api_key}"
        )
        self.api_base = os.environ.get("INTERN_API_BASE", DEFAULT_API_BASE)
        self.model = os.environ.get("INTERN_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        self.retry = retry
        self.default_args = dict(default_args or {})
        self.default_args.update(request_args)

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        *,
        thinking_mode: Optional[bool] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **request_args: Any,
    ) -> ChatResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }
        payload.update(self.default_args)
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if thinking_mode is not None:
            payload["thinking_mode"] = thinking_mode
        if tools is not None:
            payload["tools"] = tools
        payload.update(request_args)
        payload["messages"] = messages

        headers = {
            "Content-Type": "application/json",
            "Authorization": self.authorization,
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.retry):
            try:
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                message = data["choices"][0]["message"]
                if "tool_calls" in message:
                    return message
                return message["content"]
            except Exception as exc:  # pragma: no cover - network path
                last_error = exc
                if attempt + 1 < self.retry:
                    time.sleep(2**attempt)

        raise RuntimeError(
            f"Chat completion failed after {self.retry} attempts: {last_error}"
        )
