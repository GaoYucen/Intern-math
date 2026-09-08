import json
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import requests

DEFAULT_API_BASE = "https://chat.intern-ai.org.cn/api/v1/chat/completions"
DEFAULT_MODEL = "intern-s2-preview"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 8192
DEFAULT_CONNECT_TIMEOUT = 20.0
DEFAULT_READ_TIMEOUT = 360.0

ChatMessage = Dict[str, Any]
ChatResponse = Union[str, ChatMessage]
TimeoutSpec = Union[int, float, Tuple[float, float]]


class InternChatClient:
    """Small OpenAI-compatible client for the Intern challenge API.

    R1 keeps exactly one HTTP attempt so benchmark request counts remain
    interpretable. Long 397B reasoning requests use separate connect/read
    timeouts: a short connect timeout plus a longer read timeout. Response
    metadata is recorded out-of-band and never changes the returned answer.
    """

    def __init__(
        self,
        timeout: Optional[TimeoutSpec] = None,
        retry: int = 1,
        default_args: Optional[Mapping[str, Any]] = None,
        **request_args: Any,
    ) -> None:
        raw_api_key = os.environ.get("INTERN_API_KEY")
        if not raw_api_key:
            raise RuntimeError("Missing API key. Set INTERN_API_KEY.")
        if retry < 1:
            raise ValueError("retry must be >= 1")
        self.authorization = (
            raw_api_key if raw_api_key.startswith("Bearer ") else f"Bearer {raw_api_key}"
        )
        self.api_base = os.environ.get("INTERN_API_BASE", DEFAULT_API_BASE)
        self.model = os.environ.get("INTERN_MODEL", DEFAULT_MODEL)
        self.timeout = self._resolve_timeout(timeout)
        self.retry = retry
        self.default_args = dict(default_args or {})
        self.default_args.update(request_args)
        self.last_response_meta: Dict[str, Any] = {}

    @staticmethod
    def _resolve_timeout(timeout: Optional[TimeoutSpec]) -> TimeoutSpec:
        if timeout is not None:
            return timeout
        connect_timeout = float(
            os.environ.get("INTERN_CONNECT_TIMEOUT", str(DEFAULT_CONNECT_TIMEOUT))
        )
        read_timeout = float(
            os.environ.get("INTERN_READ_TIMEOUT", str(DEFAULT_READ_TIMEOUT))
        )
        if connect_timeout <= 0 or read_timeout <= 0:
            raise ValueError("INTERN_CONNECT_TIMEOUT and INTERN_READ_TIMEOUT must be positive")
        return (connect_timeout, read_timeout)

    @staticmethod
    def _serializable_timeout(timeout: TimeoutSpec) -> Any:
        if isinstance(timeout, tuple):
            return {"connect": timeout[0], "read": timeout[1]}
        return timeout

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

        self.last_response_meta = {}
        last_error: Optional[Exception] = None
        for attempt in range(self.retry):
            started = time.monotonic()
            try:
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=self.timeout,
                )
                latency = time.monotonic() - started
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                message = choice["message"]
                self.last_response_meta = {
                    "attempts_used": attempt + 1,
                    "http_status": response.status_code,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": data.get("usage"),
                    "latency_seconds": round(latency, 3),
                    "timeout_seconds": self._serializable_timeout(self.timeout),
                }
                if "tool_calls" in message:
                    return message
                return message["content"]
            except Exception as exc:  # pragma: no cover - network path
                latency = time.monotonic() - started
                last_error = exc
                self.last_response_meta = {
                    "attempts_used": attempt + 1,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "latency_seconds": round(latency, 3),
                    "timeout_seconds": self._serializable_timeout(self.timeout),
                }
                if attempt + 1 < self.retry:
                    time.sleep(2**attempt)

        raise RuntimeError(
            f"Chat completion failed after {self.retry} attempts: {last_error}"
        )
