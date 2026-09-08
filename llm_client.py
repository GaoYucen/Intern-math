import json
import os
import random
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import requests

DEFAULT_API_BASE = "https://chat.intern-ai.org.cn/api/v1/chat/completions"
DEFAULT_MODEL = "intern-s2-preview"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 8192
DEFAULT_CONNECT_TIMEOUT = 20.0
DEFAULT_READ_TIMEOUT = 360.0
DEFAULT_TRANSPORT_ATTEMPTS = 2
DEFAULT_RETRY_BACKOFF_MIN = 2.0
DEFAULT_RETRY_BACKOFF_MAX = 5.0
RETRYABLE_HTTP_STATUS = {429, 502, 503, 504}

ChatMessage = Dict[str, Any]
ChatResponse = Union[str, ChatMessage]
TimeoutSpec = Union[int, float, Tuple[float, float]]


class ChatCompletionError(RuntimeError):
    """Terminal API/transport failure with per-call telemetry attached."""

    def __init__(self, message: str, telemetry: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.telemetry = dict(telemetry)


class InternChatClient:
    """Small OpenAI-compatible client for the Intern challenge API.

    R1 performs exactly one semantic solve per problem. A semantic solve may use
    one additional transport attempt only when the previous attempt produced no
    usable completion because of a retryable transport/server failure.
    """

    def __init__(
        self,
        timeout: Optional[TimeoutSpec] = None,
        retry: Optional[int] = None,
        default_args: Optional[Mapping[str, Any]] = None,
        **request_args: Any,
    ) -> None:
        raw_api_key = os.environ.get("INTERN_API_KEY")
        if not raw_api_key:
            raise RuntimeError("Missing API key. Set INTERN_API_KEY.")
        resolved_retry = (
            int(os.environ.get("INTERN_TRANSPORT_ATTEMPTS", str(DEFAULT_TRANSPORT_ATTEMPTS)))
            if retry is None
            else retry
        )
        if resolved_retry < 1:
            raise ValueError("retry/INTERN_TRANSPORT_ATTEMPTS must be >= 1")

        self.authorization = (
            raw_api_key if raw_api_key.startswith("Bearer ") else f"Bearer {raw_api_key}"
        )
        self.api_base = os.environ.get("INTERN_API_BASE", DEFAULT_API_BASE)
        self.model = os.environ.get("INTERN_MODEL", DEFAULT_MODEL)
        self.timeout = self._resolve_timeout(timeout)
        self.retry = resolved_retry
        self.default_args = dict(default_args or {})
        self.default_args.update(request_args)
        self._telemetry_local = threading.local()

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

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            return True
        if isinstance(exc, requests.exceptions.HTTPError):
            response = getattr(exc, "response", None)
            return getattr(response, "status_code", None) in RETRYABLE_HTTP_STATUS
        return False

    @staticmethod
    def _retry_backoff_seconds() -> float:
        low = float(
            os.environ.get("INTERN_RETRY_BACKOFF_MIN", str(DEFAULT_RETRY_BACKOFF_MIN))
        )
        high = float(
            os.environ.get("INTERN_RETRY_BACKOFF_MAX", str(DEFAULT_RETRY_BACKOFF_MAX))
        )
        if low < 0 or high < low:
            raise ValueError("Retry backoff must satisfy 0 <= min <= max")
        return random.uniform(low, high)

    def _set_last_response_meta(self, meta: Mapping[str, Any]) -> None:
        self._telemetry_local.last_response_meta = dict(meta)

    def get_last_response_meta(self) -> Dict[str, Any]:
        return dict(getattr(self._telemetry_local, "last_response_meta", {}))

    @property
    def last_response_meta(self) -> Dict[str, Any]:
        """Compatibility accessor backed by thread-local per-call state."""
        return self.get_last_response_meta()

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

        self._set_last_response_meta({})
        attempts: List[Dict[str, Any]] = []
        started_all = time.monotonic()
        last_error: Optional[Exception] = None

        for attempt_index in range(self.retry):
            started = time.monotonic()
            status_code: Optional[int] = None
            try:
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=self.timeout,
                )
                status_code = response.status_code
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                message = choice["message"]
                latency = time.monotonic() - started
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "status": "success",
                        "http_status": status_code,
                        "latency_seconds": round(latency, 3),
                    }
                )
                meta = {
                    "semantic_request_count": 1,
                    "http_attempt_count": attempt_index + 1,
                    "retry_count": attempt_index,
                    "attempts": attempts,
                    "http_status": status_code,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": data.get("usage"),
                    "total_latency_seconds": round(time.monotonic() - started_all, 3),
                    "timeout_seconds": self._serializable_timeout(self.timeout),
                }
                self._set_last_response_meta(meta)
                if "tool_calls" in message:
                    return message
                return message["content"]
            except Exception as exc:  # pragma: no cover - network path
                latency = time.monotonic() - started
                last_error = exc
                retryable = self._is_retryable(exc)
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "status": "error",
                        "http_status": status_code,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "retryable": retryable,
                        "latency_seconds": round(latency, 3),
                    }
                )

                should_retry = retryable and (attempt_index + 1 < self.retry)
                if should_retry:
                    backoff = self._retry_backoff_seconds()
                    attempts[-1]["retry_backoff_seconds"] = round(backoff, 3)
                    time.sleep(backoff)
                    continue

                meta = {
                    "semantic_request_count": 1,
                    "http_attempt_count": attempt_index + 1,
                    "retry_count": attempt_index,
                    "attempts": attempts,
                    "terminal_error_type": type(exc).__name__,
                    "terminal_error": str(exc),
                    "terminal_error_retryable": retryable,
                    "total_latency_seconds": round(time.monotonic() - started_all, 3),
                    "timeout_seconds": self._serializable_timeout(self.timeout),
                }
                self._set_last_response_meta(meta)
                raise ChatCompletionError(
                    f"Chat completion failed after {attempt_index + 1} transport attempts: {exc}",
                    telemetry=meta,
                ) from exc

        raise ChatCompletionError(
            f"Chat completion failed: {last_error}",
            telemetry=self.get_last_response_meta(),
        )
