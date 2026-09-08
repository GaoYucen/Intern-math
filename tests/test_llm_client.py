import os
import unittest
from unittest.mock import patch

import requests

from llm_client import ChatCompletionError, InternChatClient


class FakeResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"HTTP {self.status_code}", response=self
            )

    def json(self):
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "FINAL_ANSWER: 42"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }


class InternChatClientTest(unittest.TestCase):
    def base_env(self):
        return {
            "INTERN_API_KEY": "test-key",
            "INTERN_CONNECT_TIMEOUT": "20",
            "INTERN_READ_TIMEOUT": "360",
            "INTERN_TRANSPORT_ATTEMPTS": "2",
            "INTERN_RETRY_BACKOFF_MIN": "0",
            "INTERN_RETRY_BACKOFF_MAX": "0",
        }

    def test_default_transport_attempt_budget_is_two(self):
        with patch.dict(os.environ, self.base_env(), clear=False):
            client = InternChatClient()
        self.assertEqual(client.retry, 2)

    def test_default_timeout_separates_connect_and_long_read(self):
        with patch.dict(os.environ, self.base_env(), clear=False):
            client = InternChatClient()
        self.assertEqual(client.timeout, (20.0, 360.0))

    def test_chat_records_metadata_without_changing_text_return(self):
        with patch.dict(os.environ, self.base_env(), clear=False):
            client = InternChatClient()
            with patch("llm_client.requests.post", return_value=FakeResponse()) as post:
                result = client.chat([{"role": "user", "content": "6*7?"}])

        self.assertEqual(result, "FINAL_ANSWER: 42")
        self.assertEqual(post.call_args.kwargs["timeout"], (20.0, 360.0))
        meta = client.get_last_response_meta()
        self.assertEqual(meta["finish_reason"], "stop")
        self.assertEqual(meta["usage"]["total_tokens"], 15)
        self.assertEqual(meta["http_attempt_count"], 1)
        self.assertEqual(meta["semantic_request_count"], 1)
        self.assertEqual(meta["retry_count"], 0)

    def test_connection_drop_is_retried_once_then_succeeds(self):
        with patch.dict(os.environ, self.base_env(), clear=False):
            client = InternChatClient()
            with patch(
                "llm_client.requests.post",
                side_effect=[requests.exceptions.ConnectionError("dropped"), FakeResponse()],
            ) as post:
                result = client.chat([{"role": "user", "content": "6*7?"}])

        self.assertEqual(result, "FINAL_ANSWER: 42")
        self.assertEqual(post.call_count, 2)
        meta = client.get_last_response_meta()
        self.assertEqual(meta["http_attempt_count"], 2)
        self.assertEqual(meta["retry_count"], 1)
        self.assertEqual(meta["attempts"][0]["status"], "error")
        self.assertTrue(meta["attempts"][0]["retryable"])
        self.assertEqual(meta["attempts"][1]["status"], "success")

    def test_retryable_503_is_retried(self):
        with patch.dict(os.environ, self.base_env(), clear=False):
            client = InternChatClient()
            with patch(
                "llm_client.requests.post",
                side_effect=[FakeResponse(503), FakeResponse(200)],
            ) as post:
                result = client.chat([{"role": "user", "content": "6*7?"}])
        self.assertEqual(result, "FINAL_ANSWER: 42")
        self.assertEqual(post.call_count, 2)

    def test_nonretryable_400_stops_after_one_attempt(self):
        with patch.dict(os.environ, self.base_env(), clear=False):
            client = InternChatClient()
            with patch("llm_client.requests.post", return_value=FakeResponse(400)) as post:
                with self.assertRaises(ChatCompletionError) as ctx:
                    client.chat([{"role": "user", "content": "6*7?"}])
        self.assertEqual(post.call_count, 1)
        self.assertEqual(ctx.exception.telemetry["http_attempt_count"], 1)
        self.assertFalse(ctx.exception.telemetry["terminal_error_retryable"])

    def test_two_connection_drops_preserve_failure_telemetry(self):
        with patch.dict(os.environ, self.base_env(), clear=False):
            client = InternChatClient()
            with patch(
                "llm_client.requests.post",
                side_effect=[
                    requests.exceptions.ConnectionError("drop-1"),
                    requests.exceptions.ConnectionError("drop-2"),
                ],
            ):
                with self.assertRaises(ChatCompletionError) as ctx:
                    client.chat([{"role": "user", "content": "6*7?"}])
        meta = ctx.exception.telemetry
        self.assertEqual(meta["http_attempt_count"], 2)
        self.assertEqual(meta["retry_count"], 1)
        self.assertEqual(len(meta["attempts"]), 2)
        self.assertTrue(meta["terminal_error_retryable"])

    def test_explicit_single_transport_attempt_remains_supported(self):
        with patch.dict(os.environ, self.base_env(), clear=False):
            client = InternChatClient(retry=1)
        self.assertEqual(client.retry, 1)

    def test_retry_must_be_positive(self):
        with patch.dict(os.environ, self.base_env(), clear=False):
            with self.assertRaises(ValueError):
                InternChatClient(retry=0)


if __name__ == "__main__":
    unittest.main()
