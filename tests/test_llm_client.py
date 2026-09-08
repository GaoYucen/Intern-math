import os
import unittest
from unittest.mock import patch

from llm_client import InternChatClient


class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

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
    def test_default_retry_budget_is_one(self):
        with patch.dict(os.environ, {"INTERN_API_KEY": "test-key"}, clear=False):
            client = InternChatClient()
        self.assertEqual(client.retry, 1)

    def test_default_timeout_separates_connect_and_long_read(self):
        env = {
            "INTERN_API_KEY": "test-key",
            "INTERN_CONNECT_TIMEOUT": "20",
            "INTERN_READ_TIMEOUT": "360",
        }
        with patch.dict(os.environ, env, clear=False):
            client = InternChatClient()
        self.assertEqual(client.timeout, (20.0, 360.0))

    def test_timeout_env_override(self):
        env = {
            "INTERN_API_KEY": "test-key",
            "INTERN_CONNECT_TIMEOUT": "7.5",
            "INTERN_READ_TIMEOUT": "420",
        }
        with patch.dict(os.environ, env, clear=False):
            client = InternChatClient()
        self.assertEqual(client.timeout, (7.5, 420.0))

    def test_chat_records_metadata_without_changing_text_return(self):
        env = {
            "INTERN_API_KEY": "test-key",
            "INTERN_CONNECT_TIMEOUT": "20",
            "INTERN_READ_TIMEOUT": "360",
        }
        with patch.dict(os.environ, env, clear=False):
            client = InternChatClient()
            with patch("llm_client.requests.post", return_value=FakeResponse()) as post:
                result = client.chat([{"role": "user", "content": "6*7?"}])

        self.assertEqual(result, "FINAL_ANSWER: 42")
        self.assertEqual(post.call_args.kwargs["timeout"], (20.0, 360.0))
        self.assertEqual(client.last_response_meta["finish_reason"], "stop")
        self.assertEqual(client.last_response_meta["usage"]["total_tokens"], 15)
        self.assertEqual(client.last_response_meta["attempts_used"], 1)
        self.assertEqual(client.last_response_meta["timeout_seconds"]["read"], 360.0)
        self.assertIn("latency_seconds", client.last_response_meta)

    def test_explicit_non_r1_retry_budget_remains_supported(self):
        with patch.dict(os.environ, {"INTERN_API_KEY": "test-key"}, clear=False):
            client = InternChatClient(retry=2)
        self.assertEqual(client.retry, 2)

    def test_retry_must_be_positive(self):
        with patch.dict(os.environ, {"INTERN_API_KEY": "test-key"}, clear=False):
            with self.assertRaises(ValueError):
                InternChatClient(retry=0)

    def test_default_timeout_must_be_positive(self):
        env = {
            "INTERN_API_KEY": "test-key",
            "INTERN_CONNECT_TIMEOUT": "20",
            "INTERN_READ_TIMEOUT": "0",
        }
        with patch.dict(os.environ, env, clear=False):
            with self.assertRaises(ValueError):
                InternChatClient()


if __name__ == "__main__":
    unittest.main()
