import os
import unittest
from unittest.mock import patch

from llm_client import InternChatClient


class InternChatClientTest(unittest.TestCase):
    def test_default_retry_budget_is_one(self):
        with patch.dict(os.environ, {"INTERN_API_KEY": "test-key"}, clear=False):
            client = InternChatClient()
        self.assertEqual(client.retry, 1)

    def test_explicit_non_r1_retry_budget_remains_supported(self):
        with patch.dict(os.environ, {"INTERN_API_KEY": "test-key"}, clear=False):
            client = InternChatClient(retry=2)
        self.assertEqual(client.retry, 2)

    def test_retry_must_be_positive(self):
        with patch.dict(os.environ, {"INTERN_API_KEY": "test-key"}, clear=False):
            with self.assertRaises(ValueError):
                InternChatClient(retry=0)


if __name__ == "__main__":
    unittest.main()
