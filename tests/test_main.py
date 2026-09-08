import unittest

from llm_client import ChatCompletionError
from main import solve_item


class FakeClient:
    def get_last_response_meta(self):
        return {}


class FailingAgent:
    def __init__(self):
        self.client = FakeClient()

    def solve(self, problem, metadata):
        del problem, metadata
        telemetry = {
            "semantic_request_count": 1,
            "http_attempt_count": 2,
            "retry_count": 1,
            "terminal_error_type": "ConnectionError",
            "attempts": [{"attempt": 1}, {"attempt": 2}],
        }
        raise ChatCompletionError("transport failed", telemetry)


class MainRunnerTest(unittest.TestCase):
    def test_failure_record_preserves_transport_telemetry(self):
        record = solve_item(FailingAgent(), {"idx": 7, "problem": "x?"})
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["trace"][0]["content"]["request_count"], 1)
        self.assertEqual(record["trace"][0]["content"]["http_attempt_count"], 2)
        self.assertEqual(
            record["trace"][0]["content"]["client_telemetry"]["retry_count"], 1
        )


if __name__ == "__main__":
    unittest.main()
