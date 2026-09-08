import os
import unittest
from unittest.mock import patch

from user_agent import AgentConfig, ReasoningAgent


class FakeClient:
    def __init__(self, response="Reasoning.\nFINAL_ANSWER: 42", telemetry=None):
        self.calls = []
        self.model = "fake"
        self.response = response
        self.last_response_meta = dict(telemetry or {})

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.response


class AgentTest(unittest.TestCase):
    def test_r1_is_exactly_one_call(self):
        client = FakeClient()
        agent = ReasoningAgent(client)
        out = agent.solve("6*7?", {"idx": 1})

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(out["final_response"], "Reasoning.\nFINAL_ANSWER: 42")
        self.assertNotIn("6*7", str(out["trace"]))
        self.assertEqual(out["trace"][0]["content"]["request_count"], 1)

    def test_r1_defaults_match_frozen_baseline(self):
        client = FakeClient()
        agent = ReasoningAgent(client)
        agent.solve("6*7?", {})

        kwargs = client.calls[0][1]
        self.assertEqual(agent.config.mode, "direct")
        self.assertTrue(kwargs["thinking_mode"])
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertEqual(kwargs["max_tokens"], 8192)

    def test_environment_only_controls_inference_settings(self):
        env = {
            "INTERN_THINKING_MODE": "0",
            "AGENT_TEMPERATURE": "0.2",
            "AGENT_MAX_TOKENS": "4096",
            "AGENT_MODE": "self_refine",  # must have no effect in R1
        }
        with patch.dict(os.environ, env, clear=False):
            client = FakeClient()
            agent = ReasoningAgent(client)
            agent.solve("6*7?", {})

        self.assertEqual(agent.config.mode, "direct")
        self.assertEqual(len(client.calls), 1)
        kwargs = client.calls[0][1]
        self.assertFalse(kwargs["thinking_mode"])
        self.assertEqual(kwargs["temperature"], 0.2)
        self.assertEqual(kwargs["max_tokens"], 4096)

    def test_primary_response_is_preserved(self):
        response = "work in progress but useful evidence\n\\boxed{42}"
        client = FakeClient(response=response)
        out = ReasoningAgent(client).solve("6*7?", {})
        self.assertEqual(out["final_response"], response)
        self.assertEqual(len(client.calls), 1)

    def test_client_telemetry_is_side_channel_only(self):
        telemetry = {
            "finish_reason": "stop",
            "usage": {"total_tokens": 123},
            "latency_seconds": 12.5,
            "attempts_used": 1,
        }
        client = FakeClient(telemetry=telemetry)
        out = ReasoningAgent(client).solve("6*7?", {})

        self.assertEqual(out["final_response"], "Reasoning.\nFINAL_ANSWER: 42")
        self.assertEqual(len(client.calls), 1)
        trace = out["trace"][0]["content"]
        self.assertEqual(trace["request_count"], 1)
        self.assertEqual(trace["client_telemetry"], telemetry)

    def test_empty_response_fails_without_retry(self):
        client = FakeClient(response="   ")
        with self.assertRaises(ValueError):
            ReasoningAgent(client).solve("6*7?", {})
        self.assertEqual(len(client.calls), 1)

    def test_non_text_response_fails_without_retry(self):
        client = FakeClient(response={"answer": 42})
        with self.assertRaises(TypeError):
            ReasoningAgent(client).solve("6*7?", {})
        self.assertEqual(len(client.calls), 1)


if __name__ == "__main__":
    unittest.main()
