import os
import unittest
from unittest.mock import patch

from user_agent import AgentConfig, ReasoningAgent


class FakeClient:
    def __init__(self):
        self.calls = []
        self.model = "fake"

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if len(self.calls) == 1:
            return "Reasoning.\nFINAL_ANSWER: 42"
        return "Audited.\nFINAL_ANSWER: 42"


class OfficialLikeClientWithoutThinkingKeyword:
    """Minimal client matching the guaranteed chat arguments in the official README."""

    def __init__(self):
        self.calls = []

    def chat(self, messages, temperature=None, max_tokens=None):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return "FINAL_ANSWER: 42"


class AgentTest(unittest.TestCase):
    def test_submission_defaults_are_direct_thinking_off_4096(self):
        with patch.dict(os.environ, {}, clear=True):
            config = AgentConfig.from_env()
        self.assertEqual(config.mode, "direct")
        self.assertFalse(config.thinking_mode)
        self.assertEqual(config.max_tokens, 4096)
        self.assertEqual(config.temperature, 0.15)

    def test_default_submission_agent_is_one_call(self):
        client = FakeClient()
        with patch.dict(os.environ, {}, clear=True):
            agent = ReasoningAgent(client)
        out = agent.solve("6*7?", {"idx": 1})
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][1]["max_tokens"], 4096)
        self.assertFalse(client.calls[0][1]["thinking_mode"])
        self.assertIn("FINAL_ANSWER: 42", out["final_response"])
        self.assertNotIn("6*7", str(out["trace"]))

    def test_direct_can_enable_thinking_for_local_ablation(self):
        client = FakeClient()
        agent = ReasoningAgent(client, AgentConfig(mode="direct", thinking_mode=True))
        out = agent.solve("6*7?", {"idx": 1})
        self.assertEqual(len(client.calls), 1)
        self.assertIn("FINAL_ANSWER: 42", out["final_response"])
        self.assertTrue(client.calls[0][1]["thinking_mode"])

    def test_self_refine_is_two_calls(self):
        client = FakeClient()
        agent = ReasoningAgent(client, AgentConfig(mode="self_refine", thinking_mode=False))
        out = agent.solve("6*7?", {"idx": 1})
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(out["final_response"].splitlines()[-1], "FINAL_ANSWER: 42")

    def test_compatible_with_client_without_thinking_mode_keyword(self):
        client = OfficialLikeClientWithoutThinkingKeyword()
        agent = ReasoningAgent(client, AgentConfig())
        out = agent.solve("6*7?", {"idx": 1}, ignored_by_runner=True) if False else agent.solve("6*7?", {"idx": 1})
        self.assertEqual(out["final_response"], "FINAL_ANSWER: 42")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["max_tokens"], 4096)

    def test_constructor_tolerates_extra_runner_arguments(self):
        client = FakeClient()
        agent = ReasoningAgent(client, None, "unused", runner_flag=True)
        out = agent.solve("6*7?", {"idx": 2})
        self.assertTrue(out["final_response"].strip())


if __name__ == "__main__":
    unittest.main()
