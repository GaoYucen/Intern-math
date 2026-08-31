import unittest

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
    def test_submission_defaults_are_self_refine_thinking_off_4096(self):
        config = AgentConfig()
        self.assertEqual(config.mode, "self_refine")
        self.assertFalse(config.thinking_mode)
        self.assertEqual(config.max_tokens, 4096)
        self.assertEqual(config.temperature, 0.15)
        self.assertEqual(config.refine_temperature, 0.0)

    def test_default_submission_agent_is_two_calls(self):
        client = FakeClient()
        agent = ReasoningAgent(client)
        out = agent.solve("6*7?", {"idx": 1})
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0][1]["max_tokens"], 4096)
        self.assertEqual(client.calls[1][1]["max_tokens"], 4096)
        self.assertFalse(client.calls[0][1]["thinking_mode"])
        self.assertFalse(client.calls[1][1]["thinking_mode"])
        self.assertEqual(out["final_response"].splitlines()[-1], "FINAL_ANSWER: 42")
        self.assertNotIn("6*7", str(out["trace"]))

    def test_direct_mode_remains_available_for_ablation(self):
        client = FakeClient()
        agent = ReasoningAgent(client, AgentConfig(mode="direct", thinking_mode=False))
        out = agent.solve("6*7?", {"idx": 1})
        self.assertEqual(len(client.calls), 1)
        self.assertIn("FINAL_ANSWER: 42", out["final_response"])

    def test_compatible_with_client_without_thinking_mode_keyword(self):
        client = OfficialLikeClientWithoutThinkingKeyword()
        agent = ReasoningAgent(client)
        out = agent.solve("6*7?", {"idx": 1})
        self.assertEqual(out["final_response"], "FINAL_ANSWER: 42")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0]["max_tokens"], 4096)
        self.assertEqual(client.calls[1]["max_tokens"], 4096)

    def test_constructor_tolerates_extra_runner_arguments(self):
        client = FakeClient()
        agent = ReasoningAgent(client, None, "unused", runner_flag=True)
        out = agent.solve("6*7?", {"idx": 2})
        self.assertTrue(out["final_response"].strip())


if __name__ == "__main__":
    unittest.main()
