import unittest

from user_agent import AgentConfig, ReasoningAgent


class FakeClient:
    def __init__(self):
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "FINAL_ANSWER: 42"


class OfficialLikeClientWithoutThinkingKeyword:
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
    def test_submission_defaults_match_b0(self):
        config = AgentConfig()
        self.assertFalse(config.thinking_mode)
        self.assertEqual(config.temperature, 0.15)
        self.assertEqual(config.max_tokens, 4096)

    def test_default_agent_uses_one_call(self):
        client = FakeClient()
        agent = ReasoningAgent(client)
        out = agent.solve("6*7?", {"idx": 1})
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][1]["max_tokens"], 4096)
        self.assertEqual(client.calls[0][1]["temperature"], 0.15)
        self.assertFalse(client.calls[0][1]["thinking_mode"])
        self.assertEqual(out["final_response"], "FINAL_ANSWER: 42")
        self.assertNotIn("6*7", str(out["trace"]))

    def test_compatible_with_client_without_thinking_keyword(self):
        client = OfficialLikeClientWithoutThinkingKeyword()
        agent = ReasoningAgent(client)
        out = agent.solve("6*7?", {"idx": 1})
        self.assertEqual(out["final_response"], "FINAL_ANSWER: 42")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["temperature"], 0.15)
        self.assertEqual(client.calls[0]["max_tokens"], 4096)

    def test_constructor_tolerates_runner_extensions(self):
        client = FakeClient()
        agent = ReasoningAgent(client, None, "unused", runner_flag=True)
        out = agent.solve("6*7?", {"idx": 2})
        self.assertTrue(out["final_response"].strip())


if __name__ == "__main__":
    unittest.main()
