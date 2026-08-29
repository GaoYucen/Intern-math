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


class AgentTest(unittest.TestCase):
    def test_direct_is_one_call(self):
        client = FakeClient()
        agent = ReasoningAgent(client, AgentConfig(mode="direct", thinking_mode=True))
        out = agent.solve("6*7?", {"idx": 1})
        self.assertEqual(len(client.calls), 1)
        self.assertIn("FINAL_ANSWER: 42", out["final_response"])
        self.assertNotIn("6*7", str(out["trace"]))
        self.assertTrue(client.calls[0][1]["thinking_mode"])

    def test_self_refine_is_two_calls(self):
        client = FakeClient()
        agent = ReasoningAgent(client, AgentConfig(mode="self_refine", thinking_mode=True))
        out = agent.solve("6*7?", {"idx": 1})
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(out["final_response"].splitlines()[-1], "FINAL_ANSWER: 42")


if __name__ == "__main__":
    unittest.main()
