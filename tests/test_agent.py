import unittest

from user_agent import AgentConfig, ReasoningAgent


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class AgentTest(unittest.TestCase):
    def test_agreement_stops_after_two_calls(self):
        client = FakeClient([
            "short derivation\nFINAL_ANSWER: 42",
            "independent derivation\nFINAL_ANSWER: 42",
        ])
        agent = ReasoningAgent(client, AgentConfig())
        out = agent.solve("6*7?", {"idx": 1})
        self.assertEqual(len(client.calls), 2)
        self.assertIn("FINAL_ANSWER: 42", out["final_response"])
        self.assertEqual(out["trace"][-1]["step"], "agreement_gate")
        self.assertFalse(client.calls[0][1]["thinking_mode"])
        self.assertFalse(client.calls[1][1]["thinking_mode"])

    def test_disagreement_triggers_bounded_chooser(self):
        client = FakeClient([
            "FINAL_ANSWER: 41",
            "FINAL_ANSWER: 42",
            "A has an arithmetic error.\nFINAL_ANSWER: 42",
        ])
        agent = ReasoningAgent(client, AgentConfig())
        out = agent.solve("6*7?", {"idx": 2})
        self.assertEqual(len(client.calls), 3)
        self.assertIn("FINAL_ANSWER: 42", out["final_response"])
        self.assertEqual(client.calls[2][1]["max_tokens"], 2048)

    def test_missing_marker_uses_chooser(self):
        client = FakeClient([
            "I think it is 41",
            "FINAL_ANSWER: 42",
            "FINAL_ANSWER: 42",
        ])
        agent = ReasoningAgent(client, AgentConfig())
        out = agent.solve("6*7?", {"idx": 3})
        self.assertEqual(len(client.calls), 3)
        self.assertIn("FINAL_ANSWER: 42", out["final_response"])

    def test_one_failed_solver_preserves_other(self):
        client = FakeClient([
            RuntimeError("gateway timeout"),
            "FINAL_ANSWER: 42",
        ])
        agent = ReasoningAgent(client, AgentConfig())
        out = agent.solve("6*7?", {"idx": 4})
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(out["final_response"], "FINAL_ANSWER: 42")

    def test_direct_a_is_one_call(self):
        client = FakeClient(["FINAL_ANSWER: 42"])
        agent = ReasoningAgent(client, AgentConfig(mode="direct_a"))
        out = agent.solve("6*7?", {"idx": 5})
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(out["final_response"], "FINAL_ANSWER: 42")

    def test_normalization_handles_boxed_and_spaces(self):
        self.assertEqual(
            ReasoningAgent._normalize_answer(" $\\boxed{42}$ "),
            ReasoningAgent._normalize_answer("42"),
        )


if __name__ == "__main__":
    unittest.main()
