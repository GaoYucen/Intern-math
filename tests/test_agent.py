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
    def test_hybrid_agreement_stops_after_two_calls(self):
        client = FakeClient([
            "short derivation\nFINAL_ANSWER: 42",
            "independent derivation\nFINAL_ANSWER: 42",
        ])
        agent = ReasoningAgent(client, AgentConfig(mode="hybrid"))
        out = agent.solve("6*7?", {"idx": 1})
        self.assertEqual(len(client.calls), 2)
        self.assertIn("FINAL_ANSWER: 42", out["final_response"])
        self.assertEqual(out["trace"][-1]["step"], "agreement_gate")
        self.assertFalse(client.calls[0][1]["thinking_mode"])
        self.assertFalse(client.calls[1][1]["thinking_mode"])

    def test_hybrid_disagreement_triggers_deep_thinking(self):
        client = FakeClient([
            "FINAL_ANSWER: 41",
            "FINAL_ANSWER: 42",
            "checked carefully\nFINAL_ANSWER: 42",
        ])
        cfg = AgentConfig(mode="hybrid", deep_tokens=4096, deep_thinking=True)
        agent = ReasoningAgent(client, cfg)
        out = agent.solve("6*7?", {"idx": 2})
        self.assertEqual(len(client.calls), 3)
        self.assertIn("FINAL_ANSWER: 42", out["final_response"])
        self.assertEqual(out["trace"][-1]["step"], "deep_escalation")
        self.assertTrue(client.calls[2][1]["thinking_mode"])
        self.assertEqual(client.calls[2][1]["max_tokens"], 4096)

    def test_hybrid_one_failed_solver_escalates(self):
        client = FakeClient([
            RuntimeError("gateway timeout"),
            "FINAL_ANSWER: 41",
            "FINAL_ANSWER: 42",
        ])
        agent = ReasoningAgent(client, AgentConfig(mode="hybrid"))
        out = agent.solve("6*7?", {"idx": 3})
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(out["final_response"], "FINAL_ANSWER: 42")
        self.assertEqual(out["trace"][-1]["step"], "deep_escalation")

    def test_hybrid_unfinished_deep_uses_delivery_chooser(self):
        client = FakeClient([
            "FINAL_ANSWER: 41",
            "FINAL_ANSWER: 42",
            "long reasoning without a final marker",
            "FINAL_ANSWER: 42",
        ])
        agent = ReasoningAgent(client, AgentConfig(mode="hybrid"))
        out = agent.solve("6*7?", {"idx": 4})
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(out["final_response"], "FINAL_ANSWER: 42")
        self.assertEqual(out["trace"][-1]["step"], "chooser")

    def test_legacy_dual_disagreement_uses_short_chooser(self):
        client = FakeClient([
            "FINAL_ANSWER: 41",
            "FINAL_ANSWER: 42",
            "FINAL_ANSWER: 42",
        ])
        agent = ReasoningAgent(client, AgentConfig(mode="dual"))
        out = agent.solve("6*7?", {"idx": 5})
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(client.calls[2][1]["max_tokens"], 2048)
        self.assertFalse(client.calls[2][1]["thinking_mode"])
        self.assertIn("FINAL_ANSWER: 42", out["final_response"])

    def test_direct_a_is_one_call(self):
        client = FakeClient(["FINAL_ANSWER: 42"])
        agent = ReasoningAgent(client, AgentConfig(mode="direct_a"))
        out = agent.solve("6*7?", {"idx": 6})
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(out["final_response"], "FINAL_ANSWER: 42")

    def test_normalization_handles_boxed_and_spaces(self):
        self.assertEqual(
            ReasoningAgent._normalize_answer(" $\\boxed{42}$ "),
            ReasoningAgent._normalize_answer("42"),
        )


if __name__ == "__main__":
    unittest.main()
