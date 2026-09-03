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

    def test_hybrid_preserves_closed_primary_without_finalizer(self):
        client = FakeClient(["deep proof\nFINAL_ANSWER: 42"])
        cfg = AgentConfig(mode="hybrid", solver_a_thinking=True, solver_tokens=6144)
        out = ReasoningAgent(client, cfg).solve("6*7?", {"idx": 6})
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(out["final_response"], "deep proof\nFINAL_ANSWER: 42")
        self.assertEqual(out["trace"][-1]["content"]["status"], "primary_closed")

    def test_hybrid_normalizes_terminal_boxed_primary(self):
        client = FakeClient([r"derivation ends with \boxed{42}"])
        cfg = AgentConfig(mode="hybrid", solver_a_thinking=True)
        out = ReasoningAgent(client, cfg).solve("6*7?", {"idx": 7})
        self.assertEqual(len(client.calls), 1)
        self.assertIn(r"\boxed{42}", out["final_response"])
        self.assertTrue(out["final_response"].endswith("FINAL_ANSWER: 42"))
        self.assertEqual(out["trace"][-1]["step"], "delivery_normalization")

    def test_nonterminal_box_does_not_skip_finalizer(self):
        client = FakeClient([
            r"We obtain \boxed{6}. Now we still must multiply by 7.",
            "The remaining local step gives 42.\nFINAL_ANSWER: 42",
        ])
        cfg = AgentConfig(mode="hybrid", solver_a_thinking=True)
        out = ReasoningAgent(client, cfg).solve("6*7?", {"idx": 70})
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(out["final_response"].endswith("FINAL_ANSWER: 42"))
        self.assertEqual(out["trace"][-1]["content"]["status"], "finalizer_closed")

    def test_hybrid_finalizes_unclosed_primary(self):
        client = FakeClient([
            "We reduce the expression and obtain 42 but need to state it cleanly",
            "The prior derivation supports 42.\nFINAL_ANSWER: 42",
        ])
        cfg = AgentConfig(
            mode="hybrid",
            solver_a_thinking=True,
            solver_tokens=4096,
            finalizer_tokens=1536,
        )
        out = ReasoningAgent(client, cfg).solve("6*7?", {"idx": 8})
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(client.calls[0][1]["thinking_mode"])
        self.assertFalse(client.calls[1][1]["thinking_mode"])
        self.assertEqual(client.calls[1][1]["max_tokens"], 1536)
        self.assertIn("FINAL_ANSWER: 42", out["final_response"])
        self.assertEqual(out["trace"][-1]["content"]["status"], "finalizer_closed")

    def test_hybrid_normalizes_terminal_boxed_finalizer(self):
        client = FakeClient([
            "We reached the last step but did not state it.",
            r"Completing that step gives \boxed{42}.",
        ])
        cfg = AgentConfig(mode="hybrid", solver_a_thinking=True)
        out = ReasoningAgent(client, cfg).solve("6*7?", {"idx": 80})
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(out["final_response"].endswith("FINAL_ANSWER: 42"))
        self.assertEqual(out["trace"][-1]["step"], "delivery_normalization")

    def test_hybrid_failed_finalizer_does_not_destroy_primary(self):
        primary = "long useful derivation ending before the final marker"
        client = FakeClient([primary, RuntimeError("timeout")])
        cfg = AgentConfig(mode="hybrid", solver_a_thinking=True)
        out = ReasoningAgent(client, cfg).solve("hard problem", {"idx": 9})
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(out["final_response"], primary)
        self.assertEqual(out["trace"][-1]["content"]["status"], "finalizer_failed")

    def test_normalization_handles_boxed_and_spaces(self):
        self.assertEqual(
            ReasoningAgent._normalize_answer(" $\\boxed{42}$ "),
            ReasoningAgent._normalize_answer("42"),
        )


if __name__ == "__main__":
    unittest.main()
