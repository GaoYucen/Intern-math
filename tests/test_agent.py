import unittest

from user_agent import AgentConfig, ReasoningAgent


class FakeClient:
    def __init__(self, responses=None, errors=None):
        self.calls = []
        self.model = "fake"
        self.responses = list(responses or [])
        self.errors = list(errors or [])

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.errors:
            err = self.errors.pop(0)
            if err is not None:
                raise err
        if self.responses:
            return self.responses.pop(0)
        return "FINAL_ANSWER: 42"


class AgentTest(unittest.TestCase):
    def test_completed_answer_is_never_rescued(self):
        client = FakeClient(["Reasoning.\nFINAL_ANSWER: 42"])
        agent = ReasoningAgent(client, AgentConfig(mode="rescue", thinking_mode=True))
        out = agent.solve("6*7?", {"idx": 1})
        self.assertEqual(len(client.calls), 1)
        self.assertIn("FINAL_ANSWER: 42", out["final_response"])
        self.assertNotIn("6*7", str(out["trace"]))

    def test_missing_marker_triggers_rescue(self):
        client = FakeClient([
            "I derive the answer to be 42 but forgot the marker.",
            "FINAL_ANSWER: 42",
        ])
        agent = ReasoningAgent(client, AgentConfig(mode="rescue", thinking_mode=True))
        out = agent.solve("6*7?", {"idx": 2})
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(out["final_response"], "FINAL_ANSWER: 42")
        self.assertEqual(client.calls[1][1]["max_tokens"], 4096)

    def test_unfinished_rescue_does_not_replace_first(self):
        first = "Partial useful derivation without a final marker."
        client = FakeClient([first, "Still thinking without a marker"])
        agent = ReasoningAgent(client, AgentConfig(mode="rescue", thinking_mode=True))
        out = agent.solve("hard problem", {"idx": 3})
        self.assertEqual(out["final_response"], first)

    def test_initial_api_error_gets_one_fallback_solve(self):
        client = FakeClient(
            responses=["FINAL_ANSWER: 42"],
            errors=[RuntimeError("timeout"), None],
        )
        agent = ReasoningAgent(client, AgentConfig(mode="rescue", thinking_mode=True))
        out = agent.solve("6*7?", {"idx": 4})
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(out["final_response"], "FINAL_ANSWER: 42")
        self.assertEqual(client.calls[1][1]["max_tokens"], 4096)

    def test_direct_mode_does_not_rescue_missing_marker(self):
        first = "answer is 42"
        client = FakeClient([first])
        agent = ReasoningAgent(client, AgentConfig(mode="direct", thinking_mode=True))
        out = agent.solve("6*7?", {"idx": 5})
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(out["final_response"], first)


if __name__ == "__main__":
    unittest.main()
