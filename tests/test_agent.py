import unittest

from user_agent import AgentConfig, ReasoningAgent


class SequenceClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if not self.responses:
            raise RuntimeError("no fake response left")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


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
    def test_defaults(self):
        config = AgentConfig()
        self.assertEqual(config.mode, "adaptive_verify")
        self.assertFalse(config.thinking_mode)
        self.assertEqual(config.temperature, 0.15)
        self.assertEqual(config.second_temperature, 0.0)
        self.assertEqual(config.verifier_temperature, 0.0)
        self.assertEqual(config.max_tokens, 4096)

    def test_explicit_proof_protects_direct_answer(self):
        client = SequenceClient(["Proof.\nFINAL_ANSWER: true"])
        out = ReasoningAgent(client).solve("Prove that the statement is true.", {"idx": 1})
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(out["final_response"].splitlines()[-1], "FINAL_ANSWER: true")
        self.assertEqual(out["trace"][-1]["content"]["route"], "proof_direct")

    def test_independent_agreement_stops_after_two_calls(self):
        client = SequenceClient([
            "FINAL_ANSWER: 1/2",
            "Work.\nFINAL_ANSWER: 1/2",
        ])
        out = ReasoningAgent(client).solve("Find the value.", {"idx": 2})
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(out["final_response"], "FINAL_ANSWER: 1/2")
        self.assertEqual(out["trace"][-1]["content"]["route"], "independent_agreement")

    def test_numeric_fraction_agreement(self):
        client = SequenceClient([
            "FINAL_ANSWER: \\frac{1}{2}",
            "FINAL_ANSWER: 0.5",
        ])
        out = ReasoningAgent(client).solve("Calculate the value.", {"idx": 3})
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(out["trace"][-1]["content"]["route"], "independent_agreement")

    def test_disagreement_triggers_verifier(self):
        client = SequenceClient([
            "FINAL_ANSWER: 72",
            "FINAL_ANSWER: 75",
            "Checked independently.\nFINAL_ANSWER: 72",
        ])
        out = ReasoningAgent(client).solve("How many elements are there?", {"idx": 4})
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(out["final_response"].splitlines()[-1], "FINAL_ANSWER: 72")
        self.assertEqual(out["trace"][-1]["content"]["route"], "disagreement_verified")

    def test_second_solver_failure_falls_back_to_direct(self):
        client = SequenceClient([
            "FINAL_ANSWER: 7",
            RuntimeError("temporary failure"),
        ])
        out = ReasoningAgent(client).solve("Find x.", {"idx": 5})
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(out["final_response"], "FINAL_ANSWER: 7")
        self.assertEqual(out["trace"][-1]["content"]["status"], "fallback_to_direct")

    def test_boxed_answer_extraction(self):
        client = SequenceClient([
            "Thus \\boxed{-\\frac{1}{8}}.",
            "FINAL_ANSWER: -\\frac{1}{8}",
        ])
        out = ReasoningAgent(client).solve("Calculate the residue.", {"idx": 6})
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(out["trace"][-1]["content"]["route"], "independent_agreement")

    def test_client_without_thinking_keyword(self):
        client = OfficialLikeClientWithoutThinkingKeyword()
        out = ReasoningAgent(client).solve("6*7?", {"idx": 7})
        self.assertTrue(out["final_response"].strip())
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0]["max_tokens"], 4096)

    def test_constructor_tolerates_runner_extensions(self):
        client = SequenceClient(["FINAL_ANSWER: 42", "FINAL_ANSWER: 42"])
        agent = ReasoningAgent(client, None, "unused", runner_flag=True)
        out = agent.solve("6*7?", {"idx": 8})
        self.assertTrue(out["final_response"].strip())


if __name__ == "__main__":
    unittest.main()
