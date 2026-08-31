from dataclasses import dataclass
from typing import Any, Dict, List


BASE_SYSTEM_PROMPT = """You are a rigorous mathematical problem-solving agent.
Solve the problem independently and prioritize correctness.

Use the model's internal reasoning to work carefully, but keep the visible response extremely concise.

Rules:
1. For multiple-choice, numeric, symbolic, short-answer, and yes/no problems, do not expose a long derivation. Return only one final line beginning with `FINAL_ANSWER:` followed by the actual answer.
2. For proof/derivation problems, give only a concise proof containing the essential argument, then one final `FINAL_ANSWER:` line with the conclusion.
3. Never spend tokens exploring abandoned approaches, repeatedly checking settled work, or narrating uncertainty.
4. Check signs, domains, assumptions, edge cases, and option labels internally before answering.
5. Do not literally copy placeholders such as `<answer>`.
6. The final line is mandatory. Examples of format only:
   FINAL_ANSWER: B
   FINAL_ANSWER: -1
   FINAL_ANSWER: x^2+1
   FINAL_ANSWER: No
The text after `FINAL_ANSWER:` must be the actual requested answer, not an explanation or meta-comment.
"""


@dataclass(frozen=True)
class AgentConfig:
    """Fixed, submission-safe configuration matching the validated B0 run."""

    thinking_mode: bool = False
    temperature: float = 0.15
    max_tokens: int = 4096


class ReasoningAgent:
    """Competition-compatible one-call mathematical reasoning agent.

    This submission intentionally matches the validated B0 configuration:
    direct solve, thinking mode off, temperature 0.15, max_tokens 4096.
    It does not depend on environment variables, hidden answers, cross-problem
    state, local absolute paths, or private client fields.
    """

    def __init__(
        self,
        client: Any,
        config: AgentConfig | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        del args, kwargs
        self.client = client
        self.config = config or AgentConfig()

    def solve(self, problem: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        del metadata
        response = self._chat(
            [
                {"role": "system", "content": BASE_SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ]
        )
        text = self._require_text(response)
        trace: List[Dict[str, Any]] = [
            {
                "step": "direct_solver",
                "content": {
                    "status": "completed",
                    "response_chars": len(text),
                    "thinking_mode": self.config.thinking_mode,
                },
            }
        ]
        return {"final_response": text, "trace": trace}

    def _chat(self, messages: List[Dict[str, Any]]) -> Any:
        try:
            return self.client.chat(
                messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                thinking_mode=self.config.thinking_mode,
            )
        except TypeError as exc:
            if "thinking_mode" not in str(exc):
                raise
            return self.client.chat(
                messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

    @staticmethod
    def _require_text(response: Any) -> str:
        if not isinstance(response, str):
            raise TypeError("Expected a text completion; tool-call responses are unsupported.")
        text = response.strip()
        if not text:
            raise ValueError("Model returned an empty response.")
        return text
