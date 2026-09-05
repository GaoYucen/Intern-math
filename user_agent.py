import os
from dataclasses import dataclass
from typing import Any, Dict, List

from llm_client import InternChatClient


R1_SYSTEM_PROMPT = """You are a rigorous mathematical problem-solving agent.
Solve the problem independently and prioritize correctness.

Use the model's internal reasoning carefully, but make sure the submitted response actually contains a concrete final answer before the inference budget ends.

Rules:
1. Always finish with exactly one final line beginning with `FINAL_ANSWER:` followed by the actual requested answer or conclusion.
2. For multiple-choice, numeric, symbolic, short-answer, and yes/no problems, keep the visible response concise and put the requested answer after `FINAL_ANSWER:`.
3. For proof/derivation problems, give only the essential argument, then finish with the `FINAL_ANSWER:` line.
4. Do not repeatedly restart, explore many abandoned approaches, or continue searching after a well-justified answer has been obtained.
5. Reserve enough budget to state the final answer. If the budget is becoming tight, stop further exploration and commit to the best justified answer immediately.
6. Check signs, domains, assumptions, edge cases, and option labels before finishing.
7. Never output a placeholder such as `<answer>`.

Examples of format only:
FINAL_ANSWER: B
FINAL_ANSWER: -1
FINAL_ANSWER: x^2+1
FINAL_ANSWER: No
"""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class AgentConfig:
    """R1 Robust Baseline configuration.

    R1 is deliberately restricted to exactly one model request per problem.
    Only the inference settings below are configurable; there is no mode switch,
    finalizer, verifier, self-refine pass, retry loop, or multi-agent path.
    """

    thinking_mode: bool = True
    temperature: float = 0.0
    max_tokens: int = 8192

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            thinking_mode=_env_bool("INTERN_THINKING_MODE", True),
            temperature=float(os.environ.get("AGENT_TEMPERATURE", "0.0")),
            max_tokens=int(os.environ.get("AGENT_MAX_TOKENS", "8192")),
        )


class ReasoningAgent:
    """Competition-compatible single-call R1 baseline.

    The platform injects the official client/model. R1 never overrides the
    model and never makes a second request. The complete non-empty primary
    response is preserved as ``final_response`` so later processing cannot
    destroy evidence already produced by the solver.
    """

    def __init__(
        self,
        client: InternChatClient,
        config: AgentConfig | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        del args, kwargs
        self.client = client
        self.config = config or AgentConfig.from_env()

    def solve(self, problem: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        del metadata
        trace: List[Dict[str, Any]] = []

        response = self.client.chat(
            [
                {"role": "system", "content": R1_SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            thinking_mode=self.config.thinking_mode,
        )
        final_response = self._require_text(response)

        trace.append(
            {
                "step": "r1_single_solver",
                "content": {
                    "status": "completed",
                    "response_chars": len(final_response),
                    "thinking_mode": self.config.thinking_mode,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "request_count": 1,
                },
            }
        )
        return {"final_response": final_response, "trace": trace}

    @staticmethod
    def _require_text(response: Any) -> str:
        if not isinstance(response, str):
            raise TypeError("Expected a text completion; tool-call responses are unsupported.")
        text = response.strip()
        if not text:
            raise ValueError("Model returned an empty response.")
        return text
