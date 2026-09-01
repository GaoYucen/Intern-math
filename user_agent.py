import os
from dataclasses import dataclass
from typing import Any, Dict, List

from llm_client import InternChatClient


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

REFINE_SYSTEM_PROMPT = """You are a mathematical solution auditor.
Independently verify the candidate answer. Think carefully but keep the visible
response concise. Correct it if needed. For objective-answer questions, return
only one line beginning with `FINAL_ANSWER:` followed by the actual answer. For
a proof problem, give a concise repaired proof and then the final conclusion.
Never output a placeholder.
"""


def _optional_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else None


def _optional_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else None


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for controlled experiments."""

    mode: str = "direct"  # direct | self_refine
    thinking_mode: bool = True
    temperature: float = 0.15
    max_tokens: int = 8192
    refine_temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None

    @classmethod
    def from_env(cls) -> "AgentConfig":
        thinking_raw = os.environ.get("INTERN_THINKING_MODE", "1").strip().lower()
        thinking = thinking_raw not in {"0", "false", "no", "off"}
        return cls(
            mode=os.environ.get("AGENT_MODE", "direct").strip().lower(),
            thinking_mode=thinking,
            temperature=float(os.environ.get("AGENT_TEMPERATURE", "0.15")),
            max_tokens=int(os.environ.get("AGENT_MAX_TOKENS", "8192")),
            refine_temperature=float(os.environ.get("REFINE_TEMPERATURE", "0.0")),
            top_p=_optional_float("AGENT_TOP_P"),
            top_k=_optional_int("AGENT_TOP_K"),
        )


class ReasoningAgent:
    """Competition-compatible reasoning agent for controlled calibration."""

    def __init__(
        self,
        client: InternChatClient,
        config: AgentConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or AgentConfig.from_env()
        if self.config.mode not in {"direct", "self_refine"}:
            raise ValueError(
                f"Unsupported AGENT_MODE={self.config.mode!r}; "
                "expected 'direct' or 'self_refine'."
            )

    def solve(self, problem: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        idx = metadata.get("idx", 0)
        trace: List[Dict[str, Any]] = []

        first = self._solve_once(problem)
        trace.append(
            {
                "step": "direct_solver",
                "content": {
                    "status": "completed",
                    "response_chars": len(first),
                    "thinking_mode": self.config.thinking_mode,
                },
            }
        )

        if self.config.mode == "direct":
            return {"final_response": first, "trace": trace}

        refined = self._refine(problem, first, idx)
        trace.append(
            {
                "step": "self_refine",
                "content": {
                    "status": "completed",
                    "response_chars": len(refined),
                    "thinking_mode": self.config.thinking_mode,
                },
            }
        )
        return {"final_response": refined, "trace": trace}

    def _sampling_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            kwargs["top_k"] = self.config.top_k
        return kwargs

    def _solve_once(self, problem: str) -> str:
        response = self.client.chat(
            [
                {"role": "system", "content": BASE_SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            thinking_mode=self.config.thinking_mode,
            **self._sampling_kwargs(),
        )
        return self._require_text(response)

    def _refine(self, problem: str, candidate: str, idx: Any) -> str:
        del idx
        response = self.client.chat(
            [
                {"role": "system", "content": REFINE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "PROBLEM:\n"
                        f"{problem}\n\n"
                        "CANDIDATE SOLUTION:\n"
                        f"{candidate}\n\n"
                        "Audit and, if necessary, correct the candidate."
                    ),
                },
            ],
            temperature=self.config.refine_temperature,
            max_tokens=self.config.max_tokens,
            thinking_mode=self.config.thinking_mode,
            **self._sampling_kwargs(),
        )
        return self._require_text(response)

    @staticmethod
    def _require_text(response: Any) -> str:
        if not isinstance(response, str):
            raise TypeError("Expected a text completion; tool-call responses are unsupported.")
        text = response.strip()
        if not text:
            raise ValueError("Model returned an empty response.")
        return text
