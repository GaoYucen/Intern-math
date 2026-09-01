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

RESCUE_SYSTEM_PROMPT = """You are a mathematical completion agent.
A strong solver already attempted the problem but did not produce the required final marker.
Preserve any mathematically correct work already present. Do not restart from scratch unless the prior work is unusable.
If the prior work already contains the answer, simply extract and format it. If it is incomplete, finish only the missing mathematical steps.
Be decisive and concise. End with exactly one line beginning `FINAL_ANSWER:` followed by the actual requested answer.
For proof problems, include only the minimal missing argument before that final line.
"""

FALLBACK_SYSTEM_PROMPT = """Solve this mathematical problem carefully but efficiently.
Prioritize reaching a final answer within the remaining budget. Avoid exploratory digressions.
End with exactly one line beginning `FINAL_ANSWER:` followed by the actual requested answer.
"""


def _optional_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else None


def _optional_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else None


@dataclass(frozen=True)
class AgentConfig:
    """Submission configuration with conditional rescue only on observable failure."""

    mode: str = "rescue"  # rescue | direct
    thinking_mode: bool = True
    temperature: float = 0.15
    max_tokens: int = 8192
    rescue_max_tokens: int = 4096
    rescue_temperature: float = 0.15
    top_p: float | None = None
    top_k: int | None = None

    @classmethod
    def from_env(cls) -> "AgentConfig":
        thinking_raw = os.environ.get("INTERN_THINKING_MODE", "1").strip().lower()
        thinking = thinking_raw not in {"0", "false", "no", "off"}
        return cls(
            mode=os.environ.get("AGENT_MODE", "rescue").strip().lower(),
            thinking_mode=thinking,
            temperature=float(os.environ.get("AGENT_TEMPERATURE", "0.15")),
            max_tokens=int(os.environ.get("AGENT_MAX_TOKENS", "8192")),
            rescue_max_tokens=int(os.environ.get("AGENT_RESCUE_MAX_TOKENS", "4096")),
            rescue_temperature=float(os.environ.get("AGENT_RESCUE_TEMPERATURE", "0.15")),
            top_p=_optional_float("AGENT_TOP_P"),
            top_k=_optional_int("AGENT_TOP_K"),
        )


class ReasoningAgent:
    """Competition-compatible 397B agent with conservative failure recovery."""

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
        if self.config.mode not in {"direct", "rescue"}:
            raise ValueError(
                f"Unsupported AGENT_MODE={self.config.mode!r}; expected 'direct' or 'rescue'."
            )

    def solve(self, problem: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        del metadata
        trace: List[Dict[str, Any]] = []

        try:
            first = self._solve_once(problem)
        except Exception as exc:
            trace.append(
                {
                    "step": "direct_solver",
                    "content": {"status": "error", "error_type": type(exc).__name__},
                }
            )
            if self.config.mode != "rescue":
                raise
            recovered = self._fallback_solve(problem)
            trace.append(
                {
                    "step": "api_failure_recovery",
                    "content": {
                        "status": "completed",
                        "response_chars": len(recovered),
                        "thinking_mode": self.config.thinking_mode,
                    },
                }
            )
            return {"final_response": recovered, "trace": trace}

        has_final = self._has_final_marker(first)
        trace.append(
            {
                "step": "direct_solver",
                "content": {
                    "status": "completed",
                    "response_chars": len(first),
                    "thinking_mode": self.config.thinking_mode,
                    "has_final_marker": has_final,
                },
            }
        )

        # High-confidence path: never touch a completed answer.
        if self.config.mode == "direct" or has_final:
            return {"final_response": first, "trace": trace}

        try:
            rescued = self._rescue(problem, first)
        except Exception as exc:
            trace.append(
                {
                    "step": "conditional_rescue",
                    "content": {"status": "error", "error_type": type(exc).__name__},
                }
            )
            return {"final_response": first, "trace": trace}

        rescue_has_final = self._has_final_marker(rescued)
        trace.append(
            {
                "step": "conditional_rescue",
                "content": {
                    "status": "completed",
                    "response_chars": len(rescued),
                    "has_final_marker": rescue_has_final,
                },
            }
        )
        # Never replace a usable first response with another unfinished response.
        return {"final_response": rescued if rescue_has_final else first, "trace": trace}

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

    def _rescue(self, problem: str, candidate: str) -> str:
        response = self.client.chat(
            [
                {"role": "system", "content": RESCUE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "PROBLEM:\n"
                        f"{problem}\n\n"
                        "PRIOR ATTEMPT:\n"
                        f"{candidate}\n\n"
                        "Complete or extract the answer now."
                    ),
                },
            ],
            temperature=self.config.rescue_temperature,
            max_tokens=self.config.rescue_max_tokens,
            thinking_mode=self.config.thinking_mode,
            **self._sampling_kwargs(),
        )
        return self._require_text(response)

    def _fallback_solve(self, problem: str) -> str:
        response = self.client.chat(
            [
                {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ],
            temperature=self.config.rescue_temperature,
            max_tokens=self.config.rescue_max_tokens,
            thinking_mode=self.config.thinking_mode,
            **self._sampling_kwargs(),
        )
        return self._require_text(response)

    @staticmethod
    def _has_final_marker(text: str) -> bool:
        return "FINAL_ANSWER:" in text.upper()

    @staticmethod
    def _require_text(response: Any) -> str:
        if not isinstance(response, str):
            raise TypeError("Expected a text completion; tool-call responses are unsupported.")
        text = response.strip()
        if not text:
            raise ValueError("Model returned an empty response.")
        return text
