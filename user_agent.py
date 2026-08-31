import os
from dataclasses import dataclass
from typing import Any, Dict, List


BASE_SYSTEM_PROMPT = """You are a rigorous mathematical problem-solving agent.
Solve the problem independently and prioritize correctness.

Use careful internal reasoning, but keep the visible response concise and easy to grade.

Rules:
1. For multiple-choice, numeric, symbolic, short-answer, and yes/no problems, return only one final line beginning with `FINAL_ANSWER:` followed by the actual answer.
2. For proof/derivation problems, give only a concise proof containing the essential argument, then one final `FINAL_ANSWER:` line with the conclusion.
3. Never spend tokens exploring abandoned approaches, repeatedly checking settled work, or narrating uncertainty.
4. Check signs, domains, assumptions, edge cases, and option labels before answering.
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


@dataclass(frozen=True)
class AgentConfig:
    """Submission-safe configuration with reproducible B0 defaults.

    The validated competition baseline is a single direct call with thinking mode
    disabled and a 4096-token cap. More expensive self-refine behavior remains
    available only as an explicit local experiment.
    """

    mode: str = "direct"  # direct | self_refine
    thinking_mode: bool = False
    temperature: float = 0.15
    max_tokens: int = 4096
    refine_temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "AgentConfig":
        thinking_raw = os.environ.get("INTERN_THINKING_MODE", "0").strip().lower()
        thinking = thinking_raw not in {"0", "false", "no", "off"}
        return cls(
            mode=os.environ.get("AGENT_MODE", "direct").strip().lower(),
            thinking_mode=thinking,
            temperature=float(os.environ.get("AGENT_TEMPERATURE", "0.15")),
            max_tokens=int(os.environ.get("AGENT_MAX_TOKENS", "4096")),
            refine_temperature=float(os.environ.get("REFINE_TEMPERATURE", "0.0")),
        )


class ReasoningAgent:
    """Competition-compatible mathematical reasoning agent.

    The official runner supplies the client and calls solve(problem, metadata).
    This implementation does not depend on hidden answers, cross-problem state,
    local absolute paths, or private client fields.
    """

    def __init__(
        self,
        client: Any,
        config: AgentConfig | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        del args, kwargs  # tolerate harmless runner-side constructor extensions
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

    def _chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> Any:
        """Call the official-like client while tolerating older compatible clients.

        Current Intern clients support thinking_mode. If a runner provides a
        compatible client whose Python signature does not expose that keyword,
        retry without it; the first TypeError occurs before a network request.
        """
        try:
            return self.client.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking_mode=self.config.thinking_mode,
            )
        except TypeError as exc:
            if "thinking_mode" not in str(exc):
                raise
            return self.client.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    def _solve_once(self, problem: str) -> str:
        response = self._chat(
            [
                {"role": "system", "content": BASE_SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return self._require_text(response)

    def _refine(self, problem: str, candidate: str, idx: Any) -> str:
        del idx  # kept in signature so future trace/session logic can use it safely.
        response = self._chat(
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
