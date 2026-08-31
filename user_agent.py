from dataclasses import dataclass
from typing import Any, Dict, List


BASE_SYSTEM_PROMPT = """You are a rigorous mathematical problem-solving agent.
Solve the problem independently and prioritize correctness.

Use careful reasoning, but keep the visible response concise and easy to grade.

Rules:
1. For multiple-choice, numeric, symbolic, short-answer, and yes/no problems, end with one final line beginning with `FINAL_ANSWER:` followed by the actual answer.
2. For proof/derivation problems, give only the essential proof or derivation, then one final `FINAL_ANSWER:` line with the conclusion.
3. Never spend tokens exploring abandoned approaches, repeatedly checking settled work, or narrating uncertainty.
4. Check signs, domains, assumptions, edge cases, counting overlaps, and option labels before answering.
5. Do not literally copy placeholders such as `<answer>`.
6. The final line is mandatory. Examples of format only:
   FINAL_ANSWER: B
   FINAL_ANSWER: -1
   FINAL_ANSWER: x^2+1
   FINAL_ANSWER: No
The text after `FINAL_ANSWER:` must be the actual requested answer, not an explanation or meta-comment.
"""

REFINE_SYSTEM_PROMPT = """You are a strict mathematical solution auditor.
Independently re-check the problem and candidate answer from scratch before accepting it.
Look especially for algebraic slips, sign errors, domain/assumption mistakes, double counting,
missing cases, unjustified proof steps, and mismatches between the requested quantity and the
candidate's final answer. If the candidate is wrong, repair it.

For objective-answer questions, keep the visible response concise and end with exactly one
`FINAL_ANSWER:` line containing the corrected answer. For proof/derivation questions, give only
a concise repaired argument containing the essential steps, then the `FINAL_ANSWER:` line.
Never output a placeholder.
"""


@dataclass(frozen=True)
class AgentConfig:
    """Fixed competition submission configuration.

    submission-v0 uses a two-call solve-then-audit pipeline. Thinking mode is
    disabled because earlier controlled tests showed more stable completion and
    final-answer delivery with a 4096-token cap.
    """

    mode: str = "self_refine"  # submission default; direct remains available for tests
    thinking_mode: bool = False
    temperature: float = 0.15
    max_tokens: int = 4096
    refine_temperature: float = 0.0


class ReasoningAgent:
    """Competition-compatible mathematical reasoning agent.

    The official runner supplies the client and calls solve(problem, metadata).
    This implementation does not depend on hidden answers, cross-problem state,
    local absolute paths, environment-specific model settings, or private client fields.
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
        self.config = config or AgentConfig()
        if self.config.mode not in {"direct", "self_refine"}:
            raise ValueError(
                f"Unsupported mode={self.config.mode!r}; expected 'direct' or 'self_refine'."
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
        """Call the official-like client and tolerate older compatible signatures."""
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
        del idx
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
                        "Independently audit the candidate and return the corrected final response."
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
