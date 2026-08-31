import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional


BASE_SYSTEM_PROMPT = """You are a rigorous mathematical problem-solving agent.
Solve the problem independently and prioritize correctness.

Use careful reasoning, but keep the visible response concise and easy to grade.
For objective-answer questions, end with one final line beginning with `FINAL_ANSWER:` followed by the actual answer.
For proof/derivation problems, give only the essential argument and then a `FINAL_ANSWER:` conclusion line.
Check signs, domains, assumptions, edge cases, counting overlaps, and option labels before answering.
Never output a placeholder.
"""

SECOND_SYSTEM_PROMPT = """Solve the mathematics problem independently from scratch.
Do not assume any other solver's work is correct or available. Prioritize correctness over speed.
Check algebra, signs, domains, assumptions, edge cases, counting overlaps, and the exact quantity requested.
Keep the visible answer concise and end objective questions with `FINAL_ANSWER:` followed by the actual answer.
"""

VERIFIER_SYSTEM_PROMPT = """You are an independent mathematical adjudicator.
You will receive one problem and two independently produced candidate solutions.
First solve/check the problem yourself. Then compare the candidates mathematically.
Do not choose by verbosity or majority wording. If both are wrong, repair the solution yourself.
Return a concise final response suitable for grading. For objective questions, end with exactly one
`FINAL_ANSWER:` line containing the answer. Never output a placeholder.
"""


@dataclass(frozen=True)
class AgentConfig:
    """Adaptive-v1 experimental configuration.

    Explicit proof questions are protected by the validated one-call B0 path.
    Other questions receive two independent solutions. Agreement ends early;
    disagreement triggers a third, independent adjudication call.
    """

    mode: str = "adaptive_verify"
    thinking_mode: bool = False
    temperature: float = 0.15
    second_temperature: float = 0.0
    verifier_temperature: float = 0.0
    max_tokens: int = 4096


class ReasoningAgent:
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
        if self.config.mode != "adaptive_verify":
            raise ValueError("adaptive-v1 supports adaptive_verify mode only")

    def solve(self, problem: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        del metadata
        trace: List[Dict[str, Any]] = []

        first = self._solve_once(problem, BASE_SYSTEM_PROMPT, self.config.temperature)
        trace.append(self._trace_step("direct_solver", first))

        # The all-question self-refine experiment disproportionately damaged
        # proof answers. Protect explicit proofs and derivations with B0 direct.
        if self._looks_like_proof(problem):
            trace.append({
                "step": "adaptive_router",
                "content": {"route": "proof_direct", "model_calls": 1},
            })
            return {"final_response": first, "trace": trace}

        try:
            second = self._solve_once(
                problem, SECOND_SYSTEM_PROMPT, self.config.second_temperature
            )
        except Exception as exc:
            trace.append({
                "step": "independent_solver",
                "content": {
                    "status": "fallback_to_direct",
                    "error_type": type(exc).__name__,
                    "model_calls": 2,
                },
            })
            return {"final_response": first, "trace": trace}

        trace.append(self._trace_step("independent_solver", second))
        answer_a = self._extract_answer(first)
        answer_b = self._extract_answer(second)
        agree = self._answers_equivalent(answer_a, answer_b)

        trace.append({
            "step": "agreement_check",
            "content": {
                "agree": agree,
                "first_answer_extracted": answer_a is not None,
                "second_answer_extracted": answer_b is not None,
            },
        })
        if agree:
            trace.append({
                "step": "adaptive_router",
                "content": {"route": "independent_agreement", "model_calls": 2},
            })
            return {"final_response": first, "trace": trace}

        try:
            verified = self._verify(problem, first, second)
        except Exception as exc:
            trace.append({
                "step": "disagreement_verifier",
                "content": {
                    "status": "fallback_to_direct",
                    "error_type": type(exc).__name__,
                    "model_calls": 3,
                },
            })
            return {"final_response": first, "trace": trace}

        trace.append(self._trace_step("disagreement_verifier", verified))
        trace.append({
            "step": "adaptive_router",
            "content": {"route": "disagreement_verified", "model_calls": 3},
        })
        return {"final_response": verified, "trace": trace}

    def _solve_once(self, problem: str, system_prompt: str, temperature: float) -> str:
        response = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem},
            ],
            temperature=temperature,
        )
        return self._require_text(response)

    def _verify(self, problem: str, first: str, second: str) -> str:
        response = self._chat(
            [
                {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "PROBLEM:\n"
                        f"{problem}\n\n"
                        "CANDIDATE A:\n"
                        f"{first}\n\n"
                        "CANDIDATE B:\n"
                        f"{second}\n\n"
                        "Independently adjudicate and return the best corrected final response."
                    ),
                },
            ],
            temperature=self.config.verifier_temperature,
        )
        return self._require_text(response)

    def _chat(self, messages: List[Dict[str, Any]], *, temperature: float) -> Any:
        try:
            return self.client.chat(
                messages,
                temperature=temperature,
                max_tokens=self.config.max_tokens,
                thinking_mode=self.config.thinking_mode,
            )
        except TypeError as exc:
            if "thinking_mode" not in str(exc):
                raise
            return self.client.chat(
                messages,
                temperature=temperature,
                max_tokens=self.config.max_tokens,
            )

    def _trace_step(self, name: str, response: str) -> Dict[str, Any]:
        return {
            "step": name,
            "content": {
                "status": "completed",
                "response_chars": len(response),
                "thinking_mode": self.config.thinking_mode,
            },
        }

    @staticmethod
    def _looks_like_proof(problem: str) -> bool:
        text = problem.lower()
        english = re.search(
            r"\b(prove|proof|demonstrate\s+that|establish\s+that|show\s+that)\b",
            text,
        )
        chinese = any(token in problem for token in ("证明", "求证", "论证", "证实"))
        return bool(english or chinese)

    @classmethod
    def _extract_answer(cls, response: str) -> Optional[str]:
        matches = list(re.finditer(r"FINAL_ANSWER\s*:\s*(.+)", response, flags=re.I))
        if matches:
            value = matches[-1].group(1).strip()
            if value:
                return value

        boxes = cls._balanced_boxed_values(response)
        if boxes:
            return boxes[-1].strip()

        # Conservative fallback: short final non-empty line only. Long prose is
        # not treated as an answer signature, forcing verifier adjudication.
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        if lines and len(lines[-1]) <= 120:
            return lines[-1]
        return None

    @staticmethod
    def _balanced_boxed_values(text: str) -> List[str]:
        values: List[str] = []
        start = 0
        while True:
            pos = text.find("\\boxed{", start)
            if pos < 0:
                break
            i = pos + len("\\boxed{")
            depth = 1
            j = i
            while j < len(text) and depth:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            if depth == 0:
                values.append(text[i : j - 1])
                start = j
            else:
                break
        return values

    @classmethod
    def _answers_equivalent(cls, a: Optional[str], b: Optional[str]) -> bool:
        if a is None or b is None:
            return False
        na = cls._normalize_answer(a)
        nb = cls._normalize_answer(b)
        if not na or not nb:
            return False
        if na == nb:
            return True
        xa = cls._simple_number(na)
        xb = cls._simple_number(nb)
        return xa is not None and xb is not None and xa == xb

    @staticmethod
    def _normalize_answer(value: str) -> str:
        text = value.strip().lower()
        text = text.replace("−", "-").replace("–", "-")
        text = text.replace("\\dfrac", "\\frac")
        text = text.replace("\\left", "").replace("\\right", "")
        text = text.replace("\\,", "").replace("\\!", "")
        text = text.replace("$", "").replace("\\(", "").replace("\\)", "")
        text = re.sub(r"\s+", "", text)
        text = text.strip("`*_.,;:。；， ")
        return text

    @staticmethod
    def _simple_number(value: str) -> Optional[Fraction]:
        text = value
        m = re.fullmatch(r"([+-]?)\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}", text)
        if m and int(m.group(3)) != 0:
            sign = -1 if m.group(1) == "-" else 1
            return Fraction(sign * int(m.group(2)), int(m.group(3)))
        try:
            if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
                return Fraction(text)
        except Exception:
            pass
        return None

    @staticmethod
    def _require_text(response: Any) -> str:
        if not isinstance(response, str):
            raise TypeError("Expected a text completion; tool-call responses are unsupported.")
        text = response.strip()
        if not text:
            raise ValueError("Model returned an empty response.")
        return text
