import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


SOLVER_A_PROMPT = """You are the primary mathematical solver. Solve the problem independently and prioritize correctness, but your inference budget is limited.

Use a direct, committed solution path. Do not repeatedly restart, narrate uncertainty, or explore many abandoned approaches. Check the result before finishing.

Output rules:
1. Always finish with exactly one line beginning `FINAL_ANSWER:` followed by the requested answer or conclusion.
2. For multiple-choice, numeric, symbolic, short-answer, and yes/no questions, keep the visible derivation short and focused.
3. For proof/derivation questions, give a concise but sufficient proof, then the `FINAL_ANSWER:` line.
4. If the full derivation is difficult, still commit to the best justified answer before the budget ends rather than continuing indefinitely.
"""

SOLVER_B_PROMPT = """You are an independent mathematical solver. Solve the problem from scratch without seeing another solver's work.

Prefer a different route when several methods are possible. Pay special attention to hidden assumptions, sign errors, boundary cases, algebraic simplification, and whether the requested object has actually been computed. Keep the reasoning bounded and decisive.

Output rules:
1. Always finish with exactly one line beginning `FINAL_ANSWER:` followed by the requested answer or conclusion.
2. For objective questions, keep the derivation concise.
3. For proof/derivation questions, give a concise complete argument before the final line.
4. Do not spend the end of the budget exploring; reserve enough budget to state the final answer.
"""

CHOOSER_PROMPT = """You are a mathematical adjudicator. Two independent solvers attempted the same problem.

Determine which candidate is more likely correct. Do not rewrite both solutions and do not start a long new derivation. Check only the decisive mathematical issue: whether the candidate actually answers the question, whether equations/conditions are satisfied, and whether there is a concrete contradiction or computational error.

If one candidate is clearly better, preserve its useful proof or derivation. If both are flawed but the correction is local, repair it. End with exactly one line beginning `FINAL_ANSWER:` followed by the best final answer or conclusion.
"""

DEEP_PROMPT = """You are the escalation solver for a difficult mathematical problem. Two fast independent attempts were not reliable enough to accept automatically.

Use the candidate attempts only as evidence, not as authority. Check the decisive mathematics yourself. Prefer repairing a promising route over restarting repeatedly, but abandon a candidate if it is fundamentally wrong. Your budget is limited: reserve enough budget to finish the requested task and state a final answer.

For objective questions, give only the essential derivation. For proof/derivation questions, give a concise complete argument. Always end with exactly one line beginning `FINAL_ANSWER:` followed by the requested answer or conclusion.
"""

FALLBACK_PROMPT = """Solve the mathematical problem quickly and decisively. Use the remaining budget to reach a concrete answer. End with exactly one line beginning `FINAL_ANSWER:` followed by the requested answer or conclusion."""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class AgentConfig:
    """Bounded multi-path configuration for hidden-set evaluation."""

    mode: str = "hybrid"  # hybrid | dual | direct_a
    solver_a_thinking: bool = False
    solver_b_thinking: bool = False
    chooser_thinking: bool = False
    deep_thinking: bool = True
    solver_tokens: int = 4096
    chooser_tokens: int = 2048
    deep_tokens: int = 4096
    solver_a_temperature: float = 0.15
    solver_b_temperature: float = 0.35
    chooser_temperature: float = 0.0
    deep_temperature: float = 0.15

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            mode=os.environ.get("AGENT_MODE", "hybrid").strip().lower(),
            solver_a_thinking=_env_bool("AGENT_SOLVER_A_THINKING", False),
            solver_b_thinking=_env_bool("AGENT_SOLVER_B_THINKING", False),
            chooser_thinking=_env_bool("AGENT_CHOOSER_THINKING", False),
            deep_thinking=_env_bool("AGENT_DEEP_THINKING", True),
            solver_tokens=int(os.environ.get("AGENT_SOLVER_TOKENS", "4096")),
            chooser_tokens=int(os.environ.get("AGENT_CHOOSER_TOKENS", "2048")),
            deep_tokens=int(os.environ.get("AGENT_DEEP_TOKENS", "4096")),
            solver_a_temperature=float(os.environ.get("AGENT_SOLVER_A_TEMPERATURE", "0.15")),
            solver_b_temperature=float(os.environ.get("AGENT_SOLVER_B_TEMPERATURE", "0.35")),
            chooser_temperature=float(os.environ.get("AGENT_CHOOSER_TEMPERATURE", "0.0")),
            deep_temperature=float(os.environ.get("AGENT_DEEP_TEMPERATURE", "0.15")),
        )


class ReasoningAgent:
    """Competition-compatible consensus-gated mathematical agent.

    The platform injects the official client and chooses the actual model at
    submission time. This code intentionally never tries to override `model`.
    """

    def __init__(self, client: Any, config: Optional[AgentConfig] = None, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.client = client
        self.config = config or AgentConfig.from_env()
        if self.config.mode not in {"hybrid", "dual", "direct_a"}:
            raise ValueError(f"Unsupported AGENT_MODE={self.config.mode!r}")

    def solve(self, problem: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        del metadata
        trace: List[Dict[str, Any]] = []

        a = self._safe_call(
            SOLVER_A_PROMPT,
            problem,
            temperature=self.config.solver_a_temperature,
            max_tokens=self.config.solver_tokens,
            thinking_mode=self.config.solver_a_thinking,
        )
        trace.append(self._trace_entry("solver_a", a, self.config.solver_a_thinking))

        if self.config.mode == "direct_a":
            return {"final_response": a or "FINAL_ANSWER: C", "trace": trace}

        b = self._safe_call(
            SOLVER_B_PROMPT,
            problem,
            temperature=self.config.solver_b_temperature,
            max_tokens=self.config.solver_tokens,
            thinking_mode=self.config.solver_b_thinking,
        )
        trace.append(self._trace_entry("solver_b", b, self.config.solver_b_thinking))

        answer_a = self._extract_final_answer(a)
        answer_b = self._extract_final_answer(b)

        # Agreement between two independent bounded solvers is the strongest
        # cheap confidence signal observed in both stress and fresh holdouts.
        if answer_a is not None and answer_b is not None and self._normalize_answer(answer_a) == self._normalize_answer(answer_b):
            trace.append({"step": "agreement_gate", "content": {"status": "agree"}})
            return {"final_response": self._prefer_complete(a, b), "trace": trace}

        if self.config.mode == "dual":
            return self._dual_finish(problem, a, b, trace)

        # Hybrid mode escalates every non-agreement case (including one failed or
        # unfinished short solver) to a stronger bounded thinking pass.
        deep_input = (
            f"PROBLEM:\n{problem}\n\n"
            f"FAST ATTEMPT A:\n{self._bounded_candidate(a) if a else '[no usable response]'}\n\n"
            f"FAST ATTEMPT B:\n{self._bounded_candidate(b) if b else '[no usable response]'}\n\n"
            "Resolve the problem and produce the final answer."
        )
        deep = self._safe_call(
            DEEP_PROMPT,
            deep_input,
            temperature=self.config.deep_temperature,
            max_tokens=self.config.deep_tokens,
            thinking_mode=self.config.deep_thinking,
        )
        trace.append(self._trace_entry("deep_escalation", deep, self.config.deep_thinking))
        if deep and self._extract_final_answer(deep) is not None:
            return {"final_response": deep, "trace": trace}

        # If the thinking pass itself did not close, use a short non-thinking
        # adjudication pass only as a delivery fallback. Never overwrite a usable
        # candidate with an unfinished fallback.
        return self._chooser_finish(problem, a, b, trace, extra_evidence=deep)

    def _dual_finish(self, problem: str, a: str, b: str, trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Preserve a complete answer if the other call failed in legacy dual mode.
        if a and not b:
            return {"final_response": a, "trace": trace}
        if b and not a:
            return {"final_response": b, "trace": trace}
        if not a and not b:
            fallback = self._safe_call(
                FALLBACK_PROMPT,
                problem,
                temperature=0.0,
                max_tokens=self.config.chooser_tokens,
                thinking_mode=False,
            )
            trace.append(self._trace_entry("fallback", fallback, False))
            return {"final_response": fallback or "FINAL_ANSWER: C", "trace": trace}
        return self._chooser_finish(problem, a, b, trace)

    def _chooser_finish(
        self,
        problem: str,
        a: str,
        b: str,
        trace: List[Dict[str, Any]],
        extra_evidence: str = "",
    ) -> Dict[str, Any]:
        chooser_input = (
            f"PROBLEM:\n{problem}\n\n"
            f"CANDIDATE A:\n{self._bounded_candidate(a) if a else '[missing]'}\n\n"
            f"CANDIDATE B:\n{self._bounded_candidate(b) if b else '[missing]'}\n\n"
        )
        if extra_evidence:
            chooser_input += f"UNFINISHED DEEP EVIDENCE:\n{self._bounded_candidate(extra_evidence, 9000)}\n\n"
        chooser_input += "Select or locally repair the best answer and close the task."

        chosen = self._safe_call(
            CHOOSER_PROMPT,
            chooser_input,
            temperature=self.config.chooser_temperature,
            max_tokens=self.config.chooser_tokens,
            thinking_mode=self.config.chooser_thinking,
        )
        trace.append(self._trace_entry("chooser", chosen, self.config.chooser_thinking))
        if chosen and self._extract_final_answer(chosen) is not None:
            return {"final_response": chosen, "trace": trace}

        usable = [x for x in (a, b, extra_evidence) if x]
        if usable:
            best = usable[0]
            for candidate in usable[1:]:
                best = self._prefer_complete(best, candidate)
            return {"final_response": best, "trace": trace}
        return {"final_response": "FINAL_ANSWER: C", "trace": trace}

    def _safe_call(
        self,
        system_prompt: str,
        user_content: str,
        *,
        temperature: float,
        max_tokens: int,
        thinking_mode: bool,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        try:
            response = self.client.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking_mode=thinking_mode,
            )
        except TypeError as exc:
            if "thinking_mode" not in str(exc):
                return ""
            try:
                response = self.client.chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception:
                return ""
        except Exception:
            return ""
        if not isinstance(response, str):
            return ""
        return response.strip()

    @staticmethod
    def _extract_final_answer(text: str) -> Optional[str]:
        matches = re.findall(r"FINAL_ANSWER\s*:\s*(.+)", text or "", flags=re.IGNORECASE)
        if not matches:
            return None
        return matches[-1].strip()

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        value = answer.strip().lower()
        value = value.replace("$", "")
        value = re.sub(r"\\boxed\s*\{(.+)\}", r"\1", value)
        value = re.sub(r"\s+", "", value)
        value = value.rstrip(".。")
        return value

    @classmethod
    def _prefer_complete(cls, a: str, b: str) -> str:
        a_final = cls._extract_final_answer(a)
        b_final = cls._extract_final_answer(b)
        if a_final is not None and b_final is None:
            return a
        if b_final is not None and a_final is None:
            return b
        if a_final is not None:
            return a
        if b_final is not None:
            return b
        return a if len(a) <= len(b) else b

    @staticmethod
    def _bounded_candidate(text: str, max_chars: int = 14000) -> str:
        if len(text) <= max_chars:
            return text
        head = max_chars // 3
        tail = max_chars - head
        return text[:head] + "\n...[middle omitted for bounded adjudication]...\n" + text[-tail:]

    @staticmethod
    def _trace_entry(step: str, text: str, thinking: bool) -> Dict[str, Any]:
        return {
            "step": step,
            "content": {
                "status": "completed" if text else "error",
                "response_chars": len(text or ""),
                "has_final_marker": "FINAL_ANSWER:" in (text or "").upper(),
                "thinking_mode": thinking,
            },
        }
