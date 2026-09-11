"""R2: budgeted mathematical computation with conservative answer preservation."""
from __future__ import annotations
from dataclasses import dataclass
import json
import re
import time
from typing import Any
from math_tools import run_math

SYSTEM = r'''You are a rigorous university-level mathematical problem solver.
Solve the actual question, including every subquestion. State necessary domains,
assumptions, branches and boundary conditions. Distinguish exact and approximate
answers. If asked for a proof, supply a complete essential argument, not just the
claimed conclusion. Never use a numerical spot check as a proof.
Use a short, purposeful derivation; do not repeatedly restart or explore discarded
approaches. Once the solution is established, stop. A complete correct answer is
more important than lengthy commentary.
End the finished solution with FINAL_ANSWER: followed by the requested result.
Put all parts of a multi-part result together. Do not output placeholders.
Do not claim to have executed code unless a tool result has actually been returned.
'''
TOOL_RULES = r'''
For substantial arithmetic, polynomial algebra, sums, matrices, combinatorial
counts, integrals or checking a proposed formula, use the mathematical Python
worker rather than trusting mental calculation. To request a calculation, give a
brief mathematical setup, then ONE fenced ```python code block and STOP. The
worker result will be returned before you finish. Code must print useful results.
Available: Python builtins, sympy (also prebound as sp), math, fractions, itertools.
No files, network, input, exec/eval, sympify/parse_expr or lambdify. Construct symbolic
expressions directly with symbols(), Rational(), Matrix(), etc. Keep calculations
small. You may give a complete solution directly when no computation is useful.
'''
HINTS = [
    (r'partial differential|\\partial|偏微分|boundary|初边值', 'For differential equations, check the PDE/ODE and all initial/boundary conditions; account for constants and allowed modes. Do not merely name a method.'),
    (r'probability|variance|expectation|概率|期望|方差|distribution|置信', 'Identify the probability model and support. Distinguish independence from disjointness, ordered from unordered outcomes, and variance from standard deviation.'),
    (r'group|field|ring|ideal|群|有限域|环|理想', 'Check hypotheses of every algebra theorem. Distinguish elements from subgroups, minimal polynomials from arbitrary polynomials, and necessary from sufficient conditions.'),
    (r'integral|residue|\\int|留数|积分|limit|极限|series', 'Check convergence, singularities, endpoints and domains before algebraic manipulation. State the requested value, not only an unevaluated reformulation.'),
    (r'matrix|eigen|rank|矩阵|特征值|秩', 'Keep exact arithmetic where possible. Distinguish algebraic/geometric multiplicity and verify dimensions and parameter cases.'),
]
CODE = re.compile(r'```python\s*\n(.*?)```',re.S | re.I)
FINAL = re.compile(r'^\s*(?:\*\*)?FINAL_ANSWER\s*[:：]\s*(.+)',re.M | re.I)


def visible(response: Any) -> str:
    if isinstance(response, str):
        text = response
    elif isinstance(response, dict):
        text = response.get('content') or ''
    else:
        return ''
    if not isinstance(text,str):
        return ''
    # Strip only a closed reasoning segment. An unclosed segment is not a final answer.
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    if '<think>' in text:
        text = text.split('<think>',1)[0]
    return text.strip()


def complete(text: str) -> bool:
    if not text or CODE.search(text):
        return False
    hits = list(FINAL.finditer(text))
    if hits:
        answer = hits[-1].group(1).strip().strip('*')
        return answer.lower() not in {'<answer>','unknown','n/a','none provided','...'}
    # Boxed expressions may contain nested braces. Require closing balance.
    start = text.rfind(r'\boxed{')
    if start >= 0:
        depth = 1
        for c in text[start+7:]:
            depth += (c == '{') - (c == '}')
            if depth == 0:
                return True
    return False


@dataclass(frozen=True)
class R2Config:
    primary_tokens: int = 4096
    followup_tokens: int = 4096
    rescue_tokens: int = 2048
    max_calls: int = 3
    max_tool_runs: int = 2
    soft_seconds: float = 600.0
    temperature: float = 0.1
    # Explicit submission default, never dependent on a workflow environment variable.
    thinking_mode: bool = False
    tools: bool = True


class ReasoningAgent:
    def __init__(self, client, *args, config: R2Config | None = None, **kwargs):
        self.client = client
        self.config = config or R2Config()

    def solve(self, problem: str, metadata: dict) -> dict:
        del metadata  # No gold, topic labels, ordering, or cross-question state.
        started = time.monotonic()
        c = self.config
        hint = next((v for p,v in HINTS if re.search(p,problem,re.I)), '')
        messages = [{'role':'system','content': SYSTEM + ('\n'+hint if hint else '') + (TOOL_RULES if c.tools else '')},
                    {'role':'user','content':problem}]
        trace = []
        best = ''
        partial = ''
        tools = 0
        calls = 0
        last_tool = None
        for turn in range(c.max_calls):
            # Leave time for the platform to persist the result. Client timeouts remain official.
            if turn and time.monotonic()-started > c.soft_seconds:
                break
            closing = turn == c.max_calls-1 or tools >= c.max_tool_runs
            if closing:
                messages.append({'role':'user','content':'No further tool calls. Finish the solution now using established results. Give a concise essential derivation and the explicit FINAL_ANSWER. Do not print code or repeat abandoned approaches.'})
            budget = c.primary_tokens if turn == 0 else (c.rescue_tokens if closing else c.followup_tokens)
            calls += 1
            try:
                raw = self.client.chat(messages=messages, temperature=c.temperature,
                    max_tokens=budget, thinking_mode=c.thinking_mode)
                text = visible(raw)
                trace.append({'step':'model','content':{'call':calls,'status':'completed','response_chars':len(text),'thinking_mode':c.thinking_mode,'max_tokens':budget}})
            except Exception as exc:
                # Do not log messages/credentials from exception strings or private client fields.
                trace.append({'step':'model','content':{'call':calls,'status':'error','error_type':type(exc).__name__}})
                if best:
                    break
                messages.append({'role':'user','content':'Please solve the original problem directly and return a complete concise answer. No code.'})
                continue
            if not text:
                messages.append({'role':'user','content':'The response contained no usable answer. Solve the original problem concisely and end with FINAL_ANSWER. No code.'})
                continue
            partial = text
            code_match = CODE.search(text) if c.tools else None
            # Keep a completed answer rather than replace it with an incomplete later response.
            if complete(text):
                best = text
                break
            messages.append({'role':'assistant','content':text[:18000]})
            if code_match and not closing and tools < c.max_tool_runs:
                tools += 1
                result = run_math(code_match.group(1))
                last_tool = result
                trace.append({'step':'math_tool','content':{'run':tools,'ok':result['ok'],'error_type':result.get('error'),'output_chars':len(result.get('stdout',''))}})
                messages.append({'role':'user','content':'Mathematical Python worker returned:\n'+json.dumps(result,ensure_ascii=False)+'\nUse this result to finish ALL requested parts. If code failed, correct the issue or solve analytically. A successful execution does not by itself validate the mathematical setup.'})
            else:
                # Short closure rescue, not unconditional rewriting of successful answers.
                messages.append({'role':'user','content':'The response is missing a clearly completed final answer. Finish the mathematics and state the actual requested answer, preserving necessary proof and all subparts. Do not simply restate the question. No code.'})
        final = best or partial
        if not final:
            final = 'Unable to obtain a mathematical answer within the available model calls.'
        trace.append({'step':'finalize','content':{'calls':calls,'tool_runs':tools,
            'complete':complete(final),'fallback':not bool(best),'seconds':round(time.monotonic()-started,3)}})
        return {'final_response':final, 'trace':trace}
