"""R2.3: preserve the remaining tool opportunity after an incomplete reply."""
from __future__ import annotations
from dataclasses import dataclass
import json,re,time
from typing import Any
from math_tools import run_math
SYSTEM=r'''You are a rigorous university-level mathematical problem solver.
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
TOOL_RULES=r'''
Use the mathematical Python worker early for numerical roots, iterations,
probabilities, matrices, integrals and nontrivial arithmetic. Do NOT hand-estimate
trigonometric/exponential values or write a long derivation before requesting a
calculation. Give at most 150 words of setup, then ONE fenced ```python block
and STOP; the actual result will be returned. Print all needed results and residuals.
Allowed packages: math, sympy (also sp), numpy, scipy.stats, scipy.optimize,
mpmath, fractions, itertools. Only public mathematical functions are exposed.
Use scipy.optimize.brentq on a continuous sign-changing interval for generic
numeric roots. If a SPECIFIC numerical method is requested, implement that method
and print its iterates. Do not substitute a different method. For d decimal
places, compute at least d+4 reliable digits, check residuals, and for bisection
continue until BOTH bracket endpoints round to the same requested value. A short
interval alone can straddle a rounding threshold. Verify first/positive/root-domain
requirements rather than accepting the first numerical root a solver happens to find.
For statistics, explicitly set covariance ddof and test assumptions. Do not confuse
successful code execution with a correct mathematical formulation.
No files, network, input, eval/exec, sympify/parse_expr, lambdify, dir/globals/getattr,
or introspection. Construct symbolic expressions directly. Define all variables.
Keep code short. No tools are needed for a purely conceptual answer or a proof.
'''
HINTS=[
 (r'partial differential|\\partial|偏微分|boundary|初边值','For differential equations, check the PDE/ODE and all initial/boundary conditions; account for constants and allowed modes. Do not merely name a method.'),
 (r'probability|variance|expectation|概率|期望|方差|distribution|置信','Identify the probability model and support. Distinguish independence from disjointness, ordered from unordered outcomes, and variance from standard deviation.'),
 (r'group|field|ring|ideal|群|有限域|环|理想','Check hypotheses of every algebra theorem. Distinguish elements from subgroups, minimal polynomials from arbitrary polynomials, and necessary from sufficient conditions.'),
 (r'integral|residue|\\int|留数|积分|limit|极限|series','Check convergence, singularities, endpoints and domains before algebraic manipulation. State the requested value, not only an unevaluated reformulation.'),
 (r'matrix|eigen|rank|矩阵|特征值|秩','Keep exact arithmetic where possible. Distinguish algebraic/geometric multiplicity and verify dimensions and parameter cases.')]
CODE=re.compile(r'```python\s*\n(.*?)```',re.S|re.I)
FINAL=re.compile(r'^\s*(?:\*\*)?FINAL_ANSWER\s*[:：]\s*(.+)',re.M|re.I)

def visible(response: Any) -> str:
    if isinstance(response,str): text=response
    elif isinstance(response,dict): text=response.get('content') or ''
    else:return ''
    if not isinstance(text,str):return ''
    text=re.sub(r'<think>.*?</think>','',text,flags=re.S)
    if '<think>' in text:text=text.split('<think>',1)[0]
    return text.strip()

def complete(text: str) -> bool:
    if not text or CODE.search(text):return False
    hits=list(FINAL.finditer(text))
    if hits:
        answer=hits[-1].group(1).strip().strip('*')
        return answer.lower() not in {'<answer>','unknown','n/a','none provided','...'}
    start=text.rfind(r'\boxed{')
    if start>=0:
        depth=1
        for ch in text[start+7:]:
            depth+=(ch=='{')-(ch=='}')
            if depth==0:return True
    return False

@dataclass(frozen=True)
class R2Config:
    primary_tokens:int=4096
    followup_tokens:int=4096
    rescue_tokens:int=2048
    max_calls:int=3
    max_tool_runs:int=2
    soft_seconds:float=600.0
    temperature:float=0.1
    thinking_mode:bool=False
    tools:bool=True

class ReasoningAgent:
    def __init__(self,client,*args,config:R2Config|None=None,**kwargs):
        self.client=client;self.config=config or R2Config()
    def solve(self,problem:str,metadata:dict)->dict:
        del metadata
        started=time.monotonic();c=self.config
        hint=next((v for p,v in HINTS if re.search(p,problem,re.I)),'')
        messages=[{'role':'system','content':SYSTEM+('\n'+hint if hint else '')+(TOOL_RULES if c.tools else '')},{'role':'user','content':problem}]
        trace=[];best='';partial='';tools=0;calls=0;last_tool=None
        for turn in range(c.max_calls):
            if turn and time.monotonic()-started>c.soft_seconds:break
            closing=turn==c.max_calls-1 or tools>=c.max_tool_runs
            if closing:
                draft=partial if len(partial)<=12000 else partial[:3000]+'\n[earlier draft shortened]\n'+partial[-9000:]
                evidence=json.dumps(last_tool,ensure_ascii=False) if last_tool else '(no tool result)'
                messages=[
                    {'role':'system','content':SYSTEM+'\nFINAL RESPONSE NOW: put FINAL_ANSWER: and the actual requested result on the FIRST line, then a concise essential justification (normally under 500 words). Preserve all required proof or numerical-method steps. No code, no restarts, no lengthy re-derivation. Do not invent a numerical result or treat an unverified draft as fact.'},
                    {'role':'user','content':problem},
                    {'role':'user','content':'Previous draft (may contain mistakes):\n'+draft+'\nActual last tool result:\n'+evidence[:12000]}]
            budget=c.primary_tokens if turn==0 else (c.rescue_tokens if closing else c.followup_tokens)
            calls+=1
            try:
                text=visible(self.client.chat(messages=messages,temperature=c.temperature,max_tokens=budget,thinking_mode=c.thinking_mode))
                trace.append({'step':'model','content':{'call':calls,'status':'completed','response_chars':len(text),'thinking_mode':c.thinking_mode,'max_tokens':budget}})
            except Exception as exc:
                trace.append({'step':'model','content':{'call':calls,'status':'error','error_type':type(exc).__name__}})
                if best:break
                messages.append({'role':'user','content':'Please solve the original problem directly and return a complete concise answer. No code.'});continue
            if not text:
                messages.append({'role':'user','content':'The response contained no usable answer. Solve the original problem concisely and end with FINAL_ANSWER. No code.'});continue
            partial=text
            code_match=CODE.search(text) if c.tools else None
            if complete(text):best=text;break
            messages.append({'role':'assistant','content':text[:18000]})
            if code_match and not closing and tools<c.max_tool_runs:
                tools+=1;result=run_math(code_match.group(1));last_tool=result
                trace.append({'step':'math_tool','content':{'run':tools,'ok':result['ok'],'error_type':result.get('error'),'output_chars':len(result.get('stdout',''))}})
                messages.append({'role':'user','content':'Mathematical Python worker returned:\n'+json.dumps(result,ensure_ascii=False)+'\nUse the actual result to finish ALL requested parts concisely, normally under 500 words. Do not repeat the whole derivation. If code failed, correct it using the permitted packages. Check numerical residuals and rounding before committing. Execution success does not validate the mathematical setup.'})
            else:
                if c.tools and not closing and tools<c.max_tool_runs:
                    messages.append({'role':'user','content':'The previous response is unfinished and no calculation was executed. Do NOT restart the derivation or hand-estimate numerical values. If a numerical step remains, immediately give a short Python block using the formula already derived and the permitted packages; print the requested result with adequate precision and a residual check. At most 80 words before the code. If this is a pure proof, finish the essential argument and FINAL_ANSWER instead.'})
                else:
                    messages.append({'role':'user','content':'Finish the existing argument concisely and state the actual FINAL_ANSWER. Preserve every requested part. No code.'})
        final=best or partial or 'Unable to obtain a mathematical answer within the available model calls.'
        trace.append({'step':'finalize','content':{'calls':calls,'tool_runs':tools,'complete':complete(final),'fallback':not bool(best),'seconds':round(time.monotonic()-started,3)}})
        return {'final_response':final,'trace':trace}
