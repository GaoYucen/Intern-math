"""Offline-only checks with explicit reference provenance and precision.
Original data/historical reports unchanged. No proof or method certification.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass
import math
import re
from typing import Literal

@dataclass(frozen=True)
class NumericReference:
    value: str
    kind: Literal['exact','rounded','unverified'] = 'unverified'
    decimals: int | None = None
    method_required: bool = False

def parse_real(text: str) -> float | None:
    """Small arithmetic AST interpreter; never eval/sympify generated strings."""
    s=str(text).strip().strip('$`')
    if len(s)>256: return None
    s=s.replace(r'\left','').replace(r'\right','').replace(r'\pi','pi')
    s=s.replace('−','-').replace(r'\cdot','*').replace(r'\times','*')
    for _ in range(6):
        updated=re.sub(r'\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}',r'((\1)/(\2))',s)
        updated=re.sub(r'\\sqrt\{([^{}]+)\}',r'sqrt(\1)',updated)
        if updated==s: break
        s=updated
    s=s.replace('^','**')
    constants={'pi':math.pi,'e':math.e,'E':math.e}
    functions={'sqrt':math.sqrt,'exp':math.exp,'log':math.log,'ln':math.log,'erf':math.erf,'erfc':math.erfc}
    def walk(node):
        if isinstance(node,ast.Constant) and type(node.value) in {int,float}:
            val=float(node.value)
        elif isinstance(node,ast.Name) and node.id in constants:
            val=constants[node.id]
        elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            val=walk(node.operand)*(1 if isinstance(node.op,ast.UAdd) else -1)
        elif isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow)):
            a,b=walk(node.left),walk(node.right)
            if isinstance(node.op,ast.Add): val=a+b
            elif isinstance(node.op,ast.Sub): val=a-b
            elif isinstance(node.op,ast.Mult): val=a*b
            elif isinstance(node.op,ast.Div): val=a/b
            else:
                if abs(b)>1000: raise ValueError('exponent limit')
                val=a**b
        elif isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in functions and len(node.args)==1 and not node.keywords:
            val=functions[node.func.id](walk(node.args[0]))
        else: raise ValueError('unsupported expression')
        if isinstance(val,complex) or not math.isfinite(val) or abs(val)>1e100:
            raise ValueError('nonfinite/oversize')
        return val
    try:
        tree=ast.parse(s,mode='eval')
        if len(list(ast.walk(tree)))>100: return None
        return walk(tree.body)
    except (ValueError,TypeError,SyntaxError,OverflowError,ZeroDivisionError):
        return None

def check_numeric(prediction: str, reference: NumericReference) -> bool | None:
    if reference.method_required or reference.kind=='unverified': return None
    pred,gold=parse_real(prediction),parse_real(reference.value)
    if pred is None or gold is None: return None
    if reference.kind=='exact':
        return math.isclose(pred,gold,rel_tol=1e-12,abs_tol=1e-12)
    if reference.decimals is None or not 0<=reference.decimals<=12:
        raise ValueError('Rounded references require verified decimal precision')
    return abs(pred-gold)<=0.5*10**(-reference.decimals)+1e-12

def single_choice(text: str, labels: str='ABCD') -> str | None:
    s=str(text).strip().strip('*`$')
    m=re.match(r'(?:option\s*)?[\[(]?(['+re.escape(labels)+r'])[\])]?(?=$|[\s.:：,，])',s,re.I)
    if not m: return None
    first=m.group(1).upper()
    if re.search(r'(?<![A-Za-z])['+re.escape(labels)+r'](?![A-Za-z])',s[m.end():],re.I): return None
    return first
