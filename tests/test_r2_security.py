import pytest
from math_tools import run_math
from dataclasses import asdict
from r2_agent import R2Config
from user_agent import ReasoningAgent

@pytest.mark.parametrize('code',[
    'print(S("1+1"))', 'print(sp.solve("x**2-4"))', 'x="x"; print(x)',
    'symbols=sp.solve', 'print=sp.solve', 'from sympy import solve as Symbol',
    'sp.Symbol=sp.solve', 's=f"{2}";print(s)', 'print(sp.Symbol("__x"))',
])
def test_no_implicit_expression_parser(code):
    assert not run_math(code)['ok']

def test_safe_string_constructors():
    r=run_math('from sympy import *\nx=symbols("x",real=True)\nprint(solve(x**2-4,x))\nprint(Rational("1/2"))')
    assert r['ok'] and '-2' in r['stdout'] and '1/2' in r['stdout']

def test_submission_defaults_equal_evaluated_defaults(monkeypatch):
    monkeypatch.setenv('INTERN_THINKING_MODE','1')
    agent=ReasoningAgent(object())
    assert asdict(agent.config)==asdict(R2Config())
    assert agent.config.mode=='r2_budgeted_tools'
