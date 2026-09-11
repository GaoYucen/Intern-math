import os
import pytest
from math_tools import run_math, validate
from r2_agent import ReasoningAgent, R2Config, complete, visible

class Fake:
    def __init__(self, replies):
        self.replies=iter(replies); self.calls=[]
    def chat(self, **kwargs):
        self.calls.append(kwargs)
        r=next(self.replies)
        if isinstance(r,Exception): raise r
        return r

def test_fixed_config(monkeypatch):
    monkeypatch.setenv('INTERN_THINKING_MODE','1')
    f=Fake(['FINAL_ANSWER: 2'])
    r=ReasoningAgent(f).solve('1+1',{'answer':'999','subject':'secret'})
    assert f.calls[0]['thinking_mode'] is False
    assert len(f.calls)==1 and '999' not in str(f.calls)
    assert r['final_response']=='FINAL_ANSWER: 2'

def test_tool_roundtrip():
    f=Fake(['```python\nprint(2**10)\n```','FINAL_ANSWER: 1024'])
    r=ReasoningAgent(f).solve('2^10',{})
    assert r['final_response']=='FINAL_ANSWER: 1024'
    assert any(t['step']=='math_tool' and t['content']['ok'] for t in r['trace'])

def test_closure():
    f=Fake(['An unfinished derivation','FINAL_ANSWER: 4'])
    assert complete(ReasoningAgent(f).solve('2+2',{})['final_response'])

def test_failure_keeps_partial():
    f=Fake(['A useful partial derivation',RuntimeError('SECRET'),RuntimeError('SECRET')])
    r=ReasoningAgent(f).solve('problem',{})
    assert r['final_response']=='A useful partial derivation'
    assert 'SECRET' not in str(r)
    assert len(f.calls)==3

def test_no_answer_is_not_counted_complete():
    f=Fake(['','',''])
    assert not complete(ReasoningAgent(f).solve('p',{})['final_response'])

def test_budget_and_no_cross_question_state():
    f=Fake(['FINAL_ANSWER: 1','FINAL_ANSWER: 2'])
    a=ReasoningAgent(f)
    a.solve('first',{}); a.solve('second',{})
    assert 'first' not in str(f.calls[1]['messages'])
    assert f.calls[0]['max_tokens']==4096

@pytest.mark.parametrize('s',[r'\boxed{\frac{1}{2}}','FINAL_ANSWER: 0','FINAL_ANSWER: No'])
def test_complete(s): assert complete(s)
@pytest.mark.parametrize('s',['','FINAL_ANSWER: <answer>',r'\boxed{\frac{1}{2}', '```python\nprint(1)\n```'])
def test_incomplete(s): assert not complete(s)
def test_reasoning_strip():
    assert visible('<think>draft</think>FINAL_ANSWER: 3')=='FINAL_ANSWER: 3'
    assert visible('<think>unfinished')==''

@pytest.mark.parametrize('code',["import os", "open('/etc/passwd')", "sp.sympify('1')", "().__class__", "getattr(sp,'sympify')('1')", "import sympy as _s"])
def test_reject(code):
    assert run_math(code)['ok'] is False

def test_sympy():
    r=run_math('import sympy as sp\nx=sp.symbols("x")\nprint(sp.integrate(x**2,(x,0,3)))')
    assert r['ok'] and r['stdout'].strip()=='9'

def test_import_star_and_matrix():
    r=run_math('from sympy import *\nprint(Matrix([[1,2],[3,4]]).det())')
    assert r['ok'] and r['stdout'].strip()=='-2'

def test_timeout():
    r=run_math('while True: pass', timeout=0.15)
    assert r['ok'] is False and r['error']=='Timeout'
