import pytest
from math_tools import run_math
from r2_agent import ReasoningAgent

@pytest.mark.parametrize('code,expected',[
 ('def f(x):\n    """Harmless function documentation."""\n    return x*x\nprint(f(3))','9'),
 ('import mpmath as mp\nprint(0.5*mp.erfc(2/mp.sqrt(10)))','0.185546'),
 ('import numpy as np\na=np.array([[1,2],[2,3],[3,3],[4,4]])\nprint(np.trace(np.cov(a,rowvar=False)))','2.333333'),
 ('from scipy import stats\nprint(stats.t.ppf(0.975,10))','2.228138'),
 ('from scipy.optimize import brentq\nprint(brentq(lambda x:x*x-2,1,2))','1.414213'),
 ('import scipy.optimize as opt\nprint(opt.brentq(lambda x:x*x-2,1,2))','1.414213'),
 ('from scipy.stats import ttest_ind\nprint(ttest_ind([1,2,3],[2,3,4],equal_var=False).pvalue)','0.28786')])
def test_standard_functions(code,expected):
    result=run_math(code)
    assert result['ok'],result
    assert expected in result['stdout']

@pytest.mark.parametrize('code',['import scipy.io','import numpy.random','import mpmath as mp\nmp.mp.dps=100','import numpy as np\nnp.save("leak",[1])','import numpy as np\nnp.load("secret")','from scipy import stats\nprint(stats.__dict__)','print(dir())','import mpmath as mp\nprint(mp.findroot("x",1))'])
def test_no_io_or_introspection(code):assert not run_math(code)['ok']

def test_forced_closure_keeps_problem_and_requests_answer_first():
    class Fake:
        def __init__(self):self.calls=[]
        def chat(self,**kwargs):
            self.calls.append(kwargs)
            return 'unfinished' if len(self.calls)<3 else 'FINAL_ANSWER: 2'
    client=Fake();out=ReasoningAgent(client).solve('sqrt(4)',{})
    assert out['final_response']=='FINAL_ANSWER: 2'
    assert 'FIRST line' in client.calls[-1]['messages'][0]['content']
    assert client.calls[-1]['messages'][1]['content']=='sqrt(4)'
