import pytest
from math_tools import run_math

@pytest.mark.parametrize('code,expected',[
    ('x=1.23456789\nprint(f"x={x:.6f}")','x=1.234568'),
    ('from math import erfc,sqrt\nprint(0.5*erfc(2/sqrt(10)))','0.18554668476134878'),
    ('from sympy import symbols,nroots\nx=symbols("x")\nprint(nroots(x**2-4))','[-2.00000000000000, 2.00000000000000]'),
    ('for i in range(2):\n    print(f"{i:2d} {i/3:.5f}")','1 0.33333'),
])
def test_math_compat(code,expected):
    result=run_math(code)
    assert result['ok'],result
    assert expected in result['stdout']

@pytest.mark.parametrize('code',[
    'x=1.2\ns=f"{x:.3f}"\nprint(s)',
    'x=1.2\nprint(f"{x:{10}.3f}")',
    'x=1.2\nprint(f"{x:100000000f}")',
    'import os',
    'print(sp.sympify("1+1"))',
    'print(sp.lambdify(1,1))',
    'print(sp.nroots("x**2-4"))',
    'print=sp.solve\nprint(f"{2}")',
])
def test_safety_preserved(code):
    assert not run_math(code)['ok']
