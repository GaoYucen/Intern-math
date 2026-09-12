"""Independent numerical/algebraic checks on flagged PUBLIC regression items.
No model calls or reference mutations. This does not certify all model proofs.
"""
from __future__ import annotations
import json,math
import sympy as sp

def bisect(fn,lo,hi):
    left,right=fn(lo),fn(hi)
    if left*right>=0: raise ValueError('Sign-changing continuous bracket required')
    for _ in range(70):
        mid=(lo+hi)/2;val=fn(mid)
        if val==0: return mid
        if left*val<0: hi=mid
        else: lo,left=mid,val
    return (lo+hi)/2

def verify():
    x=sp.symbols('x',real=True)
    quintic=bisect(lambda y:y**5-y+1,-1.5,-1)
    tangent=bisect(lambda y:math.tan(y)+math.tanh(y),2.3,2.4)
    ode=lambda t:8*math.cos(2*t)+64*math.sin(2*t)-788*math.exp(-t/4)
    crossing=bisect(ode,10.01,10.1)
    threshold=4*math.log(788/(8*math.sqrt(65)))
    assert 10.01<threshold<10.1 and ode(threshold)<0<ode(10.1)
    # Amplitude rules out every earlier crossing; derivative positive here.
    assert 6*math.pi<2*threshold<20.2<6*math.pi+math.atan(8)
    U=1/sp.cosh(3*x/sp.sqrt(2))
    reduced=sp.trigsimp(sp.diff(U,x,2)-sp.Rational(9,2)*U+9*U**3)
    assert sp.simplify(reduced)==0
    bad=1/sp.cosh(3*x)
    bad_residual=sp.simplify((sp.Rational(9,2)*sp.diff(bad,x,2)-9*sp.diff(bad**3,x,2)-sp.diff(bad,x,4)).subs(x,0))
    assert bad_residual!=0
    result={
        '47_probability':.5*math.erfc(2/math.sqrt(10)),
        '121_correct_reduced_residual':str(reduced),'121_r1_pde_residual_at_origin':str(bad_residual),
        '177_crossing':crossing,'177_first_possible_amplitude_bound':threshold,
        '177_residual_at_10_067':ode(10.067),
        '200_likelihood_ratio':250*math.log(.0049/.0036),
        '224_exact_A':str(-sp.pi/3),'241_Hoeffding_bound':2*math.exp(-1.8),
        '250_root':quintic,'250_gold_residual':1,'250_derivative_at_minus_1_5':5*(-1.5)**4-1,
        '263_first_positive_root':tangent,'272_cooling_time':math.log(13/8)/math.log(13/12),
        '279_entropy_bits':(3*math.log2(3)+4)/8,'281_flux':float(48*sp.pi/5),
        '170_counterexample_to_written_E':'f_n=g=0; f=1 at one point. A_k=E_k=empty, so the written construction omits the null set. The theorem is true, but this proof is not silently repaired.'}
    assert round(quintic,3)==-1.167 and round(tangent,2)==2.37 and round(crossing,3)==10.066
    return result

if __name__=='__main__':print(json.dumps(verify(),indent=2,ensure_ascii=False))
