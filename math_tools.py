"""Restricted mathematical worker; defense in depth, not an OS sandbox.
Generated code receives selected calculation APIs, no credentials, files or network.
"""
from __future__ import annotations
import ast,json,os,re,signal,subprocess,sys,tempfile
from pathlib import Path
SYMPY_NAMES=set('erf erfc nroots Symbol symbols Rational Integer Float Matrix ImmutableMatrix Eq Ne Lt Le Gt Ge And Or Not sqrt root real_root sin cos tan asin acos atan atan2 sinh cosh tanh exp log Abs sign floor ceiling factorial factorial2 binomial ff rf fibonacci lucas bell bernoulli harmonic gamma beta zeta pi E I oo N simplify expand factor cancel together apart collect solve linsolve nonlinsolve solveset reduce_inequalities diff integrate limit summation product residue series conjugate re im transpose det eye zeros ones diag gcd lcm mod_inverse isprime nextprime prevprime primerange factorint totient divisors divisor_count degree Poly resultant discriminant Function Derivative Integral Sum Product FiniteSet Interval Union Intersection Complement EmptySet Reals Integers Piecewise dsolve simplify_logic kronecker_symbol legendre_symbol'.split())
MATH_NAMES=set('erf erfc expm1 log1p sqrt isqrt gcd lcm factorial comb perm sin cos tan asin acos atan atan2 sinh cosh tanh exp log log2 log10 ceil floor fabs fsum prod pi e inf isclose'.split())
METHODS=set('subs diff integrate simplify expand factor cancel together apart collect evalf doit det inv eigenvals eigenvects charpoly nullspace rank rref LUsolve diagonalize trace transpose adjugate dot cross norm row col jacobian applyfunc as_real_imag as_numer_denom coeff all_coeffs degree factor_list count_ops has equals is_integer is_real is_positive is_negative is_zero free_symbols shape T rows cols numerator denominator append extend count index items keys values sort reverse copy'.split())
BUILTINS=set('abs all any bool dict enumerate float int len list map max min pow print range reversed round set sorted sum tuple zip'.split())
NUMPY_NAMES=set('array asarray mean var std cov trace sum sqrt log exp sin cos tan tanh arange linspace hstack vstack concatenate eye zeros ones diag pi abs dot prod min max linalg'.split())
SCIPY_STATS={'ttest_ind','t','norm','chi2'}
SCIPY_OPTIMIZE={'brentq','bisect','newton'}
MPMATH_NAMES=set('sqrt exp log sin cos tan tanh erf erfc pi findroot quad betainc gamma'.split())
EXTRA_NAMES=NUMPY_NAMES|SCIPY_STATS|SCIPY_OPTIMIZE|MPMATH_NAMES|set('stats optimize cdf sf ppf pdf eigvals eigvalsh eig solve det statistic pvalue df converged root'.split())
BANNED=set('open exec eval compile globals locals vars dir getattr setattr delattr hasattr type object super input help breakpoint memoryview __import__'.split())
IT_NAMES={'combinations','permutations','product','combinations_with_replacement','accumulate','chain','islice','repeat'}
IMPORT_NAMES={'numpy':NUMPY_NAMES,'numpy.linalg':{'eigvals','eigvalsh','eig','solve','det'},'scipy':{'stats','optimize'},'scipy.stats':SCIPY_STATS,'scipy.optimize':SCIPY_OPTIMIZE,'mpmath':MPMATH_NAMES,'sympy':SYMPY_NAMES,'math':MATH_NAMES,'fractions':{'Fraction'},'itertools':IT_NAMES}

def validate(code: str) -> ast.Module:
    if not isinstance(code,str) or len(code)>10000: raise ValueError('code size limit')
    tree=ast.parse(code)
    if len(list(ast.walk(tree)))>2500: raise ValueError('code complexity limit')
    parents={child:parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    reserved={'print','symbols','Symbol','Function','Integer','Float','Rational','Fraction'}
    for n in ast.walk(tree):
        if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store) and n.id in reserved: raise ValueError('cannot rebind protected function')
        if isinstance(n,(ast.FunctionDef,ast.arg)) and getattr(n,'name',getattr(n,'arg',None)) in reserved: raise ValueError('cannot shadow protected function')
        if isinstance(n,ast.alias) and n.asname in reserved and n.asname!=n.name: raise ValueError('cannot alias protected function')
        if isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Store): raise ValueError('cannot mutate attributes')
        if isinstance(n,ast.JoinedStr):
            parent=parents.get(n);root=n
            if isinstance(parent,ast.FormattedValue) and parent.format_spec is n:
                if not all(isinstance(v,ast.Constant) and isinstance(v.value,str) for v in n.values): raise ValueError('dynamic format specs are unsupported')
                spec=''.join(v.value for v in n.values)
                if not re.fullmatch(r'[+ -]?(?:[0-9]{1,2})?(?:\.[0-9]{1,2})?[eEfFgGd%]?',spec): raise ValueError('unsupported numeric format spec')
                root=parents.get(parent);parent=parents.get(root)
            if not (isinstance(root,ast.JoinedStr) and isinstance(parent,ast.Call) and isinstance(parent.func,ast.Name) and parent.func.id=='print'): raise ValueError('formatted strings only allowed directly in print')
        if isinstance(n,ast.Constant) and isinstance(n.value,str):
            if len(n.value)>4000: raise ValueError('string limit')
            parent=parents.get(n);target=''
            if isinstance(parent,ast.Call): target=getattr(parent.func,'id',getattr(parent.func,'attr',''))
            elif isinstance(parent,ast.keyword):
                if '__' not in n.value and re.fullmatch(r'[A-Za-z0-9_ +:,-]*',n.value): continue
            elif isinstance(parent,ast.JoinedStr): continue
            if isinstance(parent,ast.Expr):
                owner=parents.get(parent)
                if isinstance(owner,(ast.Module,ast.FunctionDef)) and owner.body and owner.body[0] is parent:
                    if len(n.value)>2000: raise ValueError('docstring limit')
                    continue
            if target=='print': pass
            elif target in {'symbols','Symbol','Function'}:
                if '__' in n.value or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_ ,:]*',n.value): raise ValueError('unsafe symbol name')
            elif target in {'Integer','Float','Rational','Fraction'}:
                if not re.fullmatch(r'[0-9eE.+/ -]+',n.value): raise ValueError('unsafe numeric literal')
            else: raise ValueError('expression strings are unsupported; construct expressions directly')
        if isinstance(n,(ast.ClassDef,ast.With,ast.AsyncWith,ast.AsyncFunctionDef,ast.Await,ast.Global,ast.Nonlocal,ast.Delete)): raise ValueError('unsupported statement')
        if isinstance(n,ast.Name) and (n.id.startswith('_') or n.id in BANNED): raise ValueError('forbidden name')
        if isinstance(n,ast.Attribute):
            if n.attr.startswith('_') or n.attr not in SYMPY_NAMES|MATH_NAMES|METHODS|EXTRA_NAMES|IT_NAMES|{'Fraction'}: raise ValueError('unsupported attribute')
        if isinstance(n,ast.Import) and any(a.name not in IMPORT_NAMES for a in n.names): raise ValueError('unsupported import')
        if isinstance(n,ast.ImportFrom):
            if n.level or n.module not in IMPORT_NAMES or any(a.name!='*' and a.name not in IMPORT_NAMES[n.module] for a in n.names): raise ValueError('unsupported import')
        if isinstance(n,ast.alias) and n.asname and n.asname.startswith('_'): raise ValueError('forbidden alias')
    return tree

def run_math(code: str, timeout: float=12) -> dict:
    try: validate(code)
    except Exception as exc: return {'ok':False,'error':type(exc).__name__,'detail':str(exc)[:120]}
    env={'PATH':os.defpath,'PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    try:
        with tempfile.TemporaryDirectory(prefix='math_calc_') as work:
            proc=subprocess.Popen([sys.executable,'-I',str(Path(__file__).resolve()),'--worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,cwd=work,env=env,start_new_session=True)
            try: out,_=proc.communicate(code,timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.communicate()
                return {'ok':False,'error':'Timeout'}
            if proc.returncode!=0: return {'ok':False,'error':'WorkerFailure'}
            return json.loads(out)
    except Exception as exc: return {'ok':False,'error':type(exc).__name__}

def _worker():
    import builtins,fractions,itertools,math,resource
    from types import SimpleNamespace
    resource.setrlimit(resource.RLIMIT_CPU,(8,9))
    resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,768*1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
    resource.setrlimit(resource.RLIMIT_NOFILE,(32,32))
    import sympy
    import numpy as np
    import scipy.stats as stats
    import scipy.optimize as optimize
    import mpmath
    def facade(module,names): return SimpleNamespace(**{k:getattr(module,k) for k in names if hasattr(module,k)})
    np_api=facade(np,NUMPY_NAMES-{'linalg'});np_api.linalg=facade(np.linalg,IMPORT_NAMES['numpy.linalg'])
    stats_api=facade(stats,{'ttest_ind'})
    for name in ['t','norm','chi2']: setattr(stats_api,name,facade(getattr(stats,name),{'cdf','sf','ppf','pdf'}))
    optimize_api=facade(optimize,SCIPY_OPTIMIZE)
    scipy_api=SimpleNamespace(stats=stats_api,optimize=optimize_api)
    modules={'sympy':facade(sympy,SYMPY_NAMES),'math':facade(math,MATH_NAMES),'fractions':facade(fractions,{'Fraction'}),'itertools':facade(itertools,IT_NAMES),'numpy':np_api,'numpy.linalg':np_api.linalg,'scipy':scipy_api,'scipy.stats':stats_api,'scipy.optimize':optimize_api,'mpmath':facade(mpmath,MPMATH_NAMES)}
    def restricted_import(name,globals=None,locals=None,fromlist=(),level=0):
        if level or name not in modules: raise ValueError('unsupported import')
        return modules[name] if fromlist or '.' not in name else modules[name.split('.')[0]]
    output=[];used=0
    def bounded_print(*args,sep=' ',end='\n',**kwargs):
        nonlocal used
        text=sep.join(map(str,args))+end;used+=len(text)
        if used>12000: raise ValueError('output limit')
        output.append(text)
    safe={k:getattr(builtins,k) for k in BUILTINS};safe['print']=bounded_print;safe['__import__']=restricted_import
    scope={'__builtins__':safe,'sp':modules['sympy'],'sympy':modules['sympy'],'math':modules['math'],**vars(modules['sympy'])}
    try:
        tree=validate(sys.stdin.read(10001))
        exec(compile(tree,'<math calculation>','exec'),scope,scope)
        result={'ok':True,'stdout':''.join(output) or '(no printed result)'}
    except Exception as exc: result={'ok':False,'error':type(exc).__name__,'detail':str(exc)[:200]}
    sys.stdout.write(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':_worker()
