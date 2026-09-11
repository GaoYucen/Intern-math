"""Small mathematical Python worker, with a restricted surface and resource limits.

This is defense in depth, not a replacement for the competition's OS sandbox.
No API credentials, filesystem operations, network modules or dynamic evaluation
functions are exposed to generated code. Each calculation has fresh state.
"""
from __future__ import annotations
import ast
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

SYMPY_NAMES = set('Symbol symbols Rational Integer Float Matrix ImmutableMatrix Eq Ne Lt Le Gt Ge And Or Not sqrt root real_root sin cos tan asin acos atan atan2 sinh cosh tanh exp log Abs sign floor ceiling factorial factorial2 binomial ff rf fibonacci lucas bell bernoulli harmonic gamma beta zeta pi E I oo N simplify expand factor cancel together apart collect solve linsolve nonlinsolve solveset reduce_inequalities diff integrate limit summation product residue series conjugate re im transpose det eye zeros ones diag gcd lcm mod_inverse isprime nextprime prevprime primerange factorint totient divisors divisor_count degree Poly resultant discriminant Function Derivative Integral Sum Product FiniteSet Interval Union Intersection Complement EmptySet Reals Integers Piecewise dsolve simplify_logic kronecker_symbol legendre_symbol'.split())
MATH_NAMES = set('sqrt isqrt gcd lcm factorial comb perm sin cos tan asin acos atan atan2 sinh cosh tanh exp log log2 log10 ceil floor fabs fsum prod pi e inf isclose'.split())
METHODS = set('subs diff integrate simplify expand factor cancel together apart collect evalf doit det inv eigenvals eigenvects charpoly nullspace rank rref LUsolve diagonalize trace transpose adjugate dot cross norm row col jacobian applyfunc as_real_imag as_numer_denom coeff all_coeffs degree factor_list count_ops has equals is_integer is_real is_positive is_negative is_zero free_symbols shape T rows cols numerator denominator append extend count index items keys values sort reverse copy'.split())
BUILTINS = set('abs all any bool dict enumerate float int len list map max min pow print range reversed round set sorted sum tuple zip'.split())
BANNED = set('open exec eval compile globals locals vars dir getattr setattr delattr hasattr type object super input help breakpoint memoryview __import__'.split())


def validate(code: str) -> ast.Module:
    if not isinstance(code, str) or len(code) > 10000:
        raise ValueError('code size limit')
    tree = ast.parse(code)
    if len(list(ast.walk(tree))) > 2500:
        raise ValueError('code complexity limit')
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    reserved = {'print', 'symbols', 'Symbol', 'Function', 'Integer', 'Float', 'Rational', 'Fraction'}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id in reserved:
            raise ValueError('cannot rebind protected function')
        if isinstance(n, (ast.FunctionDef, ast.arg)) and getattr(n, 'name', getattr(n, 'arg', None)) in reserved:
            raise ValueError('cannot shadow protected function')
        if isinstance(n, ast.alias) and n.asname in reserved and n.asname != n.name:
            raise ValueError('cannot alias protected function')
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store):
            raise ValueError('cannot mutate attributes')
        if isinstance(n, ast.JoinedStr):
            parent = parents.get(n)
            if not (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id == 'print'):
                raise ValueError('formatted strings only allowed in print')
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            parent = parents.get(n)
            # SymPy implicitly parses strings in many functions. Permit strings
            # only as safe constructor names/numbers or direct print labels.
            target = ''
            if isinstance(parent, ast.Call):
                target = getattr(parent.func, 'id', getattr(parent.func, 'attr', ''))
            elif isinstance(parent, ast.keyword):
                if '__' not in n.value and re.fullmatch(r'[A-Za-z0-9_ +:,-]*', n.value):
                    continue
            elif isinstance(parent, ast.JoinedStr):
                continue
            if target == 'print':
                pass
            elif target in {'symbols', 'Symbol', 'Function'}:
                if '__' in n.value or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_ ,:]*', n.value):
                    raise ValueError('unsafe symbol name')
            elif target in {'Integer', 'Float', 'Rational', 'Fraction'}:
                if not re.fullmatch(r'[0-9eE.+/ -]+', n.value):
                    raise ValueError('unsafe numeric literal')
            else:
                raise ValueError('expression strings are unsupported; construct expressions directly')
        if isinstance(n, (ast.ClassDef, ast.With, ast.AsyncWith, ast.AsyncFunctionDef,
                          ast.Await, ast.Global, ast.Nonlocal, ast.Delete)):
            raise ValueError('unsupported statement')
        if isinstance(n, ast.Name) and (n.id.startswith('_') or n.id in BANNED):
            raise ValueError('forbidden name')
        if isinstance(n, ast.Attribute):
            if n.attr.startswith('_') or n.attr not in SYMPY_NAMES | MATH_NAMES | METHODS | {'Fraction','combinations','permutations','combinations_with_replacement','accumulate','chain','islice','repeat'}:
                raise ValueError('unsupported attribute')
        if isinstance(n, ast.Import):
            if any(a.name not in {'sympy','math','itertools','fractions'} for a in n.names):
                raise ValueError('unsupported import')
        if isinstance(n, ast.ImportFrom):
            allowed = {'sympy': SYMPY_NAMES, 'math': MATH_NAMES,
                       'fractions': {'Fraction'}, 'itertools': {'combinations','permutations','product','combinations_with_replacement','accumulate','chain','islice','repeat'}}
            if n.level or n.module not in allowed or any(a.name != '*' and a.name not in allowed[n.module] for a in n.names):
                raise ValueError('unsupported import')
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) > 4000:
            raise ValueError('string limit')
        if isinstance(n, ast.alias) and n.asname and n.asname.startswith('_'):
            raise ValueError('forbidden alias')
    return tree


def run_math(code: str, timeout: float = 12) -> dict:
    try:
        validate(code)
    except Exception as exc:
        return {'ok': False, 'error': type(exc).__name__, 'detail': str(exc)[:120]}
    env = {'PATH': os.defpath, 'PYTHONIOENCODING': 'utf-8', 'OPENBLAS_NUM_THREADS': '1',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    try:
        with tempfile.TemporaryDirectory(prefix='math_calc_') as work:
            process = subprocess.Popen([sys.executable, '-I', str(Path(__file__).resolve()), '--worker'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, cwd=work, env=env, start_new_session=True)
            try:
                out, _ = process.communicate(code, timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                return {'ok': False, 'error': 'Timeout'}
            if process.returncode != 0:
                return {'ok': False, 'error': 'WorkerFailure'}
            return json.loads(out)
    except Exception as exc:
        return {'ok': False, 'error': type(exc).__name__}


def _worker() -> None:
    import builtins
    import fractions
    import itertools
    import math
    import resource
    from types import SimpleNamespace
    resource.setrlimit(resource.RLIMIT_CPU, (8, 9))
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024**2, 768 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    import sympy
    def facade(module, names):
        return SimpleNamespace(**{k: getattr(module,k) for k in names if hasattr(module,k)})
    modules = {'sympy': facade(sympy,SYMPY_NAMES), 'math': facade(math,MATH_NAMES),
        'fractions': facade(fractions,{'Fraction'}),
        'itertools': facade(itertools,{'combinations','permutations','product','combinations_with_replacement','accumulate','chain','islice','repeat'})}
    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in modules:
            raise ValueError('unsupported import')
        return modules[name]
    output = []
    used = 0
    def bounded_print(*args, sep=' ', end='\n', **kwargs):
        nonlocal used
        text = sep.join(map(str,args)) + end
        used += len(text)
        if used > 12000:
            raise ValueError('output limit')
        output.append(text)
    safe = {k: getattr(builtins,k) for k in BUILTINS}
    safe['print'] = bounded_print
    safe['__import__'] = restricted_import
    scope = {'__builtins__': safe, 'sp': modules['sympy'], 'sympy': modules['sympy'],
             'math': modules['math'], **vars(modules['sympy'])}
    try:
        code = sys.stdin.read(10001)
        tree = validate(code)
        exec(compile(tree,'<math calculation>','exec'),scope,scope)
        result = {'ok': True, 'stdout': ''.join(output) or '(no printed result)'}
    except Exception as exc:
        result = {'ok': False, 'error': type(exc).__name__, 'detail': str(exc)[:200]}
    sys.stdout.write(json.dumps(result,ensure_ascii=False))

if __name__ == '__main__':
    _worker()
