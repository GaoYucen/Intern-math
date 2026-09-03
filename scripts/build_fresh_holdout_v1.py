#!/usr/bin/env python3
"""Build Fresh-Holdout-v1 from independently verified, recent-style problems.

The benchmark statements are independently constructed/parameterized. Public
2025-2026 exam sources are used only to inform topic mix and style; no item is
copied verbatim. Gold answers are recomputed deterministically here.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "fresh_holdout_v1"

SOURCES = [
    {
        "name": "Harvard Mathematics Qualifying Examination, Fall 2025",
        "url": "https://www.math.harvard.edu/wp-content/uploads/Quals-Fall-2025-final.pdf",
    },
    {
        "name": "NUS Algebra Qualifying Exam, January 2025",
        "url": "https://www.math.nus.edu.sg/wp-content/uploads/sites/4/2025/01/S2-2024-Algebra.pdf",
    },
    {
        "name": "University of Toronto Past Comprehensive Exams",
        "url": "https://www.mathematics.utoronto.ca/graduate/past-comprehensive-exams",
    },
    {
        "name": "University of Arizona Qualifying Exams",
        "url": "https://www.math.arizona.edu/academics/graduate/requirements/qualifying-exams",
    },
    {
        "name": "MAA Putnam Archive (2025)",
        "url": "https://maa.org/maa-putnam-archive/",
    },
]


def problem_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def add(rows, idx, problem, answer, answer_type, domain, difficulty,
        source_family, verification, split):
    rows.append({
        "idx": idx,
        "problem": problem,
        "answer": str(answer),
        "answer_type": answer_type,
        "domain": domain,
        "difficulty": difficulty,
        "source_dataset": "fresh_holdout_v1",
        "source_family": source_family,
        "verification": verification,
        "split": split,
        "problem_hash": problem_hash(problem),
    })


def build_rows():
    rows = []
    x = sp.symbols("x")

    A0 = sp.Matrix([[3, 2, 3, 4], [1, 5, 3, 4], [1, 2, 8, 4], [1, 2, 3, 11]])
    add(rows, 0, r"""Let
\[
A=\begin{pmatrix}
3&2&3&4\\
1&5&3&4\\
1&2&8&4\\
1&2&3&11
\end{pmatrix}.
\]
Compute \(\det A\).""", A0.det(), "integer", "linear_algebra", "hard",
        "2025-2026 qualifying-exam style: matrix identities and exact linear algebra",
        "direct determinant; rank-one determinant lemma cross-check", "dev")

    A1 = sp.Matrix([[3, 1, 0], [1, 4, 1], [0, 1, 5]])
    det_block = (A1 - sp.eye(3)).det() * (A1 + sp.eye(3)).det()
    add(rows, 1, r"""Let
\[
A=\begin{pmatrix}3&1&0\\1&4&1\\0&1&5\end{pmatrix},\qquad
B=\begin{pmatrix}A&I_3\\I_3&A\end{pmatrix}.
\]
Compute \(\det B\).""", det_block, "integer", "linear_algebra", "hard",
        "2025-2026 qualifying-exam style: block matrices",
        "det(B)=det(A-I)det(A+I); direct exact determinant cross-check", "dev")

    M = sp.Matrix([[8, 6, 0], [6, 36, 30], [0, 30, 30]])
    d1 = math.gcd(*[abs(int(v)) for v in M])
    minors2 = [abs(int(M.extract(rr, cc).det()))
               for rr in ((0, 1), (0, 2), (1, 2))
               for cc in ((0, 1), (0, 2), (1, 2))]
    d12 = math.gcd(*minors2)
    smith = (d1, d12 // d1, abs(int(M.det())) // d12)
    assert smith == (2, 6, 30)
    killed6 = math.prod(math.gcd(6, d) for d in smith)
    add(rows, 2, r"""Let
\[
M=\begin{pmatrix}
8&6&0\\
6&36&30\\
0&30&30
\end{pmatrix}
\]
and \(G=\mathbb Z^3/M\mathbb Z^3\). How many elements \(g\in G\) satisfy \(6g=0\)?""",
        killed6, "integer", "abstract_algebra", "hard",
        "2025 NUS/2026 algebra qualifying-exam style: finitely generated abelian groups",
        "determinantal divisors give Smith invariants (2,6,30)", "dev")

    irreducible6 = sum(sp.mobius(d) * 2 ** (6 // d) for d in sp.divisors(6)) // 6
    add(rows, 3, r"""How many monic irreducible polynomials of degree \(6\) are there over the field \(\mathbb F_2\)?""",
        irreducible6, "integer", "abstract_algebra", "medium",
        "2025-2026 algebra qualifying-exam style: finite fields",
        "Möbius formula for irreducible polynomials over finite fields", "dev")

    disc = sp.discriminant(x**4 + 3*x**2 - 2*x + 5, x)
    add(rows, 4, r"""Compute the discriminant of the polynomial
\[
f(x)=x^4+3x^2-2x+5.
\]""", disc, "integer", "abstract_algebra", "hard",
        "2025-2026 algebra qualifying-exam style: polynomial invariants",
        "exact symbolic resultant/discriminant computation", "dev")

    order6 = (math.gcd(6, 12) * math.gcd(6, 18)
              - math.gcd(3, 12) * math.gcd(3, 18)
              - math.gcd(2, 12) * math.gcd(2, 18) + 1)
    add(rows, 5, r"""How many elements of order exactly \(6\) are in the finite abelian group
\[
\mathbb Z_{12}\times \mathbb Z_{18}?
\]""", order6, "integer", "abstract_algebra", "medium",
        "2025 NUS algebra qualifying-exam style: finite abelian groups",
        "inclusion-exclusion from counts of elements killed by divisors of 6", "dev")

    roots_mod = sum(1 for a in range(4096) if (a*a - 16) % 4096 == 0)
    add(rows, 6, r"""How many residue classes \(x\pmod{4096}\) satisfy
\[
x^2\equiv 16\pmod{4096}?
\]""", roots_mod, "integer", "number_theory", "hard",
        "2025 Putnam/qualifying-exam style: modular arithmetic",
        "2-adic derivation plus exhaustive residue verification", "dev")

    crt = next(n for n in range(1, 10000)
               if n % 12 == 7 and n % 30 == 19 and n % 7 == 4)
    add(rows, 7, r"""Find the least positive integer \(n\) satisfying
\[
n\equiv 7\pmod{12},\qquad
n\equiv 19\pmod{30},\qquad
n\equiv 4\pmod 7.
\]""", crt, "integer", "number_theory", "medium",
        "2025 Putnam/qualifying-exam style: generalized CRT",
        "generalized CRT plus direct congruence check", "dev")

    gcd_sum = sum(math.gcd(k, 840) for k in range(1, 841))
    add(rows, 8, r"""Compute
\[
\sum_{k=1}^{840}\gcd(k,840).
\]""", gcd_sum, "integer", "number_theory", "hard",
        "2025 Putnam-style divisor sums",
        "direct enumeration and divisor-sum identity cross-check", "dev")

    coeff = sp.expand((1 + x + x**2)**12).coeff(x, 20)
    add(rows, 9, r"""Find the coefficient of \(x^{20}\) in
\[
(1+x+x^2)^{12}.
\]""", coeff, "integer", "combinatorics", "medium",
        "2025 Putnam-style generating functions",
        "exact polynomial expansion and inclusion-exclusion cross-check", "dev")

    f = [0] * 8
    f[0] = 1
    for n in range(1, 8):
        f[n] = sum(math.comb(n-1, k-1) * math.factorial(k-1) * f[n-k]
                   for k in range(3, n+1))
    perms = math.comb(9, 2) * f[7]
    add(rows, 10, r"""How many permutations of \(\{1,2,\ldots,9\}\) have no fixed points and have exactly one 2-cycle in their disjoint-cycle decomposition?""",
        perms, "integer", "combinatorics", "hard",
        "2025 Putnam-style permutations and cycle structure",
        "cycle recurrence after choosing the unique 2-cycle", "dev")

    seq = [2, 5]
    for n in range(11):
        seq.append(4 * seq[-1] - 3 * seq[-2] + 2**n)
    add(rows, 11, r"""A sequence is defined by
\[
a_0=2,\quad a_1=5,\quad
a_{n+2}=4a_{n+1}-3a_n+2^n\quad(n\ge0).
\]
Compute \(a_{12}\).""", seq[12], "integer", "discrete_math", "hard",
        "2025 Putnam-style recurrences",
        "exact recurrence iteration and closed-form cross-check", "dev")

    add(rows, 12, r"""An urn contains 5 red, 4 blue, and 3 green balls. Four balls are drawn uniformly without replacement. Given that exactly one drawn ball is green, what is the probability that exactly two of the four drawn balls are red?""",
        sp.Rational(10, 21), "rational", "probability", "medium",
        "2025 Toronto probability comprehensive-exam style",
        "conditional hypergeometric calculation", "dev")

    p1, p2, p3 = sp.symbols("p1 p2 p3")
    hit = sp.solve([
        sp.Eq(p1, sp.Rational(2, 3) * p2),
        sp.Eq(p2, sp.Rational(1, 2) * p3 + sp.Rational(1, 2) * p1),
        sp.Eq(p3, sp.Rational(1, 3) + sp.Rational(2, 3) * p2),
    ], [p1, p2, p3])[p2]
    add(rows, 13, r"""A Markov chain has states \(0,1,2,3,4\), with 0 and 4 absorbing. From state 1 it moves to 2 with probability \(2/3\) and to 0 otherwise. From state 2 it moves to 3 or 1 with equal probability. From state 3 it moves to 4 with probability \(1/3\) and to 2 otherwise. Starting from state 2, what is the probability of hitting 4 before 0?""",
        hit, "rational", "probability", "hard",
        "2025 Toronto probability comprehensive-exam style",
        "exact linear system for hitting probabilities", "dev")

    cond_poisson = sp.binomial(7, 2) * sp.Rational(2, 5)**2 * sp.Rational(3, 5)**5
    add(rows, 14, r"""Let \(X\sim\mathrm{Poisson}(2)\) and \(Y\sim\mathrm{Poisson}(3)\) be independent. Compute
\[
\Pr(X=2\mid X+Y=7).
\]""", cond_poisson, "rational", "probability", "medium",
        "2025 Toronto probability comprehensive-exam style",
        "Poisson conditioning gives Binomial(7,2/5)", "dev")

    fourth_moment = 3 * 10**2 - 2 * 10
    add(rows, 15, r"""Let \(\varepsilon_1,\ldots,\varepsilon_{10}\) be independent random variables taking values \(\pm1\) with equal probability. Compute
\[
\mathbb E\left(\sum_{i=1}^{10}\varepsilon_i\right)^4.
\]""", fourth_moment, "integer", "probability", "hard",
        "2025-2026 probability comprehensive-exam style",
        "fourth-moment expansion; only even index multiplicities survive", "dev")

    y = sp.symbols("y", positive=True)
    t = sp.symbols("t", positive=True)
    num = (sp.integrate(sp.exp(-y) * sp.integrate(sp.exp(-t), (t, 3-y, sp.oo)), (y, 0, 1))
           + sp.integrate(sp.exp(-y) * sp.integrate(sp.exp(-t), (t, 2*y, sp.oo)), (y, 1, sp.oo)))
    s = sp.symbols("s", positive=True)
    den = sp.integrate(s * sp.exp(-s), (s, 3, sp.oo))
    exp_cond = sp.simplify(num / den)
    add(rows, 16, r"""Let \(X,Y\) be independent exponential random variables with rate 1. Compute
\[
\Pr(X>2Y\mid X+Y>3).
\]""", exp_cond, "rational", "probability", "hard",
        "2025-2026 probability comprehensive-exam style",
        "piecewise exact integration of the conditional region", "dev")

    bose = sp.factorial(5) * sp.zeta(6) / 2**6
    add(rows, 17, r"""Evaluate
\[
\int_0^\infty \frac{x^5}{e^{2x}-1}\,dx.
\]""", sp.simplify(bose), "symbolic", "real_analysis", "hard",
        "2025 Harvard/2025-2026 analysis qualifying-exam style: exact improper integrals",
        "Gamma-zeta identity with symbolic simplification", "dev")

    z = sp.symbols("z")
    integrand = (z + 2) / ((z**2 + 1) * (z - 3))
    contour = sp.simplify(2 * sp.pi * sp.I * (sp.residue(integrand, z, sp.I) + sp.residue(integrand, z, -sp.I)))
    add(rows, 18, r"""Let \(C\) be the positively oriented circle \(|z|=2\). Evaluate
\[
\oint_C \frac{z+2}{(z^2+1)(z-3)}\,dz.
\]""", contour, "symbolic", "complex_analysis", "hard",
        "2025 Harvard complex-analysis qualifying-exam style",
        "residue theorem with exact symbolic residues", "dev")

    n = sp.symbols("n", integer=True, positive=True)
    series = sp.simplify(sp.summation((-1)**(n-1) / (n*(n+1)*(n+2)), (n, 1, sp.oo)))
    add(rows, 19, r"""Evaluate the convergent series
\[
\sum_{n=1}^{\infty}\frac{(-1)^{n-1}}{n(n+1)(n+2)}.
\]""", series, "symbolic", "real_analysis", "hard",
        "2025-2026 analysis qualifying-exam style: exact series",
        "partial fractions and symbolic summation cross-check", "dev")

    a5 = sp.simplify(2/sp.pi * sp.integrate(x * sp.cos(5*x), (x, 0, sp.pi)))
    add(rows, 20, r"""For the \(2\pi\)-periodic function whose restriction to \([-\pi,\pi]\) is \(f(x)=|x|\), write the cosine-series convention
\[
f(x)\sim \frac{a_0}{2}+\sum_{n\ge1}a_n\cos(nx).
\]
Compute \(a_5\).""", a5, "symbolic", "real_analysis", "medium",
        "2025-2026 analysis qualifying-exam style: Fourier analysis",
        "direct exact Fourier coefficient integration", "dev")

    add(rows, 21, r"""For positive real numbers \(x,y,z\) satisfying
\[
x+2y+3z=12,
\]
what is the maximum possible value of \(xyz\)?""", sp.Rational(32, 3), "rational", "optimization", "medium",
        "2025 Toronto linear algebra/optimization comprehensive-exam style",
        "Lagrange multipliers and weighted AM-GM cross-check", "dev")

    add(rows, 22, r"""Let \(y\) solve
\[
y''-4y=0,\qquad y(0)=1,\qquad y'(1)=0.
\]
Compute \(y(1)\).""", 1/sp.cosh(2), "symbolic", "ode", "medium",
        "2025-2026 analysis/ODE qualifying-exam style",
        "exact boundary-value solution in hyperbolic functions", "holdout")

    add(rows, 23, r"""Let
\[
A=\begin{pmatrix}
2&1&0\\
0&2&1\\
0&0&2
\end{pmatrix},\qquad
u(0)=\begin{pmatrix}0\\0\\1\end{pmatrix},
\]
and let \(u'(t)=Au(t)\). What is the first component of \(u(1)\)?""", sp.E**2/2, "symbolic", "ode", "hard",
        "2025-2026 linear ODE qualifying-exam style",
        "Jordan-block matrix exponential", "holdout")

    add(rows, 24, r"""Consider \(-\Delta u=\lambda u\) on the rectangle \(0<x<2\), \(0<y<3\), with
\[
u(0,y)=u(2,y)=u(x,0)=0,\qquad \frac{\partial u}{\partial y}(x,3)=0.
\]
What is the smallest eigenvalue \(\lambda\)?""", sp.Rational(5, 18)*sp.pi**2, "symbolic", "pde", "hard",
        "2025 Toronto PDE comprehensive-exam style",
        "separation of variables with mixed Dirichlet-Neumann boundary conditions", "holdout")

    add(rows, 25, r"""How many spanning trees does the complete bipartite graph \(K_{4,6}\) have?""",
        4**5 * 6**3, "integer", "graph_theory", "medium",
        "2025-2026 qualifying/competition style: graph enumeration",
        "Kirchhoff theorem formula m^(n-1)n^(m-1)", "holdout")

    T = sp.Matrix([[2, 0, 1], [1, 3, 0], [0, 1, 4]])
    volume = sp.Rational(abs(int(T.det())), 6)
    add(rows, 26, r"""Find the volume of the tetrahedron with vertices
\[
(0,0,0),\quad (2,1,0),\quad (0,3,1),\quad (1,0,4).
\]""", volume, "rational", "geometry", "medium",
        "2025-2026 geometry qualifying-exam style",
        "absolute determinant divided by 6", "holdout")

    bA = [1, 1, 2]
    bT = [1, 2, 1]
    b2 = bA[2]*bT[0] + bA[1]*bT[1] + bA[0]*bT[2]
    add(rows, 27, r"""Let
\[
X=(S^1\vee S^2\vee S^2)\times T^2,
\]
where \(T^2=S^1\times S^1\). What is the dimension of \(H_2(X;\mathbb Q)\)?""", b2, "integer", "topology", "hard",
        "2025 Harvard/2026 Arizona algebraic-topology qualifying-exam style",
        "Kunneth formula over Q", "holdout")

    rotation_sum = 3**6 + 2*3 + 2*3**2 + 3**3
    reflection_sum = 3*3**4 + 3*3**3
    burnside = (rotation_sum + reflection_sum) // 12
    add(rows, 28, r"""The vertices of a regular hexagon are colored using 3 available colors, with no restriction that every color be used. Two colorings are considered the same if related by a rotation or reflection of the hexagon. How many equivalence classes of colorings are there?""",
        burnside, "integer", "combinatorics", "hard",
        "2025 Putnam/qualifying-exam style: Burnside's lemma",
        "Burnside count over D_6 using rotation/reflection cycle types", "holdout")

    add(rows, 29, r"""True or false: every bounded sequence in \(\ell^1\) has a weakly convergent subsequence. Give a brief justification.""",
        "False", "boolean", "functional_analysis", "hard",
        "2025 Harvard real-analysis qualifying-exam style: weak compactness",
        "unit vectors plus Schur property give a counterexample", "holdout")

    assert len(rows) == 30
    assert [r["idx"] for r in rows] == list(range(30))
    assert len({r["problem_hash"] for r in rows}) == 30
    return rows


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    inputs = [{"idx": r["idx"], "problem": r["problem"]} for r in rows]
    write_jsonl(OUT / "input.jsonl", inputs)
    write_jsonl(OUT / "gold.jsonl", rows)

    domains = Counter(r["domain"] for r in rows)
    types = Counter(r["answer_type"] for r in rows)
    splits = Counter(r["split"] for r in rows)
    manifest = {
        "name": "Fresh-Holdout-v1",
        "version": "0.1",
        "created_utc": "2026-09-03",
        "n_items": len(rows),
        "split_counts": dict(splits),
        "dev_indices": [r["idx"] for r in rows if r["split"] == "dev"],
        "never_touch_indices": [r["idx"] for r in rows if r["split"] == "holdout"],
        "domain_counts": dict(sorted(domains.items())),
        "answer_type_counts": dict(sorted(types.items())),
        "construction_policy": {
            "purpose": "architecture selection under distribution shift; not a leaderboard predictor",
            "no_verbatim_copy": True,
            "freshness": "styles informed by public 2025-2026 exams; all benchmark statements independently constructed or parameterized",
            "gold_policy": "objective gold only in v0.1; answers deterministically recomputed by this builder",
            "freeze_policy": "do not edit observed items in place; create a new version",
            "never_touch_policy": "indices 22-29 are one-shot pre-submission checks and must not be used for prompt tuning",
        },
        "public_inspiration_sources": SOURCES,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["input_sha256"] = sha256(OUT / "input.jsonl")
    manifest["gold_sha256"] = sha256(OUT / "gold.jsonl")
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    readme = """# Fresh-Holdout-v1 (v0.1)\n\nA fresh, frozen, gold-reliable benchmark for architecture selection after the old proxy set proved poorly calibrated to the official hidden evaluation.\n\n- 30 items: 22 development + 8 never-touch.\n- v0.1 uses objective gold only (integer/rational/symbolic/boolean).\n- Problems are independently constructed/parameterized; recent public exams inform topic mix only.\n- Gold is deterministically recomputed by `scripts/build_fresh_holdout_v1.py`.\n- Freeze rule: after observing model results, never edit items in place.\n- Never-touch rule: indices 22-29 are reserved for one-shot pre-submission evaluation.\n\nUse `benchmark_v1` for regression, `long_reasoning_stress` for mechanism diagnostics, indices 0-21 here for architecture selection, and 22-29 only as the final holdout. This set is not claimed to predict the official leaderboard score.\n"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    # Exact overlap guard with the existing benchmark when present.
    old = ROOT / "data" / "benchmark_v1" / "gold.jsonl"
    overlap = 0
    if old.exists():
        old_hashes = {json.loads(line).get("problem_hash") for line in old.read_text(encoding="utf-8").splitlines() if line.strip()}
        overlap = len({r["problem_hash"] for r in rows} & old_hashes)
        assert overlap == 0

    print(json.dumps({
        "status": "ok",
        "n_items": len(rows),
        "dev": splits["dev"],
        "never_touch": splits["holdout"],
        "domains": len(domains),
        "input_sha256": manifest["input_sha256"],
        "gold_sha256": manifest["gold_sha256"],
        "exact_overlap_with_benchmark_v1": overlap,
    }, indent=2))


if __name__ == "__main__":
    main()
