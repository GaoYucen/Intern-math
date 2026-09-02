#!/usr/bin/env python3
"""Build Fresh-Holdout-v1: 72 deterministic, newly written math problems.

Design goals:
- 18 subject areas x 4 problems each.
- Mixed Chinese/English wording.
- Mostly exact/symbolic answers, with a small number of conceptual items.
- No copying of hidden evaluation questions or exact-answer lookup assets.
- Frozen deterministic output for fair architecture comparisons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def add(rows, subject, problem, answer, answer_type, difficulty="hard", language="zh", track="exact"):
    rows.append({
        "idx": len(rows),
        "problem": problem,
        "answer": str(answer),
        "subject": subject,
        "answer_type": answer_type,
        "difficulty": difficulty,
        "language": language,
        "track": track,
        "source": "fresh_synthetic_v1",
    })


def build_rows():
    r = []

    # 1. PDE
    s = "偏微分方程"
    add(r,s,"求解 u_t=4u_xx, 0<x<π, u(0,t)=u(π,t)=0, u(x,0)=3sin(2x)-2sin(5x)。给出 u(x,t)。","3*exp(-16*t)*sin(2*x)-2*exp(-100*t)*sin(5*x)","symbolic")
    add(r,s,"求一阶方程 u_x-3u_y=0 的解，满足 u(0,y)=exp(2y)。","exp(2*(y+3*x))","symbolic")
    add(r,s,"Solve Laplace's equation in the unit disk with boundary value 2 cos(3θ)-sin(4θ). Give u(r,θ).","2*r^3*cos(3*theta)-r^4*sin(4*theta)","symbolic",language="en")
    add(r,s,"Solve u_t+2u_x=0 on R with u(x,0)=cos x.","cos(x-2*t)","symbolic",language="en")

    # 2. Complex analysis
    s = "复分析"
    add(r,s,"计算逆时针积分 ∮_{|z|=2} e^{2z}/(z-1)^3 dz。","4*pi*i*e^2","symbolic")
    add(r,s,"计算 ∮_{|z|=2} z^3/(z^2+1) dz，逆时针。","-2*pi*i","symbolic")
    add(r,s,"Find the coefficient of z^{-1} in the Laurent expansion of z^5 exp(1/z) about z=0.","1/720","symbolic",language="en")
    add(r,s,"How many zeros, counted with multiplicity, does z^5+10z+1 have in |z|<1?","1","integer",language="en")

    # 3. Topology
    s = "拓扑学"
    add(r,s,"闭可定向亏格为3的曲面的第一 Betti 数 b_1 是多少？","6","integer")
    add(r,s,"4 个环面的连通和的 Euler 示性数是多少？","-6","integer")
    add(r,s,"A continuous bijection from a compact space to a Hausdorff space is necessarily a homeomorphism. True or false?","True","boolean",language="en")
    add(r,s,"Identify [0,1] with its two endpoints glued together. Is the quotient homeomorphic to S^1?","True","boolean",language="en")

    # 4. Operations research
    s = "运筹学"
    add(r,s,"线性规划 max 4x+3y, s.t. x+y≤5, 2x+y≤8, x,y≥0。求最大值。","18","integer",track="tool")
    add(r,s,"0-1 背包：重量 [3,4,6,7]，价值 [5,6,10,11]，容量10。求最大价值。","16","integer",track="tool")
    add(r,s,"For the weighted graph with edges AB=4, AC=2, BC=1, BD=7, CD=5, CE=8, DE=1, DF=3, EF=2, find the shortest-path distance from A to F.","9","integer",language="en",track="tool")
    add(r,s,"Solve the 3x3 assignment problem with cost matrix [[4,1,3],[2,0,5],[3,2,2]]. Give the minimum total cost.","5","integer",language="en",track="tool")

    # 5. Real analysis
    s = "实分析"
    add(r,s,"求 lim_{n→∞} ∫_0^1 x^n/(1+x^n) dx。","0","numeric")
    add(r,s,"函数级数 ∑_{n=1}^∞ sin(nx)/n^2 是否在 R 上一致收敛？只回答 True/False。","True","boolean")
    add(r,s,"Evaluate ∫_0^∞ x^2 e^{-3x} dx exactly.","2/27","symbolic",language="en")
    add(r,s,"Does f_n(x)=x^n converge uniformly to 0 on [0,0.8]?","True","boolean",language="en")

    # 6. Linear algebra
    s = "线性代数"
    add(r,s,"求 det([[2,1,0],[1,2,1],[0,1,2]])。","4","integer",track="tool")
    add(r,s,"矩阵 A 的对角元全为4，非对角元全为1，A 为3×3。写出全部特征值（含重数）。","6,3,3","multiple_values",track="tool")
    add(r,s,"Find the rank of the 4x4 Vandermonde matrix built from distinct nodes 1,2,3,4.","4","integer",language="en")
    add(r,s,"A matrix has Jordan blocks J_2(2), J_1(2), J_2(-1). Give its minimal polynomial.","(x-2)^2*(x+1)^2","symbolic",language="en")

    # 7. Abstract algebra
    s = "抽象代数"
    add(r,s,"在 F_{125} 中，有多少元素 α 满足 F_5(α)=F_{125}？","120","integer")
    add(r,s,"循环群 Z_60 中阶恰为12的元素有多少个？","4","integer")
    add(r,s,"How many group homomorphisms are there from Z_12 to Z_18?","6","integer",language="en")
    add(r,s,"How many units does the ring Z/72Z have?","24","integer",language="en")

    # 8. Probability
    s = "概率论"
    add(r,s,"X1,X2,X3 独立服从 Uniform(0,1)。求 E[max(X1,X2,X3)]。","3/4","symbolic")
    add(r,s,"5 次独立 Bernoulli 试验，每次成功概率1/3。恰好2次成功的概率是多少？","80/243","symbolic")
    add(r,s,"If X~Exp(rate=2), compute P(X>3 | X>1).","exp(-4)","symbolic",language="en")
    add(r,s,"Let X,Y be independent N(0,1). Compute Cov(X+Y, X-Y).","0","numeric",language="en")

    # 9. Statistics
    s = "数理统计"
    add(r,s,"先验 p~Beta(2,3)，观察到10次 Bernoulli 中7次成功。求后验均值。","3/5","symbolic")
    add(r,s,"指数分布 rate λ 的样本为 1,2,3,4。求 λ 的 MLE。","2/5","symbolic")
    add(r,s,"For the sample 2,4,6,8, compute the unbiased sample variance.","20/3","symbolic",language="en",track="tool")
    add(r,s,"Known σ=2, n=100, sample mean=10. Using z=1.96, give the two endpoints of the 95% confidence interval for μ.","9.608,10.392","multiple_values",language="en",track="tool")

    # 10. Numerical analysis
    s = "数值分析"
    add(r,s,"用 Newton 法求 sqrt(10)，初值 x0=3，只做一步。给出 x1 的精确分数。","19/6","symbolic")
    add(r,s,"显式 Euler 法求 y'=-5y 的绝对稳定步长区间（h>0）。","0<h<2/5","symbolic")
    add(r,s,"What is the 2-norm condition number of diag(2,7)?","7/2","symbolic",language="en")
    add(r,s,"Use the composite trapezoidal rule with n=2 subintervals to approximate ∫_0^1 x^2 dx.","3/8","symbolic",language="en",track="tool")

    # 11. Differential geometry
    s = "微分几何"
    add(r,s,"平面曲线 y=x^2 在 x=1 处的曲率是多少？","2/(5*sqrt(5))","symbolic")
    add(r,s,"曲面 z=x^2+y^2 在原点的 Gauss 曲率 K 是多少？","4","numeric")
    add(r,s,"What is the Gaussian curvature of a sphere of radius 3?","1/9","symbolic",language="en")
    add(r,s,"Find the area of a standard torus with major radius R=5 and minor radius r=2.","40*pi^2","symbolic",language="en")

    # 12. Number theory
    s = "数论"
    add(r,s,"求满足 x≡1 (mod 4), x≡2 (mod 5), x≡3 (mod 7) 的最小非负整数。","17","integer",track="tool")
    add(r,s,"求 3^100 mod 7。","4","integer",track="tool")
    add(r,s,"Find the fundamental positive solution (x,y) of x^2-13y^2=1.","649,180","multiple_values",language="en",track="tool")
    add(r,s,"Compute Euler's totient φ(840).","192","integer",language="en",track="tool")

    # 13. Combinatorics
    s = "组合数学"
    add(r,s,"非负整数解 x1+x2+x3=15 且每个 xi≤7，共有多少组？","28","integer",track="tool")
    add(r,s,"长度8的二元项链，在旋转等价下共有多少种？","36","integer",track="tool")
    add(r,s,"How many derangements are there on 6 labeled objects?","265","integer",language="en")
    add(r,s,"Compute the Catalan number C_8.","1430","integer",language="en")

    # 14. Graph theory
    s = "图论"
    add(r,s,"完全二分图 K_{3,5} 有多少棵生成树？","2025","integer",track="tool")
    add(r,s,"用3种有标号颜色对七边形环图 C7 正常顶点着色，共有多少种？","126","integer")
    add(r,s,"How many edges does a tree on 17 vertices have?","16","integer",language="en")
    add(r,s,"How many perfect matchings does K_{4,4} have?","24","integer",language="en")

    # 15. Mathematical logic
    s = "数理逻辑"
    add(r,s,"命题 ((P→Q)∧P)→Q 是否为永真式？","True","boolean")
    add(r,s,"P XOR Q 在两个布尔变量上共有多少个满足赋值？","2","integer")
    add(r,s,"Negate ∀x∃y R(x,y), pushing the negation all the way to the predicate.","exists x forall y not R(x,y)","text",language="en",track="judge")
    add(r,s,"For classical first-order logic, Γ entails φ iff Γ union {not φ} is unsatisfiable. True or false?","True","boolean",language="en")

    # 16. ODE
    s = "常微分方程"
    add(r,s,"解 y'+3y=6, y(0)=1。","2-exp(-3*t)","symbolic")
    add(r,s,"解 y''+4y=0, y(0)=0, y'(0)=6。","3*sin(2*t)","symbolic")
    add(r,s,"Solve x'=-2y, y'=2x with x(0)=1,y(0)=0.","x=cos(2*t),y=sin(2*t)","text",language="en",track="judge")
    add(r,s,"For y''+λy=0, y(0)=y(π)=0, what is the third positive eigenvalue?","9","integer",language="en")

    # 17. Functional analysis
    s = "泛函分析"
    add(r,s,"在 C[0,1] 配 sup 范数上，φ(f)=∫_0^1 x^2 f(x)dx。求 ||φ||。","1/3","symbolic")
    add(r,s,"ℓ^2 上左移算子 L(x1,x2,...)= (x2,x3,...) 的算子范数是多少？","1","numeric")
    add(r,s,"On l^2 define T(e_n)=(1/n)e_n. Is T compact?","True","boolean",language="en")
    add(r,s,"The functional f(x1,x2)=3x1+4x2 acts on R^2 with Euclidean norm. Find ||f||.","5","numeric",language="en")

    # 18. Discrete mathematics
    s = "离散数学"
    add(r,s,"递推 T(n)=3T(n/2)+n 的渐近阶是什么？","Theta(n^log2(3))","text",track="judge")
    add(r,s,"6 元集合有多少种集合划分？即 Bell 数 B6。","203","integer")
    add(r,s,"How many binary strings of length 10 contain no two consecutive 1s?","144","integer",language="en")
    add(r,s,"Solve a_n=2a_{n-1}+1 with a_0=0.","2^n-1","symbolic",language="en")

    assert len(r) == 72, len(r)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/fresh_holdout_v1/scored.jsonl")
    args = p.parse_args()
    rows = build_rows()
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
    out.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    summary = {
        "n": len(rows),
        "sha256": digest,
        "subjects": dict(Counter(x["subject"] for x in rows)),
        "answer_types": dict(Counter(x["answer_type"] for x in rows)),
        "languages": dict(Counter(x["language"] for x in rows)),
        "tracks": dict(Counter(x["track"] for x in rows)),
    }
    (out.parent / "scored_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
