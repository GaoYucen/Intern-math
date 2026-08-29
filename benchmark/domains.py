"""Working domain taxonomy for proxy-benchmark construction.

The competition document states that the organizer's problems span 18
subfields, but the document we have does not enumerate all 18.  Therefore this
module deliberately treats the taxonomy below as a *working proxy taxonomy*,
not as an official list.
"""

SUBJECT_TO_DOMAIN = {
    "数学分析": "mathematical_analysis",
    "高等代数": "advanced_algebra",
    "抽象代数": "abstract_algebra",
    "复分析": "complex_analysis",
    "泛函分析": "functional_analysis",
    "测度论": "measure_theory",
    "常微分方程": "ode",
    "偏微分方程": "pde",
    "概率论": "probability_theory",
    "统计推断": "statistical_inference",
    "随机过程": "stochastic_processes",
    "数值分析": "numerical_analysis",
    "运筹学": "operations_research",
    "离散数学": "discrete_mathematics",
    "拓扑学": "topology",
    "微分几何": "differential_geometry",
    "回归分析": "regression_analysis",
    "回归": "regression_analysis",
}

WORKING_TARGET_DOMAINS = [
    "mathematical_analysis",
    "advanced_algebra",
    "abstract_algebra",
    "complex_analysis",
    "functional_analysis",
    "measure_theory",
    "ode",
    "pde",
    "probability_theory",
    "statistical_inference",
    "stochastic_processes",
    "numerical_analysis",
    "operations_research",
    "discrete_mathematics",
    "topology",
    "differential_geometry",
    "regression_analysis",
]
