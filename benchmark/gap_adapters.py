from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from benchmark.adapters import make_row, normalize_problem_text


def _format_options(options: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(options, (list, tuple)):
        return "", {}
    pairs = [(chr(ord("A") + i), value) for i, value in enumerate(options) if value is not None]
    return "\n".join(f"{key}. {value}" for key, value in pairs), dict(pairs)


def normalize_orqa(item: Mapping[str, Any], source_id: Any) -> Optional[dict]:
    """Normalize ORQA into a competition-shaped multiple-choice candidate."""
    option_text, option_map = _format_options(item.get("OPTIONS"))
    context = normalize_problem_text(item.get("CONTEXT"))
    question = normalize_problem_text(item.get("QUESTION"))
    if not context or not question or not option_text:
        return None
    try:
        target_index = int(item.get("TARGET_ANSWER"))
    except (TypeError, ValueError):
        return None
    answer_letter = chr(ord("A") + target_index)
    if answer_letter not in option_map:
        return None
    problem = f"{context}\n\nQuestion:\n{question}\n\nOptions:\n{option_text}"
    return make_row(
        problem=problem,
        answer=answer_letter,
        answer_type="choice",
        domain="operations_research",
        source_dataset="orqa",
        source_id=source_id,
        solution=item.get("REASONING"),
        source_meta={
            "question_type": item.get("QUESTION_TYPE"),
            "option_map": option_map,
            "target_index_zero_based": target_index,
        },
    )


DEEPMATH_GAP_RULES = [
    ("pde", ("partial differential equations", "partial differential equation", "pdes")),
    ("differential_geometry", ("differential geometry", "riemannian geometry")),
    ("measure_theory", ("measure theory", "lebesgue integration")),
    ("functional_analysis", ("functional analysis", "banach spaces", "hilbert spaces")),
    ("regression_analysis", ("regression analysis", "linear regression")),
]


def deepmath_gap_domain(topic: Any) -> Optional[str]:
    text = str(topic or "").lower()
    for domain, keys in DEEPMATH_GAP_RULES:
        if any(key in text for key in keys):
            return domain
    return None


def _deepmath_answer_type(answer: Any) -> str:
    text = str(answer or "").strip()
    low = text.lower()
    if low in {"yes", "no", "true", "false"}:
        return "boolean"
    if re.fullmatch(r"[-+]?\d+", text):
        return "integer"
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][-+]?\d+)?", text):
        return "numeric"
    return "symbolic"


def normalize_deepmath_gap(item: Mapping[str, Any], source_id: Any) -> Optional[dict]:
    """Use DeepMath only as a secondary gap-filling candidate source."""
    if item.get("processing_success") is False:
        return None
    domain = deepmath_gap_domain(item.get("topic"))
    answer = item.get("final_answer")
    if domain is None or answer in (None, ""):
        return None
    return make_row(
        problem=item.get("question"),
        answer=answer,
        answer_type=_deepmath_answer_type(answer),
        domain=domain,
        source_dataset="deepmath_gap",
        source_id=source_id,
        difficulty=item.get("difficulty"),
        source_meta={
            "topic": item.get("topic"),
            "failed_count": item.get("failed_count"),
            "selection_role": "secondary_gap_filling",
        },
    )


def normalize_ma_proofbench(item: Mapping[str, Any]) -> Optional[dict]:
    """Normalize expert-reviewed MA-ProofBench gap domains.

    MA-ProofBench supplies a natural-language problem and an independently
    expert-reviewed Lean formalization.  It does not provide an informal gold
    proof, so the formal theorem statement is kept as the proof specification
    in the gold file and these rows must be judged separately from exact-answer
    questions.
    """
    topic = str(item.get("topic") or "").strip().lower()
    if topic in {"functional analysis", "operator theory"}:
        domain = "functional_analysis"
    elif topic in {"measure and integration", "measure & integration"}:
        domain = "measure_theory"
    else:
        return None
    problem = normalize_problem_text(item.get("informal_statement"))
    formal = normalize_problem_text(item.get("formal_statement"))
    if not problem or not formal:
        return None
    return make_row(
        problem=problem,
        answer=formal,
        answer_type="proof",
        domain=domain,
        source_dataset="ma_proofbench",
        source_id=item.get("id", ""),
        difficulty=item.get("split"),
        source_meta={
            "split": item.get("split"),
            "topic": item.get("topic"),
            "tag": item.get("tag"),
            "formal_statement": formal,
            "mathlib_version": item.get("version"),
            "selection_role": "expert_reviewed_gap_filling",
            "proof_gold_kind": "formal_theorem_specification",
        },
    )


def normalize_hardmath2(item: Mapping[str, Any]) -> Optional[dict]:
    """Keep only HARDMath2's peer-verified nonlinear PDE problems."""
    kind = str(item.get("type") or "").strip().lower()
    if kind not in {"nonlinear_pde", "nonlinear_pdes"}:
        return None
    problem = normalize_problem_text(item.get("prompt"))
    solution = normalize_problem_text(item.get("solution"))
    if not problem or not solution:
        return None
    return make_row(
        problem=problem,
        answer=solution,
        answer_type="symbolic",
        domain="pde",
        source_dataset="hardmath2",
        source_id=item.get("index", ""),
        difficulty="graduate",
        solution=solution,
        source_meta={
            "type": item.get("type"),
            "parameters": item.get("parameters"),
            "selection_role": "peer_verified_pde",
        },
    )
