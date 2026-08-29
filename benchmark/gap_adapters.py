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
    """Normalize ORQA into a competition-shaped multiple-choice candidate.

    ORQA TARGET_ANSWER is a zero-based option index.  The context and question
    are concatenated because neither is self-contained alone.
    """
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
    """Use DeepMath only as a gap-filling candidate source.

    This adapter deliberately ignores the many already-covered DeepMath topics.
    Rows remain pending and require human review because topic annotations and
    gold expressions in large aggregated math corpora can be noisy.
    """
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
            "selection_role": "gap_filling_only",
        },
    )
