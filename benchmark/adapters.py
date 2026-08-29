from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, Mapping, Optional


def normalize_problem_text(text: Any) -> str:
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value


def problem_hash(problem: str) -> str:
    value = re.sub(r"\s+", " ", problem).strip().lower()
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_answer_type(raw_type: Any, answer: Any) -> str:
    t = str(raw_type or "").strip().lower().replace("-", " ").replace("_", " ")
    if t in {"int", "integer"}:
        return "integer"
    if t == "float":
        return "float"
    if t in {"number", "numeric", "decimal"}:
        return "numeric"
    if "bool" in t or t in {"judgement", "judgment", "true false"}:
        return "boolean"
    if "choice" in t:
        return "choice"
    if any(k in t for k in ("list", "tuple", "set", "multiple")):
        return "multiple_values"
    if "proof" in t:
        return "proof"
    if any(k in t for k in ("symbol", "expression", "latex")):
        return "symbolic"
    if isinstance(answer, bool):
        return "boolean"
    if isinstance(answer, int) and not isinstance(answer, bool):
        return "integer"
    if isinstance(answer, float):
        return "float"
    if isinstance(answer, (list, tuple, set)):
        return "multiple_values"
    return "text"


_DOMAIN_RULES = [
    ("complex_analysis", ("complex analysis", "complex variable", "holomorphic")),
    ("functional_analysis", ("functional analysis", "banach space", "hilbert space", "operator theory")),
    ("measure_theory", ("measure theory", "lebesgue", "measure and integration")),
    ("pde", ("partial differential equation", "partial differential equations", "pde")),
    ("ode", ("ordinary differential equation", "ordinary differential equations", "dynamical systems")),
    ("stochastic_processes", ("stochastic process", "markov process", "brownian motion")),
    ("probability_theory", ("probability theory", "probability", "random variable")),
    ("statistical_inference", ("statistical inference", "mathematical statistics", "hypothesis testing", "estimation theory")),
    ("regression_analysis", ("regression analysis", "linear regression", "generalized linear model")),
    ("numerical_analysis", ("numerical analysis", "numerical mathematics", "scientific computing", "numerical methods")),
    ("operations_research", ("operations research", "operational research", "linear programming", "integer programming", "convex optimization", "optimization theory")),
    ("discrete_mathematics", ("discrete mathematics", "combinatorics", "graph theory")),
    ("topology", ("algebraic topology", "general topology", "point-set topology", "topology")),
    ("differential_geometry", ("differential geometry", "riemannian geometry", "smooth manifold")),
    ("abstract_algebra", ("abstract algebra", "group theory", "ring theory", "field theory", "commutative algebra", "galois theory")),
    ("advanced_algebra", ("linear algebra", "matrix theory", "multilinear algebra")),
    ("mathematical_analysis", ("mathematical analysis", "real analysis", "calculus", "harmonic analysis", "fourier analysis")),
]


def infer_domain(*parts: Any) -> Optional[str]:
    text = " | ".join(str(p or "") for p in parts).lower()
    for domain, keys in _DOMAIN_RULES:
        if any(key in text for key in keys):
            return domain
    return None


def make_row(*, problem: Any, answer: Any, answer_type: Any, domain: Optional[str],
             source_dataset: str, source_id: Any, difficulty: Any = None,
             solution: Any = None, source_meta: Optional[Mapping[str, Any]] = None,
             review_status: str = "pending") -> Optional[Dict[str, Any]]:
    problem_text = normalize_problem_text(problem)
    if not problem_text or answer is None or domain is None:
        return None
    row: Dict[str, Any] = {
        "problem": problem_text,
        "answer": answer,
        "answer_type": canonical_answer_type(answer_type, answer),
        "domain": domain,
        "difficulty": str(difficulty or "unknown").lower(),
        "source_dataset": source_dataset,
        "source_id": str(source_id),
        "problem_hash": problem_hash(problem_text),
        "review_status": review_status,
    }
    if solution not in (None, "", "NONE"):
        row["reference_solution"] = solution
    if source_meta:
        row["source_meta"] = dict(source_meta)
    return row


THEOREMQA_SUBFIELD_MAP = {
    "algebra": "advanced_algebra",
    "mathematical analysis": "mathematical_analysis",
    "calculus": "mathematical_analysis",
    "complex analysis": "complex_analysis",
    "functional analysis": "functional_analysis",
    "numerical analysis": "numerical_analysis",
    "probability theory": "probability_theory",
    "stochastic process": "stochastic_processes",
    "combinatorics": "discrete_mathematics",
    "graph theory": "discrete_mathematics",
    "statistics": "statistical_inference",
}


def normalize_theoremqa(item: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if str(item.get("field", "")).strip().lower() != "math":
        return None
    # The competition proxy is text-only.  A non-null Picture means the item is
    # not self-contained in the JSON problem string, so exclude it rather than
    # accidentally measuring missing-image handling.
    if item.get("Picture") not in (None, "", "NONE"):
        return None
    subfield = str(item.get("subfield", "")).strip().lower()
    domain = THEOREMQA_SUBFIELD_MAP.get(subfield) or infer_domain(item.get("subfield"), item.get("theorem"))
    return make_row(
        problem=item.get("Question"), answer=item.get("Answer"), answer_type=item.get("Answer_type"),
        domain=domain, source_dataset="theoremqa", source_id=item.get("id", ""),
        solution=item.get("explanation"),
        source_meta={"field": item.get("field"), "subfield": item.get("subfield"), "theorem": item.get("theorem"),
                     "original_source": item.get("source"), "picture": item.get("Picture")},
    )


SCIBENCH_FILE_DOMAIN = {"calculus": "mathematical_analysis", "diff": "ode"}


def normalize_scibench(item: Mapping[str, Any], source_file: str) -> Optional[Dict[str, Any]]:
    domain = SCIBENCH_FILE_DOMAIN.get(source_file)
    if domain is None:
        return None
    answer = item.get("answer_number")
    answer_type = "numeric"
    answer_latex = str(item.get("answer_latex") or "").strip()
    if answer in (None, "") and answer_latex:
        answer, answer_type = answer_latex, "symbolic"
    return make_row(
        problem=item.get("problem_text"), answer=answer, answer_type=answer_type, domain=domain,
        source_dataset="scibench", source_id=f"{source_file}:{str(item.get('problemid', '')).strip()}",
        source_meta={"book_key": source_file, "answer_latex": answer_latex,
                     "unit": str(item.get("unit") or "").strip(), "comment": item.get("comment")},
    )


PROOFNET_TEXTBOOK_DOMAIN = {
    "artin": "abstract_algebra",
    "axler": "advanced_algebra",
    "munkres": "topology",
    "rudin": "mathematical_analysis",
}


def normalize_proofnet(item: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    textbook = str(item.get("textbook") or "").strip()
    domain = PROOFNET_TEXTBOOK_DOMAIN.get(textbook.lower()) or infer_domain(textbook, item.get("name"), item.get("informal_stmt"))
    return make_row(
        problem=item.get("informal_stmt"), answer=item.get("informal_proof"), answer_type="proof",
        domain=domain, source_dataset="proofnet_verified", source_id=item.get("name", item.get("index", "")),
        solution=item.get("informal_proof"),
        source_meta={"textbook": textbook, "formal_stmt": item.get("formal_stmt"), "index": item.get("index")},
    )


def _format_options(options: Any) -> tuple[str, Dict[str, Any]]:
    if isinstance(options, Mapping):
        pairs = [(str(k), v) for k, v in options.items()]
    elif isinstance(options, (list, tuple)):
        pairs = [(chr(ord("A") + i), v) for i, v in enumerate(options)]
    else:
        return "", {}
    return "\n".join(f"{k}. {v}" for k, v in pairs), {k: v for k, v in pairs}


def normalize_supergpqa(item: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if str(item.get("discipline") or "").strip().lower() not in {"mathematics", "math"}:
        return None
    domain = infer_domain(item.get("field"), item.get("subfield"))
    if domain is None:
        return None
    option_text, option_map = _format_options(item.get("options"))
    problem = normalize_problem_text(item.get("question"))
    if option_text:
        problem = f"{problem}\n\nOptions:\n{option_text}"
    answer_letter = str(item.get("answer_letter") or "").strip()
    if not answer_letter:
        return None
    return make_row(
        problem=problem, answer=answer_letter.upper(), answer_type="choice", domain=domain,
        source_dataset="supergpqa", source_id=item.get("uuid", ""), difficulty=item.get("difficulty"),
        source_meta={"discipline": item.get("discipline"), "field": item.get("field"),
                     "subfield": item.get("subfield"), "is_calculation": item.get("is_calculation"),
                     "option_map": option_map, "answer_text": item.get("answer")},
    )


def deduplicate(rows: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    seen: set[str] = set()
    out: list[Dict[str, Any]] = []
    for row in rows:
        h = row.get("problem_hash") or problem_hash(str(row.get("problem", "")))
        if h in seen:
            continue
        seen.add(h)
        out.append(row)
    return out
