from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

try:
    import sympy as sp
except Exception:  # pragma: no cover
    sp = None

FINAL_PATTERNS = [
    re.compile(r"FINAL_ANSWER\s*:\s*(.+)", re.IGNORECASE),
    re.compile(r"最终答案\s*[:：]\s*(.+)", re.IGNORECASE),
]


@dataclass
class Score:
    correct: Optional[bool]
    predicted: str
    gold: str
    answer_type: str
    reason: str = ""


def extract_final_answer(text: str) -> str:
    """Extract the explicit final answer while retaining a safe fallback.

    The benchmark prompt asks for FINAL_ANSWER. If a model ignores it, using the
    final non-empty line is safer than declaring failure immediately.
    """
    if not isinstance(text, str):
        return ""
    matches = []
    for pattern in FINAL_PATTERNS:
        matches.extend(pattern.findall(text))
    if matches:
        return _strip_wrappers(matches[-1].strip())

    boxed = _extract_last_boxed(text)
    if boxed:
        return _strip_wrappers(boxed)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return _strip_wrappers(lines[-1] if lines else "")


def _extract_last_boxed(text: str) -> str:
    # Handles the common single-level \boxed{...} form. Nested braces are not
    # parsed here; FINAL_ANSWER is the preferred contract.
    found = re.findall(r"\\boxed\{([^{}]+)\}", text)
    return found[-1].strip() if found else ""


def _strip_wrappers(value: str) -> str:
    value = value.strip().strip("`$")
    value = re.sub(r"^[\s]*(?:答案|answer)\s*[:：]\s*", "", value, flags=re.I)
    value = value.rstrip("。.;；")
    return value.strip()


def normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("−", "-").replace("–", "-")
    text = re.sub(r"\s+", "", text)
    text = text.strip("。.;；,，")
    return text


def _to_float(value: str) -> Optional[float]:
    s = normalize_text(value)
    s = s.replace("\\pi", "pi").replace("π", "pi")
    if sp is not None:
        try:
            expr = sp.sympify(s.replace("^", "**"), locals={"pi": sp.pi})
            if expr.is_real is False:
                return None
            return float(sp.N(expr))
        except Exception:
            pass
    try:
        return float(s)
    except Exception:
        return None


def _split_set(value: str) -> list[str]:
    s = value.strip()
    s = s.strip("{}[]()")
    parts = re.split(r"[,，;；]|\band\b|和", s, flags=re.I)
    return [normalize_text(p) for p in parts if p.strip()]


def _symbolically_equal(a: str, b: str) -> Optional[bool]:
    if sp is None:
        return None
    try:
        local_dict = {"pi": sp.pi, "e": sp.E, "i": sp.I}
        ea = sp.sympify(
            normalize_text(a).replace("\\pi", "pi").replace("π", "pi").replace("^", "**"),
            locals=local_dict,
        )
        eb = sp.sympify(
            normalize_text(b).replace("\\pi", "pi").replace("π", "pi").replace("^", "**"),
            locals=local_dict,
        )
        return bool(sp.simplify(ea - eb) == 0)
    except Exception:
        return None


def score_answer(
    prediction_text: str,
    gold: Any,
    answer_type: str = "text",
    tolerance: float = 1e-8,
) -> Score:
    pred = extract_final_answer(prediction_text)
    gold_text = str(gold).strip()
    kind = (answer_type or "text").strip().lower()

    if kind in {"integer", "float", "numeric", "number", "rational"}:
        p = _to_float(pred)
        g = _to_float(gold_text)
        if p is None or g is None:
            return Score(None, pred, gold_text, kind, "numeric parse failed")
        ok = math.isclose(p, g, rel_tol=tolerance, abs_tol=tolerance)
        return Score(ok, pred, gold_text, kind)

    if kind in {"symbolic", "expression"}:
        eq = _symbolically_equal(pred, gold_text)
        if eq is None:
            return Score(None, pred, gold_text, kind, "symbolic parse failed")
        return Score(eq, pred, gold_text, kind)

    if kind in {"set", "unordered", "multiple_values", "roots"}:
        p = sorted(_split_set(pred))
        g = sorted(_split_set(gold_text))
        return Score(p == g, pred, gold_text, kind)

    if kind in {"choice", "multiple_choice"}:
        p = re.findall(r"\b([A-Z])\b", pred.upper())
        g = re.findall(r"\b([A-Z])\b", gold_text.upper())
        if not p or not g:
            return Score(None, pred, gold_text, kind, "choice label parse failed")
        return Score(p[-1] == g[-1], pred, gold_text, kind)

    if kind in {"bool", "boolean", "judgement", "true_false"}:
        def norm_bool(v: str) -> Optional[bool]:
            t = normalize_text(v)
            if t in {"true", "yes", "正确", "对", "成立"}:
                return True
            if t in {"false", "no", "错误", "错", "不成立"}:
                return False
            return None
        p = norm_bool(pred)
        g = norm_bool(gold_text)
        if p is None or g is None:
            return Score(None, pred, gold_text, kind, "boolean parse failed")
        return Score(p == g, pred, gold_text, kind)

    if kind in {"proof", "explanation", "derivation"}:
        return Score(None, pred, gold_text, kind, "requires judge/manual review")

    return Score(normalize_text(pred) == normalize_text(gold_text), pred, gold_text, kind)
