#!/usr/bin/env python3
"""Automatically review normalized benchmark candidates with strict source rules.

This is intentionally conservative.  It approves rows from curated/verified
sources after schema and self-containedness checks, and applies extra semantic
filters to the noisier DeepMath gap source.  The script is deterministic so a
frozen benchmark can be reproduced from the same upstream snapshots.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.io import read_jsonl, write_jsonl

TRUSTED_SOURCES = {
    "theoremqa",
    "proofnet_verified",
    "scibench",
    "supergpqa",
    "orqa",
    "ma_proofbench",
    "hardmath2",
}

PDE_TERMS = (
    "partial differential", "pde", "heat equation", "wave equation",
    "laplace equation", "poisson equation", "dirichlet", "neumann",
    "burgers", "hamilton-jacobi", "navier-stokes", "diffusion equation",
)
DG_TERMS = (
    "differential geometry", "riemannian", "geodesic", "gauss curvature",
    "gaussian curvature", "sectional curvature", "ricci", "levi-civita",
    "connection", "metric tensor", "second fundamental form",
    "mean curvature", "shape operator", "christoffel", "tangent bundle",
)
FA_TERMS = (
    "functional analysis", "banach", "hilbert", "bounded operator",
    "compact operator", "linear functional", "weak topology", "weak convergence",
)
MEASURE_TERMS = (
    "measure theory", "lebesgue", "measurable", "sigma-algebra", "σ-algebra",
    "radon-nikodym", "dominated convergence", "monotone convergence",
)
REGRESSION_TERMS = (
    "regression", "least squares", "linear model", "glm", "generalized linear",
)

IMAGE_PHRASES = (
    "shown in the figure", "see the figure", "shown in figure", "attached image",
    "image below", "diagram below", "as shown below",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Automatically review normalized benchmark rows.")
    p.add_argument("input")
    p.add_argument("--output", required=True, help="all rows with approved/rejected review_status")
    p.add_argument("--approved-output", help="approved rows only")
    p.add_argument("--report", help="JSON review report")
    return p.parse_args()


def _reject(reason: str) -> tuple[bool, str]:
    return False, reason


def _choice_is_consistent(row: dict[str, Any]) -> bool:
    if row.get("answer_type") != "choice":
        return True
    meta = row.get("source_meta") or {}
    option_map = meta.get("option_map")
    if not isinstance(option_map, dict) or len(option_map) < 2:
        return False
    return str(row.get("answer") or "").strip().upper() in option_map


def _deepmath_semantic_check(row: dict[str, Any]) -> tuple[bool, str]:
    text = str(row.get("problem") or "").lower()
    domain = row.get("domain")
    terms = {
        "pde": PDE_TERMS,
        "differential_geometry": DG_TERMS,
        "functional_analysis": FA_TERMS,
        "measure_theory": MEASURE_TERMS,
        "regression_analysis": REGRESSION_TERMS,
    }.get(domain)
    if not terms:
        return _reject("deepmath_domain_not_whitelisted")
    if not any(term in text for term in terms):
        return _reject("deepmath_problem_not_semantically_confirmed")
    return True, "deepmath_topic_and_problem_agree"


def review_row(row: dict[str, Any]) -> tuple[bool, str]:
    problem = str(row.get("problem") or "").strip()
    answer = row.get("answer")
    source = str(row.get("source_dataset") or "")
    domain = str(row.get("domain") or "")

    if len(problem) < 20:
        return _reject("problem_too_short")
    if len(problem) > 30000:
        return _reject("problem_too_long")
    if answer is None or str(answer).strip() == "":
        return _reject("missing_answer")
    if not row.get("problem_hash"):
        return _reject("missing_problem_hash")
    if not domain:
        return _reject("missing_domain")
    if any(phrase in problem.lower() for phrase in IMAGE_PHRASES):
        # Some text explicitly depends on a visual that our competition-shaped
        # input does not carry.  Reject rather than silently measuring OCR/image loss.
        return _reject("external_visual_dependency")
    if not _choice_is_consistent(row):
        return _reject("choice_answer_not_in_options")

    meta = row.get("source_meta") or {}
    if source == "theoremqa" and meta.get("picture") not in (None, "", "NONE"):
        return _reject("theoremqa_picture_dependency")

    if source == "ma_proofbench":
        formal = str(meta.get("formal_statement") or "")
        if "theorem" not in formal or "sorry" not in formal:
            return _reject("ma_proofbench_missing_verified_formal_spec")
        if domain not in {"functional_analysis", "measure_theory"}:
            return _reject("ma_proofbench_unexpected_domain")
        return True, "expert_reviewed_formal_benchmark"

    if source == "hardmath2":
        if str(meta.get("type") or "").lower() not in {"nonlinear_pde", "nonlinear_pdes"}:
            return _reject("hardmath2_not_pde_subset")
        if domain != "pde":
            return _reject("hardmath2_domain_mismatch")
        return True, "peer_verified_course_benchmark"

    if source == "deepmath_gap":
        return _deepmath_semantic_check(row)

    if source in TRUSTED_SOURCES:
        return True, "curated_source_schema_pass"

    return _reject("unrecognized_source")


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    reviewed: list[dict[str, Any]] = []
    approved: list[dict[str, Any]] = []
    reasons = Counter()
    by_domain = defaultdict(Counter)
    by_source = defaultdict(Counter)

    for row in rows:
        item = dict(row)
        ok, reason = review_row(item)
        item["review_status"] = "approved" if ok else "rejected"
        item["review_method"] = "automatic_source_schema_semantic_v1"
        item["review_reason"] = reason
        reviewed.append(item)
        reasons[reason] += 1
        by_domain[item.get("domain", "unknown")][item["review_status"]] += 1
        by_source[item.get("source_dataset", "unknown")][item["review_status"]] += 1
        if ok:
            approved.append(item)

    write_jsonl(args.output, reviewed)
    if args.approved_output:
        write_jsonl(args.approved_output, approved)

    report = {
        "review_method": "automatic_source_schema_semantic_v1",
        "n_input": len(reviewed),
        "n_approved": len(approved),
        "n_rejected": len(reviewed) - len(approved),
        "approval_rate": round(len(approved) / len(reviewed), 6) if reviewed else 0,
        "reasons": dict(sorted(reasons.items())),
        "by_domain": {k: dict(v) for k, v in sorted(by_domain.items())},
        "by_source": {k: dict(v) for k, v in sorted(by_source.items())},
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
