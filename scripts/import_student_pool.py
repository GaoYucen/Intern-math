#!/usr/bin/env python3
"""Normalize the student's combined public-math pool into Intern-math schema.

This importer does not declare the imported questions benchmark-ready.  It
preserves audit status and marks every record as a candidate.  The intended
workflow is import -> audit coverage -> manually approve -> build held-out set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.domains import SUBJECT_TO_DOMAIN, WORKING_TARGET_DOMAINS
from benchmark.io import write_jsonl


def normalize_problem_hash(problem: str) -> str:
    normalized = re.sub(r"\s+", " ", problem.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def normalize_answer_type(raw: str, answer: Any) -> str:
    kind = (raw or "").strip().lower().replace("-", "_")
    aliases = {
        "integer": "integer",
        "int": "integer",
        "float": "float",
        "number": "numeric",
        "numeric": "numeric",
        "bool": "boolean",
        "boolean": "boolean",
        "choice": "choice",
        "option": "choice",
        "multiple_choice": "choice",
        "list of integer": "set",
        "list_of_integer": "set",
        "list of float": "set",
        "list_of_float": "set",
        "set": "set",
        "roots": "set",
        "symbolic": "symbolic",
        "expression": "symbolic",
        "proof": "proof",
    }
    if kind in aliases:
        return aliases[kind]

    # Hendrycks MATH is imported as free_form even when the boxed answer is
    # numeric.  Infer only conservative cases; otherwise leave it as text for
    # later audit rather than pretending the scorer can safely judge it.
    if kind == "free_form":
        s = str(answer).strip().strip("$")
        if re.fullmatch(r"[+-]?\d+", s):
            return "integer"
        if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)", s):
            return "float"
        if s.lower() in {"true", "false"}:
            return "boolean"
        if re.fullmatch(r"[A-E]", s.upper()):
            return "choice"
        # Fractions and simple pi expressions are safely handled by SymPy.
        if re.fullmatch(r"[0-9+\-*/(). ^\\piπ]+", s):
            return "symbolic"
        return "text"

    return "text"


def normalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("data_status") != "mapped":
        return None
    subject = row.get("subject")
    domain = SUBJECT_TO_DOMAIN.get(subject)
    if not domain:
        return None
    problem = str(row.get("problem", "")).strip()
    if not problem or "answer" not in row:
        return None

    source_dataset = row.get("source_dataset", "unknown")
    review_status = row.get("review_status", "pending")
    answer = row.get("answer")
    return {
        "idx": -1,
        "problem": problem,
        "answer": answer,
        "answer_type": normalize_answer_type(row.get("answer_type", ""), answer),
        "domain": domain,
        "subject_original": subject,
        "difficulty": row.get("difficulty", "unknown"),
        "language": row.get("language", "unknown"),
        "source_dataset": source_dataset,
        "source_record_key": row.get("source_record_key", row.get("source", "")),
        "source": row.get("source", ""),
        "problem_hash": row.get("problem_hash") or normalize_problem_hash(problem),
        "review_status": review_status,
        "benchmark_status": "candidate",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="student combined_math_all.jsonl")
    p.add_argument("--output", required=True, help="normalized candidate JSONL")
    p.add_argument("--coverage", required=True, help="coverage report JSON")
    p.add_argument(
        "--exclude-source",
        action="append",
        default=[],
        help="source_dataset to exclude; can be repeated",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    excluded = set(args.exclude_source)
    seen_hashes: set[str] = set()
    normalized = []
    duplicate_count = 0

    with Path(args.input).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source_dataset") in excluded:
                continue
            item = normalize_row(row)
            if item is None:
                continue
            if item["problem_hash"] in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(item["problem_hash"])
            item["idx"] = len(normalized)
            normalized.append(item)

    write_jsonl(args.output, normalized)

    by_domain = Counter(r["domain"] for r in normalized)
    by_source = Counter(r["source_dataset"] for r in normalized)
    by_type = Counter(r["answer_type"] for r in normalized)
    by_review = Counter(r["review_status"] for r in normalized)
    domain_source: dict[str, Counter] = defaultdict(Counter)
    for r in normalized:
        domain_source[r["domain"]][r["source_dataset"]] += 1

    coverage = {
        "n_records": len(normalized),
        "deduplicated_records": duplicate_count,
        "working_taxonomy_note": (
            "The competition document says 18 subfields but does not enumerate "
            "them; these are working proxy domains, not an official taxonomy."
        ),
        "by_domain": dict(sorted(by_domain.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_answer_type": dict(sorted(by_type.items())),
        "by_review_status": dict(sorted(by_review.items())),
        "domain_source": {d: dict(sorted(c.items())) for d, c in sorted(domain_source.items())},
        "coverage_gaps": [d for d in WORKING_TARGET_DOMAINS if by_domain[d] < 20],
    }
    coverage_path = Path(args.coverage)
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(coverage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
