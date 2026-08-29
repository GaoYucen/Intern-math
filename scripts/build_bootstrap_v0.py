#!/usr/bin/env python3
"""Build the 160-question exploratory bootstrap set from the student's normalized pool.

This is intentionally NOT Benchmark-v1.  It exists so the runner/evaluator can
be tested immediately on a competition-shaped input while the audited public
benchmark is being assembled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.io import read_jsonl, write_jsonl

ALLOWED_SOURCES = {"MathBench", "TheoremQA", "mathbench", "theoremqa"}
DEFAULT_DOMAINS = [
    "advanced_algebra",
    "complex_analysis",
    "discrete_mathematics",
    "mathematical_analysis",
    "numerical_analysis",
    "ode",
    "probability_theory",
    "statistical_inference",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build exploratory bootstrap_v0.")
    p.add_argument("candidate_pool", help="normalized output of scripts/import_student_pool.py")
    p.add_argument("--output-dir", default="data/bootstrap_v0")
    p.add_argument("--per-domain", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_diverse(items: list[dict], n: int, rng: random.Random) -> list[dict]:
    by_source: dict[str, deque] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in items:
        grouped[row.get("source_dataset", "unknown")].append(row)
    for source, rows in grouped.items():
        rng.shuffle(rows)
        by_source[source] = deque(rows)
    source_order = sorted(by_source)
    rng.shuffle(source_order)
    selected = []
    while len(selected) < n:
        progress = False
        for source in source_order:
            if by_source[source] and len(selected) < n:
                selected.append(by_source[source].popleft())
                progress = True
        if not progress:
            break
    return selected


def main() -> None:
    args = parse_args()
    if args.per_domain < 1:
        raise SystemExit("--per-domain must be positive")
    rows = read_jsonl(args.candidate_pool)
    by_domain: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    for row in rows:
        if row.get("source_dataset") not in ALLOWED_SOURCES:
            continue
        domain = row.get("domain")
        if domain not in DEFAULT_DOMAINS:
            continue
        key = row.get("problem_hash") or row.get("problem")
        if key in seen:
            continue
        seen.add(key)
        by_domain[domain].append(row)

    short = {d: len(by_domain[d]) for d in DEFAULT_DOMAINS if len(by_domain[d]) < args.per_domain}
    if short:
        raise SystemExit(f"Insufficient candidate coverage for bootstrap_v0: {short}")

    rng = random.Random(args.seed)
    selected = []
    for domain in DEFAULT_DOMAINS:
        selected.extend(source_diverse(by_domain[domain], args.per_domain, rng))
    rng.shuffle(selected)

    input_rows, gold_rows = [], []
    for idx, row in enumerate(selected):
        input_rows.append({"idx": idx, "problem": row["problem"]})
        gold = dict(row)
        gold["idx"] = idx
        gold_rows.append(gold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path, gold_path = output_dir / "input.jsonl", output_dir / "gold.jsonl"
    write_jsonl(input_path, input_rows)
    write_jsonl(gold_path, gold_rows)
    manifest = {
        "name": "bootstrap_v0",
        "status": "exploratory_unreviewed",
        "n_rows": len(selected),
        "per_domain": args.per_domain,
        "seed": args.seed,
        "selection": "MathBench + TheoremQA only; Hendrycks MATH excluded; fixed 8 domains",
        "by_domain": dict(sorted(Counter(r["domain"] for r in selected).items())),
        "by_source": dict(sorted(Counter(r["source_dataset"] for r in selected).items())),
        "input_contract": ["idx", "problem"],
        "input_sha256": sha256_file(input_path),
        "gold_sha256": sha256_file(gold_path),
        "limitations": [
            "not human-reviewed",
            "covers only 8 working proxy domains",
            "not Benchmark-v1",
            "proof/PDE/topology/etc. gaps remain",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
