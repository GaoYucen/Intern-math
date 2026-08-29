#!/usr/bin/env python3
"""Build a deterministic, audited, balanced held-out proxy benchmark.

The builder intentionally prefers domain-specific, higher-trust sources over a
pure random draw from the merged approved pool. This avoids letting broad
aggregated labels (for example SuperGPQA's ``Geometry and Topology``) dominate a
proxy domain with questions that are mathematically valid but belong to a
neighboring field.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import read_jsonl, write_jsonl
from benchmark.domains import WORKING_TARGET_DOMAINS


# Benchmark-v1 source quotas. Counts are targets, not hard requirements: if a
# preferred source is short after quality filtering, the remaining slots are
# filled from other approved sources in trust order.
SOURCE_QUOTAS: dict[str, list[tuple[str, int]]] = {
    "abstract_algebra": [("theoremqa", 11), ("proofnet_verified", 9)],
    "advanced_algebra": [("theoremqa", 20)],
    "complex_analysis": [("theoremqa", 20)],
    "differential_geometry": [("deepmath_gap", 20)],
    "discrete_mathematics": [("theoremqa", 20)],
    "functional_analysis": [("theoremqa", 10), ("ma_proofbench", 10)],
    "mathematical_analysis": [("theoremqa", 15), ("scibench", 3), ("proofnet_verified", 2)],
    "measure_theory": [("theoremqa", 4), ("ma_proofbench", 16)],
    "numerical_analysis": [("theoremqa", 20)],
    "ode": [("scibench", 20)],
    "operations_research": [("orqa", 20)],
    "pde": [("hardmath2", 10), ("deepmath_gap", 10)],
    "probability_theory": [("theoremqa", 19), ("supergpqa", 1)],
    "regression_analysis": [("mmlu_regression", 20)],
    "statistical_inference": [("theoremqa", 20)],
    "stochastic_processes": [("theoremqa", 12), ("supergpqa", 8)],
    "topology": [("proofnet_verified", 8), ("supergpqa", 12)],
}

SOURCE_TRUST_ORDER = {
    "theoremqa": 0,
    "proofnet_verified": 1,
    "ma_proofbench": 1,
    "hardmath2": 1,
    "scibench": 2,
    "orqa": 2,
    "mmlu_regression": 2,
    "deepmath_gap": 3,
    "supergpqa": 4,
}

TOPOLOGY_TERMS = re.compile(
    r"\b(topolog(?:y|ical)?|homeomorph|homotop|fundamental group|covering space|"
    r"torus|manifold|connected|compact|open set|closed set|critical point|"
    r"euler characteristic|deformation retract|simply connected|projective plane|"
    r"quotient space|separation axiom|metric space)\b",
    flags=re.I,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a reproducible balanced benchmark from normalized JSONL files."
    )
    p.add_argument("inputs", nargs="+", help="Normalized source JSONL files")
    p.add_argument("--output", required=True)
    p.add_argument("--per-domain", type=int, default=25)
    p.add_argument("--min-per-domain", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--domains",
        default=",".join(WORKING_TARGET_DOMAINS),
        help="comma-separated proxy domains; defaults to the working taxonomy",
    )
    p.add_argument(
        "--allow-pending",
        action="store_true",
        help="allow rows whose review_status is not approved (exploration only)",
    )
    p.add_argument(
        "--allow-shortfall",
        action="store_true",
        help="write an incomplete benchmark even when a target domain is under-covered",
    )
    p.add_argument(
        "--exclude-source",
        action="append",
        default=[],
        help="source_dataset to exclude; can be repeated",
    )
    return p.parse_args()


def _benchmark_eligible(row: dict) -> bool:
    """Apply benchmark-specific quality filters after source-level review."""
    problem = str(row.get("problem", "")).strip()
    if not problem:
        return False

    # Synthetic OEIS-style sequence prompts were a major source of very long,
    # unrepresentative reasoning in the first smoke run.
    if problem.lower().startswith("we now define an algorithm:"):
        return False

    # A small number of aggregate rows carry obvious scrape markers.
    if re.search(r"\btc\d{3,}\b", problem, flags=re.I):
        return False

    # SuperGPQA combines ordinary geometry and topology under one subfield. Only
    # retain rows whose question text itself contains topology/manifold language.
    if row.get("domain") == "topology" and row.get("source_dataset") == "supergpqa":
        return bool(TOPOLOGY_TERMS.search(problem))

    return True


def _source_rank(row: dict) -> tuple[int, str, str]:
    source = str(row.get("source_dataset", ""))
    return (
        SOURCE_TRUST_ORDER.get(source, 99),
        source,
        str(row.get("problem_hash", "")),
    )


def _choose_domain_rows(pool: list[dict], domain: str, take: int, rng: random.Random) -> list[dict]:
    """Choose rows using source quotas, then trusted-source fallback."""
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        by_source[str(row.get("source_dataset", "unknown"))].append(row)
    for source_pool in by_source.values():
        rng.shuffle(source_pool)

    chosen: list[dict] = []
    chosen_hashes: set[str] = set()

    for source, quota in SOURCE_QUOTAS.get(domain, []):
        for row in by_source.get(source, [])[:quota]:
            chosen.append(row)
            chosen_hashes.add(str(row.get("problem_hash", "")))
            if len(chosen) >= take:
                return chosen

    if len(chosen) < take:
        remaining = [
            row
            for row in pool
            if str(row.get("problem_hash", "")) not in chosen_hashes
        ]
        # Stable trust ordering with a seeded random tiebreak inside each source.
        rng.shuffle(remaining)
        remaining.sort(key=_source_rank)
        chosen.extend(remaining[: take - len(chosen)])

    return chosen[:take]


def main() -> None:
    args = parse_args()
    if args.per_domain < 1 or args.min_per_domain < 1:
        raise SystemExit("per-domain and min-per-domain must be positive")
    if args.min_per_domain > args.per_domain:
        raise SystemExit("min-per-domain cannot exceed per-domain")

    target_domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    excluded = set(args.exclude_source)
    rng = random.Random(args.seed)
    pools = defaultdict(list)
    seen_hashes: set[str] = set()

    for path in args.inputs:
        for row in read_jsonl(path):
            domain = row.get("domain")
            if domain not in target_domains:
                continue
            if row.get("source_dataset") in excluded:
                continue
            status = row.get("review_status", "approved")
            if not args.allow_pending and status != "approved":
                continue
            if not _benchmark_eligible(row):
                continue
            problem_hash = row.get("problem_hash")
            if problem_hash and problem_hash in seen_hashes:
                continue
            if problem_hash:
                seen_hashes.add(problem_hash)
            pools[domain].append(row)

    shortfalls = {
        domain: len(pools[domain])
        for domain in target_domains
        if len(pools[domain]) < args.min_per_domain
    }
    if shortfalls and not args.allow_shortfall:
        formatted = ", ".join(f"{d}={n}" for d, n in shortfalls.items())
        raise SystemExit(
            "Refusing to build a benchmark with insufficient audited coverage: "
            f"{formatted}. Add data/audit rows, change --domains, or use "
            "--allow-shortfall for exploratory runs only."
        )

    selected = []
    selected_counts = Counter()
    selected_sources: dict[str, Counter] = defaultdict(Counter)
    for domain in target_domains:
        pool = list(pools[domain])
        take = min(len(pool), args.per_domain)
        chosen = _choose_domain_rows(pool, domain, take, rng)
        selected.extend(chosen)
        selected_counts[domain] = len(chosen)
        selected_sources[domain].update(row.get("source_dataset", "unknown") for row in chosen)

    rng.shuffle(selected)
    output_rows = []
    for idx, row in enumerate(selected):
        item = dict(row)
        item["idx"] = idx
        output_rows.append(item)

    write_jsonl(args.output, output_rows)
    print(f"Wrote {len(output_rows)} rows to {args.output}")
    for domain in target_domains:
        source_summary = ", ".join(
            f"{source}={count}" for source, count in sorted(selected_sources[domain].items())
        )
        print(
            f"{domain}: selected {selected_counts[domain]} / {len(pools[domain])}"
            f" [{source_summary}]"
        )
    if shortfalls:
        print("WARNING: exploratory benchmark contains coverage shortfalls:")
        for domain, n in shortfalls.items():
            print(f"  {domain}: {n} (< {args.min_per_domain})")


if __name__ == "__main__":
    main()
