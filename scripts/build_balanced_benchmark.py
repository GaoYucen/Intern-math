#!/usr/bin/env python3
"""Build a deterministic, audited, balanced held-out proxy benchmark."""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import read_jsonl, write_jsonl
from benchmark.domains import WORKING_TARGET_DOMAINS


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
    for domain in target_domains:
        pool = list(pools[domain])
        rng.shuffle(pool)
        take = min(len(pool), args.per_domain)
        chosen = pool[:take]
        selected.extend(chosen)
        selected_counts[domain] = take

    rng.shuffle(selected)
    output_rows = []
    for idx, row in enumerate(selected):
        item = dict(row)
        item["idx"] = idx
        output_rows.append(item)

    write_jsonl(args.output, output_rows)
    print(f"Wrote {len(output_rows)} rows to {args.output}")
    for domain in target_domains:
        print(f"{domain}: selected {selected_counts[domain]} / {len(pools[domain])}")
    if shortfalls:
        print("WARNING: exploratory benchmark contains coverage shortfalls:")
        for domain, n in shortfalls.items():
            print(f"  {domain}: {n} (< {args.min_per_domain})")


if __name__ == "__main__":
    main()
