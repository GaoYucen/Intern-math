#!/usr/bin/env python3
import argparse
import random
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import read_jsonl, write_jsonl

TARGET_DOMAINS = [
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
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a reproducible balanced benchmark from normalized JSONL files."
    )
    p.add_argument("inputs", nargs="+", help="Normalized source JSONL files")
    p.add_argument("--output", required=True)
    p.add_argument("--per_domain", type=int, default=25)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    pools = defaultdict(list)
    for path in args.inputs:
        for row in read_jsonl(path):
            domain = row.get("domain")
            if domain:
                pools[domain].append(row)

    selected = []
    for domain in sorted(pools):
        pool = list(pools[domain])
        rng.shuffle(pool)
        selected.extend(pool[: args.per_domain])

    rng.shuffle(selected)
    for idx, row in enumerate(selected):
        row = dict(row)
        row["idx"] = idx
        selected[idx] = row

    write_jsonl(args.output, selected)
    print(f"Wrote {len(selected)} rows to {args.output}")
    for domain in sorted(pools):
        print(f"{domain}: selected {min(len(pools[domain]), args.per_domain)} / {len(pools[domain])}")


if __name__ == "__main__":
    main()
