#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.domains import WORKING_TARGET_DOMAINS
from benchmark.io import read_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit proxy-dataset coverage without running a model.")
    p.add_argument("dataset")
    p.add_argument("--min-per-domain", type=int, default=20)
    p.add_argument("--output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.dataset)
    by_domain = Counter(r.get("domain", "unknown") for r in rows)
    by_source = Counter(r.get("source_dataset", "unknown") for r in rows)
    by_type = Counter(r.get("answer_type", "text") for r in rows)
    by_review = Counter(r.get("review_status", "unknown") for r in rows)
    by_domain_source = defaultdict(Counter)
    by_domain_difficulty = defaultdict(Counter)
    for r in rows:
        domain = r.get("domain", "unknown")
        by_domain_source[domain][r.get("source_dataset", "unknown")] += 1
        by_domain_difficulty[domain][r.get("difficulty", "unknown")] += 1

    gaps = {
        d: by_domain[d]
        for d in WORKING_TARGET_DOMAINS
        if by_domain[d] < args.min_per_domain
    }
    report = {
        "n_records": len(rows),
        "min_per_domain": args.min_per_domain,
        "by_domain": dict(sorted(by_domain.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_answer_type": dict(sorted(by_type.items())),
        "by_review_status": dict(sorted(by_review.items())),
        "domain_source": {k: dict(v) for k, v in sorted(by_domain_source.items())},
        "domain_difficulty": {k: dict(v) for k, v in sorted(by_domain_difficulty.items())},
        "coverage_gaps": gaps,
        "benchmark_ready": not gaps and by_review.get("approved", 0) == len(rows),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
