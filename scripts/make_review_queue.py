#!/usr/bin/env python3
"""Create a deterministic, source-diverse human review queue."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.domains import WORKING_TARGET_DOMAINS
from benchmark.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a source-diverse review queue.")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--output", default="data/review_queue_v1.jsonl")
    p.add_argument("--csv-output", default="data/review_queue_v1.csv")
    p.add_argument("--per-domain", type=int, default=30)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--domains", default=",".join(WORKING_TARGET_DOMAINS))
    return p.parse_args()


def round_robin_source_sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row.get("source_dataset", "unknown")].append(row)
    queues = {}
    for source, items in by_source.items():
        rng.shuffle(items)
        queues[source] = deque(items)
    source_order = sorted(queues)
    rng.shuffle(source_order)

    selected: list[dict] = []
    while len(selected) < n:
        made_progress = False
        for source in source_order:
            if queues[source] and len(selected) < n:
                selected.append(queues[source].popleft())
                made_progress = True
        if not made_progress:
            break
    return selected


def main() -> None:
    args = parse_args()
    if args.per_domain < 1:
        raise SystemExit("--per-domain must be positive")
    domains = [x.strip() for x in args.domains.split(",") if x.strip()]
    rng = random.Random(args.seed)

    by_domain: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for path in args.inputs:
        for row in read_jsonl(path):
            domain = row.get("domain")
            if domain not in domains:
                continue
            h = row.get("problem_hash") or row["problem"]
            if h in seen:
                continue
            seen.add(h)
            by_domain[domain].append(row)

    selected: list[dict] = []
    for domain in domains:
        selected.extend(round_robin_source_sample(by_domain[domain], args.per_domain, rng))

    rng.shuffle(selected)
    for idx, row in enumerate(selected):
        row["idx"] = idx
        row["review_status"] = "pending"
    write_jsonl(args.output, selected)

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["idx", "domain", "source_dataset", "source_id", "difficulty", "answer_type",
              "problem", "answer", "review_status", "review_notes"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in selected:
            item = dict(row)
            item.setdefault("review_notes", "")
            writer.writerow(item)

    counts = Counter(row["domain"] for row in selected)
    print(f"Wrote {len(selected)} rows to {args.output}")
    print(f"Wrote CSV review queue to {args.csv_output}")
    for domain in domains:
        print(f"{domain}: review {counts[domain]} / available {len(by_domain[domain])}")


if __name__ == "__main__":
    main()
