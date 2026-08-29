#!/usr/bin/env python3
"""Export normalized benchmark data into competition-shaped input plus local gold."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export competition-compatible benchmark bundle.")
    p.add_argument("benchmark")
    p.add_argument("--output-dir", default="data/benchmark_v1")
    p.add_argument("--allow-unapproved", action="store_true")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.benchmark)
    if not rows:
        raise SystemExit("Benchmark is empty")

    if not args.allow_unapproved:
        bad = [r for r in rows if r.get("review_status") != "approved"]
        if bad:
            raise SystemExit(
                f"Refusing to export: {len(bad)} rows are not review_status=approved. "
                "Use --allow-unapproved only for exploratory runs."
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_rows, gold_rows = [], []
    for idx, row in enumerate(rows):
        problem = row.get("problem")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"row {idx}: missing non-empty problem")
        if "answer" not in row:
            raise ValueError(f"row {idx}: missing gold answer")
        input_rows.append({"idx": idx, "problem": problem})
        gold = dict(row)
        gold["idx"] = idx
        gold_rows.append(gold)

    input_path = output_dir / "input.jsonl"
    gold_path = output_dir / "gold.jsonl"
    write_jsonl(input_path, input_rows)
    write_jsonl(gold_path, gold_rows)

    forbidden = {"answer", "gold", "reference_solution", "solution", "label"}
    for row in input_rows:
        if forbidden.intersection(row):
            raise AssertionError("Gold leakage in competition input")
        if set(row) != {"idx", "problem"}:
            raise AssertionError(f"Unexpected input keys: {sorted(row)}")

    manifest = {
        "schema_version": 1,
        "n_rows": len(rows),
        "input_contract": {"required_keys": ["idx", "problem"]},
        "output_contract": {
            "required_keys": ["idx", "status", "final_response"],
            "note": "main.py writes one JSON result per problem; final_response must be non-empty on success"
        },
        "by_domain": dict(sorted(Counter(r.get("domain", "unknown") for r in rows).items())),
        "by_source": dict(sorted(Counter(r.get("source_dataset", "unknown") for r in rows).items())),
        "by_answer_type": dict(sorted(Counter(r.get("answer_type", "unknown") for r in rows).items())),
        "input_sha256": sha256_file(input_path),
        "gold_sha256": sha256_file(gold_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Competition-shaped input: {input_path}")
    print(f"Local gold file: {gold_path}")


if __name__ == "__main__":
    main()
